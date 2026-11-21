"""Adapter for the official Pokémon Showdown server.

poke-env already speaks the protocol and ships the official server's address,
so this is a thin subclass of its ``PSClient`` rather than a new websocket
client. It changes only what the public server needs and the local simulator
does not:

* **Login** follows ``PROTOCOL.md`` (POST to ``/api/login``, then ``/trn``),
  runs off the event loop, and turns every failure into a :class:`LoginError`
  instead of an unhandled exception in a background task.
* **Nothing secret is logged.** poke-env logs each outgoing message, which
  would include the login assertion; here it is redacted, twice over.
* **Rate limit.** The server queues messages sent less than 600 ms apart and
  drops them once six are queued (``server/users.ts``), so we space ours out.
* **Safety rails.** The password is only ever sent to the official login
  server, and an account not registered as a bot is refused.

* **Reconnect.** A dropped connection is retried a few times with growing
  pauses, logging in again each time. The server then puts us back into any
  battle still running (``Rooms.global.rejoinGames``).

Challenges and battles are handled by ``psbot.battle_adapters.session``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from poke_env import AccountConfiguration, ServerConfiguration, ShowdownServerConfiguration
from poke_env.concurrency import POKE_LOOP, create_in_poke_loop, handle_threaded_coroutines
from poke_env.data import to_id_str
from poke_env.player import Player
from poke_env.ps_client import PSClient
from websockets.exceptions import ConnectionClosed

from psbot.battle_adapters.credentials import REGISTERED_VAR, BotCredentials, CredentialsError
from psbot.config import BATTLE_FORMAT

logger = logging.getLogger(__name__)

OFFICIAL_SERVER: ServerConfiguration = ShowdownServerConfiguration
LOGIN_URL = "https://play.pokemonshowdown.com/api/login"
# The server's throttle is 600 ms for ordinary accounts; stay clear of it.
MIN_SEND_INTERVAL = 0.65
HANDSHAKE_NAME = "psbot-handshake"
# Pauses before each reconnection attempt; giving up after the last one.
RECONNECT_DELAYS: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0, 30.0)
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_TRN = re.compile(r"(/trn [^,|\n]*,\d+,)[^\n]+")

AssertionFetcher = Callable[[str, str, str], str]
OutgoingFilter = Callable[[str, str], "str | None"]


def identity_name(identity: str) -> str:
    """``" Name"`` / ``"+Name@!"`` -> ``"Name"``: drop the rank character and
    the away marker Showdown puts around usernames."""
    return identity[1:].removesuffix("@!")


class LoginError(RuntimeError):
    """Connecting or logging in failed. The message never contains a secret."""


def redact(text: str) -> str:
    """Hide the login assertion in a protocol message."""
    return _TRN.sub(r"\1<redacted>", text)


class RedactingFilter(logging.Filter):
    """Second line of defence: scrubs assertions from anything this logger emits."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        if "/trn " in message:
            record.msg, record.args = redact(message), ()
        return True


def parse_login_response(body: str) -> str:
    """Extract the assertion from the login server's reply (``]`` + JSON).

    A failed login still returns a reply; its assertion is missing or starts
    with ``;`` followed by the reason.
    """
    if not body.startswith("]"):
        raise LoginError("login server sent an unexpected reply")
    try:
        data = json.loads(body[1:])
    except json.JSONDecodeError:
        raise LoginError("login server sent an unreadable reply") from None
    assertion = data.get("assertion") if isinstance(data, dict) else None
    if not isinstance(assertion, str) or not assertion:
        raise LoginError("login rejected: wrong username or password")
    if assertion.startswith(";"):
        reason = assertion.lstrip(";").strip()[:200] or "no reason given"
        raise LoginError(f"login rejected by the server: {reason}")
    if isinstance(data, dict) and data.get("actionsuccess") is False:
        raise LoginError("login rejected: wrong username or password")
    return assertion


def fetch_assertion(
    username: str, password: str, challstr: str, url: str = LOGIN_URL, timeout: float = 10.0
) -> str:
    """Exchange the password for a one-time login assertion (blocking HTTP)."""
    body = urlencode({"name": username, "pass": password, "challstr": challstr}).encode()
    request = Request(url, data=body, method="POST", headers={"User-Agent": "showdown-client"})
    try:
        with urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
    except HTTPError as exc:
        raise LoginError(f"login server answered HTTP {exc.code}") from None
    except (URLError, OSError) as exc:
        raise LoginError(f"login server unreachable ({type(exc).__name__})") from None
    return parse_login_response(text)


_default_fetch_assertion = fetch_assertion


class OfficialPSClient(PSClient):
    """poke-env's client with public-server login, redaction and throttling.

    :param credentials: the bot account, or None to only open the connection
        and read the server's greeting without logging in.
    :param fetch_assertion: replaces the HTTP login call (tests). Required for
        any server other than the official one, so the real password can never
        be sent somewhere else by mistake.
    """

    def __init__(
        self,
        credentials: BotCredentials | None,
        *,
        server_configuration: ServerConfiguration = OFFICIAL_SERVER,
        fetch_assertion: AssertionFetcher | None = None,
        min_send_interval: float = MIN_SEND_INTERVAL,
        reconnect_delays: tuple[float, ...] = RECONNECT_DELAYS,
        start_listening: bool = True,
        **kwargs: Any,
    ) -> None:
        official = server_configuration.websocket_url == OFFICIAL_SERVER.websocket_url
        if not official:
            host = urlsplit(server_configuration.websocket_url).hostname
            if host not in _LOOPBACK_HOSTS or fetch_assertion is None:
                raise ValueError(
                    "OfficialPSClient connects to the official server only "
                    "(a loopback test server needs an injected fetch_assertion)"
                )
        elif credentials is not None and not credentials.registered:
            raise CredentialsError(
                "this account is not marked as a registered bot. Showdown requires bots to be "
                "registered (in the Showdown client type /ht, then 'Something else' > 'I want to "
                f"register my bot'). Once staff confirm, set {REGISTERED_VAR}=1."
            )
        self._credentials = credentials
        self._fetch_assertion: AssertionFetcher = fetch_assertion or _default_fetch_assertion
        self._min_send_interval = min_send_interval
        self._last_send = 0.0
        self._reconnect_delays = reconnect_delays
        self._reconnects_left = len(reconnect_delays)
        self._closing = False
        self.closed = False  # True once the listener has stopped for good
        self.n_reconnects = 0
        # Set by the battle session: may rewrite or drop (None) a room message,
        # and is told about every successful login.
        self.outgoing_filter: OutgoingFilter | None = None
        self.on_login: Callable[[], None] | None = None
        # Challenge feedback: private notices (sender, text) and popups (text).
        self.on_notice: Callable[[str, str], None] | None = None
        self.on_popup: Callable[[str], None] | None = None
        self._login_error: LoginError | None = None
        self.confirmed_username: str | None = None
        loop = kwargs.get("loop", POKE_LOOP)
        self._greeted: asyncio.Event = create_in_poke_loop(asyncio.Event, loop)
        self._login_settled: asyncio.Event = create_in_poke_loop(asyncio.Event, loop)
        name = credentials.username if credentials is not None else HANDSHAKE_NAME
        super().__init__(
            # The password stays out of poke-env's (printable) account tuple.
            AccountConfiguration(name, None),
            server_configuration=server_configuration,
            start_listening=False,
            **kwargs,
        )
        if start_listening:
            self.start()

    def start(self) -> None:
        self._listening_coroutine = asyncio.run_coroutine_threadsafe(self.listen(), self.loop)

    def _create_logger(self, log_level: int | None) -> logging.Logger:
        # No handler of our own: records reach whatever the application set up.
        client_logger = logging.getLogger(self.username)
        if log_level is not None:
            client_logger.setLevel(log_level)
        # The websockets library logs every frame at DEBUG, assertion included.
        for target in (client_logger, logging.getLogger("websockets.client")):
            if not any(isinstance(f, RedactingFilter) for f in target.filters):
                target.addFilter(RedactingFilter())
        return client_logger

    # ------------------------------------------------------------ connection

    async def listen(self) -> None:
        """Listen until told to stop, reconnecting after a dropped connection.

        Only a connection that had logged in is retried: a failed first
        connection or a rejected login is reported instead of repeated.
        """
        try:
            while True:
                await super().listen()  # returns when the socket closes
                if (
                    self._closing
                    or self.confirmed_username is None  # never got in: report, don't retry
                    or self._login_error is not None
                    or not self._reconnects_left
                ):
                    break
                delay = self._reconnect_delays[-self._reconnects_left]
                self._reconnects_left -= 1
                self.n_reconnects += 1
                self.logger.warning("connection lost; reconnecting in %.0fs", delay)
                self.logged_in.clear()
                self._login_settled.clear()
                await asyncio.sleep(delay)
                if self._closing:
                    break
        finally:
            self.closed = True
            if not self._login_settled.is_set():
                self._fail("connection failed or closed before login finished")
            self._greeted.set()

    async def _stop_listening(self) -> None:
        self._closing = True
        websocket = getattr(self, "websocket", None)
        if websocket is not None:
            await websocket.close()

    def _fail(self, reason: str) -> None:
        if self._login_error is None:
            self._login_error = LoginError(reason)
        self._login_settled.set()

    # ----------------------------------------------------------------- login

    async def log_in(self, split_message: list[str]) -> None:
        """Answer the server's ``|challstr|`` greeting."""
        self._greeted.set()
        if self._credentials is None or len(split_message) < 4:
            return
        challstr = f"{split_message[2]}|{split_message[3]}"
        try:
            assertion = await asyncio.to_thread(
                self._fetch_assertion,
                self._credentials.username,
                self._credentials.password,
                challstr,
            )
        except LoginError as exc:
            self._login_error = exc
        except Exception as exc:  # never let the cause (it may hold the request) escape
            self._login_error = LoginError(f"login failed ({type(exc).__name__})")
        if self._login_error is not None:
            self.logger.error("%s", self._login_error)
            self._login_settled.set()
            await self._stop_listening()
            return
        await self.send_message(f"/trn {self.username},0,{assertion}")

    async def _handle_message(self, message: str) -> None:
        parts = message.split("\n", 1)[0].split("|")
        kind = parts[1] if len(parts) > 1 else ""
        if kind == "updateuser" and len(parts) > 3:
            self._on_updateuser(parts[2], parts[3])
            return
        if kind == "nametaken":
            reason = parts[3] if len(parts) > 3 else "name unavailable"
            self._fail(f"the server refused the username: {reason[:200]}")
            self.logger.error("%s", self._login_error)
            await self._stop_listening()
            return
        if kind == "pm":
            # One message can carry several lines (every pending challenge is
            # re-sent after a login); poke-env only dispatches single lines.
            for line in message.split("\n"):
                fields = line.split("|")
                if len(fields) <= 4 or fields[1] != "pm":
                    continue
                body = "|".join(fields[4:])
                if body.startswith("/challenge"):
                    await self._handle_challenge_request(fields)
                elif body.startswith(("/nonotify ", "/text ")) and self.on_notice is not None:
                    # e.g. "/nonotify Name rejected the challenge."
                    self.on_notice(identity_name(fields[2]), body.split(" ", 1)[1])
            return
        if kind == "popup" and self.on_popup is not None:
            self.on_popup(message.split("|popup|", 1)[-1].replace("||", "\n"))
        await super()._handle_message(message)

    def _on_updateuser(self, user: str, named: str) -> None:
        """``|updateuser|USER|NAMED|...``: USER is a rank character plus the
        name, optionally followed by ``@!`` when away."""
        name = user[1:].removesuffix("@!")
        if named != "1":
            return  # still a guest; the server sends this before we log in
        if self._credentials is not None and to_id_str(name) == self._credentials.user_id:
            if name != self.username:
                self.logger.warning(
                    "the server spells the username %r; set PSBOT_USERNAME to match", name
                )
            self.confirmed_username = name
            self._reconnects_left = len(self._reconnect_delays)
            self.logged_in.set()
            self._login_settled.set()
            if self.on_login is not None:
                self.on_login()
        else:
            self._fail(f"the server logged us in under a different name: {name!r}")

    async def _await(self, event: asyncio.Event, timeout: float, what: str) -> None:
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except asyncio.TimeoutError:
            raise LoginError(f"timed out after {timeout:.0f}s waiting for {what}") from None

    async def wait_for_greeting(self, timeout: float = 15.0) -> None:
        """Return once the server has greeted us; proves the connection works."""
        await handle_threaded_coroutines(
            self._await(self._greeted, timeout, "the server"), self.loop
        )
        if self._login_error is not None and not self.logged_in.is_set():
            raise self._login_error

    async def wait_until_logged_in(self, timeout: float = 30.0) -> str:
        """Return the username the server confirmed, or raise :class:`LoginError`."""
        await handle_threaded_coroutines(
            self._await(self._login_settled, timeout, "login"), self.loop
        )
        if self._login_error is not None:
            raise self._login_error
        assert self.confirmed_username is not None
        return self.confirmed_username

    # --------------------------------------------------------------- sending

    async def send_message(self, message: str, room: str = "", message_2: str | None = None) -> None:
        """Send one message, redacted in the log and spaced for the rate limit."""
        if room and self.outgoing_filter is not None:
            filtered = self.outgoing_filter(room, message)
            if filtered is None:
                return
            message = filtered
        to_send = "|".join([room, message, message_2] if message_2 else [room, message])
        async with self._sending_lock:
            wait = self._last_send + self._min_send_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self.logger.info(">>> %s", redact(to_send))
            try:
                await self.websocket.send(to_send)
            except ConnectionClosed:
                # Not retried here: after a reconnect the server re-sends any
                # request it still needs an answer to.
                self.logger.warning("not sent, connection is down: %s", redact(to_send))
                return
            self._last_send = time.monotonic()


def create_official_player(bot: str, credentials: BotCredentials, **client_kwargs: Any) -> Player:
    """Build one of our agents, unchanged, connected to the official server.

    ``bot`` is ``heuristic``, another scripted player, or a checkpoint path
    (see ``psbot.opponents.make_opponent``). The player logs in and then does
    nothing: challenges are accepted or sent only through an ``OfficialSession``
    (``psbot.battle_adapters.session``).
    """
    from psbot.opponents import make_opponent  # imports torch; keep login checks light

    server = client_kwargs.pop("server_configuration", OFFICIAL_SERVER)
    player = make_opponent(
        bot,
        account_configuration=AccountConfiguration(credentials.username, None),
        battle_format=BATTLE_FORMAT,
        server_configuration=server,
        max_concurrent_battles=1,
        start_listening=False,
    )
    player.ps_client = OfficialPSClient(
        credentials,
        server_configuration=server,
        on_battle_message=player._handle_battle_message,
        on_update_challenges=player._update_challenges,
        on_challenge_request=player._handle_challenge_request,
        **client_kwargs,
    )
    return player
