"""The official-server adapter, exercised against a fake server on loopback.

No test here touches the internet or needs an account.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import websockets
from poke_env import ServerConfiguration

from psbot.battle_adapters.credentials import (
    BotCredentials,
    CredentialsError,
    load_credentials,
    read_env_file,
)
from psbot.battle_adapters.official import (
    OFFICIAL_SERVER,
    LoginError,
    OfficialPSClient,
    RedactingFilter,
    create_official_player,
    parse_login_response,
    redact,
)
from psbot.players import HeuristicPlayer

PASSWORD = "hunter2-not-a-real-password"
ASSERTION = "0123456789abcdef-secret-assertion"
CREDS = BotCredentials("TestBot", PASSWORD, registered=True)


# ------------------------------------------------------------- credentials


def test_credentials_from_environment() -> None:
    env = {"PSBOT_USERNAME": "My Bot", "PSBOT_PASSWORD": "pw", "PSBOT_BOT_REGISTERED": "1"}
    creds = load_credentials(env, env_file=None)
    assert (creds.username, creds.password, creds.registered) == ("My Bot", "pw", True)
    assert creds.user_id == "mybot"


def test_credentials_from_file_and_environment_wins(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('# bot account\nPSBOT_USERNAME=FileBot\nexport PSBOT_PASSWORD="file pw"\n')
    env_file.chmod(0o600)
    creds = load_credentials({}, env_file=env_file)
    assert (creds.username, creds.password, creds.registered) == ("FileBot", "file pw", False)
    assert load_credentials({"PSBOT_USERNAME": "EnvBot"}, env_file=env_file).username == "EnvBot"


def test_missing_credentials_name_the_variables(tmp_path: Path) -> None:
    with pytest.raises(CredentialsError, match="PSBOT_USERNAME and PSBOT_PASSWORD"):
        load_credentials({}, env_file=tmp_path / "absent")
    with pytest.raises(CredentialsError, match="PSBOT_PASSWORD"):
        load_credentials({"PSBOT_USERNAME": "x"}, env_file=None)
    with pytest.raises(CredentialsError, match="1-18"):
        load_credentials({"PSBOT_USERNAME": "x" * 19, "PSBOT_PASSWORD": "p"}, env_file=None)


def test_password_never_appears_in_repr() -> None:
    assert PASSWORD not in repr(CREDS)
    assert PASSWORD not in str(CREDS)


def test_world_readable_env_file_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("PSBOT_USERNAME=a\n")
    env_file.chmod(0o644)
    with caplog.at_level(logging.WARNING):
        read_env_file(env_file)
    assert "chmod 600" in caplog.text


# --------------------------------------------------------------- redaction


def test_redact_hides_only_the_assertion() -> None:
    assert redact(f"|/trn TestBot,0,{ASSERTION}") == "|/trn TestBot,0,<redacted>"
    assert redact("battle-x|/choose move 1") == "battle-x|/choose move 1"


def test_logging_filter_redacts_formatted_arguments() -> None:
    record = logging.LogRecord("x", logging.INFO, "", 0, ">>> %s", (f"|/trn A,0,{ASSERTION}",), None)
    assert RedactingFilter().filter(record)
    assert ASSERTION not in record.getMessage()


# ---------------------------------------------------------- login response


def test_parse_login_response() -> None:
    assert parse_login_response(']{"actionsuccess":true,"assertion":"abc"}') == "abc"
    with pytest.raises(LoginError, match="wrong username or password"):
        parse_login_response(']{"actionsuccess":false,"assertion":""}')
    with pytest.raises(LoginError, match="name is taken"):
        parse_login_response(']{"assertion":";;name is taken"}')
    with pytest.raises(LoginError, match="unexpected"):
        parse_login_response("<html>error</html>")
    with pytest.raises(LoginError, match="unreadable"):
        parse_login_response("]not json")


# ------------------------------------------------------------- safety rails


def test_refuses_unregistered_bot_on_the_official_server() -> None:
    with pytest.raises(CredentialsError, match="register"):
        OfficialPSClient(BotCredentials("TestBot", PASSWORD, registered=False))


def test_refuses_other_servers() -> None:
    remote = ServerConfiguration("wss://example.com/showdown/websocket", "https://example.com")
    with pytest.raises(ValueError, match="official server only"):
        OfficialPSClient(CREDS, server_configuration=remote, fetch_assertion=lambda *a: "x")
    local = ServerConfiguration("ws://127.0.0.1:1/showdown/websocket", "unused")
    with pytest.raises(ValueError, match="fetch_assertion"):
        OfficialPSClient(CREDS, server_configuration=local)  # would use the real password


def test_official_address_is_the_one_poke_env_ships() -> None:
    assert OFFICIAL_SERVER.websocket_url == "wss://sim3.psim.us/showdown/websocket"


# -------------------------------------------------------------- fake server


class FakeShowdown:
    """Greets like Showdown and accepts ``/trn`` with the right assertion."""

    def __init__(self, accept_name: str = "TestBot") -> None:
        self.accept_name = accept_name
        self.received: list[tuple[float, str]] = []

    async def handler(self, connection: websockets.ServerConnection) -> None:
        await connection.send("|updateuser| Guest 42|0|1|{}")
        await connection.send("|challstr|4|deadbeef")
        async for message in connection:
            self.received.append((time.monotonic(), str(message)))
            if str(message).startswith("|/trn "):
                name, _, assertion = str(message)[6:].split(",", 2)
                if assertion == ASSERTION:
                    await connection.send(f"|updateuser| {self.accept_name}|1|1|{{}}")
                else:
                    await connection.send(f"|nametaken|{name}|Your verification failed.")

    @property
    def messages(self) -> list[str]:
        return [m for _, m in self.received]


@asynccontextmanager
async def fake_server(**kwargs):
    fake = FakeShowdown(**kwargs)
    async with websockets.serve(fake.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield fake, ServerConfiguration(f"ws://127.0.0.1:{port}/showdown/websocket", "unused")


def good_fetch(username: str, password: str, challstr: str) -> str:
    assert (username, password, challstr) == ("TestBot", PASSWORD, "4|deadbeef")
    return ASSERTION


def test_login_succeeds_and_logs_no_secret(caplog: pytest.LogCaptureFixture) -> None:
    async def run() -> tuple[str, list[str]]:
        async with fake_server() as (fake, config):
            client = OfficialPSClient(
                CREDS, server_configuration=config, fetch_assertion=good_fetch,
                min_send_interval=0, log_level=logging.DEBUG,
            )  # fmt: skip
            try:
                await client.wait_for_greeting(5)
                return await client.wait_until_logged_in(5), fake.messages
            finally:
                await client.stop_listening()

    with caplog.at_level(logging.DEBUG):
        name, messages = asyncio.run(run())
    assert name == "TestBot"
    assert messages == [f"|/trn TestBot,0,{ASSERTION}"]
    # Everything our side logged, down to the websocket library's DEBUG frames.
    # (The fake server's own logger sees the assertion, as the real server does.)
    ours = "\n".join(r.getMessage() for r in caplog.records if r.name != "websockets.server")
    assert "/trn TestBot,0,<redacted>" in ours
    assert "websockets.client" in {r.name for r in caplog.records}
    assert ASSERTION not in ours
    assert PASSWORD not in ours


def test_wrong_password_raises_and_sends_nothing() -> None:
    def rejecting_fetch(*_: str) -> str:
        raise LoginError("login rejected: wrong username or password")

    async def run() -> list[str]:
        async with fake_server() as (fake, config):
            client = OfficialPSClient(CREDS, server_configuration=config, fetch_assertion=rejecting_fetch)
            try:
                with pytest.raises(LoginError, match="wrong username or password"):
                    await client.wait_until_logged_in(5)
            finally:
                await client.stop_listening()
            return fake.messages

    assert asyncio.run(run()) == []


def test_unexpected_fetch_error_does_not_leak_its_message() -> None:
    def exploding_fetch(*_: str) -> str:
        raise RuntimeError(f"request body pass={PASSWORD}")

    async def run() -> str:
        async with fake_server() as (_, config):
            client = OfficialPSClient(CREDS, server_configuration=config, fetch_assertion=exploding_fetch)
            try:
                with pytest.raises(LoginError) as info:
                    await client.wait_until_logged_in(5)
                return str(info.value)
            finally:
                await client.stop_listening()

    assert PASSWORD not in asyncio.run(run())


def test_server_refusing_the_name_raises() -> None:
    async def run() -> None:
        async with fake_server() as (_, config):
            client = OfficialPSClient(
                CREDS, server_configuration=config, fetch_assertion=lambda *a: "stale", min_send_interval=0
            )
            try:
                with pytest.raises(LoginError, match="verification failed"):
                    await client.wait_until_logged_in(5)
            finally:
                await client.stop_listening()

    asyncio.run(run())


def test_login_under_a_different_name_is_an_error() -> None:
    async def run() -> None:
        async with fake_server(accept_name="SomeoneElse") as (_, config):
            client = OfficialPSClient(
                CREDS, server_configuration=config, fetch_assertion=good_fetch, min_send_interval=0
            )
            try:
                with pytest.raises(LoginError, match="different name"):
                    await client.wait_until_logged_in(5)
            finally:
                await client.stop_listening()

    asyncio.run(run())


def test_server_spelling_of_the_name_is_accepted() -> None:
    async def run() -> str:
        async with fake_server(accept_name="Test Bot@!") as (_, config):
            client = OfficialPSClient(
                CREDS, server_configuration=config, fetch_assertion=good_fetch, min_send_interval=0
            )
            try:
                return await client.wait_until_logged_in(5)
            finally:
                await client.stop_listening()

    assert asyncio.run(run()) == "Test Bot"


def test_handshake_only_sends_nothing() -> None:
    async def run() -> list[str]:
        async with fake_server() as (fake, config):
            client = OfficialPSClient(None, server_configuration=config, fetch_assertion=good_fetch)
            try:
                await client.wait_for_greeting(5)
                await asyncio.sleep(0.1)
                assert not client.logged_in.is_set()
            finally:
                await client.stop_listening()
            return fake.messages

    assert asyncio.run(run()) == []


def test_unreachable_server_raises_login_error() -> None:
    async def run() -> None:
        config = ServerConfiguration("ws://127.0.0.1:1/showdown/websocket", "unused")
        client = OfficialPSClient(CREDS, server_configuration=config, fetch_assertion=good_fetch)
        try:
            with pytest.raises(LoginError, match="connection failed"):
                await client.wait_until_logged_in(5)
        finally:
            await client.stop_listening()

    logging.disable(logging.CRITICAL)  # poke-env logs the refused connection with a traceback
    try:
        asyncio.run(run())
    finally:
        logging.disable(logging.NOTSET)


def test_messages_are_spaced_for_the_rate_limit() -> None:
    interval = 0.1

    async def run() -> list[float]:
        async with fake_server() as (fake, config):
            client = OfficialPSClient(
                CREDS, server_configuration=config, fetch_assertion=good_fetch, min_send_interval=interval
            )
            try:
                await client.wait_until_logged_in(5)

                async def burst() -> None:
                    await asyncio.gather(*(client.send_message(f"/cmd {i}") for i in range(3)))

                await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(burst(), client.loop))
                await asyncio.sleep(0.1)
            finally:
                await client.stop_listening()
            return [t for t, _ in fake.received]

    times = asyncio.run(run())
    assert len(times) == 4
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert min(gaps) >= interval * 0.8


def test_existing_agent_connects_unchanged_and_stays_idle() -> None:
    async def run() -> tuple[type, list[str], int]:
        async with fake_server() as (fake, config):
            player = create_official_player(
                "heuristic", CREDS, server_configuration=config,
                fetch_assertion=good_fetch, min_send_interval=0,
            )  # fmt: skip
            try:
                await player.ps_client.wait_until_logged_in(5)
                await asyncio.sleep(0.1)
            finally:
                await player.ps_client.stop_listening()
            return type(player), fake.messages, len(player.battles)

    kind, messages, battles = asyncio.run(run())
    assert kind is HeuristicPlayer
    assert messages == [f"|/trn TestBot,0,{ASSERTION}"]  # no challenge, search or join
    assert battles == 0


# ------------------------------------------------------------ HTTP request


def test_fetch_assertion_posts_the_documented_form() -> None:
    """PROTOCOL.md: POST name, pass and challstr; the reply is ``]`` + JSON."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs

    from psbot.battle_adapters.official import fetch_assertion

    seen: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"])).decode()
            seen.update(path=self.path, form=parse_qs(body))
            ok = parse_qs(body)["pass"] == [PASSWORD]
            reply = '{"actionsuccess":true,"assertion":"abc"}' if ok else '{"actionsuccess":false}'
            self.send_response(200)
            self.end_headers()
            self.wfile.write(("]" + reply).encode())

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/api/login"
    try:
        assert fetch_assertion("TestBot", PASSWORD, "4|dead beef", url=url) == "abc"
        assert seen["form"] == {"name": ["TestBot"], "pass": [PASSWORD], "challstr": ["4|dead beef"]}
        assert seen["path"] == "/api/login"  # nothing secret in the URL
        with pytest.raises(LoginError, match="wrong username or password"):
            fetch_assertion("TestBot", "wrong", "4|x", url=url)
    finally:
        server.shutdown()
        server.server_close()
    with pytest.raises(LoginError, match="unreachable"):
        fetch_assertion("TestBot", PASSWORD, "4|x", url=url, timeout=1)
