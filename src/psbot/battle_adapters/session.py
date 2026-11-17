"""Consent-based battles on the official server: challenges in, challenges out.

An :class:`OfficialSession` sits between the server connection and one of our
agents. The agent still decides every move exactly as it does locally; the
session only decides *whether a battle happens* and keeps the conversation
with the server correct:

* it accepts a challenge only from a username on the approved list, in our
  format, one battle at a time, and declines everything else;
* it challenges someone only when asked to, once;
* it checks that the battle room that opens is in our format and against the
  expected opponent, and forfeits anything else;
* it never answers the same request twice (:class:`ChoiceGuard`);
* after a reconnect it rebuilds the battle from the log the server re-sends.

There is no ladder search here, and no way to add one from the command line.

Everything in this module runs on poke-env's background event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable

from poke_env.concurrency import handle_threaded_coroutines
from poke_env.data import to_id_str
from poke_env.player import Player

from psbot.battle_adapters.official import OfficialPSClient, identity_name
from psbot.config import BATTLE_FORMAT

logger = logging.getLogger(__name__)

Lines = list[list[str]]

# Server replies to a choice that can never succeed by trying again
# (``server/room-battle.ts``); poke-env would otherwise retry them forever.
_HOPELESS_ERRORS = ("too late", "nothing to choose", "Can't do anything")


@dataclass(frozen=True)
class BattleOutcome:
    battle_tag: str
    opponent: str
    won: bool | None  # None: tie, or the result was not seen
    turns: int
    remaining: int
    note: str = ""

    def __str__(self) -> str:
        result = {True: "won", False: "lost", None: "no result"}[self.won]
        extra = f" ({self.note})" if self.note else ""
        return (
            f"{result} vs {self.opponent} in {self.turns} turns, "
            f"{self.remaining} Pokémon left{extra}"
        )


def request_id(line: list[str]) -> int | None:
    """The ``rqid`` of a ``|request|JSON`` line, or None."""
    try:
        rqid = json.loads("|".join(line[2:])).get("rqid")
    except (ValueError, AttributeError):
        return None
    return rqid if isinstance(rqid, int) else None


class ChoiceGuard:
    """Makes sure each request from the server is answered exactly once.

    The server numbers its requests (``rqid``). We remember the newest number
    per room and the number we last answered, and:

    * tag each outgoing choice with its request number, so the server discards
      it if the turn has moved on;
    * drop a second choice for a request already answered;
    * drop a request the server repeats after a reconnect when it also says it
      has our choice (``|sentchoice|``); if it does not, our choice was lost
      and the request is let through to be answered again;
    * stop poke-env's retry loop after ``max_errors`` rejected choices.
    """

    def __init__(self, max_errors: int = 3) -> None:
        self.max_errors = max_errors
        self.latest: dict[str, int] = {}
        self.answered: dict[str, int] = {}
        self.errors: dict[str, int] = {}
        self.n_suppressed = 0

    def incoming(self, room: str, lines: Lines) -> Lines:
        """Filter the lines of one battle message (header line excluded)."""
        server_has_choice = any(len(line) > 1 and line[1] == "sentchoice" for line in lines)
        kept: Lines = []
        for line in lines:
            kind = line[1] if len(line) > 1 else ""
            detail = line[2] if len(line) > 2 else ""
            if kind == "sentchoice":
                continue
            if kind == "request" and detail:
                rqid = request_id(line)
                if rqid is not None:
                    if self.answered.get(room) == rqid:
                        if server_has_choice:
                            logger.info("%s: request %s repeated, already answered", room, rqid)
                            continue
                        logger.warning("%s: choice for request %s was lost; redoing", room, rqid)
                        del self.answered[room]
                    if rqid != self.latest.get(room):
                        self.errors[room] = 0
                    self.latest[room] = rqid
            elif kind == "error" and detail.startswith("[Invalid choice]"):
                self.errors[room] = self.errors.get(room, 0) + 1
                if any(text in detail for text in _HOPELESS_ERRORS):
                    logger.warning("%s: %s", room, detail)
                    continue
                if self.errors[room] > self.max_errors:
                    logger.error("%s: giving up on this request after: %s", room, detail)
                    continue
                self.answered.pop(room, None)  # the server wants another answer
            kept.append(line)
        return kept

    def outgoing(self, room: str, message: str) -> str | None:
        """Tag a choice with its request number, or return None to drop it."""
        if not message.startswith("/choose"):
            return message
        rqid = self.latest.get(room)
        if rqid is None:
            return message
        if self.answered.get(room) == rqid:
            self.n_suppressed += 1
            logger.warning("%s: duplicate choice for request %s not sent", room, rqid)
            return None
        self.answered[room] = rqid
        return f"{message}|{rqid}"

    def forget(self, room: str) -> None:
        for table in (self.latest, self.answered, self.errors):
            table.pop(room, None)


class OfficialSession:
    """Runs up to ``max_battles`` consent-based battles for ``player``.

    :param player: an agent built by ``create_official_player``.
    :param approved: usernames allowed to battle the bot. A user we challenge
        ourselves must be on it too.
    :param accept_timeout: seconds to wait for a battle after accepting.
    :param challenge_timeout: seconds before an unanswered challenge of ours
        is withdrawn.
    :param stall_timeout: seconds of silence in a battle before we start the
        battle timer, so an opponent who left cannot hold the room forever.
    :param rejoin_timeout: seconds after a reconnect to wait for the server to
        put us back into the battle before treating it as over.
    """

    def __init__(
        self,
        player: Player,
        approved: Iterable[str],
        *,
        max_battles: int = 1,
        battle_format: str = BATTLE_FORMAT,
        accept_timeout: float = 30.0,
        challenge_timeout: float = 120.0,
        stall_timeout: float = 180.0,
        rejoin_timeout: float = 15.0,
        reject_cooldown: float = 60.0,
    ) -> None:
        client = player.ps_client
        if not isinstance(client, OfficialPSClient):
            raise TypeError("player must come from create_official_player")
        self.player = player
        self.client: OfficialPSClient = client
        self.me = to_id_str(player.username)
        self.approved = frozenset(to_id_str(u) for u in approved) - {"", self.me}
        if not self.approved:
            raise ValueError("at least one approved opponent is required")
        if max_battles < 1:
            raise ValueError("max_battles must be >= 1")
        self.max_battles = max_battles
        self.format = battle_format
        self.accept_timeout = accept_timeout
        self.challenge_timeout = challenge_timeout
        self.stall_timeout = stall_timeout
        self.rejoin_timeout = rejoin_timeout
        self.reject_cooldown = reject_cooldown

        self.guard = ChoiceGuard()
        self.outcomes: list[BattleOutcome] = []
        self.active: str | None = None  # room of the battle in progress
        self.opponent: str | None = None
        self.n_rejected = 0
        self._expected: str | None = None  # id we accepted or challenged
        self._expected_until = 0.0
        self._outgoing_open = False  # our own challenge is still pending
        self._refused: set[str] = set()  # rooms we forfeited and left
        self._verified: set[str] = set()
        self._rejected_at: dict[str, float] = {}
        self._last_activity = 0.0
        self._last_init = 0.0
        self._lost_connection_at: float | None = None
        self._timer_started = False

        client._on_battle_message = self._on_battle_message
        client._on_challenge_request = self._on_challenge
        client._on_update_challenges = _ignore
        client.outgoing_filter = self.guard.outgoing
        client.on_login = self._on_login

    # ------------------------------------------------------------ challenges

    @property
    def _busy(self) -> bool:
        return self.active is not None or self._expected is not None

    @property
    def _slots_left(self) -> int:
        return self.max_battles - len(self.outcomes) - int(self._busy)

    async def _on_challenge(self, fields: list[str]) -> None:
        """``|pm|FROM|TO|/challenge FORMAT|...`` or a bare ``/challenge``,
        which means the challenge between FROM and TO no longer exists."""
        sender, body = to_id_str(identity_name(fields[2])), fields[4]
        challenge_format = body.removeprefix("/challenge").strip()
        if sender == self.me:
            if not challenge_format:
                self._outgoing_open = False  # accepted, declined or withdrawn
            return
        if not challenge_format:
            return  # they withdrew; if we had accepted, accept_timeout cleans up
        if sender == self._expected:
            return  # the same challenge announced again: already accepted
        reason = self._refusal(sender, to_id_str(challenge_format))
        if reason is not None:
            await self._reject(sender, reason)
            return
        self._expect(sender, self.accept_timeout)
        logger.info("accepting challenge from %s", sender)
        await self.client.send_message("/utm null")
        await self.client.send_message(f"/accept {sender}")

    def _refusal(self, sender: str, challenge_format: str) -> str | None:
        if sender not in self.approved:
            return "not on the approved list"
        if challenge_format != self.format:
            return f"format {challenge_format} is not {self.format}"
        if self._busy:
            return "already in a battle"
        if self._slots_left < 1:
            return "this session has played its battles"
        return None

    async def _reject(self, sender: str, reason: str) -> None:
        now = time.monotonic()
        if now - self._rejected_at.get(sender, -1e9) < self.reject_cooldown:
            return  # one reply per challenger per cooldown: no reject spam
        self._rejected_at[sender] = now
        self.n_rejected += 1
        logger.info("declining challenge from %s: %s", sender, reason)
        await self.client.send_message(f"/reject {sender}")

    def _expect(self, user_id: str, timeout: float) -> None:
        self._expected, self._expected_until = user_id, time.monotonic() + timeout

    async def challenge(self, username: str) -> None:
        """Send one challenge to an approved user."""
        await handle_threaded_coroutines(self._challenge(username), self.client.loop)

    async def _challenge(self, username: str) -> None:
        target = to_id_str(username)
        if target not in self.approved:
            raise ValueError(f"{username!r} is not on the approved list")
        if self._busy or self._slots_left < 1:
            raise RuntimeError("a battle or challenge is already in progress")
        await self.client.logged_in.wait()
        self._expect(target, self.challenge_timeout)
        self._outgoing_open = True
        logger.info("challenging %s", target)
        await self.client.send_message("/utm null")
        await self.client.send_message(f"/challenge {target}, {self.format}")

    # --------------------------------------------------------------- battles

    async def _on_battle_message(self, messages: Lines) -> None:
        room = messages[0][0][1:]
        is_init = len(messages) > 1 and messages[1][1:2] == ["init"]
        if not room.startswith(f"battle-{self.format}-") or room in self._refused:
            if is_init and room not in self._refused:
                await self._refuse(room, "not our format", forfeit=False)
            return
        known = room in self.player.battles
        if is_init:
            self._last_init = time.monotonic()
            if known:
                # After a reconnect the server sends the whole battle log
                # again; replaying it onto the old state would double-count,
                # so start that battle's state afresh.
                logger.info("%s: rejoined, rebuilding battle state", room)
                self._drop_battle(room)
        elif not known:
            return  # a room we are not tracking; poke-env would wait forever

        lines = self.guard.incoming(room, messages[1:])
        if room not in self._verified:
            problem = self._verify(room, lines)
            if problem is not None:
                await self._refuse(room, problem, forfeit=True)
                return
        self._last_activity = time.monotonic()
        self._timer_started = False
        await self.player._handle_battle_message([messages[0], *lines])
        if any(len(line) > 1 and line[1] in ("win", "tie") for line in lines):
            self._finish(room)

    def _verify(self, room: str, lines: Lines) -> str | None:
        """Check who the battle is against. Returns a problem, or None if fine
        or not yet knowable (the names arrive in the first messages)."""
        names = [line[3] for line in lines if len(line) > 3 and line[1] == "player" and line[3]]
        if len(names) < 2:
            titles = [line[2] for line in lines if len(line) > 2 and line[1] == "title"]
            names = titles[0].split(" vs. ") if titles else []
        ids = [to_id_str(name) for name in names]
        if len(ids) != 2:
            return None
        if self.me not in ids:
            return "we are not a player in it"
        index = 1 - ids.index(self.me)
        if ids[index] not in self.approved:
            return f"opponent {ids[index]} is not approved"
        if self.active not in (None, room):
            return "another battle is already in progress"
        if self.active is None and ids[index] != self._expected:
            return f"no battle with {ids[index]} was agreed"
        self._verified.add(room)
        self.active, self.opponent = room, names[index]
        self._expected, self._outgoing_open = None, False
        logger.info("%s: battle against %s", room, self.opponent)
        return None

    async def _refuse(self, room: str, why: str, forfeit: bool) -> None:
        logger.error("%s: leaving, %s", room, why)
        self._refused.add(room)
        self._drop_battle(room)
        self.guard.forget(room)
        if forfeit:
            await self.client.send_message("/forfeit", room)
        await self.client.send_message(f"/leave {room}")

    def _drop_battle(self, room: str) -> None:
        """Remove a battle from poke-env's bookkeeping (its slot counter is a
        queue that ``_create_battle`` fills and the end of a battle empties)."""
        if self.player._battles.pop(room, None) is None:
            return
        queue = self.player._battle_count_queue
        try:
            queue.get_nowait()
            queue.task_done()
        except (asyncio.QueueEmpty, ValueError):
            pass

    def _finish(self, room: str, note: str = "") -> None:
        if any(outcome.battle_tag == room for outcome in self.outcomes):
            return
        battle = self.player.battles.get(room)
        outcome = BattleOutcome(
            battle_tag=room,
            opponent=self.opponent or "?",
            won=battle.won if battle is not None else None,
            turns=battle.turn if battle is not None else 0,
            remaining=sum(1 for m in battle.team.values() if not m.fainted) if battle else 0,
            note=note,
        )
        self.outcomes.append(outcome)
        logger.info("%s: %s", room, outcome)
        self.guard.forget(room)
        if self.active == room:
            self.active = self.opponent = None

    # ------------------------------------------------------------ connection

    def _on_login(self) -> None:
        if self.active is not None and self.client.n_reconnects:
            self._lost_connection_at = time.monotonic()

    async def _tick(self) -> None:
        """Time-based housekeeping, called a few times per second."""
        now = time.monotonic()
        if self._expected is not None and now > self._expected_until:
            who, self._expected = self._expected, None
            logger.warning("no battle with %s started in time", who)
            if self._outgoing_open:
                self._outgoing_open = False
                await self.client.send_message(f"/cancelchallenge {who}")
        if self.active is None or not self.client.logged_in.is_set():
            return
        if self._lost_connection_at is not None:
            if self._last_init >= self._lost_connection_at:
                self._lost_connection_at = None  # the server put us back in
            elif now - self._lost_connection_at > self.rejoin_timeout:
                self._lost_connection_at = None
                room = self.active
                self._finish(room, note="battle ended while disconnected")
                self._drop_battle(room)
            return
        if not self._timer_started and now - self._last_activity > self.stall_timeout:
            self._timer_started = True
            logger.info("%s: no activity for %.0fs, starting the timer", self.active, self.stall_timeout)
            await self.client.send_message("/timer on", self.active)

    async def run(self, timeout: float | None = None) -> list[BattleOutcome]:
        """Wait until the session's battles are done, the connection is gone
        for good, or ``timeout`` seconds pass with no battle in progress."""
        return await handle_threaded_coroutines(self._run(timeout), self.client.loop)

    async def _run(self, timeout: float | None) -> list[BattleOutcome]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while len(self.outcomes) < self.max_battles and not self.client.closed:
            if deadline is not None and time.monotonic() > deadline and self.active is None:
                break
            await self._tick()
            await asyncio.sleep(0.2)
        return list(self.outcomes)

    async def close(self) -> None:
        """Withdraw any challenge, forfeit a battle in progress, disconnect."""
        await handle_threaded_coroutines(self._close(), self.client.loop)
        await self.client.stop_listening()

    async def _close(self) -> None:
        if self.client.closed or not self.client.logged_in.is_set():
            return
        if self._outgoing_open and self._expected is not None:
            await self.client.send_message(f"/cancelchallenge {self._expected}")
        if self.active is not None:
            logger.warning("%s: forfeiting, the session is closing", self.active)
            await self.client.send_message("/forfeit", self.active)
            self._finish(self.active, note="forfeited on shutdown")
        self._expected, self._outgoing_open = None, False


async def _ignore(_: Any) -> None:
    return None
