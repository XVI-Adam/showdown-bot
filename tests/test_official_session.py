"""Challenges and battles through ``OfficialSession``.

Three layers: the choice guard on its own, challenge decisions against a
scripted fake server, and whole battles (including a dropped connection)
against the real local simulator. Nothing here touches the internet.
"""

from __future__ import annotations

import asyncio
import json
import random
import string
from contextlib import asynccontextmanager
from typing import Any

import pytest
import websockets
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import RandomPlayer

from psbot.battle_adapters.credentials import BotCredentials
from psbot.battle_adapters.official import create_official_player
from psbot.battle_adapters.session import ChoiceGuard, OfficialSession, request_id
from psbot.config import BATTLE_FORMAT, local_server_configuration
from psbot.server import is_server_reachable

ROOM = "battle-gen9randombattle-1"


def request(rqid: int, **extra: Any) -> list[str]:
    return ["", "request", json.dumps({"rqid": rqid, **extra})]


# ------------------------------------------------------------ choice guard


def test_request_id_reads_the_number() -> None:
    assert request_id(request(7)) == 7
    assert request_id(["", "request", "not json"]) is None
    assert request_id(["", "request", "{}"]) is None


def test_choice_is_tagged_and_sent_once_per_request() -> None:
    guard = ChoiceGuard()
    guard.incoming(ROOM, [request(3)])
    assert guard.outgoing(ROOM, "/choose move 1") == "/choose move 1|3"
    assert guard.outgoing(ROOM, "/choose move 2") is None
    assert guard.n_suppressed == 1
    guard.incoming(ROOM, [request(4)])
    assert guard.outgoing(ROOM, "/choose switch 2") == "/choose switch 2|4"


def test_other_messages_and_other_rooms_pass_untouched() -> None:
    guard = ChoiceGuard()
    guard.incoming(ROOM, [request(3)])
    assert guard.outgoing(ROOM, "/forfeit") == "/forfeit"
    assert guard.outgoing(ROOM, "/timer on") == "/timer on"
    assert guard.outgoing("battle-gen9randombattle-2", "/choose move 1") == "/choose move 1"


def test_repeated_request_is_dropped_when_the_server_has_our_choice() -> None:
    guard = ChoiceGuard()
    guard.incoming(ROOM, [request(3)])
    guard.outgoing(ROOM, "/choose move 1")
    # What the server sends a player who reconnects (room-battle.ts onConnect).
    kept = guard.incoming(ROOM, [request(3), ["", "sentchoice", "move 1"]])
    assert kept == []
    assert guard.outgoing(ROOM, "/choose move 1") is None


def test_repeated_request_is_answered_again_when_our_choice_was_lost() -> None:
    guard = ChoiceGuard()
    guard.incoming(ROOM, [request(3)])
    guard.outgoing(ROOM, "/choose move 1")
    kept = guard.incoming(ROOM, [request(3)])  # no |sentchoice|: it never arrived
    assert kept == [request(3)]
    assert guard.outgoing(ROOM, "/choose move 1") == "/choose move 1|3"


def test_rejected_choice_may_be_retried_a_limited_number_of_times() -> None:
    guard = ChoiceGuard(max_errors=2)
    guard.incoming(ROOM, [request(3)])
    error = ["", "error", "[Invalid choice] Can't move: Earthquake is disabled"]
    for _ in range(2):
        assert guard.outgoing(ROOM, "/choose move 1") is not None
        assert guard.incoming(ROOM, [error]) == [error]  # poke-env sees it and retries
    assert guard.outgoing(ROOM, "/choose move 1") is not None
    assert guard.incoming(ROOM, [error]) == []  # third rejection: stop the loop


def test_too_late_error_is_never_retried() -> None:
    guard = ChoiceGuard()
    guard.incoming(ROOM, [request(3)])
    guard.outgoing(ROOM, "/choose move 1")
    late = ["", "error", "[Invalid choice] Sorry, too late to make a different move; the next turn has already started"]
    assert guard.incoming(ROOM, [late]) == []
    assert guard.outgoing(ROOM, "/choose move 1") is None


# ------------------------------------------------- challenges, fake server

CREDS = BotCredentials("TestBot", "unused", registered=True)


class FakeShowdown:
    """Logs any name in, then lets the test script the server's side."""

    def __init__(self) -> None:
        self.received: list[str] = []
        self.connection: websockets.ServerConnection | None = None
        self.logged_in = asyncio.Event()

    async def handler(self, connection: websockets.ServerConnection) -> None:
        self.connection = connection
        await connection.send("|challstr|4|deadbeef")
        async for message in connection:
            self.received.append(str(message))
            if str(message).startswith("|/trn "):
                await connection.send(f"|updateuser| {str(message)[6:].split(',')[0]}|1|1|{{}}")
                self.logged_in.set()

    async def send(self, message: str) -> None:
        assert self.connection is not None
        await self.connection.send(message)

    def sent_by_bot(self) -> list[str]:
        return [m for m in self.received if not m.startswith("|/trn ")]


@asynccontextmanager
async def scripted_session(approved: list[str], **session_kwargs: Any):
    fake = FakeShowdown()
    async with websockets.serve(fake.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        config = ServerConfiguration(f"ws://127.0.0.1:{port}/showdown/websocket", "unused")
        player = create_official_player(
            "heuristic", CREDS, server_configuration=config,
            fetch_assertion=lambda *_: "ok", min_send_interval=0,
        )  # fmt: skip
        session = OfficialSession(player, approved, **session_kwargs)
        try:
            await player.ps_client.wait_until_logged_in(5)
            await fake.logged_in.wait()
            yield fake, session
        finally:
            await player.ps_client.stop_listening()


def challenge_pm(sender: str, fmt: str = BATTLE_FORMAT) -> str:
    return f"|pm| {sender}| TestBot|/challenge {fmt}|{fmt}|||"


async def settle() -> None:
    await asyncio.sleep(0.25)


def test_approved_challenge_is_accepted_exactly_once() -> None:
    async def run() -> list[str]:
        async with scripted_session(["Alice"]) as (fake, _):
            await fake.send(challenge_pm("Alice"))
            await fake.send(challenge_pm("Alice@!"))  # announced again
            await fake.send(challenge_pm("Alice") + "\n" + challenge_pm("Alice"))  # as after a login
            await settle()
            return fake.sent_by_bot()

    assert asyncio.run(run()) == ["|/utm null", "|/accept alice"]


def test_unapproved_challenger_is_declined_once() -> None:
    async def run() -> tuple[list[str], int]:
        async with scripted_session(["Alice"]) as (fake, session):
            for _ in range(3):
                await fake.send(challenge_pm("Mallory"))
            await settle()
            return fake.sent_by_bot(), session.n_rejected

    assert asyncio.run(run()) == (["|/reject mallory"], 1)


def test_other_formats_are_declined() -> None:
    async def run() -> list[str]:
        async with scripted_session(["Alice"]) as (fake, _):
            await fake.send(challenge_pm("Alice", "gen9ou"))
            await settle()
            return fake.sent_by_bot()

    assert asyncio.run(run()) == ["|/reject alice"]


def test_second_challenger_is_declined_while_one_is_pending() -> None:
    async def run() -> list[str]:
        async with scripted_session(["Alice", "Bob"]) as (fake, _):
            await fake.send(challenge_pm("Alice"))
            await settle()
            await fake.send(challenge_pm("Bob"))
            await settle()
            return fake.sent_by_bot()

    assert asyncio.run(run()) == ["|/utm null", "|/accept alice", "|/reject bob"]


def test_withdrawn_and_own_challenge_notices_are_ignored() -> None:
    async def run() -> list[str]:
        async with scripted_session(["Alice"]) as (fake, _):
            await fake.send("|pm| Alice| TestBot|/challenge")
            await fake.send("|pm| TestBot| Alice|/challenge gen9randombattle|gen9randombattle|||")
            await fake.send("|pm| Alice| TestBot|hello there")
            await settle()
            return fake.sent_by_bot()

    assert asyncio.run(run()) == []


def test_outgoing_challenge_goes_only_to_approved_users_and_expires() -> None:
    async def run() -> list[str]:
        async with scripted_session(["Alice"], challenge_timeout=0.3) as (fake, session):
            with pytest.raises(ValueError, match="not on the approved list"):
                await session.challenge("Mallory")
            await session.challenge("Alice")
            with pytest.raises(RuntimeError, match="already in progress"):
                await session.challenge("Alice")
            assert await session.run(timeout=0.8) == []
            return fake.sent_by_bot()

    assert asyncio.run(run()) == [
        "|/utm null",
        "|/challenge alice, gen9randombattle",
        "|/cancelchallenge alice",
    ]


def test_battle_nobody_agreed_to_is_forfeited_and_left() -> None:
    async def run() -> tuple[list[str], int]:
        async with scripted_session(["Alice"]) as (fake, session):
            await fake.send(f">{ROOM}\n|init|battle\n|title|TestBot vs. Mallory")
            await fake.send(">battle-gen9ou-5\n|init|battle\n|title|TestBot vs. Alice")
            await fake.send(f">{ROOM}\n|request|" + json.dumps({"rqid": 1}))  # ignored now
            await settle()
            return fake.sent_by_bot(), len(session.player.battles)

    sent, battles = asyncio.run(run())
    assert sent == [f"{ROOM}|/forfeit", f"|/leave {ROOM}", "|/leave battle-gen9ou-5"]
    assert battles == 0


def test_session_requires_an_approved_opponent() -> None:
    async def run() -> None:
        async with scripted_session(["Alice"]) as (_, session):
            with pytest.raises(ValueError, match="at least one approved"):
                OfficialSession(session.player, ["TestBot", " "])

    asyncio.run(run())


# ------------------------------------------- whole battles, local simulator

needs_simulator = [
    pytest.mark.integration,
    pytest.mark.skipif(not is_server_reachable(), reason="local Showdown server not running"),
]


def unique(prefix: str) -> str:
    return prefix + "".join(random.choices(string.ascii_lowercase + string.digits, k=6))


def local_bot(approved: list[str], **kwargs: Any) -> OfficialSession:
    """Our agent behind the official adapter, pointed at the local simulator
    (whose password-less login takes an empty assertion)."""
    client_kwargs = {k: kwargs.pop(k) for k in ("reconnect_delays",) if k in kwargs}
    player = create_official_player(
        "heuristic", BotCredentials(unique("SessBot"), "unused", registered=True),
        server_configuration=local_server_configuration(),
        fetch_assertion=lambda *_: "", min_send_interval=0, **client_kwargs,
    )  # fmt: skip
    return OfficialSession(player, approved, **kwargs)


def visitor(cls: type = RandomPlayer) -> RandomPlayer:
    return cls(
        account_configuration=AccountConfiguration(unique("Visitor"), None),
        battle_format=BATTLE_FORMAT,
        server_configuration=local_server_configuration(),
    )


class SlowPlayer(RandomPlayer):
    """Takes a moment per move, like a person, so a battle lasts a few seconds."""

    async def choose_move(self, battle):  # type: ignore[override]
        await asyncio.sleep(0.15)
        return self.choose_random_move(battle)


@pytest.mark.parametrize("who_challenges", ["visitor", "bot"])
def test_full_battle_against_an_approved_opponent(who_challenges: str) -> None:
    async def run() -> OfficialSession:
        human = visitor()
        session = local_bot([human.username])
        bot_name = session.player.username
        try:
            await session.client.wait_until_logged_in(10)
            if who_challenges == "visitor":
                other = human.send_challenges(bot_name, 1)
            else:
                other = human.accept_challenges(bot_name, 1)
                await session.challenge(human.username)
            await asyncio.wait_for(asyncio.gather(session.run(timeout=60), other), 90)
        finally:
            await session.close()
            await human.ps_client.stop_listening()
        assert human.n_finished_battles == 1
        assert to_id(session.outcomes[0].opponent) == to_id(human.username)
        return session

    session = asyncio.run(run())
    (outcome,) = session.outcomes
    assert outcome.won is not None and outcome.turns > 0
    assert outcome.battle_tag.startswith("battle-gen9randombattle-")
    assert session.guard.n_suppressed == 0  # nothing needed suppressing in a clean battle
    assert session.active is None and session.n_rejected == 0
    assert session.player.n_finished_battles == 1


def to_id(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def test_stranger_is_declined_and_no_battle_starts() -> None:
    async def run() -> OfficialSession:
        stranger = visitor()
        session = local_bot(["SomeoneElse"])
        try:
            await session.client.wait_until_logged_in(10)
            pending = asyncio.ensure_future(stranger.send_challenges(session.player.username, 1))
            assert await session.run(timeout=3) == []
            pending.cancel()
        finally:
            await session.close()
            await stranger.ps_client.stop_listening()
        assert stranger.n_finished_battles == 0
        return session

    session = asyncio.run(run())
    assert session.n_rejected == 1
    assert len(session.player.battles) == 0


def test_battle_survives_a_dropped_connection() -> None:
    async def run() -> OfficialSession:
        human = visitor(SlowPlayer)
        session = local_bot([human.username], reconnect_delays=(0.3,))
        client = session.client
        try:
            await client.wait_until_logged_in(10)
            other = asyncio.ensure_future(human.send_challenges(session.player.username, 1))
            for _ in range(200):  # wait until the battle is a few turns in
                battles = list(session.player.battles.values())
                if battles and battles[0].turn >= 3:
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("battle did not start")
            # Cut the socket without telling the client it is deliberate.
            asyncio.run_coroutine_threadsafe(client.websocket.close(), client.loop)
            await asyncio.wait_for(asyncio.gather(session.run(timeout=60), other), 90)
        finally:
            await session.close()
            await human.ps_client.stop_listening()
        assert human.n_finished_battles == 1
        return session

    session = asyncio.run(run())
    assert session.client.n_reconnects == 1
    (outcome,) = session.outcomes
    assert outcome.won is not None and outcome.note == ""
    assert outcome.turns >= 3
    assert len(session.player.battles) == 1  # rebuilt, not duplicated


for _test in (
    test_full_battle_against_an_approved_opponent,
    test_stranger_is_declined_and_no_battle_starts,
    test_battle_survives_a_dropped_connection,
):
    for _mark in needs_simulator:
        _test = _mark(_test)
