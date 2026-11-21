"""Challenging consenting players: the list, the limits, the runner."""

from __future__ import annotations

import asyncio
import random
import string
from dataclasses import dataclass
from pathlib import Path

import pytest
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer

from psbot.battle_adapters.challenges import (
    ChallengeRunner, DailyState, Limits, add_consent, read_consent, refusal, remove_consent,
)  # fmt: skip
from psbot.config import BATTLE_FORMAT, local_server_configuration
from psbot.server import is_server_reachable


# ------------------------------------------------------------ consent list


def test_consent_list_add_read_remove(tmp_path: Path) -> None:
    path = tmp_path / "consenting.txt"
    assert read_consent(path) == []
    assert add_consent("Alice", "asked in chat", path) and add_consent("Bob", path=path)
    assert not add_consent("ALICE", path=path)  # same player, any spelling
    path.write_text(path.read_text() + "# a comment line\n\n  Carol  # added by hand\n")
    assert read_consent(path) == ["Alice", "Bob", "Carol"]
    assert "agreed" in path.read_text() and "asked in chat" in path.read_text()
    assert remove_consent("bob", path) and not remove_consent("bob", path)
    assert read_consent(path) == ["Alice", "Carol"]


# ------------------------------------------------------------------ limits


def test_state_resets_each_day(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = DailyState("2026-11-21", sent=7)
    state.record_decline("Alice", 100.0)
    state.save(path)
    same = DailyState.load(path, today="2026-11-21")
    assert same.sent == 7 and same.declines == {"alice": [100.0]}
    assert DailyState.load(path, today="2026-11-22").sent == 0
    assert DailyState.load(tmp_path / "missing.json", today="2026-11-22").sent == 0


def test_refusal_rules() -> None:
    limits = Limits(daily_cap=3, max_declines=2, decline_cooldown=600)
    state = DailyState("d")
    assert refusal(state, limits, "Alice", 1000.0) is None
    state.record_decline("Alice", 1000.0)
    why = refusal(state, limits, "Alice", 1100.0)
    assert why is not None and why[1] == 1600.0  # back after the cooldown
    assert refusal(state, limits, "Alice", 1600.0) is None
    state.record_decline("Alice", 1700.0)
    assert refusal(state, limits, "Alice", 99_999.0)[1] == float("inf")  # done for the day
    assert refusal(state, limits, "Bob", 0.0) is None
    state.sent = 3
    assert "daily cap" in refusal(state, limits, "Bob", 0.0)[0]


# ------------------------------------------------------------------ runner


@dataclass
class Outcome:
    opponent: str

    def __str__(self) -> str:
        return f"won vs {self.opponent}"


class FakeSession:
    """Answers challenges from a script: player -> list of answers in order."""

    def __init__(self, answers: dict[str, list[str]]) -> None:
        self.answers = {k.lower(): list(v) for k, v in answers.items()}
        self.outcomes: list[Outcome] = []
        self.active = self._expected = None
        self.challenge_status = None
        self.challenge_detail = ""
        self.sent: list[str] = []
        self.client = type("C", (), {"closed": False})()

    async def challenge(self, name: str) -> None:
        self.sent.append(name)
        answer = self.answers[name.lower()].pop(0) if self.answers[name.lower()] else "declined"
        self.challenge_status = answer
        if answer == "accepted":
            self.outcomes.append(Outcome(name))


def runner(session, targets, games, **limits) -> ChallengeRunner:
    defaults = dict(min_gap=0.0, decline_cooldown=0.2)
    return ChallengeRunner(session, targets, games, Limits(**{**defaults, **limits}),
                           state=DailyState("d"), save=lambda s: None, idle_timeout=5, say=lambda _: None)  # fmt: skip


def test_rotation_takes_turns_and_stops_at_the_game_count() -> None:
    session = FakeSession({"A": ["accepted"] * 5, "B": ["accepted"] * 5})
    asyncio.run(runner(session, ["A", "B"], 4).run())
    assert session.sent == ["A", "B", "A", "B"] and len(session.outcomes) == 4


def test_two_declines_end_it_for_the_day_with_a_pause_between() -> None:
    session = FakeSession({"A": ["declined", "expired", "accepted"]})
    r = runner(session, ["A"], 3, decline_cooldown=0.3)
    started = asyncio.get_event_loop_policy().new_event_loop().time()
    attempts = asyncio.run(r.run())
    assert [a.status for a in attempts] == ["declined", "expired"]  # never asked a third time
    assert session.sent == ["A", "A"] and not session.outcomes
    assert r.state.declines["a"][1] - r.state.declines["a"][0] >= 0.3  # waited out the cooldown


def test_offline_players_are_skipped_without_a_strike() -> None:
    session = FakeSession({"Gone": ["unavailable"], "Here": ["accepted", "accepted"]})
    r = runner(session, ["Gone", "Here"], 2)
    asyncio.run(r.run())
    assert session.sent == ["Gone", "Here", "Here"]
    assert "gone" not in r.state.declines


def test_daily_cap_stops_the_run() -> None:
    session = FakeSession({"A": ["accepted"] * 9})
    r = runner(session, ["A"], 9, daily_cap=3)
    asyncio.run(r.run())
    assert len(session.sent) == 3 and r.state.sent == 3


def test_server_throttle_is_retried_and_not_counted() -> None:
    session = FakeSession({"A": ["too_soon", "accepted"]})
    r = runner(session, ["A"], 1)
    asyncio.run(r.run())
    assert session.sent == ["A", "A"] and r.state.sent == 1 and len(session.outcomes) == 1


# --------------------------------------------- the real thing, locally

needs_simulator = pytest.mark.skipif(not is_server_reachable(), reason="local Showdown server not running")


def unique(prefix: str) -> str:
    return prefix + "".join(random.choices(string.ascii_lowercase, k=6))


class Decliner(RandomPlayer):
    """A friend who says no."""

    async def _handle_challenge_request(self, split_message):
        challenger = split_message[2].strip()
        if challenger != self.username:
            await self.ps_client.send_message(f"/reject {challenger}")


@needs_simulator
@pytest.mark.integration
def test_rotation_on_the_local_server() -> None:
    from psbot.battle_adapters.credentials import BotCredentials
    from psbot.battle_adapters.official import create_official_player
    from psbot.battle_adapters.session import OfficialSession

    async def play():
        common = dict(battle_format=BATTLE_FORMAT, server_configuration=local_server_configuration())
        friend = RandomPlayer(account_configuration=AccountConfiguration(unique("Friend"), None), **common)
        shy = Decliner(account_configuration=AccountConfiguration(unique("Shy"), None), **common)
        offline = unique("Offline")  # never connects
        bot = create_official_player(
            "heuristic", BotCredentials(unique("RotBot"), "unused", registered=True),
            server_configuration=local_server_configuration(), fetch_assertion=lambda *_: "", min_send_interval=0,
        )  # fmt: skip
        targets = [offline, shy.username, friend.username]
        session = OfficialSession(bot, targets, max_battles=2)
        housekeeping = None
        try:
            await bot.ps_client.wait_until_logged_in(10)
            accepting = asyncio.ensure_future(friend.accept_challenges(bot.username, 2))
            housekeeping = asyncio.ensure_future(session.run(timeout=None))
            r = ChallengeRunner(session, targets, 2, Limits(min_gap=0.5, decline_cooldown=1.0),
                                state=DailyState("d"), save=lambda s: None, idle_timeout=30, say=lambda _: None)  # fmt: skip
            attempts = await asyncio.wait_for(r.run(), 120)
            await asyncio.wait_for(accepting, 30)
            return attempts, session, r
        finally:
            await session.close()
            if housekeeping is not None:
                housekeeping.cancel()
            for p in (friend, shy):
                await p.ps_client.stop_listening()

    attempts, session, r = asyncio.run(play())
    by_player = {}
    for a in attempts:
        kind = next(k for k in ("Offline", "Shy", "Friend") if a.player.startswith(k))
        by_player.setdefault(kind, []).append(a.status)
    print(by_player)
    assert by_player["Offline"] == ["unavailable"]  # the server said "not found"; no strike
    assert by_player["Shy"] and set(by_player["Shy"]) == {"declined"} and len(by_player["Shy"]) <= 2
    assert by_player["Friend"] == ["accepted", "accepted"]
    assert len(session.outcomes) == 2 and all(o.won is not None for o in session.outcomes)
