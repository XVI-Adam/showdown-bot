"""Opponent profiles: counted from logs, and silent when the sample is small."""

from __future__ import annotations

import json
from pathlib import Path

from psbot.analytics import MIN_BATTLES_FOR_PROFILE, BattleView, build_profile, build_profiles, load_battles, render_profile

# We are p2. Their Garchomp sets up, attacks into an immunity and a resistance,
# pivots out with U-turn, and their Skarmory replaces a fainted Pokémon.
EVENTS = [
    "switch|p1a: Garchomp|Garchomp, L77|100/100",
    "switch|p2a: Skarmory|Skarmory, L80|300/300",
    "turn|1",
    "move|p1a: Garchomp|Swords Dance|p1a: Garchomp",
    "move|p2a: Skarmory|Roost|p2a: Skarmory",
    "turn|2",
    "move|p1a: Garchomp|Earthquake|p2a: Skarmory",
    "-immune|p2a: Skarmory",
    "move|p2a: Skarmory|Brave Bird|p1a: Garchomp",
    "turn|3",
    "move|p1a: Garchomp|Earthquake|p2a: Skarmory",
    "-immune|p2a: Skarmory",
    "turn|4",
    "move|p1a: Garchomp|Iron Head|p2a: Skarmory",
    "-resisted|p2a: Skarmory",
    "turn|5",
    "switch|p1a: Rotom|Rotom-Wash, L82|100/100",
    "move|p2a: Skarmory|Brave Bird|p1a: Rotom",
    "turn|6",
    "move|p1a: Rotom|Volt Switch|p2a: Skarmory",
    "switch|p1a: Garchomp|Garchomp, L77|80/100",
    "turn|7",
    "move|p1a: Garchomp|Earthquake|p2a: Skarmory|[from] move: Copycat",
    "move|p2a: Skarmory|Brave Bird|p1a: Garchomp",
    "faint|p1a: Garchomp",
    "switch|p1a: Rotom|Rotom-Wash, L82|70/100",
    "turn|8",
    "move|p1a: Rotom|Hydro Pump|p2a: Skarmory",
    "faint|p2a: Skarmory",
    "win|Them",
]


def battle(i: int, opponent: str = "Them", won: bool | None = False, full_log: bool = True) -> BattleView:
    return BattleView(f"battle-{i}", opponent, "p2", won, [e.split("|") for e in EVENTS], full_log)


def test_one_battle_gives_counts_and_no_conclusions() -> None:
    profile = build_profile("Them", [battle(1)])
    assert profile["confidence"] == "insufficient" and profile["battles"] == 1
    assert "too few" in profile["note"]
    for interpreted in ("tendencies", "aggression", "switching", "predictability", "mistakes"):
        assert interpreted not in profile
    for species_level in ("leads", "most_used_moves", "troublesome_pokemon"):
        assert species_level not in profile  # random teams: these describe the draw, not the player
    text = render_profile(profile)
    assert "insufficient" in text and "Tendencies" not in text


def test_counts_from_the_log() -> None:
    n = MIN_BATTLES_FOR_PROFILE
    profile = build_profile("Them", [battle(i) for i in range(n)])
    assert profile["confidence"] == "low"
    # Per battle: 6 chosen moves by them + 1 voluntary switch (turn 5). The
    # Volt Switch exit and the replacement after the faint are not choices.
    assert profile["switching"]["rate_per_action"]["count"] == n
    assert profile["switching"]["rate_per_action"]["n"] == 7 * n
    assert profile["setup"]["rate_per_action"]["count"] == n
    assert profile["setup"]["battles_with_setup"]["value"] == 1.0
    assert profile["aggression"]["count"] == 5 * n  # 3 attacks + Volt Switch + Hydro Pump
    mistakes = profile["mistakes"]
    assert (mistakes["immune_attacks"]["count"], mistakes["resisted_attacks"]["count"]) == (2 * n, n)
    assert mistakes["immune_attacks"]["n"] == 5 * n and mistakes["complete"] is True
    assert profile["predictability"]["repeat_rate"]["count"] == n  # Earthquake after Earthquake
    assert profile["our_win_rate"]["value"] == 0.0
    low, high = profile["our_win_rate"]["ci95"]
    assert low == 0.0 and high > 0.3  # five losses still leave a wide interval


def test_tendencies_need_a_clear_interval() -> None:
    few = build_profile("Them", [battle(i) for i in range(5)])
    many = build_profile("Them", [battle(i) for i in range(40)])
    immune = "often attacks into an immunity"
    assert any(immune in t for t in many["tendencies"])  # 80 of 200 attacks
    # Same rates, 35 choices: nothing about switching can be claimed either way.
    assert not any("switch" in t for t in few["tendencies"])
    assert all("(" in t for t in many["tendencies"])  # every claim shows its numbers


def test_older_logs_report_missing_metrics_as_unavailable() -> None:
    profile = build_profile("Them", [battle(i, full_log=False) for i in range(6)])
    assert profile["mistakes"]["immune_attacks"] is None and profile["mistakes"]["complete"] is False
    assert "not available" in render_profile(profile)


def test_profiles_are_per_opponent_with_an_overall_one() -> None:
    battles = [battle(i, "Alice", won=True) for i in range(6)] + [battle(10, "Bob")]
    profiles = build_profiles(battles)
    assert set(profiles) == {"Alice", "Bob", "*"}
    assert profiles["Alice"]["our_record"] == {"wins": 6, "losses_or_ties": 0}
    assert profiles["Bob"]["confidence"] == "insufficient"
    assert profiles["*"]["battles"] == 7


def test_loads_both_record_layouts_and_skips_duplicates(tmp_path: Path) -> None:
    events = [e.split("|") for e in EVENTS]
    new = {
        "schema": "psbot.human_battle/1", "battle_id": "battle-1", "role": "p2", "events": events,
        "opponent": {"username": "Them"}, "outcome": {"won": True}, "agent": {"name": "heuristic"},
    }  # fmt: skip
    old = {"battle_tag": "battle-2", "opponent": "Them", "role": "p2", "won": False, "events": events}
    (tmp_path / "a.jsonl").write_text(json.dumps(new) + "\n" + json.dumps(new) + "\n")
    (tmp_path / "b.jsonl").write_text(json.dumps(old) + "\n")
    loaded = list(load_battles(tmp_path, tmp_path / "missing"))
    assert [(b.battle_id, b.full_log, b.won) for b in loaded] == [("battle-1", True, True), ("battle-2", False, False)]
