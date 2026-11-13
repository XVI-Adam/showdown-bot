"""Weakness analysis on small hand-written battle records."""

import json

import pytest

from psbot.analysis import (
    battle_factors,
    build_timeline,
    compare_wins_and_losses,
    hybrid_deviations,
    load_records,
    move_kind,
    prediction_calibration,
    unexpected_ko_causes,
)


def mon(species: str, hp: float) -> dict:
    return {"species": species, "hp": hp, "status": None}


def decision(turn, active, opponent, action, forced=False, **predicted) -> dict:
    d = {"turn": turn, "active": active, "opponent": opponent, "forced_switch": forced, "action": action}
    if predicted:
        d["predicted"] = predicted
    return d


# We are p1. The opposing Dragonite sets up on turn 1 and then knocks out three of ours.
SWEPT_LOSS = {
    "role": "p1",
    "won": False,
    "turns": 4,
    "remaining": 0,
    "opponent_remaining": 3,
    "events": [
        ["switch", "p1a: Blissey", "Blissey, L86, F", "100/100"],
        ["switch", "p2a: Dragonite", "Dragonite, L74, M", "100/100"],
        ["turn", "1"],
        ["move", "p2a: Dragonite", "Dragon Dance", "p2a: Dragonite"],
        ["move", "p1a: Blissey", "Seismic Toss", "p2a: Dragonite"],
        ["turn", "2"],
        ["move", "p2a: Dragonite", "Outrage", "p1a: Blissey"],
        ["faint", "p1a: Blissey"],
        ["switch", "p1a: Weavile", "Weavile, L79, M", "100/100"],
        ["turn", "3"],
        ["move", "p2a: Dragonite", "Extreme Speed", "p1a: Weavile"],
        ["faint", "p1a: Weavile"],
        ["switch", "p1a: Gengar", "Gengar, L80, M", "100/100"],
        ["turn", "4"],
        ["move", "p1a: Gengar", "Shadow Ball", "p2a: Dragonite"],
        ["-miss", "p1a: Gengar", "p2a: Dragonite"],
        ["move", "p2a: Dragonite", "Outrage", "p1a: Gengar"],
        ["faint", "p1a: Gengar"],
        ["win", "Opponent"],
    ],
    "decisions": [
        decision(1, mon("blissey", 1.0), mon("dragonite", 1.0), "move seismictoss", first=False, threat=0.3, threat_ko=0.0, dealt=0.25, ko=0.0),
        # We expected to survive (KO chance 10%) and were knocked out: an unexpected KO.
        decision(2, mon("blissey", 1.0), mon("dragonite", 0.75), "move seismictoss", first=False, threat=0.45, threat_ko=0.1, dealt=0.25, ko=0.0),
        decision(2, mon("blissey", 0.0), mon("dragonite", 0.75), "switch weavile", forced=True),
        # We expected to move first and were outsped by priority.
        decision(3, mon("weavile", 1.0), mon("dragonite", 0.75), "move iceshard", first=True, threat=0.5, threat_ko=0.0, dealt=0.6, ko=0.0),
        decision(3, mon("weavile", 0.0), mon("dragonite", 0.75), "switch gengar", forced=True),
        decision(4, mon("gengar", 1.0), mon("dragonite", 0.75), "move shadowball", first=True, threat=1.2, threat_ko=1.0, dealt=0.4, ko=0.0),
    ],
}

CLEAN_WIN = {
    "role": "p2",
    "won": True,
    "turns": 3,
    "remaining": 3,
    "opponent_remaining": 0,
    "events": [
        ["switch", "p1a: Pikachu", "Pikachu, L93, M", "100/100"],
        ["switch", "p2a: Garchomp", "Garchomp, L74, F", "100/100"],
        ["turn", "1"],
        ["move", "p2a: Garchomp", "Stealth Rock", "p1a: Pikachu"],
        ["-sidestart", "p1: Opponent", "move: Stealth Rock"],
        ["move", "p1a: Pikachu", "Thunderbolt", "p2a: Garchomp"],
        ["-immune", "p2a: Garchomp"],
        ["turn", "2"],
        ["switch", "p2a: Blastoise", "Blastoise, L80, M", "100/100"],
        ["move", "p1a: Pikachu", "Volt Switch", "p2a: Blastoise"],
        ["turn", "3"],
        ["switch", "p2a: Garchomp", "Garchomp, L74, F", "100/100"],
        ["move", "p1a: Pikachu", "Surf", "p2a: Garchomp"],
        ["faint", "p1a: Pikachu"],
        ["faint", "p1a: Raichu"],
        ["win", "Us"],
    ],
    "decisions": [
        decision(1, mon("garchomp", 1.0), mon("pikachu", 1.0), "move stealthrock", first=True, threat=0.0, threat_ko=0.0, dealt=0.0, ko=0.0),
        decision(2, mon("garchomp", 1.0), mon("pikachu", 1.0), "switch blastoise", entry=0.1, target_hp=1.0),
        decision(3, mon("blastoise", 0.5), mon("pikachu", 1.0), "switch garchomp", entry=0.0, target_hp=1.0),
        decision(4, mon("garchomp", 1.0), mon("pikachu", 0.0), "move earthquake", first=True, threat=0.0, threat_ko=0.0, dealt=1.0, ko=1.0),
    ],
}


def test_move_kinds() -> None:
    assert move_kind("Dragon Dance") == "setup"
    assert move_kind("Stealth Rock") == "hazard"
    assert move_kind("Thunder Wave") == "status"
    assert move_kind("Recover") == "recovery"
    assert move_kind("Earthquake") == "attack"
    assert move_kind("Protect") == "other"
    assert move_kind("recharge") == "other"


def test_timeline_attributes_knockouts_to_the_opposing_active_pokemon() -> None:
    timeline = build_timeline(SWEPT_LOSS)
    assert timeline.kos_by == {"dragonite": 3}
    assert timeline.setup_by == {"dragonite": 1}
    assert timeline.lead_history == [-1, -2, -3]
    assert timeline.failed_moves == 1  # Shadow Ball missed
    assert [side for side, _, _ in timeline.moves[3]] == ["them"]


def test_timeline_respects_which_side_we_are() -> None:
    timeline = build_timeline(CLEAN_WIN)  # we are p2 here
    assert timeline.kos_by == {}
    assert timeline.lead_history == [1, 2]
    assert timeline.hazards_on_our_side is False  # the rocks are on their side
    assert timeline.failed_moves == 0  # the immunity was ours, not a failed move of ours
    assert timeline.our_move_kinds["hazard"] == 1


def test_battle_factors_for_a_setup_sweep() -> None:
    factors = battle_factors(SWEPT_LOSS)
    assert factors["swept"] == factors["swept_after_setup"] == 1.0
    assert factors["opponent_used_setup"] == 1.0
    assert factors["trailed_by_2"] == 1.0 and factors["led_by_2"] == 0.0
    # Blissey (10% predicted) and Weavile (0%) were unexpected; Gengar's KO was predicted.
    assert factors["n_unexpected_kos"] == 2.0
    assert factors["n_failed_moves"] == 1.0
    assert factors["n_voluntary_switches"] == 0.0  # both switches were forced


def test_speed_surprise_needs_both_sides_to_move() -> None:
    # Turn 3: predicted first, but only the opponent moved (we fainted) -> not counted.
    # Turn 4: predicted first and we did move first -> a check, not a surprise.
    factors = battle_factors(SWEPT_LOSS)
    assert factors["n_speed_checks"] == 1.0
    assert factors["n_speed_surprises"] == 0.0


def test_battle_factors_for_a_clean_win() -> None:
    factors = battle_factors(CLEAN_WIN)
    assert factors["swept"] == 0.0
    assert factors["led_by_2"] == 1.0
    assert factors["n_voluntary_switches"] == 2.0
    assert factors["n_chained_switches"] == 1.0
    assert factors["n_our_hazard_moves"] == 1.0
    assert factors["close_finish"] == 0.0


def test_compare_wins_and_losses_averages_each_group() -> None:
    result = compare_wins_and_losses([SWEPT_LOSS, CLEAN_WIN, CLEAN_WIN])
    assert (result["wins"], result["losses"]) == (2, 1)
    assert result["factors"]["swept"] == {"wins": 0.0, "losses": 1.0}
    assert result["factors"]["n_voluntary_switches"]["wins"] == 2.0


def test_calibration_compares_predictions_with_outcomes() -> None:
    result = prediction_calibration([SWEPT_LOSS, CLEAN_WIN])
    # Blissey turn 2: predicted 45%, lost 100% (fainted) -> underestimate by 0.55.
    # Weavile turn 3: predicted 50%, lost 100% -> underestimate by 0.50.
    # Garchomp turn 1 (the win): predicted 0%, lost 0% -> correct.
    # Blissey turn 1 is skipped: the opponent used a setup move, not an attack.
    assert result["threat"]["n"] == 3
    assert result["threat"]["under"] == pytest.approx(2 / 3)
    assert result["threat"]["ok"] == pytest.approx(1 / 3)
    assert result["threat"]["mean_error"] == pytest.approx((0.55 + 0.5 + 0.0) / 3)
    # Blastoise's entry: predicted 10%, actually lost 50% -> underestimate.
    # Garchomp's entry: predicted 0%, lost 0% -> correct.
    assert result["entry"]["n"] == 2
    assert result["entry"]["under"] == 0.5
    assert result["entry"]["fainted_on_entry"] == 0.0
    assert result["speed"]["n"] == 2  # Gengar's turn 4 and Garchomp's turn 1
    assert result["speed"]["moved_second"] == 0.0


def test_hybrid_deviations_are_grouped_by_kind() -> None:
    record = {
        "decisions": [
            {"action": "move earthquake", "agrees_with_heuristic": True},
            {"action": "switch gliscor", "agrees_with_heuristic": False, "heuristic_action": "move earthquake"},
            {"action": "move earthquake +tera", "agrees_with_heuristic": False, "heuristic_action": "move earthquake"},
            {"action": "move recharge"},  # forced: not a choice
        ]
    }
    result = hybrid_deviations([record])
    assert (result["choices"], result["disagreements"]) == (3, 2)
    assert result["kinds"] == {"move -> switch": 1, "move -> move + tera": 1}


def test_load_records_reads_json_lines(tmp_path) -> None:
    path = tmp_path / "battles.jsonl"
    path.write_text(json.dumps(SWEPT_LOSS) + "\n\n" + json.dumps(CLEAN_WIN) + "\n")
    assert [r["won"] for r in load_records(path)] == [False, True]


def test_unexpected_knockouts_are_explained() -> None:
    # Blissey (turn 2) fell to Outrage, which Dragonite had not attacked with before.
    # Weavile (turn 3) fell to Extreme Speed, also new.
    result = unexpected_ko_causes([SWEPT_LOSS])
    assert result["total"] == 2
    assert result["shares"]["unseen_move"] == 1.0


def test_critical_hits_and_known_moves_are_told_apart() -> None:
    record = {
        "role": "p1",
        "events": [
            ["switch", "p1a: Blissey", "Blissey, L86, F", "100/100"],
            ["switch", "p2a: Dragonite", "Dragonite, L74, M", "100/100"],
            ["turn", "1"],
            ["move", "p2a: Dragonite", "Outrage", "p1a: Blissey"],
            ["turn", "2"],
            ["move", "p2a: Dragonite", "Outrage", "p1a: Blissey"],
            ["faint", "p1a: Blissey"],
            ["switch", "p1a: Snorlax", "Snorlax, L82, M", "100/100"],
            ["turn", "3"],
            ["move", "p2a: Dragonite", "Outrage", "p1a: Snorlax"],
            ["-crit", "p1a: Snorlax"],
            ["faint", "p1a: Snorlax"],
        ],
        "decisions": [
            decision(2, mon("blissey", 0.6), mon("dragonite", 1.0), "move softboiled", first=False, threat=0.4, threat_ko=0.0, dealt=0.0, ko=0.0),
            decision(3, mon("snorlax", 1.0), mon("dragonite", 1.0), "move bodyslam", first=False, threat=0.5, threat_ko=0.0, dealt=0.3, ko=0.0),
        ],
    }
    shares = unexpected_ko_causes([record])["shares"]
    assert shares["known_move_underestimated"] == 0.5  # Blissey: Outrage was already seen
    assert shares["critical_hit"] == 0.5  # Snorlax
