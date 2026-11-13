"""End-to-end decisions of the heuristic engine on hand-built positions."""

import json

from factories import make_battle, make_mon
from poke_env.battle import Move, PokemonType

from psbot.heuristics import HeuristicEngine, ScoreComponent
from psbot.recording import BattleRecorder, opponent_actions

TYRANITAR = ["stoneedge", "crunch", "earthquake", "firepunch"]
VENUSAUR = ["gigadrain", "sludgebomb", "sleeppowder", "synthesis"]


def best(battle):
    return HeuristicEngine().evaluate(battle)[0]


def test_picks_the_super_effective_stab_move() -> None:
    me = make_mon("charizard", ["flamethrower", "airslash", "earthquake", "roost"])
    battle = make_battle(me, make_mon("venusaur", VENUSAUR, own=False), bench=[make_mon("blastoise", ["surf"])])
    assert best(battle).label == "move flamethrower"


def test_uses_priority_to_ko_when_it_would_be_outsped_and_kod() -> None:
    me = make_mon("dragonite", ["outrage", "extremespeed"], hp=0.2)
    weavile = make_mon("weavile", ["iciclecrash", "knockoff", "lowkick", "swordsdance"], own=False, hp=0.1)
    assert best(make_battle(me, weavile)).label == "move extremespeed"


def test_switches_out_of_a_losing_matchup_into_a_resist() -> None:
    me = make_mon("charizard", ["flamethrower", "airslash"])
    battle = make_battle(
        me,
        make_mon("tyranitar", TYRANITAR, own=False),
        bench=[make_mon("hippowdon", ["earthquake", "slackoff"])],
    )
    assert best(battle).label == "switch hippowdon"


def test_stays_in_a_winning_matchup() -> None:
    me = make_mon("hippowdon", ["earthquake", "slackoff"])
    battle = make_battle(
        me,
        make_mon("tyranitar", TYRANITAR, own=False),
        bench=[make_mon("charizard", ["flamethrower", "airslash"])],
    )
    assert best(battle).label == "move earthquake"


def test_forced_switch_chooses_the_best_matchup() -> None:
    bench = [make_mon("talonflame", ["bravebird", "flareblitz"]), make_mon("hippowdon", ["earthquake"])]
    battle = make_battle(None, make_mon("tyranitar", TYRANITAR, own=False), bench=bench, force_switch=True)
    evaluations = HeuristicEngine().evaluate(battle)
    assert [e.label for e in evaluations] == ["switch hippowdon", "switch talonflame"]


def test_heals_when_low_against_a_weak_attacker() -> None:
    me = make_mon("blissey", ["softboiled", "seismictoss"], hp=0.3)
    battle = make_battle(me, make_mon("umbreon", ["foulplay", "protect", "toxic", "wish"], own=False))
    assert best(battle).label == "move softboiled"


def test_does_not_use_a_status_move_on_an_immune_target() -> None:
    me = make_mon("rotomwash", ["thunderwave", "hydropump"])
    battle = make_battle(me, make_mon("garchomp", ["earthquake", "outrage", "stoneedge", "swordsdance"], own=False))
    assert best(battle).label == "move hydropump"


def test_single_legal_action_is_returned_without_scoring() -> None:
    me = make_mon("snorlax", ["bodyslam"])
    battle = make_battle(me, make_mon("mew", own=False))
    battle._available_moves = [Move("recharge", gen=9)]  # pseudo-move with no move data
    (only,) = HeuristicEngine().evaluate(battle)
    assert only.label == "move recharge"
    assert only.components[0].name == "forced"


def test_no_legal_actions_gives_an_empty_list() -> None:
    battle = make_battle(None, make_mon("mew", own=False), force_switch=True)
    assert HeuristicEngine().evaluate(battle) == []


def test_terastallizes_to_power_up_the_chosen_attack() -> None:
    me = make_mon("charizard", ["flamethrower", "airslash"], tera_type=PokemonType.FIRE)
    battle = make_battle(me, make_mon("blissey", ["seismictoss", "softboiled", "toxic", "stealthrock"], own=False), can_tera=True)
    chosen = best(battle)
    assert chosen.label == "move flamethrower +tera"
    assert any("Tera Fire" in line for line in chosen.reasons())


def test_keeps_tera_when_the_new_type_is_weak_to_the_opponent() -> None:
    me = make_mon("charizard", ["flamethrower", "airslash"], tera_type=PokemonType.FIRE)
    battle = make_battle(me, make_mon("blastoise", ["surf", "icebeam", "rapidspin", "shellsmash"], own=False), can_tera=True)
    assert not best(battle).terastallize


def test_every_decision_is_explained() -> None:
    me = make_mon("charizard", ["flamethrower", "airslash"])
    battle = make_battle(me, make_mon("venusaur", VENUSAUR, own=False), bench=[make_mon("blastoise", ["surf"])])
    for evaluation in HeuristicEngine().evaluate(battle):
        assert evaluation.components
        assert evaluation.total == sum(c.value for c in evaluation.components)
        assert all(c.reason for c in evaluation.components)


def test_scorers_are_pluggable() -> None:
    """The engine is just a sum of components, so strategies can be swapped in."""
    base_power = lambda ctx, move: ScoreComponent("power", move.base_power, "raw base power")
    engine = HeuristicEngine(move_scorers=[base_power], switch_scorers=[])
    me = make_mon("charizard", ["flamethrower", "earthquake"])
    battle = make_battle(me, make_mon("venusaur", VENUSAUR, own=False))
    assert engine.evaluate(battle)[0].label == "move earthquake"  # 100 BP beats 90 BP


def test_recorder_writes_one_json_line_per_battle(tmp_path) -> None:
    me = make_mon("charizard", ["flamethrower", "airslash"])
    battle = make_battle(me, make_mon("venusaur", VENUSAUR, own=False), bench=[make_mon("blastoise", ["surf"])])
    battle._replay_data = [
        ["", "turn", "1"],
        ["", "move", "p1a: Charizard", "Flamethrower", "p2a: Venusaur"],
        ["", "move", "p2a: Venusaur", "Sludge Bomb", "p1a: Charizard"],
        ["", "turn", "2"],
        ["", "switch", "p2a: Blissey", "Blissey, L86, F", "100/100"],
    ]
    assert opponent_actions(battle) == [
        {"turn": 1, "type": "move", "detail": "Sludge Bomb"},
        {"turn": 2, "type": "switch", "detail": "Blissey"},
    ]

    recorder = BattleRecorder(tmp_path / "log" / "battles.jsonl")
    recorder.record_decision(battle, HeuristicEngine().evaluate(battle))
    recorder.record_decision(battle, [], fallback="engine error")
    battle._won = True
    recorder.finish_battle(battle, opponent_kind="test")

    (line,) = recorder.path.read_text().splitlines()
    record = json.loads(line)
    assert record["won"] is True
    assert record["opponent_kind"] == "test"
    assert record["remaining"] == 2
    first, second = record["decisions"]
    assert first["action"] == "move flamethrower"
    assert first["active"] == {"species": "charizard", "hp": 1.0, "status": None}
    assert first["reasons"] and first["alternatives"]
    assert second["action"] == "fallback"
    assert len(record["opponent_actions"]) == 2


def test_reads_our_tera_type_from_the_server_request() -> None:
    """poke-env leaves ``tera_type`` unset in random battles; the request has it."""
    me = make_mon("charizard", ["flamethrower", "airslash"])
    assert me.tera_type is None
    battle = make_battle(me, make_mon("blissey", ["seismictoss", "softboiled", "toxic", "stealthrock"], own=False), can_tera=True)
    battle._last_request = {"side": {"pokemon": [{"ident": "p1: Charizard", "teraType": "Fire"}]}}
    assert best(battle).terastallize
