"""Phase 7 metrics: counted from the protocol log exactly as defined."""

from __future__ import annotations

import pytest
from factories import make_battle, make_mon

from psbot.evaluation.metrics import battle_metrics
from psbot.evaluation.probe import ko_check
from psbot.evaluation.summary import difference, mean, rate, summarise
from psbot.rl.observation import embed_battle

# We are p1.
LOG = """switch|p1a: Garchomp|Garchomp, L77|250/250
switch|p2a: Dragonite|Dragonite, L74|100/100
turn|1
move|p2a: Dragonite|Dragon Dance|p2a: Dragonite
move|p1a: Garchomp|Earthquake|p2a: Dragonite
-immune|p2a: Dragonite
turn|2
switch|p1a: Skarmory|Skarmory, L80|260/260
move|p2a: Dragonite|Fire Punch|p1a: Skarmory
-damage|p1a: Skarmory|100/260
turn|3
move|p2a: Dragonite|Fire Punch|p1a: Skarmory
-damage|p1a: Skarmory|0 fnt
faint|p1a: Skarmory
switch|p1a: Garchomp|Garchomp, L77|250/250
turn|4
move|p2a: Dragonite|Extreme Speed|p1a: Garchomp
-damage|p1a: Garchomp|0 fnt
faint|p1a: Garchomp
switch|p1a: Rotom|Rotom-Wash, L82|240/240
turn|5
move|p1a: Rotom|Volt Switch|p2a: Dragonite
-damage|p2a: Dragonite|20/100
switch|p1a: Blissey|Blissey, L85|400/400
move|p2a: Dragonite|Earthquake|p1a: Blissey
-damage|p1a: Blissey|300/400
turn|6
move|p1a: Blissey|Seismic Toss|p2a: Dragonite
-damage|p2a: Dragonite|0 fnt
faint|p2a: Dragonite
switch|p2a: Mew|Mew, L80|100/100
turn|7
move|p1a: Blissey|Toxic|p2a: Mew
-miss|p1a: Blissey|p2a: Mew"""


@pytest.fixture
def metrics():
    return battle_metrics([line.split("|") for line in LOG.splitlines()], "p1")


def test_moves_switches_and_mistakes(metrics) -> None:
    assert metrics["our_moves"] == 4 and metrics["immune"] == 1 and metrics["missed"] == 1
    # Turn 2 is the only voluntary switch: turn-3/4 ones replace a faint, the
    # turn-5 one is Volt Switch's.
    assert metrics["voluntary_switches"] == 1
    assert metrics["switch_big_hit"] == 1  # Skarmory lost 160/260 coming in
    assert metrics["switch_fainted_before_acting"] == 1  # and fainted without moving


def test_knockouts_counted_by_attack(metrics) -> None:
    assert (metrics["our_faints"], metrics["their_faints"]) == (2, 1)
    assert metrics["kos_suffered_by_attack"] == 2


def test_first_use_is_per_move(metrics) -> None:
    events = [line.split("|") for line in LOG.splitlines()]
    m = battle_metrics(events, "p1")
    # Turn 3 KO by a repeated Fire Punch (not first use); turn 4 KO by Extreme Speed (first use).
    assert m["kos_suffered_first_use"] == 1


def test_setup_sweeper_and_stages(metrics) -> None:
    assert metrics["faced_setup_sweeper"] and metrics["kos_by_setup_sweepers"] == 2
    assert not metrics["swept"]
    assert metrics["stages"]["early"] == {"dealt": 1, "suffered": 2}
    assert metrics["max_deficit"] == 2 and metrics["max_lead"] == 0
    assert metrics["endgame"] is None


def test_endgame_is_entered_once_at_two_each() -> None:
    lines = ["switch|p1a: A|A, L80|100/100", "switch|p2a: B|B, L80|100/100", "turn|1"]
    for i in range(4):
        lines += [f"faint|p1a: A{i}", f"faint|p2a: B{i}"]
    lines += ["faint|p2a: C"]
    m = battle_metrics([l.split("|") for l in lines], "p1")
    assert m["endgame"] == "level"


def test_perspective_flips_with_the_role(metrics) -> None:
    other = battle_metrics([line.split("|") for line in LOG.splitlines()], "p2")
    assert (other["our_faints"], other["their_faints"]) == (1, 2)
    assert other["voluntary_switches"] == 0


# ------------------------------------------------------------------ probe


def test_sure_ko_detection_and_probe_does_not_change_the_observation() -> None:
    me = make_mon("garchomp", ["earthquake", "swordsdance"])
    opp = make_mon("heatran", ["magmastorm"], own=False, hp=0.3)
    battle = make_battle(me, opp)
    before = embed_battle(battle).copy()
    quake, dance = me.moves["earthquake"], me.moves["swordsdance"]

    class Order:
        def __init__(self, order):
            self.order = order

    assert ko_check(battle, Order(quake)) == (True, False)
    assert ko_check(battle, Order(dance)) == (True, True)
    assert (embed_battle(battle) == before).all()  # what the PPO agents see is untouched
    healthy = make_battle(make_mon("garchomp", ["earthquake"]), make_mon("blissey", ["softboiled"], own=False))
    assert ko_check(healthy, Order(quake)) == (False, False)


# ---------------------------------------------------------------- summary


def test_intervals() -> None:
    r = rate(80, 100)
    assert r["value"] == 0.8 and r["ci"][0] < 0.8 < r["ci"][1]
    assert mean([1.0, 1.0, 1.0])["ci"] == [1.0, 1.0]
    d = difference(rate(55, 100), rate(50, 100))
    assert d["value"] == pytest.approx(0.05) and d["ci"][0] < 0 < d["ci"][1]
    assert difference(rate(900, 1000), rate(500, 1000))["ci"][0] > 0


def test_summary_counts(metrics) -> None:
    row = {**metrics, "won": False, "remaining": 4, "sure_ko_chances": 2, "sure_ko_missed": 1}
    s = summarise([row, {**row, "won": True}])
    assert s["win_rate"]["value"] == 0.5 and s["missed_kos"]["k"] == 2 and s["missed_kos"]["n"] == 4
    assert s["setup"]["faced"]["value"] == 1.0 and s["switches"]["fainted_before_acting"]["value"] == 1.0
