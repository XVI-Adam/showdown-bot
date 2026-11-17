"""The forward projection: duels, the sweep check, the endgame search."""

from __future__ import annotations

from dataclasses import replace

import pytest
from factories import make_battle, make_mon

from psbot.heuristics import HeuristicEngine
from psbot.heuristics.context import BattleContext, HeuristicWeights, v2_weights
from psbot.heuristics.lookahead import WIN_VALUE, Lookahead

from poke_env.battle import Status

TYRANITAR = ["stoneedge", "crunch", "earthquake", "firepunch"]


def add_fainted(battle, n: int) -> None:
    """Pad our team with fainted Pokémon, so the team size (which tells the
    engine how many of theirs are still unseen) is what the test needs."""
    for i, species in enumerate(["pikachu", "eevee", "snorlax", "mew"][:n]):
        mon = make_mon(species, hp=0.0)
        mon._status = Status.FNT
        battle._team[f"p1: fainted{i}"] = mon
NEW_COMPONENTS = {"sweep", "endgame", "coverage"}


def test_new_behaviour_is_off_by_default() -> None:
    """The default weights must stay the baseline engine: the trained hybrids
    take its scores as input."""
    weights = HeuristicWeights()
    assert (weights.move_model, weights.sweep, weights.endgame) == (False, 0.0, 0.0)
    assert (weights.immunity_aware, weights.switch_in_risk) == (False, 0.0)
    me = make_mon("charizard", ["flamethrower", "earthquake"])
    battle = make_battle(
        me, make_mon("tyranitar", TYRANITAR, own=False),
        bench=[make_mon("hippowdon", ["earthquake"])], opp_bench=[make_mon("skarmory", ["bravebird"], own=False)],
    )  # fmt: skip
    assert BattleContext(battle).projection is None
    for evaluation in HeuristicEngine().evaluate(battle):
        assert not NEW_COMPONENTS & {c.name for c in evaluation.components}


def test_duel_goes_to_the_better_matchup() -> None:
    hippowdon = make_mon("hippowdon", ["earthquake", "slackoff"])
    charizard = make_mon("charizard", ["flamethrower", "airslash"])
    tyranitar = make_mon("tyranitar", TYRANITAR, own=False)
    look = Lookahead(BattleContext(make_battle(hippowdon, tyranitar, bench=[charizard]), v2_weights()))
    ours, theirs = look.duel(hippowdon, 1.0, tyranitar, 1.0)
    assert ours > 0 and theirs == 0
    ours, theirs = look.duel(charizard, 1.0, tyranitar, 1.0)
    assert ours == 0 and theirs > 0


def test_sweep_check_sends_in_the_pokemon_that_stops_it() -> None:
    """Forced switch against a boosted attacker: one of ours beats it, the
    other only feeds it another knockout."""
    tyranitar = make_mon("tyranitar", TYRANITAR, own=False, boosts={"spe": 2})
    bench = [make_mon("charizard", ["flamethrower", "airslash"]), make_mon("hippowdon", ["earthquake", "slackoff"])]
    battle = make_battle(None, tyranitar, bench=bench, force_switch=True)
    add_fainted(battle, 4)  # a team of six, so five of theirs are still unseen
    ctx = BattleContext(battle, replace(HeuristicWeights(), move_model=True, sweep=1.0))
    projection = ctx.projection
    assert projection is not None and projection.mode == "sweep"
    by_species = {mon.species: projection.values[("switch", id(mon))] for mon in bench}
    assert by_species["hippowdon"] > by_species["charizard"]

    evaluations = HeuristicEngine(ctx.weights).evaluate(battle, ctx)
    assert evaluations[0].label == "switch hippowdon"
    penalty = next(c for e in evaluations if e.label == "switch charizard" for c in e.components if c.name == "sweep")
    assert penalty.value < 0 and "tyranitar" in penalty.reason
    assert all(c.value <= 0 for e in evaluations for c in e.components if c.name in NEW_COMPONENTS)


class Scripted(Lookahead):
    """Pair facts set by hand, so the search itself is what is tested."""

    def __init__(self, ctx, our_hits, their_hits, first):
        super().__init__(ctx)
        self._hits, self._taken, self._order = our_hits, their_hits, first

    def our_hit(self, mine, theirs):
        return self._hits[(mine.species, theirs.species)], 0

    def their_hit(self, theirs, mine):
        return self._taken[(theirs.species, mine.species)]

    def we_first(self, mine, theirs):
        return self._order[(mine.species, theirs.species)]


def test_endgame_search_finds_the_winning_order() -> None:
    """2 v 2. Blastoise beats Tyranitar but loses to Venusaur; Charizard beats
    Venusaur but loses to Tyranitar. Tyranitar is in. Leading with Blastoise
    wins; leading with Charizard loses it for nothing and then loses."""
    blastoise, charizard = make_mon("blastoise", ["surf"]), make_mon("charizard", ["flamethrower"])
    tyranitar, venusaur = make_mon("tyranitar", own=False), make_mon("venusaur", own=False)
    battle = make_battle(None, tyranitar, bench=[blastoise, charizard], opp_bench=[venusaur], force_switch=True)
    # team size 2 v 2, so nothing of theirs is unseen
    ctx = BattleContext(battle, replace(HeuristicWeights(), move_model=True, endgame=1.0))
    assert ctx.opp_remaining == 2
    look = Scripted(
        ctx,
        our_hits={("blastoise", "tyranitar"): 0.6, ("blastoise", "venusaur"): 0.1,
                  ("charizard", "tyranitar"): 0.1, ("charizard", "venusaur"): 0.6},
        their_hits={("tyranitar", "blastoise"): 0.2, ("tyranitar", "charizard"): 1.0,
                    ("venusaur", "blastoise"): 0.6, ("venusaur", "charizard"): 0.2},
        first={(a, b): True for a in ("blastoise", "charizard") for b in ("tyranitar", "venusaur")},
    )  # fmt: skip
    projection = look.project()
    assert projection is not None and projection.mode == "endgame"
    lead_blastoise = projection.values[("switch", id(blastoise))]
    lead_charizard = projection.values[("switch", id(charizard))]
    assert lead_blastoise >= WIN_VALUE and "win" in projection.notes[("switch", id(blastoise))]
    assert lead_charizard <= -WIN_VALUE and "lose" in projection.notes[("switch", id(charizard))]


def test_endgame_assumes_they_choose_the_replacement_worst_for_us() -> None:
    """1 v 2 after we knock out their active: they may answer with Venusaur
    (which beats our Blastoise) or Tyranitar (which loses). Minimax must
    assume Venusaur."""
    blastoise = make_mon("blastoise", ["surf"])
    skarmory = make_mon("skarmory", own=False, hp=0.1)
    tyranitar, venusaur = make_mon("tyranitar", own=False), make_mon("venusaur", own=False)
    battle = make_battle(blastoise, skarmory, opp_bench=[tyranitar, venusaur])
    add_fainted(battle, 2)  # a team of three: all of theirs are seen
    ctx = BattleContext(battle, replace(HeuristicWeights(), move_model=True, endgame=1.0))
    look = Scripted(
        ctx,
        our_hits={("blastoise", "skarmory"): 0.5, ("blastoise", "tyranitar"): 0.6, ("blastoise", "venusaur"): 0.1},
        their_hits={("skarmory", "blastoise"): 0.0, ("tyranitar", "blastoise"): 0.2, ("venusaur", "blastoise"): 0.6},
        first={("blastoise", s): True for s in ("skarmory", "tyranitar", "venusaur")},
    )
    assert look.all_seen
    projection = look.project()
    assert projection is not None and projection.mode == "endgame"
    assert projection.values[("move", "surf")] <= -WIN_VALUE


def test_switch_in_risk_marks_down_a_move_their_bench_is_immune_to() -> None:
    me = make_mon("garchomp", ["earthquake", "outrage"])
    opp = make_mon("blissey", ["softboiled", "seismictoss"], own=False)
    skarmory = make_mon("skarmory", ["bravebird"], own=False)  # Flying: immune to Ground
    weights = replace(HeuristicWeights(), switch_in_risk=0.15)

    def coverage(battle) -> dict[str, float]:
        return {
            e.label: c.value
            for e in HeuristicEngine(weights).evaluate(battle)
            for c in e.components
            if c.name == "coverage"
        }

    exposed = coverage(make_battle(me, opp, opp_bench=[skarmory]))
    assert set(exposed) == {"move earthquake"} and exposed["move earthquake"] < 0
    assert coverage(make_battle(me, opp)) == {}  # nothing known on their bench: nothing assumed
    low = make_mon("blissey", ["softboiled"], own=False, hp=0.05)
    assert coverage(make_battle(me, low, opp_bench=[skarmory])) == {}  # a sure knockout is taken anyway


@pytest.mark.parametrize("name", ["heuristic", "heuristic_v2"])
def test_both_engines_are_registered(name: str) -> None:
    from psbot.opponents import SCRIPTED_OPPONENTS
    from psbot.players.heuristic import HeuristicPlayer

    assert issubclass(SCRIPTED_OPPONENTS[name], HeuristicPlayer)
