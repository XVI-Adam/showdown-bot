"""The opponent set belief: probabilities from counts, inference kept apart
from revelation."""

from __future__ import annotations

import json

import pytest
from factories import make_battle, make_mon
from poke_env.battle import Pokemon

from psbot.heuristics.context import BattleContext, HeuristicWeights, v2_weights
from psbot.heuristics.damage import MixtureEstimate, estimate_damage
from psbot.heuristics.randbats import SETS_PATH
from psbot.opponent_model import sets as model
from psbot.opponent_model.sets import belief_for, set_belief

# Counts shaped like the real table: (moves, ability, item, how often).
TABLE = {
    "garchomp": [
        (frozenset({"earthquake", "swordsdance", "scaleshot", "ironhead"}), "roughskin", "loadeddice", 60.0),
        (frozenset({"earthquake", "outrage", "spikes", "stealthrock"}), "roughskin", "rockyhelmet", 40.0),
    ],
    "typhlosion": [
        (frozenset({"eruption", "focusblast", "shadowball", "fireblast"}), "blaze", "choicescarf", 50.0),
        (frozenset({"eruption", "focusblast", "shadowball", "fireblast"}), "flashfire", "choicescarf", 50.0),
    ],
}


@pytest.fixture
def table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model, "_measured", lambda: TABLE)


def garchomp(*moves: str) -> Pokemon:
    return make_mon("garchomp", moves, own=False)


def test_probabilities_are_shares_of_the_measured_sets(table) -> None:
    belief = belief_for(garchomp())
    assert belief.measured and belief.revealed == frozenset()
    probabilities = belief.move_probabilities()
    assert probabilities["earthquake"] == pytest.approx(1.0)
    assert probabilities["swordsdance"] == pytest.approx(0.6)
    assert probabilities["outrage"] == pytest.approx(0.4)
    assert belief.item_probabilities() == pytest.approx({"loadeddice": 0.6, "rockyhelmet": 0.4})
    assert belief.ability_probabilities() == pytest.approx({"roughskin": 1.0})


def test_a_revealed_move_rules_out_other_sets(table) -> None:
    belief = belief_for(garchomp("swordsdance"))
    probabilities = belief.move_probabilities()
    assert "swordsdance" not in probabilities  # revealed, not a belief
    assert belief.move_probability("swordsdance") == 1.0
    assert probabilities["scaleshot"] == pytest.approx(1.0)
    assert "outrage" not in probabilities and belief.move_probability("outrage") == 0.0
    assert belief.item_probabilities() == pytest.approx({"loadeddice": 1.0})


def test_item_evidence_is_used_and_an_empty_slot_is_not(table) -> None:
    helmet = make_mon("garchomp", own=False, item="rockyhelmet")
    assert belief_for(helmet).move_probability("spikes") == pytest.approx(1.0)
    knocked_off = make_mon("garchomp", own=False)
    knocked_off._item = None
    assert belief_for(knocked_off).move_probability("spikes") == pytest.approx(0.4)


def test_evidence_matching_nothing_is_dropped_not_obeyed(table) -> None:
    belief = set_belief(garchomp(), ["surf"])  # e.g. an Illusion, or a newer server
    assert belief.relaxed == ("item", "ability", "moves")
    assert belief.move_probability("earthquake") == pytest.approx(1.0)
    odd_item = set_belief(garchomp(), ["swordsdance"], item="leftovers")
    assert odd_item.relaxed == ("item",) and odd_item.move_probability("scaleshot") == pytest.approx(1.0)


def test_without_measurements_probabilities_come_from_the_move_pool(monkeypatch) -> None:
    monkeypatch.setattr(model, "_measured", lambda: {})
    sets = json.loads(SETS_PATH.read_text())
    species = next(k for k, v in sets.items() if len(v["sets"]) == 1 and len(v["sets"][0]["movepool"]) == 5)
    belief = belief_for(make_mon(species, own=False))
    probabilities = belief.move_probabilities()
    assert not belief.measured and belief.item_probabilities() == {}
    assert len(probabilities) == 5
    assert all(p == pytest.approx(0.8) for p in probabilities.values())  # 4 of 5, uniformly
    assert sum(probabilities.values()) == pytest.approx(4.0)


def test_an_inferred_move_is_never_marked_revealed(table) -> None:
    opp = garchomp("swordsdance")
    battle = make_battle(make_mon("blissey", ["softboiled"]), opp)
    beliefs = {m.id: (p, shown) for m, p, shown in BattleContext(battle, v2_weights()).move_beliefs(opp)}
    assert beliefs["swordsdance"] == (1.0, True)
    assert beliefs["scaleshot"] == (pytest.approx(1.0), False)  # certain per the table, still not seen
    assert set(opp.moves) == {"swordsdance"}  # the model does not write to the Pokémon


def test_modelled_threat_sees_the_knockout_a_flat_discount_hides(table) -> None:
    """The report's biggest weakness: 80% of losses included a Pokémon lost
    that the engine expected to survive, mostly to a move not yet revealed."""
    opp = garchomp()
    me = make_mon("blissey", ["softboiled", "seismictoss"])
    battle = make_battle(me, opp)
    quake = estimate_damage(battle, opp, me, [m for m, _, _ in BattleContext(battle, v2_weights()).move_beliefs(opp) if m.id == "earthquake"][0])
    me._current_hp = int(me.max_hp * quake.min_fraction * 0.95)  # Earthquake always knocks out from here
    hp = me.current_hp_fraction

    baseline = BattleContext(battle, HeuristicWeights()).threat(me)
    modelled = BattleContext(battle, v2_weights()).threat(me)
    # Flat discount: 0.7 x the damage mostly no longer reaches the knockout.
    assert baseline.damage.ko_probability(hp) < 0.5
    # Model: Earthquake is on every measured set, so the knockout is expected.
    assert modelled.damage.ko_probability(hp) == pytest.approx(1.0)
    assert isinstance(modelled.damage, MixtureEstimate)
    assert modelled.revealed is False  # expected, not confirmed


def test_mixture_weighs_moves_by_probability_of_being_their_strongest() -> None:
    from psbot.heuristics.damage import DamageEstimate

    strong, weak = DamageEstimate(0.6, 0.7), DamageEstimate(0.2, 0.3)
    mixture = MixtureEstimate(0.6, 0.7, 1.0, ((0.4, strong), (0.6, weak)))
    assert mixture.expected == pytest.approx(0.4 * 0.65 + 0.6 * 0.25)
    assert mixture.ko_probability(0.5) == pytest.approx(0.4)
    assert mixture.scaled(0.5).expected == pytest.approx(mixture.expected / 2)


def test_unknown_ability_discounts_a_move_it_might_absorb(table) -> None:
    opp = make_mon("typhlosion", own=False)
    assert opp.ability is None
    me = make_mon("charizard", ["flamethrower", "airslash"])
    battle = make_battle(me, opp)
    flamethrower, airslash = me.moves["flamethrower"], me.moves["airslash"]
    plain = BattleContext(battle, HeuristicWeights())
    aware = BattleContext(battle, v2_weights())
    assert plain.damage(me, flamethrower).accuracy == pytest.approx(1.0)
    assert aware.damage(me, flamethrower).accuracy == pytest.approx(0.5)  # Flash Fire half the time
    assert aware.damage(me, airslash).accuracy == pytest.approx(plain.damage(me, airslash).accuracy)
    assert opp.ability is None  # the guess was not left on the Pokémon


def test_evasion_lowers_the_chance_to_hit_unless_the_move_cannot_miss() -> None:
    opp = make_mon("blissey", own=False, boosts={"evasion": 1})
    me = make_mon("blastoise", ["hydropump", "aurasphere"])
    aware = BattleContext(make_battle(me, opp), v2_weights())
    assert aware.damage(me, me.moves["hydropump"]).accuracy == pytest.approx(0.8 * 0.75)
    assert aware.damage(me, me.moves["aurasphere"]).accuracy == pytest.approx(1.0)
