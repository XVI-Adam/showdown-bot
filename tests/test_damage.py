"""Damage estimation: the adapter around poke-env's damage calculator."""

import pytest
from factories import make_battle, make_mon
from poke_env.battle import Move

from psbot.heuristics.damage import DamageEstimate, estimate_damage, fill_unknowns
from psbot.heuristics.randbats import SETS_PATH, certain_ability, estimate_stats, possible_moves

needs_sets = pytest.mark.skipif(not SETS_PATH.exists(), reason="simulator data not installed")


def move(move_id: str) -> Move:
    return Move(move_id, gen=9)


class TestDamageEstimate:
    def test_expected_includes_accuracy(self) -> None:
        estimate = DamageEstimate(0.4, 0.6, accuracy=0.8)
        assert estimate.mean_fraction == pytest.approx(0.5)
        assert estimate.expected == pytest.approx(0.4)

    def test_guaranteed_ko_when_min_roll_is_enough(self) -> None:
        assert DamageEstimate(0.5, 0.6).ko_probability(0.5) == 1.0

    def test_no_ko_when_max_roll_falls_short(self) -> None:
        assert DamageEstimate(0.5, 0.6).ko_probability(0.61) == 0.0

    def test_partial_ko_chance_interpolates_between_rolls(self) -> None:
        assert DamageEstimate(0.4, 0.6).ko_probability(0.5) == pytest.approx(0.5)

    def test_ko_chance_includes_accuracy(self) -> None:
        assert DamageEstimate(0.9, 1.0, accuracy=0.7).ko_probability(0.5) == pytest.approx(0.7)

    def test_no_ko_on_fainted_target_or_zero_damage(self) -> None:
        assert DamageEstimate(0.5, 0.6).ko_probability(0.0) == 0.0
        assert DamageEstimate().ko_probability(0.1) == 0.0

    def test_scaled_discounts_damage_not_accuracy(self) -> None:
        scaled = DamageEstimate(0.4, 0.6, accuracy=0.9).scaled(0.5)
        assert (scaled.min_fraction, scaled.max_fraction, scaled.accuracy) == (0.2, 0.3, 0.9)


class TestEstimateDamage:
    def test_super_effective_beats_resisted(self) -> None:
        attacker = make_mon("charizard", ["flamethrower"])
        grass = make_mon("venusaur", own=False)
        water = make_mon("blastoise", own=False)
        battle = make_battle(attacker, grass, opp_bench=[water])
        vs_grass = estimate_damage(battle, attacker, grass, move("flamethrower"))
        vs_water = estimate_damage(battle, attacker, water, move("flamethrower"))
        assert vs_grass.mean_fraction > 3 * vs_water.mean_fraction > 0

    def test_type_immunity_deals_nothing(self) -> None:
        attacker = make_mon("garchomp", ["earthquake"])
        flyer = make_mon("charizard", own=False)
        battle = make_battle(attacker, flyer)
        assert estimate_damage(battle, attacker, flyer, move("earthquake")).max_fraction == 0

    def test_stab_is_a_fifty_percent_bonus(self) -> None:
        # Flamethrower and Ice Beam are both 90 BP special moves; Mew is neutral to both.
        attacker = make_mon("charizard", ["flamethrower", "icebeam"])
        target = make_mon("mew", own=False)
        battle = make_battle(attacker, target)
        stab = estimate_damage(battle, attacker, target, move("flamethrower")).mean_fraction
        plain = estimate_damage(battle, attacker, target, move("icebeam")).mean_fraction
        assert stab / plain == pytest.approx(1.5, rel=0.03)

    def test_status_moves_deal_no_damage(self) -> None:
        attacker = make_mon("charizard", ["willowisp"])
        target = make_mon("mew", own=False)
        battle = make_battle(attacker, target)
        assert estimate_damage(battle, attacker, target, move("willowisp")).max_fraction == 0

    def test_accuracy_is_carried_through(self) -> None:
        attacker = make_mon("tyranitar", ["stoneedge"])
        target = make_mon("mew", own=False)
        battle = make_battle(attacker, target)
        assert estimate_damage(battle, attacker, target, move("stoneedge")).accuracy == 0.8

    def test_opponent_attacks_use_estimated_stats(self) -> None:
        ours = make_mon("charizard")
        theirs = make_mon("tyranitar", ["stoneedge"], own=False)
        battle = make_battle(ours, theirs)
        estimate = estimate_damage(battle, theirs, ours, move("stoneedge"))
        assert estimate.min_fraction > 1  # 4x effective: a certain KO if it hits
        assert all(isinstance(v, int) for v in theirs.stats.values())

    def test_attack_boosts_increase_damage(self) -> None:
        attacker = make_mon("garchomp", ["earthquake"])
        target = make_mon("mew", own=False)
        battle = make_battle(attacker, target)
        base = estimate_damage(battle, attacker, target, move("earthquake")).mean_fraction
        attacker._boosts["atk"] = 2  # +2 doubles the stat
        plus2 = estimate_damage(battle, attacker, target, move("earthquake")).mean_fraction
        assert plus2 / base == pytest.approx(2.0, rel=0.03)

    def test_required_item_is_inferred_so_judgment_gets_its_type(self) -> None:
        ours = make_mon("latias")  # Dragon/Psychic: Bug is super effective
        arceus = make_mon("arceusbug", ["judgment"], own=False)
        battle = make_battle(ours, arceus)
        fill_unknowns(battle, arceus)
        assert arceus.item == "insectplate"
        typed = estimate_damage(battle, arceus, ours, move("judgment")).mean_fraction
        arceus._item = "leftovers"  # without the plate Judgment is Normal type
        untyped = estimate_damage(battle, arceus, ours, move("judgment")).mean_fraction
        assert typed > 2 * untyped


class TestRandomBattleKnowledge:
    def test_stat_formula_matches_random_battle_spread(self) -> None:
        # Mew has 100 in every base stat. With 85 EVs and 31 IVs at level 100:
        # other stats = floor((2*100 + 31 + 21) * 100/100) + 5 = 257; HP = 252 + 100 + 10.
        stats = estimate_stats(make_mon("mew", level=100))
        assert stats["hp"] == 362
        assert all(stats[s] == 257 for s in ("atk", "def", "spa", "spd", "spe"))

    @needs_sets
    def test_possible_moves_come_from_the_simulator_sets(self) -> None:
        ids = {m.id for m in possible_moves(make_mon("umbreon", own=False))}
        assert {"foulplay", "toxic", "wish", "protect"} <= ids

    @needs_sets
    def test_cosmetic_formes_fall_back_to_base_species(self) -> None:
        assert possible_moves(make_mon("alcremierubycream", own=False))

    @needs_sets
    def test_ability_known_when_species_has_only_one(self) -> None:
        assert certain_ability(make_mon("gholdengo", own=False)) == "goodasgold"
