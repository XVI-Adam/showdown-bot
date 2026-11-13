"""Mechanics rules: each test states a documented Gen 9 rule."""

import pytest
from factories import make_battle, make_mon
from poke_env.battle import Effect, Field, Move, SideCondition, Status

from psbot.heuristics.mechanics import (
    boost_multiplier,
    effective_speed,
    hazard_entry_fraction,
    moves_first,
    status_block_reason,
)


def move(move_id: str) -> Move:
    return Move(move_id, gen=9)


@pytest.mark.parametrize(
    ("stage", "expected"),
    [(0, 1.0), (1, 1.5), (2, 2.0), (6, 4.0), (-1, 2 / 3), (-2, 0.5), (-6, 0.25)],
)
def test_boost_multiplier(stage: int, expected: float) -> None:
    assert boost_multiplier(stage) == pytest.approx(expected)


class TestSpeed:
    def test_paralysis_halves_speed(self) -> None:
        healthy = make_mon("jolteon")
        paralysed = make_mon("jolteon", status=Status.PAR)
        battle = make_battle(healthy, make_mon("snorlax", own=False))
        par_battle = make_battle(paralysed, make_mon("snorlax", own=False))
        assert effective_speed(par_battle, paralysed) == pytest.approx(
            effective_speed(battle, healthy) / 2
        )

    def test_choice_scarf_and_tailwind_multiply_speed(self) -> None:
        # Separate battles: two of the same species cannot share a team.
        plain = make_mon("garchomp")
        battle = make_battle(plain, make_mon("snorlax", own=False))
        base = effective_speed(battle, plain)
        scarfed = make_mon("garchomp", item="choicescarf")
        scarf_battle = make_battle(scarfed, make_mon("snorlax", own=False))
        assert effective_speed(scarf_battle, scarfed) == pytest.approx(base * 1.5)
        battle.side_conditions[SideCondition.TAILWIND] = 1
        assert effective_speed(battle, plain) == pytest.approx(base * 2)

    def test_faster_pokemon_moves_first(self) -> None:
        fast, slow = make_mon("jolteon"), make_mon("snorlax", own=False)
        battle = make_battle(fast, slow)
        assert moves_first(battle, fast, slow)
        assert not moves_first(battle, slow, fast)

    def test_priority_beats_speed(self) -> None:
        fast, slow = make_mon("jolteon"), make_mon("snorlax", own=False)
        battle = make_battle(fast, slow)
        assert moves_first(battle, slow, fast, priority=1)

    def test_trick_room_reverses_order(self) -> None:
        fast, slow = make_mon("jolteon"), make_mon("snorlax", own=False)
        battle = make_battle(fast, slow)
        battle._fields[Field.TRICK_ROOM] = 1
        assert moves_first(battle, slow, fast)
        assert not moves_first(battle, fast, slow)

    def test_speed_tie_is_treated_as_moving_second(self) -> None:
        a, b = make_mon("jolteon"), make_mon("jolteon", own=False)
        battle = make_battle(a, b)
        assert not moves_first(battle, a, b)


class TestEntryHazards:
    @pytest.mark.parametrize(
        ("species", "expected"),
        [
            ("charizard", 1 / 2),  # Fire/Flying: Rock is 4x
            ("blastoise", 1 / 8),  # Water: neutral
            ("lucario", 1 / 32),  # Fighting/Steel: Rock is 0.25x
        ],
    )
    def test_stealth_rock_scales_with_rock_effectiveness(self, species: str, expected: float) -> None:
        mon = make_mon(species)
        battle = make_battle(make_mon("snorlax"), make_mon("mew", own=False), bench=[mon])
        battle.side_conditions[SideCondition.STEALTH_ROCK] = 1
        assert hazard_entry_fraction(battle, mon) == pytest.approx(expected)

    @pytest.mark.parametrize(("layers", "expected"), [(1, 1 / 8), (2, 1 / 6), (3, 1 / 4)])
    def test_spikes_damage_by_layer(self, layers: int, expected: float) -> None:
        mon = make_mon("blastoise")
        battle = make_battle(make_mon("snorlax"), make_mon("mew", own=False), bench=[mon])
        battle.side_conditions[SideCondition.SPIKES] = layers
        assert hazard_entry_fraction(battle, mon) == pytest.approx(expected)

    def test_flying_types_ignore_spikes(self) -> None:
        mon = make_mon("charizard")
        battle = make_battle(make_mon("snorlax"), make_mon("mew", own=False), bench=[mon])
        battle.side_conditions[SideCondition.SPIKES] = 3
        assert hazard_entry_fraction(battle, mon) == 0

    def test_heavy_duty_boots_ignore_hazards(self) -> None:
        mon = make_mon("charizard", item="heavydutyboots")
        battle = make_battle(make_mon("snorlax"), make_mon("mew", own=False), bench=[mon])
        battle.side_conditions[SideCondition.STEALTH_ROCK] = 1
        assert hazard_entry_fraction(battle, mon) == 0

    def test_no_hazards_no_damage(self) -> None:
        mon = make_mon("charizard")
        battle = make_battle(make_mon("snorlax"), make_mon("mew", own=False), bench=[mon])
        assert hazard_entry_fraction(battle, mon) == 0


class TestStatusBlocking:
    @staticmethod
    def reason(move_id: str, target_species: str, **target_kwargs) -> str | None:
        target = make_mon(target_species, own=False, **target_kwargs)
        battle = make_battle(make_mon("mew"), target)
        return status_block_reason(battle, move(move_id), target)

    @pytest.mark.parametrize(
        ("move_id", "target"),
        [
            ("thunderwave", "garchomp"),  # Ground is immune to Thunder Wave
            ("thunderwave", "jolteon"),  # Electric types cannot be paralysed
            ("toxic", "skarmory"),  # Steel types cannot be poisoned
            ("toxic", "weezing"),  # Poison types cannot be poisoned
            ("willowisp", "arcanine"),  # Fire types cannot be burned
            ("spore", "venusaur"),  # Grass types are immune to powder moves
        ],
    )
    def test_type_immunities(self, move_id: str, target: str) -> None:
        assert self.reason(move_id, target) is not None

    @pytest.mark.parametrize(
        ("move_id", "target"),
        [("thunderwave", "gyarados"), ("toxic", "snorlax"), ("willowisp", "garchomp"), ("spore", "snorlax")],
    )
    def test_status_lands_on_susceptible_targets(self, move_id: str, target: str) -> None:
        assert self.reason(move_id, target) is None

    def test_already_statused_target(self) -> None:
        assert "already" in self.reason("toxic", "snorlax", status=Status.BRN)

    def test_known_blocking_ability(self) -> None:
        assert "goodasgold" in self.reason("thunderwave", "gholdengo", ability="goodasgold")

    def test_substitute_blocks_status_moves(self) -> None:
        target = make_mon("snorlax", own=False)
        target._effects[Effect.SUBSTITUTE] = 1
        battle = make_battle(make_mon("mew"), target)
        assert "Substitute" in status_block_reason(battle, move("toxic"), target)

    def test_sleep_clause(self) -> None:
        target = make_mon("snorlax", own=False)
        sleeper = make_mon("blissey", own=False, status=Status.SLP)
        battle = make_battle(make_mon("mew"), target, opp_bench=[sleeper])
        assert "Sleep Clause" in status_block_reason(battle, move("spore"), target)
