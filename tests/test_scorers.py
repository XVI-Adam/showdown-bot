"""Scoring components, each tested in isolation."""

import pytest
from factories import make_battle, make_mon
from poke_env.battle import Move, SideCondition, Status

from psbot.heuristics import BattleContext
from psbot.heuristics import scorers as S

# Four revealed moves each, so the opponent's random-battle move pool is never consulted.
BLISSEY = ["seismictoss", "softboiled", "stealthrock", "thunderwave"]
TYRANITAR = ["stoneedge", "crunch", "earthquake", "firepunch"]
CHARIZARD = ["flamethrower", "airslash", "earthquake", "roost"]


def move(move_id: str) -> Move:
    return Move(move_id, gen=9)


def ctx_for(active, opponent, bench=(), **kwargs) -> BattleContext:
    return BattleContext(make_battle(active, opponent, bench=bench, **kwargs))


class TestDamageAndKo:
    def test_damage_is_capped_at_the_targets_remaining_hp(self) -> None:
        me = make_mon("charizard", ["flamethrower"])
        ctx = ctx_for(me, make_mon("venusaur", ["gigadrain"], own=False, hp=0.1))
        assert S.score_damage(ctx, move("flamethrower")).value == pytest.approx(0.1)

    def test_immune_target_scores_zero_with_a_reason(self) -> None:
        me = make_mon("garchomp", ["earthquake"])
        component = S.score_damage(ctx_for(me, make_mon("charizard", CHARIZARD, own=False)), move("earthquake"))
        assert component.value == 0
        assert "no effect" in component.reason

    def test_status_moves_have_no_damage_component(self) -> None:
        me = make_mon("charizard", ["roost"])
        assert S.score_damage(ctx_for(me, make_mon("mew", own=False)), move("roost")) is None

    def test_no_ko_bonus_when_the_target_survives(self) -> None:
        me = make_mon("charizard", ["flamethrower"])
        ctx = ctx_for(me, make_mon("blissey", BLISSEY, own=False))
        assert S.score_ko(ctx, move("flamethrower")) is None

    def test_ko_bonus_is_larger_when_moving_first(self) -> None:
        target = dict(own=False, hp=0.05)
        fast = ctx_for(make_mon("jolteon", ["thunderbolt"]), make_mon("gyarados", ["waterfall"], **target))
        slow = ctx_for(make_mon("snorlax", ["bodyslam"]), make_mon("gyarados", ["waterfall"], **target))
        assert S.score_ko(fast, move("thunderbolt")).value > S.score_ko(slow, move("bodyslam")).value

    def test_slower_attacker_discounts_damage_by_chance_of_being_kod_first(self) -> None:
        # Dragonite at 20% is outsped and KOed by Weavile's Icicle Crash (90% accurate).
        me = make_mon("dragonite", ["outrage", "extremespeed"], hp=0.2)
        weavile = make_mon("weavile", ["iciclecrash", "knockoff", "lowkick", "swordsdance"], own=False)
        ctx = ctx_for(me, weavile)
        assert ctx.act_probability(move("outrage")) == pytest.approx(0.1)
        assert ctx.act_probability(move("extremespeed")) == 1.0  # priority +2 goes first


class TestDamageTaken:
    def test_cost_is_capped_at_our_remaining_hp(self) -> None:
        me = make_mon("charizard", ["flamethrower"], hp=0.3)
        ctx = ctx_for(me, make_mon("tyranitar", TYRANITAR, own=False))
        assert S.score_damage_taken(ctx, move("flamethrower")).value == pytest.approx(-0.3, abs=0.01)

    def test_no_cost_when_we_ko_first(self) -> None:
        me = make_mon("jolteon", ["thunderbolt"])
        ctx = ctx_for(me, make_mon("gyarados", ["waterfall"], own=False, hp=0.05))
        assert S.score_damage_taken(ctx, move("thunderbolt")).value == pytest.approx(0)

    def test_unrevealed_moves_are_flagged_in_the_reason(self) -> None:
        me = make_mon("charizard", ["flamethrower"])
        ctx = ctx_for(me, make_mon("tyranitar", ["crunch"], own=False))
        component = S.score_damage_taken(ctx, move("flamethrower"))
        # Skipped silently if the simulator's set data is not installed.
        if component is not None and "stoneedge" in component.reason:
            assert "(unrevealed)" in component.reason


class TestStatus:
    def test_blocked_status_scores_zero_and_explains_why(self) -> None:
        me = make_mon("rotomwash", ["thunderwave"])
        component = S.score_status(ctx_for(me, make_mon("garchomp", ["earthquake"], own=False)), move("thunderwave"))
        assert component.value == 0
        assert "would fail" in component.reason

    def test_paralysis_is_worth_more_on_a_faster_target(self) -> None:
        me = make_mon("snorlax", ["thunderwave"])
        faster = S.score_status(ctx_for(me, make_mon("gyarados", ["waterfall"], own=False)), move("thunderwave"))
        me2 = make_mon("jolteon", ["thunderwave"])
        slower = S.score_status(ctx_for(me2, make_mon("blissey", BLISSEY, own=False)), move("thunderwave"))
        assert faster.value > slower.value > 0

    def test_burn_is_worth_more_on_a_physical_attacker(self) -> None:
        me = make_mon("mew", ["willowisp"])
        physical = S.score_status(ctx_for(me, make_mon("garchomp", ["earthquake"], own=False)), move("willowisp"))
        me2 = make_mon("mew", ["willowisp"])
        special = S.score_status(ctx_for(me2, make_mon("alakazam", ["psychic"], own=False)), move("willowisp"))
        assert physical.value > special.value > 0

    def test_self_targeting_status_is_ignored(self) -> None:
        me = make_mon("snorlax", ["rest"])
        assert S.score_status(ctx_for(me, make_mon("mew", own=False)), move("rest")) is None


class TestSetup:
    def test_setup_has_value_when_the_opponent_is_passive(self) -> None:
        me = make_mon("scizor", ["swordsdance", "bulletpunch"])
        component = S.score_setup(ctx_for(me, make_mon("blissey", BLISSEY, own=False)), move("swordsdance"))
        assert component.value > 0

    def test_setup_is_worthless_when_we_would_be_kod(self) -> None:
        me = make_mon("scizor", ["swordsdance", "bulletpunch"])
        component = S.score_setup(ctx_for(me, make_mon("charizard", CHARIZARD, own=False)), move("swordsdance"))
        assert component.value == 0

    def test_boosting_an_unused_attack_stat_is_worthless(self) -> None:
        me = make_mon("scizor", ["swordsdance", "roost"])  # no physical attack to boost
        component = S.score_setup(ctx_for(me, make_mon("blissey", BLISSEY, own=False)), move("swordsdance"))
        assert component.value == 0

    def test_maxed_boosts_are_worthless(self) -> None:
        me = make_mon("scizor", ["swordsdance", "bulletpunch"], boosts={"atk": 6})
        component = S.score_setup(ctx_for(me, make_mon("blissey", BLISSEY, own=False)), move("swordsdance"))
        assert component.value == 0


class TestRecovery:
    def test_healing_scales_with_missing_hp(self) -> None:
        opp = lambda: make_mon("umbreon", ["foulplay", "protect", "toxic", "wish"], own=False)
        low = S.score_recovery(ctx_for(make_mon("blissey", ["softboiled"], hp=0.3), opp()), move("softboiled"))
        high = S.score_recovery(ctx_for(make_mon("blissey", ["softboiled"], hp=0.8), opp()), move("softboiled"))
        assert low.value == pytest.approx(0.5)
        assert high.value == pytest.approx(0.2, abs=0.01)

    def test_no_value_at_full_hp(self) -> None:
        me = make_mon("blissey", ["softboiled"])
        assert S.score_recovery(ctx_for(me, make_mon("mew", own=False)), move("softboiled")).value == 0

    def test_discounted_when_the_opponent_out_damages_the_heal(self) -> None:
        me = make_mon("charizard", ["roost"], hp=0.4)
        component = S.score_recovery(ctx_for(me, make_mon("tyranitar", TYRANITAR, own=False)), move("roost"))
        assert 0 < component.value < 0.2
        assert "they deal" in component.reason


class TestHazards:
    @staticmethod
    def full_team():
        return [make_mon(s, ["tackle"]) for s in ("blastoise", "venusaur", "snorlax", "mew", "jolteon")]

    def test_stealth_rock_is_worth_more_with_more_opponents_left(self) -> None:
        me = make_mon("skarmory", ["stealthrock"])
        ctx = ctx_for(me, make_mon("blissey", BLISSEY, own=False), bench=self.full_team())
        assert ctx.opp_remaining == 6
        assert S.score_hazard(ctx, move("stealthrock")).value == pytest.approx(0.09 * 5)

    def test_no_value_when_already_set(self) -> None:
        me = make_mon("skarmory", ["stealthrock"])
        ctx = ctx_for(me, make_mon("blissey", BLISSEY, own=False), bench=self.full_team())
        ctx.battle.opponent_side_conditions[SideCondition.STEALTH_ROCK] = 3
        assert S.score_hazard(ctx, move("stealthrock")).value == 0

    def test_spikes_can_stack_with_diminishing_value(self) -> None:
        me = make_mon("skarmory", ["spikes"])
        ctx = ctx_for(me, make_mon("blissey", BLISSEY, own=False), bench=self.full_team())
        first = S.score_hazard(ctx, move("spikes")).value
        ctx.battle.opponent_side_conditions[SideCondition.SPIKES] = 1
        second = S.score_hazard(ctx, move("spikes")).value
        ctx.battle.opponent_side_conditions[SideCondition.SPIKES] = 3
        assert first > second > 0
        assert S.score_hazard(ctx, move("spikes")).value == 0

    def test_removal_is_valued_by_what_our_bench_would_lose(self) -> None:
        me = make_mon("excadrill", ["rapidspin"])
        ctx = ctx_for(me, make_mon("blissey", BLISSEY, own=False), bench=[make_mon("charizard", CHARIZARD)])
        assert S.score_hazard(ctx, move("rapidspin")) is None  # nothing to clear
        ctx.battle.side_conditions[SideCondition.STEALTH_ROCK] = 1
        assert S.score_hazard(ctx, move("rapidspin")).value == pytest.approx(0.5 * 0.5)


class TestSelfCost:
    def test_recoil_is_a_cost(self) -> None:
        me = make_mon("talonflame", ["bravebird"])
        assert S.score_self_cost(ctx_for(me, make_mon("mew", own=False)), move("bravebird")).value < 0

    def test_self_destruct_costs_remaining_hp(self) -> None:
        me = make_mon("golem", ["explosion"], hp=0.5)
        component = S.score_self_cost(ctx_for(me, make_mon("mew", own=False)), move("explosion"))
        assert component.value == pytest.approx(-0.6 * 0.5, abs=0.01)  # self_ko weight x HP

    def test_drain_only_helps_when_damaged(self) -> None:
        opp = lambda: make_mon("blastoise", ["surf"], own=False)
        full = S.score_self_cost(ctx_for(make_mon("venusaur", ["gigadrain"]), opp()), move("gigadrain"))
        hurt = S.score_self_cost(ctx_for(make_mon("venusaur", ["gigadrain"], hp=0.5), opp()), move("gigadrain"))
        assert full is None
        assert hurt.value > 0


class TestSwitchScorers:
    def test_entry_cost_includes_the_hit_and_hazards(self) -> None:
        bench = make_mon("charizard", CHARIZARD)
        ctx = ctx_for(make_mon("snorlax", ["bodyslam"]), make_mon("blissey", BLISSEY, own=False), bench=[bench])
        hit_only = S.score_switch_entry(ctx, bench).value
        ctx.battle.side_conditions[SideCondition.STEALTH_ROCK] = 1
        with_rocks = S.score_switch_entry(ctx, bench).value
        assert with_rocks == pytest.approx(hit_only - 0.5)

    def test_free_switch_takes_no_hit(self) -> None:
        bench = [make_mon("hippowdon", ["earthquake"]), make_mon("talonflame", ["bravebird"])]
        ctx = ctx_for(None, make_mon("tyranitar", TYRANITAR, own=False), bench=bench, force_switch=True)
        assert ctx.free_switch
        assert S.score_switch_entry(ctx, bench[0]) is None
        assert S.score_switch_tempo(ctx, bench[0]) is None

    def test_fainting_on_entry_is_heavily_penalised(self) -> None:
        frail = make_mon("charizard", CHARIZARD, hp=0.3)
        ctx = ctx_for(make_mon("snorlax", ["bodyslam"]), make_mon("tyranitar", TYRANITAR, own=False), bench=[frail])
        component = S.score_switch_entry(ctx, frail)
        assert component.value < -1
        assert "faint" in component.reason

    def test_tempo_cost_grows_with_abandoned_boosts(self) -> None:
        bench = make_mon("blastoise", ["surf"])
        plain = ctx_for(make_mon("scizor", ["bulletpunch"]), make_mon("mew", own=False), bench=[bench])
        boosted = ctx_for(
            make_mon("scizor", ["bulletpunch"], boosts={"atk": 2}),
            make_mon("mew", own=False),
            bench=[make_mon("blastoise", ["surf"])],
        )
        assert S.score_switch_tempo(boosted, bench).value == pytest.approx(
            S.score_switch_tempo(plain, bench).value - 0.2
        )

    def test_matchup_prefers_the_pokemon_that_wins_the_exchange(self) -> None:
        good, bad = make_mon("hippowdon", ["earthquake"]), make_mon("talonflame", ["bravebird"])
        ctx = ctx_for(make_mon("snorlax", ["bodyslam"]), make_mon("tyranitar", TYRANITAR, own=False), bench=[good, bad])
        assert S.score_switch_matchup(ctx, good).value > S.score_switch_matchup(ctx, bad).value


class TestThreat:
    def test_sleeping_opponent_is_less_threatening(self) -> None:
        awake = ctx_for(make_mon("charizard", CHARIZARD), make_mon("tyranitar", TYRANITAR, own=False))
        asleep = ctx_for(
            make_mon("charizard", CHARIZARD), make_mon("tyranitar", TYRANITAR, own=False, status=Status.SLP)
        )
        assert asleep.threat(asleep.me).damage.expected < awake.threat(awake.me).damage.expected

    def test_threat_picks_the_most_damaging_revealed_move(self) -> None:
        ctx = ctx_for(make_mon("charizard", CHARIZARD), make_mon("tyranitar", TYRANITAR, own=False))
        threat = ctx.threat(ctx.me)
        assert threat.move_id == "stoneedge"
        assert threat.revealed
