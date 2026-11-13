"""Battle observations: shape, range, and alignment with the action space."""

import numpy as np
import pytest
from factories import make_battle, make_mon
from poke_env.battle import SideCondition, Status
from poke_env.environment import SinglesEnv

from psbot.rl.env import PSBotEnv, RewardConfig
from psbot.rl.observation import (
    ACTIVE_FEATURES,
    GLOBAL_FEATURES,
    MOVE_FEATURES,
    OBSERVATION_SIZE,
    TEAM_FEATURES,
    embed_battle,
    feature_names,
)

NAMES = feature_names()
CHARIZARD = ["flamethrower", "airslash", "earthquake", "roost"]


def feature(obs: np.ndarray, name: str) -> float:
    return float(obs[NAMES.index(name)])


def standard_battle(**kwargs):
    me = make_mon("charizard", CHARIZARD)
    opp = make_mon("venusaur", ["gigadrain", "sludgebomb"], own=False)
    bench = [make_mon("blastoise", ["surf"]), make_mon("snorlax", ["bodyslam"], hp=0.5)]
    return make_battle(me, opp, bench=bench, **kwargs)


class TestShape:
    def test_size_matches_the_documented_layout(self) -> None:
        expected = 4 * len(MOVE_FEATURES) + 6 * len(TEAM_FEATURES) + 2 * len(ACTIVE_FEATURES) + len(GLOBAL_FEATURES)
        assert OBSERVATION_SIZE == expected == 112
        assert len(NAMES) == OBSERVATION_SIZE
        assert len(set(NAMES)) == OBSERVATION_SIZE  # every index has a unique name

    def test_vector_is_float32_of_the_right_shape_within_bounds(self) -> None:
        obs = embed_battle(standard_battle())
        assert obs.shape == (OBSERVATION_SIZE,)
        assert obs.dtype == np.float32
        assert obs.min() >= -1.0 and obs.max() <= 1.0

    def test_extreme_values_stay_in_bounds(self) -> None:
        me = make_mon("charizard", ["explosion", "extremespeed"], boosts={"atk": 6, "spe": -6})
        battle = make_battle(me, make_mon("shedinja", own=False, boosts={"def": -6}))
        battle.side_conditions[SideCondition.SPIKES] = 3
        obs = embed_battle(battle)
        assert obs.min() >= -1.0 and obs.max() <= 1.0

    def test_handles_missing_pokemon(self) -> None:
        bench = [make_mon("blastoise", ["surf"]), make_mon("snorlax", ["bodyslam"])]
        no_active = embed_battle(make_battle(None, make_mon("venusaur", own=False), bench=bench, force_switch=True))
        no_opponent = embed_battle(make_battle(make_mon("charizard", CHARIZARD), None))
        assert np.isfinite(no_active).all() and np.isfinite(no_opponent).all()


class TestMoveFeatures:
    def test_type_effectiveness_stab_and_immunity(self) -> None:
        obs = embed_battle(standard_battle())
        # Flamethrower: Fire vs Grass/Poison is 2x, and Charizard is a Fire type.
        assert feature(obs, "move0.effectiveness") == pytest.approx(0.5)
        assert feature(obs, "move0.stab") == 1.0
        assert feature(obs, "move0.special") == 1.0
        # Earthquake: Ground vs Grass/Poison is neutral, no STAB, physical.
        assert feature(obs, "move2.effectiveness") == 0.0
        assert feature(obs, "move2.stab") == 0.0
        assert feature(obs, "move2.physical") == 1.0
        # Roost is a status move.
        assert feature(obs, "move3.base_power") == 0.0
        assert feature(obs, "move3.physical") == feature(obs, "move3.special") == 0.0

    def test_immune_target_is_flagged(self) -> None:
        me = make_mon("garchomp", ["earthquake", "outrage"])
        obs = embed_battle(make_battle(me, make_mon("charizard", own=False)))
        assert feature(obs, "move0.immune") == 1.0
        assert feature(obs, "move1.immune") == 0.0

    def test_accuracy_and_priority(self) -> None:
        me = make_mon("dragonite", ["extremespeed", "stoneedge"])
        obs = embed_battle(make_battle(me, make_mon("mew", own=False)))
        assert feature(obs, "move0.priority") == pytest.approx(2 / 3)
        assert feature(obs, "move1.accuracy") == pytest.approx(0.8)

    def test_disabled_move_keeps_its_slot(self) -> None:
        """Features must stay on the action they describe when a move is unavailable."""
        battle = standard_battle()
        battle._available_moves = [m for m in battle.available_moves if m.id != "airslash"]
        obs = embed_battle(battle)
        assert [feature(obs, f"move{i}.available") for i in range(4)] == [1.0, 0.0, 1.0, 1.0]
        # Earthquake is still described in slot 2, the slot action 8 selects.
        assert feature(obs, "move2.physical") == 1.0

    def test_unused_move_slots_are_zero(self) -> None:
        me = make_mon("ditto", ["transform"])
        obs = embed_battle(make_battle(me, make_mon("mew", own=False)))
        start = NAMES.index("move1.available")
        assert not obs[start : start + 3 * len(MOVE_FEATURES)].any()


class TestTeamFeatures:
    def test_slots_follow_team_order(self) -> None:
        obs = embed_battle(standard_battle())
        assert feature(obs, "team0.is_active") == 1.0  # charizard
        assert feature(obs, "team1.is_active") == 0.0  # blastoise
        assert feature(obs, "team2.hp") == pytest.approx(0.5, abs=0.01)  # snorlax
        assert feature(obs, "team3.hp") == 0.0  # empty slot

    def test_matchup_features(self) -> None:
        obs = embed_battle(standard_battle())
        # Charizard's Fire/Flying hits Venusaur 2x; Venusaur's Grass/Poison is neutral at best.
        assert feature(obs, "team0.offense") == pytest.approx(0.5)
        assert feature(obs, "team0.weakness") == 0.0
        # Blastoise's Water is resisted; Venusaur's Grass hits it 2x.
        assert feature(obs, "team1.offense") == pytest.approx(-0.5)
        assert feature(obs, "team1.weakness") == pytest.approx(0.5)

    def test_fainted_pokemon_are_not_switchable(self) -> None:
        battle = standard_battle()
        fainted = list(battle.team.values())[1]
        fainted.faint()
        battle._available_switches = [m for m in battle.available_switches if m is not fainted]
        obs = embed_battle(battle)
        assert feature(obs, "team1.fainted") == 1.0
        assert feature(obs, "team1.switchable") == 0.0
        assert feature(obs, "team2.switchable") == 1.0
        assert feature(obs, "global.our_fainted") == pytest.approx(1 / 6)


class TestActiveAndGlobalFeatures:
    def test_hp_status_and_boosts(self) -> None:
        me = make_mon("charizard", CHARIZARD, hp=0.5, status=Status.BRN, boosts={"atk": 2, "spe": -1})
        opp = make_mon("venusaur", own=False, hp=0.25, status=Status.TOX)
        obs = embed_battle(make_battle(me, opp))
        assert feature(obs, "me.hp") == pytest.approx(0.5, abs=0.01)
        assert feature(obs, "me.brn") == 1.0 and feature(obs, "me.par") == 0.0
        assert feature(obs, "me.boost_atk") == pytest.approx(2 / 6)
        assert feature(obs, "me.boost_spe") == pytest.approx(-1 / 6)
        assert feature(obs, "opp.hp") == pytest.approx(0.25)
        assert feature(obs, "opp.psn") == 1.0  # toxic shares the poison flag

    def test_speed_share_reflects_who_is_faster(self) -> None:
        fast = embed_battle(make_battle(make_mon("jolteon", ["thunderbolt"]), make_mon("snorlax", own=False)))
        slow = embed_battle(make_battle(make_mon("snorlax", ["bodyslam"]), make_mon("jolteon", own=False)))
        assert feature(fast, "global.speed_share") > 0.5 > feature(slow, "global.speed_share")

    def test_hazards_and_flags(self) -> None:
        battle = standard_battle(can_tera=True)
        battle.side_conditions[SideCondition.STEALTH_ROCK] = 4
        battle.opponent_side_conditions[SideCondition.SPIKES] = 2
        obs = embed_battle(battle)
        assert feature(obs, "global.our_stealth_rock") == 1.0
        assert feature(obs, "global.their_spikes") == pytest.approx(2 / 3)
        assert feature(obs, "global.their_stealth_rock") == 0.0
        assert feature(obs, "global.can_tera") == 1.0
        assert feature(obs, "global.force_switch") == 0.0
        assert feature(obs, "global.their_revealed") == pytest.approx(1 / 6)


class TestAlignmentWithActionMask:
    """The observation's availability flags must agree with poke-env's mask."""

    def test_mask_has_one_entry_per_action(self) -> None:
        assert len(SinglesEnv.get_action_mask(standard_battle())) == 26 == SinglesEnv.get_action_space_size(9)

    def test_availability_features_match_the_mask(self) -> None:
        battle = standard_battle(can_tera=True)
        battle._available_moves = [m for m in battle.available_moves if m.id != "roost"]
        mask = SinglesEnv.get_action_mask(battle)
        obs = embed_battle(battle)
        for slot in range(4):
            assert feature(obs, f"move{slot}.available") == mask[6 + slot]  # plain move
            assert mask[22 + slot] == mask[6 + slot]  # same move with Terastallize
        for slot in range(6):
            assert feature(obs, f"team{slot}.switchable") == mask[slot]
        assert not any(mask[10:22])  # mega, Z-move and dynamax do not exist in Gen 9

    def test_forced_switch_only_allows_switches(self) -> None:
        bench = [make_mon("blastoise", ["surf"]), make_mon("snorlax", ["bodyslam"])]
        battle = make_battle(None, make_mon("venusaur", own=False), bench=bench, force_switch=True)
        mask = SinglesEnv.get_action_mask(battle)
        assert [i for i, legal in enumerate(mask) if legal] == [0, 1]


class TestReward:
    @pytest.fixture
    def env(self):
        environment = PSBotEnv(start_listening=False)  # never connects
        yield environment

    def test_default_weights_keep_the_official_proportions(self) -> None:
        cfg = RewardConfig()
        assert cfg.victory == 1.0
        assert cfg.fainted / cfg.hp == pytest.approx(2.0)
        assert cfg.hp / cfg.status == pytest.approx(2.0)

    def test_reward_is_the_change_in_state_value(self, env) -> None:
        me = make_mon("charizard", CHARIZARD)
        opp = make_mon("venusaur", own=False)
        battle = make_battle(me, opp)
        env.calc_reward(battle)  # establishes the baseline
        assert env.calc_reward(battle) == 0.0  # nothing changed

        opp._current_hp = 40  # we dealt 60%
        assert env.calc_reward(battle) == pytest.approx(0.6 * RewardConfig().hp)

        me._current_hp = me.max_hp // 2  # we took ~50%
        assert env.calc_reward(battle) == pytest.approx(-0.5 * RewardConfig().hp, abs=1e-3)

    def test_knockouts_and_winning_are_rewarded(self, env) -> None:
        me = make_mon("charizard", CHARIZARD)
        opp = make_mon("venusaur", own=False)
        battle = make_battle(me, opp)
        env.calc_reward(battle)
        opp.faint()
        battle._won = True
        cfg = RewardConfig()
        assert env.calc_reward(battle) == pytest.approx(cfg.hp + cfg.fainted + cfg.victory)

    def test_observation_space_matches_the_encoder(self, env) -> None:
        space = env.observation_spaces[env.possible_agents[0]]
        assert space["observation"].shape == (OBSERVATION_SIZE,)
        assert space["action_mask"].shape == (26,)
        obs = embed_battle(standard_battle())
        assert space["observation"].contains(obs)
