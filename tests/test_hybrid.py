"""Hybrid agent: heuristic features, action-score alignment, and the logit prior."""

import numpy as np
import pytest
import torch
from factories import make_battle, make_mon
from gymnasium import spaces
from poke_env.battle import Move, PokemonType
from poke_env.environment import SinglesEnv

from psbot.heuristics import HeuristicEngine
from psbot.rl.env import REWARD_PRESETS, PSBotEnv
from psbot.rl.hybrid import (
    HEURISTIC_FEATURE_SIZE,
    HYBRID_OBSERVATION_SIZE,
    PRIOR_CLIP,
    PRIOR_SLICE,
    TERA_NUDGE,
    embed_battle_hybrid,
    encoder_for_size,
    heuristic_action_scores,
    heuristic_features,
    hybrid_feature_names,
)
from psbot.rl.observation import OBSERVATION_SIZE, embed_battle
from psbot.rl.policy import HybridActorCriticPolicy, MaskedActorCriticPolicy

ENGINE = HeuristicEngine()
NAMES = hybrid_feature_names()
TYRANITAR = ["stoneedge", "crunch", "earthquake", "firepunch"]
VENUSAUR = ["gigadrain", "sludgebomb", "sleeppowder", "synthesis"]
N_ACTIONS = 26


def feature(obs: np.ndarray, name: str) -> float:
    return float(obs[NAMES.index(name)])


def standard_battle(**kwargs):
    me = make_mon("charizard", ["flamethrower", "airslash", "earthquake", "roost"])
    opp = make_mon("venusaur", VENUSAUR, own=False)
    bench = [make_mon("blastoise", ["surf"]), make_mon("snorlax", ["bodyslam"], hp=0.5)]
    return make_battle(me, opp, bench=bench, **kwargs)


class TestActionScores:
    def test_scores_land_on_the_actions_they_describe(self) -> None:
        battle = standard_battle()
        scores = heuristic_action_scores(battle, ENGINE)
        by_label = {e.label: e.total for e in ENGINE.evaluate(battle)}
        assert scores.shape == (N_ACTIONS,)
        # Moves: actions 6-9 in the active Pokémon's move order.
        for slot, move_id in enumerate(["flamethrower", "airslash", "earthquake", "roost"]):
            assert scores[6 + slot] == pytest.approx(by_label[f"move {move_id}"])
        # Switches: actions 0-5 in team order (0 is the active Pokémon itself).
        assert scores[1] == pytest.approx(by_label["switch blastoise"])
        assert scores[2] == pytest.approx(by_label["switch snorlax"])

    def test_best_scoring_action_is_the_heuristic_agents_choice(self) -> None:
        for battle in (
            standard_battle(),
            make_battle(
                make_mon("charizard", ["flamethrower", "airslash"]),
                make_mon("tyranitar", TYRANITAR, own=False),
                bench=[make_mon("hippowdon", ["earthquake", "slackoff"])],
            ),
        ):
            scores = heuristic_action_scores(battle, ENGINE)
            mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
            best_action = int(np.argmax(np.where(mask, scores, -np.inf)))
            order = SinglesEnv.action_to_order(np.int64(best_action), battle, fake=True)
            chosen = ENGINE.evaluate(battle)[0]
            assert order.order is chosen.target

    def test_illegal_actions_score_zero(self) -> None:
        battle = standard_battle()
        scores = heuristic_action_scores(battle, ENGINE)
        mask = np.array(SinglesEnv.get_action_mask(battle))
        assert not scores[mask == 0].any()

    def test_disabled_move_does_not_shift_other_scores(self) -> None:
        battle = standard_battle()
        full = heuristic_action_scores(battle, ENGINE)
        battle._available_moves = [m for m in battle.available_moves if m.id != "airslash"]
        partial = heuristic_action_scores(battle, ENGINE)
        assert partial[7] == 0  # airslash, now illegal
        assert partial[8] == pytest.approx(full[8])  # earthquake stays in its slot

    def test_tera_variant_is_nudged_up_only_when_recommended(self) -> None:
        me = make_mon("charizard", ["flamethrower", "airslash"], tera_type=PokemonType.FIRE)
        opp = make_mon("blissey", ["seismictoss", "softboiled", "toxic", "stealthrock"], own=False)
        scores = heuristic_action_scores(make_battle(me, opp, can_tera=True), ENGINE)
        assert scores[22] == pytest.approx(scores[6] + TERA_NUDGE)  # Tera Fire Flamethrower
        assert scores[23] == pytest.approx(scores[7] - TERA_NUDGE)  # Air Slash gains nothing

    def test_no_tera_scores_when_tera_is_unavailable(self) -> None:
        scores = heuristic_action_scores(standard_battle(can_tera=False), ENGINE)
        assert not scores[22:].any()

    def test_forced_single_action(self) -> None:
        me = make_mon("snorlax", ["bodyslam"])
        battle = make_battle(me, make_mon("mew", own=False))
        battle._available_moves = [Move("recharge", gen=9)]
        scores = heuristic_action_scores(battle, ENGINE)
        assert not scores.any()  # nothing to rank; the mask leaves one action


class TestHybridObservation:
    def test_layout(self) -> None:
        assert HEURISTIC_FEATURE_SIZE == 26 + 4 * 3 + 6 * 2 + 3 == 53
        assert HYBRID_OBSERVATION_SIZE == OBSERVATION_SIZE + 53 == 165
        assert len(NAMES) == len(set(NAMES)) == HYBRID_OBSERVATION_SIZE
        assert PRIOR_SLICE == slice(112, 138)

    def test_starts_with_the_basic_observation(self) -> None:
        battle = standard_battle()
        obs = embed_battle_hybrid(battle)
        assert obs.shape == (HYBRID_OBSERVATION_SIZE,) and obs.dtype == np.float32
        assert np.array_equal(obs[:OBSERVATION_SIZE], embed_battle(battle))
        assert obs.min() >= -1.0 and obs.max() <= 1.0

    def test_prior_slice_holds_the_scaled_action_scores(self) -> None:
        battle = standard_battle()
        obs = embed_battle_hybrid(battle)
        scores = heuristic_action_scores(battle, ENGINE)
        assert obs[PRIOR_SLICE] * PRIOR_CLIP == pytest.approx(np.clip(scores, -PRIOR_CLIP, PRIOR_CLIP), abs=1e-5)

    def test_detail_features(self) -> None:
        obs = embed_battle_hybrid(standard_battle())
        # Flamethrower is super effective: the most expected damage of the four moves.
        damages = [feature(obs, f"move{i}.expected_damage") for i in range(4)]
        assert damages[0] == max(damages) > 0.5
        assert feature(obs, "move3.expected_damage") == 0.0  # Roost
        assert feature(obs, "move0.act_probability") == 1.0  # Charizard outspeeds Venusaur
        # Blastoise would switch into a super-effective Giga Drain.
        assert feature(obs, "team1.entry_damage") > 0.3
        assert feature(obs, "team0.entry_damage") == 0.0  # the active Pokémon is not a switch
        assert 0 < feature(obs, "heuristic.threat_expected") < 1

    def test_engine_failure_degrades_to_the_basic_observation(self) -> None:
        class BrokenEngine(HeuristicEngine):
            def evaluate(self, battle, ctx=None):
                raise RuntimeError("boom")

        features = heuristic_features(standard_battle(), BrokenEngine())
        assert features.shape == (HEURISTIC_FEATURE_SIZE,)
        assert not features.any()

    def test_handles_forced_switch_and_missing_opponent(self) -> None:
        bench = [make_mon("hippowdon", ["earthquake"]), make_mon("talonflame", ["bravebird"])]
        forced = embed_battle_hybrid(
            make_battle(None, make_mon("tyranitar", TYRANITAR, own=False), bench=bench, force_switch=True)
        )
        no_opponent = embed_battle_hybrid(make_battle(make_mon("charizard", ["flamethrower", "roost"]), None))
        assert np.isfinite(forced).all() and np.isfinite(no_opponent).all()
        # Hippowdon (slot 0) is the better switch into Tyranitar.
        assert forced[PRIOR_SLICE][0] > forced[PRIOR_SLICE][1]

    def test_encoder_is_identified_by_observation_size(self) -> None:
        assert encoder_for_size(OBSERVATION_SIZE) is embed_battle
        assert encoder_for_size(HYBRID_OBSERVATION_SIZE) is embed_battle_hybrid
        with pytest.raises(ValueError):
            encoder_for_size(7)


class TestHybridPolicy:
    SPACE = spaces.Dict(
        {
            "observation": spaces.Box(-1.0, 1.0, shape=(HYBRID_OBSERVATION_SIZE,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(N_ACTIONS,), dtype=np.int8),
        }
    )

    def make_policy(self, **kwargs) -> HybridActorCriticPolicy:
        torch.manual_seed(0)
        return HybridActorCriticPolicy(self.SPACE, spaces.Discrete(N_ACTIONS), lambda _: 3e-4, **kwargs)

    @staticmethod
    def obs_for(battle) -> dict[str, np.ndarray]:
        return {
            "observation": embed_battle_hybrid(battle),
            "action_mask": np.array(SinglesEnv.get_action_mask(battle), dtype=np.int8),
        }

    def test_prior_scale_is_a_trained_parameter(self) -> None:
        policy = self.make_policy(prior_scale=7.0)
        assert policy.prior_scale.item() == 7.0
        optimised = [p for group in policy.optimizer.param_groups for p in group["params"]]
        assert any(p is policy.prior_scale for p in optimised)

    def test_prior_scale_can_be_frozen(self) -> None:
        assert not self.make_policy(learn_prior_scale=False).prior_scale.requires_grad

    def test_untrained_policy_plays_the_heuristics_choice(self) -> None:
        policy = self.make_policy()
        battles = [
            standard_battle(),
            make_battle(
                make_mon("charizard", ["flamethrower", "airslash"]),
                make_mon("tyranitar", TYRANITAR, own=False),
                bench=[make_mon("hippowdon", ["earthquake", "slackoff"])],
            ),
            make_battle(
                make_mon("blissey", ["softboiled", "seismictoss"], hp=0.3),
                make_mon("umbreon", ["foulplay", "protect", "toxic", "wish"], own=False),
            ),
        ]
        for battle in battles:
            action, _ = policy.predict(self.obs_for(battle), deterministic=True)
            order = SinglesEnv.action_to_order(np.int64(action), battle, fake=True)
            assert order.order is ENGINE.evaluate(battle)[0].target

    def test_learned_logits_can_overrule_the_prior(self) -> None:
        """The heuristic is a starting point, not a constraint."""
        policy = self.make_policy()
        battle = standard_battle()
        obs = self.obs_for(battle)
        heuristic_choice, _ = policy.predict(obs, deterministic=True)
        with torch.no_grad():  # as if training had learned to prefer switching to Blastoise
            policy.action_net.bias[1] = 50.0
        overruled, _ = policy.predict(obs, deterministic=True)
        assert int(heuristic_choice) == 6 and int(overruled) == 1

    def test_zero_prior_scale_ignores_the_heuristic(self) -> None:
        hybrid = self.make_policy(prior_scale=0.0)
        obs = {k: torch.as_tensor(v).unsqueeze(0) for k, v in self.obs_for(standard_battle()).items()}
        hybrid._remember(obs)
        latent = torch.zeros(1, 128)
        assert torch.equal(hybrid._action_logits(latent), hybrid.action_net(latent))

    def test_never_picks_illegal_actions(self) -> None:
        policy = self.make_policy()
        rng = np.random.default_rng(0)
        mask = np.zeros((256, N_ACTIONS), dtype=np.int8)
        for row in mask:
            row[rng.choice(N_ACTIONS, size=rng.integers(1, 14), replace=False)] = 1
        obs = {
            "observation": rng.uniform(-1, 1, size=(256, HYBRID_OBSERVATION_SIZE)).astype(np.float32),
            "action_mask": mask,
        }
        for deterministic in (True, False):
            actions, _ = policy.predict(obs, deterministic=deterministic)
            assert mask[np.arange(256), actions].all()

    def test_features_only_variant_accepts_the_hybrid_observation(self) -> None:
        policy = MaskedActorCriticPolicy(self.SPACE, spaces.Discrete(N_ACTIONS), lambda _: 3e-4)
        assert policy.mlp_extractor.policy_net[0].in_features == HYBRID_OBSERVATION_SIZE
        action, _ = policy.predict(self.obs_for(standard_battle()), deterministic=True)
        assert 0 <= int(action) < N_ACTIONS


class TestEnvironmentOptions:
    def test_hybrid_environment_declares_the_larger_observation(self) -> None:
        env = PSBotEnv(observation="hybrid", start_listening=False)
        space = env.observation_spaces[env.possible_agents[0]]
        assert space["observation"].shape == (HYBRID_OBSERVATION_SIZE,)
        assert space["observation"].contains(env.embed_battle(standard_battle()))

    def test_unknown_observation_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown observation"):
            PSBotEnv(observation="telepathy", start_listening=False)

    def test_reward_presets(self) -> None:
        sparse, shaped, light = (REWARD_PRESETS[k] for k in ("sparse", "shaped", "light"))
        assert (sparse.fainted, sparse.hp, sparse.status) == (0.0, 0.0, 0.0)
        assert sparse.victory == shaped.victory == light.victory == 1.0
        assert light.hp == pytest.approx(shaped.hp / 2)


class TestCriticWarmup:
    def test_freezing_the_actor_leaves_the_critic_trainable(self) -> None:
        from psbot.rl.callbacks import actor_parameters, set_actor_trainable

        policy = TestHybridPolicy().make_policy()
        set_actor_trainable(policy, False)
        assert not any(p.requires_grad for p in actor_parameters(policy))
        assert not policy.prior_scale.requires_grad
        assert all(p.requires_grad for p in policy.mlp_extractor.value_net.parameters())
        assert all(p.requires_grad for p in policy.value_net.parameters())

        set_actor_trainable(policy, True)
        assert all(p.requires_grad for p in actor_parameters(policy))

    def test_a_frozen_prior_scale_stays_frozen(self) -> None:
        from psbot.rl.callbacks import actor_parameters, set_actor_trainable

        policy = TestHybridPolicy().make_policy(learn_prior_scale=False)
        assert all(p is not policy.prior_scale for p in actor_parameters(policy))
        set_actor_trainable(policy, True)
        assert not policy.prior_scale.requires_grad


class TestActionConversionFallback:
    """A rare poke-env mapping failure must not crash training, and must be counted."""

    @staticmethod
    def env_and_battle():
        env = PSBotEnv(start_listening=False)
        battle = standard_battle()
        battle._player_username = env.agent2.username  # this is the opponent's battle
        return env, battle

    def test_unmappable_order_falls_back_to_a_legal_action_and_is_counted(self) -> None:
        from poke_env.player import Player

        env, battle = self.env_and_battle()
        # The opponent picks a legal move that is not among the first four known moves.
        extra = Move("energyball", gen=9)
        battle._available_moves = [*battle.available_moves, extra]
        action = env.order_to_action(Player.create_order(extra), battle)
        assert SinglesEnv.get_action_mask(battle)[int(action)] == 1
        assert (env.opponent_conversion_errors, env.agent_conversion_errors) == (1, 0)

    def test_valid_orders_convert_normally_without_counting(self) -> None:
        from poke_env.player import Player

        env, battle = self.env_and_battle()
        action = env.order_to_action(Player.create_order(battle.available_moves[2]), battle)
        assert int(action) == 8
        assert env.opponent_conversion_errors == 0

    def test_illegal_agent_action_is_replaced_and_counted(self) -> None:
        env = PSBotEnv(start_listening=False)
        battle = standard_battle()
        battle._player_username = env.agent1.username
        order = env.action_to_order(np.int64(15), battle)  # Z-move slot: never legal in Gen 9
        assert str(order) in {str(o) for o in battle.valid_orders}
        assert (env.agent_conversion_errors, env.opponent_conversion_errors) == (1, 0)
