"""Play battles with a trained policy, either plain PPO or the hybrid.

Mirrors the official example's ``PolicyPlayer``: build the same observation
dict the policy was trained on, ask it for an action, and let poke-env turn
the action index back into a battle order.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from poke_env.battle import AbstractBattle, Move, Pokemon
from poke_env.environment import SinglesEnv
from poke_env.player import BattleOrder, DefaultBattleOrder, Player
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

from psbot.recording import BattleRecorder
from psbot.rl.hybrid import HYBRID_OBSERVATION_SIZE, PRIOR_SLICE, encoder_for_size

logger = logging.getLogger(__name__)


class PPOPlayer(Player):
    """Chooses actions with a masked PPO policy.

    ``deterministic=True`` plays the most probable action (best for
    evaluation); ``False`` samples from the policy, as during training.
    Any action that turns out to be illegal is counted in
    ``n_illegal_actions`` and replaced by a random legal one.
    """

    def __init__(
        self,
        *args: Any,
        policy: BasePolicy,
        deterministic: bool = True,
        recorder: BattleRecorder | None = None,
        opponent_kind: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.deterministic = deterministic
        self.recorder = recorder
        self.opponent_kind = opponent_kind
        # Basic or hybrid observation, whichever this policy was trained on.
        size = int(policy.observation_space["observation"].shape[0])
        self._encode = encoder_for_size(size)
        self._is_hybrid = size == HYBRID_OBSERVATION_SIZE
        self.n_decisions = 0
        self.n_illegal_actions = 0
        # Hybrid only: decisions with a real choice, and how many of those
        # matched the heuristic engine's top-scoring action.
        self.n_choices = 0
        self.n_heuristic_agreements = 0

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kwargs: Any) -> "PPOPlayer":
        """Load a policy saved by ``scripts/train_ppo.py``."""
        model = PPO.load(str(path), device="cpu")
        return cls(policy=model.policy, **kwargs)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.wait:
            return DefaultBattleOrder()
        self.n_decisions += 1
        mask = np.asarray(SinglesEnv.get_action_mask(battle), dtype=np.int8)
        observation = {"observation": self._encode(battle), "action_mask": mask}
        action, _ = self.policy.predict(observation, deterministic=self.deterministic)
        action = np.int64(action)
        extra: dict[str, Any] = {}
        if self._is_hybrid and mask.sum() > 1:
            scores = np.where(mask == 1, observation["observation"][PRIOR_SLICE], -np.inf)
            agrees = bool(scores[action] >= scores.max())
            self.n_choices += 1
            self.n_heuristic_agreements += int(agrees)
            extra["agrees_with_heuristic"] = agrees
            if not agrees and self.recorder is not None:
                extra["heuristic_action"] = self._describe(np.int64(np.argmax(scores)), battle)
        try:
            if mask[action] != 1:
                raise ValueError(f"policy chose masked action {action}")
            order = SinglesEnv.action_to_order(action, battle, strict=True)
        except ValueError as exc:
            self.n_illegal_actions += 1
            logger.warning("illegal action on turn %s of %s: %s", battle.turn, battle.battle_tag, exc)
            order = self.choose_random_move(battle)
            extra["illegal"] = True
        if self.recorder is not None:
            self.recorder.record_action(battle, self._label(order), extra)
        return order

    @staticmethod
    def _label(order: BattleOrder) -> str:
        """'move earthquake', 'move earthquake +tera' or 'switch gliscor'."""
        target = getattr(order, "order", None)
        if isinstance(target, Move):
            return f"move {target.id}" + (" +tera" if getattr(order, "terastallize", False) else "")
        if isinstance(target, Pokemon):
            return f"switch {target.species}"
        return str(order)

    def _describe(self, action: np.int64, battle: AbstractBattle) -> str:
        try:
            return self._label(SinglesEnv.action_to_order(action, battle, fake=True, strict=True))
        except ValueError:
            return f"action {int(action)}"

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        if self.recorder is not None:
            self.recorder.finish_battle(battle, self.opponent_kind)
