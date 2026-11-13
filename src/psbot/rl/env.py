"""Gymnasium environment for training, built on poke-env's ``SinglesEnv``.

poke-env supplies the hard parts: the 26-action space, the legal-action mask,
and translating actions to battle orders. We only define what the agent sees
(``embed_battle``) and what it is rewarded for (``calc_reward``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from gymnasium.spaces import Box
from poke_env.battle import AbstractBattle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import BattleOrder
from stable_baselines3.common.monitor import Monitor

from psbot.config import BATTLE_FORMAT, local_server_configuration
from psbot.opponents import make_opponent
from psbot.rl.hybrid import ENCODERS


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights.

    Same proportions as the official poke-env example (win 30, faint 2, HP 1,
    status 0.5), divided by 30 so a win is worth exactly 1 and a whole game's
    return stays within about +/-1.6. Small targets keep PPO's value loss
    well-scaled without reward normalisation.

    The reward is the *change* in a state value, so it is zero-sum and dense:
    damage dealt and taken register on the turn they happen, while the win
    bonus keeps the final outcome dominant.
    """

    victory: float = 1.0
    fainted: float = 2.0 / 30
    hp: float = 1.0 / 30
    status: float = 0.5 / 30


# Named reward schemes to compare. "shaped" is the default above.
REWARD_PRESETS: dict[str, RewardConfig] = {
    "shaped": RewardConfig(),
    # Half the shaping: progress still registers, the result matters more.
    "light": RewardConfig(fainted=1.0 / 30, hp=0.5 / 30, status=0.25 / 30),
    # Win or lose only. Unbiased, but the signal arrives once per battle.
    "sparse": RewardConfig(fainted=0.0, hp=0.0, status=0.0),
}


class PSBotEnv(SinglesEnv):
    """Two-player battle environment with our observation and reward."""

    def __init__(
        self,
        reward: RewardConfig | None = None,
        observation: str = "basic",
        embed_opponent: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        :param observation: ``"basic"`` (112 features) or ``"hybrid"`` (basic
            plus the heuristic engine's analysis).
        :param embed_opponent: set False when the second player is scripted,
            as in ``make_env``: its observation is never used, and skipping it
            saves a full heuristic evaluation per step.
        """
        super().__init__(**kwargs)
        if observation not in ENCODERS:
            raise ValueError(f"unknown observation {observation!r}; choose from {sorted(ENCODERS)}")
        self.reward_config = reward or RewardConfig()
        self.observation_kind = observation
        self.embed_opponent = embed_opponent
        self._observation_size, self._encoder = ENCODERS[observation]
        self.agent_conversion_errors = 0
        self.opponent_conversion_errors = 0
        # poke-env wraps this into Dict(observation=..., action_mask=...).
        self.observation_spaces = {
            agent: Box(-1.0, 1.0, shape=(self._observation_size,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def calc_reward(self, battle: AbstractBattle) -> float:
        cfg = self.reward_config
        return self.reward_computing_helper(
            battle,
            fainted_value=cfg.fainted,
            hp_value=cfg.hp,
            status_value=cfg.status,
            victory_value=cfg.victory,
        )

    # --- action conversion -------------------------------------------------
    # poke-env maps move actions onto the active Pokémon's first four known
    # moves. In rare states (a Pokémon tracked with more than four moves) a
    # legal order has no action index, or vice versa, and strict conversion
    # raises. One such turn in millions should not end a training run, so we
    # substitute a random legal move for that turn and count it. The counts
    # are reported with every finished battle and saved in the training log.

    def action_to_order(
        self, action: np.int64, battle: Any, fake: bool = False, strict: bool = True
    ) -> BattleOrder:
        try:
            return SinglesEnv.action_to_order(action, battle, fake=fake, strict=True)
        except ValueError:
            self._count_conversion_error(battle)
            return SinglesEnv.action_to_order(action, battle, fake=fake, strict=False)

    def order_to_action(
        self, order: BattleOrder, battle: Any, fake: bool = False, strict: bool = True
    ) -> np.int64:
        try:
            return SinglesEnv.order_to_action(order, battle, fake=fake, strict=True)
        except ValueError:
            self._count_conversion_error(battle)
            return SinglesEnv.order_to_action(order, battle, fake=fake, strict=False)

    def _count_conversion_error(self, battle: Any) -> None:
        if battle.player_username == self.agent1.username:
            self.agent_conversion_errors += 1
        else:
            self.opponent_conversion_errors += 1

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        if not self.embed_opponent and battle.player_username == self.agent2.username:
            return np.zeros(self._observation_size, dtype=np.float32)
        return self._encoder(battle)

    def get_additional_info(self) -> dict[str, dict[str, Any]]:
        """Report the result when a battle ends, so training can log win rate."""
        info: dict[str, dict[str, Any]] = {agent: {} for agent in self.possible_agents}
        battle = self.battle1
        if battle is not None and battle.finished:
            info[self.possible_agents[0]] = {
                "won": bool(battle.won),
                "turns": battle.turn,
                # Cumulative for this environment since it was created.
                "agent_conversion_errors": self.agent_conversion_errors,
                "opponent_conversion_errors": self.opponent_conversion_errors,
            }
        return info


def make_env(
    opponent_kind: str,
    reward: RewardConfig | None = None,
    monitor_path: str | None = None,
    observation: str = "basic",
) -> Monitor:
    """Build one single-agent training environment against a scripted opponent.

    The opponent never connects to the server (``start_listening=False``): the
    wrapper asks it for a move directly, using the environment's second player.
    """
    opponent = make_opponent(opponent_kind, battle_format=BATTLE_FORMAT, start_listening=False)
    env = PSBotEnv(
        reward=reward,
        observation=observation,
        embed_opponent=False,
        battle_format=BATTLE_FORMAT,
        server_configuration=local_server_configuration(),
        log_level=logging.ERROR,
        open_timeout=None,
    )
    return Monitor(
        SingleAgentWrapper(env, opponent),
        filename=monitor_path,
        info_keywords=("won", "turns", "agent_conversion_errors", "opponent_conversion_errors"),
    )


def env_factory(
    opponent_kind: str,
    reward: RewardConfig | None = None,
    monitor_path: str | None = None,
    observation: str = "basic",
) -> Callable[[], Monitor]:
    """A zero-argument factory that ``SubprocVecEnv`` can ship to a subprocess."""

    def _factory() -> Monitor:
        return make_env(opponent_kind, reward, monitor_path, observation)

    return _factory
