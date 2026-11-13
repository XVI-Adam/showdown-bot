"""Training callbacks: win-rate logging and critic warm-up."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class WinRateCallback(BaseCallback):
    """Log the win rate over the most recent finished battles.

    Reads the ``won`` flag the environment attaches to each finished episode;
    SB3 keeps the last 100 episodes in ``ep_info_buffer``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.history: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        episodes = [e for e in self.model.ep_info_buffer if "won" in e]
        if not episodes:
            return
        win_rate = float(np.mean([e["won"] for e in episodes]))
        mean_reward = float(np.mean([e["r"] for e in episodes]))
        self.logger.record("rollout/win_rate", win_rate)
        self.history.append(
            {"timesteps": self.num_timesteps, "win_rate": win_rate, "ep_rew_mean": mean_reward}
        )


def actor_parameters(policy: ActorCriticPolicy) -> list[nn.Parameter]:
    """Parameters that decide actions (as opposed to estimating value)."""
    params = list(policy.mlp_extractor.policy_net.parameters())
    params += list(policy.action_net.parameters())
    prior_scale = getattr(policy, "prior_scale", None)
    if prior_scale is not None and getattr(policy, "_learn_prior_scale", True):
        params.append(prior_scale)
    return params


def set_actor_trainable(policy: ActorCriticPolicy, trainable: bool) -> None:
    for param in actor_parameters(policy):
        param.requires_grad_(trainable)


class CriticWarmupCallback(BaseCallback):
    """Train only the value function for the first ``warmup_steps``.

    PPO improves the policy using the critic's judgement of which actions
    turned out better than expected. A hybrid policy starts out good (it plays
    the heuristic) while its critic starts out random, so the first updates
    would push a good policy around with meaningless advantages. Freezing the
    actor until the critic has seen some battles avoids that.
    """

    def __init__(self, warmup_steps: int) -> None:
        super().__init__()
        self.warmup_steps = warmup_steps
        self.frozen = False

    def _on_training_start(self) -> None:
        if self.warmup_steps > 0 and self.num_timesteps < self.warmup_steps:
            set_actor_trainable(self.model.policy, False)
            self.frozen = True

    def _on_rollout_start(self) -> None:
        if self.frozen and self.num_timesteps >= self.warmup_steps:
            set_actor_trainable(self.model.policy, True)
            self.frozen = False

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if self.frozen:  # never leave a saved model with a frozen actor
            set_actor_trainable(self.model.policy, True)
            self.frozen = False
