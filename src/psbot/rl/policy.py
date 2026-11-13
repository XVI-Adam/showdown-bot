"""PPO policy with action masking, following the official poke-env example.

The network outputs one logit per action. Before sampling we add ``-inf`` to
the logits of illegal actions, which makes their probability exactly zero, so
the agent cannot choose them and gradients never push toward them.

One addition to the official example: ``get_distribution`` is overridden too.
SB3's ``predict`` goes through it rather than ``forward``, so without the
override ``model.predict`` would silently reuse a stale mask.
"""

from __future__ import annotations

from typing import Any

import torch
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from psbot.rl.hybrid import PRIOR_CLIP, PRIOR_SLICE

NET_ARCH = [128, 128]


class ObservationExtractor(BaseFeaturesExtractor):
    """Pass the ``observation`` entry of the dict observation to the MLP."""

    def __init__(self, observation_space: spaces.Dict) -> None:
        features_dim = int(observation_space["observation"].shape[0])
        super().__init__(observation_space, features_dim=features_dim)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return obs["observation"]


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set illegal actions' logits to ``-inf``.

    A row with no legal action at all would become all ``-inf`` and produce
    NaN probabilities, so such rows are left unmasked. poke-env always offers
    at least one action, so this is a safety net.
    """
    legal = mask > 0
    has_legal = legal.any(dim=-1, keepdim=True)
    legal = legal | ~has_legal
    return logits.masked_fill(~legal, float("-inf"))


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic MLP that never assigns probability to masked actions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("net_arch", NET_ARCH)
        kwargs.setdefault("features_extractor_class", ObservationExtractor)
        super().__init__(*args, **kwargs)
        self._mask: torch.Tensor | None = None

    def _remember(self, obs: dict[str, torch.Tensor]) -> None:
        """Stash what ``_get_action_dist_from_latent`` needs from the observation."""
        self._mask = obs["action_mask"]

    def forward(self, obs: dict[str, torch.Tensor], deterministic: bool = False):
        self._remember(obs)
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs: dict[str, torch.Tensor], actions: torch.Tensor):
        self._remember(obs)
        return super().evaluate_actions(obs, actions)

    def get_distribution(self, obs: dict[str, torch.Tensor]) -> Distribution:
        self._remember(obs)
        return super().get_distribution(obs)

    def _action_logits(self, latent_pi: torch.Tensor) -> torch.Tensor:
        return self.action_net(latent_pi)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        logits = self._action_logits(latent_pi)
        if self._mask is not None:
            logits = masked_logits(logits, self._mask)
        return self.action_dist.proba_distribution(action_logits=logits)


class HybridActorCriticPolicy(MaskedActorCriticPolicy):
    """Masked policy whose logits start from the heuristic engine's scores.

    ``logits = network(observation) + prior_scale * heuristic_score``

    The network's output layer starts near zero, so an untrained hybrid plays
    almost exactly like the heuristic agent. Training then learns a residual:
    where to agree with the heuristic and where to overrule it. Nothing is
    forbidden; a large enough learned logit beats any heuristic score, and
    ``prior_scale`` itself is learned, so the agent can also decide how much
    to trust the heuristic overall.
    """

    def __init__(
        self,
        *args: Any,
        prior_scale: float = 5.0,
        learn_prior_scale: bool = True,
        **kwargs: Any,
    ) -> None:
        # Plain attributes, read by ``_build`` (which runs inside super().__init__).
        self._prior_scale_init = float(prior_scale)
        self._learn_prior_scale = bool(learn_prior_scale)
        self._prior: torch.Tensor | None = None
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Any) -> None:
        # Created before the optimizer so that it is trained with the network.
        self.prior_scale = nn.Parameter(
            torch.tensor(self._prior_scale_init), requires_grad=self._learn_prior_scale
        )
        super()._build(lr_schedule)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(prior_scale=self._prior_scale_init, learn_prior_scale=self._learn_prior_scale)
        return data

    def _remember(self, obs: dict[str, torch.Tensor]) -> None:
        super()._remember(obs)
        # Stored scaled into [-1, 1]; undo that to get scores in HP-bar units.
        self._prior = obs["observation"][..., PRIOR_SLICE] * PRIOR_CLIP

    def _action_logits(self, latent_pi: torch.Tensor) -> torch.Tensor:
        logits = super()._action_logits(latent_pi)
        if self._prior is not None:
            logits = logits + self.prior_scale * self._prior
        return logits
