"""Action masking in the PPO policy."""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from psbot.rl.observation import OBSERVATION_SIZE
from psbot.rl.policy import MaskedActorCriticPolicy, masked_logits

N_ACTIONS = 26
OBS_SPACE = spaces.Dict(
    {
        "observation": spaces.Box(-1.0, 1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32),
        "action_mask": spaces.Box(0, 1, shape=(N_ACTIONS,), dtype=np.int8),
    }
)


@pytest.fixture
def policy() -> MaskedActorCriticPolicy:
    torch.manual_seed(0)
    return MaskedActorCriticPolicy(OBS_SPACE, spaces.Discrete(N_ACTIONS), lr_schedule=lambda _: 3e-4)


def random_batch(n: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random observations with 1-13 legal actions each, like real battles."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((n, N_ACTIONS), dtype=np.int8)
    for row in mask:
        row[rng.choice(N_ACTIONS, size=rng.integers(1, 14), replace=False)] = 1
    observation = rng.uniform(-1, 1, size=(n, OBSERVATION_SIZE)).astype(np.float32)
    return {"observation": torch.as_tensor(observation), "action_mask": torch.as_tensor(mask)}


def test_masked_logits_sets_illegal_actions_to_minus_infinity() -> None:
    logits = torch.zeros(1, 4)
    out = masked_logits(logits, torch.tensor([[1, 0, 1, 0]]))
    assert out.tolist() == [[0.0, float("-inf"), 0.0, float("-inf")]]


def test_masked_logits_leaves_a_row_with_no_legal_action_untouched() -> None:
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = masked_logits(logits, torch.tensor([[0, 0], [0, 1]]))
    assert out[0].tolist() == [1.0, 2.0]  # safety net: no NaNs
    assert out[1].tolist() == [float("-inf"), 4.0]


def test_network_dimensions(policy) -> None:
    assert policy.features_extractor.features_dim == OBSERVATION_SIZE
    assert policy.action_net.out_features == N_ACTIONS
    assert policy.mlp_extractor.policy_net[0].in_features == OBSERVATION_SIZE


@pytest.mark.parametrize("deterministic", [True, False])
def test_forward_only_picks_legal_actions(policy, deterministic: bool) -> None:
    obs = random_batch(512)
    actions, values, log_probs = policy.forward(obs, deterministic=deterministic)
    chosen_legal = obs["action_mask"].gather(1, actions.unsqueeze(1)).squeeze(1)
    assert chosen_legal.all()
    assert values.shape == (512, 1)
    assert torch.isfinite(log_probs).all()


@pytest.mark.parametrize("deterministic", [True, False])
def test_predict_only_picks_legal_actions(policy, deterministic: bool) -> None:
    obs = {k: v.numpy() for k, v in random_batch(256, seed=1).items()}
    actions, _ = policy.predict(obs, deterministic=deterministic)
    assert obs["action_mask"][np.arange(256), actions].all()


def test_predict_uses_the_current_mask_not_a_stale_one(policy) -> None:
    """Regression: SB3's predict() bypasses forward(), so the mask must be
    captured in get_distribution() too."""
    first = random_batch(1, seed=2)
    policy.forward(first)  # leaves first's mask stashed on the policy
    only_action_5 = np.zeros(N_ACTIONS, dtype=np.int8)
    only_action_5[5] = 1
    obs = {"observation": first["observation"][0].numpy(), "action_mask": only_action_5}
    for _ in range(20):
        action, _ = policy.predict(obs, deterministic=False)
        assert int(action) == 5


def test_single_unbatched_observation_is_accepted(policy) -> None:
    batch = random_batch(1, seed=3)
    obs = {k: v[0].numpy() for k, v in batch.items()}
    action, _ = policy.predict(obs, deterministic=True)
    assert obs["action_mask"][int(action)] == 1


def test_evaluate_actions_gives_finite_values_for_legal_actions(policy) -> None:
    obs = random_batch(128, seed=4)
    actions, _, _ = policy.forward(obs)
    values, log_probs, entropy = policy.evaluate_actions(obs, actions)
    assert torch.isfinite(values).all()
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropy).all()


def test_forced_single_action_has_probability_one(policy) -> None:
    obs = random_batch(1, seed=5)
    obs["action_mask"] = torch.zeros(1, N_ACTIONS, dtype=torch.int8)
    obs["action_mask"][0, 7] = 1
    actions, _, log_probs = policy.forward(obs)
    assert int(actions[0]) == 7
    assert log_probs[0].item() == pytest.approx(0.0, abs=1e-6)
