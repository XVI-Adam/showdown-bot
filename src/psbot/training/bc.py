"""Behaviour cloning into the masked PPO policy.

The learner is the exact network the PPO agents use (``MaskedActorCriticPolicy``,
MLP 128-128 over the 112 basic features), inside an ordinary SB3 ``PPO``
object. So a cloned model is saved, loaded, played and evaluated like any
other checkpoint, and could later be fine-tuned with ``train_ppo.py --resume``.

Training minimises the negative log-likelihood of the teacher's action under
the masked policy. Illegal actions have probability exactly zero, so the loss
can never be lowered by putting weight on them and the cloned agent cannot
choose them. The critic is fitted on the side to the discounted battle outcome
(``DecisionDataset.value_targets``); the actor does not use it, but a PPO
fine-tune would start from a critic that already roughly knows which states
win, instead of a random one.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from psbot.rl.observation import OBSERVATION_SIZE
from psbot.rl.policy import MaskedActorCriticPolicy
from psbot.training.dataset import N_ACTIONS, DecisionDataset

# Matches the saved PPO agents exactly (see checkpoints/ppo-v1).
OBSERVATION_SPACE = spaces.Dict(
    {
        "action_mask": spaces.Box(0, 1, (N_ACTIONS,), np.int8),
        "observation": spaces.Box(-1.0, 1.0, (OBSERVATION_SIZE,), np.float32),
    }
)
ACTION_SPACE = spaces.Discrete(N_ACTIONS)

# poke-env's action layout: 0-5 switch to team slot, 6-9 move, 22-25 move + Tera.
SWITCH, MOVE, TERA = range(0, 6), range(6, 10), range(22, 26)


class _SpacesOnly(gym.Env):
    """Lets SB3 build a PPO model without a battle server. Never stepped."""

    observation_space = OBSERVATION_SPACE
    action_space = ACTION_SPACE

    def reset(self, *, seed=None, options=None):  # pragma: no cover - never called
        raise RuntimeError("behaviour-cloning model has no environment")

    def step(self, action):  # pragma: no cover - never called
        raise RuntimeError("behaviour-cloning model has no environment")


def new_model(seed: int = 0) -> PPO:
    """A fresh PPO model with the PPO agents' policy and spaces."""
    return PPO(MaskedActorCriticPolicy, _SpacesOnly(), seed=seed, device="cpu", n_steps=512, batch_size=256)


def _tensors(data: DecisionDataset, rows: np.ndarray | slice) -> dict[str, torch.Tensor]:
    return {
        "observation": torch.as_tensor(data.obs[rows]),
        "action_mask": torch.as_tensor(data.mask[rows]),
    }


@dataclass(frozen=True)
class BCConfig:
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    value_coef: float = 0.5
    gamma: float = 0.99
    patience: int = 4  # stop after this many epochs without validation improvement
    seed: int = 0


def train(model: PPO, train_data: DecisionDataset, val_data: DecisionDataset, config: BCConfig) -> list[dict[str, Any]]:
    """Fit ``model``'s policy to the teacher. Keeps the weights of the epoch
    with the lowest validation loss. Returns per-epoch history."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    actions = torch.as_tensor(train_data.action)
    values = torch.as_tensor(train_data.value_targets(config.gamma))
    best_loss, best_state, stale, history = math.inf, None, 0, []
    for epoch in range(1, config.epochs + 1):
        started = time.monotonic()
        policy.train()
        order = rng.permutation(len(train_data))
        total_nll = total_value = 0.0
        for start in range(0, len(order), config.batch_size):
            rows = order[start : start + config.batch_size]
            predicted_values, log_prob, _ = policy.evaluate_actions(_tensors(train_data, rows), actions[rows])
            nll = -log_prob.mean()
            value_loss = torch.nn.functional.mse_loss(predicted_values.flatten(), values[rows])
            loss = nll + config.value_coef * value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total_nll += nll.item() * len(rows)
            total_value += value_loss.item() * len(rows)
        val = evaluate(model, val_data, config.gamma)
        entry = {
            "epoch": epoch,
            "train_nll": round(total_nll / len(train_data), 4),
            "train_value_mse": round(total_value / len(train_data), 4),
            "val_nll": val["nll"],
            "val_accuracy": val["accuracy"],
            "val_value_mse": val["value_mse"],
            "seconds": round(time.monotonic() - started, 1),
        }
        history.append(entry)
        print(
            f"epoch {epoch:2d}: train nll {entry['train_nll']:.3f} | val nll {val['nll']:.3f}, "
            f"agreement {val['accuracy']:.1%}, value mse {val['value_mse']:.3f} ({entry['seconds']}s)",
            flush=True,
        )
        if val["nll"] < best_loss - 1e-4:
            best_loss, stale = val["nll"], 0
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is not None:
        policy.load_state_dict(best_state)
    policy.eval()
    return history


def _kind(action: int) -> str:
    if action in SWITCH:
        return "switch"
    if action in TERA:
        return "move+tera"
    return "move" if action in MOVE else "other"


@torch.no_grad()
def evaluate(model: PPO, data: DecisionDataset, gamma: float = 0.99) -> dict[str, Any]:
    """How closely the policy reproduces the teacher on ``data``.

    ``accuracy`` is agreement on the single most likely action (how the agent
    plays when deterministic). ``illegal_mass`` is the total probability put on
    masked actions, which must be zero. Breakdowns are by battle stage, by
    forced vs free choice, and by the kind of action the teacher took. Only
    decisions with at least two legal actions count: the rest are not choices.
    """
    policy = model.policy
    policy.eval()
    choice = data.mask.sum(axis=1) > 1
    rows = np.flatnonzero(choice)
    if not len(rows):
        return {"decisions": 0}
    probs, values = [], []
    for start in range(0, len(rows), 4096):
        batch = rows[start : start + 4096]
        obs = _tensors(data, batch)
        distribution = policy.get_distribution(obs).distribution
        probs.append(distribution.probs.numpy())
        values.append(policy.predict_values(obs).numpy().flatten())
    p = np.concatenate(probs)
    v = np.concatenate(values)
    taught = data.action[rows]
    predicted = p.argmax(axis=1)
    hit = predicted == taught
    top2 = (np.argsort(-p, axis=1)[:, :2] == taught[:, None]).any(axis=1)
    nll = -np.log(np.clip(p[np.arange(len(rows)), taught], 1e-12, None))
    illegal = float((p * (data.mask[rows] == 0)).sum(axis=1).max())

    def share(selector: np.ndarray) -> dict[str, Any]:
        n = int(selector.sum())
        return {"n": n, "accuracy": round(float(hit[selector].mean()), 4) if n else None}

    turn = data.turn[rows]
    stages = {"early (turns 1-5)": turn <= 5, "mid (6-15)": (turn > 5) & (turn <= 15), "late (16+)": turn > 15}
    kinds = np.array([_kind(int(a)) for a in taught])
    predicted_kinds = np.array([_kind(int(a)) for a in predicted])
    confusion: dict[str, dict[str, int]] = {}
    for t, q in zip(kinds, predicted_kinds):
        confusion.setdefault(t, {}).setdefault(q, 0)
        confusion[t][q] += 1
    targets = data.value_targets(gamma)[rows]
    outcome_known = data.outcome[rows] != 0
    win_pred = (v > 0) == (data.outcome[rows] > 0)
    return {
        "decisions": int(len(rows)),
        "battles": int(len(set(data.battle[rows].tolist()))),
        "accuracy": round(float(hit.mean()), 4),
        "top2_accuracy": round(float(top2.mean()), 4),
        "nll": round(float(nll.mean()), 4),
        "illegal_mass": illegal,
        "by_stage": {k: share(s) for k, s in stages.items()},
        "forced": share(data.forced[rows]),
        "free": share(~data.forced[rows]),
        "by_teacher_action": {k: share(kinds == k) for k in sorted(set(kinds))},
        "action_kind_confusion": confusion,
        "value_mse": round(float(((v - targets) ** 2).mean()), 4),
        "value_sign_matches_outcome": round(float(win_pred[outcome_known].mean()), 4) if outcome_known.any() else None,
    }
