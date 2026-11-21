"""Training data, behaviour cloning and opponent adaptation, without a server."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO

from psbot.training.bc import BCConfig, evaluate, new_model, train
from psbot.training.dataset import DecisionDataset, from_records, leakage_report, split_by_battle
from psbot.training.opponent_adaptation import CLASSES, Choice, evaluate as adapt, extract_choices, learning_curve


def toy(n_battles: int = 60, per_battle: int = 20, seed: int = 0) -> DecisionDataset:
    """A teacher that picks the legal action with the largest feature value."""
    rng = np.random.default_rng(seed)
    n = n_battles * per_battle
    obs = rng.uniform(-1, 1, (n, 112)).astype(np.float32)
    mask = (rng.random((n, 26)) < 0.25).astype(np.int8)
    mask[:, 6] = 1
    action = np.array([max(np.flatnonzero(m), key=lambda a: o[a]) for o, m in zip(obs, mask)])
    battle = np.repeat(np.arange(n_battles, dtype=np.int32), per_battle)
    to_end = np.tile(np.arange(per_battle - 1, -1, -1, dtype=np.int16), n_battles)
    outcome = np.where(battle % 2 == 0, 1.0, -1.0).astype(np.float32)
    turn = np.tile(np.arange(1, per_battle + 1, dtype=np.int16), n_battles)
    return DecisionDataset(obs, mask, action, outcome, to_end, turn, np.zeros(n, bool), battle,
                           [f"battle-{i}" for i in range(n_battles)], {"source": "toy"})  # fmt: skip


# ------------------------------------------------------------------ dataset


def test_split_keeps_every_battle_whole_and_is_stable() -> None:
    data = toy()
    train_a, val_a = split_by_battle(data, 0.2)
    train_b, val_b = split_by_battle(copy.deepcopy(data), 0.2)
    assert train_a.battles_used().isdisjoint(val_a.battles_used())
    assert train_a.battles_used() | val_a.battles_used() == set(data.battle_ids)
    assert val_a.battles_used() == val_b.battles_used()  # depends only on battle ids
    report = leakage_report(train_a, val_a)
    assert report["shared_battles"] == 0 and report["identical_observations"] == 0
    assert leakage_report(train_a, train_a)["shared_battles"] == train_a.n_battles


def test_value_targets_are_discounted_outcomes() -> None:
    data = toy(n_battles=2, per_battle=3)
    assert data.value_targets(0.5).tolist() == [0.25, 0.5, 1.0, -0.25, -0.5, -1.0]


def test_save_load_and_concatenate(tmp_path: Path) -> None:
    a, b = toy(4, 5, seed=1), toy(3, 5, seed=2)
    a.save(tmp_path / "a.npz")
    loaded = DecisionDataset.load(tmp_path / "a.npz")
    assert np.array_equal(loaded.obs, a.obs) and loaded.battle_ids == a.battle_ids
    both = DecisionDataset.concatenate([a, b])
    assert len(both) == 35 and both.n_battles == 7 and both.battle.max() == 6


def record(decisions: list[dict], won: bool = True) -> dict:
    return {"battle_id": "battle-x", "agent": {"name": "heuristic"}, "outcome": {"won": won}, "decisions": decisions}


def decision(index: int | None, mask: list[int] | None, size: int = 112, fallback: str | None = None) -> dict:
    return {
        "turn": 1, "forced_switch": False, "action_mask": mask,
        "selected": {"action": "move x", "index": index},
        "observation": {"size": size, "values": [0.0] * size},
        "heuristic": {"fallback": fallback, "actions": []}, "policy": None,
    }  # fmt: skip


def test_records_become_rows_and_bad_rows_are_counted_not_guessed() -> None:
    legal = [0] * 26
    legal[6] = legal[7] = 1
    data = from_records(
        [record([
            decision(6, legal),
            decision(8, legal),  # chose something illegal
            decision(None, legal),  # no index
            decision(6, None),  # no mask
            decision(6, legal, size=165),  # a hybrid's observation
            decision(7, legal, fallback="engine error"),  # random action
        ])],
        "human",
    )  # fmt: skip
    assert len(data) == 1 and data.action.tolist() == [6] and data.outcome.tolist() == [1.0]
    assert sum(data.meta["skipped"].values()) == 5
    assert data.meta["skipped"]["selected action not in the mask"] == 1


# ------------------------------------------------------- behaviour cloning


def test_cloning_learns_the_teacher_and_never_chooses_illegal_actions(tmp_path: Path) -> None:
    data = toy(n_battles=150)
    train_data, val_data = split_by_battle(data, 0.2)
    model = new_model()
    before = evaluate(model, val_data)
    train(model, train_data, val_data, BCConfig(epochs=25, patience=25))
    after = evaluate(model, val_data)
    assert after["accuracy"] > before["accuracy"] + 0.15
    assert after["illegal_mass"] == 0.0 and before["illegal_mass"] == 0.0
    assert after["value_mse"] < before["value_mse"]

    model.save(tmp_path / "final")
    reloaded = PPO.load(tmp_path / "final.zip", device="cpu")
    assert evaluate(reloaded, val_data)["accuracy"] == after["accuracy"]
    assert reloaded.observation_space["observation"].shape == (112,)
    assert type(reloaded.policy).__name__ == "MaskedActorCriticPolicy"


def test_evaluation_ignores_decisions_without_a_choice() -> None:
    data = toy(n_battles=2)
    data.mask[:] = 0
    data.mask[:, 6] = 1
    data.action[:] = 6
    assert evaluate(new_model(), data) == {"decisions": 0}


# ------------------------------------------------------ opponent adaptation


def test_choices_skip_forced_replacements_and_called_moves() -> None:
    rec = {
        "battle_id": "b1", "opponent": {"username": "Alice", "kind": "human"},
        "decisions": [
            {"opponent_active": {"hp": 0.9, "fainted": False}, "opponent_actions": [{"type": "move", "id": "earthquake"}]},
            {"opponent_active": {"hp": 0.2, "fainted": False}, "opponent_actions": [{"type": "switch", "id": "skarmory"}]},
            {"opponent_active": {"hp": 0.5, "fainted": False}, "opponent_actions": [{"type": "move", "id": "swordsdance"}]},
            {"opponent_active": {"hp": 0.0, "fainted": True}, "opponent_actions": [{"type": "switch", "id": "garchomp"}]},
            {"opponent_active": {"hp": 0.8, "fainted": False}, "opponent_actions": [{"type": "move", "id": "tackle", "called": True}]},
            {"opponent_active": {"hp": 0.8, "fainted": False}, "opponent_actions": [{"type": "move", "id": "uturn"}]},
        ],
    }  # fmt: skip
    labels = [(c.context, CLASSES[c.label]) for c in extract_choices([rec])]
    assert labels == [(2, "attack"), (0, "switch"), (1, "support"), (2, "attack")]


def synthetic_choices(rng, styles: dict[str, list[float]], battles: int, per_battle: int = 20) -> list[Choice]:
    return [
        Choice(who, f"{who}-{b}", int(rng.integers(3)), int(rng.choice(3, p=p)))
        for who, p in styles.items()
        for b in range(battles)
        for _ in range(per_battle)
    ]


def test_personal_model_helps_only_when_opponents_differ() -> None:
    rng = np.random.default_rng(0)
    different = synthetic_choices(rng, {"a": [0.8, 0.1, 0.1], "b": [0.2, 0.7, 0.1]}, battles=15)
    result = adapt(different)
    assert result["personal_gain_ci95"][0] > 0
    alike = synthetic_choices(rng, {"a": [0.6, 0.3, 0.1], "b": [0.6, 0.3, 0.1]}, battles=15)
    low, high = adapt(alike)["personal_gain_ci95"]
    assert low < 0.01 and high < 0.05  # no real gain to find


def test_learning_curve_improves_with_battles_seen() -> None:
    rng = np.random.default_rng(1)
    choices = synthetic_choices(rng, {"a": [0.9, 0.05, 0.05], "b": [0.1, 0.8, 0.1], "c": [0.4, 0.3, 0.3]}, battles=25)
    curve = learning_curve(choices, ks=(0, 1, 10))
    assert curve[10] < curve[1] < curve[0]
