"""Decision datasets for supervised learning, and how they are split.

One row per decision our agent made:

``obs``      the 112-feature observation (``psbot.rl.observation``), exactly
             what the PPO agents see
``mask``     the 26 legal actions at that moment (poke-env's action space)
``action``   the action taken, as an index into the same space
``outcome``  +1 the battle was won, -1 lost, 0 tie or unknown
``to_end``   decisions left in the battle after this one
``turn``, ``forced``, ``battle`` (index into ``battle_ids``)

What these rows are, and are not:

* The label is what *our agent* chose, so cloning it learns that agent's
  policy. It is not "the right move": the teacher loses about one battle in
  five even against bots.
* A human opponent's actions are not in here. From our side of a battle we see
  neither their observation nor their legal actions, so their choices cannot
  be imitation targets. They feed ``opponent_adaptation`` instead.
* The outcome is attached to every decision of a battle. It is a sparse,
  noisy signal about any one decision, kept for warm-starting a value
  function, not for judging moves.

Splits are always by battle: every decision of a battle lands in the same
split, since consecutive decisions of one battle are nearly identical and
would otherwise leak the answer from training into evaluation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv

from psbot.config import PROJECT_ROOT
from psbot.rl.observation import OBSERVATION_SIZE, embed_battle

logger = logging.getLogger(__name__)

SYNTHETIC_DIR: Path = PROJECT_ROOT / "data" / "synthetic"
N_ACTIONS = 26
_ARRAYS = ("obs", "mask", "action", "outcome", "to_end", "turn", "forced", "battle")


@dataclass
class DecisionDataset:
    obs: np.ndarray
    mask: np.ndarray
    action: np.ndarray
    outcome: np.ndarray
    to_end: np.ndarray
    turn: np.ndarray
    forced: np.ndarray
    battle: np.ndarray
    battle_ids: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.action.shape[0])

    @property
    def n_battles(self) -> int:
        return len(set(self.battle.tolist()))

    def subset(self, rows: np.ndarray) -> "DecisionDataset":
        return DecisionDataset(
            *(getattr(self, name)[rows] for name in _ARRAYS), self.battle_ids, dict(self.meta)
        )

    def battles_used(self) -> set[str]:
        return {self.battle_ids[i] for i in set(self.battle.tolist())}

    def value_targets(self, gamma: float = 0.99) -> np.ndarray:
        """Discounted outcome: what PPO's critic would learn to predict under
        a win/loss-only reward with discount ``gamma``."""
        return (self.outcome * gamma ** self.to_end.astype(np.float32)).astype(np.float32)

    # --------------------------------------------------------------- disk

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{name: getattr(self, name) for name in _ARRAYS})
        path.with_suffix(".json").write_text(
            json.dumps({**self.meta, "battle_ids": self.battle_ids}, indent=1)
        )

    @classmethod
    def load(cls, path: Path) -> "DecisionDataset":
        arrays = np.load(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        battle_ids = meta.pop("battle_ids")
        return cls(*(arrays[name] for name in _ARRAYS), battle_ids, meta)

    @classmethod
    def concatenate(cls, parts: Iterable["DecisionDataset"]) -> "DecisionDataset":
        parts = list(parts)
        ids: list[str] = []
        offsets = []
        for part in parts:
            offsets.append(len(ids))
            ids.extend(part.battle_ids)
        arrays = {
            name: np.concatenate([getattr(p, name) for p in parts]) for name in _ARRAYS if name != "battle"
        }
        arrays["battle"] = np.concatenate([p.battle + off for p, off in zip(parts, offsets)])
        meta = {"parts": [p.meta for p in parts]}
        return cls(**arrays, battle_ids=ids, meta=meta)


class _Builder:
    """Accumulates rows battle by battle."""

    def __init__(self) -> None:
        self.rows: dict[str, list[Any]] = {name: [] for name in _ARRAYS}
        self.battle_ids: list[str] = []
        self.skipped: dict[str, int] = {}

    def skip(self, why: str) -> None:
        self.skipped[why] = self.skipped.get(why, 0) + 1

    def add_battle(self, battle_id: str, decisions: list[tuple], won: bool | None) -> None:
        """``decisions``: (obs, mask, action, turn, forced) in order."""
        if not decisions:
            return
        index = len(self.battle_ids)
        self.battle_ids.append(battle_id)
        outcome = 1.0 if won is True else -1.0 if won is False else 0.0
        for k, (obs, mask, action, turn, forced) in enumerate(decisions):
            for name, value in (
                ("obs", obs), ("mask", mask), ("action", action), ("outcome", outcome),
                ("to_end", len(decisions) - 1 - k), ("turn", turn), ("forced", forced), ("battle", index),
            ):  # fmt: skip
                self.rows[name].append(value)

    def build(self, meta: dict[str, Any]) -> DecisionDataset:
        n = len(self.rows["action"])
        dtypes = {
            "obs": np.float32, "mask": np.int8, "action": np.int64, "outcome": np.float32,
            "to_end": np.int16, "turn": np.int16, "forced": np.bool_, "battle": np.int32,
        }  # fmt: skip
        arrays = {}
        for name in _ARRAYS:
            if n:
                arrays[name] = np.asarray(self.rows[name], dtype=dtypes[name])
            else:
                width = {"obs": OBSERVATION_SIZE, "mask": N_ACTIONS}.get(name)
                arrays[name] = np.zeros((0, width) if width else (0,), dtype=dtypes[name])
        return DecisionDataset(**arrays, battle_ids=self.battle_ids, meta={**meta, "skipped": self.skipped})


def _check_row(builder: _Builder, obs: Any, mask: Any, action: Any) -> bool:
    """A row is usable only with a full observation, a mask, and a legal action."""
    if obs is None or len(obs) != OBSERVATION_SIZE:
        builder.skip("no 112-feature observation")
        return False
    if mask is None or len(mask) != N_ACTIONS:
        builder.skip("no action mask")
        return False
    if action is None or not 0 <= action < N_ACTIONS:
        builder.skip("no action index")
        return False
    if not mask[action]:
        builder.skip("selected action not in the mask")
        return False
    return True


def from_records(records: Iterable[dict[str, Any]], source: str) -> DecisionDataset:
    """Our agent's decisions from human-battle records (``psbot.dataset``).

    Decisions without a 112-feature observation (e.g. a hybrid's 165) or
    without a legal action index are skipped and counted, never guessed.
    """
    builder = _Builder()
    agents: dict[str, int] = {}
    for record in records:
        agents[record["agent"]["name"]] = agents.get(record["agent"]["name"], 0) + 1
        rows = []
        for decision in record["decisions"]:
            heuristic = decision.get("heuristic")
            if heuristic is not None and heuristic.get("fallback"):
                builder.skip("engine fallback (a random action)")
                continue
            observation = (decision.get("observation") or {}).get("values")
            mask, action = decision.get("action_mask"), decision["selected"].get("index")
            if not _check_row(builder, observation, mask, action):
                continue
            rows.append((observation, mask, action, decision["turn"], decision["forced_switch"]))
        builder.add_battle(record["battle_id"], rows, record["outcome"]["won"])
    return builder.build({"source": source, "agents": agents})


class TeacherRecorder:
    """Collects decisions straight into arrays while an agent plays.

    Has the recorder interface the agents call (``record_decision``,
    ``record_action``, ``finish_battle``) but keeps only what training needs,
    so tens of thousands of battles fit in memory and on disk.
    """

    def __init__(self) -> None:
        self._builder = _Builder()
        self._pending: dict[str, list[tuple]] = {}
        self.n_written = 0

    def _add(self, battle: AbstractBattle, order: Any) -> None:
        if order is None:
            self._builder.skip("no order")
            return
        try:
            mask = [int(x) for x in SinglesEnv.get_action_mask(battle)]
            action = int(SinglesEnv.order_to_action(order, battle, fake=True, strict=False))
            obs = embed_battle(battle)
        except Exception:
            logger.debug("could not encode a decision", exc_info=True)
            self._builder.skip("could not encode")
            return
        if _check_row(self._builder, obs, mask, action):
            self._pending.setdefault(battle.battle_tag, []).append(
                (obs, mask, action, battle.turn, bool(battle.force_switch))
            )

    def record_decision(self, battle, evaluations, fallback=None, extra=None, order=None) -> None:
        if not evaluations:
            self._builder.skip("engine fallback (a random action)")
            return
        self._add(battle, order)

    def record_action(self, battle, action, extra=None, order=None) -> None:
        self._add(battle, order)

    def finish_battle(self, battle: AbstractBattle, opponent_kind: str | None = None) -> None:
        self._builder.add_battle(battle.battle_tag, self._pending.pop(battle.battle_tag, []), battle.won)
        self.n_written += 1

    def dataset(self, meta: dict[str, Any]) -> DecisionDataset:
        return self._builder.build(meta)


# ------------------------------------------------------------------ splits


def _bucket(battle_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{battle_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_by_battle(
    data: DecisionDataset, validation: float = 0.1, salt: str = "psbot"
) -> tuple[DecisionDataset, DecisionDataset]:
    """Deterministic train/validation split: a battle's place depends only on
    its id, so re-running never moves a battle between splits."""
    in_validation = np.array([_bucket(b, salt) < validation for b in data.battle_ids])
    rows = in_validation[data.battle]
    return data.subset(np.flatnonzero(~rows)), data.subset(np.flatnonzero(rows))


def leakage_report(train: DecisionDataset, other: DecisionDataset) -> dict[str, Any]:
    """Evidence that ``other`` shares nothing with ``train``.

    Shared battle ids would be outright leakage. Identical observations can
    also occur legitimately (the same lead matchup in two battles), so their
    share is reported for judgement rather than treated as an error.
    """
    shared = sorted(train.battles_used() & other.battles_used())
    seen = {row.tobytes() for row in train.obs}
    duplicates = sum(row.tobytes() in seen for row in other.obs)
    return {
        "shared_battles": len(shared),
        "examples": shared[:5],
        "identical_observations": duplicates,
        "identical_share": round(duplicates / len(other), 4) if len(other) else 0.0,
    }
