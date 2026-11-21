"""Does knowing who we are playing help predict what they will do?

This is the offline test for opponent-specific adaptation, run before any
agent is allowed to act on it. Each time an opponent chose freely (not a
replacement after a faint), the log shows what kind of action they took:

``attack``   a damaging move (pivots such as U-turn included)
``switch``   a voluntary switch
``support``  anything else: setup, status, hazards, recovery

Three predictors of that choice, given how healthy their active Pokémon is:

``uniform``       one third each
``population``    frequencies over every other battle, of every opponent
``personal``      this opponent's own frequencies, shrunk toward the
                  population by ``prior_strength`` pseudo-observations, so a
                  person seen once is predicted almost like everyone else

Each battle is predicted from all *other* battles (leave one battle out), so
no battle helps predict itself. Scores are log-loss (lower is better) and
accuracy, and the personal-minus-population difference gets a 95% interval
by resampling battles.

What this does not do: it does not show that acting on a prediction wins
more. A tendency can be real and still not be exploitable.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from poke_env.data import to_id_str

from psbot.analysis import move_kind
from psbot.analytics.opponents import PIVOT_MOVES

CLASSES = ("attack", "switch", "support")
CONTEXTS = ("hp<1/3", "hp<2/3", "hp>=2/3")


@dataclass(frozen=True)
class Choice:
    opponent: str
    battle: str
    context: int
    label: int


def _context(hp: float) -> int:
    return 0 if hp < 1 / 3 else 1 if hp < 2 / 3 else 2


def _label(action: dict[str, Any]) -> int | None:
    if action["type"] == "switch":
        return CLASSES.index("switch")
    if action["type"] != "move" or action.get("called"):
        return None
    if action["id"] in PIVOT_MOVES or move_kind(action["id"]) == "attack":
        return CLASSES.index("attack")
    return CLASSES.index("support")


def extract_choices(records: Iterable[dict[str, Any]], identity: str = "username") -> list[Choice]:
    """Free choices of the opponent, from Phase 4 records.

    ``identity`` is ``"username"`` for people, or ``"kind"`` to treat each bot
    type as one opponent (used to check the method on synthetic battles).
    """
    choices = []
    for record in records:
        who = record["opponent"][identity] or "?"
        for decision in record["decisions"]:
            active = decision.get("opponent_active")
            if not active or active["fainted"] or active["hp"] <= 0:
                continue  # their next switch is a forced replacement
            for action in decision.get("opponent_actions", []):
                if action["type"] in ("drag", "cant"):
                    break
                label = _label(action)
                if label is not None:
                    choices.append(Choice(who, record["battle_id"], _context(active["hp"]), label))
                    break  # one choice per turn: the first thing they did
    return choices


def _counts(choices: Iterable[Choice]) -> np.ndarray:
    table = np.zeros((len(CONTEXTS), len(CLASSES)))
    for c in choices:
        table[c.context, c.label] += 1
    return table


def _normalise(table: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    table = table + smoothing
    return table / table.sum(axis=1, keepdims=True)


def evaluate(choices: list[Choice], prior_strength: float = 20.0, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Leave-one-battle-out comparison of the three predictors."""
    by_battle: dict[str, list[Choice]] = defaultdict(list)
    for c in choices:
        by_battle[c.battle].append(c)
    total = _counts(choices)
    by_opponent = defaultdict(lambda: np.zeros_like(total))
    battles_of = defaultdict(set)
    for c in choices:
        by_opponent[c.opponent][c.context, c.label] += 1
        battles_of[c.opponent].add(c.battle)

    per_battle = []  # (opponent, n, loss uniform, loss population, loss personal, hits...)
    for battle, items in by_battle.items():
        own = _counts(items)
        who = items[0].opponent
        population = _normalise(total - own)
        mine = by_opponent[who] - own
        personal = (mine + prior_strength * population) / (mine.sum(axis=1, keepdims=True) + prior_strength)
        losses = np.zeros(3)
        hits = np.zeros(3)
        for c in items:
            for k, p in enumerate((np.full(len(CLASSES), 1 / len(CLASSES)), population[c.context], personal[c.context])):
                losses[k] -= math.log(p[c.label])
                hits[k] += float(np.argmax(p) == c.label)
        per_battle.append((who, len(items), losses, hits))

    n = sum(item[1] for item in per_battle)
    loss = sum(item[2] for item in per_battle) / n
    acc = sum(item[3] for item in per_battle) / n
    gain_low, gain_high = _bootstrap_gain(per_battle, n_boot, seed)
    opponents = {}
    for who in sorted(battles_of):
        rows = [item for item in per_battle if item[0] == who]
        m = sum(r[1] for r in rows)
        opponents[who] = {
            "battles": len(battles_of[who]),
            "choices": m,
            "population_logloss": round(float(sum(r[2][1] for r in rows) / m), 4),
            "personal_logloss": round(float(sum(r[2][2] for r in rows) / m), 4),
            "class_shares": {cls: round(float(by_opponent[who][:, i].sum() / by_opponent[who].sum()), 3) for i, cls in enumerate(CLASSES)},
        }
    return {
        "choices": n,
        "battles": len(by_battle),
        "opponents": len(battles_of),
        "prior_strength": prior_strength,
        "logloss": dict(zip(("uniform", "population", "personal"), np.round(loss, 4).tolist())),
        "accuracy": dict(zip(("uniform", "population", "personal"), np.round(acc, 4).tolist())),
        # positive = personal model predicts better than the population model
        "personal_gain_nats": round(float(loss[1] - loss[2]), 4),
        "personal_gain_ci95": [round(gain_low, 4), round(gain_high, 4)],
        "by_opponent": opponents,
    }


def _bootstrap_gain(per_battle: list, n_boot: int, seed: int) -> tuple[float, float]:
    """95% interval of (population - personal) log-loss per choice, resampling battles."""
    rng = np.random.default_rng(seed)
    sizes = np.array([b[1] for b in per_battle], dtype=float)
    diffs = np.array([b[2][1] - b[2][2] for b in per_battle])
    if len(per_battle) < 2:
        return float("nan"), float("nan")
    gains = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(per_battle), len(per_battle))
        gains.append(diffs[pick].sum() / sizes[pick].sum())
    low, high = np.percentile(gains, [2.5, 97.5])
    return float(low), float(high)


def learning_curve(choices: list[Choice], ks: Iterable[int] = (0, 1, 2, 5, 10, 20), prior_strength: float = 20.0) -> dict[int, float | None]:
    """Log-loss on an opponent's later battles when only their first ``k``
    battles are known. ``k = 0`` is the population model. Shows how many
    battles adaptation needs before it starts to pay."""
    order: dict[str, list[str]] = defaultdict(list)
    for c in choices:
        if c.battle not in order[c.opponent]:
            order[c.opponent].append(c.battle)
    out: dict[int, float | None] = {}
    for k in ks:
        loss = count = 0.0
        for who, battles in order.items():
            if len(battles) <= k:
                continue
            known, later = set(battles[:k]), set(battles[k:])
            others = [c for c in choices if c.opponent != who]
            population = _normalise(_counts(others))
            mine = _counts(c for c in choices if c.opponent == who and c.battle in known)
            personal = (mine + prior_strength * population) / (mine.sum(axis=1, keepdims=True) + prior_strength)
            for c in choices:
                if c.opponent == who and c.battle in later:
                    loss -= math.log(personal[c.context, c.label])
                    count += 1
        out[k] = round(loss / count, 4) if count else None
    return out
