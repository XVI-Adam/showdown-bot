"""Aggregate per-battle metrics into rates with 95% intervals.

Proportions use the Wilson interval; means use a normal interval
(1.96 standard errors); differences between two agents' win rates use
Newcombe's method (hybrid Wilson), suited to independent samples.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from psbot.evaluation.metrics import STAGES
from psbot.runner import wilson_interval

Z = 1.96


def rate(successes: int, n: int) -> dict[str, Any]:
    low, high = wilson_interval(successes, n)
    return {"value": successes / n if n else None, "k": successes, "n": n, "ci": [low, high]}


def mean(values: list[float]) -> dict[str, Any]:
    n = len(values)
    if not n:
        return {"value": None, "n": 0, "ci": [None, None]}
    mu = sum(values) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    half = Z * sd / math.sqrt(n)
    return {"value": mu, "n": n, "ci": [mu - half, mu + half]}


def difference(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Newcombe interval for p_a - p_b from two ``rate`` results."""
    pa, pb = a["value"], b["value"]
    if pa is None or pb is None:
        return {"value": None, "ci": [None, None]}
    la, ua = a["ci"]
    lb, ub = b["ci"]
    d = pa - pb
    low = d - math.sqrt((pa - la) ** 2 + (ub - pb) ** 2)
    high = d + math.sqrt((ua - pa) ** 2 + (pb - lb) ** 2)
    return {"value": d, "ci": [low, high]}


def summarise(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    n = len(rows)
    won = [r for r in rows if r["won"] is True]
    endgames = [r for r in rows if r["endgame"] is not None]
    setup = [r for r in rows if r["faced_setup_sweeper"]]
    first_use = [r for r in rows if r["kos_suffered_first_use"] > 0]

    def wins(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return rate(sum(r["won"] is True for r in subset), len(subset))

    def total(key: str) -> int:
        return sum(r[key] for r in rows)

    return {
        "battles": n,
        "win_rate": rate(len(won), n),
        "remaining": mean([r["remaining"] for r in rows]),
        "remaining_when_won": mean([r["remaining"] for r in won]),
        "turns": mean([r["turns"] for r in rows]),
        "stages": {
            name: {
                "net_kos": mean([r["stages"][name]["dealt"] - r["stages"][name]["suffered"] for r in rows]),
                "dealt": mean([r["stages"][name]["dealt"] for r in rows]),
                "suffered": mean([r["stages"][name]["suffered"] for r in rows]),
            }
            for name, _, _ in STAGES
        },
        "setup": {
            "faced": rate(len(setup), n),
            "win_rate_when_faced": wins(setup),
            "win_rate_otherwise": wins([r for r in rows if not r["faced_setup_sweeper"]]),
            "kos_per_battle_faced": mean([r["kos_by_setup_sweepers"] for r in setup]),
        },
        "swept": rate(sum(r["swept"] for r in rows), n),
        "unrevealed": {
            "kos_to_first_use_moves": rate(total("kos_suffered_first_use"), total("kos_suffered_by_attack")),
            "win_rate_with_such_ko": wins(first_use),
            "win_rate_without": wins([r for r in rows if r["kos_suffered_first_use"] == 0]),
        },
        "endgame": {
            "reached": rate(len(endgames), n),
            "win_rate": wins(endgames),
            **{f"from_{state}": wins([r for r in endgames if r["endgame"] == state]) for state in ("ahead", "level", "behind")},
        },
        "led_by_2_and_lost": rate(sum(r["max_lead"] >= 2 and r["won"] is False for r in rows),
                                  sum(r["max_lead"] >= 2 for r in rows)),  # fmt: skip
        "missed_kos": rate(total("sure_ko_missed"), total("sure_ko_chances")),
        "immune_moves": rate(total("immune"), total("our_moves")),
        "missed_or_failed_moves": rate(total("missed") + total("failed"), total("our_moves")),
        "switches": {
            "per_battle": mean([r["voluntary_switches"] for r in rows]),
            "fainted_before_acting": rate(total("switch_fainted_before_acting"), total("voluntary_switches")),
            "lost_half_hp_on_entry": rate(total("switch_big_hit"), total("voluntary_switches")),
        },
    }
