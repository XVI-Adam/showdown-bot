"""Test whether knowing the opponent helps predict their choices.

Usage:
    uv run python scripts/analyze_opponent_adaptation.py              # recorded human battles
    uv run python scripts/analyze_opponent_adaptation.py --source data/some-bot-records/ --by kind

Compares a population model of opponent choices (attack / switch / support,
given their active Pokémon's HP) with a per-opponent model, predicting each
battle from all the others. Opponents are numbered in the output, not named.
See src/psbot/training/opponent_adaptation.py for the method and its limits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from psbot.dataset.human import HUMAN_DIR
from psbot.dataset.schema import load_dataset
from psbot.training.opponent_adaptation import evaluate, extract_choices, learning_curve


def main(args: argparse.Namespace) -> int:
    path = Path(args.source)
    records = [record for _, record in load_dataset(path)] if path.exists() else []
    choices = extract_choices(records, identity=args.by)
    if not choices:
        print("no opponent choices recorded yet")
        return 0
    result = evaluate(choices, prior_strength=args.prior_strength)
    loss, acc = result["logloss"], result["accuracy"]
    low, high = result["personal_gain_ci95"]
    print(f"{result['choices']} free choices by {result['opponents']} opponent(s) in {result['battles']} battle(s)\n")
    print("                 log-loss   accuracy")
    for name in ("uniform", "population", "personal"):
        print(f"  {name:12s}   {loss[name]:.3f}      {acc[name]:.1%}")
    print(f"\npersonal model's gain over population: {result['personal_gain_nats']:+.3f} nats per choice "
          f"(95% interval {low:+.3f} to {high:+.3f})")  # fmt: skip
    verdict = (
        "knowing the opponent helps" if low > 0
        else "knowing the opponent makes predictions worse" if high < 0
        else "no detectable difference at this sample size"
    )
    print(f"verdict: {verdict}\n")
    for i, (who, row) in enumerate(result["by_opponent"].items(), 1):
        label = who if args.by == "kind" else f"opponent {i}"
        shares = ", ".join(f"{k} {v:.0%}" for k, v in row["class_shares"].items())
        print(f"  {label}: {row['battles']} battle(s), {row['choices']} choices ({shares}); "
              f"log-loss population {row['population_logloss']:.3f} vs personal {row['personal_logloss']:.3f}")  # fmt: skip
    curve = learning_curve(choices)
    print("\nlog-loss on later battles after seeing an opponent's first k (k=0: population only):")
    print("  " + ", ".join(f"k={k}: {v:.3f}" for k, v in curve.items() if v is not None))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=str(HUMAN_DIR), help="Phase 4 records: a .jsonl file or directory")
    parser.add_argument("--by", choices=("username", "kind"), default="username", help="what counts as one opponent")
    parser.add_argument("--prior-strength", type=float, default=20.0)
    sys.exit(main(parser.parse_args()))
