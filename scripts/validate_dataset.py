"""Validate and summarise the human-battle dataset.

Usage:
    uv run python scripts/validate_dataset.py                 # everything in data/human/
    uv run python scripts/validate_dataset.py data/human/official-20261117-120000.jsonl
    uv run python scripts/validate_dataset.py -v              # also list warnings

Checks every record against psbot/dataset/schema.py: required fields, legal
and selected actions, score arithmetic, policy probabilities, and that every
fact marked "revealed" about the opponent is in the battle log at that point.
Exits with status 1 if any record has an error.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from psbot.dataset.human import HUMAN_DIR
from psbot.dataset.schema import load_dataset, validate_record


def main(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"nothing to validate: {path} does not exist")
        return 0
    seen: Counter[str] = Counter()
    by_agent: Counter[str] = Counter()
    by_opponent: Counter[str] = Counter()
    results: Counter[str] = Counter()
    knowledge: Counter[str] = Counter()
    n_decisions = n_errors = n_warnings = 0

    for location, record in load_dataset(path):
        issues = validate_record(record)
        battle_id = str(record.get("battle_id", location))
        seen[battle_id] += 1
        if seen[battle_id] == 2:
            print(f"error: {location}: duplicate battle_id {battle_id}")
            n_errors += 1
        for issue in issues:
            if issue.level == "error":
                n_errors += 1
                print(f"{location}: {issue}")
            else:
                n_warnings += 1
                if args.verbose:
                    print(f"{location}: {issue}")
        if any(issue.level == "error" for issue in issues):
            continue
        n_decisions += len(record["decisions"])
        by_agent[record["agent"]["name"]] += 1
        by_opponent[record["opponent"]["username"]] += 1
        won = record["outcome"]["won"]
        results["won" if won else "lost" if won is False else "no result"] += 1
        for mon in record["opponent_team"]:
            for field in ("ability", "item"):
                knowledge[f"{field} {mon[field]['source']}"] += 1

    n_battles = sum(seen.values())
    print(f"\n{path}: {n_battles} battle(s), {n_decisions} decision(s) in valid records")
    if n_battles:
        print("  results:   " + ", ".join(f"{k} {v}" for k, v in sorted(results.items())))
        print("  agents:    " + ", ".join(f"{k} x{v}" for k, v in by_agent.most_common()))
        print(f"  opponents: {len(by_opponent)} distinct")
        print("  opposing Pokémon knowledge at battle end: "
              + ", ".join(f"{k} {v}" for k, v in sorted(knowledge.items())))  # fmt: skip
    print(f"  {n_errors} error(s), {n_warnings} warning(s)" + ("" if args.verbose or not n_warnings else " (-v to list)"))
    return 1 if n_errors else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", default=str(HUMAN_DIR), help="a .jsonl file or a directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="list warnings too")
    sys.exit(main(parser.parse_args()))
