"""Phase 7: round-robin of every agent, with the extended metrics.

Usage:
    uv run python scripts/evaluate_all.py --run-dir logs/eval/phase7-x -n 1000 --shard 0/4   # one of four
    uv run python scripts/evaluate_all.py --run-dir logs/eval/phase7-x --report               # after all shards

Eight participants play every other participant ``n`` times. Each battle is
measured from both sides by psbot.evaluation.probe, so every metric exists for
every agent, scripted bots included. Shards split the pairings between
processes; each appends to its own files in the run directory, and --report
combines whatever is there into reports/phase7-evaluation.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

from poke_env import AccountConfiguration

from psbot.config import BATTLE_FORMAT, PROJECT_ROOT, local_server_configuration
from psbot.evaluation.probe import MetricsSink, attach
from psbot.logging_setup import setup_logging
from psbot.opponents import make_opponent
from psbot.runner import BattleRunError, run_battles
from psbot.server import ServerUnavailableError, require_server

PARTICIPANTS = {
    "random": "random",
    "max_base_power": "max_base_power",
    "simple_heuristics": "simple_heuristics",
    "heuristic": "heuristic",
    "heuristic_v2": "heuristic_v2",
    "ppo": "checkpoints/ppo-v1/final.zip",
    "hybrid": "checkpoints/hybrid-v2-sparse/final.zip",
    "bc": "checkpoints/bc-v1/final.zip",
}
REPORT = PROJECT_ROOT / "reports" / "phase7-evaluation.md"


def simulator_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT / "vendor" / "pokemon-showdown"), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return "unknown"


async def play(a: str, b: str, n: int, sink: MetricsSink, concurrency: int) -> dict:
    common = dict(
        battle_format=BATTLE_FORMAT,
        server_configuration=local_server_configuration(),
        max_concurrent_battles=concurrency,
    )
    players = {}
    for name, other in ((a, b), (b, a)):
        player = make_opponent(PARTICIPANTS[name], account_configuration=AccountConfiguration.generate(name[:8], rand=True), **common)
        players[name] = attach(player, sink, name, other)
    started = time.monotonic()
    summary = await run_battles(players[a], players[b], n_battles=n, timeout_per_battle=300)
    counters = {
        name: {
            "decisions": getattr(p, "n_decisions", None),
            "illegal_actions": getattr(p, "n_illegal_actions", None),
            "fallback_decisions": getattr(p, "n_fallback_decisions", None),
        }
        for name, p in players.items()
    }
    return {"a": a, "b": b, "n": summary.n_battles, "wins_a": summary.wins, "ties": summary.ties,
            "seconds": round(time.monotonic() - started, 1), "counters": counters}  # fmt: skip


async def run_shard(args: argparse.Namespace) -> int:
    index, count = (int(x) for x in args.shard.split("/"))
    pairings = list(combinations(PARTICIPANTS, 2))[index::count]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sink = MetricsSink(run_dir / f"battles-{index}.jsonl")
    done_path = run_dir / f"pairings-{index}.jsonl"
    done = set()
    if done_path.exists():  # resume an interrupted shard
        done = {(p["a"], p["b"]) for p in map(json.loads, done_path.read_text().splitlines())}
    for a, b in pairings:
        if (a, b) in done:
            continue
        result = await play(a, b, args.n_battles, sink, args.concurrency)
        with done_path.open("a") as fh:
            fh.write(json.dumps(result) + "\n")
        print(f"[shard {index}] {a:>17} vs {b:<17} {result['wins_a']:>4}/{result['n']} "
              f"({result['wins_a'] / result['n']:.1%}) {result['seconds']:.0f}s", flush=True)  # fmt: skip
    meta = run_dir / "meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({"participants": PARTICIPANTS, "n_battles": args.n_battles,
                                    "simulator_commit": simulator_commit()}, indent=1))  # fmt: skip
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("-n", "--n-battles", type=int, default=1000, help="battles per pairing")
    parser.add_argument("--shard", default="0/1", help="i/k: play every k-th pairing starting at i")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--report", action="store_true", help="build the report from the run directory")
    args = parser.parse_args()
    setup_logging(verbose=False)
    if args.report:
        from psbot.evaluation.report import build

        from psbot.report import write_report  # keeps hand-written text after the marker

        write_report(REPORT, build(Path(args.run_dir)))
        print(f"report: {REPORT}")
        return 0
    missing = [spec for spec in PARTICIPANTS.values() if spec.endswith(".zip") and not Path(spec).is_file()]
    if missing:
        print(f"error: checkpoint not found: {', '.join(missing)}", file=sys.stderr)
        return 1
    try:
        require_server()
        return asyncio.run(run_shard(args))
    except (ServerUnavailableError, BattleRunError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
