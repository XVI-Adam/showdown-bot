"""Phase 5: round-robin evaluation of all agents and a performance report.

Usage:
    uv run python scripts/final_evaluation.py                       # 500 battles per pairing
    uv run python scripts/final_evaluation.py -n 100                # quicker, wider intervals
    uv run python scripts/final_evaluation.py --hybrid checkpoints/hybrid-v3/final.zip
    uv run python scripts/final_evaluation.py --report-only logs/eval/final-20261112-200000

Six participants play each other: random, max_base_power, simple_heuristics,
heuristic, ppo and hybrid. Battles of the heuristic and hybrid agents are
logged with decisions and events for the weakness analysis.

Outputs:
    logs/eval/<run>/results.json     raw numbers for every pairing
    logs/eval/<run>/battles.jsonl    battle logs of the heuristic and hybrid agents
    reports/final-evaluation.md      the report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from poke_env import AccountConfiguration
from poke_env.player import Player

from psbot.config import BATTLE_FORMAT, LOG_DIR, PROJECT_ROOT, local_server_configuration
from psbot.logging_setup import setup_logging
from psbot.opponents import make_opponent
from psbot.recording import BattleRecorder
from psbot.report import build_report, write_report
from psbot.runner import BattleRunError, run_battles, summarise
from psbot.server import ServerUnavailableError, require_server

RECORDED = ("heuristic", "hybrid")  # agents whose battles are logged for analysis
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "final-evaluation.md"


def simulator_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT / "vendor" / "pokemon-showdown"), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def make_player(name: str, spec: str, opponent: str, battles_path: Path, concurrency: int) -> Player:
    player = make_opponent(
        spec,
        account_configuration=AccountConfiguration.generate(name[:8], rand=True),
        battle_format=BATTLE_FORMAT,
        server_configuration=local_server_configuration(),
        max_concurrent_battles=concurrency,
    )
    if name in RECORDED and hasattr(player, "recorder"):
        player.recorder = BattleRecorder(
            battles_path, player_kind=name, save_events=True, keep_in_memory=False
        )
        player.opponent_kind = opponent
    return player


async def play_pairing(
    a: str, b: str, participants: dict[str, str], n: int, battles_path: Path, concurrency: int
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    player_a = make_player(a, participants[a], b, battles_path, concurrency)
    player_b = make_player(b, participants[b], a, battles_path, concurrency)
    summary_a = await run_battles(player_a, player_b, n_battles=n)
    summary_b = summarise(player_b, player_a, player_b.battles.values())
    pairing = {
        "a": a, "b": b, "n": summary_a.n_battles,
        "wins_a": summary_a.wins, "wins_b": summary_b.wins, "ties": summary_a.ties,
        "remaining_a": summary_a.avg_remaining_pokemon,
        "remaining_b": summary_b.avg_remaining_pokemon,
        "avg_turns": summary_a.avg_turns,
    }  # fmt: skip
    counters = {}
    for name, player in ((a, player_a), (b, player_b)):
        if hasattr(player, "n_decisions"):
            counters[name] = {
                "decisions": player.n_decisions,
                "illegal_actions": getattr(player, "n_illegal_actions", None),
                "fallback_decisions": getattr(player, "n_fallback_decisions", None),
            }
    return pairing, counters


async def run(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    participants = {
        "random": "random",
        "max_base_power": "max_base_power",
        "simple_heuristics": "simple_heuristics",
        "heuristic": "heuristic",
        "ppo": args.ppo,
        "hybrid": args.hybrid,
    }
    battles_path = run_dir / "battles.jsonl"
    pairings: list[dict[str, Any]] = []
    agents: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    for a, b in combinations(participants, 2):
        pairing, counters = await play_pairing(a, b, participants, args.n_battles, battles_path, args.concurrency)
        pairings.append(pairing)
        for name, values in counters.items():
            total = agents.setdefault(name, {"decisions": 0, "illegal_actions": None, "fallback_decisions": None})
            total["decisions"] += values["decisions"]
            for key in ("illegal_actions", "fallback_decisions"):
                if values[key] is not None:
                    total[key] = (total[key] or 0) + values[key]
        print(
            f"{a:>18} vs {b:<18} {pairing['wins_a']:>4}-{pairing['wins_b']:<4} "
            f"({pairing['wins_a'] / pairing['n']:.1%})",
            flush=True,
        )
    return {
        "run_id": run_dir.name,
        "n_battles": args.n_battles,
        "participants": participants,
        "pairings": pairings,
        "agents": agents,
        "elapsed_s": round(time.monotonic() - started, 1),
        "simulator_commit": simulator_commit(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ppo", default="checkpoints/ppo-v1/final.zip")
    parser.add_argument("--hybrid", default="checkpoints/hybrid-v2-sparse/final.zip")
    parser.add_argument("-n", "--n-battles", type=int, default=500, help="battles per pairing")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--report-only", type=Path, help="rebuild the report from an existing run directory")
    args = parser.parse_args()
    setup_logging(verbose=False)

    if args.report_only:
        run_dir = args.report_only
        results = json.loads((run_dir / "results.json").read_text())
    else:
        missing = [p for p in (args.ppo, args.hybrid) if not Path(p).is_file()]
        if missing:
            print(f"error: checkpoint not found: {', '.join(missing)}", file=sys.stderr)
            return 1
        run_dir = LOG_DIR / "eval" / f"final-{datetime.now():%Y%m%d-%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            require_server()
            results = asyncio.run(run(args, run_dir))
        except (ServerUnavailableError, BattleRunError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    write_report(args.report, build_report(results, run_dir / "battles.jsonl"))
    print(f"\nresults: {run_dir / 'results.json'}\nreport:  {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
