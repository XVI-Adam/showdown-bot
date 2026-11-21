"""Record a teacher agent's decisions in local battles, for behaviour cloning.

Usage:
    uv run python scripts/generate_teacher_data.py --name train -n 800
    uv run python scripts/generate_teacher_data.py --name test -n 200 --teacher heuristic_v2

Plays ``n`` battles against each opponent on the local simulator and saves the
teacher's (observation, legal mask, action) rows to data/synthetic/<name>.npz.
These are synthetic battles: they are kept apart from data/human/ and never
mixed into it. Start the simulator first (scripts/start_server.sh).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone

from poke_env import AccountConfiguration

from psbot.config import BATTLE_FORMAT, local_server_configuration
from psbot.logging_setup import setup_logging
from psbot.opponents import make_opponent
from psbot.runner import run_battles
from psbot.server import ServerUnavailableError, require_server
from psbot.training.dataset import SYNTHETIC_DIR, DecisionDataset, TeacherRecorder

DEFAULT_OPPONENTS = ["heuristic", "simple_heuristics", "max_base_power", "random", "checkpoints/ppo-v1/final.zip"]


async def main(args: argparse.Namespace) -> int:
    path = SYNTHETIC_DIR / f"{args.name}.npz"
    if path.exists() and not args.overwrite:
        print(f"error: {path} exists; pick another --name or pass --overwrite", file=sys.stderr)
        return 1
    parts = []
    for opponent in args.opponents:
        common = dict(
            battle_format=BATTLE_FORMAT,
            server_configuration=local_server_configuration(),
            max_concurrent_battles=args.concurrency,
        )
        teacher = make_opponent(args.teacher, account_configuration=AccountConfiguration.generate("Teach", rand=True), **common)
        foe = make_opponent(opponent, account_configuration=AccountConfiguration.generate("Foe", rand=True), **common)
        recorder = TeacherRecorder()
        teacher.recorder = recorder
        started = time.monotonic()
        summary = await run_battles(teacher, foe, n_battles=args.n_battles, timeout_per_battle=300)
        part = recorder.dataset({"teacher": args.teacher, "opponent": opponent, "battles": summary.n_battles, "wins": summary.wins})
        parts.append(part)
        print(f"{summary} -> {len(part):,} decisions ({time.monotonic() - started:.0f}s)", flush=True)
    data = DecisionDataset.concatenate(parts)
    data.meta.update(
        source="synthetic", teacher=args.teacher, created=datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    data.save(path)
    print(f"saved {len(data):,} decisions from {data.n_battles:,} battles to {path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="output name under data/synthetic/")
    parser.add_argument("--teacher", default="heuristic_v2")
    parser.add_argument("--opponents", nargs="+", default=DEFAULT_OPPONENTS)
    parser.add_argument("-n", "--n-battles", type=int, default=800, help="battles per opponent")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    setup_logging(verbose=False)
    try:
        require_server()
        sys.exit(asyncio.run(main(args)))
    except ServerUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
