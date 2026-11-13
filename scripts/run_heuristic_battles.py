"""Phase 2: play the heuristic agent against baseline opponents and save statistics.

Usage:
    uv run python scripts/run_heuristic_battles.py              # 20 battles per opponent
    uv run python scripts/run_heuristic_battles.py -n 100
    uv run python scripts/run_heuristic_battles.py -n 1 --show  # print one battle's reasoning

Outputs (under logs/):
    battles/heuristic-<timestamp>.jsonl   one record per battle, with decision reasons
    stats/heuristic-<timestamp>.json      summary per opponent
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Any

from poke_env import AccountConfiguration
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer

from psbot.config import BATTLE_FORMAT, LOG_DIR, local_server_configuration
from psbot.logging_setup import setup_logging
from psbot.players import HeuristicPlayer
from psbot.recording import BattleRecorder
from psbot.runner import BattleRunError, run_battles
from psbot.server import ServerUnavailableError, require_server

OPPONENTS: dict[str, type[Player]] = {
    "random": RandomPlayer,
    "max_base_power": MaxBasePowerPlayer,
    "simple_heuristics": SimpleHeuristicsPlayer,
}


async def play_matchup(
    opponent_kind: str, n_battles: int, recorder: BattleRecorder, concurrency: int
) -> dict[str, Any]:
    server = local_server_configuration()
    player = HeuristicPlayer(
        account_configuration=AccountConfiguration.generate("Heuristic", rand=True),
        battle_format=BATTLE_FORMAT,
        server_configuration=server,
        max_concurrent_battles=concurrency,
        recorder=recorder,
        opponent_kind=opponent_kind,
    )
    opponent = OPPONENTS[opponent_kind](
        account_configuration=AccountConfiguration.generate(opponent_kind[:10], rand=True),
        battle_format=BATTLE_FORMAT,
        server_configuration=server,
        max_concurrent_battles=concurrency,
    )
    summary = await run_battles(player, opponent, n_battles=n_battles)
    return {
        **asdict(summary),
        "opponent_kind": opponent_kind,
        "win_rate": summary.win_rate,
        "decisions": player.n_decisions,
        "fallback_decisions": player.n_fallback_decisions,
    }


def show_battle(record: dict[str, Any]) -> None:
    result = "WON" if record["won"] else "LOST"
    print(f"\n{record['battle_tag']} vs {record['opponent_kind']}: {result} in {record['turns']} turns")
    for d in record["decisions"]:
        me, opp = d["active"], d["opponent"]
        matchup = f"{me['species']} {me['hp']:.0%}" if me else "-"
        matchup += f" vs {opp['species']} {opp['hp']:.0%}" if opp else ""
        print(f"  turn {d['turn']:>2} | {matchup} -> {d['action']} ({d['score']})")
        for reason in d["reasons"]:
            print(f"            {reason}")


async def main(args: argparse.Namespace) -> int:
    run_id = datetime.now().strftime("heuristic-%Y%m%d-%H%M%S")
    recorder = BattleRecorder(LOG_DIR / "battles" / f"{run_id}.jsonl")
    results = []
    for kind in args.opponents:
        result = await play_matchup(kind, args.n_battles, recorder, args.concurrency)
        results.append(result)
        print(
            f"vs {kind:<18} {result['wins']:>3}W-{result['losses']:>3}L "
            f"({result['win_rate']:.0%})  avg turns {result['avg_turns']:.1f}  "
            f"avg remaining {result['avg_remaining_pokemon']:.2f}  "
            f"fallbacks {result['fallback_decisions']}/{result['decisions']}"
        )

    stats_path = LOG_DIR / "stats" / f"{run_id}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps({"run_id": run_id, "results": results}, indent=2))
    print(f"\nbattle log: {recorder.path}\nstatistics: {stats_path}")

    if args.show:
        show_battle(recorder.records[0])
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--n-battles", type=int, default=20, help="battles per opponent")
    parser.add_argument("--opponents", nargs="+", choices=list(OPPONENTS), default=list(OPPONENTS))
    parser.add_argument("--concurrency", type=int, default=10, help="simultaneous battles")
    parser.add_argument("--show", action="store_true", help="print the first battle's reasoning")
    parser.add_argument("-v", "--verbose", action="store_true")
    cli_args = parser.parse_args()

    setup_logging(verbose=cli_args.verbose)
    try:
        require_server()
        sys.exit(asyncio.run(main(cli_args)))
    except (ServerUnavailableError, BattleRunError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
