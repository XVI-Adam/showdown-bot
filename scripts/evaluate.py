"""Evaluate agents in fresh battles that were not part of training.

Usage:
    uv run python scripts/evaluate.py checkpoints/hybrid-v1/final.zip
    uv run python scripts/evaluate.py heuristic checkpoints/ppo-v1/final.zip checkpoints/hybrid-v1/final.zip -n 500
    uv run python scripts/evaluate.py checkpoints/hybrid-v1/final.zip --opponents heuristic checkpoints/ppo-v1/final.zip

Agents and opponents are each a scripted player name (random, max_base_power,
simple_heuristics, heuristic) or a path to a checkpoint, so any agent can be
measured against any other, including previously trained ones. Results are
saved to logs/stats/eval-<timestamp>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from poke_env import AccountConfiguration

from psbot.config import BATTLE_FORMAT, LOG_DIR, local_server_configuration
from psbot.logging_setup import setup_logging
from psbot.opponents import SCRIPTED_OPPONENTS, make_opponent
from psbot.runner import BattleRunError, run_battles, wilson_interval
from psbot.server import ServerUnavailableError, require_server


def short_name(spec: str) -> str:
    """'checkpoints/ppo-v1/final.zip' -> 'ppo-v1/final'; scripted names unchanged."""
    path = Path(spec)
    return f"{path.parent.name}/{path.stem}" if path.suffix == ".zip" else spec


async def evaluate(
    agent_spec: str, opponent_spec: str, n_battles: int, concurrency: int, deterministic: bool
) -> dict[str, Any]:
    server = local_server_configuration()
    common = dict(battle_format=BATTLE_FORMAT, server_configuration=server, max_concurrent_battles=concurrency)
    agent = make_opponent(
        agent_spec, account_configuration=AccountConfiguration.generate("Agent", rand=True), **common
    )
    foe = make_opponent(
        opponent_spec, account_configuration=AccountConfiguration.generate("Opp", rand=True), **common
    )
    if hasattr(agent, "deterministic"):
        agent.deterministic = deterministic
    summary = await run_battles(agent, foe, n_battles=n_battles)
    low, high = wilson_interval(summary.wins, summary.n_battles)
    # Learned agents count illegal actions; the heuristic counts engine fallbacks.
    decisions = getattr(agent, "n_decisions", None)
    illegal = getattr(agent, "n_illegal_actions", getattr(agent, "n_fallback_decisions", None))
    return {
        "agent": agent_spec,
        "opponent": opponent_spec,
        "n_battles": summary.n_battles,
        "wins": summary.wins,
        "win_rate": summary.win_rate,
        "ci95": [low, high],
        "avg_turns": summary.avg_turns,
        "avg_remaining_pokemon": summary.avg_remaining_pokemon,
        "decisions": decisions,
        "illegal_or_fallback_actions": illegal,
        "deterministic": deterministic if hasattr(agent, "deterministic") else None,
        # Hybrid agents only: share of real choices matching the heuristic's top action.
        "heuristic_agreement": (
            agent.n_heuristic_agreements / agent.n_choices if getattr(agent, "n_choices", 0) else None
        ),
    }


async def main(args: argparse.Namespace) -> int:
    results = []
    print(f"{'agent':<28} {'opponent':<24} {'wins':>9} {'win rate':>9} {'95% CI':>14} {'remaining':>10} {'illegal':>9} {'agree':>6}")
    for agent in args.agents:
        for opponent in args.opponents:
            r = await evaluate(agent, opponent, args.n_battles, args.concurrency, not args.stochastic)
            results.append(r)
            agree = "-" if r["heuristic_agreement"] is None else f"{r['heuristic_agreement']:.0%}"
            illegal = "-" if r["decisions"] is None else f"{r['illegal_or_fallback_actions']}/{r['decisions']}"
            print(
                f"{short_name(agent):<28} {short_name(opponent):<24} "
                f"{r['wins']:>4}/{r['n_battles']:<4} {r['win_rate']:>8.1%} "
                f"{r['ci95'][0]:>6.1%}-{r['ci95'][1]:<6.1%} {r['avg_remaining_pokemon']:>10.2f} {illegal:>9} {agree:>6}"
            )
    out = LOG_DIR / "stats" / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2))
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    scripted = ", ".join(SCRIPTED_OPPONENTS)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("agents", nargs="+", help=f"scripted names ({scripted}) or checkpoint paths")
    parser.add_argument(
        "--opponents", nargs="+", default=list(SCRIPTED_OPPONENTS), help="same choices as agents"
    )
    parser.add_argument("-n", "--n-battles", type=int, default=200, help="battles per pairing")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--stochastic", action="store_true", help="learned agents sample actions instead of taking the most likely"
    )
    cli_args = parser.parse_args()

    setup_logging(verbose=False)
    try:
        require_server()
        sys.exit(asyncio.run(main(cli_args)))
    except (ServerUnavailableError, BattleRunError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
