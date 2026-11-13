"""Phase 1 smoke test: play Gen 9 Random Battles on the local server.

Usage:
    uv run python scripts/run_random_battle.py            # 1 battle, verbose
    uv run python scripts/run_random_battle.py -n 20 -q   # 20 battles, summary only
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer

from psbot.config import BATTLE_FORMAT, local_server_configuration
from psbot.logging_setup import setup_logging
from psbot.players import InspectingRandomPlayer
from psbot.runner import BattleRunError, BattleSummary, run_battles
from psbot.server import ServerUnavailableError, require_server


async def main(n_battles: int) -> BattleSummary:
    server = local_server_configuration()
    # rand=True appends a random suffix so re-runs never collide on a username
    # that is still logged in from a previous run.
    player = InspectingRandomPlayer(
        account_configuration=AccountConfiguration.generate("Inspector", rand=True),
        battle_format=BATTLE_FORMAT,
        server_configuration=server,
    )
    opponent = RandomPlayer(
        account_configuration=AccountConfiguration.generate("Random", rand=True),
        battle_format=BATTLE_FORMAT,
        server_configuration=server,
    )
    return await run_battles(player, opponent, n_battles=n_battles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--n-battles", type=int, default=1)
    parser.add_argument("-q", "--quiet", action="store_true", help="hide per-turn logs")
    args = parser.parse_args()

    setup_logging(verbose=not args.quiet)
    try:
        require_server()
        summary = asyncio.run(main(args.n_battles))
    except (ServerUnavailableError, BattleRunError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(summary)
