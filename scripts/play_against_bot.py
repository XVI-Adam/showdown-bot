"""Play against a bot yourself, in the Showdown web client.

Usage:
    uv run python scripts/play_against_bot.py                       # the heuristic agent
    uv run python scripts/play_against_bot.py --bot checkpoints/hybrid-v2-sparse/final.zip
    uv run python scripts/play_against_bot.py --bot checkpoints/ppo-v1/final.zip -n 5

The bot waits for challenges on the local server (start it first with
scripts/start_server.sh) and accepts them from anyone who connects. It plays
instantly; you set your own pace. Finished battles are saved to
logs/human/<name>-<time>.jsonl with the bot's reasoning for each turn.

The bot does not learn from these games. Training happens in
scripts/train_ppo.py; these logs are for studying how the bot played.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from psbot.config import SERVER_HOST, SERVER_PORT
from psbot.hosting import DEFAULT_BOT_NAME, create_hosted_bot, default_log_path
from psbot.logging_setup import setup_logging
from psbot.server import ServerUnavailableError, require_server


async def main(args: argparse.Namespace) -> int:
    bot, recorder = create_hosted_bot(args.bot, args.name, default_log_path(args.name))
    print(
        f"\n{args.name} ({args.bot}) is online and waiting for {args.n_battles} challenge(s).\n\n"
        f"  1. Open  http://localhost:{SERVER_PORT}  in your browser\n"
        f"  2. Choose any username when asked (no password is needed)\n"
        f"  3. Click 'Find a user', type  {args.name}  and press Challenge\n"
        f"  4. Pick the format  [Gen 9] Random Battle  and send the challenge\n\n"
        f"Other formats are ignored. Press Ctrl-C to stop.\n",
        flush=True,
    )
    try:
        await bot.accept_challenges(None, args.n_battles)
    finally:
        await bot.ps_client.stop_listening()
    print(
        f"\nFinished {bot.n_finished_battles} battle(s): the bot won {bot.n_won_battles}, "
        f"you won {bot.n_lost_battles}."
    )
    if recorder is not None and recorder.n_written:
        print(f"Battle log: {recorder.path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bot", default="heuristic", help="'heuristic', another scripted bot, or a checkpoint .zip"
    )
    parser.add_argument("--name", default=DEFAULT_BOT_NAME, help="the bot's username (max 18 characters)")
    parser.add_argument("-n", "--n-battles", type=int, default=10, help="battles to play before exiting")
    cli_args = parser.parse_args()

    setup_logging(verbose=False)
    try:
        require_server(SERVER_HOST, SERVER_PORT)
        sys.exit(asyncio.run(main(cli_args)))
    except KeyboardInterrupt:
        print("\nstopped")
    except (ServerUnavailableError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
