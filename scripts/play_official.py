"""Play consent-based battles on the official Pokémon Showdown server.

Usage:
    # wait for a challenge from an approved user, play one battle, exit
    uv run python scripts/play_official.py --accept-from FriendName

    # challenge an approved user once
    uv run python scripts/play_official.py --challenge FriendName

    # the hybrid agent, up to three battles with either of two people
    uv run python scripts/play_official.py --bot checkpoints/hybrid-v2-sparse/final.zip \
        --accept-from FriendOne,FriendTwo -n 3

Only [Gen 9] Random Battle challenges from the named users are accepted;
everything else is declined. The bot never searches the ladder and never
challenges anyone you did not name. Ctrl-C withdraws any challenge, forfeits a
battle in progress and logs out.

Needs the bot account in .env (see README, "Official server"). Finished
battles are saved to data/human/ with the bot's reasoning for each turn and
what was known about the opponent at that moment.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from psbot.battle_adapters.credentials import CredentialsError, load_credentials
from psbot.battle_adapters.official import LoginError, create_official_player
from psbot.battle_adapters.session import OfficialSession
from psbot.config import BATTLE_FORMAT
from psbot.logging_setup import setup_logging
from psbot.dataset import HumanBattleRecorder, agent_info


def names(values: list[str] | None) -> list[str]:
    return [name.strip() for value in values or [] for name in value.split(",") if name.strip()]


async def main(args: argparse.Namespace) -> int:
    credentials = load_credentials()
    approved = names(args.accept_from) + ([args.challenge] if args.challenge else [])
    if not approved:
        raise ValueError("name at least one opponent with --accept-from or --challenge")

    player = create_official_player(
        args.bot, credentials, log_level=logging.INFO if args.verbose else None
    )
    session = OfficialSession(player, approved, max_battles=args.n_battles)
    recorder = None
    if hasattr(player, "recorder"):
        recorder = HumanBattleRecorder(agent_info(args.bot, player), source="official")
        player.recorder = recorder
        player.opponent_kind = "human"

    try:
        name = await player.ps_client.wait_until_logged_in()
        if name != credentials.username:
            raise LoginError(
                f"the server spells the username {name!r}; set PSBOT_USERNAME to exactly that"
            )
        print(f"logged in as {name}, playing {BATTLE_FORMAT} as '{args.bot}'")
        print(f"approved opponents: {', '.join(sorted(session.approved))}")
        if args.challenge:
            await session.challenge(args.challenge)
            print(f"challenged {args.challenge}; waiting for them to accept")
        else:
            print("waiting for a challenge (Ctrl-C to stop)")
        outcomes = await session.run(timeout=args.timeout * 60 if args.timeout else None)
    finally:
        await session.close()

    for outcome in session.outcomes:
        print(f"  {outcome.battle_tag}: {outcome}")
    if len(outcomes) < args.n_battles:
        why = "the connection was lost" if player.ps_client.closed else "time ran out"
        print(f"stopped after {len(outcomes)} of {args.n_battles} battle(s): {why}")
    if session.n_rejected:
        print(f"declined {session.n_rejected} challenge(s)")
    if recorder is not None and recorder.n_written:
        print(f"battle record: {recorder.path}")
        print(f"check it with: uv run python scripts/validate_dataset.py {recorder.path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bot", default="heuristic", help="'heuristic', another scripted bot, or a checkpoint .zip"
    )
    parser.add_argument(
        "--accept-from", action="append", metavar="USER[,USER]",
        help="usernames whose challenges are accepted (repeatable)",
    )  # fmt: skip
    parser.add_argument("--challenge", metavar="USER", help="challenge this user once")
    parser.add_argument("-n", "--n-battles", type=int, default=1, help="battles to play before exiting")
    parser.add_argument("--timeout", type=float, default=30, help="minutes to wait while idle (0: forever)")
    parser.add_argument("-v", "--verbose", action="store_true", help="show the protocol (redacted)")
    cli_args = parser.parse_args()

    setup_logging(verbose=True)
    try:
        sys.exit(asyncio.run(main(cli_args)))
    except KeyboardInterrupt:
        print("\nstopped")
    except (CredentialsError, LoginError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
