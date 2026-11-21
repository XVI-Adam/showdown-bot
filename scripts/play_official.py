"""Play consent-based battles on the official Pokémon Showdown server.

Usage:
    # wait for a challenge from an approved user, play one battle, exit
    uv run python scripts/play_official.py --accept-from FriendName

    # challenge a player from the consent list once, or for a series of games
    uv run python scripts/play_official.py --challenge FriendName
    uv run python scripts/play_official.py --series FriendName -n 5

    # challenge everyone on the consent list in turn, 6 games in total
    uv run python scripts/play_official.py --rotate -n 6

    # the hybrid agent, up to three battles with either of two people
    uv run python scripts/play_official.py --bot checkpoints/hybrid-v2-sparse/final.zip \
        --accept-from FriendOne,FriendTwo -n 3

Only [Gen 9] Random Battle challenges from the named users or the consent list
are accepted; everything else is declined. The bot never searches the ladder
and only challenges players on the consent list (scripts/challenge_list.py),
within the limits in psbot/battle_adapters/challenges.py: one pending
challenge at a time, 15 s apart, a 10-minute pause after a decline, no more
after 2 declines in a day, and at most 20 challenges a day. Ctrl-C withdraws any challenge, forfeits a
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
from poke_env.data import to_id_str

from psbot.battle_adapters.challenges import ChallengeRunner, DailyState, Limits, read_consent
from psbot.config import BATTLE_FORMAT
from psbot.logging_setup import setup_logging
from psbot.dataset import HumanBattleRecorder, agent_info


def names(values: list[str] | None) -> list[str]:
    return [name.strip() for value in values or [] for name in value.split(",") if name.strip()]


def outgoing_targets(args: argparse.Namespace, consenting: list[str]) -> list[str]:
    """Who this run challenges. Only players on the consent list, ever."""
    wanted = [args.series] if args.series else [args.challenge] if args.challenge else consenting if args.rotate else []
    on_list = {to_id_str(n) for n in consenting}
    missing = [w for w in wanted if to_id_str(w) not in on_list]
    if missing:
        raise ValueError(
            f"{', '.join(missing)} not on the consent list. Ask them first, then: "
            f"uv run python scripts/challenge_list.py add {missing[0]}"
        )
    if args.rotate and not wanted:
        raise ValueError("the consent list is empty; add players with scripts/challenge_list.py add NAME")
    return wanted


async def main(args: argparse.Namespace) -> int:
    credentials = load_credentials()
    consenting = read_consent()
    targets = outgoing_targets(args, consenting)
    # Anyone on the consent list or named here may also challenge the bot.
    approved = names(args.accept_from) + consenting + targets
    if not approved:
        raise ValueError("name an opponent with --accept-from, --challenge, --series or --rotate")

    player = create_official_player(
        args.bot, credentials, log_level=logging.INFO if args.verbose else None
    )
    session = OfficialSession(player, approved, max_battles=args.n_battles)
    recorder = None
    if hasattr(player, "recorder"):
        recorder = HumanBattleRecorder(agent_info(args.bot, player), source="official")
        player.recorder = recorder
        player.opponent_kind = "human"

    housekeeping = None
    try:
        name = await player.ps_client.wait_until_logged_in()
        if name != credentials.username:
            raise LoginError(
                f"the server spells the username {name!r}; set PSBOT_USERNAME to exactly that"
            )
        print(f"logged in as {name}, playing {BATTLE_FORMAT} as '{args.bot}'")
        print(f"approved opponents: {', '.join(sorted(session.approved))}")
        if targets:
            limits = Limits(daily_cap=args.daily_cap)
            mode = "rotation through " + ", ".join(targets) if args.rotate else f"series against {targets[0]}"
            print(f"{mode}: up to {args.n_battles} game(s); today {DailyState.load().sent} of {limits.daily_cap} challenges used")
            # The session's own loop keeps time (challenge expiry, idle battles)
            # while the runner decides whom to challenge next.
            housekeeping = asyncio.ensure_future(session.run(timeout=None))
            runner = ChallengeRunner(
                session, targets, args.n_battles, limits,
                idle_timeout=args.timeout * 60 if args.timeout else float("inf"),
            )  # fmt: skip
            await runner.run()
            outcomes = list(session.outcomes)
        else:
            print("waiting for a challenge (Ctrl-C to stop)")
            outcomes = await session.run(timeout=args.timeout * 60 if args.timeout else None)
    finally:
        await session.close()
        if housekeeping is not None:
            housekeeping.cancel()

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
    parser.add_argument("--challenge", metavar="USER", help="challenge this user once (must be on the consent list)")
    parser.add_argument("--series", metavar="USER", help="challenge this user again after each game, up to -n games")
    parser.add_argument("--rotate", action="store_true", help="challenge everyone on the consent list in turn")
    parser.add_argument("--daily-cap", type=int, default=Limits().daily_cap, help="challenges per day, at most")
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
