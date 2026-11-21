"""Check that the bot account can reach and log in to the official server.

Usage:
    uv run python scripts/check_official_login.py --handshake-only   # no account needed
    uv run python scripts/check_official_login.py                    # log in, then log out

The login check needs PSBOT_USERNAME, PSBOT_PASSWORD and PSBOT_BOT_REGISTERED=1
in the environment or in .env (see psbot/battle_adapters/credentials.py). It
connects, logs in, confirms the username the server reports, and disconnects.
It joins no room, sends no challenge and plays no battle.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from psbot.battle_adapters.credentials import CredentialsError, load_credentials
from psbot.battle_adapters.official import OFFICIAL_SERVER, LoginError, OfficialPSClient
from psbot.logging_setup import setup_logging


async def main(args: argparse.Namespace) -> int:
    credentials = None if args.handshake_only else load_credentials()
    level = logging.INFO if args.verbose else None
    client = OfficialPSClient(credentials, log_level=level)
    print(f"connecting to {OFFICIAL_SERVER.websocket_url}")
    try:
        await client.wait_for_greeting()
        print("connected: the server sent its login challenge")
        if credentials is not None:
            name = await client.wait_until_logged_in()
            print(f"logged in as {name}")
    finally:
        await client.stop_listening()
    print("disconnected")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--handshake-only", action="store_true", help="connect without logging in")
    parser.add_argument("-v", "--verbose", action="store_true", help="show the protocol (redacted)")
    cli_args = parser.parse_args()

    setup_logging(verbose=True)
    try:
        sys.exit(asyncio.run(main(cli_args)))
    except KeyboardInterrupt:
        print("\nstopped")
    except (CredentialsError, LoginError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
