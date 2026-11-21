"""Manage the players who agreed to be challenged by the bot.

Usage:
    uv run python scripts/challenge_list.py                    # show the list and today's limits
    uv run python scripts/challenge_list.py add SomeName "asked on Discord"
    uv run python scripts/challenge_list.py remove SomeName

Add someone only after they have said yes. The list lives in
data/challenges/consenting.txt (git-ignored); you can also edit it by hand.
"""

from __future__ import annotations

import sys
import time

from psbot.battle_adapters.challenges import (
    CONSENT_FILE, DailyState, Limits, add_consent, read_consent, refusal, remove_consent,
)  # fmt: skip


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] in ("add", "remove"):
        name = argv[1]
        if argv[0] == "add":
            added = add_consent(name, " ".join(argv[2:]))
            print(f"added {name}" if added else f"{name} is already on the list")
        else:
            print(f"removed {name}" if remove_consent(name) else f"{name} was not on the list")
        return 0
    if argv:
        print(__doc__)
        return 1
    names = read_consent()
    state, limits = DailyState.load(), Limits()
    print(f"{len(names)} player(s) on {CONSENT_FILE}")
    print(f"today: {state.sent} of {limits.daily_cap} challenges sent\n")
    for name in names:
        why = refusal(state, limits, name, time.time())
        print(f"  {name:20} {'can be challenged' if why is None else why[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
