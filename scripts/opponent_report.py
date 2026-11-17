"""Build opponent profiles from recorded human battles.

Usage:
    uv run python scripts/opponent_report.py                 # summary of everyone
    uv run python scripts/opponent_report.py SomeUsername    # one person's profile
    uv run python scripts/opponent_report.py --write         # also save profiles

Reads data/human/ and the older logs/human/. Profiles are saved to
data/profiles/ (git-ignored: they name real people). An opponent with fewer
than 5 battles gets counts only: one battle is mostly the random matchup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from poke_env.data import to_id_str

from psbot.analytics import build_profiles, load_battles, render_profile
from psbot.config import LOG_DIR
from psbot.dataset.human import DATA_DIR, HUMAN_DIR

PROFILE_DIR = DATA_DIR / "profiles"


def main(args: argparse.Namespace) -> int:
    sources = [Path(p) for p in args.source] or [HUMAN_DIR, LOG_DIR / "human"]
    profiles = build_profiles(load_battles(*sources))
    if not profiles:
        print("no human battles recorded yet")
        return 0
    if args.opponent:
        wanted = to_id_str(args.opponent)
        match = next((p for name, p in profiles.items() if to_id_str(name) == wanted), None)
        if match is None:
            print(f"no battles against {args.opponent!r}", file=sys.stderr)
            return 1
        print(render_profile(match))
    else:
        overall = profiles["*"]
        print(f"{overall['battles']} human battle(s) against {len(profiles) - 1} opponent(s)\n")
        for name, profile in profiles.items():
            record = profile["our_record"]
            label = "ALL OPPONENTS" if name == "*" else name
            tendencies = "; ".join(profile.get("tendencies", [])) or profile.get("note", "")
            print(f"  {label}: {profile['battles']} battle(s), we are {record['wins']}-{record['losses_or_ties']}, "
                  f"confidence {profile['confidence']}. {tendencies}")  # fmt: skip
    if args.write:
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        for name, profile in profiles.items():
            stem = "all-opponents" if name == "*" else to_id_str(name)
            (PROFILE_DIR / f"{stem}.json").write_text(json.dumps(profile, indent=2))
            (PROFILE_DIR / f"{stem}.md").write_text(render_profile(profile))
        print(f"\nwrote {len(profiles)} profile(s) to {PROFILE_DIR}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("opponent", nargs="?", help="show this opponent's full profile")
    parser.add_argument("--source", action="append", default=[], help="a .jsonl file or directory (repeatable)")
    parser.add_argument("--write", action="store_true", help="save profiles to data/profiles/")
    sys.exit(main(parser.parse_args()))
