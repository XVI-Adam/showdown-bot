"""Measure how the simulator actually builds random-battle sets.

Usage:
    uv run python scripts/build_randbats_stats.py              # 200,000 teams, about 2 minutes
    uv run python scripts/build_randbats_stats.py -n 20000     # quicker, coarser

Asks the pinned Pokémon Showdown checkout to generate teams with its own team
generator and counts, per species, every (moves, ability, item) combination it
produces. The opponent move model (psbot/opponent_model) turns those counts
into probabilities: "this Garchomp has shown Earthquake and Swords Dance, so it
has Scale Shot 81% of the time".

Output: data/randbats/gen9-sets.json (git-ignored, a few MB). Re-run after
changing the simulator commit in scripts/setup_showdown.sh.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from psbot.config import PROJECT_ROOT
from psbot.opponent_model.sets import STATS_PATH

PS_DIR = PROJECT_ROOT / "vendor" / "pokemon-showdown"

NODE_SCRIPT = r"""
const {Teams} = require('./dist/sim');
const n = Number(process.argv[1]);
const out = {};
for (let i = 0; i < n; i++) {
  for (const set of Teams.generate('gen9randombattle')) {
    const species = out[set.speciesId] || (out[set.speciesId] = {n: 0, level: set.level, sets: {}});
    species.n++;
    const key = [...set.moves].sort().join(',') + '|' + set.ability + '|' + set.item;
    species.sets[key] = (species.sets[key] || 0) + 1;
  }
}
process.stdout.write(JSON.stringify(out));
"""


def main(args: argparse.Namespace) -> int:
    if not (PS_DIR / "dist" / "sim").is_dir():
        print("error: simulator not built; run scripts/setup_showdown.sh and start it once", file=sys.stderr)
        return 1
    commit = subprocess.run(
        ["git", "-C", str(PS_DIR), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(f"generating {args.n_teams:,} teams with the simulator at {commit[:10]} ...")
    result = subprocess.run(
        ["node", "-e", NODE_SCRIPT, str(args.n_teams)], cwd=PS_DIR, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stderr[-2000:], file=sys.stderr)
        return 1
    raw = json.loads(result.stdout)
    species = {}
    for species_id, entry in raw.items():
        sets = []
        for key, count in sorted(entry["sets"].items(), key=lambda kv: -kv[1]):
            moves, ability, item = key.split("|")
            sets.append([moves.split(","), ability, item, count])
        species[species_id] = {"n": entry["n"], "level": entry["level"], "sets": sets}
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(
        json.dumps({"simulator_commit": commit, "n_teams": args.n_teams, "species": species})
    )
    counts = sorted(entry["n"] for entry in species.values())
    print(
        f"{len(species)} species; samples per species: min {counts[0]}, "
        f"median {counts[len(counts) // 2]}, max {counts[-1]}"
    )
    print(f"wrote {STATS_PATH} ({STATS_PATH.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--n-teams", type=int, default=200_000)
    sys.exit(main(parser.parse_args()))
