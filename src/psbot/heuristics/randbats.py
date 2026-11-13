"""Knowledge specific to Gen 9 Random Battles.

We never see the opponent's stats or unrevealed moves, but random battles are
generated from public data that ships with the simulator. Reading it turns
"unknown" into "one of a short list":

* every stat has 85 EVs and 31 IVs with a neutral nature
  (``data/random-battles/gen9/teams.ts``: ``evs = {hp: 85, ...}``);
* every species draws its moves from a fixed pool (``sets.json``).
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from poke_env.battle import Move, Pokemon
from poke_env.data import GenData, to_id_str
from poke_env.stats import compute_raw_stats

from psbot.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

GEN = 9
SETS_PATH: Path = (
    PROJECT_ROOT / "vendor" / "pokemon-showdown" / "data" / "random-battles" / "gen9" / "sets.json"
)
RANDBATS_EV = 85
RANDBATS_IV = 31
STAT_NAMES: tuple[str, ...] = ("hp", "atk", "def", "spa", "spd", "spe")


@lru_cache(maxsize=1)
def _load_sets() -> dict[str, Any]:
    try:
        with SETS_PATH.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "random-battle sets unavailable (%s); opponents' unrevealed moves "
            "will be ignored. Run scripts/setup_showdown.sh to install them.",
            exc,
        )
        return {}


def _species_entry(mon: Pokemon) -> dict[str, Any] | None:
    """Look up a species, falling back to its base species for cosmetic formes."""
    sets = _load_sets()
    for key in (mon.species, to_id_str(mon.base_species)):
        if key in sets:
            return sets[key]
    return None


@lru_cache(maxsize=None)
def _move(move_id: str) -> Move:
    return Move(move_id, gen=GEN)


def possible_moves(mon: Pokemon) -> tuple[Move, ...]:
    """Every move this species can be generated with, across all its roles."""
    entry = _species_entry(mon)
    if entry is None:
        return ()
    ids = {to_id_str(name) for s in entry["sets"] for name in s["movepool"]}
    return tuple(_move(move_id) for move_id in sorted(ids))


def certain_ability(mon: Pokemon) -> str | None:
    """The species' ability if every generated set has the same one."""
    entry = _species_entry(mon)
    if entry is None:
        return None
    abilities = {to_id_str(a) for s in entry["sets"] for a in s["abilities"]}
    return next(iter(abilities)) if len(abilities) == 1 else None


def certain_item(mon: Pokemon) -> str | None:
    """The item a forme cannot exist without (Arceus plates, Ogerpon masks...).

    Matters for damage: Judgment takes its type from the held plate.
    """
    entry = GenData.from_gen(GEN).pokedex.get(mon.species, {})
    required = entry.get("requiredItem") or next(iter(entry.get("requiredItems", [])), None)
    return to_id_str(required) if required else None


def estimate_stats(mon: Pokemon) -> dict[str, int]:
    """Raw stats assuming the standard random-battle spread.

    Slightly high for the few sets generated with 0 Attack or 0 Speed EVs/IVs,
    which only affects stats those sets do not use (or want low).
    """
    raw = compute_raw_stats(
        mon.species,
        [RANDBATS_EV] * 6,
        [RANDBATS_IV] * 6,
        mon.level,
        "hardy",  # neutral nature
        GenData.from_gen(GEN),
    )
    return dict(zip(STAT_NAMES, raw))
