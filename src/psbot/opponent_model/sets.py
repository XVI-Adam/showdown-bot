"""A belief over which set an opposing Pokémon is running.

Random-battle sets are not arbitrary: the simulator builds each one from a
short list per species. ``scripts/build_randbats_stats.py`` asks the simulator
to do that a few hundred thousand times and counts the results, so for each
species we hold every (moves, ability, item) combination it produces and how
often. Conditioning those counts on what a Pokémon has shown is then plain
counting:

    P(has move m | shown) = weight of consistent sets containing m
                            / weight of consistent sets

A revealed move has probability 1 and is flagged ``revealed``. Every other
number here is an inference and stays labelled as one: a move with
probability 1.0 that has not been used is still not "confirmed", because the
table can be out of date (a newer server) or the Pokémon can be an Illusion.

If the measured table is missing, the same interface is served from the
species' move pools in ``sets.json``, assuming the four moves are drawn
uniformly from a role's pool. That is cruder (the generator enforces STAB and
avoids some pairs) and says nothing about items.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any, Iterable

from poke_env.battle import Pokemon
from poke_env.data import GenData, to_id_str

from psbot.config import PROJECT_ROOT
from psbot.heuristics.randbats import GEN, _species_entry

logger = logging.getLogger(__name__)

STATS_PATH: Path = PROJECT_ROOT / "data" / "randbats" / "gen9-sets.json"
UNKNOWN_ITEM: str = GenData.from_gen(GEN).UNKNOWN_ITEM
# Enumerating every 4-move subset of a pool is fine up to this size (C(10,4)=210).
_MAX_POOL_TO_ENUMERATE = 10

# Shown to the opponent under another forme's name, so both must be candidates.
_DISPLAYED_AS = {"greninjabond": "greninja"}

Candidate = tuple[frozenset[str], str, str, float]  # moves, ability, item, weight


@lru_cache(maxsize=1)
def _measured() -> dict[str, list[Candidate]]:
    try:
        raw = json.loads(STATS_PATH.read_text(encoding="utf-8"))["species"]
    except (OSError, ValueError, KeyError) as exc:
        logger.warning(
            "measured set statistics unavailable (%s); using move pools instead. "
            "Build them with: uv run python scripts/build_randbats_stats.py",
            type(exc).__name__,
        )
        return {}
    table: dict[str, list[Candidate]] = {}
    for species, entry in raw.items():
        sets = [
            (frozenset(moves), to_id_str(ability), to_id_str(item), float(count))
            for moves, ability, item, count in entry["sets"]
        ]
        table.setdefault(species, []).extend(sets)
        shown_as = _DISPLAYED_AS.get(species)
        if shown_as:
            table.setdefault(shown_as, []).extend(sets)
    return table


@lru_cache(maxsize=None)
def _from_pools(species: str, base_species: str) -> tuple[Candidate, ...]:
    """Fallback candidates: every 4-move subset of each role's pool, equally
    likely within a role, roles equally likely. Ability and item unknown."""
    mon = Pokemon(gen=GEN, species=species)
    entry = _species_entry(mon)
    if entry is None:
        return ()
    candidates: list[Candidate] = []
    roles = entry["sets"]
    for role in roles:
        pool = sorted({to_id_str(name) for name in role["movepool"]})
        size = min(4, len(pool))
        if len(pool) > _MAX_POOL_TO_ENUMERATE:
            continue
        weight = 1.0 / (len(roles) * comb(len(pool), size))
        abilities = [to_id_str(a) for a in role["abilities"]]
        for moves in combinations(pool, size):
            for ability in abilities:
                candidates.append((frozenset(moves), ability, "", weight / len(abilities)))
    return tuple(candidates)


def candidates_for(mon: Pokemon) -> tuple[list[Candidate] | tuple[Candidate, ...], bool]:
    """Candidate sets for a species, and whether they come from measurement."""
    table = _measured()
    for key in (mon.species, to_id_str(mon.base_species)):
        if key in table:
            return table[key], True
    return _from_pools(mon.species, to_id_str(mon.base_species)), False


@dataclass(frozen=True)
class SetBelief:
    """Sets still consistent with what a Pokémon has shown, with weights."""

    revealed: frozenset[str]
    candidates: tuple[Candidate, ...]
    measured: bool  # False: derived from move pools; items unknown
    relaxed: tuple[str, ...] = ()  # evidence dropped because nothing matched it

    @property
    def total(self) -> float:
        return sum(c[3] for c in self.candidates)

    def move_probabilities(self) -> dict[str, float]:
        """Probability of each move not yet revealed (revealed ones are 1 by
        definition and left out)."""
        total = self.total
        if total <= 0:
            return {}
        weights: dict[str, float] = {}
        for moves, _, _, weight in self.candidates:
            for move in moves - self.revealed:
                weights[move] = weights.get(move, 0.0) + weight
        return {move: min(1.0, w / total) for move, w in sorted(weights.items())}

    def move_probability(self, move_id: str) -> float:
        if move_id in self.revealed:
            return 1.0
        return self.move_probabilities().get(move_id, 0.0)

    def _marginal(self, index: int) -> dict[str, float]:
        total = self.total
        if total <= 0:
            return {}
        weights: dict[str, float] = {}
        for candidate in self.candidates:
            if candidate[index]:
                weights[candidate[index]] = weights.get(candidate[index], 0.0) + candidate[3]
        return {k: w / total for k, w in sorted(weights.items(), key=lambda kv: -kv[1])}

    def item_probabilities(self) -> dict[str, float]:
        """Empty when unknown (pool-derived beliefs carry no item data)."""
        return self._marginal(2)

    def ability_probabilities(self) -> dict[str, float]:
        return self._marginal(1)


def set_belief(
    mon: Pokemon,
    revealed_moves: Iterable[str],
    ability: str | None = None,
    item: str | None = None,
) -> SetBelief:
    """Condition a species' candidate sets on evidence.

    ``ability`` and ``item`` should be passed only when known. Evidence that
    no candidate matches is dropped, least reliable first (the item, which can
    be swapped or knocked off; then the ability, which can be overwritten;
    then the moves themselves), and listed in ``relaxed``.
    """
    candidates, measured = candidates_for(mon)
    revealed = frozenset(revealed_moves)
    ability = to_id_str(ability) if ability else None
    item = to_id_str(item) if item else None

    def consistent(use_moves: bool, use_ability: bool, use_item: bool) -> list[Candidate]:
        return [
            c
            for c in candidates
            if (not use_moves or revealed <= c[0])
            and (not use_ability or ability is None or not c[1] or c[1] == ability)
            and (not use_item or item is None or not c[2] or c[2] == item)
        ]

    for relaxed, flags in (
        ((), (True, True, True)),
        (("item",), (True, True, False)),
        (("item", "ability"), (True, False, False)),
        (("item", "ability", "moves"), (False, False, False)),
    ):
        kept = consistent(*flags)
        if kept:
            return SetBelief(revealed, tuple(kept), measured, relaxed)
    return SetBelief(revealed, (), measured, ("item", "ability", "moves"))


def belief_for(mon: Pokemon) -> SetBelief:
    """The belief for an opposing Pokémon as poke-env currently describes it.

    Uses its revealed moves, and its ability and item when they are set. An
    item slot that is empty (consumed, knocked off) is not evidence about the
    original set and is ignored.
    """
    item: Any = mon.item
    if not item or item == UNKNOWN_ITEM:
        item = None
    return set_belief(mon, mon.moves.keys(), mon.ability, item)
