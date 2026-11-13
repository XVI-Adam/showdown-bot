"""Build poke-env battle states offline, without a server.

These set private attributes on poke-env objects. That is acceptable in tests
(it is how poke-env's own unit tests build states) and keeps every scoring
test fast and deterministic.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

from poke_env.battle import Battle, Move, Pokemon, PokemonType, Status

from psbot.heuristics.randbats import estimate_stats

GEN = 9


def make_mon(
    species: str,
    moves: Sequence[str] = (),
    *,
    own: bool = True,
    level: int = 80,
    hp: float = 1.0,
    status: Status | None = None,
    item: str | None = None,
    ability: str | None = None,
    boosts: dict[str, int] | None = None,
    tera_type: PokemonType | None = None,
) -> Pokemon:
    """A Pokémon as we would see it: exact stats if ours, percentages if not."""
    mon = Pokemon(gen=GEN, species=species)
    mon._level = level
    if own:
        stats = estimate_stats(mon)
        mon.stats = stats
        mon._max_hp = stats["hp"]
        mon._current_hp = round(stats["hp"] * hp)
    else:
        mon._max_hp = 100
        mon._current_hp = round(100 * hp)
    for move_id in moves:
        mon._moves[move_id] = Move(move_id, gen=GEN)
    mon._status = status
    if item is not None:
        mon._item = item
    if ability is not None:
        mon._ability = ability
    for stat, stage in (boosts or {}).items():
        mon._boosts[stat] = stage
    if tera_type is not None:
        mon._terastallized_type = tera_type
    return mon


def make_battle(
    active: Pokemon | None,
    opponent: Pokemon | None,
    bench: Iterable[Pokemon] = (),
    opp_bench: Iterable[Pokemon] = (),
    *,
    force_switch: bool = False,
    can_tera: bool = False,
) -> Battle:
    """A singles battle where we are p1, with legal actions filled in."""
    battle = Battle("battle-gen9randombattle-1", "tester", logging.getLogger("tests"), gen=GEN)
    battle.player_role = "p1"
    bench = list(bench)
    if active is not None:
        active._active = True
    if opponent is not None:
        opponent._active = True
    ours = ([active] if active is not None else []) + bench
    theirs = ([opponent] if opponent is not None else []) + list(opp_bench)
    battle._team = {m.identifier("p1"): m for m in ours}
    battle._opponent_team = {m.identifier("p2"): m for m in theirs}
    battle._force_switch = force_switch
    battle._can_tera = can_tera
    if active is not None and not force_switch:
        battle._available_moves = list(active.moves.values())
    battle._available_switches = [m for m in bench if not m.fainted]
    return battle
