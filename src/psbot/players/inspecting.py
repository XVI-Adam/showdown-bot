"""A random player that logs what it can see each turn.

This is the Phase 1 "battle interface" proof: it reads the battle state,
enumerates the legal moves and switches, and submits a legal action. Later
phases replace the random choice with a heuristic and then a learned policy;
the information available to them is exactly what is logged here.
"""

from __future__ import annotations

import logging

from poke_env.battle import AbstractBattle, Move, Pokemon
from poke_env.player import BattleOrder, RandomPlayer

logger = logging.getLogger(__name__)


def _describe_pokemon(mon: Pokemon | None) -> str:
    if mon is None:
        return "none"
    types = "/".join(t.name.title() for t in mon.types if t is not None)
    status = f" {mon.status.name}" if mon.status is not None else ""
    return f"{mon.species} L{mon.level} [{types}] {mon.current_hp_fraction:.0%}{status}"


def _describe_move(move: Move) -> str:
    return f"{move.id}({move.type.name.title()} {move.category.name.title()} bp={move.base_power})"


def describe_state(battle: AbstractBattle) -> str:
    """One-line, human-readable snapshot of the decision the player faces."""
    moves = ", ".join(_describe_move(m) for m in battle.available_moves) or "-"
    switches = ", ".join(m.species for m in battle.available_switches) or "-"
    return (
        f"turn {battle.turn}: {_describe_pokemon(battle.active_pokemon)} "
        f"vs {_describe_pokemon(battle.opponent_active_pokemon)} "
        f"| moves: {moves} | switches: {switches}"
    )


class InspectingRandomPlayer(RandomPlayer):
    """Picks uniformly among legal actions, logging the state it chose from."""

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        order = self.choose_random_move(battle)
        logger.info("%s -> %s", describe_state(battle), order.message)
        return order
