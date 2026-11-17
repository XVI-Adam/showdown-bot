"""Registry of opponents, shared by training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer

from psbot.players.heuristic import HeuristicPlayer, HeuristicV2Player
from psbot.players.ppo import PPOPlayer

SCRIPTED_OPPONENTS: dict[str, type[Player]] = {
    "random": RandomPlayer,
    "max_base_power": MaxBasePowerPlayer,
    "simple_heuristics": SimpleHeuristicsPlayer,
    "heuristic": HeuristicPlayer,
    "heuristic_v2": HeuristicV2Player,
}


def make_opponent(kind: str, **player_kwargs: Any) -> Player:
    """Create an opponent by name, or a PPO agent from a checkpoint path.

    ``kind`` is a key of ``SCRIPTED_OPPONENTS`` or a path to a ``.zip``
    checkpoint, which lets a new agent play against previously trained ones.
    """
    if kind in SCRIPTED_OPPONENTS:
        return SCRIPTED_OPPONENTS[kind](**player_kwargs)
    path = Path(kind)
    if path.suffix == ".zip" and path.is_file():
        return PPOPlayer.from_checkpoint(path, **player_kwargs)
    raise ValueError(
        f"unknown opponent {kind!r}: expected one of {sorted(SCRIPTED_OPPONENTS)} "
        "or a path to a .zip checkpoint"
    )
