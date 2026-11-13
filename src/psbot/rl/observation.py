"""Encode a battle as a fixed-size vector for the neural network.

The layout mirrors the action space so the network can connect "what I know
about slot i" to "what happens if I pick action i":

* ``MOVES``: 4 slots in the order poke-env maps move actions (6-9, 22-25):
  the active Pokémon's known moves, *not* ``battle.available_moves``. The
  official example uses the latter, which shifts features onto the wrong
  action whenever a move is disabled.
* ``TEAM``: 6 slots in the order poke-env maps switch actions (0-5).
* ``ACTIVE``: both active Pokémon: HP, status, boosts.
* ``GLOBAL``: speed order, team counts, hazards, flags.

Only directly observable facts and basic mechanics go in here. The heuristic
engine's judgement (expected damage, threats, action scores) is deliberately
left out so Phase 4 can measure what adding it is worth.

Every feature lies in [-1, 1].
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from poke_env.battle import AbstractBattle, Move, MoveCategory, Pokemon, SideCondition, Status

from psbot.heuristics.mechanics import effective_speed

N_MOVE_SLOTS = 4
N_TEAM_SLOTS = 6

MOVE_FEATURES = (
    "available",
    "base_power",
    "effectiveness",
    "immune",
    "stab",
    "accuracy",
    "priority",
    "physical",
    "special",
)
TEAM_FEATURES = (
    "is_active",
    "fainted",
    "switchable",
    "hp",
    "has_status",
    "offense",  # best effectiveness of its own types against their active
    "weakness",  # worst effectiveness of their active's types against it
)
ACTIVE_FEATURES = (
    "hp",
    "brn",
    "par",
    "psn",
    "slp",
    "frz",
    "boost_atk",
    "boost_def",
    "boost_spa",
    "boost_spd",
    "boost_spe",
    "terastallized",
)
GLOBAL_FEATURES = (
    "speed_share",  # our speed / (ours + theirs); > 0.5 means we move first
    "our_fainted",
    "their_fainted",
    "their_revealed",
    "can_tera",
    "force_switch",
    "our_stealth_rock",
    "our_spikes",
    "their_stealth_rock",
    "their_spikes",
)

OBSERVATION_SIZE = (
    N_MOVE_SLOTS * len(MOVE_FEATURES)
    + N_TEAM_SLOTS * len(TEAM_FEATURES)
    + 2 * len(ACTIVE_FEATURES)
    + len(GLOBAL_FEATURES)
)

_BOOST_STATS = ("atk", "def", "spa", "spd", "spe")
_STATUS_ORDER = (Status.BRN, Status.PAR, (Status.PSN, Status.TOX), Status.SLP, Status.FRZ)


def feature_names() -> list[str]:
    """Name of every index of the observation vector, for debugging and tests."""
    names = [f"move{i}.{f}" for i in range(N_MOVE_SLOTS) for f in MOVE_FEATURES]
    names += [f"team{i}.{f}" for i in range(N_TEAM_SLOTS) for f in TEAM_FEATURES]
    names += [f"{side}.{f}" for side in ("me", "opp") for f in ACTIVE_FEATURES]
    names += [f"global.{f}" for f in GLOBAL_FEATURES]
    return names


def _effectiveness(multiplier: float) -> float:
    """Map a type multiplier to [-1, 1]: 0.25x -> -1, 1x -> 0, 4x -> +1. Immune -> 0."""
    if multiplier <= 0:
        return 0.0
    return max(-1.0, min(1.0, math.log2(multiplier) / 2))


def action_move_slots(battle: AbstractBattle) -> list[Move]:
    """The moves behind move actions, in poke-env's action order."""
    active = battle.active_pokemon
    return list(active.moves.values())[:N_MOVE_SLOTS] if active is not None else []


def _encode_move(move: Move, battle: AbstractBattle, available_ids: set[str]) -> list[float]:
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    damaging = move.category != MoveCategory.STATUS
    multiplier = opp.damage_multiplier(move) if (opp is not None and damaging) else 1.0
    return [
        float(move.id in available_ids),
        min(move.base_power / 250, 1.0),
        _effectiveness(multiplier),
        float(multiplier == 0),
        float(damaging and me is not None and move.type in me.types),
        float(move.accuracy),
        max(-1.0, min(1.0, move.priority / 3)),
        float(move.category == MoveCategory.PHYSICAL),
        float(move.category == MoveCategory.SPECIAL),
    ]


def _encode_team_slot(
    mon: Pokemon, battle: AbstractBattle, switchable_species: set[str]
) -> list[float]:
    opp = battle.opponent_active_pokemon
    offense = weakness = 0.0
    if opp is not None and not mon.fainted:
        offense = _effectiveness(max(opp.damage_multiplier(t) for t in mon.types if t is not None))
        weakness = _effectiveness(max(mon.damage_multiplier(t) for t in opp.types if t is not None))
    return [
        float(bool(mon.active)),
        float(mon.fainted),
        float(mon.base_species in switchable_species),
        mon.current_hp_fraction,
        float(mon.status is not None and not mon.fainted),
        offense,
        weakness,
    ]


def _encode_active(mon: Pokemon | None) -> list[float]:
    if mon is None or mon.fainted:
        return [0.0] * len(ACTIVE_FEATURES)
    statuses = [
        float(mon.status in s if isinstance(s, tuple) else mon.status == s) for s in _STATUS_ORDER
    ]
    boosts = [mon.boosts[stat] / 6 for stat in _BOOST_STATS]
    return [mon.current_hp_fraction, *statuses, *boosts, float(mon.is_terastallized)]


def _encode_global(battle: AbstractBattle) -> list[float]:
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    speed_share = 0.5
    if me is not None and opp is not None and not me.fainted and not opp.fainted:
        ours, theirs = effective_speed(battle, me), effective_speed(battle, opp)
        if ours + theirs > 0:
            speed_share = ours / (ours + theirs)
    ours_side, theirs_side = battle.side_conditions, battle.opponent_side_conditions
    return [
        speed_share,
        sum(m.fainted for m in battle.team.values()) / N_TEAM_SLOTS,
        sum(m.fainted for m in battle.opponent_team.values()) / N_TEAM_SLOTS,
        len(battle.opponent_team) / N_TEAM_SLOTS,
        float(bool(battle.can_tera)),
        float(bool(battle.force_switch)),
        float(SideCondition.STEALTH_ROCK in ours_side),
        ours_side.get(SideCondition.SPIKES, 0) / 3,
        float(SideCondition.STEALTH_ROCK in theirs_side),
        theirs_side.get(SideCondition.SPIKES, 0) / 3,
    ]


def embed_battle(battle: AbstractBattle) -> npt.NDArray[np.float32]:
    """Encode ``battle`` as a float32 vector of length ``OBSERVATION_SIZE``."""
    features: list[float] = []

    available_ids = {m.id for m in battle.available_moves}
    moves = action_move_slots(battle)
    for slot in range(N_MOVE_SLOTS):
        if slot < len(moves):
            features += _encode_move(moves[slot], battle, available_ids)
        else:
            features += [0.0] * len(MOVE_FEATURES)

    switchable = set() if battle.trapped else {m.base_species for m in battle.available_switches}
    team = list(battle.team.values())[:N_TEAM_SLOTS]
    for slot in range(N_TEAM_SLOTS):
        if slot < len(team):
            features += _encode_team_slot(team[slot], battle, switchable)
        else:
            features += [0.0] * len(TEAM_FEATURES)

    features += _encode_active(battle.active_pokemon)
    features += _encode_active(battle.opponent_active_pokemon)
    features += _encode_global(battle)

    return np.clip(np.asarray(features, dtype=np.float32), -1.0, 1.0)
