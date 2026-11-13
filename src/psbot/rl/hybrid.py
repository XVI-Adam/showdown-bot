"""Hybrid observation: the basic encoding plus the heuristic engine's analysis.

The hybrid agent sees everything the PPO agent sees (112 features) and, in
addition, what the heuristic engine concluded about this turn:

* ``prior``: the engine's total score for each of the 26 actions. The hybrid
  policy also adds these to its logits (see ``HybridActorCriticPolicy``).
* per-move details: expected damage, KO chance, chance to act at all.
* per-switch details: damage taken on entry, expected exchange afterwards.
* the threat to our active Pokémon and its current exchange.

This is information, not a decision. The network is free to ignore it.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import numpy.typing as npt
from poke_env.battle import AbstractBattle, Move

from psbot.heuristics.context import BattleContext
from psbot.heuristics.engine import HeuristicEngine, terastallize_reason
from psbot.heuristics.scorers import entry_damage
from psbot.rl.observation import (
    N_MOVE_SLOTS,
    N_TEAM_SLOTS,
    OBSERVATION_SIZE,
    action_move_slots,
    embed_battle,
    feature_names,
)

logger = logging.getLogger(__name__)

N_ACTIONS = 26
SWITCH_ACTION_OFFSET = 0  # actions 0-5 switch to team slot i
MOVE_ACTION_OFFSET = 6  # actions 6-9 use move slot i
TERA_ACTION_OFFSET = 22  # actions 22-25 use move slot i and Terastallize

# Heuristic scores are in HP-bar units, almost always within +/-4. They are
# divided by this to fit the observation's [-1, 1] range.
PRIOR_CLIP = 4.0
# The engine recommends Tera or not; reflect that as a small nudge on the Tera
# variant of a move, weak enough that learning can easily overrule it.
TERA_NUDGE = 0.05

HYBRID_MOVE_FEATURES = ("expected_damage", "ko_probability", "act_probability")
HYBRID_SWITCH_FEATURES = ("entry_damage", "exchange")
HYBRID_GLOBAL_FEATURES = ("threat_expected", "threat_ko_probability", "exchange")

HEURISTIC_FEATURE_SIZE = (
    N_ACTIONS
    + N_MOVE_SLOTS * len(HYBRID_MOVE_FEATURES)
    + N_TEAM_SLOTS * len(HYBRID_SWITCH_FEATURES)
    + len(HYBRID_GLOBAL_FEATURES)
)
HYBRID_OBSERVATION_SIZE = OBSERVATION_SIZE + HEURISTIC_FEATURE_SIZE
# Where the per-action scores sit inside the hybrid observation vector.
PRIOR_SLICE = slice(OBSERVATION_SIZE, OBSERVATION_SIZE + N_ACTIONS)

_DEFAULT_ENGINE = HeuristicEngine()


def hybrid_feature_names() -> list[str]:
    names = feature_names()
    names += [f"prior.action{i}" for i in range(N_ACTIONS)]
    names += [f"move{i}.{f}" for i in range(N_MOVE_SLOTS) for f in HYBRID_MOVE_FEATURES]
    names += [f"team{i}.{f}" for i in range(N_TEAM_SLOTS) for f in HYBRID_SWITCH_FEATURES]
    names += [f"heuristic.{f}" for f in HYBRID_GLOBAL_FEATURES]
    return names


def _move_action_slots(battle: AbstractBattle) -> list[Move]:
    """Moves behind the move actions, following poke-env's ``action_to_order``:
    normally the active Pokémon's known moves, but a lone special move such as
    Struggle or recharge takes slot 0."""
    known = action_move_slots(battle)
    available = battle.available_moves
    if len(available) == 1 and available[0].id not in {m.id for m in known}:
        return list(available)
    return known


def heuristic_action_scores(
    battle: AbstractBattle, engine: HeuristicEngine, ctx: BattleContext | None = None
) -> npt.NDArray[np.float32]:
    """The engine's score for each of the 26 actions (0 for illegal ones)."""
    scores = np.zeros(N_ACTIONS, dtype=np.float32)
    ctx = ctx or BattleContext(battle, engine.weights)
    move_ids = [m.id for m in _move_action_slots(battle)]
    team_species = [m.base_species for m in battle.team.values()]

    for evaluation in engine.evaluate(battle, ctx):
        target = evaluation.target
        if isinstance(target, Move):
            if target.id not in move_ids:
                continue
            slot = move_ids.index(target.id)
            scores[MOVE_ACTION_OFFSET + slot] = evaluation.total
            if battle.can_tera:
                recommended = (
                    len(evaluation.components) > 0
                    and evaluation.components[0].name != "forced"
                    and terastallize_reason(ctx, target) is not None
                )
                nudge = TERA_NUDGE if recommended else -TERA_NUDGE
                scores[TERA_ACTION_OFFSET + slot] = evaluation.total + nudge
        elif target.base_species in team_species:
            slot = team_species.index(target.base_species)
            if slot < N_TEAM_SLOTS:
                scores[SWITCH_ACTION_OFFSET + slot] = evaluation.total
    return scores


def _detail_features(battle: AbstractBattle, ctx: BattleContext) -> list[float]:
    features: list[float] = []
    me, opp = ctx.me, ctx.opp
    in_play = me is not None and opp is not None and not me.fainted

    available_ids = {m.id for m in battle.available_moves}
    moves = action_move_slots(battle)
    for slot in range(N_MOVE_SLOTS):
        if in_play and slot < len(moves) and moves[slot].id in available_ids:
            estimate = ctx.damage(me, moves[slot])
            features += [
                min(estimate.expected, 1.0),
                estimate.ko_probability(ctx.opp_hp),
                ctx.act_probability(moves[slot]),
            ]
        else:
            features += [0.0] * len(HYBRID_MOVE_FEATURES)

    switchable = {id(m) for m in battle.available_switches}
    team = list(battle.team.values())[:N_TEAM_SLOTS]
    for slot in range(N_TEAM_SLOTS):
        if slot < len(team) and id(team[slot]) in switchable:
            mon = team[slot]
            hazards, hit = entry_damage(ctx, mon)
            exchange = ctx.exchange(mon, hp=mon.current_hp_fraction - hazards - hit)
            features += [min(hazards + hit, 1.0), exchange]
        else:
            features += [0.0] * len(HYBRID_SWITCH_FEATURES)

    if in_play:
        threat = ctx.threat(me).damage
        features += [min(threat.expected, 1.0), threat.ko_probability(ctx.my_hp), ctx.exchange(me)]
    else:
        features += [0.0] * len(HYBRID_GLOBAL_FEATURES)
    return features


def heuristic_features(
    battle: AbstractBattle, engine: HeuristicEngine | None = None
) -> npt.NDArray[np.float32]:
    """The heuristic part of the hybrid observation, scaled into [-1, 1].

    If the engine fails on some unusual state the features are zero, which
    leaves the policy with exactly the basic observation for that turn.
    """
    engine = engine or _DEFAULT_ENGINE
    try:
        ctx = BattleContext(battle, engine.weights)
        scores = heuristic_action_scores(battle, engine, ctx) / PRIOR_CLIP
        details = np.asarray(_detail_features(battle, ctx), dtype=np.float32)
        features = np.concatenate([scores, details])
    except Exception:
        logger.warning("heuristic features failed on turn %s", battle.turn, exc_info=True)
        features = np.zeros(HEURISTIC_FEATURE_SIZE, dtype=np.float32)
    return np.clip(features, -1.0, 1.0).astype(np.float32)


def embed_battle_hybrid(
    battle: AbstractBattle, engine: HeuristicEngine | None = None
) -> npt.NDArray[np.float32]:
    """Basic observation followed by the heuristic features."""
    return np.concatenate([embed_battle(battle), heuristic_features(battle, engine)])


Encoder = Callable[[AbstractBattle], npt.NDArray[np.float32]]
ENCODERS: dict[str, tuple[int, Encoder]] = {
    "basic": (OBSERVATION_SIZE, embed_battle),
    "hybrid": (HYBRID_OBSERVATION_SIZE, embed_battle_hybrid),
}


def encoder_for_size(size: int) -> Encoder:
    """The encoder a saved policy was trained with, identified by its input size."""
    for expected, encoder in ENCODERS.values():
        if expected == size:
            return encoder
    raise ValueError(f"no observation encoder produces {size} features")
