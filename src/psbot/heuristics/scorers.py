"""Individual scoring components.

Each scorer looks at one aspect of one action and returns a
``ScoreComponent`` (a number plus the reason for it), or ``None`` when the
aspect does not apply. The engine sums the components. Keeping them separate
means each can be unit-tested, re-weighted or replaced on its own, and every
decision can be explained by listing its components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from poke_env.battle import Move, MoveCategory, Pokemon, SideCondition, Target

from psbot.heuristics.context import BattleContext
from psbot.heuristics.damage import max_hp_points
from psbot.heuristics.mechanics import (
    HAZARD_REMOVAL_MOVES,
    MAX_LAYERS,
    hazard_entry_fraction,
    status_block_reason,
)

# Healing moves whose amount is not in the move data's ``heal`` field.
# Moonlight/Morning Sun/Synthesis/Shore Up restore 1/2 max HP in neutral
# weather; Rest restores all of it.
EXTRA_RECOVERY_FRACTIONS = {
    "rest": 1.0,
    "moonlight": 0.5,
    "morningsun": 0.5,
    "synthesis": 0.5,
    "shoreup": 0.5,
}


@dataclass(frozen=True)
class ScoreComponent:
    """One named contribution to an action's score, with its justification."""

    name: str
    value: float
    reason: str

    def __str__(self) -> str:
        return f"{self.name} {self.value:+.2f} ({self.reason})"


MoveScorer = Callable[[BattleContext, Move], ScoreComponent | None]
SwitchScorer = Callable[[BattleContext, Pokemon], ScoreComponent | None]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


# --------------------------------------------------------------------------
# Move scorers
# --------------------------------------------------------------------------


def score_damage(ctx: BattleContext, move: Move) -> ScoreComponent | None:
    """Expected damage: type effectiveness, STAB, power and accuracy, via the calc."""
    if move.category == MoveCategory.STATUS or ctx.me is None or ctx.opp is None:
        return None
    estimate = ctx.damage(ctx.me, move)
    if estimate.max_fraction <= 0:
        return ScoreComponent("damage", 0.0, "no effect on the target")
    p_act = ctx.act_probability(move)
    dealt = min(estimate.mean_fraction, ctx.opp_hp) * estimate.accuracy
    reason = (
        f"{estimate.min_fraction:.0%}-{estimate.max_fraction:.0%} of max HP, "
        f"{estimate.accuracy:.0%} accurate"
    )
    if p_act < 1:
        reason += f", {p_act:.0%} to get the move off"
    return ScoreComponent("damage", ctx.weights.damage * dealt * p_act, reason)


def score_ko(ctx: BattleContext, move: Move) -> ScoreComponent | None:
    """Bonus for knocking the target out, larger if we strike first (speed/priority)."""
    if ctx.me is None or ctx.opp is None:
        return None
    p_ko = ctx.damage(ctx.me, move).ko_probability(ctx.opp_hp)
    if p_ko <= 0:
        return None
    first = ctx.outspeeds(ctx.me, move.priority)
    value = ctx.weights.ko * p_ko * ctx.act_probability(move)
    reason = f"{p_ko:.0%} to KO"
    if first:
        value += ctx.weights.ko_first * p_ko
        reason += " before they move" + (" (priority)" if move.priority > 0 else "")
    return ScoreComponent("ko", value, reason)


def score_damage_taken(ctx: BattleContext, move: Move) -> ScoreComponent | None:
    """Cost of the hit we expect to take this turn by staying in."""
    if ctx.me is None or ctx.opp is None:
        return None
    threat = ctx.threat(ctx.me)
    taken = min(threat.damage.expected, ctx.my_hp)
    if taken <= 0:
        return None
    reason = f"{ctx.opp.species} may hit with {threat.move_id} for ~{threat.damage.expected:.0%}"
    if not threat.revealed:
        reason += " (unrevealed)"
    if ctx.outspeeds(ctx.me, move.priority):
        p_ko = ctx.damage(ctx.me, move).ko_probability(ctx.opp_hp)
        if p_ko > 0:
            taken *= 1 - p_ko
            reason += f", avoided {p_ko:.0%} of the time by KOing first"
    return ScoreComponent("damage_taken", -ctx.weights.damage_taken * taken, reason)


def score_matchup(ctx: BattleContext, move: Move) -> ScoreComponent | None:
    """Next turn's expected exchange if we keep this Pokémon in."""
    if ctx.me is None or ctx.opp is None:
        return None
    p_ko = ctx.damage(ctx.me, move).ko_probability(ctx.opp_hp)
    exchange = ctx.exchange(ctx.me)
    value = ctx.weights.continuation * exchange * (1 - p_ko)
    return ScoreComponent("matchup", value, f"staying in trades {exchange:+.0%} per turn")


def score_status(ctx: BattleContext, move: Move) -> ScoreComponent | None:
    """Value of inflicting sleep, paralysis, burn or poison."""
    if (
        move.category != MoveCategory.STATUS
        or move.status is None
        or move.target == Target.SELF
        or ctx.me is None
        or ctx.opp is None
    ):
        return None
    blocked = status_block_reason(ctx.battle, move, ctx.opp)
    if blocked is not None:
        return ScoreComponent("status", 0.0, f"would fail: {blocked}")
    w = ctx.weights
    value = w.status_value.get(move.status, 0.0)
    reason = f"inflicts {move.status.name}"
    if move.status.name == "PAR" and not ctx.outspeeds(ctx.me):
        value += w.par_speed_bonus
        reason += " on a faster target"
    if move.status.name == "BRN" and ctx.opp.base_stats["atk"] > ctx.opp.base_stats["spa"]:
        value += w.brn_physical_bonus
        reason += " on a physical attacker"
    # A status on something nearly fainted is worth little.
    value *= float(move.accuracy) * _clamp(ctx.opp_hp * 1.5) * ctx.act_probability(move)
    return ScoreComponent("status", value, f"{reason}, {float(move.accuracy):.0%} accurate")


def score_setup(ctx: BattleContext, move: Move) -> ScoreComponent | None:
    """Value of boosting our own stats, when it is safe enough to spend the turn."""
    if (
        move.category != MoveCategory.STATUS
        or move.target != Target.SELF
        or not move.boosts
        or ctx.me is None
    ):
        return None
    me = ctx.me
    categories = {m.category for m in me.moves.values()}
    stages = 0.0
    for stat, amount in move.boosts.items():
        if amount < 0:
            stages += 0.25 * amount  # e.g. Shell Smash's defence drops
            continue
        usable = min(amount, 6 - me.boosts.get(stat, 0))
        if stat == "atk":
            weight = 1.0 if MoveCategory.PHYSICAL in categories else 0.0
        elif stat == "spa":
            weight = 1.0 if MoveCategory.SPECIAL in categories else 0.0
        elif stat == "spe":
            weight = 0.25 if ctx.outspeeds(me) else 1.0
        elif stat in ("def", "spd"):
            weight = 0.5
        else:
            weight = 0.0
        if me.boosts.get(stat, 0) >= 2:
            weight *= 0.5  # diminishing returns: start attacking
        stages += usable * weight
    if stages <= 0:
        return ScoreComponent("setup", 0.0, "no useful boost left to gain")
    w = ctx.weights
    threat = ctx.threat(me).damage.expected
    safety = _clamp(1 - threat / max(w.setup_danger * ctx.my_hp, 1e-6))
    value = w.setup_per_stage * stages * safety * ctx.act_probability(move)
    reason = f"{stages:.1f} useful boost stages, safety {safety:.0%} vs ~{threat:.0%} hits"
    return ScoreComponent("setup", value, reason)


def score_recovery(ctx: BattleContext, move: Move) -> ScoreComponent | None:
    """Value of healing, discounted if the opponent out-damages it."""
    if move.category != MoveCategory.STATUS or move.target != Target.SELF or ctx.me is None:
        return None
    heal = move.heal or EXTRA_RECOVERY_FRACTIONS.get(move.id, 0.0)
    if heal <= 0:
        return None
    gain = min(heal, 1 - ctx.my_hp)
    if gain < 0.05:
        return ScoreComponent("recovery", 0.0, "HP is already nearly full")
    w = ctx.weights
    value = w.recovery * gain
    reason = f"restores ~{gain:.0%} HP"
    threat = ctx.threat(ctx.me).damage.expected
    if threat >= gain:
        value *= w.recovery_outdamaged_scale
        reason += f", but they deal ~{threat:.0%} per turn"
    if move.id == "rest":
        value *= w.rest_scale
        reason += ", asleep for two turns"
    return ScoreComponent("recovery", value * ctx.act_probability(move), reason)


def score_hazard(ctx: BattleContext, move: Move) -> ScoreComponent | None:
    """Value of setting entry hazards, or clearing them from our side."""
    if ctx.me is None:
        return None
    w = ctx.weights
    condition = move.side_condition
    if condition in MAX_LAYERS and move.target == Target.FOE_SIDE:
        side = ctx.battle.opponent_side_conditions
        if condition in (SideCondition.SPIKES, SideCondition.TOXIC_SPIKES):
            layers = side.get(condition, 0)
        else:
            layers = 1 if condition in side else 0
        if layers >= MAX_LAYERS[condition]:
            return ScoreComponent("hazard", 0.0, "already set")
        targets = ctx.opp_remaining - 1
        if targets <= 0:
            return ScoreComponent("hazard", 0.0, "nothing left to switch into it")
        value = w.hazard_per_target[condition] * targets * 0.5**layers
        reason = f"hits up to {targets} future switch-ins"
        return ScoreComponent("hazard", value * ctx.act_probability(move), reason)
    if move.id in HAZARD_REMOVAL_MOVES:
        saved = sum(hazard_entry_fraction(ctx.battle, mon) for mon in ctx.my_bench)
        if saved <= 0:
            return None
        value = w.hazard_removal * saved * ctx.act_probability(move)
        return ScoreComponent("hazard", value, f"clears hazards costing our bench {saved:.0%}")
    return None


def score_self_cost(ctx: BattleContext, move: Move) -> ScoreComponent | None:
    """Recoil, HP drain and self-KO moves."""
    if ctx.me is None or ctx.opp is None:
        return None
    if move.self_destruct:
        value = -ctx.weights.self_ko * ctx.my_hp
        return ScoreComponent("self_cost", value, "the user faints")
    if not (move.recoil or move.drain):
        return None
    estimate = ctx.damage(ctx.me, move)
    dealt = min(estimate.mean_fraction, ctx.opp_hp) * estimate.accuracy
    # Recoil/drain are fractions of damage dealt in HP points; convert to our HP bar.
    ratio = max_hp_points(ctx.battle, ctx.opp) / max(max_hp_points(ctx.battle, ctx.me), 1)
    if move.recoil:
        cost = min(move.recoil * dealt * ratio, ctx.my_hp)
        return ScoreComponent("self_cost", -cost, f"{move.recoil:.0%} recoil")
    gain = min(move.drain * dealt * ratio, 1 - ctx.my_hp)
    if gain <= 0:
        return None
    return ScoreComponent("self_cost", gain, f"drains {move.drain:.0%} of damage dealt")


# --------------------------------------------------------------------------
# Switch scorers
# --------------------------------------------------------------------------


def entry_damage(ctx: BattleContext, mon: Pokemon) -> tuple[float, float]:
    """(hazard damage, attack damage) ``mon`` takes while switching in."""
    hp = mon.current_hp_fraction
    hazards = min(hazard_entry_fraction(ctx.battle, mon), hp)
    if ctx.free_switch:
        return hazards, 0.0
    hit = min(ctx.threat(mon).damage.expected, hp - hazards)
    return hazards, hit


def score_switch_entry(ctx: BattleContext, mon: Pokemon) -> ScoreComponent | None:
    """Cost of coming in: entry hazards plus the attack aimed at the outgoing Pokémon."""
    hazards, hit = entry_damage(ctx, mon)
    total = hazards + hit
    if total <= 0:
        return None
    value = -ctx.weights.damage_taken * total
    parts = []
    if hazards > 0:
        parts.append(f"hazards {hazards:.0%}")
    if hit > 0:
        parts.append(f"takes {ctx.threat(mon).move_id} for ~{hit:.0%}")
    if total >= mon.current_hp_fraction:
        value -= ctx.weights.entry_faint_penalty
        parts.append("would faint on entry")
    return ScoreComponent("entry", value, ", ".join(parts))


def score_switch_matchup(ctx: BattleContext, mon: Pokemon) -> ScoreComponent | None:
    """Expected exchange once the new Pokémon is in, including who moves first."""
    if ctx.opp is None:
        value = 0.1 * mon.current_hp_fraction
        return ScoreComponent("matchup", value, "no opposing Pokémon in play; prefer healthy")
    hazards, hit = entry_damage(ctx, mon)
    exchange = ctx.exchange(mon, hp=mon.current_hp_fraction - hazards - hit)
    move, _ = ctx.best_attack(mon)
    speed = "outspeeds" if ctx.outspeeds(mon, move.priority if move else 0) else "is outsped"
    reason = f"trades {exchange:+.0%} per turn with {move.id if move else 'nothing'}, {speed}"
    return ScoreComponent("matchup", ctx.weights.continuation * exchange, reason)


def score_switch_tempo(ctx: BattleContext, mon: Pokemon) -> ScoreComponent | None:
    """Cost of giving up this turn's attack, our boosts, and switching repeatedly."""
    if ctx.free_switch or ctx.me is None:
        return None
    w = ctx.weights
    value = -w.switch_tempo
    parts = ["spends the turn"]
    if ctx.me.first_turn:
        value -= w.switch_again_penalty
        parts.append("just switched in")
    boosts = sum(v for v in ctx.me.boosts.values() if v > 0)
    if boosts:
        value -= w.boost_loss * boosts
        parts.append(f"abandons +{boosts} boosts")
    return ScoreComponent("tempo", value, ", ".join(parts))


DEFAULT_MOVE_SCORERS: tuple[MoveScorer, ...] = (
    score_damage,
    score_ko,
    score_status,
    score_setup,
    score_recovery,
    score_hazard,
    score_self_cost,
    score_damage_taken,
    score_matchup,
)
DEFAULT_SWITCH_SCORERS: tuple[SwitchScorer, ...] = (
    score_switch_entry,
    score_switch_matchup,
    score_switch_tempo,
)
