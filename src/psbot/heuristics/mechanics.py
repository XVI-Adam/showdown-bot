"""Battle mechanics the damage calculator does not cover.

Each rule here is a documented Gen 9 mechanic. Where a rule depends on
something we cannot see (the opponent's item or ability), we assume the
common case and say so.
"""

from __future__ import annotations

from poke_env.battle import (
    AbstractBattle,
    Effect,
    Field,
    Move,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
)

from psbot.heuristics.damage import fill_unknowns, is_own

# Moves that clear entry hazards from the user's side.
HAZARD_REMOVAL_MOVES = frozenset({"rapidspin", "defog", "mortalspin", "tidyup"})

# Fraction of max HP lost on switch-in, by number of Spikes layers.
SPIKES_DAMAGE = {1: 1 / 8, 2: 1 / 6, 3: 1 / 4}
MAX_LAYERS = {
    SideCondition.STEALTH_ROCK: 1,
    SideCondition.SPIKES: 3,
    SideCondition.TOXIC_SPIKES: 2,
    SideCondition.STICKY_WEB: 1,
}

# Types that cannot receive a status condition.
STATUS_IMMUNE_TYPES: dict[Status, frozenset[PokemonType]] = {
    Status.BRN: frozenset({PokemonType.FIRE}),
    Status.PAR: frozenset({PokemonType.ELECTRIC}),
    Status.PSN: frozenset({PokemonType.POISON, PokemonType.STEEL}),
    Status.TOX: frozenset({PokemonType.POISON, PokemonType.STEEL}),
    Status.FRZ: frozenset({PokemonType.ICE}),
}
# Abilities that prevent a status condition (only checked when known).
STATUS_IMMUNE_ABILITIES: dict[Status, frozenset[str]] = {
    Status.BRN: frozenset({"waterveil", "waterbubble", "thermalexchange"}),
    Status.PAR: frozenset({"limber"}),
    Status.PSN: frozenset({"immunity", "pastelveil"}),
    Status.TOX: frozenset({"immunity", "pastelveil"}),
    Status.SLP: frozenset({"insomnia", "vitalspirit", "sweetveil"}),
}
# Abilities that stop any status condition, or bounce/block status moves.
BLOCK_ALL_STATUS_ABILITIES = frozenset({"comatose", "purifyingsalt", "goodasgold", "magicbounce"})


def boost_multiplier(stage: int) -> float:
    """Stat multiplier for a stage from -6 to +6: (2+n)/2 up, 2/(2-n) down."""
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def effective_speed(battle: AbstractBattle, mon: Pokemon) -> float:
    """Speed after boosts, paralysis, Choice Scarf and Tailwind.

    An opponent's unrevealed Choice Scarf is not assumed.
    """
    fill_unknowns(battle, mon)
    speed = float(mon.stats["spe"] or 0) * boost_multiplier(mon.boosts["spe"])
    if mon.status == Status.PAR and mon.ability != "quickfeet":
        speed *= 0.5
    if mon.item == "choicescarf":
        speed *= 1.5
    side = battle.side_conditions if is_own(battle, mon) else battle.opponent_side_conditions
    if SideCondition.TAILWIND in side:
        speed *= 2
    return speed


def moves_first(
    battle: AbstractBattle,
    mon: Pokemon,
    other: Pokemon,
    priority: int = 0,
    other_priority: int = 0,
) -> bool:
    """Whether ``mon`` acts before ``other``. Speed ties count as moving second."""
    if priority != other_priority:
        return priority > other_priority
    mine, theirs = effective_speed(battle, mon), effective_speed(battle, other)
    if Field.TRICK_ROOM in battle.fields:
        return mine < theirs
    return mine > theirs


def own_tera_type(battle: AbstractBattle, mon: Pokemon) -> PokemonType | None:
    """The Tera type of one of our Pokémon.

    poke-env 0.16 only fills ``Pokemon.tera_type`` from a teambuilder team or
    after Terastallizing, so in random battles we read it from the server's
    request, where each of our Pokémon carries a ``teraType`` field.
    """
    if mon.tera_type is not None:
        return mon.tera_type
    role = battle.player_role
    request = battle.last_request or {}
    if role is None:
        return None
    ident = mon.identifier(role)
    for entry in request.get("side", {}).get("pokemon", []):
        if entry.get("ident") == ident and entry.get("teraType"):
            return PokemonType.from_name(entry["teraType"])
    return None


def hazard_entry_fraction(battle: AbstractBattle, mon: Pokemon) -> float:
    """Fraction of max HP ``mon`` loses to entry hazards when it switches in."""
    side = battle.side_conditions if is_own(battle, mon) else battle.opponent_side_conditions
    if mon.item == "heavydutyboots" or mon.ability == "magicguard":
        return 0.0
    damage = 0.0
    if SideCondition.STEALTH_ROCK in side:
        damage += mon.damage_multiplier(PokemonType.ROCK) / 8
    layers = side.get(SideCondition.SPIKES, 0)
    if layers and battle.is_grounded(mon):
        damage += SPIKES_DAMAGE[min(layers, 3)]
    return damage


def status_block_reason(battle: AbstractBattle, move: Move, target: Pokemon) -> str | None:
    """Why a status-inflicting move would fail on ``target``, or None if it lands."""
    status = move.status
    if status is None:
        return "move inflicts no status"
    if target.status is not None:
        return f"target already has {target.status.name}"
    if Effect.SUBSTITUTE in target.effects and "bypasssub" not in move.flags:
        return "target is behind a Substitute"
    target_side = (
        battle.side_conditions if is_own(battle, target) else battle.opponent_side_conditions
    )
    if SideCondition.SAFEGUARD in target_side:
        return "target is protected by Safeguard"
    types = {t for t in target.types if t is not None}
    immune = STATUS_IMMUNE_TYPES.get(status, frozenset()) & types
    if immune:
        return f"{next(iter(immune)).name.title()} types cannot be {status.name}"
    if "powder" in move.flags and PokemonType.GRASS in types:
        return "Grass types are immune to powder moves"
    if move.id == "thunderwave" and target.damage_multiplier(move) == 0:
        return "Thunder Wave does not affect this type"
    if target.ability in BLOCK_ALL_STATUS_ABILITIES:
        return f"blocked by {target.ability}"
    if target.ability in STATUS_IMMUNE_ABILITIES.get(status, frozenset()):
        return f"{target.ability} prevents {status.name}"
    if battle.is_grounded(target):
        if Field.MISTY_TERRAIN in battle.fields:
            return "Misty Terrain prevents status"
        if status == Status.SLP and Field.ELECTRIC_TERRAIN in battle.fields:
            return "Electric Terrain prevents sleep"
    if status == Status.SLP:
        # Sleep Clause Mod is part of the gen9randombattle ruleset.
        team = battle.team if is_own(battle, target) else battle.opponent_team
        if any(m.status == Status.SLP and not m.fainted for m in team.values()):
            return "Sleep Clause: another Pokémon is already asleep"
    return None
