"""Expected damage, as a fraction of the defender's maximum HP.

The numbers come from poke-env's port of the Smogon damage calculator, so
type effectiveness, STAB, items, abilities, weather, screens, burn and stat
boosts are handled by a maintained implementation rather than by us. This
module only adapts it: it fills in what we cannot see about the opponent and
converts HP points into fractions so scores are comparable across Pokémon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from poke_env.battle import AbstractBattle, Move, MoveCategory, Pokemon
from poke_env.calc.damage_calc_gen9 import calculate_damage

from poke_env.data import GenData

from psbot.heuristics.randbats import GEN, certain_ability, certain_item, estimate_stats

logger = logging.getLogger(__name__)

UNKNOWN_ITEM: str = GenData.from_gen(GEN).UNKNOWN_ITEM

_warned_moves: set[str] = set()


@dataclass(frozen=True)
class DamageEstimate:
    """Damage range of one move use, in fractions of the defender's max HP."""

    min_fraction: float = 0.0
    max_fraction: float = 0.0
    accuracy: float = 1.0

    @property
    def mean_fraction(self) -> float:
        return (self.min_fraction + self.max_fraction) / 2

    @property
    def expected(self) -> float:
        """Average damage including the chance to miss."""
        return self.mean_fraction * self.accuracy

    def ko_probability(self, hp_fraction: float) -> float:
        """Chance this move knocks out a defender with ``hp_fraction`` HP left.

        Damage rolls are treated as uniform between min and max.
        """
        if hp_fraction <= 0 or self.max_fraction <= 0:
            return 0.0
        if self.min_fraction >= hp_fraction:
            roll = 1.0
        elif self.max_fraction < hp_fraction:
            roll = 0.0
        else:
            roll = (self.max_fraction - hp_fraction) / (self.max_fraction - self.min_fraction)
        return roll * self.accuracy

    def scaled(self, weight: float) -> "DamageEstimate":
        """Discount for uncertainty, e.g. a move the opponent might not have."""
        return DamageEstimate(self.min_fraction * weight, self.max_fraction * weight, self.accuracy)


NO_DAMAGE = DamageEstimate()


def is_own(battle: AbstractBattle, mon: Pokemon) -> bool:
    return any(mon is m for m in battle.team.values())


def fill_unknowns(battle: AbstractBattle, mon: Pokemon) -> None:
    """Give an opponent Pokémon our best estimate of its hidden attributes.

    Our own Pokémon already carry exact stats from the server. Opponents get
    the random-battle stat spread, their ability when the species only has one,
    and their item when the forme requires it. poke-env overwrites the ability
    and item if the real ones are ever revealed.
    """
    if is_own(battle, mon):
        return
    mon.stats = estimate_stats(mon)
    if mon.ability is None:
        ability = certain_ability(mon)
        if ability is not None:
            mon.ability = ability
    if mon.item == UNKNOWN_ITEM:
        item = certain_item(mon)
        if item is not None:
            mon.item = item


def max_hp_points(battle: AbstractBattle, mon: Pokemon) -> int:
    """Max HP in real points. Opponent HP is reported as a percentage, so we
    use the estimated stat instead of ``mon.max_hp`` (which would be 100)."""
    if is_own(battle, mon) and mon.max_hp:
        return mon.max_hp
    return estimate_stats(mon)["hp"]


def estimate_damage(
    battle: AbstractBattle, attacker: Pokemon, defender: Pokemon, move: Move
) -> DamageEstimate:
    """Estimate what ``move`` does when ``attacker`` uses it on ``defender``."""
    if move.category == MoveCategory.STATUS or move.id == "recharge":
        return NO_DAMAGE  # "recharge" is the forced no-op turn after Hyper Beam etc.
    fill_unknowns(battle, attacker)
    fill_unknowns(battle, defender)
    attacker_role = battle.player_role if is_own(battle, attacker) else battle.opponent_role
    defender_role = battle.player_role if is_own(battle, defender) else battle.opponent_role
    if attacker_role is None or defender_role is None:
        return NO_DAMAGE
    try:
        low, high = calculate_damage(
            attacker.identifier(attacker_role), defender.identifier(defender_role), move, battle
        )
    except Exception as exc:  # the calculator does not cover every move
        if move.id not in _warned_moves:
            _warned_moves.add(move.id)
            logger.warning("damage calc failed for %s (%r); treating as 0", move.id, exc)
        return NO_DAMAGE
    hp = max_hp_points(battle, defender)
    if hp <= 0:
        return NO_DAMAGE
    return DamageEstimate(low / hp, high / hp, float(move.accuracy))
