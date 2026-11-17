"""Shared analysis of one decision point.

``BattleContext`` answers the questions every scorer needs — how hard do we
hit, how hard do they hit, who moves first — and caches the answers, so each
scorer stays a few lines of judgement instead of repeating calculations.

All scores share one unit: **fractions of a Pokémon's HP bar**. Dealing 40%
is worth +0.40, expecting to take 25% is worth -0.25. Weights rescale those
terms; they are tuning knobs, not game mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property

from poke_env.battle import AbstractBattle, Move, MoveCategory, Pokemon, SideCondition, Status

from psbot.heuristics.damage import NO_DAMAGE, DamageEstimate, MixtureEstimate, estimate_damage
from psbot.heuristics.mechanics import moves_first
from psbot.heuristics.randbats import _move, possible_moves

# Moves that only work on the user's first turn on the field.
FIRST_TURN_ONLY_MOVES = frozenset({"fakeout", "firstimpression"})


@dataclass(frozen=True)
class HeuristicWeights:
    """Tunable judgement calls. Every value is in HP-bar units or a 0-1 scale."""

    damage: float = 1.0  # value of dealing one HP bar of damage
    ko: float = 0.4  # extra value of securing a knockout
    ko_first: float = 0.3  # ...and more if we strike before they can act
    damage_taken: float = 1.0  # cost of taking one HP bar of damage
    continuation: float = 1.0  # weight of next turn's expected exchange

    unrevealed_move_weight: float = 0.7  # trust in moves the opponent might not have
    incapacitated_threat_scale: float = 0.35  # sleeping/frozen opponents rarely act

    setup_per_stage: float = 0.15  # value of one useful boost stage
    setup_danger: float = 0.5  # setup is worthless if they deal this share of our HP
    recovery: float = 1.0  # value of healing one HP bar
    recovery_outdamaged_scale: float = 0.25  # ...when they out-damage the heal
    rest_scale: float = 0.6  # Rest heals fully but costs two turns asleep

    status_value: dict[Status, float] = field(
        default_factory=lambda: {
            Status.SLP: 0.6,
            Status.TOX: 0.35,
            Status.PAR: 0.3,
            Status.BRN: 0.3,
            Status.PSN: 0.2,
        }
    )
    par_speed_bonus: float = 0.15  # paralysing something faster than us
    brn_physical_bonus: float = 0.2  # burning a physical attacker

    hazard_per_target: dict[SideCondition, float] = field(
        default_factory=lambda: {
            SideCondition.STEALTH_ROCK: 0.09,
            SideCondition.SPIKES: 0.05,
            SideCondition.TOXIC_SPIKES: 0.04,
            SideCondition.STICKY_WEB: 0.05,
        }
    )
    hazard_removal: float = 0.5  # share of our bench's future hazard damage we count
    self_ko: float = 0.6  # cost of a move that faints the user, times its HP

    switch_tempo: float = 0.15  # flat cost of spending a turn switching
    switch_again_penalty: float = 0.3  # extra cost of switching right after entering
    boost_loss: float = 0.1  # cost per positive boost stage abandoned
    entry_faint_penalty: float = 1.0  # switching into something that faints on entry

    # ---- Phase 5 additions. All off by default, so the default weights are
    # ---- still exactly the baseline engine the trained hybrids were built on.
    # Weigh the opponent's unrevealed moves by how likely each is (measured
    # set statistics) instead of a flat discount on the single worst one.
    move_model: bool = False
    # Weight of the forward projection (psbot.heuristics.lookahead) while the
    # opponent has unseen Pokémon: HP bars an action is projected to cost us
    # against their active Pokémon, relative to our best option.
    sweep: float = 0.0
    # Weight of the same projection once all their remaining Pokémon are known
    # and few are left, when the rest of the battle is searched by minimax.
    endgame: float = 0.0
    # Expected damage also accounts for (a) an opposing ability we have not
    # seen that would absorb or ignore the move, weighted by how likely the
    # species is to have it, and (b) accuracy and evasion stages.
    immunity_aware: bool = False
    # Assumed chance the opponent switches this turn. An attack that one of
    # their revealed, healthy bench Pokémon is immune to loses that share of
    # its damage value. 0 disables the check.
    switch_in_risk: float = 0.0


# The Phase 5 engine: the baseline plus the move model, the forward projection
# and the damage refinements. The weights are first guesses, not tuned values:
# see reports/phase5-engine.md for what was measured.
V2_OVERRIDES: dict[str, float | bool] = {
    "move_model": True,
    "sweep": 1.0,
    "endgame": 1.0,
    "immunity_aware": True,
    "switch_in_risk": 0.15,
}


def v2_weights() -> HeuristicWeights:
    return replace(HeuristicWeights(), **V2_OVERRIDES)


@dataclass(frozen=True)
class Threat:
    """The opponent's most damaging option against one of our Pokémon."""

    move_id: str | None = None
    damage: DamageEstimate = NO_DAMAGE
    revealed: bool = False
    probability: float = 1.0  # that they have ``move_id`` at all (1 if revealed)


class BattleContext:
    """Cached analysis of the current turn from our point of view."""

    def __init__(self, battle: AbstractBattle, weights: HeuristicWeights | None = None) -> None:
        self.battle = battle
        self.weights = weights or HeuristicWeights()
        self.me: Pokemon | None = battle.active_pokemon
        self.opp: Pokemon | None = battle.opponent_active_pokemon
        if self.opp is not None and self.opp.fainted:
            self.opp = None
        self._damage_cache: dict[tuple[int, str], DamageEstimate] = {}
        self._threat_cache: dict[int, Threat] = {}
        self._belief_cache: dict[int, list[tuple[Move, float, bool]]] = {}

    # --- basic state -----------------------------------------------------

    @property
    def my_hp(self) -> float:
        return self.me.current_hp_fraction if self.me is not None else 0.0

    @property
    def opp_hp(self) -> float:
        return self.opp.current_hp_fraction if self.opp is not None else 0.0

    @cached_property
    def free_switch(self) -> bool:
        """True when switching costs no turn (our active Pokémon fainted)."""
        return bool(self.battle.force_switch) or self.me is None or self.me.fainted

    @cached_property
    def my_bench(self) -> list[Pokemon]:
        return [m for m in self.battle.team.values() if not m.fainted and not m.active]

    @cached_property
    def opp_remaining(self) -> int:
        """Opposing Pokémon not yet fainted, counting unrevealed ones."""
        team_size = max(len(self.battle.team), len(self.battle.opponent_team))
        fainted = sum(1 for m in self.battle.opponent_team.values() if m.fainted)
        return team_size - fainted

    # --- offence ---------------------------------------------------------

    def damage(self, attacker: Pokemon, move: Move) -> DamageEstimate:
        """What one of our Pokémon's moves does to the opposing active Pokémon."""
        if self.opp is None:
            return NO_DAMAGE
        key = (id(attacker), move.id)
        if key not in self._damage_cache:
            if (
                move.id in FIRST_TURN_ONLY_MOVES
                and attacker is self.me
                and not attacker.first_turn
            ):
                estimate = NO_DAMAGE
            else:
                estimate = estimate_damage(self.battle, attacker, self.opp, move)
                if self.weights.immunity_aware and estimate.max_fraction > 0:
                    estimate = self._hit_chance_adjusted(attacker, move, estimate)
            self._damage_cache[key] = estimate
        return self._damage_cache[key]

    def _hit_chance_adjusted(
        self, attacker: Pokemon, move: Move, estimate: DamageEstimate
    ) -> DamageEstimate:
        """Fold into the estimate's accuracy the chances that the move does
        nothing for reasons the calculator cannot see.

        * Unknown ability: when the target's ability has not been shown and
          its species has several, ask the calculator again under each one
          and discount by the probability of those that nullify the move.
          The guess is used for this arithmetic only and removed again.
        * Accuracy/evasion stages, for moves that can miss.
        """
        assert self.opp is not None
        opp, chance = self.opp, estimate.accuracy
        if opp.ability is None:
            from psbot.opponent_model import belief_for

            abilities = belief_for(opp).ability_probabilities()
            if len(abilities) > 1:
                blocked = 0.0
                for ability, probability in abilities.items():
                    opp._ability = ability
                    try:
                        if estimate_damage(self.battle, attacker, opp, move).max_fraction <= 0:
                            blocked += probability
                    finally:
                        opp._ability = None
                chance *= 1 - min(blocked, 1.0)
        if move.entry.get("accuracy") is not True:
            stage = attacker.boosts.get("accuracy", 0) - opp.boosts.get("evasion", 0)
            stage = max(-6, min(6, stage))
            chance *= (3 + stage) / 3 if stage >= 0 else 3 / (3 - stage)
        chance = max(0.0, min(1.0, chance))
        return estimate if chance == estimate.accuracy else replace(estimate, accuracy=chance)

    def usable_moves(self, mon: Pokemon) -> list[Move]:
        """Moves ``mon`` could pick: the legal ones if active, else all it knows."""
        if mon is self.me:
            return list(self.battle.available_moves)
        return list(mon.moves.values())

    def best_attack(self, mon: Pokemon) -> tuple[Move | None, DamageEstimate]:
        """The move of ``mon`` with the highest expected damage on the opponent."""
        best: tuple[Move | None, DamageEstimate] = (None, NO_DAMAGE)
        for move in self.usable_moves(mon):
            estimate = self.damage(mon, move)
            if estimate.expected > best[1].expected:
                best = (move, estimate)
        return best

    # --- defence ---------------------------------------------------------

    def threat(self, mon: Pokemon) -> Threat:
        """The worst the opposing active Pokémon is likely to do to ``mon``.

        Revealed moves count in full. While fewer than four are revealed, the
        rest of the species' random-battle move pool is considered too, at a
        discount because the opponent only has some of them.
        """
        if self.opp is None:
            return Threat()
        key = id(mon)
        if key not in self._threat_cache:
            self._threat_cache[key] = self._compute_threat(mon)
        return self._threat_cache[key]

    def _compute_threat(self, mon: Pokemon) -> Threat:
        assert self.opp is not None
        if self.opp.must_recharge:
            return Threat()
        if self.weights.move_model:
            return self._modelled_threat(self.opp, mon)
        revealed = dict(self.opp.moves)
        candidates: list[tuple[Move, bool]] = [(m, True) for m in revealed.values()]
        if len(revealed) < 4:
            candidates += [(m, False) for m in possible_moves(self.opp) if m.id not in revealed]

        scale = 1.0
        if self.opp.status in (Status.SLP, Status.FRZ):
            scale = self.weights.incapacitated_threat_scale

        best = Threat()
        for move, is_revealed in candidates:
            if move.category == MoveCategory.STATUS:
                continue
            estimate = estimate_damage(self.battle, self.opp, mon, move)
            weight = scale * (1.0 if is_revealed else self.weights.unrevealed_move_weight)
            estimate = estimate.scaled(weight)
            if estimate.expected > best.damage.expected:
                best = Threat(move.id, estimate, is_revealed)
        return best

    def move_beliefs(self, attacker: Pokemon) -> list[tuple[Move, float, bool]]:
        """``(move, probability, revealed)`` for every move an opposing Pokémon
        may have. Revealed moves are certain; the rest are inferences."""
        from psbot.opponent_model import belief_for  # avoids an import cycle

        key = id(attacker)
        if key not in self._belief_cache:
            beliefs = [(move, 1.0, True) for move in attacker.moves.values()]
            if len(attacker.moves) < 4:
                likely = belief_for(attacker).move_probabilities()
                if likely:
                    beliefs += [(_move(m), p, False) for m, p in likely.items()]
                else:  # species missing from the tables: fall back to its pool
                    flat = self.weights.unrevealed_move_weight
                    beliefs += [
                        (m, flat, False)
                        for m in possible_moves(attacker)
                        if m.id not in attacker.moves
                    ]
            self._belief_cache[key] = beliefs
        return self._belief_cache[key]

    def _modelled_threat(self, attacker: Pokemon, mon: Pokemon) -> Threat:
        """The hit ``mon`` should expect from ``attacker``, over the moves it
        may have.

        Assume it uses its most damaging move. Sort the candidates by damage:
        the i-th is the one we take if it has that move and none of the
        stronger ones, so its weight is p_i * prod(1 - p_j for stronger j).
        Treating moves as independent is an approximation; a revealed move
        (p = 1) correctly cuts off everything weaker.
        """
        scored = []
        for move, probability, revealed in self.move_beliefs(attacker):
            if move.category == MoveCategory.STATUS or probability <= 0:
                continue
            estimate = estimate_damage(self.battle, attacker, mon, move)
            if estimate.expected > 0:
                scored.append((estimate, probability, move, revealed))
        if not scored:
            return Threat()
        scored.sort(key=lambda item: item[0].expected, reverse=True)
        parts, none_stronger = [], 1.0
        for estimate, probability, _, _ in scored:
            parts.append((none_stronger * probability, estimate))
            none_stronger *= 1 - probability
            if none_stronger < 1e-3:
                break
        top_estimate, top_probability, top_move, top_revealed = scored[0]
        mixture = MixtureEstimate(
            top_estimate.min_fraction, top_estimate.max_fraction, top_estimate.accuracy, tuple(parts)
        )
        if attacker.status in (Status.SLP, Status.FRZ):
            mixture = mixture.scaled(self.weights.incapacitated_threat_scale)
        return Threat(top_move.id, mixture, top_revealed, top_probability)

    @cached_property
    def projection(self) -> "Projection | None":  # noqa: F821
        """Forward projection of every legal action, if any weight asks for it."""
        if self.weights.sweep <= 0 and self.weights.endgame <= 0:
            return None
        from psbot.heuristics.lookahead import Lookahead

        return Lookahead(self).project()

    # --- turn order ------------------------------------------------------

    def outspeeds(self, mon: Pokemon, priority: int = 0) -> bool:
        """Whether ``mon`` would act before the opposing active Pokémon.

        Assumes the opponent uses a priority-0 move.
        """
        if self.opp is None:
            return True
        return moves_first(self.battle, mon, self.opp, priority=priority)

    def act_probability(self, move: Move) -> float:
        """Chance our active Pokémon actually gets ``move`` off this turn.

        Moving second means we might be knocked out first. Paralysis fails 25%
        of the time and a frozen Pokémon thaws 20% of the time; a sleeping one
        only acts with Sleep Talk-style moves or if it wakes (1-3 turns).
        """
        if self.me is None:
            return 0.0
        chance = 1.0
        if self.me.status == Status.PAR:
            chance *= 0.75
        elif self.me.status == Status.FRZ:
            chance *= 0.2
        elif self.me.status == Status.SLP and not move.sleep_usable:
            chance *= 1 / 3
        if not self.outspeeds(self.me, move.priority):
            chance *= 1 - self.threat(self.me).damage.ko_probability(self.my_hp)
        return chance

    # --- matchup ---------------------------------------------------------

    def exchange(self, mon: Pokemon, hp: float | None = None) -> float:
        """Expected HP-bar swing of one turn of ``mon`` trading blows with the
        opposing active Pokémon: damage dealt minus damage taken.

        Accounts for turn order: a slower Pokémon that gets knocked out deals
        nothing, and a faster one that knocks out takes nothing.
        """
        if self.opp is None:
            return 0.0
        hp = mon.current_hp_fraction if hp is None else hp
        if hp <= 0:
            return -1.0
        move, attack = self.best_attack(mon)
        threat = self.threat(mon).damage
        first = self.outspeeds(mon, move.priority if move is not None else 0)
        dealt = min(attack.mean_fraction, self.opp_hp) * attack.accuracy
        taken = min(threat.expected, hp)
        if first:
            taken *= 1 - attack.ko_probability(self.opp_hp)
        else:
            dealt *= 1 - threat.ko_probability(hp)
        return dealt - taken
