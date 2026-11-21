"""Looking past this turn: who beats whom if the fight simply continues.

The scorers judge one turn. That misses the two situations the evaluation
report blamed for most losses: an opposing Pokémon that will knock out our
team one after another (a sweep), and endgames where the order we use our last
Pokémon in decides the battle.

Both are answered here by playing the battle forward in a simplified form:

* each side's active Pokémon uses its most damaging move every turn;
* damage is the expected value (accuracy and average roll included); for the
  opponent it is weighted over the moves it may have (the move model);
* the faster Pokémon strikes first; burn and poison tick each turn;
* nobody switches out, except as the first action we are evaluating;
* a Pokémon coming in takes entry hazards.

Two modes, chosen by what we know:

**Sweep check** (``sweep_value``): the opponent still has Pokémon we have not
seen, so we can only reason about the one in front of us. We send our Pokémon
at it in turn, starting with the action being scored, and count the HP it
costs us before it goes down, or everything if it never does.

**Endgame** (``endgame_value``): every remaining opposing Pokémon has been
seen and few are left. Then the whole rest of the battle is played out, with
both sides choosing their replacements by minimax: we pick the replacement
that is best for us, they pick the one that is worst for us.

Deliberate simplifications: no setup, recovery, Protect or status moves in the
projection, no mid-projection switching, and an unrevealed Choice Scarf is not
assumed. The projection ranks our options; it does not predict the battle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from poke_env.battle import Move, MoveCategory, Pokemon, Status

from psbot.heuristics.mechanics import hazard_entry_fraction, moves_first

RESIDUAL = {Status.BRN: 1 / 16, Status.PSN: 1 / 8, Status.TOX: 1 / 8}
MAX_DUEL_TURNS = 12
# Endgame search runs only when both sides together have at most this many
# Pokémon left (the search branches on every replacement choice).
ENDGAME_MAX_TOTAL = 6
WIN_VALUE = 2.0  # HP bars a projected win is worth beyond the HP left over
SWEPT_PENALTY = 1.0  # extra cost, in HP bars, when the sweep check never stops them

Action = tuple[str, Hashable]  # ("move", move id) or ("switch", id(pokemon))


@dataclass(frozen=True)
class Projection:
    """Projected values of each legal action, higher is better, in HP bars."""

    mode: str  # "sweep" or "endgame"
    values: dict[Action, float]
    notes: dict[Action, str]

    @property
    def best(self) -> float:
        return max(self.values.values()) if self.values else 0.0


class Lookahead:
    """Forward projection from one decision point. Built once per turn."""

    def __init__(self, ctx: "BattleContext") -> None:  # noqa: F821 (import cycle)
        self.ctx = ctx
        self.battle = ctx.battle
        self.ours = [m for m in self.battle.team.values() if not m.fainted]
        self.theirs = [m for m in self.battle.opponent_team.values() if not m.fainted]
        self._our_hit: dict[tuple[int, int], tuple[float, int]] = {}
        self._their_hit: dict[tuple[int, int], float] = {}
        self._first: dict[tuple[int, int], bool] = {}

    # ---------------------------------------------------------- pair facts

    @property
    def all_seen(self) -> bool:
        """True when every opposing Pokémon still standing has been revealed."""
        return self.ctx.opp_remaining == len(self.theirs)

    def our_hit(self, mine: Pokemon, theirs: Pokemon) -> tuple[float, int]:
        """(expected damage per turn as a fraction of their max HP, priority)
        of our best move."""
        key = (id(mine), id(theirs))
        if key not in self._our_hit:
            from psbot.heuristics.damage import estimate_damage

            best = (0.0, 0)
            for move in self.ctx.usable_moves(mine):
                if move.category == MoveCategory.STATUS:
                    continue
                expected = estimate_damage(self.battle, mine, theirs, move).expected
                if expected > best[0]:
                    best = (expected, move.priority)
            self._our_hit[key] = best
        return self._our_hit[key]

    def their_hit(self, theirs: Pokemon, mine: Pokemon) -> float:
        """Expected damage per turn they deal to ``mine``, over the moves they may have."""
        key = (id(theirs), id(mine))
        if key not in self._their_hit:
            self._their_hit[key] = self.ctx._modelled_threat(theirs, mine).damage.expected
        return self._their_hit[key]

    def we_first(self, mine: Pokemon, theirs: Pokemon) -> bool:
        key = (id(mine), id(theirs))
        if key not in self._first:
            priority = self.our_hit(mine, theirs)[1]
            self._first[key] = moves_first(self.battle, mine, theirs, priority=priority)
        return self._first[key]

    def entry_hp(self, mon: Pokemon, hp: float) -> float:
        return hp - min(hazard_entry_fraction(self.battle, mon), hp)

    # ---------------------------------------------------------------- duel

    def duel(
        self,
        mine: Pokemon,
        my_hp: float,
        theirs: Pokemon,
        their_hp: float,
        opening: float | None = None,
    ) -> tuple[float, float]:
        """Fight until one side faints. Returns both HP fractions afterwards
        (0 for the loser). ``opening`` replaces our first turn's damage: the
        damage of the specific move being scored, or 0 for a non-attack."""
        dealt, taken = self.our_hit(mine, theirs)[0], self.their_hit(theirs, mine)
        first = self.we_first(mine, theirs)
        my_tick = RESIDUAL.get(mine.status, 0.0)
        their_tick = RESIDUAL.get(theirs.status, 0.0)
        for turn in range(MAX_DUEL_TURNS):
            hit = opening if (turn == 0 and opening is not None) else dealt
            if first:
                their_hp -= hit
                if their_hp <= 0:
                    return max(my_hp, 0.0), 0.0
                my_hp -= taken
                if my_hp <= 0:
                    return 0.0, their_hp
            else:
                my_hp -= taken
                if my_hp <= 0:
                    return 0.0, their_hp
                their_hp -= hit
                if their_hp <= 0:
                    return max(my_hp, 0.0), 0.0
            my_hp -= my_tick
            their_hp -= their_tick
            if their_hp <= 0:
                return max(my_hp, 0.0), 0.0
            if my_hp <= 0:
                return 0.0, their_hp
            if dealt <= 0 and taken <= 0 and not my_tick and not their_tick:
                break
        return my_hp, their_hp  # neither can finish the other

    # ---------------------------------------------------------- sweep check

    def _sweep_cost(self, first: Pokemon, first_hp: float, opening: float | None) -> tuple[float, bool]:
        """HP bars we lose to their active Pokémon before it is stopped, with
        ``first`` facing it now. Returns (cost, stopped)."""
        foe = self.ctx.opp
        assert foe is not None
        hp = {id(m): m.current_hp_fraction for m in self.ours}
        hp[id(first)] = first_hp
        foe_hp, cost, current = foe.current_hp_fraction, 0.0, first
        while True:
            before = hp[id(current)]
            after, foe_hp = self.duel(current, before, foe, foe_hp, opening)
            opening = None
            cost += before - after
            hp[id(current)] = after
            if foe_hp <= 0 or after > 0:
                return cost, True  # it fainted, or it cannot break through
            standing = [m for m in self.ours if hp[id(m)] > 0]
            if not standing:
                return cost + SWEPT_PENALTY, False

            # Next in: whoever beats it with the most HP left, else whoever
            # hurts it most on the way down.
            def outcome(mon: Pokemon) -> tuple[float, float]:
                mine, theirs = self.duel(mon, self.entry_hp(mon, hp[id(mon)]), foe, foe_hp)
                return (mine, -theirs)

            current = max(standing, key=outcome)
            entered = self.entry_hp(current, hp[id(current)])
            cost += hp[id(current)] - entered
            hp[id(current)] = entered

    # -------------------------------------------------------------- endgame

    def _resolve(
        self,
        mine: Pokemon,
        theirs: Pokemon,
        our_hp: dict[int, float],
        their_hp: dict[int, float],
        opening: float | None = None,
    ) -> float:
        """Value of the rest of the battle from here, by minimax over
        replacement choices. Positive is good for us."""
        a, b = self.duel(mine, our_hp[id(mine)], theirs, their_hp[id(theirs)], opening)
        our_hp = {**our_hp, id(mine): a}
        their_hp = {**their_hp, id(theirs): b}
        ours_left = [m for m in self.ours if our_hp[id(m)] > 0]
        theirs_left = [m for m in self.theirs if their_hp[id(m)] > 0]
        margin = sum(our_hp[id(m)] for m in ours_left) - sum(their_hp[id(m)] for m in theirs_left)
        if not theirs_left:
            return WIN_VALUE + margin
        if not ours_left:
            return -WIN_VALUE + margin
        if a > 0 and b > 0:
            return margin  # stalemate between the two actives
        if b <= 0:  # they choose the replacement that is worst for us
            return min(
                self._resolve(mine, r, our_hp, {**their_hp, id(r): self.entry_hp(r, their_hp[id(r)])})
                for r in theirs_left
            )
        return max(  # we choose the replacement that is best for us
            self._resolve(r, theirs, {**our_hp, id(r): self.entry_hp(r, our_hp[id(r)])}, their_hp)
            for r in ours_left
        )

    # ------------------------------------------------------------- actions

    def project(self) -> Projection | None:
        """Value every legal action, or None when there is nothing to project
        (no opposing Pokémon on the field)."""
        ctx, foe = self.ctx, self.ctx.opp
        if foe is None or not self.ours:
            return None
        endgame = self.all_seen and len(self.ours) + len(self.theirs) <= ENDGAME_MAX_TOTAL
        our_hp = {id(m): m.current_hp_fraction for m in self.ours}
        their_hp = {id(m): m.current_hp_fraction for m in self.theirs}
        their_hp.setdefault(id(foe), foe.current_hp_fraction)
        values: dict[Action, float] = {}
        notes: dict[Action, str] = {}

        def evaluate(key: Action, mon: Pokemon, hp: float, opening: float | None) -> None:
            if hp <= 0:
                values[key] = (-WIN_VALUE if endgame else -SWEPT_PENALTY) - sum(our_hp.values())
                notes[key] = "faints coming in"
                return
            if endgame:
                value = self._resolve(mon, foe, {**our_hp, id(mon): hp}, their_hp, opening)
                values[key] = value
                # A projected win is at least WIN_VALUE; a loss at most -WIN_VALUE.
                if value >= WIN_VALUE:
                    notes[key] = "projected to win the endgame"
                elif value <= -WIN_VALUE:
                    notes[key] = "projected to lose the endgame"
                else:
                    notes[key] = "endgame projected as a stalemate"
            else:
                cost, stopped = self._sweep_cost(mon, hp, opening)
                cost += mon.current_hp_fraction - hp  # damage taken coming in
                values[key] = -cost
                notes[key] = (
                    f"{foe.species} costs us {cost:.0%} of an HP bar before it is stopped"
                    if stopped
                    else f"{foe.species} is projected to sweep"
                )

        me = ctx.me
        if me is not None and not me.fainted and not ctx.free_switch:
            for move in self.battle.available_moves:
                opening = 0.0 if move.category == MoveCategory.STATUS else ctx.damage(me, move).expected
                evaluate(("move", move.id), me, me.current_hp_fraction, opening)
        for mon in self.battle.available_switches:
            hp = self.entry_hp(mon, mon.current_hp_fraction)
            if not ctx.free_switch:
                hp -= self.their_hit(foe, mon)  # the attack lands on the Pokémon coming in
            evaluate(("switch", id(mon)), mon, hp, None)
        if not values:
            return None
        return Projection("endgame" if endgame else "sweep", values, notes)


def action_key(target: Move | Pokemon) -> Action:
    return ("move", target.id) if isinstance(target, Move) else ("switch", id(target))
