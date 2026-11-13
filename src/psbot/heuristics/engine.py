"""The heuristic decision engine: score every legal action, rank them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from poke_env.battle import AbstractBattle, Move, MoveCategory, Pokemon
from poke_env.data import GenData

from psbot.heuristics.context import BattleContext, HeuristicWeights
from psbot.heuristics.mechanics import own_tera_type
from psbot.heuristics.randbats import GEN
from psbot.heuristics.scorers import (
    DEFAULT_MOVE_SCORERS,
    DEFAULT_SWITCH_SCORERS,
    MoveScorer,
    ScoreComponent,
    SwitchScorer,
)


@dataclass(frozen=True)
class ActionEvaluation:
    """A legal action, its score, and the components that produced it."""

    kind: Literal["move", "switch"]
    target: Move | Pokemon
    components: tuple[ScoreComponent, ...]
    terastallize: bool = False
    tera_reason: str | None = None

    @property
    def total(self) -> float:
        return sum(c.value for c in self.components)

    @property
    def label(self) -> str:
        name = self.target.id if isinstance(self.target, Move) else self.target.species
        return f"{self.kind} {name}" + (" +tera" if self.terastallize else "")

    def reasons(self) -> list[str]:
        """Human-readable justification, largest contributions first."""
        ordered = sorted(self.components, key=lambda c: abs(c.value), reverse=True)
        lines = [str(c) for c in ordered]
        if self.tera_reason:
            lines.append(f"tera ({self.tera_reason})")
        return lines


def terastallize_reason(ctx: BattleContext, move: Move) -> str | None:
    """Why to Terastallize with ``move`` now, or None to keep it in reserve.

    Deliberately simple and offensive-only: use Tera when it powers up the
    attack we already chose, the attack would not KO anyway, and the new type
    is not weak to the opponent's own types.
    """
    me, opp = ctx.me, ctx.opp
    if not ctx.battle.can_tera or me is None or opp is None:
        return None
    tera_type = own_tera_type(ctx.battle, me)
    if tera_type is None:
        return None
    if move.category == MoveCategory.STATUS:
        return None
    estimate = ctx.damage(me, move)
    if estimate.max_fraction <= 0 or estimate.ko_probability(ctx.opp_hp) >= 0.9:
        return None
    if move.type != tera_type and move.id != "terablast":
        return None
    chart = GenData.from_gen(GEN).type_chart
    for opp_type in opp.types:
        if opp_type is not None and opp_type.damage_multiplier(tera_type, type_chart=chart) > 1:
            return None
    return f"Tera {tera_type.name.title()} powers up {move.id}"


class HeuristicEngine:
    """Scores legal actions with a configurable list of scoring components."""

    def __init__(
        self,
        weights: HeuristicWeights | None = None,
        move_scorers: Sequence[MoveScorer] = DEFAULT_MOVE_SCORERS,
        switch_scorers: Sequence[SwitchScorer] = DEFAULT_SWITCH_SCORERS,
    ) -> None:
        self.weights = weights or HeuristicWeights()
        self.move_scorers = tuple(move_scorers)
        self.switch_scorers = tuple(switch_scorers)

    def evaluate(
        self, battle: AbstractBattle, ctx: BattleContext | None = None
    ) -> list[ActionEvaluation]:
        """Return every legal action, best first. Empty if there is none.

        Pass ``ctx`` to reuse an existing analysis of this turn (and its caches).
        """
        legal: list[Move | Pokemon] = [*battle.available_moves, *battle.available_switches]
        if len(legal) == 1:
            # Nothing to decide (e.g. the "recharge" turn after Hyper Beam, whose
            # pseudo-move carries no move data to score).
            only = legal[0]
            kind = "move" if isinstance(only, Move) else "switch"
            forced = ScoreComponent("forced", 0.0, "only legal action")
            return [ActionEvaluation(kind, only, (forced,))]

        ctx = ctx or BattleContext(battle, self.weights)
        evaluations: list[ActionEvaluation] = []

        for move in battle.available_moves:
            components = tuple(c for s in self.move_scorers if (c := s(ctx, move)) is not None)
            evaluations.append(ActionEvaluation("move", move, components))

        for mon in battle.available_switches:
            components = tuple(c for s in self.switch_scorers if (c := s(ctx, mon)) is not None)
            evaluations.append(ActionEvaluation("switch", mon, components))

        evaluations.sort(key=lambda e: e.total, reverse=True)

        if evaluations and evaluations[0].kind == "move":
            best = evaluations[0]
            assert isinstance(best.target, Move)
            reason = terastallize_reason(ctx, best.target)
            if reason is not None:
                evaluations[0] = ActionEvaluation(
                    best.kind, best.target, best.components, terastallize=True, tera_reason=reason
                )
        return evaluations
