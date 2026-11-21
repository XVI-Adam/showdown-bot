"""The heuristic agent: plays the highest-scoring action from the engine."""

from __future__ import annotations

import logging
from typing import Any

from poke_env.battle import AbstractBattle, Move, Pokemon
from poke_env.player import BattleOrder, Player

from psbot.heuristics.context import BattleContext, v2_weights
from psbot.heuristics.engine import ActionEvaluation, HeuristicEngine
from psbot.heuristics.scorers import entry_damage
from psbot.recording import BattleRecorder  # or psbot.dataset.HumanBattleRecorder: same methods

logger = logging.getLogger(__name__)


def predictions(ctx: BattleContext, chosen: ActionEvaluation) -> dict[str, Any]:
    """What the engine expected to happen this turn, for later comparison with
    what did happen. Never raises: missing pieces are simply left out."""
    out: dict[str, Any] = {}
    try:
        me, target = ctx.me, chosen.target
        if ctx.opp is None:
            return out
        if isinstance(target, Move) and me is not None and not me.fainted:
            threat = ctx.threat(me)
            estimate = ctx.damage(me, target)
            out["predicted"] = {
                "first": ctx.outspeeds(me, target.priority),
                "threat": round(threat.damage.expected, 3),
                "threat_ko": round(threat.damage.ko_probability(ctx.my_hp), 3),
                "threat_move": threat.move_id,
                "threat_revealed": threat.revealed,
                "dealt": round(estimate.expected, 3),
                "ko": round(estimate.ko_probability(ctx.opp_hp), 3),
            }
        elif isinstance(target, Pokemon):
            hazards, hit = entry_damage(ctx, target)
            out["predicted"] = {
                "entry": round(hazards + hit, 3),
                "target_hp": round(target.current_hp_fraction, 3),
            }
    except Exception:  # predictions are diagnostics; they must not affect play
        logger.debug("could not record predictions", exc_info=True)
    return out


class HeuristicPlayer(Player):
    """Greedy over :class:`HeuristicEngine` scores.

    If the engine raises, the player falls back to a random legal action so a
    bug costs one decision instead of the battle. Fallbacks are counted in
    ``n_fallback_decisions`` and logged, so they cannot go unnoticed.
    """

    def __init__(
        self,
        *args: Any,
        engine: HeuristicEngine | None = None,
        recorder: BattleRecorder | None = None,
        opponent_kind: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.engine = engine or HeuristicEngine()
        self.recorder = recorder
        self.opponent_kind = opponent_kind
        self.n_decisions = 0
        self.n_fallback_decisions = 0

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        self.n_decisions += 1
        try:
            ctx = BattleContext(battle, self.engine.weights)
            evaluations = self.engine.evaluate(battle, ctx)
        except Exception:
            logger.exception("heuristic engine failed on turn %s of %s", battle.turn, battle.battle_tag)
            return self._fallback(battle, "engine error")
        if not evaluations:
            return self._fallback(battle, "no legal move or switch offered")

        best = evaluations[0]
        logger.debug("turn %s: %s | %s", battle.turn, best.label, "; ".join(best.reasons()))
        order = self.create_order(best.target, terastallize=best.terastallize)
        if self.recorder is not None:
            self.recorder.record_decision(
                battle, evaluations, extra=predictions(ctx, best), order=order
            )
        return order

    def _fallback(self, battle: AbstractBattle, why: str) -> BattleOrder:
        self.n_fallback_decisions += 1
        order = self.choose_random_move(battle)
        if self.recorder is not None:
            self.recorder.record_decision(battle, [], fallback=why, order=order)
        return order

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        if self.recorder is not None:
            self.recorder.finish_battle(battle, self.opponent_kind)


class HeuristicV2Player(HeuristicPlayer):
    """The heuristic agent with the Phase 5 engine (``v2_weights``): the
    opponent move model, the sweep/endgame projection and the damage
    refinements. ``HeuristicPlayer`` stays the unchanged baseline."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("engine", HeuristicEngine(v2_weights()))
        super().__init__(*args, **kwargs)
