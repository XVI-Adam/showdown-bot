"""Measure any player's battles without changing how it plays.

``attach`` wraps a player's ``choose_move`` and battle-finished hook on that
one instance. After the player has chosen (so the choice itself is never
affected), the probe asks the baseline damage estimate whether a sure
knockout was available:

    an available move with at least a 95% chance to knock out the opposing
    Pokémon (accuracy and damage roll included) while moving first

If so, the decision is an opportunity, and it is *missed* when the chosen
action was not a move with at least a 50% chance to knock out. This is judged
by our damage calculator, so it inherits its blind spots (Focus Sash, Sturdy,
an unseen Choice Scarf); it is an estimate, applied identically to every agent.

The probe computes damage for the player's own battle object, which fills in
estimated stats for opposing Pokémon. Nothing the agents read depends on
those: the PPO observation ignores stats, items and abilities, poke-env's
bots read only their own Pokémon's stats, and the heuristics fill them in
themselves.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from poke_env.battle import AbstractBattle, Move
from poke_env.player import Player

from psbot.evaluation.metrics import battle_metrics
from psbot.heuristics.context import BattleContext, HeuristicWeights

logger = logging.getLogger(__name__)

SURE_KO = 0.95
CHOSE_KO = 0.5
_BASELINE = HeuristicWeights()


def ko_check(battle: AbstractBattle, order: Any) -> tuple[bool, bool]:
    """(sure knockout available, chosen action missed it)."""
    if battle.force_switch or getattr(battle, "_wait", False):
        return False, False
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    if me is None or opp is None or opp.fainted or not battle.available_moves:
        return False, False
    ctx = BattleContext(battle, _BASELINE)
    sure = any(
        ctx.damage(me, move).ko_probability(ctx.opp_hp) >= SURE_KO and ctx.outspeeds(me, move.priority)
        for move in battle.available_moves
    )
    if not sure:
        return False, False
    chosen = getattr(order, "order", None)
    took_it = isinstance(chosen, Move) and ctx.damage(me, chosen).ko_probability(ctx.opp_hp) >= CHOSE_KO
    return True, not took_it


class MetricsSink:
    """Appends one JSON line per finished battle."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.n_written = 0

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        self.n_written += 1


def attach(player: Player, sink: MetricsSink, agent: str, opponent: str) -> Player:
    """Measure every battle ``player`` finishes, labelled ``agent`` vs ``opponent``."""
    choose, finished = player.choose_move, player._battle_finished_callback
    tallies: dict[str, list[int]] = {}

    def probed_choose(battle: AbstractBattle):
        order = choose(battle)
        try:
            opportunity, missed = ko_check(battle, order)
        except Exception:  # diagnostics must never cost a decision
            logger.debug("probe failed", exc_info=True)
            return order
        tally = tallies.setdefault(battle.battle_tag, [0, 0, 0])
        tally[0] += 1
        tally[1] += int(opportunity)
        tally[2] += int(missed)
        return order

    def probed_finished(battle: AbstractBattle) -> None:
        finished(battle)
        try:
            events = [event[1:] for event in battle._replay_data]
            row = battle_metrics(events, battle.player_role or "p1")
        except Exception:
            logger.exception("could not measure %s", battle.battle_tag)
            return
        decisions, opportunities, missed = tallies.pop(battle.battle_tag, [0, 0, 0])
        row.update(
            agent=agent,
            opponent=opponent,
            battle_id=battle.battle_tag,
            won=battle.won,
            remaining=sum(1 for m in battle.team.values() if not m.fainted),
            decisions=decisions,
            sure_ko_chances=opportunities,
            sure_ko_missed=missed,
        )
        sink.write(row)

    player.choose_move = probed_choose  # type: ignore[method-assign]
    player._battle_finished_callback = probed_finished  # type: ignore[method-assign]
    return player
