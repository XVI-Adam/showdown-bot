"""Battle data collection: one JSON line per finished battle.

Each record holds the outcome, every decision we made with the reasons behind
it, the state at that moment (so HP changes can be reconstructed), and the
opponent's observable actions. JSON Lines keeps the log appendable and easy
to load with ``pandas.read_json(path, lines=True)`` or a plain loop.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

from poke_env.battle import AbstractBattle, Pokemon

from psbot.heuristics.engine import ActionEvaluation

logger = logging.getLogger(__name__)

TOP_ALTERNATIVES = 3

# Protocol events kept in each battle record: enough to reconstruct who moved
# when, who fainted to whom, and what failed. HP changes are omitted because
# every decision already stores both active Pokémon's HP.
RECORDED_EVENTS = frozenset(
    {
        "turn", "move", "switch", "drag", "faint", "cant", "-status", "-terastallize",
        "-sidestart", "-sideend", "-immune", "-fail", "-miss", "-crit", "win", "tie",
    }
)  # fmt: skip


def _mon_state(mon: Pokemon | None) -> dict[str, Any] | None:
    if mon is None:
        return None
    return {
        "species": mon.species,
        "hp": round(mon.current_hp_fraction, 3),
        "status": mon.status.name if mon.status is not None else None,
    }


def opponent_actions(battle: AbstractBattle) -> list[dict[str, Any]]:
    """The opponent's moves and switches, in order, from the battle's event log.

    poke-env keeps the raw protocol events it parsed in ``_replay_data``; there
    is no public accessor in 0.16, so this reads the private attribute and
    degrades to an empty list if it ever disappears.
    """
    role = battle.opponent_role
    events = getattr(battle, "_replay_data", None)
    if role is None or events is None:
        return []
    actions: list[dict[str, Any]] = []
    turn = 0
    for event in events:
        if len(event) < 2:
            continue
        if event[1] == "turn" and len(event) > 2:
            turn = int(event[2])
        elif event[1] in ("move", "switch", "drag") and len(event) > 3 and event[2].startswith(role):
            actions.append({"turn": turn, "type": event[1], "detail": event[3].split(",")[0]})
    return actions


def recorded_events(battle: AbstractBattle) -> list[list[str]]:
    """The battle's key protocol events, in order (see ``RECORDED_EVENTS``)."""
    events = getattr(battle, "_replay_data", None) or []
    return [e[1:6] for e in events if len(e) > 1 and e[1] in RECORDED_EVENTS]


class BattleRecorder:
    """Collects decisions during battles and appends finished battles to a file.

    :param player_kind: label for the agent being recorded (e.g. "heuristic").
    :param save_events: also store the battle's key protocol events.
    :param keep_in_memory: keep finished records in ``self.records``; turn off
        for large evaluation runs.
    """

    def __init__(
        self,
        path: Path,
        player_kind: str | None = None,
        save_events: bool = False,
        keep_in_memory: bool = True,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.player_kind = player_kind
        self.save_events = save_events
        self.keep_in_memory = keep_in_memory
        self._decisions: dict[str, list[dict[str, Any]]] = {}
        self._started: dict[str, float] = {}
        self.records: list[dict[str, Any]] = []
        self.n_written = 0

    def _base_entry(self, battle: AbstractBattle) -> dict[str, Any]:
        self._started.setdefault(battle.battle_tag, time.monotonic())
        return {
            "turn": battle.turn,
            "active": _mon_state(battle.active_pokemon),
            "opponent": _mon_state(battle.opponent_active_pokemon),
            "forced_switch": bool(battle.force_switch),
        }

    def record_action(
        self,
        battle: AbstractBattle,
        action: str,
        extra: dict[str, Any] | None = None,
        order: Any = None,
    ) -> None:
        """Store a decision made without the heuristic engine (learned agents).

        ``order`` is accepted for compatibility with
        ``psbot.dataset.HumanBattleRecorder`` and not used here."""
        entry = self._base_entry(battle)
        entry["action"] = action
        entry.update(extra or {})
        self._decisions.setdefault(battle.battle_tag, []).append(entry)

    def record_decision(
        self,
        battle: AbstractBattle,
        evaluations: Sequence[ActionEvaluation],
        fallback: str | None = None,
        extra: dict[str, Any] | None = None,
        order: Any = None,
    ) -> None:
        """Store one decision. ``fallback`` explains a non-heuristic choice;
        ``extra`` adds fields such as the agent's predictions for this turn;
        ``order`` is unused here (see ``record_action``)."""
        tag = battle.battle_tag
        entry = self._base_entry(battle)
        if evaluations:
            chosen = evaluations[0]
            entry.update(
                action=chosen.label,
                score=round(chosen.total, 3),
                reasons=chosen.reasons(),
                alternatives=[
                    {"action": e.label, "score": round(e.total, 3)}
                    for e in evaluations[1 : 1 + TOP_ALTERNATIVES]
                ],
            )
        else:
            entry.update(action="fallback", score=None, reasons=[fallback or "no evaluation"])
        entry.update(extra or {})
        self._decisions.setdefault(tag, []).append(entry)

    def finish_battle(self, battle: AbstractBattle, opponent_kind: str | None = None) -> None:
        """Write the finished battle to disk."""
        tag = battle.battle_tag
        started = self._started.pop(tag, None)
        team_size = max(len(battle.team), len(battle.opponent_team))
        record = {
            "battle_tag": tag,
            "player": battle.player_username,
            "player_kind": self.player_kind,
            "role": battle.player_role,
            "opponent": battle.opponent_username,
            "opponent_kind": opponent_kind,
            "won": battle.won,
            "turns": battle.turn,
            "duration_s": round(time.monotonic() - started, 3) if started is not None else None,
            "remaining": sum(1 for m in battle.team.values() if not m.fainted),
            "opponent_remaining": team_size
            - sum(1 for m in battle.opponent_team.values() if m.fainted),
            "team": [m.species for m in battle.team.values()],
            "opponent_revealed": [m.species for m in battle.opponent_team.values()],
            "decisions": self._decisions.pop(tag, []),
            "opponent_actions": opponent_actions(battle),
        }
        if self.save_events:
            record["events"] = recorded_events(battle)
        if self.keep_in_memory:
            self.records.append(record)
        self.n_written += 1
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError:
            logger.exception("could not write battle log to %s", self.path)
