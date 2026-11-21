"""Records battles against people: every decision, with what was known then.

One JSON line per finished battle (layout in ``psbot.dataset.schema``). The
recorder has the same three methods as ``psbot.recording.BattleRecorder``, so
the agents use either one without knowing which.

The rule this module exists to enforce: battles are partially observable, so
every fact about the opponent carries its source.

``revealed``  the battle log showed it (``psbot.dataset.revealed``)
``inferred``  not shown, but certain from public random-battle data (a species
              with one possible ability, a forme that requires its item)
``unknown``   neither; the value is null

Nothing about the opponent is ever written without one of those labels.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from poke_env.battle import AbstractBattle, Move, Pokemon
from poke_env.data import to_id_str
from poke_env.environment import SinglesEnv
from poke_env.player import BattleOrder

from psbot.config import PROJECT_ROOT
from psbot.dataset.revealed import Revealed, RevealTracker
from psbot.dataset.schema import DROPPED_EVENTS, SCHEMA
from psbot.heuristics.mechanics import own_tera_type
from psbot.heuristics.randbats import certain_ability, certain_item, possible_moves

logger = logging.getLogger(__name__)

DATA_DIR: Path = PROJECT_ROOT / "data"
HUMAN_DIR: Path = DATA_DIR / "human"


@dataclass(frozen=True)
class AgentInfo:
    """Which of our agents played, precisely enough to reproduce it."""

    type: str  # "heuristic", "ppo", "hybrid" or a scripted bot's name
    name: str  # "heuristic" or "<run>/<checkpoint file>"
    checkpoint: str | None = None  # path relative to the project
    checkpoint_sha256: str | None = None


def agent_info(bot: str, player: Any = None) -> AgentInfo:
    """Describe the agent built from ``bot`` (a name or a checkpoint path)."""
    path = Path(bot)
    if path.suffix != ".zip":
        return AgentInfo(type=bot, name=bot)
    kind = "hybrid" if getattr(player, "_is_hybrid", False) else "ppo"
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    try:
        shown = str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        shown = str(path)
    return AgentInfo(kind, f"{path.parent.name}/{path.name}", shown, digest)


def default_path(source: str) -> Path:
    return HUMAN_DIR / f"{source}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.jsonl"


def order_label(order: BattleOrder) -> str:
    """'move earthquake', 'move earthquake +tera' or 'switch gliscor'."""
    target = getattr(order, "order", None)
    if isinstance(target, Move):
        return f"move {target.id}" + (" +tera" if getattr(order, "terastallize", False) else "")
    if isinstance(target, Pokemon):
        return f"switch {target.species}"
    return str(getattr(order, "message", order))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _names(mapping: Any) -> dict[str, int]:
    return {key.name.lower(): int(value) for key, value in (mapping or {}).items()}


def _condition(mon: Pokemon) -> dict[str, Any]:
    """What both players can see of any Pokémon on the field or bench."""
    return {
        "species": mon.species,
        "level": mon.level,
        "hp": round(mon.current_hp_fraction, 4),
        "status": mon.status.name.lower() if mon.status is not None else None,
        "fainted": bool(mon.fainted),
        "boosts": {stat: stage for stat, stage in mon.boosts.items() if stage},
        "terastallized": bool(mon.is_terastallized),
    }


def own_pokemon(ident: str, mon: Pokemon, battle: AbstractBattle | None = None) -> dict[str, Any]:
    """One of ours: the server tells us everything, so nothing needs a label.

    The Tera type comes from the server's request when ``battle`` is given:
    poke-env only fills ``Pokemon.tera_type`` after Terastallizing.
    """
    tera = own_tera_type(battle, mon) if battle is not None else mon.tera_type
    return {
        "ident": ident,
        **_condition(mon),
        "moves": list(mon.moves),
        "ability": mon.ability,
        "item": mon.item or None,
        "tera_type": tera.name.lower() if tera is not None else None,
    }


# Species the log shows under another forme's name, so "the only ability this
# species can have" is not certain. Greninja-Bond (Battle Bond) appears to the
# opponent as plain Greninja (Protean): measured wrong 5 times in 450 battles.
_HIDDEN_FORMES = frozenset({"greninja"})


def _certain(lookup: Any, mon: Pokemon) -> str | None:
    return None if mon.species in _HIDDEN_FORMES else lookup(mon)


def _labelled(revealed: str | None, inferred: str | None) -> dict[str, Any]:
    if revealed:
        return {"value": revealed, "source": "revealed"}
    if inferred:
        return {"value": to_id_str(inferred), "source": "inferred"}
    return {"value": None, "source": "unknown"}


def opponent_pokemon(ident: str, mon: Pokemon, shown: Revealed) -> dict[str, Any]:
    """One of theirs: visible condition plus labelled knowledge."""
    item = _labelled(shown.item, _certain(certain_item, mon))
    if shown.item and shown.item_consumed:
        item["consumed"] = True
    return {
        "ident": ident,
        **_condition(mon),
        "hp_precision": "percent",  # the server reports opposing HP out of 100
        "revealed_moves": list(shown.moves),
        "ability": _labelled(shown.ability, _certain(certain_ability, mon)),
        "item": item,
        "tera_type": _labelled(shown.tera_type, None),
    }


class HumanBattleRecorder:
    """Collects decisions during battles and appends finished battles to a file.

    :param agent: which agent is playing (see :func:`agent_info`).
    :param source: ``"official"`` or ``"local"``: which server the battle is on.
    :param path: output file; a new timestamped file in ``data/human/`` if None.
    """

    wants_policy = True  # asks learned agents for their action probabilities

    def __init__(
        self,
        agent: AgentInfo,
        source: str = "official",
        path: Path | None = None,
        opponent_kind: str = "human",
        keep_in_memory: bool = True,
    ) -> None:
        self.agent = agent
        self.source = source
        self.path = path or default_path(source)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.opponent_kind = opponent_kind
        self.keep_in_memory = keep_in_memory
        self.records: list[dict[str, Any]] = []
        self.n_written = 0
        self._decisions: dict[str, list[dict[str, Any]]] = {}
        self._started: dict[str, str] = {}
        self._trackers: dict[str, tuple[int, RevealTracker]] = {}
        self._rebuilt: set[str] = set()

    # ------------------------------------------------------------- snapshots

    def _revealed(self, battle: AbstractBattle) -> RevealTracker:
        """The reveal tracker for this battle, caught up with its event log."""
        events = getattr(battle, "_replay_data", None) or []
        owner, tracker = self._trackers.get(battle.battle_tag, (0, RevealTracker()))
        if owner != id(battle) or tracker.n_events > len(events):
            if owner:
                # A reconnect replaced the battle object and replayed the log:
                # event positions recorded before this point may not line up.
                self._rebuilt.add(battle.battle_tag)
            tracker = RevealTracker()
        tracker.update(event[1:] for event in events[tracker.n_events :])
        self._trackers[battle.battle_tag] = (id(battle), tracker)
        return tracker

    def _snapshot(self, battle: AbstractBattle) -> dict[str, Any]:
        self._started.setdefault(battle.battle_tag, _now())
        tracker = self._revealed(battle)
        active, opp_active = battle.active_pokemon, battle.opponent_active_pokemon
        ours = [(ident, mon) for ident, mon in battle.team.items()]
        theirs = [(ident, mon) for ident, mon in battle.opponent_team.items()]
        entry: dict[str, Any] = {
            "turn": battle.turn,
            "event_index": tracker.n_events,  # raw; rewritten when the battle is saved
            "forced_switch": bool(battle.force_switch),
            "active": next((own_pokemon(i, m, battle) for i, m in ours if m is active), None),
            "bench": [own_pokemon(i, m, battle) for i, m in ours if m is not active],
            "opponent_active": next(
                (opponent_pokemon(i, m, tracker.get(i)) for i, m in theirs if m is opp_active), None
            ),
            "opponent_bench": [
                opponent_pokemon(i, m, tracker.get(i)) for i, m in theirs if m is not opp_active
            ],
            # How many of their Pokémon we have not seen at all. Nothing is
            # recorded about those, not even a guess.
            "opponent_unseen": max(0, 6 - len(theirs)),
            "field": {
                "weather": sorted(_names(battle.weather)),
                "fields": sorted(_names(battle.fields)),
                "side_conditions": _names(battle.side_conditions),
                "opponent_side_conditions": _names(battle.opponent_side_conditions),
            },
        }
        entry.update(self._legal(battle))
        return entry

    @staticmethod
    def _legal(battle: AbstractBattle) -> dict[str, Any]:
        """Legal actions, in the 26-action space the learned agents use."""
        try:
            mask = [int(x) for x in SinglesEnv.get_action_mask(battle)]
        except Exception:
            logger.debug("no action mask on turn %s", battle.turn, exc_info=True)
            legal = [{"index": None, "action": f"move {m.id}"} for m in battle.available_moves]
            legal += [
                {"index": None, "action": f"switch {m.species}"} for m in battle.available_switches
            ]
            return {"action_mask": None, "legal_actions": legal}
        legal = []
        for index, allowed in enumerate(mask):
            if not allowed:
                continue
            try:
                order = SinglesEnv.action_to_order(np.int64(index), battle, fake=True, strict=False)
                label = order_label(order)
            except Exception:
                label = f"action {index}"
            legal.append({"index": index, "action": label})
        return {"action_mask": mask, "legal_actions": legal}

    @staticmethod
    def _selected(battle: AbstractBattle, order: BattleOrder | None, label: str) -> dict[str, Any]:
        index = None
        if order is not None:
            label = order_label(order)
            try:
                index = int(SinglesEnv.order_to_action(order, battle, fake=True, strict=False))
            except Exception:
                logger.debug("no action index for %s", label, exc_info=True)
        return {"action": label, "index": index}

    def _observation(self, battle: AbstractBattle, extra: dict[str, Any]) -> dict[str, Any] | None:
        """The observation vector at this decision: the one the policy was fed,
        or the basic 112-feature encoding for agents that do not use one."""
        values = extra.pop("observation", None)
        try:
            if values is None:
                from psbot.rl.observation import embed_battle

                values = embed_battle(battle)
            values = [round(float(x), 4) for x in values]
        except Exception:
            logger.debug("could not encode observation", exc_info=True)
            return None
        return {"size": len(values), "values": values}

    # --------------------------------------------------------------- records

    def record_decision(
        self,
        battle: AbstractBattle,
        evaluations: Sequence[Any],
        fallback: str | None = None,
        extra: dict[str, Any] | None = None,
        order: BattleOrder | None = None,
    ) -> None:
        """A decision by the heuristic engine: every legal action's score and
        the named components behind it."""
        extra = dict(extra or {})
        entry = self._snapshot(battle)
        label = evaluations[0].label if evaluations else "fallback"
        entry["selected"] = self._selected(battle, order, label)
        entry["heuristic"] = {
            "fallback": None if evaluations else (fallback or "no evaluation"),
            "actions": [
                {
                    "action": e.label,
                    "total": round(e.total, 4),
                    "components": [
                        {"name": c.name, "value": round(c.value, 4), "reason": c.reason}
                        for c in e.components
                    ],
                }
                for e in evaluations
            ],
        }
        entry["policy"] = None
        entry["observation"] = self._observation(battle, extra)
        entry.update(extra)
        self._decisions.setdefault(battle.battle_tag, []).append(entry)

    def record_action(
        self,
        battle: AbstractBattle,
        action: str,
        extra: dict[str, Any] | None = None,
        order: BattleOrder | None = None,
    ) -> None:
        """A decision by a learned policy: its action probabilities, and for
        the hybrid the heuristic score it was shown for each action."""
        extra = dict(extra or {})
        entry = self._snapshot(battle)
        entry["selected"] = self._selected(battle, order, action)
        entry["heuristic"] = None
        policy = {
            key: extra.pop(key)
            for key in ("probabilities", "deterministic", "heuristic_scores")
            if key in extra
        }
        entry["policy"] = policy or None
        entry["observation"] = self._observation(battle, extra)
        entry.update(extra)
        self._decisions.setdefault(battle.battle_tag, []).append(entry)

    def finish_battle(self, battle: AbstractBattle, opponent_kind: str | None = None) -> None:
        """Assemble the battle's record and append it to the file."""
        tag = battle.battle_tag
        tracker = self._revealed(battle)
        raw = [list(event[1:]) for event in getattr(battle, "_replay_data", None) or []]
        keep = [bool(event) and event[0] not in DROPPED_EVENTS for event in raw]
        events = [event for event, kept in zip(raw, keep) if kept]
        # Positions in the saved (filtered) log for each position in the raw one.
        saved_index = [0]
        for kept in keep:
            saved_index.append(saved_index[-1] + int(kept))

        decisions = self._decisions.pop(tag, [])
        for decision in decisions:
            decision["event_index"] = saved_index[min(decision["event_index"], len(raw))]
        for decision, following in zip(decisions, [*decisions[1:], None]):
            end = following["event_index"] if following is not None else len(events)
            decision["opponent_actions"] = _opponent_actions(
                events[decision["event_index"] : end], battle.opponent_role
            )

        theirs = list(battle.opponent_team.items())
        opponent_team = []
        for ident, mon in theirs:
            entry = opponent_pokemon(ident, mon, tracker.get(ident))
            entry["possible_moves"] = {
                "source": "inferred",
                "note": "random-battle move pool for the species, minus revealed moves",
                "value": sorted(
                    {m.id for m in possible_moves(mon)} - set(entry["revealed_moves"])
                ),
            }
            opponent_team.append(entry)

        record = {
            "schema": SCHEMA,
            "battle_id": tag,
            "source": self.source,
            "format": tag.split("-")[1] if tag.count("-") >= 2 else None,
            "started_at": self._started.pop(tag, None),
            "finished_at": _now(),
            "agent": {**asdict(self.agent), "username": battle.player_username},
            "opponent": {
                "username": battle.opponent_username,
                "kind": opponent_kind or self.opponent_kind,
            },
            "role": battle.player_role,
            "outcome": {
                "won": battle.won,
                "turns": battle.turn,
                "remaining": sum(1 for m in battle.team.values() if not m.fainted),
                "opponent_fainted": sum(1 for _, m in theirs if m.fainted),
                "opponent_seen": len(theirs),
            },
            "state_rebuilt": tag in self._rebuilt,
            "team": [own_pokemon(ident, mon, battle) for ident, mon in battle.team.items()],
            "opponent_team": opponent_team,
            "decisions": decisions,
            "events": events,
        }
        self._trackers.pop(tag, None)
        self._rebuilt.discard(tag)
        if self.keep_in_memory:
            self.records.append(record)
        self.n_written += 1
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except (OSError, TypeError, ValueError):
            logger.exception("could not write battle record to %s", self.path)


def _opponent_actions(events: Sequence[Sequence[str]], role: str | None) -> list[dict[str, Any]]:
    """What the opponent was seen to do in a stretch of the log."""
    actions: list[dict[str, Any]] = []
    if role is None:
        return actions
    for event in events:
        if len(event) < 3 or not event[1].startswith(role):
            continue
        kind = event[0]
        if kind == "move":
            called = any(a.startswith("[from]") and "lockedmove" not in a for a in event[3:])
            actions.append({"type": "move", "id": to_id_str(event[2]), "called": called})
        elif kind in ("switch", "drag"):
            actions.append({"type": kind, "id": to_id_str(event[2].split(",")[0])})
        elif kind == "cant":
            actions.append({"type": "cant", "id": to_id_str(event[2])})
    return actions
