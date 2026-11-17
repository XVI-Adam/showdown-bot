"""Layout of a human-battle record, and the checks a record must pass.

One JSON object per line, one line per finished battle::

    schema         "psbot.human_battle/1"
    battle_id      the server's battle room id
    source         "official" | "local"
    format         "gen9randombattle"
    started_at     UTC time of our first decision      finished_at  UTC
    agent          {type, name, checkpoint, checkpoint_sha256, username}
    opponent       {username, kind}
    role           "p1" | "p2"  (ours)
    outcome        {won, turns, remaining, opponent_fainted, opponent_seen}
    state_rebuilt  true if a reconnect rebuilt the battle state mid-game
    team           our six, exactly as the server described them
    opponent_team  every opposing Pokémon seen, with labelled knowledge and
                   the species' possible moves (labelled "inferred")
    decisions      one entry per decision we made, in order (below)
    events         the public battle log, minus chat and joins

    decision:
      turn, forced_switch
      event_index       how many of ``events`` had happened when we decided
      active, bench     our Pokémon: hp, status, boosts, moves, ability, item
      opponent_active, opponent_bench
                        theirs: hp, status, boosts, revealed_moves, and
                        ability / item / tera_type as {value, source}
      opponent_unseen   how many of theirs have not appeared at all
      field             weather, terrain, hazards and screens on each side
      action_mask       26 zeros/ones: legality in the learned agents' action space
      legal_actions     [{index, action}] for every legal action
      selected          {index, action}
      heuristic         {fallback, actions: [{action, total, components}]} | null
      policy            {probabilities, deterministic, heuristic_scores} | null
      observation       {size, values}: the feature vector at that moment
      opponent_actions  what the opponent was then seen to do, up to our next
                        decision

``source`` on opponent knowledge is "revealed", "inferred" or "unknown"; see
``psbot.dataset.human``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from poke_env.data import to_id_str

from psbot.dataset.revealed import Revealed, RevealTracker

SCHEMA = "psbot.human_battle/1"
SOURCES = ("revealed", "inferred", "unknown")
ACTION_SPACE = 26
OBSERVATION_SIZES = (112, 165)

# Not saved: chat, joins and leaves, and client-only display messages. The
# battle itself is unaffected, and other people's chat is not ours to keep.
DROPPED_EVENTS = frozenset(
    {
        "", "c", "c:", "chat", "j", "J", "join", "l", "L", "leave", "n", "N", "name",
        "raw", "html", "uhtml", "uhtmlchange", "inactive", "inactiveoff", "deinit",
        "bigerror", "debug", "unlink", "hidelines", "tempnotify", "tempnotifyoff",
    }
)  # fmt: skip

# The subset of events the older log format kept (psbot.recording).
_LEGACY_EVENTS = frozenset(
    {
        "turn", "move", "switch", "drag", "faint", "cant", "-status", "-terastallize",
        "-sidestart", "-sideend", "-immune", "-fail", "-miss", "-crit", "win", "tie",
    }
)  # fmt: skip

_REQUIRED = (
    "battle_id", "source", "format", "started_at", "finished_at", "agent", "opponent",
    "role", "outcome", "team", "opponent_team", "decisions", "events",
)  # fmt: skip
_DECISION_KEYS = (
    "turn", "event_index", "forced_switch", "active", "bench", "opponent_active",
    "opponent_bench", "opponent_unseen", "field", "action_mask", "legal_actions",
    "selected", "heuristic", "policy", "observation", "opponent_actions",
)  # fmt: skip


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.where}: {self.message}"


def load_dataset(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(location, record)`` for every line of a file, or of every
    ``*.jsonl`` file in a directory."""
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for file in files:
        with file.open(encoding="utf-8") as fh:
            for number, line in enumerate(fh, 1):
                if line.strip():
                    yield f"{file.name}:{number}", json.loads(line)


def _check_label(issues: list[Issue], where: str, name: str, label: Any) -> None:
    if not isinstance(label, dict) or label.get("source") not in SOURCES:
        issues.append(Issue("error", where, f"{name} has no valid knowledge source"))
    elif label["source"] == "unknown" and label.get("value") is not None:
        issues.append(Issue("error", where, f"{name} is 'unknown' but has a value"))
    elif label["source"] != "unknown" and not isinstance(label.get("value"), str):
        issues.append(Issue("error", where, f"{name} is '{label['source']}' without a value"))


def _check_opponent(
    issues: list[Issue],
    where: str,
    mon: dict[str, Any],
    shown: Revealed | None,
    log_text: str | None,
) -> None:
    """One opposing Pokémon: labels are well formed, and (when the log is
    usable) every 'revealed' fact is in the log and no shown fact is missing."""
    where = f"{where} {mon.get('ident')}"
    for name in ("ability", "item", "tera_type"):
        _check_label(issues, where, name, mon.get(name))
    if not 0.0 <= mon.get("hp", -1) <= 1.0:
        issues.append(Issue("error", where, "hp outside [0, 1]"))
    if shown is None or log_text is None:
        return
    if list(mon.get("revealed_moves", [])) != shown.moves:
        issues.append(
            Issue("error", where, f"revealed_moves {mon.get('revealed_moves')} != log {shown.moves}")
        )
    for name, logged in (("ability", shown.ability), ("item", shown.item), ("tera_type", shown.tera_type)):
        label = mon.get(name) or {}
        if label.get("source") == "revealed":
            if label.get("value") != logged:
                issues.append(Issue("error", where, f"{name} marked revealed but the log shows {logged}"))
            elif label["value"] not in log_text:  # independent of the tracker's rules
                issues.append(Issue("error", where, f"{name} {label['value']} never appears in the log"))
        elif logged is not None:
            issues.append(Issue("error", where, f"{name} {logged} was revealed but is marked {label.get('source')}"))


def validate_record(record: dict[str, Any]) -> list[Issue]:
    """Check one battle record. An empty list means it is sound."""
    issues: list[Issue] = []
    battle = str(record.get("battle_id", "?"))

    def error(where: str, message: str) -> None:
        issues.append(Issue("error", f"{battle} {where}".strip(), message))

    def warning(where: str, message: str) -> None:
        issues.append(Issue("warning", f"{battle} {where}".strip(), message))

    if record.get("schema") != SCHEMA:
        error("", f"schema is {record.get('schema')!r}, expected {SCHEMA!r}")
        return issues
    missing = [key for key in _REQUIRED if key not in record]
    if missing:
        error("", f"missing fields: {', '.join(missing)}")
        return issues

    events: list[list[str]] = record["events"]
    outcome, agent = record["outcome"], record["agent"]
    if outcome.get("won") not in (True, False, None):
        error("outcome", "won must be true, false or null")
    if not 0 <= outcome.get("remaining", -1) <= 6:
        error("outcome", "remaining outside 0-6")
    if not record["opponent"].get("username"):
        error("opponent", "no username")
    if not agent.get("type") or not agent.get("name"):
        error("agent", "no agent type or name")
    if agent.get("type") in ("ppo", "hybrid") and not agent.get("checkpoint"):
        error("agent", "a learned agent must name its checkpoint")
    if not 1 <= len(record["team"]) <= 6:
        error("team", "expected 1-6 Pokémon")
    winners = [event[1] for event in events if len(event) > 1 and event[0] == "win"]
    if winners:
        if (winners[-1] == agent.get("username")) != (outcome.get("won") is True):
            error("outcome", "won disagrees with the log's winner")
    elif not any(event and event[0] == "tie" for event in events):
        error("events", "the log has no result")
    if not record["decisions"]:
        warning("decisions", "battle has no recorded decision")

    log_usable = not record.get("state_rebuilt")
    if not log_usable:
        warning("", "state was rebuilt after a reconnect: log positions not checked")
    tracker, text_upto, log_text = RevealTracker(), 0, ""
    last_turn, last_index = 0, 0
    for number, decision in enumerate(record["decisions"]):
        where = f"decision {number} (turn {decision.get('turn')})"
        absent = [key for key in _DECISION_KEYS if key not in decision]
        if absent:
            error(where, f"missing fields: {', '.join(absent)}")
            continue
        turn, index = decision["turn"], decision["event_index"]
        if turn < last_turn:
            error(where, "turn goes backwards")
        if not last_index <= index <= len(events):
            if log_usable:
                error(where, "event_index out of order or out of range")
            index = max(last_index, min(index, len(events)))
        last_turn = max(last_turn, turn)

        # Legal actions and the choice.
        mask, legal, selected = decision["action_mask"], decision["legal_actions"], decision["selected"]
        labels = {entry["action"] for entry in legal}
        if not legal:
            error(where, "no legal action recorded")
        if mask is None:
            warning(where, "no action mask")
        else:
            if len(mask) != ACTION_SPACE or any(bit not in (0, 1) for bit in mask):
                error(where, f"action_mask is not {ACTION_SPACE} zeros and ones")
            elif [i for i, bit in enumerate(mask) if bit] != [entry["index"] for entry in legal]:
                error(where, "legal_actions do not match action_mask")
            elif selected["index"] is None:
                warning(where, f"selected action {selected['action']!r} has no index")
            elif not 0 <= selected["index"] < ACTION_SPACE or not mask[selected["index"]]:
                error(where, f"selected action {selected['action']!r} is not legal")
            if selected["index"] is not None and selected["action"] not in labels:
                error(where, f"selected action {selected['action']!r} is not among legal_actions")

        # The reasoning.
        heuristic, policy = decision["heuristic"], decision["policy"]
        if heuristic is None and policy is None:
            error(where, "neither heuristic scores nor policy output recorded")
        if heuristic is not None:
            actions = heuristic.get("actions", [])
            if not actions and not heuristic.get("fallback"):
                error(where, "heuristic block has neither actions nor a fallback reason")
            for action in actions:
                total = sum(c["value"] for c in action["components"])
                if abs(total - action["total"]) > 1e-3 * (1 + len(action["components"])):
                    error(where, f"{action['action']}: components do not add up to the total")
            if actions and actions[0]["total"] < max(a["total"] for a in actions) - 1e-9:
                error(where, "heuristic actions are not ranked best first")
        if policy is not None and policy.get("probabilities") is not None:
            probabilities = policy["probabilities"]
            if len(probabilities) != ACTION_SPACE or abs(sum(probabilities) - 1) > 1e-3:
                error(where, "policy probabilities are not a distribution over 26 actions")
            elif mask is not None and len(mask) == ACTION_SPACE:
                if any(p > 1e-6 for p, bit in zip(probabilities, mask) if not bit):
                    error(where, "policy gives probability to an illegal action")
        observation = decision["observation"]
        if observation is None:
            warning(where, "no observation vector")
        elif observation["size"] != len(observation["values"]) or observation["size"] not in OBSERVATION_SIZES:
            error(where, f"observation has unexpected size {observation.get('size')}")
        elif any(not -1.0001 <= value <= 1.0001 for value in observation["values"]):
            error(where, "observation value outside [-1, 1]")

        # What was known about the opponent, against the log up to this point.
        shown_now: dict[str, Revealed] | None = None
        if log_usable:
            tracker.update(events[tracker.n_events : index])
            log_text += " ".join(to_id_str(" ".join(event)) for event in events[text_upto:index]) + " "
            text_upto = index
            shown_now = tracker.mons
        opponents = [decision["opponent_active"], *decision["opponent_bench"]]
        for mon in opponents:
            if mon is None:
                continue
            shown = shown_now.get(mon["ident"], Revealed()) if shown_now is not None else None
            _check_opponent(issues, f"{battle} {where}", mon, shown, log_text if log_usable else None)
        seen = sum(1 for mon in opponents if mon is not None)
        if seen + decision["opponent_unseen"] != 6:
            warning(where, "seen + unseen opposing Pokémon is not 6")
        for mon in [decision["active"], *decision["bench"]]:
            if mon is not None and not 0.0 <= mon.get("hp", -1) <= 1.0:
                error(where, f"{mon.get('ident')}: hp outside [0, 1]")
        last_index = index

    final = RevealTracker().update(events).mons if log_usable else None
    final_text = " ".join(to_id_str(" ".join(event)) for event in events)
    for mon in record["opponent_team"]:
        shown = final.get(mon["ident"], Revealed()) if final is not None else None
        _check_opponent(issues, f"{battle} opponent_team", mon, shown, final_text if log_usable else None)
        pool = mon.get("possible_moves") or {}
        if pool.get("source") != "inferred":
            error("opponent_team", f"{mon.get('ident')}: possible_moves must be labelled inferred")
        elif set(pool.get("value", [])) & set(mon.get("revealed_moves", [])):
            error("opponent_team", f"{mon.get('ident')}: a revealed move is also listed as possible")
    return issues


def to_legacy_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert to the layout ``psbot.recording`` writes, which the existing
    analysis tools read. Loses the extra detail; invents nothing."""

    def brief(mon: dict[str, Any] | None) -> dict[str, Any] | None:
        if mon is None:
            return None
        status = mon["status"].upper() if mon["status"] else None
        return {"species": mon["species"], "hp": round(mon["hp"], 3), "status": status}

    decisions = []
    for decision in record["decisions"]:
        entry: dict[str, Any] = {
            "turn": decision["turn"],
            "active": brief(decision["active"]),
            "opponent": brief(decision["opponent_active"]),
            "forced_switch": decision["forced_switch"],
            "action": decision["selected"]["action"],
        }
        heuristic = decision.get("heuristic")
        if heuristic and heuristic["actions"]:
            best = heuristic["actions"][0]
            ordered = sorted(best["components"], key=lambda c: abs(c["value"]), reverse=True)
            entry.update(
                score=round(best["total"], 3),
                reasons=[f"{c['name']} {c['value']:+.2f} ({c['reason']})" for c in ordered],
                alternatives=[
                    {"action": a["action"], "score": round(a["total"], 3)}
                    for a in heuristic["actions"][1:4]
                ],
            )
        elif heuristic:
            entry.update(action="fallback", score=None, reasons=[heuristic["fallback"]])
        for key in ("predicted", "agrees_with_heuristic", "heuristic_action", "illegal"):
            if key in decision:
                entry[key] = decision[key]
        decisions.append(entry)

    role = record["role"] or "p1"
    actions, turn = [], 0
    for event in record["events"]:
        if event[0] == "turn" and len(event) > 1:
            turn = int(event[1])
        elif event[0] in ("move", "switch", "drag") and len(event) > 2 and not event[1].startswith(role):
            actions.append({"turn": turn, "type": event[0], "detail": event[2].split(",")[0]})
    outcome = record["outcome"]
    return {
        "battle_tag": record["battle_id"],
        "player": record["agent"]["username"],
        "player_kind": record["agent"]["name"].split("/")[0],
        "role": record["role"],
        "opponent": record["opponent"]["username"],
        "opponent_kind": record["opponent"]["kind"],
        "won": outcome["won"],
        "turns": outcome["turns"],
        "duration_s": None,
        "remaining": outcome["remaining"],
        "opponent_remaining": 6 - outcome["opponent_fainted"],
        "team": [mon["species"] for mon in record["team"]],
        "opponent_revealed": [mon["species"] for mon in record["opponent_team"]],
        "decisions": decisions,
        "opponent_actions": actions,
        "events": [event[:5] for event in record["events"] if event[0] in _LEGACY_EVENTS],
    }
