"""Find patterns in recorded battles that separate losses from wins.

Input is the JSON Lines written by ``BattleRecorder`` with ``save_events``.
Everything here is observational: a factor that is much more common in losses
than in wins is a lead worth investigating, not proof of cause.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

from poke_env.battle import Move, MoveCategory, Target
from poke_env.data import to_id_str

from psbot.heuristics.mechanics import MAX_LAYERS
from psbot.heuristics.scorers import EXTRA_RECOVERY_FRACTIONS

Record = dict[str, Any]

HAZARD_NAMES = ("Stealth Rock", "Spikes", "Toxic Spikes", "Sticky Web")
# A prediction is counted as wrong when it misses by more than this much HP.
CALIBRATION_TOLERANCE = 0.2
# An "unexpected" knockout: we stayed in believing this was the most likely to happen.
UNEXPECTED_KO_THRESHOLD = 0.25


def load_records(path: Path) -> Iterator[Record]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


@lru_cache(maxsize=None)
def move_kind(move_name: str) -> str:
    """Classify a move: attack, setup, hazard, status, recovery or other."""
    try:
        move = Move(to_id_str(move_name), gen=9)
        if move.id == "recharge":
            return "other"
        if move.category != MoveCategory.STATUS:
            return "attack"
        if move.target == Target.SELF and move.boosts and any(v > 0 for v in move.boosts.values()):
            return "setup"
        if move.side_condition in MAX_LAYERS:
            return "hazard"
        if move.status is not None and move.target != Target.SELF:
            return "status"
        if move.heal > 0 or move.id in EXTRA_RECOVERY_FRACTIONS:
            return "recovery"
    except Exception:  # unknown or pseudo move
        pass
    return "other"


@dataclass
class Timeline:
    """One battle's events, organised from the recorded player's point of view."""

    moves: dict[int, list[tuple[str, str, str]]] = field(default_factory=lambda: defaultdict(list))
    """turn -> [(side, species, move name)] in the order they happened"""
    our_faints: dict[int, list[str]] = field(default_factory=lambda: defaultdict(list))
    """turn -> species of ours that fainted"""
    kos_by: Counter = field(default_factory=Counter)
    """opposing species -> how many of ours fainted while it was active"""
    setup_by: Counter = field(default_factory=Counter)
    """opposing species -> setup moves it used"""
    lead_history: list[int] = field(default_factory=list)
    """(their faints - our faints) after each faint"""
    their_tera_turns: set[int] = field(default_factory=set)
    crits_on_us: set[int] = field(default_factory=set)
    """turns on which we took a critical hit"""
    our_move_kinds: Counter = field(default_factory=Counter)
    their_move_kinds: Counter = field(default_factory=Counter)
    turns_lost: int = 0
    failed_moves: int = 0
    hazards_on_our_side: bool = False


def build_timeline(record: Record) -> Timeline:
    role = record.get("role") or "p1"
    timeline = Timeline()
    active = {"us": "", "them": ""}
    turn = 0
    ours = theirs = 0

    def side_of(ident: str) -> str:
        return "us" if ident[:2] == role else "them"

    for event in record.get("events", []):
        kind = event[0]
        if kind == "turn":
            turn = int(event[1])
        elif kind in ("switch", "drag") and len(event) > 2:
            active[side_of(event[1])] = to_id_str(event[2].split(",")[0])
        elif kind == "move" and len(event) > 2:
            side = side_of(event[1])
            timeline.moves[turn].append((side, active[side], event[2]))
            kind_of_move = move_kind(event[2])
            if side == "us":
                timeline.our_move_kinds[kind_of_move] += 1
            else:
                timeline.their_move_kinds[kind_of_move] += 1
                if kind_of_move == "setup":
                    timeline.setup_by[active["them"]] += 1
        elif kind == "faint":
            if side_of(event[1]) == "us":
                ours += 1
                timeline.our_faints[turn].append(active["us"])
                timeline.kos_by[active["them"]] += 1
            else:
                theirs += 1
            timeline.lead_history.append(theirs - ours)
        elif kind == "cant" and side_of(event[1]) == "us":
            timeline.turns_lost += 1
        elif kind == "-miss" and side_of(event[1]) == "us":
            timeline.failed_moves += 1  # event names the attacker
        elif kind == "-immune" and side_of(event[1]) == "them":
            timeline.failed_moves += 1  # event names the immune target
        elif kind == "-fail" and side_of(event[1]) == "us":
            timeline.failed_moves += 1
        elif kind == "-terastallize" and side_of(event[1]) == "them":
            timeline.their_tera_turns.add(turn)
        elif kind == "-crit" and side_of(event[1]) == "us":
            timeline.crits_on_us.add(turn)  # event names the Pokémon that was hit
        elif kind == "-sidestart" and len(event) > 2 and side_of(event[1]) == "us":
            if any(name in event[2] for name in HAZARD_NAMES):
                timeline.hazards_on_our_side = True
    return timeline


def _voluntary_switch(decision: Record) -> bool:
    active = decision.get("active")
    return (
        decision.get("action", "").startswith("switch")
        and not decision.get("forced_switch")
        and bool(active)
        and active["hp"] > 0
    )


def battle_factors(record: Record) -> dict[str, float]:
    """Yes/no (1.0/0.0) and count features of one battle."""
    timeline = build_timeline(record)
    decisions = record.get("decisions", [])
    top_killer, top_kos = (timeline.kos_by.most_common(1) or [("", 0)])[0]
    swept = top_kos >= 3
    lead = timeline.lead_history

    unexpected_kos = speed_surprises = speed_checks = 0
    for d in decisions:
        predicted = d.get("predicted") or {}
        if not d.get("action", "").startswith("move") or "threat_ko" not in predicted:
            continue
        turn, species = d["turn"], (d.get("active") or {}).get("species")
        if predicted["threat_ko"] < UNEXPECTED_KO_THRESHOLD and species in timeline.our_faints[turn]:
            unexpected_kos += 1
        sides = [side for side, _, _ in timeline.moves[turn]]
        if predicted.get("first") and "us" in sides and "them" in sides:
            speed_checks += 1
            speed_surprises += int(sides[0] == "them")

    voluntary = [_voluntary_switch(d) for d in decisions]
    chained = sum(1 for a, b in zip(voluntary, voluntary[1:]) if a and b)

    return {
        "swept": float(swept),
        "swept_after_setup": float(swept and timeline.setup_by[top_killer] > 0),
        "opponent_used_setup": float(sum(timeline.setup_by.values()) > 0),
        "hazards_on_our_side": float(timeline.hazards_on_our_side),
        "led_by_2": float(bool(lead) and max(lead) >= 2),
        "trailed_by_2": float(bool(lead) and min(lead) <= -2),
        "close_finish": float(abs(record["remaining"] - record["opponent_remaining"]) <= 1),
        "unexpected_ko": float(unexpected_kos > 0),
        "outsped_unexpectedly": float(speed_surprises > 0),
        "n_unexpected_kos": float(unexpected_kos),
        "n_speed_surprises": float(speed_surprises),
        "n_speed_checks": float(speed_checks),
        "n_turns_lost": float(timeline.turns_lost),
        "n_failed_moves": float(timeline.failed_moves),
        "n_voluntary_switches": float(sum(voluntary)),
        "n_chained_switches": float(chained),
        "n_opponent_setup_moves": float(sum(timeline.setup_by.values())),
        "n_our_setup_moves": float(timeline.our_move_kinds["setup"]),
        "n_our_hazard_moves": float(timeline.our_move_kinds["hazard"]),
        "n_our_status_moves": float(timeline.our_move_kinds["status"]),
        "n_our_recovery_moves": float(timeline.our_move_kinds["recovery"]),
        "turns": float(record["turns"]),
    }


UNEXPECTED_KO_CAUSES = (
    "no_attack",  # fainted to residual damage, recoil or hazards
    "critical_hit",
    "opponent_terastallized",
    "unseen_move",  # an attack that Pokémon had not used before in the battle
    "known_move_underestimated",
)


def unexpected_ko_causes(records: Iterable[Record]) -> dict[str, Any]:
    """Why the heuristic lost Pokémon it expected to survive the turn.

    Each unexpected knockout gets the first explanation that applies, in the
    order of ``UNEXPECTED_KO_CAUSES``, so the shares add up to 100%.
    """
    causes: Counter = Counter()
    for record in records:
        timeline = build_timeline(record)
        seen: dict[str, set[str]] = defaultdict(set)  # opposing species -> attacks used so far
        by_turn = {
            d["turn"]: d
            for d in record.get("decisions", [])
            if d.get("action", "").startswith("move") and "threat_ko" in (d.get("predicted") or {})
        }
        for turn in sorted(timeline.moves):
            attacks = [
                (species, name)
                for side, species, name in timeline.moves[turn]
                if side == "them" and move_kind(name) == "attack"
            ]
            d = by_turn.get(turn)
            fainted = (d.get("active") or {}).get("species") in timeline.our_faints[turn] if d else False
            if d and fainted and d["predicted"]["threat_ko"] < UNEXPECTED_KO_THRESHOLD:
                if not attacks:
                    causes["no_attack"] += 1
                elif turn in timeline.crits_on_us:
                    causes["critical_hit"] += 1
                elif turn in timeline.their_tera_turns:
                    causes["opponent_terastallized"] += 1
                elif any(name not in seen[species] for species, name in attacks):
                    causes["unseen_move"] += 1
                else:
                    causes["known_move_underestimated"] += 1
            for species, name in attacks:
                seen[species].add(name)
    total = sum(causes.values())
    return {
        "total": total,
        "shares": {c: (causes[c] / total if total else 0.0) for c in UNEXPECTED_KO_CAUSES},
    }


def compare_wins_and_losses(records: Iterable[Record]) -> dict[str, Any]:
    """Average every battle factor separately over wins and over losses."""
    sums: dict[bool, Counter] = {True: Counter(), False: Counter()}
    counts = {True: 0, False: 0}
    for record in records:
        if record.get("won") is None:
            continue
        won = bool(record["won"])
        counts[won] += 1
        sums[won].update(battle_factors(record))
    names = sorted(set(sums[True]) | set(sums[False]))
    return {
        "wins": counts[True],
        "losses": counts[False],
        "factors": {
            name: {
                "wins": sums[True][name] / counts[True] if counts[True] else 0.0,
                "losses": sums[False][name] / counts[False] if counts[False] else 0.0,
            }
            for name in names
        },
    }


def _bucket(predicted: float, actual: float) -> str:
    if actual - predicted > CALIBRATION_TOLERANCE:
        return "under"
    if predicted - actual > CALIBRATION_TOLERANCE:
        return "over"
    return "ok"


def prediction_calibration(records: Iterable[Record]) -> dict[str, Any]:
    """Compare the heuristic's predictions with what happened next.

    * ``threat``: damage our active Pokémon was expected to take on turns the
      opponent attacked it, against the HP it actually lost that turn.
    * ``entry``: damage a voluntary switch-in was expected to take on entry.
    * ``speed``: how often we moved second after predicting we would move first.
    """
    threat: Counter = Counter()
    entry: Counter = Counter()
    speed: Counter = Counter()
    threat_error = entry_error = 0.0
    for record in records:
        timeline = build_timeline(record)
        decisions = record.get("decisions", [])
        for d in decisions:
            if d.get("action", "").startswith("move") and (d.get("predicted") or {}).get("first"):
                sides = [side for side, _, _ in timeline.moves[d["turn"]]]
                if "us" in sides and "them" in sides:
                    speed["checked"] += 1
                    speed["moved_second"] += int(sides[0] == "them")
        for now, after in zip(decisions, decisions[1:]):
            predicted = now.get("predicted") or {}
            active_now, active_after = now.get("active"), after.get("active")
            if not active_now or not active_after:
                continue
            turn = now["turn"]
            if now.get("action", "").startswith("move") and "threat" in predicted:
                attacked = any(
                    side == "them" and move_kind(name) == "attack"
                    for side, _, name in timeline.moves[turn]
                )
                same_mon = active_after["species"] == active_now["species"]
                next_turn = after["turn"] == turn + 1 or (
                    after.get("forced_switch") and active_after["hp"] == 0
                )
                if attacked and same_mon and next_turn:
                    actual = max(active_now["hp"] - active_after["hp"], 0.0)
                    expected = min(predicted["threat"], active_now["hp"])
                    threat[_bucket(expected, actual)] += 1
                    threat["n"] += 1
                    threat_error += actual - expected
            elif _voluntary_switch(now) and "entry" in predicted:
                target = now["action"].split(" ", 1)[1]
                if active_after["species"] == target:
                    actual = max(predicted["target_hp"] - active_after["hp"], 0.0)
                    entry[_bucket(predicted["entry"], actual)] += 1
                    entry["n"] += 1
                    entry["fainted_on_entry"] += int(active_after["hp"] == 0)
                    entry_error += actual - predicted["entry"]

    def shares(counter: Counter, error: float) -> dict[str, float]:
        n = counter["n"]
        out = {k: (counter[k] / n if n else 0.0) for k in ("under", "ok", "over")}
        out.update(n=n, mean_error=error / n if n else 0.0)
        return out

    result = {"threat": shares(threat, threat_error), "entry": shares(entry, entry_error)}
    result["entry"]["fainted_on_entry"] = entry["fainted_on_entry"] / entry["n"] if entry["n"] else 0.0
    result["speed"] = {
        "n": speed["checked"],
        "moved_second": speed["moved_second"] / speed["checked"] if speed["checked"] else 0.0,
    }
    return result


def _action_kind(label: str) -> str:
    if label.startswith("switch"):
        return "switch"
    return "move + tera" if label.endswith("+tera") else "move"


def hybrid_deviations(records: Iterable[Record]) -> dict[str, Any]:
    """How a hybrid agent's choices differ from the heuristic's top choice."""
    choices = disagreements = 0
    kinds: Counter = Counter()
    for record in records:
        for d in record.get("decisions", []):
            if "agrees_with_heuristic" not in d:
                continue
            choices += 1
            if not d["agrees_with_heuristic"]:
                disagreements += 1
                heuristic = _action_kind(d.get("heuristic_action", ""))
                kinds[f"{heuristic} -> {_action_kind(d['action'])}"] += 1
    return {
        "choices": choices,
        "disagreements": disagreements,
        "rate": disagreements / choices if choices else 0.0,
        "kinds": dict(kinds.most_common()),
    }
