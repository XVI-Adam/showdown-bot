"""Measure one battle from its protocol log, from one player's side.

Input is the battle's event list as poke-env keeps it (``battle._replay_data``,
leading empty field removed) or as ``psbot.dataset`` saves it. Only what the
log shows is used. Every metric is defined here once, so the evaluation and
the human-battle report cannot drift apart:

stages           knockouts dealt and suffered in turns 1-10, 11-20, 21+
setup sweeper    an opposing Pokémon that used a setup move and then knocked
                 out at least one of ours before leaving the field
swept            one opposing Pokémon knocked out three or more of ours
unrevealed       one of ours fainted to a move that the attacker was using
                 for the first time in the battle
endgame          the first moment both sides have at most two Pokémon left
                 (and at least one); whether we were ahead, level or behind
immune turns     our move was answered by ``-immune`` on the target
switch mistakes  a voluntary switch whose incoming Pokémon fainted before it
                 acted, or lost half its HP or more on the turn it came in

A voluntary switch is a ``switch`` of ours that is not the lead, not a
replacement after a faint, and not the second half of our own pivot move
(U-turn, Volt Switch...). ``drag`` (Roar, Whirlwind) is never voluntary.
"""

from __future__ import annotations

from typing import Any, Sequence

from poke_env.data import to_id_str

from psbot.analysis import move_kind
from psbot.analytics.opponents import PIVOT_MOVES

STAGES = (("early", 1, 10), ("mid", 11, 20), ("late", 21, 10_000))
ENDGAME_MAX_LEFT = 2
TEAM_SIZE = 6
BIG_ENTRY_HIT = 0.5


def _stage(turn: int) -> str:
    for name, first, last in STAGES:
        if first <= turn <= last:
            return name
    return "early"  # turn 0: leads


def _hp(text: str) -> float | None:
    """``"123/260"``, ``"45/100 par"`` or ``"0 fnt"`` -> fraction of max HP."""
    value = text.split(" ")[0]
    if "/" in value:
        current, maximum = value.split("/", 1)
        try:
            return float(current) / float(maximum) if float(maximum) else None
        except ValueError:
            return None
    return 0.0 if value == "0" else None


def _called(args: Sequence[str]) -> bool:
    return any(a.startswith("[from]") and "lockedmove" not in a for a in args)


def battle_metrics(events: Sequence[Sequence[str]], role: str) -> dict[str, Any]:
    """Everything this module measures, for the player on ``role`` ("p1"/"p2")."""
    ours = role
    turn = 0
    active = {"us": None, "them": None}  # current ident on each side
    hp: dict[str, float] = {}
    replacement_due = {"us": True, "them": True}  # leads count as replacements
    our_pivot_this_turn = False
    last_move: dict[str, Any] | None = None  # the move whose effects follow
    used: dict[str, set[str]] = {}  # opposing ident -> moves seen
    boosted_stint: dict[str, bool] = {}
    kos_by: dict[str, int] = {}
    setup_kos: dict[str, int] = {}
    entry: dict[str, Any] | None = None  # our latest voluntary switch-in, until it acts
    faints = {"us": 0, "them": 0}
    stage_kos = {name: {"dealt": 0, "suffered": 0} for name, _, _ in STAGES}
    m: dict[str, Any] = {
        "turns": 0,
        "our_moves": 0,
        "immune": 0,
        "missed": 0,
        "failed": 0,
        "voluntary_switches": 0,
        "switch_fainted_before_acting": 0,
        "switch_big_hit": 0,
        "kos_suffered_by_attack": 0,
        "kos_suffered_first_use": 0,
        "endgame": None,  # "ahead" | "level" | "behind"
        "max_lead": 0,
        "max_deficit": 0,
    }

    def side(ident: str) -> str:
        return "us" if ident[:2] == ours else "them"

    def ident_of(text: str) -> str:
        # "p1a: Garchomp" -> "p1: Garchomp"
        return text[:2] + text[3:] if len(text) > 3 and text[2] != ":" else text

    for event in events:
        if not event:
            continue
        kind, args = event[0], list(event[1:])
        if kind == "turn" and args:
            turn = int(args[0])
            our_pivot_this_turn = False
            last_move = None
            continue
        if not args or len(args[0]) < 3 or args[0][:1] != "p":
            continue
        who = side(args[0])
        mon = ident_of(args[0])

        if kind in ("switch", "drag", "replace"):
            if kind == "replace":
                active[who] = mon
                continue
            if len(args) > 2 and _hp(args[2]) is not None:
                hp[mon] = _hp(args[2])  # type: ignore[assignment]
            if who == "them":
                boosted_stint[mon] = False
            elif kind == "switch" and not replacement_due["us"] and not our_pivot_this_turn and turn > 0:
                m["voluntary_switches"] += 1
                entry = {"mon": mon, "turn": turn, "hp": hp.get(mon, 1.0), "acted": False}
            if kind == "switch" and replacement_due[who]:
                replacement_due[who] = False
            active[who] = mon
            last_move = None
        elif kind == "move" and len(args) > 1:
            move = to_id_str(args[1])
            target = ident_of(args[2]) if len(args) > 2 and args[2].startswith("p") else None
            called = _called(args[2:])
            if who == "us":
                if not called:
                    m["our_moves"] += 1
                if move in PIVOT_MOVES:
                    our_pivot_this_turn = True
                if entry is not None and entry["mon"] == mon:
                    entry["acted"] = True
            else:
                seen = used.setdefault(mon, set())
                first_use = move not in seen and not called
                seen.add(move)
                if move_kind(args[1]) == "setup":
                    boosted_stint[mon] = True
                last_move = {"side": who, "move": move, "user": mon, "target": target, "first_use": first_use}
                continue
            last_move = {"side": who, "move": move, "user": mon, "target": target, "first_use": False}
        elif kind == "-immune":
            if who == "them" and last_move is not None and last_move["side"] == "us":
                m["immune"] += 1
        elif kind == "-miss":
            if who == "us":
                m["missed"] += 1
        elif kind == "-fail":
            if who == "us" and last_move is not None and last_move["side"] == "us":
                m["failed"] += 1
        elif kind in ("-damage", "-heal", "-sethp") and len(args) > 1:
            value = _hp(args[1])
            if value is None:
                continue
            before = hp.get(mon, 1.0)
            hp[mon] = value
            if (
                kind == "-damage"
                and entry is not None
                and entry["mon"] == mon
                and entry["turn"] == turn
                and not entry["acted"]
                and before - value > 0
            ):
                entry["lost"] = entry.get("lost", 0.0) + (before - value)
                if entry["lost"] >= BIG_ENTRY_HIT and not entry.get("counted_big"):
                    entry["counted_big"] = True
                    m["switch_big_hit"] += 1
        elif kind == "faint":
            faints[who] += 1
            stage = _stage(turn)
            if who == "us":
                stage_kos[stage]["suffered"] += 1
                killer = active["them"]
                if killer is not None:
                    kos_by[killer] = kos_by.get(killer, 0) + 1
                    if boosted_stint.get(killer):
                        setup_kos[killer] = setup_kos.get(killer, 0) + 1
                if last_move is not None and last_move["side"] == "them" and last_move["target"] == mon:
                    m["kos_suffered_by_attack"] += 1
                    m["kos_suffered_first_use"] += int(last_move["first_use"])
                if entry is not None and entry["mon"] == mon and not entry["acted"]:
                    m["switch_fainted_before_acting"] += 1
                    entry = None
            else:
                stage_kos[stage]["dealt"] += 1
            replacement_due[who] = True
            lead = faints["them"] - faints["us"]
            m["max_lead"] = max(m["max_lead"], lead)
            m["max_deficit"] = max(m["max_deficit"], -lead)
            ours_left, theirs_left = TEAM_SIZE - faints["us"], TEAM_SIZE - faints["them"]
            if m["endgame"] is None and 0 < ours_left <= ENDGAME_MAX_LEFT and 0 < theirs_left <= ENDGAME_MAX_LEFT:
                m["endgame"] = "ahead" if ours_left > theirs_left else "behind" if ours_left < theirs_left else "level"

    m["turns"] = turn
    m["our_faints"], m["their_faints"] = faints["us"], faints["them"]
    m["stages"] = stage_kos
    m["faced_setup_sweeper"] = bool(setup_kos)
    m["kos_by_setup_sweepers"] = sum(setup_kos.values())
    m["swept"] = max(kos_by.values(), default=0) >= 3
    return m
