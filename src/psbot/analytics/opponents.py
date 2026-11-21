"""Per-opponent profiles from recorded human battles.

Everything here is counted from the public battle log: which Pokémon someone
led with, which moves they clicked, when they switched. Nothing is assumed
about moves, items or abilities that were never shown.

Small samples are the main hazard. One battle says almost nothing about a
person: the teams are random, so most of what happens is the matchup. So:

* a profile is ``insufficient`` below :data:`MIN_BATTLES_FOR_PROFILE` battles
  and then reports raw counts only, with no tendencies and no conclusions;
* every rate carries its count and a 95% Wilson interval;
* a tendency is stated only when its interval clears the threshold it is
  compared with, never because the point estimate does.

Reads both record layouts: the Phase 4 dataset (``data/human``) and the older
logs (``logs/human``), which lack some event types; metrics that need those
are then reported as unavailable rather than as zero.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from poke_env.data import to_id_str

from psbot.analysis import move_kind
from psbot.runner import wilson_interval

MIN_BATTLES_FOR_PROFILE = 5  # below this: counts only
MODERATE_BATTLES = 15  # below this: "low" confidence
MIN_EVENTS_FOR_RATE = 20  # a rate over fewer observations is not interpreted

PIVOT_MOVES = frozenset(
    {"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "batonpass", "chillyreception", "shedtail"}
)  # fmt: skip


@dataclass
class BattleView:
    """One battle, reduced to what the analytics need."""

    battle_id: str
    opponent: str
    role: str  # ours: "p1" or "p2"
    won: bool | None
    events: list[list[str]]
    full_log: bool  # False for the older layout, which kept fewer event types
    agent: str = "?"


def load_battles(*paths: Path) -> Iterator[BattleView]:
    """Yield every battle in the given files or directories (``*.jsonl``)."""
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for file in sorted(path.glob("*.jsonl")) if path.is_dir() else [path]:
            with file.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    view = _view(record)
                    if view is not None and view.battle_id not in seen:
                        seen.add(view.battle_id)
                        yield view


def _view(record: dict[str, Any]) -> BattleView | None:
    if str(record.get("schema", "")).startswith("psbot.human_battle/"):
        return BattleView(
            battle_id=record["battle_id"],
            opponent=record["opponent"]["username"],
            role=record["role"] or "p1",
            won=record["outcome"]["won"],
            events=record["events"],
            full_log=True,
            agent=record["agent"]["name"],
        )
    if "battle_tag" in record and "events" in record:
        return BattleView(
            battle_id=record["battle_tag"],
            opponent=record.get("opponent") or "?",
            role=record.get("role") or "p1",
            won=record.get("won"),
            events=record["events"],
            full_log=False,
            agent=record.get("player_kind") or "?",
        )
    return None


# ------------------------------------------------------------- one battle


@dataclass
class _Tally:
    """Counts accumulated over an opponent's battles."""

    battles: int = 0
    full_log_battles: int = 0
    our_wins: int = 0
    turns: int = 0
    leads: Counter = field(default_factory=Counter)
    moves: Counter = field(default_factory=Counter)
    move_kinds: Counter = field(default_factory=Counter)
    actions: int = 0  # moves + voluntary switches: turns where they chose something
    voluntary_switches: int = 0
    switch_pairs: Counter = field(default_factory=Counter)  # (out, in)
    switched_out: Counter = field(default_factory=Counter)
    setup_battles: int = 0
    repeat_opportunities: int = 0
    repeats: int = 0
    per_mon_moves: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    # mistakes (attacks only count where the log can show them)
    attacks: int = 0
    attacks_full_log: int = 0
    resisted: int = 0
    immune: int = 0
    failed: int = 0
    switch_ins: int = 0
    fainted_on_entry_turn: int = 0
    # trouble for us
    kos_by: Counter = field(default_factory=Counter)
    appearances: Counter = field(default_factory=Counter)
    losses_with: Counter = field(default_factory=Counter)


def _tally_battle(tally: _Tally, battle: BattleView) -> None:
    ours = battle.role
    tally.battles += 1
    tally.full_log_battles += int(battle.full_log)
    tally.our_wins += int(battle.won is True)

    active = {"us": "", "them": ""}
    turn = 0
    lead_seen = False
    replacement_due = False  # their active fainted: the next switch is forced
    entered_turn: dict[str, int] = {}
    last_move: tuple[str, str] | None = None  # (species, move) of their previous action
    used_setup = False
    seen_species: set[str] = set()
    pending_attack = False  # their last event was an attack; flags may follow

    for event in battle.events:
        kind = event[0]
        if kind == "turn" and len(event) > 1:
            turn = int(event[1])
            pending_attack = False
            continue
        if len(event) < 2:
            continue
        side = "us" if event[1][:2] == ours else "them"
        if kind in ("switch", "drag") and len(event) > 2:
            species = to_id_str(event[2].split(",")[0])
            pending_attack = False
            if side == "us":
                active["us"] = species
                continue
            previous = active["them"]
            active["them"] = species
            seen_species.add(species)
            entered_turn[species] = turn
            tally.switch_ins += 1
            if not lead_seen:
                lead_seen = True
                tally.leads[species] += 1
            elif replacement_due:
                replacement_due = False
            elif kind == "switch":
                if last_move is not None and last_move[1] in PIVOT_MOVES and last_move[0] == previous:
                    pass  # the switch is part of the pivot move already counted
                else:
                    tally.voluntary_switches += 1
                    tally.actions += 1
                    tally.switch_pairs[(previous, species)] += 1
                    tally.switched_out[previous] += 1
                    last_move = None
        elif kind == "move" and len(event) > 2:
            if side == "us":
                pending_attack = False
                continue
            if any(a.startswith("[from]") and "lockedmove" not in a for a in event[3:]):
                continue  # called by another move or ability: not a choice
            move = to_id_str(event[2])
            kind_of_move = "pivot" if move in PIVOT_MOVES else move_kind(event[2])
            tally.moves[move] += 1
            tally.move_kinds[kind_of_move] += 1
            tally.per_mon_moves[active["them"]][move] += 1
            tally.actions += 1
            if last_move is not None and last_move[0] == active["them"]:
                tally.repeat_opportunities += 1
                tally.repeats += int(last_move[1] == move)
            last_move = (active["them"], move)
            used_setup = used_setup or kind_of_move == "setup"
            pending_attack = kind_of_move in ("attack", "pivot")
            if pending_attack:
                tally.attacks += 1
                tally.attacks_full_log += int(battle.full_log)
        elif kind == "-resisted" and side == "us" and pending_attack:
            tally.resisted += 1
        elif kind == "-immune" and side == "us" and pending_attack:
            tally.immune += 1
        elif kind == "-fail" and side == "them":
            tally.failed += 1
        elif kind == "faint":
            pending_attack = False
            if side == "them":
                replacement_due = True
                if entered_turn.get(active["them"]) == turn and turn > 0:
                    tally.fainted_on_entry_turn += 1
            else:
                tally.kos_by[active["them"]] += 1
    tally.turns += turn
    tally.setup_battles += int(used_setup)
    for species in seen_species:
        tally.appearances[species] += 1
        tally.losses_with[species] += int(battle.won is False)


# --------------------------------------------------------------- profile


def _rate(successes: int, n: int) -> dict[str, Any]:
    """A proportion with its sample size and 95% interval."""
    low, high = wilson_interval(successes, n)
    return {
        "value": round(successes / n, 3) if n else None,
        "count": successes,
        "n": n,
        "ci95": [round(low, 3), round(high, 3)],
    }


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    return -sum(c / total * math.log2(c / total) for c in counter.values() if c)


def _predictability(per_mon: dict[str, Counter]) -> dict[str, Any]:
    """1 minus the normalised entropy of move choice, averaged over Pokémon
    seen making at least four choices among at least two moves. 1 means the
    same move every time; 0 means all known moves used equally."""
    scores, choices = [], 0
    for moves in per_mon.values():
        n = sum(moves.values())
        if n >= 4 and len(moves) >= 2:
            scores.append(1 - _entropy(moves) / math.log2(len(moves)))
            choices += n
    return {
        "value": round(sum(scores) / len(scores), 3) if scores else None,
        "pokemon": len(scores),
        "choices": choices,
    }


def confidence(n_battles: int) -> str:
    if n_battles < MIN_BATTLES_FOR_PROFILE:
        return "insufficient"
    return "low" if n_battles < MODERATE_BATTLES else "moderate"


def _tendencies(profile: dict[str, Any]) -> list[str]:
    """Statements the data supports: each needs enough observations and an
    interval that lies wholly on one side of its threshold."""
    found: list[str] = []

    def clear(rate: dict[str, Any], above: float | None = None, below: float | None = None) -> bool:
        if rate["n"] < MIN_EVENTS_FOR_RATE:
            return False
        low, high = rate["ci95"]
        return (above is not None and low > above) or (below is not None and high < below)

    aggression, switching = profile["aggression"], profile["switching"]["rate_per_action"]
    if clear(aggression, above=0.75):
        found.append(f"attacks with most choices ({aggression['value']:.0%} of {aggression['n']})")
    elif clear(aggression, below=0.55):
        found.append(f"plays defensively: under 55% of choices are attacks ({aggression['value']:.0%} of {aggression['n']})")
    if clear(switching, above=0.25):
        found.append(f"switches often ({switching['value']:.0%} of {switching['n']} choices)")
    elif clear(switching, below=0.08):
        found.append(f"rarely switches ({switching['value']:.0%} of {switching['n']} choices)")
    setup = profile["setup"]["rate_per_action"]
    if clear(setup, above=0.10):
        found.append(f"uses setup moves often ({setup['value']:.0%} of {setup['n']} choices)")
    repeat = profile["predictability"]["repeat_rate"]
    if clear(repeat, above=0.65):
        found.append(f"usually repeats the previous move ({repeat['value']:.0%} of {repeat['n']})")
    mistakes = profile["mistakes"]
    for key, label, threshold in (
        ("resisted_attacks", "often attacks into a resistance", 0.25),
        ("immune_attacks", "often attacks into an immunity", 0.05),
    ):
        rate = mistakes[key]
        if rate is not None and clear(rate, above=threshold):
            found.append(f"{label} ({rate['value']:.0%} of {rate['n']} attacks)")
    return found


def build_profile(opponent: str, battles: Iterable[BattleView]) -> dict[str, Any]:
    """Summarise one opponent. With fewer than :data:`MIN_BATTLES_FOR_PROFILE`
    battles the result holds counts only."""
    tally = _Tally()
    battles = list(battles)
    for battle in battles:
        _tally_battle(tally, battle)

    level = confidence(tally.battles)
    profile: dict[str, Any] = {
        "opponent": opponent,
        "battles": tally.battles,
        "confidence": level,
        "our_record": {"wins": tally.our_wins, "losses_or_ties": tally.battles - tally.our_wins},
    }
    # Leads, favourite moves and which species beat ours are not reported:
    # with random teams they describe the teams dealt, not the player.
    if level == "insufficient":
        profile["note"] = (
            f"{tally.battles} battle(s): too few to describe how this person plays. "
            f"Counts only; tendencies need at least {MIN_BATTLES_FOR_PROFILE}."
        )
        return profile

    kinds, actions = tally.move_kinds, tally.actions
    full = tally.full_log_battles == tally.battles
    profile.update(
        {
            "our_win_rate": _rate(tally.our_wins, tally.battles),
            "avg_turns": round(tally.turns / tally.battles, 1),
            "action_mix": {kind: _rate(count, actions) for kind, count in kinds.most_common()},
            "aggression": _rate(kinds["attack"] + kinds["pivot"], actions),
            "switching": {
                "rate_per_action": _rate(tally.voluntary_switches, actions),
                "per_battle": round(tally.voluntary_switches / tally.battles, 2),
            },
            "setup": {
                "rate_per_action": _rate(kinds["setup"], actions),
                "battles_with_setup": _rate(tally.setup_battles, tally.battles),
            },
            "predictability": {
                "repeat_rate": _rate(tally.repeats, tally.repeat_opportunities),
                "move_choice": _predictability(tally.per_mon_moves),
            },
            "mistakes": {
                # Resisted/immune flags exist only in full logs.
                "resisted_attacks": _rate(tally.resisted, tally.attacks_full_log) if tally.attacks_full_log else None,
                "immune_attacks": _rate(tally.immune, tally.attacks_full_log) if tally.attacks_full_log else None,
                "failed_moves": _rate(tally.failed, sum(tally.moves.values())),
                "fainted_the_turn_it_switched_in": _rate(tally.fainted_on_entry_turn, tally.switch_ins),
                "complete": full,
            },
        }
    )
    profile["tendencies"] = _tendencies(profile)
    if not profile["tendencies"]:
        profile["note"] = "No tendency is clear of its threshold at this sample size."
    return profile


def build_profiles(battles: Iterable[BattleView]) -> dict[str, dict[str, Any]]:
    """One profile per opponent, plus ``"*"`` for all human battles together."""
    by_opponent: dict[str, list[BattleView]] = defaultdict(list)
    everything: list[BattleView] = []
    for battle in battles:
        by_opponent[battle.opponent].append(battle)
        everything.append(battle)
    profiles = {name: build_profile(name, games) for name, games in sorted(by_opponent.items())}
    if everything:
        profiles["*"] = build_profile("all opponents", everything)
    return profiles


# ---------------------------------------------------------------- report


def _fmt(rate: dict[str, Any] | None) -> str:
    if rate is None:
        return "not available (older log format)"
    if not rate["n"]:
        return "no observations"
    low, high = rate["ci95"]
    return f"{rate['value']:.0%} ({rate['count']}/{rate['n']}; 95% interval {low:.0%}-{high:.0%})"


def render_profile(profile: dict[str, Any]) -> str:
    """A profile as Markdown."""
    lines = [
        f"# Opponent profile: {profile['opponent']}",
        "",
        f"Battles: {profile['battles']} | Confidence: **{profile['confidence']}** | "
        f"Our record: {profile['our_record']['wins']}-{profile['our_record']['losses_or_ties']}",
        "",
    ]
    if "note" in profile:
        lines += [f"> {profile['note']}", ""]

    if profile["confidence"] == "insufficient":
        return "\n".join(lines) + "\n"

    lines += ["", "## Tendencies supported by the data", ""]
    lines += [f"- {t}" for t in profile["tendencies"]] or ["- none yet"]
    switching, setup, predict, mistakes = (
        profile["switching"], profile["setup"], profile["predictability"], profile["mistakes"],
    )  # fmt: skip
    lines += [
        "",
        "## Measurements",
        "",
        "| Measure | Value |",
        "| --- | --- |",
        f"| Our win rate | {_fmt(profile['our_win_rate'])} |",
        f"| Average turns | {profile['avg_turns']} |",
        f"| Aggression (attacks / choices) | {_fmt(profile['aggression'])} |",
        f"| Voluntary switches / choices | {_fmt(switching['rate_per_action'])} |",
        f"| Setup moves / choices | {_fmt(setup['rate_per_action'])} |",
        f"| Battles with a setup move | {_fmt(setup['battles_with_setup'])} |",
        f"| Repeats previous move | {_fmt(predict['repeat_rate'])} |",
        f"| Move-choice predictability (0-1) | {predict['move_choice']['value']} over {predict['move_choice']['pokemon']} Pokémon |",
        f"| Attacks into a resistance | {_fmt(mistakes['resisted_attacks'])} |",
        f"| Attacks into an immunity | {_fmt(mistakes['immune_attacks'])} |",
        f"| Moves that failed | {_fmt(mistakes['failed_moves'])} |",
        f"| Fainted the turn it switched in | {_fmt(mistakes['fainted_the_turn_it_switched_in'])} |",
        "",
        "Action mix: " + ", ".join(f"{k} {v['value']:.0%}" for k, v in profile["action_mix"].items()),
    ]
    return "\n".join(lines) + "\n"
