"""How an opponent plays, measured so that the random teams matter as little
as possible.

Species-level counts (their lead, their favourite move, which Pokémon beat
ours) mostly describe the teams they happened to be dealt. These measures
describe decisions instead, each one in the situation it was made:

matchup     at each free choice, compare the two Pokémon on the field.
            *Bad* for them: some damaging move of ours hits their Pokémon
            super-effectively and their Pokémon's own types do not threaten
            ours. *Good* for them: the reverse. Otherwise *even*. Types only
            (no Tera, no unrevealed moves), so it is a rough but fair yardstick.
switching   do they leave bad matchups and stay in good ones?
reads       when they switched into our attack, did the Pokémon coming in
            resist it or take nothing from it?
setup       how often they set up, and whether our Pokémon could threaten
            theirs at that moment (super-effectively)
Tera        in how many games, and how early
closing     after getting two Pokémon ahead, did they win? and comebacks
            after being two behind
mistakes    their attacks into an immunity

Only Phase 4 records (data/human) carry the per-turn context these need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from poke_env.battle import Move, MoveCategory, Pokemon
from poke_env.data import GenData, to_id_str

from psbot.analysis import move_kind
from psbot.analytics.opponents import PIVOT_MOVES
from psbot.evaluation.metrics import battle_metrics

_CHART = GenData.from_gen(9).type_chart


@lru_cache(maxsize=None)
def _types(species: str) -> tuple:
    try:
        return tuple(t for t in Pokemon(gen=9, species=species).types if t is not None)
    except Exception:
        return ()


@lru_cache(maxsize=None)
def _move(move_id: str) -> Move | None:
    try:
        move = Move(move_id, gen=9)
        return move if move.category != MoveCategory.STATUS else None
    except Exception:
        return None


def effectiveness(move_id: str, species: str) -> float | None:
    """Type effectiveness of a damaging move on a species, or None."""
    move, types = _move(to_id_str(move_id)), _types(species)
    if move is None or not types:
        return None
    return move.type.damage_multiplier(*types, type_chart=_CHART)


def stab_threat(attacker: str, defender: str) -> float:
    """Best effectiveness of the attacker's own types against the defender."""
    defending = _types(defender)
    if not defending:
        return 1.0
    return max((t.damage_multiplier(*defending, type_chart=_CHART) for t in _types(attacker)), default=1.0)


def matchup(our_species: str, our_moves: list[str], their_species: str) -> str:
    ours = [e for e in (effectiveness(m, their_species) for m in our_moves) if e is not None]
    we_threaten = max(ours, default=1.0) >= 2
    they_threaten = stab_threat(their_species, our_species) >= 2
    if we_threaten and not they_threaten:
        return "bad"
    if they_threaten and not we_threaten:
        return "good"
    return "even"


@dataclass
class StyleTally:
    battles: int = 0
    wins: int = 0  # theirs
    choices: dict[str, dict[str, int]] = field(
        default_factory=lambda: {c: {"switch": 0, "attack": 0, "support": 0} for c in ("bad", "even", "good")}
    )
    switches_into_attack: int = 0
    reads: int = 0  # ...where the incoming Pokémon resisted or was immune
    setup: int = 0
    setup_unthreatened: int = 0
    tera_battles: int = 0
    tera_turns: list[int] = field(default_factory=list)
    led_by_2: int = 0
    led_by_2_won: int = 0
    trailed_by_2: int = 0
    trailed_by_2_won: int = 0
    their_moves: int = 0
    into_immunity: int = 0
    # how their wins happened (bot's view)
    wins_by_setup_sweep: int = 0
    wins_by_sweep: int = 0

    def add(self, other: "StyleTally") -> None:
        for key, value in vars(other).items():
            if key == "choices":
                for c, kinds in value.items():
                    for k, n in kinds.items():
                        self.choices[c][k] += n
            elif isinstance(value, list):
                getattr(self, key).extend(value)
            else:
                setattr(self, key, getattr(self, key) + value)

    # convenient totals
    def total(self, context: str | None = None) -> int:
        contexts = [context] if context else list(self.choices)
        return sum(sum(self.choices[c].values()) for c in contexts)

    def kind(self, kind: str, context: str | None = None) -> int:
        contexts = [context] if context else list(self.choices)
        return sum(self.choices[c][kind] for c in contexts)


def _choice(action: dict[str, Any]) -> str | None:
    if action["type"] == "switch":
        return "switch"
    if action["type"] != "move" or action.get("called"):
        return None
    if action["id"] in PIVOT_MOVES or move_kind(action["id"]) == "attack":
        return "attack"
    return "support"


def tally_battle(record: dict[str, Any]) -> StyleTally:
    """One Phase 4 record, from the opponent's side."""
    t = StyleTally(battles=1)
    ours_won = record["outcome"]["won"]
    t.wins = int(ours_won is False)
    for decision in record["decisions"]:
        theirs, ours = decision.get("opponent_active"), decision.get("active")
        if not theirs or theirs["fainted"] or theirs["hp"] <= 0 or not ours or ours["fainted"]:
            continue  # their next switch is a forced replacement
        context = matchup(ours["species"], ours.get("moves", []), theirs["species"])
        for action in decision.get("opponent_actions", []):
            if action["type"] in ("drag", "cant"):
                break
            kind = _choice(action)
            if kind is None:
                continue
            t.choices[context][kind] += 1
            our_action = decision["selected"]["action"]
            if kind == "switch" and our_action.startswith("move "):
                our_move = our_action.split()[1]
                hit = effectiveness(our_move, action["id"])
                if hit is not None:
                    t.switches_into_attack += 1
                    t.reads += int(hit < 1)
            if kind == "support" and move_kind(action["id"]) == "setup":
                t.setup += 1
                ours_best = max((e for e in (effectiveness(m, theirs["species"]) for m in ours.get("moves", [])) if e is not None), default=1.0)
                t.setup_unthreatened += int(ours_best < 2)
            break  # one choice per turn
    their_role = "p1" if record["role"] == "p2" else "p2"
    for event in record["events"]:
        if event and event[0] == "-terastallize" and len(event) > 1 and event[1].startswith(their_role):
            t.tera_battles = 1
    turn = 0
    for event in record["events"]:
        if event and event[0] == "turn":
            turn = int(event[1])
        elif event and event[0] == "-terastallize" and event[1].startswith(their_role):
            t.tera_turns.append(turn)
            break
    theirs_view = battle_metrics(record["events"], their_role)
    t.led_by_2 = int(theirs_view["max_lead"] >= 2)
    t.led_by_2_won = int(t.led_by_2 and t.wins)
    t.trailed_by_2 = int(theirs_view["max_deficit"] >= 2)
    t.trailed_by_2_won = int(t.trailed_by_2 and t.wins)
    t.their_moves = theirs_view["our_moves"]
    t.into_immunity = theirs_view["immune"]
    if t.wins:
        ours_view = battle_metrics(record["events"], record["role"])
        t.wins_by_setup_sweep = int(ours_view["faced_setup_sweeper"])
        t.wins_by_sweep = int(ours_view["swept"])
    return t


def tally(records: list[dict[str, Any]]) -> StyleTally:
    total = StyleTally()
    for record in records:
        total.add(tally_battle(record))
    return total
