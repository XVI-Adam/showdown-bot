"""How strong was each side's team, with skill held equal?

For a recorded battle, rebuild both teams and let the *same* bot play both
sides many times. The share our team wins is that matchup's team strength:
what the dealt teams were worth on their own, before anyone made a choice.
Comparing it with the actual result separates luck of the draw from play.

Rebuilding the teams:

* **Ours** is known exactly from the record (species, level, moves, ability,
  item). Two gaps in older records are filled honestly: an item used up or
  knocked off is recovered from the log, and a Tera type not recorded (and not
  shown in the battle) is sampled from that species' random-battle options.
* **Theirs** keeps everything the log revealed. Each unrevealed move, ability,
  item and Tera type is sampled from the measured set statistics
  (``psbot.opponent_model``), conditioned on what was shown. A Pokémon that
  never appeared is a random species drawn as the generator would. Several
  samples per battle turn hidden information into an uncertainty interval.

Every team member gets the random-battle spread (85 EVs, 31 IVs, neutral
nature) and its recorded level. Teams play on the simulator's Custom Game
format with team preview answered by keeping the original lead first.

Limits: "strength" means strength when this bot plays both teams; a
species' rare set (e.g. 0 Speed IVs for Trick Room) is approximated by the
standard spread; and a Trick that swapped an item can mislead the rebuild.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from poke_env import AccountConfiguration
from poke_env.battle import Pokemon
from poke_env.data import GenData, to_id_str
from poke_env.teambuilder import Teambuilder, TeambuilderPokemon

from psbot.config import local_server_configuration
from psbot.dataset.revealed import RevealTracker
from psbot.heuristics.randbats import GEN, RANDBATS_EV, RANDBATS_IV, _load_sets
from psbot.opponent_model.sets import _measured, set_belief

CUSTOM_FORMAT = "gen9customgame"
MAX_REDRAWS = 20  # attempts to draw a hidden-information sample the validator accepts
_DEX = GenData.from_gen(GEN).pokedex


# ---------------------------------------------------------------- species


def team_species(species: str, ability: str | None = None) -> str:
    """The forme to build: battle-only formes (Palafin-Hero, Mimikyu-Busted,
    Ogerpon's Tera formes) start the battle as their base forme."""
    species = to_id_str(species)
    entry = _DEX.get(species, {})
    if entry.get("battleOnly"):
        species = to_id_str(entry["battleOnly"])
    if species == "greninja" and ability == "battlebond":
        species = "greninjabond"  # shown to both players as Greninja
    return species


def team_ability(species: str, ability: str | None) -> str:
    """The ability id the validator accepts. Battle logs name some abilities
    more loosely than their ids: "As One" is ``asoneglastrier`` or
    ``asonespectrier`` depending on the forme."""
    if not ability:
        return ""
    ability = to_id_str(ability)
    legal = [to_id_str(a) for a in _DEX.get(species, {}).get("abilities", {}).values()]
    if not legal or ability in legal:
        return ability
    return next((a for a in legal if a.startswith(ability)), ability)


def _display(species: str) -> str:
    return _DEX.get(species, {}).get("name", species)


def _tera_options(species: str, moves: Sequence[str]) -> list[str]:
    """Tera types of the random-battle roles this moveset could come from."""
    entry = _load_sets().get(species) or _load_sets().get(to_id_str(_DEX.get(species, {}).get("baseSpecies", "")))
    if not entry:
        return []
    moves = {to_id_str(m) for m in moves}
    roles = [r for r in entry["sets"] if moves <= {to_id_str(x) for x in r["movepool"]}] or entry["sets"]
    return sorted({t for r in roles for t in r.get("teraTypes", [])})


def _mon(species: str, level: int, moves: Sequence[str], ability: str, item: str, tera: str | None) -> TeambuilderPokemon:
    return TeambuilderPokemon(
        nickname=_display(species), species=_display(species), item=item or None,
        ability=team_ability(species, ability) or None,
        moves=list(moves)[:4], nature="Hardy", evs=[RANDBATS_EV] * 6, ivs=[RANDBATS_IV] * 6,
        level=level, tera_type=tera.capitalize() if tera else None,
    )  # fmt: skip


def pack(team: Sequence[TeambuilderPokemon]) -> str:
    return Teambuilder.join_team(list(team))


_VALIDATOR_JS = r"""
const {TeamValidator, Teams} = require('./dist/sim');
const validator = TeamValidator.get(process.argv[1]);
const teams = JSON.parse(require('fs').readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(teams.map(t => validator.validateTeam(Teams.unpack(t)))));
"""


def validate_teams(packed: Sequence[str]) -> list[list[str] | None]:
    """Ask the simulator's own validator about each team: None if it would
    be accepted, otherwise the reasons. One Node process for the whole batch."""
    import subprocess

    from psbot.config import PROJECT_ROOT

    result = subprocess.run(
        ["node", "-e", _VALIDATOR_JS, CUSTOM_FORMAT],
        input=json.dumps(list(packed)), capture_output=True, text=True, timeout=120,
        cwd=PROJECT_ROOT / "vendor" / "pokemon-showdown",
    )  # fmt: skip
    if result.returncode != 0:
        raise RuntimeError(f"team validator failed: {result.stderr[-500:]}")
    return json.loads(result.stdout)


# ------------------------------------------------------------- rebuilding


@dataclass
class RebuiltTeam:
    members: list[TeambuilderPokemon]
    sampled: dict[str, int] = field(default_factory=dict)  # what had to be filled in

    def note(self, what: str) -> None:
        self.sampled[what] = self.sampled.get(what, 0) + 1


def _lead_first(team: list[dict[str, Any]], lead: str | None) -> list[dict[str, Any]]:
    return sorted(team, key=lambda m: m["ident"] != lead)


def _leads(events: Sequence[Sequence[str]]) -> dict[str, str]:
    leads: dict[str, str] = {}
    for event in events:
        if event and event[0] == "switch" and len(event) > 1:
            side, ident = event[1][:2], event[1][:2] + event[1][3:]
            leads.setdefault(side, ident)
        if len(leads) == 2:
            break
    return leads


def rebuild_ours(record: dict[str, Any], rng: random.Random) -> RebuiltTeam:
    tracker = RevealTracker().update(record["events"])
    role = record["role"]
    out = RebuiltTeam([])
    for mon in _lead_first(record["team"], _leads(record["events"]).get(role)):
        shown = tracker.mons.get(mon["ident"])
        item = mon["item"] or (shown.item if shown and shown.item_consumed else None)
        if not mon["item"]:
            out.note("our item recovered from the log" if item else "our item unknown")
        tera = mon.get("tera_type") or (shown.tera_type if shown else None)
        species = team_species(mon["species"], mon["ability"])
        if not tera:
            options = _tera_options(species, mon["moves"])
            tera = rng.choice(options) if options else None
            out.note("our Tera type sampled")
        out.members.append(_mon(species, mon["level"], mon["moves"], mon["ability"] or "", item or "", tera))
    return out


def _random_member(rng: random.Random, taken: set[str]) -> tuple[str, int, tuple]:
    """A species and set as the generator would deal them, excluding species
    already on the team (one of each base species per team)."""
    table = _measured()
    raw = json.loads((__import__("psbot.opponent_model.sets", fromlist=["STATS_PATH"]).STATS_PATH).read_text())["species"]
    while True:
        species = rng.choices(list(raw), weights=[e["n"] for e in raw.values()])[0]
        base = to_id_str(_DEX.get(species, {}).get("baseSpecies", species))
        if base in taken:
            continue
        candidates = table[species]
        chosen = rng.choices(candidates, weights=[c[3] for c in candidates])[0]
        return species, raw[species]["level"], chosen


def sample_theirs(record: dict[str, Any], rng: random.Random) -> RebuiltTeam:
    opponent_role = "p2" if record["role"] == "p1" else "p1"
    out = RebuiltTeam([])
    taken = set()
    for mon in _lead_first(record["opponent_team"], _leads(record["events"]).get(opponent_role)):
        known_ability = mon["ability"]["value"] if mon["ability"]["source"] == "revealed" else None
        known_item = mon["item"]["value"] if mon["item"]["source"] == "revealed" else None
        species = team_species(mon["species"], known_ability)
        taken.add(to_id_str(_DEX.get(species, {}).get("baseSpecies", species)))
        belief = set_belief(Pokemon(gen=GEN, species=species), mon["revealed_moves"], known_ability, known_item)
        if belief.candidates:
            moves, ability, item, _ = rng.choices(belief.candidates, weights=[c[3] for c in belief.candidates])[0]
        else:
            moves, ability, item = frozenset(mon["revealed_moves"]), "", ""
        moves = list(mon["revealed_moves"]) + sorted(set(moves) - set(mon["revealed_moves"]))
        if len(mon["revealed_moves"]) < 4:
            out.note("their moves sampled")
        if not known_ability:
            out.note("their ability sampled")
        if not known_item:
            out.note("their item sampled")
        tera = mon["tera_type"]["value"]
        if not tera:
            options = _tera_options(species, moves)
            tera = rng.choice(options) if options else None
            out.note("their Tera type sampled")
        out.members.append(_mon(species, mon["level"], moves, known_ability or ability, known_item or item, tera))
    while len(out.members) < 6:
        species, level, (moves, ability, item, _) = _random_member(rng, taken)
        taken.add(to_id_str(_DEX.get(species, {}).get("baseSpecies", species)))
        options = _tera_options(species, moves)
        out.members.append(_mon(species, level, sorted(moves), ability, item, rng.choice(options) if options else None))
        out.note("their unseen Pokémon sampled")
    return out


# --------------------------------------------------------------- replaying


class Replayer:
    """Two copies of one agent that play given teams against each other on
    the local simulator. Reused across matchups to avoid logging in again."""

    def __init__(self, agent: str, concurrency: int = 10) -> None:
        from psbot.opponents import make_opponent

        common = dict(
            battle_format=CUSTOM_FORMAT,
            server_configuration=local_server_configuration(),
            max_concurrent_battles=concurrency,
            team=pack([_mon("pikachu", 50, ["thunderbolt"], "static", "", None)]),  # replaced per matchup
        )
        self.a = make_opponent(agent, account_configuration=AccountConfiguration.generate("TeamA", rand=True), **common)
        self.b = make_opponent(agent, account_configuration=AccountConfiguration.generate("TeamB", rand=True), **common)
        for player in (self.a, self.b):
            player.teampreview = lambda battle: "/team 123456"  # keep the recorded lead first
        self.fallbacks = 0
        self.abandoned = 0
        self.timeout_per_battle = 20.0

    async def play(self, team_a: str, team_b: str, n: int) -> tuple[float, int]:
        """Score of team A over ``n`` battles (a tie counts half), and battles played."""
        self.a.update_team(team_a)
        self.b.update_team(team_b)
        self.a.reset_battles()
        self.b.reset_battles()
        try:
            await asyncio.wait_for(self.a.battle_against(self.b, n_battles=n), timeout=self.timeout_per_battle * n)
        except asyncio.TimeoutError:
            # A battle neither bot can end (Custom Game has no endless-battle
            # clause). Count what finished; report the rest as abandoned.
            self.abandoned += sum(1 for b in self.a.battles.values() if not b.finished)
        battles = [b for b in self.a.battles.values() if b.finished]
        score = sum(1.0 if b.won else 0.5 if b.won is None else 0.0 for b in battles)
        self.fallbacks = getattr(self.a, "n_fallback_decisions", 0) + getattr(self.b, "n_fallback_decisions", 0)
        return score, len(battles)

    async def close(self) -> None:
        for player in (self.a, self.b):
            await player.ps_client.stop_listening()


# -------------------------------------------------------------- estimating


def _wilson(k: float, n: int) -> tuple[float, float]:
    from psbot.runner import wilson_interval

    return wilson_interval(round(k), n)


async def estimate(
    record: dict[str, Any],
    replayer: Replayer,
    samples: int = 10,
    per_sample: int = 30,
    seed: int = 0,
    their_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Team strength of one recorded battle: our team's share of wins.

    Each sample redraws the hidden parts of both teams and plays
    ``per_sample`` battles. With ``their_record`` (the opponent's own record of
    the same battle, available only for bot battles) their team is rebuilt
    exactly instead of sampled.
    """
    rng = random.Random(f"{seed}:{record['battle_id']}")
    matchups, filled, rejected = [], {}, []

    def draw() -> tuple[RebuiltTeam, RebuiltTeam]:
        ours = rebuild_ours(record, rng)
        theirs = rebuild_ours(their_record, rng) if their_record is not None else sample_theirs(record, rng)
        return ours, theirs

    for _ in range(samples):
        for attempt in range(MAX_REDRAWS):
            ours, theirs = draw()
            problems = validate_teams([pack(ours.members), pack(theirs.members)])
            if not any(problems):
                break
            rejected.append("; ".join(p for side in problems if side for p in side)[:200])
        else:
            continue  # this sample could not be made valid: leave it out, and say so
        for part in (ours, theirs):
            for key, count in part.sampled.items():
                filled[key] = filled.get(key, 0) + count
        matchups.append((pack(ours.members), pack(theirs.members)))
    scores = [await replayer.play(a, b, per_sample) for a, b in matchups]
    result = summarise_samples(record["battle_id"], scores, filled, max(len(matchups), 1), seed)
    result["rejected_rebuilds"] = rejected
    result["abandoned_replays"] = replayer.abandoned
    return result


def summarise_samples(battle_id: str, scores: list[tuple[float, int]], filled: dict[str, int], samples: int, seed: int) -> dict[str, Any]:
    won, played = sum(s for s, _ in scores), sum(n for _, n in scores)
    low, high = bootstrap_interval(scores, seed)
    return {
        "battle_id": battle_id,
        "strength": won / played if played else None,
        "ci": [low, high],
        "battles": played,
        "per_sample": [round(s / n, 3) if n else None for s, n in scores],
        "filled_in": {k: round(v / samples, 1) for k, v in filled.items()},  # per rebuild
    }


def bootstrap_interval(scores: list[tuple[float, int]], seed: int = 0, n_boot: int = 2000) -> tuple[float, float]:
    """95% interval for the win share, counting both the battle-to-battle
    noise and the spread between hidden-information samples: resample the
    samples, then each sample's battles."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rates = np.array([s / n for s, n in scores if n])
    sizes = np.array([n for _, n in scores if n])
    if not len(rates):
        return float("nan"), float("nan")
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(rates), len(rates))
        wins = rng.binomial(sizes[pick], rates[pick])
        draws.append(wins.sum() / sizes[pick].sum())
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def chance_of_at_most(wins: int, strengths: Sequence[float]) -> float:
    """P(at most ``wins`` wins) when each battle is won with its own
    probability: how likely the result is if team strength alone decided."""
    distribution = [1.0]
    for p in strengths:
        nxt = [0.0] * (len(distribution) + 1)
        for k, mass in enumerate(distribution):
            nxt[k] += mass * (1 - p)
            nxt[k + 1] += mass * p
        distribution = nxt
    return sum(distribution[: wins + 1])


# ------------------------------------------------------------ calibration


def fit_calibration(outcomes: Sequence[bool], strengths: Sequence[float]) -> dict[str, float]:
    """Map raw team strength to an actual win chance (Platt scaling).

    Fitted on battles whose outcome is known: ``P(win) = sigmoid(a + b *
    logit(strength))``. A slope ``b`` below 1 means raw strengths are too
    extreme: the replays exaggerate how much the teams decide, because real
    players (and the dice) decide more.
    """
    import numpy as np

    y = np.array([1.0 if o else 0.0 for o in outcomes])
    p = np.clip(np.array(strengths, dtype=float), 0.02, 0.98)
    x = np.log(p / (1 - p))
    a, b = 0.0, 1.0
    for _ in range(100):
        q = 1 / (1 + np.exp(-(a + b * x)))
        w = q * (1 - q)
        hessian = np.array([[w.sum(), (w * x).sum()], [(w * x).sum(), (w * x * x).sum()]])
        step = np.linalg.solve(hessian + 1e-9 * np.eye(2), np.array([(q - y).sum(), ((q - y) * x).sum()]))
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    se = float(np.sqrt(np.linalg.inv(hessian)[1, 1]))
    return {"a": float(a), "b": float(b), "b_se": se, "n": int(len(y))}


def calibrated(strength: float, calibration: dict[str, float] | None) -> float:
    if calibration is None:
        return strength
    import math

    s = min(max(strength, 0.02), 0.98)
    return 1 / (1 + math.exp(-(calibration["a"] + calibration["b"] * math.log(s / (1 - s)))))
