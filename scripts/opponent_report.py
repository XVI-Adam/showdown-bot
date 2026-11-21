"""Opponent ratings, tier list and shareable player cards.

Usage:
    uv run python scripts/opponent_report.py                  # tier list in the terminal
    uv run python scripts/opponent_report.py --write          # also write the pages
    uv run python scripts/opponent_report.py --write --name-others   # cards show other players' names too
    uv run python scripts/opponent_report.py --write --exclude MyOwnAccount

Writes to data/opponent_reports/ (git-ignored, since it names real people):
    tier-list.html       everyone, with names: for you
    <player>.html        one card per player, ready to share; other players on
                         its tier list are anonymous unless --name-others
    analyst.md           the numbers behind it all, with intervals

Ratings are adjusted for team luck using reports from
scripts/team_strength_report.py; run that first after new battles (battles not
yet measured count as an even draw, and the cards say so).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from poke_env.data import to_id_str

from psbot.analytics.cards import Game, Player, player_card, play_style, something_to_try, tier_list_page
from psbot.analytics.rating import rate
from psbot.analytics.style import tally
from psbot.analytics.team_strength import calibrated, fit_calibration
from psbot.config import LOG_DIR
from psbot.dataset.human import DATA_DIR, HUMAN_DIR
from psbot.dataset.schema import load_dataset
from psbot.evaluation.summary import rate as proportion

OUT = DATA_DIR / "opponent_reports"
STRENGTH = DATA_DIR / "team_strength"


def team_chances() -> tuple[dict[str, float], bool]:
    """Battle id -> the *opponent's* calibrated team chance, and whether a
    calibration was available."""
    if not (STRENGTH / "human.json").exists():
        return {}, False
    cache = json.loads((STRENGTH / "human.json").read_text())
    calibration = None
    if (STRENGTH / "validation.json").exists():
        battles = json.loads((STRENGTH / "validation.json").read_text())["battles"]
        calibration = fit_calibration([b["won"] for b in battles], [b["sampled"] for b in battles])
    chances = {key.split(":")[-1]: 1 - calibrated(v["strength"], calibration) for key, v in cache.items() if v["strength"] is not None}
    return chances, calibration is not None


def load_players(exclude: frozenset[str] = frozenset()) -> tuple[list[Player], int]:
    chances, _ = team_chances()
    games: dict[str, list[Game]] = defaultdict(list)
    records: dict[str, list[dict]] = defaultdict(list)
    bot_name = "the bot"
    unmeasured = 0
    if HUMAN_DIR.exists():
        for _, record in load_dataset(HUMAN_DIR):
            name = record["opponent"]["username"]
            bot_name = record["agent"]["username"] or bot_name
            won = None if record["outcome"]["won"] is None else not record["outcome"]["won"]
            chance = chances.get(record["battle_id"])
            unmeasured += chance is None
            games[name].append(Game((record["started_at"] or "")[:10], won, chance))
            records[name].append(record)
    legacy = LOG_DIR / "human"
    if legacy.exists():  # older records: the result only
        for file in sorted(legacy.glob("*.jsonl")):
            day = re.search(r"(\d{4})(\d{2})(\d{2})", file.name)
            for line in file.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    name = record.get("opponent") or "?"
                    won = None if record.get("won") is None else not record["won"]
                    games[name].append(Game("-".join(day.groups()) if day else "?", won, None))
    players = []
    for name, played in games.items():
        if to_id_str(name) in exclude:
            continue
        rating = rate([(g.won is True, g.team_chance) for g in played])
        style = tally(records[name]) if records[name] else None
        players.append(Player(name, rating, played, style, bot_name))
    return sorted(players, key=lambda p: -p.rating.rating), unmeasured


def _fmt(r: dict) -> str:
    if not r["n"]:
        return "-"
    return f"{r['value']:.0%} ({r['k']}/{r['n']}; 95% {r['ci'][0]:.0%}-{r['ci'][1]:.0%})"


def analyst(players: list[Player]) -> str:
    lines = ["# Opponents: the numbers", "",
             "Ratings: team-adjusted, bot = 1500, prior SD 1 logit. Intervals 95%. Private: names real people.", ""]  # fmt: skip
    for p in players:
        r, s = p.rating, p.style
        lines += [f"## {p.name}", "",
                  f"- Rating {r.rating} (95% {r.interval[0]}-{r.interval[1]}), power level {r.power_level}, tier {r.tier}"
                  f"{' (provisional)' if r.provisional else ''}; {r.wins}-{r.battles - r.wins} vs the bot; "
                  f"team strength measured for {r.measured} of {r.battles} games",
                  "- Their team chance per game: " + ", ".join(
                      f"{g.day} {'W' if g.won else 'L'} {'?' if g.team_chance is None else f'{g.team_chance:.0%}'}" for g in p.games)]  # fmt: skip
        if s is not None and s.total():
            lines += [
                f"- Switched out of bad matchups: {_fmt(proportion(s.kind('switch', 'bad'), s.total('bad')))}",
                f"- Stayed and attacked in good matchups: {_fmt(proportion(s.kind('attack', 'good'), s.total('good')))}",
                f"- Switch-ins that resisted our attack: {_fmt(proportion(s.reads, s.switches_into_attack))}",
                f"- Setup moves: {s.setup} ({s.setup_unthreatened} unthreatened); Tera in {s.tera_battles}/{s.battles} games, turns {sorted(s.tera_turns)}",
                f"- Converted a 2-Pokémon lead: {s.led_by_2_won}/{s.led_by_2}; came back from 2 down: {s.trailed_by_2_won}/{s.trailed_by_2}",
                f"- Attacks into an immunity: {_fmt(proportion(s.into_immunity, s.their_moves))}",
                f"- Choice mix: {_fmt(proportion(s.kind('attack'), s.total()))} attacks, {_fmt(proportion(s.kind('switch'), s.total()))} switches",
            ]
        lines += [f"- Card says to try: {something_to_try(p)}", ""]
    return "\n".join(lines)


def main(args: argparse.Namespace) -> int:
    players, unmeasured = load_players(frozenset(to_id_str(n) for n in args.exclude))
    if not players:
        print("no human battles recorded yet")
        return 0
    print(f"{len(players)} opponent(s), {sum(p.rating.battles for p in players)} battle(s) against the bot\n")
    print(f"  {'tier':4} {'player':20} {'rating':>6}  {'likely':>11}  {'power':>5}  {'W-L':>5}")
    for p in players:
        r = p.rating
        print(f"  {r.tier + ('?' if r.provisional else ''):4} {p.name[:20]:20} {r.rating:>6}  "
              f"{r.interval[0]:>5}-{r.interval[1]:<5}  {r.power_level:>5}  {r.wins:>2}-{r.battles - r.wins:<2}")  # fmt: skip
    print(f"\n  ? = provisional (fewer than 5 games). Bot = 1500.")
    if unmeasured:
        print(f"  {unmeasured} recorded battle(s) have no team-strength measurement yet: run scripts/team_strength_report.py")
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "tier-list.html").write_text(tier_list_page(players))
        for p in players:
            (OUT / f"{to_id_str(p.name) or 'player'}.html").write_text(player_card(p, players, args.name_others))
        (OUT / "analyst.md").write_text(analyst(players))
        print(f"\nwrote {len(players)} card(s), the tier list and analyst.md to {OUT}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="write the HTML pages and analyst.md")
    parser.add_argument("--name-others", action="store_true", help="show other players' names on each card")
    parser.add_argument("--exclude", action="append", default=[], metavar="NAME",
                        help="leave a player out, e.g. your own account (repeatable)")
    sys.exit(main(parser.parse_args()))
