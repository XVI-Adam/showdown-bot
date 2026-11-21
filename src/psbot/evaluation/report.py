"""Phase 7 report: every agent against the original baseline, plus people.

The baseline is the heuristic's 81.3% from the final evaluation: its win rate
against random, max_base_power, simple_heuristics, ppo and hybrid. Every
agent is scored against that same reference pool here (an agent that is
itself in the pool is scored against the other four), so the numbers can be
compared with 81.3% and with each other.

Human opponents are labelled A, B, C... in order of first battle. Their
usernames stay in data/ (git-ignored) and never enter the report.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from psbot.config import LOG_DIR
from psbot.dataset.human import HUMAN_DIR
from psbot.dataset.schema import load_dataset
from psbot.evaluation.metrics import STAGES, battle_metrics
from psbot.evaluation.summary import difference, rate, summarise

REFERENCE_POOL = ("random", "max_base_power", "simple_heuristics", "ppo", "hybrid")
BASELINE = {"name": "heuristic", "final_evaluation": 0.813, "battles": 2500}
FOCUS = ("heuristic", "heuristic_v2", "hybrid", "ppo", "bc")


def _pct(r: dict[str, Any] | None, digits: int = 1) -> str:
    if not r or r.get("value") is None or not r.get("n", 1):
        return "n/a"
    low, high = r["ci"]
    return f"{r['value']:.{digits}%} ({low:.{digits}%} to {high:.{digits}%})"


def _num(r: dict[str, Any], digits: int = 2, signed: bool = False) -> str:
    if r.get("value") is None:
        return "n/a"
    low, high = r["ci"]
    sign = "+" if signed else ""
    return f"{r['value']:{sign}.{digits}f} ({low:{sign}.{digits}f} to {high:{sign}.{digits}f})"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    meta = json.loads((run_dir / "meta.json").read_text())
    pairings = [json.loads(l) for f in sorted(run_dir.glob("pairings-*.jsonl")) for l in f.read_text().splitlines()]
    rows = [json.loads(l) for f in sorted(run_dir.glob("battles-*.jsonl")) for l in f.read_text().splitlines() if l]
    return meta, pairings, rows


# ------------------------------------------------------------------ humans


def human_rows() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Per-battle metrics for every recorded human battle, both layouts.

    The older layout (logs/human) kept fewer event types (no damage events),
    so its entry-damage metric is left out rather than counted as zero.
    """
    rows: list[dict[str, Any]] = []
    labels: dict[str, str] = {}
    sources: list[tuple[str, dict[str, Any], bool]] = []
    if HUMAN_DIR.exists():
        sources += [(r["opponent"]["username"], r, True) for _, r in load_dataset(HUMAN_DIR)]
    legacy = LOG_DIR / "human"
    if legacy.exists():
        for f in sorted(legacy.glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    sources.append((r.get("opponent") or "?", r, False))
    sources.sort(key=lambda item: (item[1].get("started_at") or item[1].get("battle_tag") or item[1].get("battle_id")))
    for username, record, full in sources:
        label = labels.setdefault(username, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[len(labels) % 26])
        role = record.get("role") or "p1"
        row = battle_metrics(record["events"], role)
        won = record["outcome"]["won"] if full else record.get("won")
        row.update(
            agent=(record["agent"]["name"] if full else record.get("player_kind")) or "?",
            opponent=label,
            won=won,
            remaining=6 - row["our_faints"],
            sure_ko_chances=0,
            sure_ko_missed=0,
            full_log=full,
        )
        if not full:
            row["switch_big_hit"] = 0
        rows.append(row)
    return rows, labels


# ------------------------------------------------------------------ report


def _metric_rows(summaries: dict[str, dict[str, Any]], columns: list[str]) -> list[list[str]]:
    def row(label: str, getter) -> list[str]:
        return [label] + [getter(summaries[c]) for c in columns]

    out = [
        row("Battles", lambda s: f"{s['battles']:,}"),
        row("Win rate", lambda s: _pct(s["win_rate"])),
        row("Average Pokémon remaining", lambda s: _num(s["remaining"])),
    ]
    for name, first, last in STAGES:
        span = f"{first}+" if last > 1000 else f"{first}-{last}"
        out.append(row(f"Net knockouts, turns {span}", lambda s, n=name: _num(s["stages"][n]["net_kos"], signed=True)))
    out += [
        row("Faced a setup sweeper (share of battles)", lambda s: _pct(s["setup"]["faced"])),
        row("Win rate when facing one", lambda s: _pct(s["setup"]["win_rate_when_faced"])),
        row("Swept (one Pokémon took 3+ of ours)", lambda s: _pct(s["swept"])),
        row("Our KOs suffered to a move's first use", lambda s: _pct(s["unrevealed"]["kos_to_first_use_moves"])),
        row("Win rate after such a KO", lambda s: _pct(s["unrevealed"]["win_rate_with_such_ko"])),
        row("Reached an endgame (both sides at 2 or fewer)", lambda s: _pct(s["endgame"]["reached"])),
        row("Endgame win rate", lambda s: _pct(s["endgame"]["win_rate"])),
        row("... entering it ahead", lambda s: _pct(s["endgame"]["from_ahead"])),
        row("... entering it level", lambda s: _pct(s["endgame"]["from_level"])),
        row("... entering it behind", lambda s: _pct(s["endgame"]["from_behind"])),
        row("Lost after leading by 2+", lambda s: _pct(s["led_by_2_and_lost"])),
        row("Missed sure KOs (share of chances)", lambda s: _pct(s["missed_kos"])),
        row("Moves into an immunity", lambda s: _pct(s["immune_moves"], 2)),
        row("Moves that missed or failed", lambda s: _pct(s["missed_or_failed_moves"])),
        row("Voluntary switches per battle", lambda s: _num(s["switches"]["per_battle"])),
        row("Switch-in fainted before acting", lambda s: _pct(s["switches"]["fainted_before_acting"])),
        row("Switch-in lost half its HP on entry", lambda s: _pct(s["switches"]["lost_half_hp_on_entry"])),
    ]
    return out


def build(run_dir: Path) -> str:
    meta, pairings, rows = load_run(run_dir)
    agents = list(meta["participants"])
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["agent"], r["opponent"])].append(r)

    def pool_rows(agent: str, pool: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
        return [r for o in pool if o != agent for r in by[(agent, o)]]

    reference = {a: summarise(pool_rows(a, REFERENCE_POOL)) for a in agents}
    overall = {a: summarise(pool_rows(a, agents)) for a in agents}
    baseline_now = reference["heuristic"]["win_rate"]
    total_battles = sum(p["n"] for p in pairings)
    seconds = sum(p["seconds"] for p in pairings)

    parts = [
        "# Phase 7 evaluation",
        f"Run `{run_dir.name}`: {total_battles:,} battles of Gen 9 Random Battles on the local simulator "
        f"(commit `{meta['simulator_commit'][:10]}`), {meta['n_battles']:,} per pairing among "
        f"{len(agents)} participants. Each battle has freshly generated random teams, and no agent "
        f"trained on any of them. Learned agents play their most likely action. "
        f"Battle time: {seconds / 60:.0f} minutes, summed over the parallel shards.",
        "Every interval is a 95% interval: Wilson for proportions, normal for means, Newcombe for "
        "differences between two win rates.",
        "## Participants",
        _table(["Name", "What it is"], [[f"`{a}`", f"`{spec}`"] for a, spec in meta["participants"].items()]),
        "## Against the baseline",
        f"The baseline is `heuristic` at **81.3%** in the final evaluation (2,500 battles against "
        f"{', '.join(f'`{p}`' for p in REFERENCE_POOL)}). Every agent is scored against the same pool "
        f"below; this run re-measures the baseline itself at {_pct(baseline_now)}.",
    ]
    headline = []
    for a in sorted(agents, key=lambda x: -(reference[x]["win_rate"]["value"] or 0)):
        # Compare with the heuristic against exactly the same opponents: an
        # agent in the pool does not play itself, so the heuristic's figure is
        # recomputed without that opponent too.
        same = [o for o in REFERENCE_POOL if o != a]
        baseline_same = summarise(pool_rows("heuristic", same))["win_rate"]
        diff = difference(reference[a]["win_rate"], baseline_same)
        in_pool = " (pool without itself)" if a in REFERENCE_POOL else ""
        delta = "-" if a == "heuristic" else (
            f"{diff['value']:+.1%} ({diff['ci'][0]:+.1%} to {diff['ci'][1]:+.1%})"
            + ("" if a not in REFERENCE_POOL else f" vs {baseline_same['value']:.1%}")
        )
        headline.append([f"`{a}`{in_pool}", f"{reference[a]['battles']:,}", _pct(reference[a]["win_rate"]),
                         delta, _pct(overall[a]["win_rate"])])  # fmt: skip
    parts.append(_table(
        ["Agent", "Battles vs pool", "Win rate vs pool", "Difference from `heuristic` vs the same opponents", "Win rate vs all others"],
        headline,
    ))  # fmt: skip

    wins = {(p["a"], p["b"]): (p["wins_a"], p["n"]) for p in pairings}
    matrix = []
    for a in agents:
        cells = []
        for b in agents:
            if a == b:
                cells.append("-")
            elif (a, b) in wins:
                w, n = wins[(a, b)]
                cells.append(f"{w / n:.1%}")
            elif (b, a) in wins:
                w, n = wins[(b, a)]
                cells.append(f"{(n - w) / n:.1%}")
            else:
                cells.append("?")
        matrix.append([f"**{a}**"] + cells)
    parts += ["## Win-rate matrix", "Row agent's win rate against the column agent.",
              _table(["row vs column"] + agents, matrix)]  # fmt: skip

    for a, b in (("heuristic_v2", "heuristic"), ("bc", "ppo"), ("heuristic_v2", "hybrid")):
        key = (a, b) if (a, b) in wins else (b, a)
        if key in wins:
            w, n = wins[key]
            w = w if key == (a, b) else n - w
            parts.append(f"- `{a}` vs `{b}` head to head: {_pct(rate(w, n))} of {n:,}.")

    focus = [a for a in FOCUS if a in agents]
    parts += [
        "## Extended metrics",
        "Each agent against every other participant. The opponent sets differ by one (an agent "
        "never plays itself), so compare large differences, not small ones. Definitions are in "
        "`src/psbot/evaluation/metrics.py` and `probe.py`. A *sure KO* is a move our damage estimate "
        "gives at least a 95% chance to knock out while moving first; it is *missed* when the agent "
        "did something without at least a 50% chance to knock out.",
        _table(["Metric"] + [f"`{a}`" for a in focus], _metric_rows(overall, focus)),
    ]
    for a in ("heuristic", "heuristic_v2"):
        per_opponent = [
            [f"`{o}`", f"{len(by[(a, o)]):,}", _pct(summarise(by[(a, o)])["win_rate"]),
             _num(summarise(by[(a, o)])["remaining"])]
            for o in agents if o != a and by[(a, o)]
        ]  # fmt: skip
        parts += [f"### `{a}` by opponent", _table(["Opponent", "Battles", "Win rate", "Pokémon remaining"], per_opponent)]

    illegal = defaultdict(lambda: [0, 0, 0])
    for p in pairings:
        for name, c in p["counters"].items():
            illegal[name][0] += c["decisions"] or 0
            illegal[name][1] += c["illegal_actions"] or 0
            illegal[name][2] += c["fallback_decisions"] or 0
    parts += [
        "## Illegal actions and fallbacks",
        _table(["Agent", "Decisions", "Illegal actions", "Engine fallbacks"],
               [[f"`{n}`", f"{d:,}", f"{i:,}", f"{f:,}"] for n, (d, i, f) in illegal.items() if d]),
    ]  # fmt: skip

    humans, labels = human_rows()
    if humans:
        human_summary = summarise(humans)
        bots = overall["heuristic"]
        per = []
        for label in sorted(set(labels.values())):
            mine = [r for r in humans if r["opponent"] == label]
            per.append([label, str(len(mine)), _pct(summarise(mine)["win_rate"]), _num(summarise(mine)["remaining"])])
        agents_used = sorted({r["agent"] for r in humans})
        parts += [
            "## Against people",
            f"{len(humans)} battles on the official server against {len(labels)} consenting people, all "
            f"played by {', '.join(f'`{a}`' for a in agents_used)}. "
            f"{sum(not r['full_log'] for r in humans)} of them were recorded in the older, shorter log "
            "format. These samples are small: an interval of plus or minus 25 points is what 11 "
            "battles can say. Missed KOs were not probed in these battles.",
            _table(["Opponent", "Battles", "Our win rate", "Pokémon remaining"], per),
            "Side by side with the same agent against bots:",
            _table(["Metric", "vs people", "vs bots (this run)"],
                   [[r[0], r[1], r[2]] for r in _metric_rows({"h": human_summary, "b": bots}, ["h", "b"])
                    if not r[0].startswith("Missed sure KOs")]),
        ]  # fmt: skip

    parts += [
        "## Limits",
        "- Bot opponents only, apart from the small human sample. People switch predictively and play "
        "around obvious moves; none of that is in the bot results.\n"
        "- KO attribution: a knockout is credited to the opposing Pokémon on the field, and counted as "
        "\"to a move\" only when that move's target fainted right after it (not recoil, hazards or "
        "residual damage).\n"
        "- The sure-KO check is our own damage estimate; it misses Focus Sash, Sturdy and unseen "
        "Choice Scarf, equally for every agent.\n"
        "- Random teams make every battle a different game; intervals are the only protection "
        "against reading noise as signal.",
    ]
    return "\n\n".join(parts) + "\n"
