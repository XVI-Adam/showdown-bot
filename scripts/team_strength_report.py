"""Team strength of recorded battles: was a result the draw or the play?

Usage:
    uv run python scripts/team_strength_report.py                 # human battles -> reports/team-strength.md
    uv run python scripts/team_strength_report.py --validate 60   # check the method on 60 bot battles first

For each recorded human battle, both teams are rebuilt (hidden opposing
details sampled from the set statistics) and the same bot plays both sides
many times on the local simulator; our team's win share is the matchup's team
strength. Results are cached in data/team_strength/ (only the win counts, no
battle logs), so re-runs only play new battles. Start the simulator first.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from poke_env import AccountConfiguration

from psbot.analytics.team_strength import Replayer, calibrated, chance_of_at_most, estimate, fit_calibration
from psbot.config import BATTLE_FORMAT, PROJECT_ROOT, local_server_configuration
from psbot.dataset import HumanBattleRecorder, agent_info
from psbot.dataset.human import DATA_DIR, HUMAN_DIR
from psbot.dataset.schema import load_dataset
from psbot.logging_setup import setup_logging
from psbot.opponents import make_opponent
from psbot.runner import run_battles
from psbot.server import ServerUnavailableError, require_server

CACHE = DATA_DIR / "team_strength" / "human.json"
VALIDATION = DATA_DIR / "team_strength" / "validation.json"  # per-battle numbers only
REPORT = PROJECT_ROOT / "reports" / "team-strength.md"
METHOD = "v2"  # bump when the rebuild or replay changes, to invalidate the cache


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


async def validate(n: int, samples: int, per_sample: int) -> int:
    """Record bot-vs-bot battles from both sides, then estimate each one twice:
    with the opponent's hidden details sampled (as for people) and with their
    true team. Report agreement and how well strength predicts the winner."""
    tmp = Path(tempfile.mkdtemp())
    common = dict(battle_format=BATTLE_FORMAT, server_configuration=local_server_configuration(), max_concurrent_battles=10)
    a = make_opponent("heuristic", account_configuration=AccountConfiguration.generate("ValA", rand=True), **common)
    b = make_opponent("heuristic", account_configuration=AccountConfiguration.generate("ValB", rand=True), **common)
    a.recorder = HumanBattleRecorder(agent_info("heuristic"), "local", tmp / "a.jsonl", opponent_kind="heuristic")
    b.recorder = HumanBattleRecorder(agent_info("heuristic"), "local", tmp / "b.jsonl", opponent_kind="heuristic")
    await run_battles(a, b, n_battles=n)
    theirs = {r["battle_id"]: r for r in b.recorder.records}
    replayer = Replayer("heuristic")
    rows = []
    rejected = 0
    started = time.monotonic()
    for record in a.recorder.records:
        sampled = await estimate(record, replayer, samples, per_sample)
        true = await estimate(record, replayer, samples, per_sample, their_record=theirs[record["battle_id"]])
        rows.append((record["outcome"]["won"], sampled["strength"], true["strength"]))
        rejected += len(sampled["rejected_rebuilds"]) + len(true["rejected_rebuilds"])
        print(f"  {len(rows)}/{len(a.recorder.records)}: won={record['outcome']['won']} sampled {sampled['strength']:.2f} "
              f"true {true['strength']:.2f}", flush=True)  # fmt: skip
    await replayer.close()
    VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION.write_text(json.dumps({
        "method": METHOD, "agent": "heuristic", "samples": samples, "per_sample": per_sample,
        "battles": [{"won": bool(w), "sampled": s, "true": t} for w, s, t in rows],
    }, indent=1))  # fmt: skip
    import numpy as np

    won = np.array([1.0 if w else 0.0 for w, _, _ in rows])
    p_s, p_t = np.array([s for _, s, _ in rows]), np.array([t for _, _, t in rows])
    print(f"\n{len(rows)} bot battles (heuristic vs heuristic), {samples}x{per_sample} replays each, "
          f"{(time.monotonic() - started) / 60:.1f} min")  # fmt: skip
    print(f"hidden-information sampling vs true teams: mean |difference| {np.abs(p_s - p_t).mean():.3f}, "
          f"correlation {np.corrcoef(p_s, p_t)[0, 1]:.2f}")  # fmt: skip
    for name, p in (("true teams", p_t), ("sampled", p_s)):
        brier = ((p - won) ** 2).mean()
        hit = ((p > 0.5) == (won > 0.5)).mean()
        print(f"{name:>11}: Brier {brier:.3f} (0.25 = no information), stronger team won {hit:.0%}")
    print(f"abandoned replays (no result within the time limit): {replayer.abandoned}; "
          f"rebuilds the validator rejected and were redrawn: {rejected}")
    print(f"spread of team strength: {np.percentile(p_t, 10):.0%} to {np.percentile(p_t, 90):.0%} (10th-90th percentile)")
    for low, high in ((0, 0.3), (0.3, 0.7), (0.7, 1.01)):
        sel = (p_t >= low) & (p_t < high)
        if sel.any():
            print(f"  strength {low:.0%}-{min(high, 1):.0%}: {sel.sum()} battles, actually won {won[sel].mean():.0%}")
    return 0


async def human(samples: int, per_sample: int) -> int:
    from psbot.evaluation.report import human_rows

    records = sorted((r for _, r in load_dataset(HUMAN_DIR)), key=lambda r: r["started_at"] or "")
    _, labels = human_rows()  # the same A/B/C labels as the Phase 7 report
    cache: dict[str, Any] = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    key = f"{METHOD}:{samples}x{per_sample}"
    todo = [r for r in records if f"{key}:{r['battle_id']}" not in cache]
    if todo:
        replayer = Replayer(records[0]["agent"]["name"])
        for i, record in enumerate(todo, 1):
            started = time.monotonic()
            cache[f"{key}:{record['battle_id']}"] = await estimate(record, replayer, samples, per_sample)
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(cache, indent=1))
            print(f"  {i}/{len(todo)} done ({time.monotonic() - started:.0f}s)", flush=True)
        await replayer.close()
    rows = [(labels[r["opponent"]["username"]], r, cache[f"{key}:{r['battle_id']}"]) for r in records]
    validation = json.loads(VALIDATION.read_text()) if VALIDATION.exists() else None
    REPORT.write_text(render(rows, samples, per_sample, validation))
    print(f"report: {REPORT}")
    return 0


def verdicts(rows: list[tuple[str, dict, dict]], calibration: dict | None) -> list[str]:
    by: dict[str, list[tuple[bool, float]]] = {}
    for label, record, est in rows:
        by.setdefault(label, []).append((record["outcome"]["won"] is True, calibrated(est["strength"], calibration)))
    out = []
    for label, games in sorted(by.items()):
        wins, expected = sum(w for w, _ in games), sum(p for _, p in games)
        chance = chance_of_at_most(wins, [p for _, p in games])
        weaker = sum(p < 0.5 for _, p in games)
        if chance < 0.05:
            text = "the bot did clearly worse than its teams deserved: the opponent's play explains it"
        elif wins > expected:
            text = "the bot won more than its teams predicted"
        else:
            text = (f"bad draws are enough to explain it ({chance:.0%} chance from team luck alone); "
                    "more battles are needed to tell whether skill played a part")  # fmt: skip
        out.append(f"- **{label}** ({len(games)} battles, bot won {wins}, expected {expected:.1f}): our team was the "
                   f"weaker one in {weaker} of {len(games)}; {text}.")  # fmt: skip
    return out


def render(rows: list[tuple[str, dict, dict]], samples: int, per_sample: int, validation: dict | None = None) -> str:
    battles = (validation or {}).get("battles", [])
    calibration = fit_calibration([b["won"] for b in battles], [b["sampled"] for b in battles]) if len(battles) >= 20 else None
    lines = [
        "# Team strength of the human battles",
        "",
        "**Question:** when the bot loses to a person, was it dealt the weaker team, or was it outplayed?",
        "",
        "**Method:** for each battle both teams were rebuilt. Ours is exact. Theirs keeps everything the battle revealed, "
        "with unrevealed moves, items, abilities, Tera types and unseen Pokémon sampled from the measured "
        "random-battle sets. The same bot then played both sides "
        f"{samples * per_sample} times ({samples} samples of the hidden details x {per_sample} battles). "
        "**Team strength** is the share our team won: 50% means an even draw. The 95% interval covers both the "
        "replay noise and the uncertainty about their hidden details. See "
        "`src/psbot/analytics/team_strength.py`; the method was first checked on bot battles where both teams are known.",
        "",
        "## Verdict",
        "",
        *verdicts(rows, calibration),
        "",
        "## Battle by battle",
        "",
        "| Opponent | Battle | Result | Our team strength (raw) | 95% interval | Calibrated win chance | Filled in per rebuild |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, record, est in rows:
        result = {True: "won", False: "lost", None: "tie"}[record["outcome"]["won"]]
        filled = ", ".join(f"{v:g} {k}" for k, v in est["filled_in"].items() if k.startswith("their")) or "-"
        lines.append(f"| {label} | {record['started_at'][:10]} | {result} | {_pct(est['strength'])} | "
                     f"{_pct(est['ci'][0])} to {_pct(est['ci'][1])} | {_pct(calibrated(est['strength'], calibration))} | {filled} |")  # fmt: skip
    lines += ["", "## By opponent", "",
              "| Opponent | Battles | Bot's wins | Expected from team strength | Chance of doing this badly or worse if teams alone decided |",
              "| --- | --- | --- | --- | --- |"]  # fmt: skip
    by: dict[str, list] = {}
    for label, record, est in rows:
        by.setdefault(label, []).append((record["outcome"]["won"] is True, calibrated(est["strength"], calibration)))
    for label, games in sorted(by.items()):
        wins = sum(w for w, _ in games)
        expected = sum(p for _, p in games)
        chance = chance_of_at_most(wins, [p for _, p in games])
        lines.append(f"| {label} | {len(games)} | {wins} | {expected:.1f} | {chance:.1%} |")
    lines += [
        "",
        "Expected wins and the last column use the *calibrated* win chance (see below). The last column is the "
        "chance of the bot winning this few if each battle went to the dice of its team strength alone. A small "
        "value (under about 5%) means the bot did worse than its teams deserved: the opponent's play explains "
        "it. A large value means bad draws are a sufficient explanation, though skill is not ruled out.",
        "",
        "Two battles recorded in the older log format (before Phase 4) are not included: they lack our full team.",
        "",
    ]
    if battles:
        import numpy as np

        won = np.array([1.0 if b["won"] else 0.0 for b in battles])
        sampled = np.array([b["sampled"] for b in battles])
        true = np.array([b["true"] for b in battles])
        lines += [
            "## How far to trust it",
            "",
            f"Checked first on {len(battles)} bot battles (`{validation['agent']}` vs itself), where both teams are known "
            f"and the winner is known ({validation['samples']} x {validation['per_sample']} replays each).",
            "",
            "| Check | Result |",
            "| --- | --- |",
            f"| Sampling their hidden details vs using their true team | correlation {np.corrcoef(sampled, true)[0, 1]:.2f}, "
            f"mean difference {np.mean(sampled - true):+.1%} (no bias) |",
            f"| Does team strength predict the winner? | Brier score {np.mean((sampled - won) ** 2):.3f} (0.25 = no information) |",
            f"| Spread of team strength across battles | {np.percentile(true, 10):.0%} to {np.percentile(true, 90):.0%} (10th to 90th percentile) |",
        ]
        for low, high in ((0, 0.3), (0.3, 0.7), (0.7, 1.01)):
            sel = (sampled >= low) & (sampled < high)
            if sel.any():
                lines.append(f"| Raw strength {low:.0%} to {min(high, 1):.0%} | {sel.sum()} battles, mean {sampled[sel].mean():.0%}, "
                             f"actually won {won[sel].mean():.0%} |")  # fmt: skip
        if calibration:
            lines += [
                "",
                f"Raw strengths are **too extreme**: the calibration slope is {calibration['b']:.2f} "
                f"(standard error {calibration['b_se']:.2f}; 1 would mean raw values are right). The replays have the same "
                "bot on both sides, so they credit the team with what is really decided by play and luck. The calibrated "
                f"win chance corrects for this. It is fitted on only {calibration['n']} battles, so treat it as approximate.",
            ]
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--validate", type=int, metavar="N", help="check the method on N bot battles instead")
    parser.add_argument("--samples", type=int, default=10, help="hidden-information samples per battle")
    parser.add_argument("--per-sample", type=int, default=30, help="replays per sample")
    args = parser.parse_args()
    setup_logging(verbose=False)
    try:
        require_server()
    except ServerUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.validate:
        return asyncio.run(validate(args.validate, args.samples, args.per_sample))
    return asyncio.run(human(args.samples, args.per_sample))


if __name__ == "__main__":
    sys.exit(main())
