"""Turn evaluation results and battle logs into a Markdown performance report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from psbot.analysis import (
    compare_wins_and_losses,
    hybrid_deviations,
    load_records,
    prediction_calibration,
    unexpected_ko_causes,
)
from psbot.runner import wilson_interval

# Text after this marker in an existing report is kept when it is regenerated,
# so hand-written interpretation survives new evaluation runs.
INTERPRETATION_MARKER = "<!-- interpretation -->"

AGENTS = ("random", "heuristic", "ppo", "hybrid")  # the four agents under evaluation

KO_CAUSE_LABELS = {
    "unseen_move": "Hit by an attack that Pokémon had not used before in the battle",
    "known_move_underestimated": "An attack it had already shown did more than predicted",
    "critical_hit": "Critical hit",
    "opponent_terastallized": "Opponent Terastallized that turn",
    "no_attack": "No attack that turn (residual damage, recoil, hazards)",
}

YES_NO_FACTORS = {
    "swept": "One opposing Pokémon knocked out 3 or more of ours",
    "swept_after_setup": "...and that Pokémon had used a setup move",
    "opponent_used_setup": "Opponent used a setup move at all",
    "trailed_by_2": "We trailed by 2+ Pokémon at some point",
    "led_by_2": "We led by 2+ Pokémon at some point",
    "hazards_on_our_side": "Opponent set entry hazards on our side",
    "close_finish": "Decided by a single Pokémon",
}
PREDICTION_YES_NO_FACTORS = {
    "unexpected_ko": "Lost a Pokémon we expected to survive the turn (KO judged under 25%)",
    "outsped_unexpectedly": "Moved second after predicting we would move first",
}
COUNT_FACTORS = {
    "turns": "Battle length in turns",
    "n_voluntary_switches": "Voluntary switches",
    "n_chained_switches": "Voluntary switches on consecutive decisions",
    "n_failed_moves": "Our moves that missed, failed or hit an immunity",
    "n_turns_lost": "Turns we could not act (sleep, paralysis, flinch, recharge)",
    "n_opponent_setup_moves": "Opponent's setup moves",
    "n_our_setup_moves": "Our setup moves",
    "n_our_hazard_moves": "Our hazard moves",
    "n_our_status_moves": "Our status-inflicting moves",
    "n_our_recovery_moves": "Our recovery moves",
}
PREDICTION_COUNT_FACTORS = {
    "n_unexpected_kos": "Unexpected knockouts suffered",
    "n_speed_surprises": "Times outsped unexpectedly",
}


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _pairing_view(results: dict[str, Any], agent: str, opponent: str) -> dict[str, Any] | None:
    """One pairing's numbers from ``agent``'s side, whichever side it played."""
    for p in results["pairings"]:
        if (p["a"], p["b"]) == (agent, opponent):
            return {
                "n": p["n"], "wins": p["wins_a"], "ties": p["ties"],
                "remaining": p["remaining_a"], "opp_remaining": p["remaining_b"], "turns": p["avg_turns"],
            }  # fmt: skip
        if (p["a"], p["b"]) == (opponent, agent):
            return {
                "n": p["n"], "wins": p["wins_b"], "ties": p["ties"],
                "remaining": p["remaining_b"], "opp_remaining": p["remaining_a"], "turns": p["avg_turns"],
            }  # fmt: skip
    return None


def _overall(results: dict[str, Any], agent: str) -> tuple[int, int]:
    wins = battles = 0
    for other in results["participants"]:
        view = _pairing_view(results, agent, other)
        if view:
            wins += view["wins"]
            battles += view["n"]
    return wins, battles


def _win_matrix(results: dict[str, Any]) -> str:
    names = list(results["participants"])
    rows = []
    for agent in names:
        cells = []
        for other in names:
            view = _pairing_view(results, agent, other)
            cells.append("-" if view is None else _pct(view["wins"] / view["n"]))
        wins, battles = _overall(results, agent)
        rows.append([f"**{agent}**", *cells, _pct(wins / battles) if battles else "-"])
    return _table(["row vs column", *names, "overall"], rows)


def _agent_detail(results: dict[str, Any], agent: str) -> str:
    rows = []
    for other in results["participants"]:
        view = _pairing_view(results, agent, other)
        if view is None:
            continue
        low, high = wilson_interval(view["wins"], view["n"])
        rows.append(
            [
                other, str(view["n"]), str(view["wins"]), _pct(view["wins"] / view["n"]),
                f"{_pct(low)} to {_pct(high)}", f"{view['remaining']:.2f}",
                f"{view['opp_remaining']:.2f}", f"{view['turns']:.1f}",
            ]  # fmt: skip
        )
    wins, battles = _overall(results, agent)
    low, high = wilson_interval(wins, battles)
    rows.append(
        ["**all opponents**", str(battles), str(wins), f"**{_pct(wins / battles)}**",
         f"{_pct(low)} to {_pct(high)}", "", "", ""]  # fmt: skip
    )
    header = ["Opponent", "Battles", "Wins", "Win rate", "95% interval",
              "Avg Pokémon left", "Avg opponent left", "Avg turns"]  # fmt: skip
    return _table(header, rows)


def _legality(results: dict[str, Any]) -> str:
    rows = []
    for agent in AGENTS:
        stats = results["agents"].get(agent, {})
        decisions = stats.get("decisions")
        if decisions is None:
            rows.append([agent, "not counted", "0 by construction", "-", "-"])
            continue
        illegal = stats.get("illegal_actions")
        fallbacks = stats.get("fallback_decisions")
        rows.append(
            [
                agent,
                f"{decisions:,}",
                "-" if illegal is None else str(illegal),
                "-" if illegal is None else f"{illegal / max(decisions, 1):.4%}",
                "-" if fallbacks is None else str(fallbacks),
            ]
        )
    return _table(["Agent", "Decisions", "Illegal actions", "Illegal action rate", "Engine fallbacks"], rows)


def _factor_tables(comparison: dict[str, Any], with_predictions: bool) -> str:
    factors = comparison["factors"]
    yes_no = dict(YES_NO_FACTORS)
    counts = dict(COUNT_FACTORS)
    if with_predictions:
        yes_no.update(PREDICTION_YES_NO_FACTORS)
        counts.update(PREDICTION_COUNT_FACTORS)

    rows = []
    ordered = sorted(yes_no, key=lambda k: factors[k]["losses"] - factors[k]["wins"], reverse=True)
    for key in ordered:
        loss, win = factors[key]["losses"], factors[key]["wins"]
        rows.append([yes_no[key], _pct(loss), _pct(win), f"{(loss - win) * 100:+.1f} pts"])
    out = _table(["Happened in the battle", "Share of losses", "Share of wins", "Difference"], rows)

    rows = []
    for key, label in counts.items():
        loss, win = factors[key]["losses"], factors[key]["wins"]
        rows.append([label, f"{loss:.2f}", f"{win:.2f}"])
    out += "\n\n" + _table(["Average per battle", "In losses", "In wins"], rows)
    return out


def _calibration(calibration: dict[str, Any]) -> str:
    threat, entry, speed = calibration["threat"], calibration["entry"], calibration["speed"]
    rows = [
        [
            "Damage taken when staying in (turns the opponent attacked)",
            f"{threat['n']:,}", _pct(threat["ok"]), _pct(threat["under"]), _pct(threat["over"]),
            f"{threat['mean_error'] * 100:+.1f} pts",
        ],
        [
            "Damage taken by a voluntary switch-in on entry",
            f"{entry['n']:,}", _pct(entry["ok"]), _pct(entry["under"]), _pct(entry["over"]),
            f"{entry['mean_error'] * 100:+.1f} pts",
        ],
    ]  # fmt: skip
    out = _table(
        ["Prediction", "Cases", "Within 20 pts", "Took 20+ pts more", "Took 20+ pts less", "Mean error"],
        rows,
    )
    out += (
        f"\n\n- Predicted to move first and both sides moved: {speed['n']:,} turns; "
        f"the opponent moved first in **{_pct(speed['moved_second'])}** of them."
        f"\n- Voluntary switch-ins that fainted on entry: **{_pct(entry['fainted_on_entry'])}**."
    )
    return out


def build_report(results: dict[str, Any], records_path: Path | None) -> str:
    """Render the full report. ``records_path`` enables the weakness analysis."""
    participants = results["participants"]
    n = results["n_battles"]
    total = sum(p["n"] for p in results["pairings"])
    ranking = sorted(
        ((_overall(results, name), name) for name in participants),
        key=lambda item: item[0][0] / max(item[0][1], 1),
        reverse=True,
    )

    parts = [
        "# Final evaluation",
        f"Run `{results['run_id']}`: {total:,} battles of Gen 9 Random Battles on a local "
        f"simulator (commit `{results.get('simulator_commit', 'unknown')[:10]}`), "
        f"{n} per pairing, {results['elapsed_s'] / 60:.1f} minutes.",
        "Every battle is a new game with freshly generated random teams for both sides. "
        "None of these battles was used for training: the learned agents' weights were frozen "
        "before the run, and they play their most likely action.",
        "## Participants",
        _table(
            ["Name", "What it is"],
            [[f"`{name}`", f"`{spec}`"] for name, spec in participants.items()],
        ),
        "The four agents under evaluation are `random`, `heuristic`, `ppo` and `hybrid`. "
        "`max_base_power` and `simple_heuristics` are poke-env's built-in bots, included as "
        "additional opponent types.",
        "## Ranking",
        _table(
            ["Rank", "Participant", "Win rate vs the other five", "Battles"],
            [
                [str(i + 1), f"`{name}`", _pct(wins / battles), f"{battles:,}"]
                for i, ((wins, battles), name) in enumerate(ranking)
            ],
        ),
        "## Win-rate matrix",
        "Each cell is the row participant's win rate against the column participant.",
        _win_matrix(results),
    ]

    for agent in AGENTS:
        parts += [f"## `{agent}` by opponent type", _agent_detail(results, agent)]

    parts += [
        "## Illegal actions",
        "For learned agents, an illegal action is a policy output that the legal-action mask "
        "or poke-env rejected (it would be replaced by a random legal move). For the heuristic, "
        "an engine fallback is a decision where scoring raised an error and a random legal "
        "move was played instead.",
        _legality(results),
    ]

    if records_path is not None and records_path.is_file():
        records = list(load_records(records_path))
        for agent, with_predictions in (("heuristic", True), ("hybrid", False)):
            mine = [r for r in records if r.get("player_kind") == agent and r.get("opponent_kind") != "random"]
            if not mine:
                continue
            comparison = compare_wins_and_losses(mine)
            opponents = sorted({r["opponent_kind"] for r in mine})
            parts += [
                f"## What separates `{agent}`'s losses from its wins",
                f"{comparison['losses']:,} losses and {comparison['wins']:,} wins against "
                f"{', '.join(f'`{o}`' for o in opponents)} (battles against `random` excluded). "
                "These are associations, not proven causes: a factor far more common in losses "
                "marks where to look.",
                _factor_tables(comparison, with_predictions),
            ]
            if with_predictions:
                causes = unexpected_ko_causes(mine)
                cause_rows = sorted(causes["shares"].items(), key=lambda kv: kv[1], reverse=True)
                parts += [
                    "### How accurate were the heuristic's predictions?",
                    _calibration(prediction_calibration(mine)),
                    "### Why did Pokémon faint when they were expected to survive?",
                    f"{causes['total']:,} knockouts where the heuristic stayed in judging the "
                    "chance of being knocked out that turn below 25%. Each is counted once, "
                    "under the first explanation that applies (critical hit, then Tera, then "
                    "unseen move).",
                    _table(
                        ["Explanation", "Share"],
                        [[KO_CAUSE_LABELS[key], _pct(share)] for key, share in cause_rows],
                    ),
                ]
            else:
                deviations = hybrid_deviations(mine)
                rows = [
                    [kind, f"{count:,}", _pct(count / max(deviations["disagreements"], 1))]
                    for kind, count in deviations["kinds"].items()
                ]
                parts += [
                    "### Where the hybrid overrules the heuristic",
                    f"The hybrid chose differently from the heuristic's top-scoring action in "
                    f"{deviations['disagreements']:,} of {deviations['choices']:,} real choices "
                    f"({_pct(deviations['rate'])}).",
                    _table(["Heuristic's pick -> hybrid's pick", "Count", "Share of deviations"], rows),
                ]

    parts += [
        "## Limits of this evaluation",
        "- All opponents are bots. Human players predict switches, bluff, and play around "
        "obvious moves; none of that is measured here.\n"
        "- The loss analysis is observational and partly depends on the opponents chosen.\n"
        "- Knockouts are attributed to the opposing Pokémon on the field, which includes "
        "faints from recoil, hazards and status.",
    ]
    return "\n\n".join(parts) + "\n"


def write_report(path: Path, body: str) -> None:
    """Write the report, keeping any hand-written text after the marker."""
    tail = ""
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if INTERPRETATION_MARKER in existing:
            tail = existing.split(INTERPRETATION_MARKER, 1)[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body + "\n" + INTERPRETATION_MARKER + tail, encoding="utf-8")
