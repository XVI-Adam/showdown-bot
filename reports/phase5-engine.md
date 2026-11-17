# Phase 5: engine changes and what they measured

Simulator commit `e5d26285ab`, Gen 9 Random Battles, local server. Every number is the new engine's win rate **against the unchanged baseline heuristic** (so 50% means no change), in fresh battles with random teams. Intervals are 95% Wilson intervals.

The baseline engine is untouched: every addition is behind a setting that is off by default, and the baseline is still the agent named `heuristic`. The new engine is the agent `heuristic_v2`.

## Result

| Engine | Battles | Win rate vs baseline | 95% interval |
| --- | --- | --- | --- |
| Baseline (mirror) | - | 50% by symmetry | - |
| + move model (B) | 3,000 | **53.7%** | 51.9% to 55.5% |
| + move model + sweep check (A) | 3,000 | 54.2% | 52.4% to 56.0% |
| + move model + endgame search (C) | 3,000 | 53.7% | 51.9% to 55.5% |
| + move model + sweep + endgame | 6,000 (two runs: 51.1%, 53.5%) | 52.3% | 51.0% to 53.5% |
| + move model + damage refinements (D) | 3,000 | 53.0% | 51.2% to 54.7% |
| Everything (`heuristic_v2`) | 3,000 | 52.8% | 51.0% to 54.5% |

Against poke-env's SimpleHeuristicsPlayer, the move model alone won 83.0% of 1,500 (81.0% to 84.8%); the baseline's figure from the final evaluation was 80.0% of 500 (76.3% to 83.3%).

**What this supports:**

- The **move model is a real improvement**: about 3.7 points head-to-head, and its interval excludes 50%.
- The **sweep check, endgame search and damage refinements show no measurable effect** on top of it. Each lands within about one point of the move model alone, which is inside the noise of a 3,000-battle run (about ±1.8 points).
- The two runs of "sweep + endgame" differ by 2.4 points from each other with identical settings. That is what run-to-run noise looks like at this size, and it is why no weight was tuned: the early 500-battle weight sweeps (49.6% to 56.0%) were not telling settings apart.

Detecting a one-point effect reliably needs roughly 10,000 battles per comparison. That, and the targeted metrics (performance against setup sweepers, endgame win rate, wasted turns), belong to the evaluation phase.

## B. Opponent move model

`scripts/build_randbats_stats.py` has the pinned simulator generate 200,000 teams with its own team generator and counts every (moves, ability, item) combination per species: 509 species, 419 to 3,312 samples each. `psbot/opponent_model/sets.py` conditions those counts on what a Pokémon has shown.

Calibration, measured at decision time against opponents' true movesets (150 battles, 21,865 predictions):

| Predicted | Predictions | Mean predicted | Actually had the move |
| --- | --- | --- | --- |
| 0-20% | 4,389 | 4% | 4% |
| 20-40% | 3,986 | 30% | 32% |
| 40-60% | 5,484 | 50% | 50% |
| 60-80% | 1,646 | 69% | 67% |
| 80-100% | 6,360 | 99% | 99% |

Brier score 0.128, against 0.278 for the baseline's flat 0.7 on every pool move.

The engine uses it for the threat to each of our Pokémon: candidates are sorted by damage, and each counts with the probability that the opponent has it and nothing stronger. The baseline instead took the single worst pool move and multiplied its damage by 0.7, which turns a certain knockout into "survives".

Not done: inference from observed damage (to narrow items or spreads) and from behaviour (for example, a Choice-locked Pokémon). The model's item probabilities exist but the engine does not yet use them for speed (Choice Scarf).

## A and C. Forward projection

`psbot/heuristics/lookahead.py` plays the fight forward with both sides using their most damaging move. While the opponent has unseen Pokémon it asks what their active Pokémon costs us before it is stopped (sweep check). Once all their remaining Pokémon are known and at most six are left in total, it searches the rest of the battle by minimax over replacement choices (endgame).

The projection is informative. In 500 baseline battles:

| Projection for our best action | Decisions | Battle then won |
| --- | --- | --- |
| Endgame: projected win | 1,641 | 79.3% |
| Endgame: projected loss | 1,099 | 20.7% |
| Sweep check: cost under 1 HP bar | 10,227 | 47.6% |
| Sweep check: cost 1 to 2 bars | 1,150 | 28.3% |
| Sweep check: cost 2 bars or more | 553 | 11.9% |

But it rarely disagrees with the baseline: in endgames the baseline already chose a projected-winning action in 1,549 of 1,602 decisions where one existed, and in the sweep check its choice was within half an HP bar of the projection's best in 96% of decisions. A projection that mostly agrees cannot move the win rate much. It also cannot find wins that need setup, recovery or a mid-fight switch, because it does not model them.

## D. Damage and immunities

In 180 baseline battles (2,714 of our moves), 39 attacks hit an immunity. 35 were type immunities and 4 were Levitate. Since the engine already scores a move at zero against an immune target, nearly all of these were the opponent switching in (or Terastallizing into) an immune Pokémon after we chose. 72 moves missed and 21 failed (6 were Sucker Punch into a non-attack).

Added, in `psbot/heuristics/context.py` and `scorers.py`:

- an unseen opposing ability that would nullify the move is weighted by how likely the species is to have it;
- accuracy and evasion stages change the chance to hit;
- an attack that a revealed, healthy bench Pokémon of theirs is immune to loses a fixed share of its value.

Accuracy, damage ranges, weather, terrain, items, abilities and stat stages were already handled by poke-env's damage calculator.

## Open issues found on the way

- The engine's "certain ability" is wrong for Greninja (Greninja-Bond is displayed as Greninja). The move model accounts for it; the damage code still assumes Protean.
- Move-pool lookups miss some formes after a forme change (Ogerpon after Terastallizing, Zoroark-Hisui).
