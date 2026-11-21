# Phase 7 evaluation

Run `phase7-20261121-001413`: 28,000 battles of Gen 9 Random Battles on the local simulator (commit `e5d26285ab`), 1,000 per pairing among 8 participants. Each battle has freshly generated random teams, and no agent trained on any of them. Learned agents play their most likely action. Battle time: 47 minutes, summed over the parallel shards.

Every interval is a 95% interval: Wilson for proportions, normal for means, Newcombe for differences between two win rates.

## Participants

| Name | What it is |
| --- | --- |
| `random` | `random` |
| `max_base_power` | `max_base_power` |
| `simple_heuristics` | `simple_heuristics` |
| `heuristic` | `heuristic` |
| `heuristic_v2` | `heuristic_v2` |
| `ppo` | `checkpoints/ppo-v1/final.zip` |
| `hybrid` | `checkpoints/hybrid-v2-sparse/final.zip` |
| `bc` | `checkpoints/bc-v1/final.zip` |

## Against the baseline

The baseline is `heuristic` at **81.3%** in the final evaluation (2,500 battles against `random`, `max_base_power`, `simple_heuristics`, `ppo`, `hybrid`). Every agent is scored against the same pool below; this run re-measures the baseline itself at 82.8% (81.7% to 83.8%).

| Agent | Battles vs pool | Win rate vs pool | Difference from `heuristic` vs the same opponents | Win rate vs all others |
| --- | --- | --- | --- | --- |
| `hybrid` (pool without itself) | 4,000 | 87.9% (86.8% to 88.9%) | -2.3% (-3.6% to -0.9%) vs 90.1% | 74.5% (73.4% to 75.5%) |
| `heuristic_v2` | 5,000 | 83.9% (82.9% to 84.9%) | +1.2% (-0.3% to +2.6%) | 78.9% (77.9% to 79.8%) |
| `heuristic` | 5,000 | 82.8% (81.7% to 83.8%) | - | 77.2% (76.2% to 78.1%) |
| `bc` | 5,000 | 65.0% (63.6% to 66.3%) | -17.8% (-19.5% to -16.1%) | 52.3% (51.2% to 53.5%) |
| `simple_heuristics` (pool without itself) | 4,000 | 64.8% (63.3% to 66.3%) | -17.9% (-19.8% to -16.0%) vs 82.7% | 48.2% (47.0% to 49.3%) |
| `ppo` (pool without itself) | 4,000 | 63.2% (61.7% to 64.7%) | -19.9% (-21.8% to -18.0%) vs 83.2% | 47.1% (45.9% to 48.2%) |
| `max_base_power` (pool without itself) | 4,000 | 31.1% (29.6% to 32.5%) | -48.2% (-50.1% to -46.3%) vs 79.2% | 19.8% (18.9% to 20.8%) |
| `random` (pool without itself) | 4,000 | 3.0% (2.5% to 3.6%) | -75.6% (-77.0% to -74.2%) vs 78.6% | 2.1% (1.8% to 2.5%) |

## Win-rate matrix

Row agent's win rate against the column agent.

| row vs column | random | max_base_power | simple_heuristics | heuristic | heuristic_v2 | ppo | hybrid | bc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **random** | - | 8.3% | 0.8% | 0.7% | 0.5% | 2.4% | 0.5% | 1.6% |
| **max_base_power** | 91.7% | - | 11.1% | 3.1% | 2.5% | 17.5% | 3.9% | 9.0% |
| **simple_heuristics** | 99.2% | 88.9% | - | 16.9% | 16.9% | 49.4% | 21.8% | 44.0% |
| **heuristic** | 99.3% | 96.9% | 83.1% | - | 47.5% | 81.3% | 53.3% | 78.7% |
| **heuristic_v2** | 99.5% | 97.5% | 83.1% | 52.5% | - | 85.6% | 54.0% | 79.8% |
| **ppo** | 97.6% | 82.5% | 50.6% | 18.7% | 14.4% | - | 22.3% | 43.5% |
| **hybrid** | 99.5% | 96.1% | 78.2% | 46.7% | 46.0% | 77.7% | - | 77.0% |
| **bc** | 98.4% | 91.0% | 56.0% | 21.3% | 20.2% | 56.5% | 23.0% | - |

- `heuristic_v2` vs `heuristic` head to head: 52.5% (49.4% to 55.6%) of 1,000.

- `bc` vs `ppo` head to head: 56.5% (53.4% to 59.5%) of 1,000.

- `heuristic_v2` vs `hybrid` head to head: 54.0% (50.9% to 57.1%) of 1,000.

## Extended metrics

Each agent against every other participant. The opponent sets differ by one (an agent never plays itself), so compare large differences, not small ones. Definitions are in `src/psbot/evaluation/metrics.py` and `probe.py`. A *sure KO* is a move our damage estimate gives at least a 95% chance to knock out while moving first; it is *missed* when the agent did something without at least a 50% chance to knock out.

| Metric | `heuristic` | `heuristic_v2` | `hybrid` | `ppo` | `bc` |
| --- | --- | --- | --- | --- | --- |
| Battles | 7,000 | 7,000 | 7,000 | 7,000 | 7,000 |
| Win rate | 77.2% (76.2% to 78.1%) | 78.9% (77.9% to 79.8%) | 74.5% (73.4% to 75.5%) | 47.1% (45.9% to 48.2%) | 52.3% (51.2% to 53.5%) |
| Average Pokémon remaining | 2.71 (2.66 to 2.75) | 2.71 (2.66 to 2.75) | 2.60 (2.56 to 2.65) | 1.34 (1.30 to 1.38) | 1.51 (1.46 to 1.55) |
| Net knockouts, turns 1-10 | +1.15 (+1.12 to +1.19) | +1.13 (+1.09 to +1.16) | +1.13 (+1.09 to +1.16) | -0.13 (-0.16 to -0.09) | +0.04 (+0.00 to +0.08) |
| Net knockouts, turns 11-20 | +0.81 (+0.77 to +0.85) | +0.87 (+0.83 to +0.91) | +0.70 (+0.66 to +0.74) | -0.07 (-0.11 to -0.03) | +0.11 (+0.07 to +0.15) |
| Net knockouts, turns 21+ | +0.20 (+0.17 to +0.23) | +0.23 (+0.20 to +0.25) | +0.17 (+0.14 to +0.20) | +0.07 (+0.05 to +0.10) | +0.04 (+0.02 to +0.07) |
| Faced a setup sweeper (share of battles) | 1.7% (1.4% to 2.0%) | 2.2% (1.9% to 2.5%) | 1.7% (1.4% to 2.0%) | 3.0% (2.6% to 3.4%) | 2.7% (2.3% to 3.1%) |
| Win rate when facing one | 66.9% (58.0% to 74.8%) | 75.7% (68.3% to 81.8%) | 76.5% (68.1% to 83.2%) | 68.6% (62.0% to 74.5%) | 72.3% (65.5% to 78.2%) |
| Swept (one Pokémon took 3+ of ours) | 16.2% (15.4% to 17.1%) | 13.3% (12.5% to 14.1%) | 18.3% (17.4% to 19.2%) | 25.4% (24.4% to 26.4%) | 23.5% (22.5% to 24.5%) |
| Our KOs suffered to a move's first use | 49.7% (49.0% to 50.3%) | 49.8% (49.1% to 50.5%) | 48.4% (47.7% to 49.0%) | 49.3% (48.8% to 49.9%) | 49.5% (48.9% to 50.1%) |
| Win rate after such a KO | 72.2% (71.0% to 73.4%) | 74.8% (73.6% to 75.9%) | 69.4% (68.2% to 70.6%) | 43.3% (42.1% to 44.6%) | 48.8% (47.6% to 50.0%) |
| Reached an endgame (both sides at 2 or fewer) | 33.0% (31.9% to 34.1%) | 33.5% (32.4% to 34.6%) | 34.0% (32.9% to 35.1%) | 44.7% (43.5% to 45.9%) | 44.2% (43.0% to 45.4%) |
| Endgame win rate | 59.2% (57.2% to 61.2%) | 62.2% (60.2% to 64.1%) | 56.0% (54.0% to 58.0%) | 47.6% (45.8% to 49.3%) | 50.5% (48.7% to 52.2%) |
| ... entering it ahead | 85.3% (82.6% to 87.6%) | 87.8% (85.3% to 89.9%) | 83.0% (80.2% to 85.6%) | 88.1% (85.3% to 90.5%) | 87.9% (85.1% to 90.2%) |
| ... entering it level | 51.6% (48.9% to 54.3%) | 55.8% (53.1% to 58.5%) | 48.7% (46.1% to 51.3%) | 49.0% (46.7% to 51.3%) | 52.4% (50.1% to 54.7%) |
| ... entering it behind | 19.0% (14.6% to 24.4%) | 19.9% (15.5% to 25.2%) | 15.9% (11.9% to 20.9%) | 10.8% (8.8% to 13.3%) | 10.3% (8.2% to 12.8%) |
| Lost after leading by 2+ | 6.6% (6.0% to 7.3%) | 5.5% (5.0% to 6.2%) | 8.0% (7.3% to 8.8%) | 9.0% (8.0% to 10.0%) | 7.8% (7.0% to 8.8%) |
| Missed sure KOs (share of chances) | 0.8% (0.7% to 0.9%) | 0.2% (0.1% to 0.2%) | 0.8% (0.7% to 1.0%) | 10.8% (10.4% to 11.3%) | 5.0% (4.7% to 5.3%) |
| Moves into an immunity | 1.59% (1.52% to 1.67%) | 1.60% (1.53% to 1.68%) | 1.74% (1.66% to 1.81%) | 3.51% (3.41% to 3.62%) | 2.82% (2.73% to 2.92%) |
| Moves that missed or failed | 3.6% (3.5% to 3.7%) | 3.5% (3.4% to 3.6%) | 4.1% (4.0% to 4.3%) | 6.0% (5.8% to 6.1%) | 4.4% (4.3% to 4.6%) |
| Voluntary switches per battle | 3.48 (3.42 to 3.54) | 3.82 (3.54 to 4.10) | 3.53 (3.47 to 3.59) | 2.01 (1.73 to 2.29) | 2.18 (2.13 to 2.23) |
| Switch-in fainted before acting | 8.7% (8.4% to 9.1%) | 6.3% (6.1% to 6.6%) | 9.0% (8.6% to 9.3%) | 12.9% (12.4% to 13.5%) | 11.4% (10.9% to 11.9%) |
| Switch-in lost half its HP on entry | 15.5% (15.1% to 16.0%) | 12.6% (12.3% to 13.1%) | 15.6% (15.2% to 16.1%) | 17.6% (17.0% to 18.2%) | 15.3% (14.8% to 15.9%) |

### `heuristic` by opponent

| Opponent | Battles | Win rate | Pokémon remaining |
| --- | --- | --- | --- |
| `random` | 1,000 | 99.3% (98.6% to 99.7%) | 4.82 (4.75 to 4.88) |
| `max_base_power` | 1,000 | 96.9% (95.6% to 97.8%) | 3.76 (3.67 to 3.84) |
| `simple_heuristics` | 1,000 | 83.1% (80.7% to 85.3%) | 2.64 (2.54 to 2.75) |
| `heuristic_v2` | 1,000 | 47.5% (44.4% to 50.6%) | 1.27 (1.17 to 1.37) |
| `ppo` | 1,000 | 81.3% (78.8% to 83.6%) | 2.59 (2.49 to 2.70) |
| `hybrid` | 1,000 | 53.3% (50.2% to 56.4%) | 1.44 (1.34 to 1.54) |
| `bc` | 1,000 | 78.7% (76.1% to 81.1%) | 2.43 (2.32 to 2.54) |

### `heuristic_v2` by opponent

| Opponent | Battles | Win rate | Pokémon remaining |
| --- | --- | --- | --- |
| `random` | 1,000 | 99.5% (98.8% to 99.8%) | 4.67 (4.60 to 4.74) |
| `max_base_power` | 1,000 | 97.5% (96.3% to 98.3%) | 3.68 (3.59 to 3.76) |
| `simple_heuristics` | 1,000 | 83.1% (80.7% to 85.3%) | 2.67 (2.57 to 2.77) |
| `heuristic` | 1,000 | 52.5% (49.4% to 55.6%) | 1.43 (1.32 to 1.53) |
| `ppo` | 1,000 | 85.6% (83.3% to 87.6%) | 2.64 (2.54 to 2.74) |
| `hybrid` | 1,000 | 54.0% (50.9% to 57.1%) | 1.44 (1.34 to 1.54) |
| `bc` | 1,000 | 79.8% (77.2% to 82.2%) | 2.42 (2.32 to 2.52) |

## Illegal actions and fallbacks

| Agent | Decisions | Illegal actions | Engine fallbacks |
| --- | --- | --- | --- |
| `ppo` | 178,193 | 0 | 0 |
| `heuristic` | 170,339 | 0 | 0 |
| `bc` | 173,236 | 0 | 0 |
| `hybrid` | 174,144 | 0 | 0 |
| `heuristic_v2` | 168,575 | 0 | 0 |

## Against people

11 battles on the official server against 3 consenting people, all played by `heuristic`. 2 of them were recorded in the older, shorter log format. These samples are small: an interval of plus or minus 25 points is what 11 battles can say. Missed KOs were not probed in these battles.

| Opponent | Battles | Our win rate | Pokémon remaining |
| --- | --- | --- | --- |
| A | 2 | 100.0% (34.2% to 100.0%) | 4.00 (4.00 to 4.00) |
| B | 5 | 0.0% (0.0% to 43.4%) | 0.00 (0.00 to 0.00) |
| C | 4 | 100.0% (51.0% to 100.0%) | 2.25 (1.31 to 3.19) |

Side by side with the same agent against bots:

| Metric | vs people | vs bots (this run) |
| --- | --- | --- |
| Battles | 11 | 7,000 |
| Win rate | 54.5% (28.0% to 78.7%) | 77.2% (76.2% to 78.1%) |
| Average Pokémon remaining | 1.55 (0.54 to 2.55) | 2.71 (2.66 to 2.75) |
| Net knockouts, turns 1-10 | +0.09 (-0.40 to +0.58) | +1.15 (+1.12 to +1.19) |
| Net knockouts, turns 11-20 | +0.36 (-0.67 to +1.40) | +0.81 (+0.77 to +0.85) |
| Net knockouts, turns 21+ | +0.00 (-0.88 to +0.88) | +0.20 (+0.17 to +0.23) |
| Faced a setup sweeper (share of battles) | 45.5% (21.3% to 72.0%) | 1.7% (1.4% to 2.0%) |
| Win rate when facing one | 80.0% (37.6% to 96.4%) | 66.9% (58.0% to 74.8%) |
| Swept (one Pokémon took 3+ of ours) | 27.3% (9.7% to 56.6%) | 16.2% (15.4% to 17.1%) |
| Our KOs suffered to a move's first use | 53.2% (39.2% to 66.7%) | 49.7% (49.0% to 50.3%) |
| Win rate after such a KO | 54.5% (28.0% to 78.7%) | 72.2% (71.0% to 73.4%) |
| Reached an endgame (both sides at 2 or fewer) | 36.4% (15.2% to 64.6%) | 33.0% (31.9% to 34.1%) |
| Endgame win rate | 50.0% (15.0% to 85.0%) | 59.2% (57.2% to 61.2%) |
| ... entering it ahead | 100.0% (20.7% to 100.0%) | 85.3% (82.6% to 87.6%) |
| ... entering it level | 50.0% (9.5% to 90.5%) | 51.6% (48.9% to 54.3%) |
| ... entering it behind | 0.0% (0.0% to 79.3%) | 19.0% (14.6% to 24.4%) |
| Lost after leading by 2+ | 0.0% (0.0% to 39.0%) | 6.6% (6.0% to 7.3%) |
| Moves into an immunity | 2.62% (1.12% to 5.98%) | 1.59% (1.52% to 1.67%) |
| Moves that missed or failed | 1.6% (0.5% to 4.5%) | 3.6% (3.5% to 3.7%) |
| Voluntary switches per battle | 2.91 (1.84 to 3.98) | 3.48 (3.42 to 3.54) |
| Switch-in fainted before acting | 12.5% (5.0% to 28.1%) | 8.7% (8.4% to 9.1%) |
| Switch-in lost half its HP on entry | 12.5% (5.0% to 28.1%) | 15.5% (15.1% to 16.0%) |

## Limits

- Bot opponents only, apart from the small human sample. People switch predictively and play around obvious moves; none of that is in the bot results.
- KO attribution: a knockout is credited to the opposing Pokémon on the field, and counted as "to a move" only when that move's target fainted right after it (not recoil, hazards or residual damage).
- The sure-KO check is our own damage estimate; it misses Focus Sash, Sturdy and unseen Choice Scarf, equally for every agent.
- Random teams make every battle a different game; intervals are the only protection against reading noise as signal.

<!-- interpretation -->
## Findings

**Against the 81.3% baseline.** Re-measured here, the baseline `heuristic` scores 82.8% (81.7% to 83.8%) against the original pool, inside the final evaluation's interval (79.7% to 82.8%), so the two runs agree. No new agent clears it by a margin this run can confirm:

- `heuristic_v2`: 83.9% against the same pool, +1.2 points (-0.3 to +2.6). Head to head it beat the baseline 52.5% of 1,000 here. Phase 5 ran the same engine head to head for 3,000 battles at 52.8%; pooled, that is 52.7% of 4,000 (51.2% to 54.2%). So it is slightly stronger in direct play, but the gain does not show against weaker opponents, who lose to both.
- `hybrid` is 2.3 points *below* the heuristic against the same four opponents (-3.6 to -0.9), and 53.3% to 46.7% behind it head to head.
- `bc` (Phase 6) plays at PPO level: 56.5% against `ppo` head to head, far below the heuristics.

**Where `heuristic_v2` measurably differs** (non-overlapping intervals, each over 7,000 battles):

| | `heuristic` | `heuristic_v2` |
| --- | --- | --- |
| Swept (one Pokémon took 3+ of ours) | 16.2% | 13.3% |
| Switch-in fainted before acting | 8.7% | 6.3% |
| Switch-in lost half its HP on entry | 15.5% | 12.6% |
| Missed sure KOs | 0.8% | 0.2% |

These are the sweep-prevention and switching weaknesses the final evaluation named. They improved, even though the win rate barely moved.

**What did not change:** moves into an immunity (1.59% vs 1.60%), losing after leading by 2 or more (6.6% vs 5.5%, overlapping), and the endgame win rate (59.2% vs 62.2%, overlapping). The Phase 5 damage refinements did not reduce wasted turns, which fits Phase 5's diagnosis that most immune hits come from the opponent switching after we choose.

**The bots under-test the main weakness.** Bot opponents produced a setup sweeper in under 3% of battles; the people did in 5 of 11 battles. The share of our knockouts that came from a move's first use is about 50% for every agent, so it does not separate agents. Sweep and setup play are therefore barely measured by bot evaluation, and that is what people use.

**Against people.** 11 battles is too few to rank anything: 6-5 overall (54.5%, interval 28% to 79%), with all five losses against one opponent (B), each with no Pokémon left. That is consistent with a single strong or well-adapted player, not with a general rate.

**Decision:** keep `heuristic` as the default agent. `heuristic_v2` is the candidate. It is no worse anywhere measured and better on the sweep and switching metrics, so it is the one to test against people next. Promote it only once human games show it holding up.
