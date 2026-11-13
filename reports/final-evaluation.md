# Final evaluation

Run `final-20261112-195857`: 7,500 battles of Gen 9 Random Battles on a local simulator (commit `e5d26285ab`), 500 per pairing, 4.5 minutes.

Every battle is a new game with freshly generated random teams for both sides. None of these battles was used for training: the learned agents' weights were frozen before the run, and they play their most likely action.

## Participants

| Name | What it is |
| --- | --- |
| `random` | `random` |
| `max_base_power` | `max_base_power` |
| `simple_heuristics` | `simple_heuristics` |
| `heuristic` | `heuristic` |
| `ppo` | `checkpoints/ppo-v1/final.zip` |
| `hybrid` | `checkpoints/hybrid-v2-sparse/final.zip` |

The four agents under evaluation are `random`, `heuristic`, `ppo` and `hybrid`. `max_base_power` and `simple_heuristics` are poke-env's built-in bots, included as additional opponent types.

## Ranking

| Rank | Participant | Win rate vs the other five | Battles |
| --- | --- | --- | --- |
| 1 | `heuristic` | 81.3% | 2,500 |
| 2 | `hybrid` | 81.0% | 2,500 |
| 3 | `simple_heuristics` | 57.3% | 2,500 |
| 4 | `ppo` | 52.3% | 2,500 |
| 5 | `max_base_power` | 25.2% | 2,500 |
| 6 | `random` | 2.9% | 2,500 |

## Win-rate matrix

Each cell is the row participant's win rate against the column participant.

| row vs column | random | max_base_power | simple_heuristics | heuristic | ppo | hybrid | overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **random** | - | 9.6% | 1.2% | 0.4% | 2.4% | 1.0% | 2.9% |
| **max_base_power** | 90.4% | - | 10.0% | 2.6% | 18.4% | 4.4% | 25.2% |
| **simple_heuristics** | 98.8% | 90.0% | - | 20.0% | 55.0% | 22.8% | 57.3% |
| **heuristic** | 99.6% | 97.4% | 80.0% | - | 81.2% | 48.2% | 81.3% |
| **ppo** | 97.6% | 81.6% | 45.0% | 18.8% | - | 18.6% | 52.3% |
| **hybrid** | 99.0% | 95.6% | 77.2% | 51.8% | 81.4% | - | 81.0% |

## `random` by opponent type

| Opponent | Battles | Wins | Win rate | 95% interval | Avg Pokémon left | Avg opponent left | Avg turns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| max_base_power | 500 | 48 | 9.6% | 7.3% to 12.5% | 0.13 | 2.89 | 29.6 |
| simple_heuristics | 500 | 6 | 1.2% | 0.6% to 2.6% | 0.01 | 3.98 | 22.7 |
| heuristic | 500 | 2 | 0.4% | 0.1% to 1.4% | 0.00 | 4.84 | 20.7 |
| ppo | 500 | 12 | 2.4% | 1.4% to 4.1% | 0.02 | 3.95 | 25.7 |
| hybrid | 500 | 5 | 1.0% | 0.4% to 2.3% | 0.01 | 4.70 | 21.4 |
| **all opponents** | 2500 | 73 | **2.9%** | 2.3% to 3.7% |  |  |  |

## `heuristic` by opponent type

| Opponent | Battles | Wins | Win rate | 95% interval | Avg Pokémon left | Avg opponent left | Avg turns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 500 | 498 | 99.6% | 98.6% to 99.9% | 4.84 | 0.00 | 20.7 |
| max_base_power | 500 | 487 | 97.4% | 95.6% to 98.5% | 3.83 | 0.03 | 16.6 |
| simple_heuristics | 500 | 400 | 80.0% | 76.3% to 83.3% | 2.56 | 0.33 | 18.4 |
| ppo | 500 | 406 | 81.2% | 77.5% to 84.4% | 2.55 | 0.34 | 20.5 |
| hybrid | 500 | 241 | 48.2% | 43.9% to 52.6% | 1.33 | 1.38 | 25.3 |
| **all opponents** | 2500 | 2032 | **81.3%** | 79.7% to 82.8% |  |  |  |

## `ppo` by opponent type

| Opponent | Battles | Wins | Win rate | 95% interval | Avg Pokémon left | Avg opponent left | Avg turns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 500 | 488 | 97.6% | 95.9% to 98.6% | 3.95 | 0.02 | 25.7 |
| max_base_power | 500 | 408 | 81.6% | 78.0% to 84.8% | 2.44 | 0.26 | 18.8 |
| simple_heuristics | 500 | 225 | 45.0% | 40.7% to 49.4% | 0.95 | 1.11 | 19.2 |
| heuristic | 500 | 94 | 18.8% | 15.6% to 22.5% | 0.34 | 2.55 | 20.5 |
| hybrid | 500 | 93 | 18.6% | 15.4% to 22.2% | 0.37 | 2.65 | 20.5 |
| **all opponents** | 2500 | 1308 | **52.3%** | 50.4% to 54.3% |  |  |  |

## `hybrid` by opponent type

| Opponent | Battles | Wins | Win rate | 95% interval | Avg Pokémon left | Avg opponent left | Avg turns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 500 | 495 | 99.0% | 97.7% to 99.6% | 4.70 | 0.01 | 21.4 |
| max_base_power | 500 | 478 | 95.6% | 93.4% to 97.1% | 3.54 | 0.07 | 17.4 |
| simple_heuristics | 500 | 386 | 77.2% | 73.3% to 80.7% | 2.50 | 0.41 | 18.8 |
| heuristic | 500 | 259 | 51.8% | 47.4% to 56.1% | 1.38 | 1.33 | 25.3 |
| ppo | 500 | 407 | 81.4% | 77.8% to 84.6% | 2.65 | 0.37 | 20.5 |
| **all opponents** | 2500 | 2025 | **81.0%** | 79.4% to 82.5% |  |  |  |

## Illegal actions

For learned agents, an illegal action is a policy output that the legal-action mask or poke-env rejected (it would be replaced by a random legal move). For the heuristic, an engine fallback is a decision where scoring raised an error and a random legal move was played instead.

| Agent | Decisions | Illegal actions | Illegal action rate | Engine fallbacks |
| --- | --- | --- | --- | --- |
| random | not counted | 0 by construction | - | - |
| heuristic | 58,141 | - | - | 0 |
| ppo | 62,752 | 0 | 0.0000% | - |
| hybrid | 59,244 | 0 | 0.0000% | - |

## What separates `heuristic`'s losses from its wins

466 losses and 1,534 wins against `hybrid`, `max_base_power`, `ppo`, `simple_heuristics` (battles against `random` excluded). These are associations, not proven causes: a factor far more common in losses marks where to look.

| Happened in the battle | Share of losses | Share of wins | Difference |
| --- | --- | --- | --- |
| We trailed by 2+ Pokémon at some point | 75.1% | 4.6% | +70.5 pts |
| One opposing Pokémon knocked out 3 or more of ours | 54.3% | 4.8% | +49.5 pts |
| Lost a Pokémon we expected to survive the turn (KO judged under 25%) | 80.3% | 51.6% | +28.7 pts |
| Decided by a single Pokémon | 33.9% | 9.9% | +24.0 pts |
| Opponent used a setup move at all | 6.9% | 2.3% | +4.5 pts |
| ...and that Pokémon had used a setup move | 2.6% | 0.0% | +2.6 pts |
| Opponent set entry hazards on our side | 14.8% | 16.0% | -1.2 pts |
| Moved second after predicting we would move first | 18.2% | 19.7% | -1.4 pts |
| We led by 2+ Pokémon at some point | 25.5% | 95.2% | -69.6 pts |

| Average per battle | In losses | In wins |
| --- | --- | --- |
| Battle length in turns | 23.71 | 19.12 |
| Voluntary switches | 3.60 | 3.10 |
| Voluntary switches on consecutive decisions | 0.08 | 0.03 |
| Our moves that missed, failed or hit an immunity | 1.20 | 0.57 |
| Turns we could not act (sleep, paralysis, flinch, recharge) | 0.47 | 0.25 |
| Opponent's setup moves | 0.27 | 0.08 |
| Our setup moves | 0.13 | 0.08 |
| Our hazard moves | 0.18 | 0.05 |
| Our status-inflicting moves | 0.42 | 0.20 |
| Our recovery moves | 1.15 | 0.63 |
| Unexpected knockouts suffered | 1.45 | 0.70 |
| Times outsped unexpectedly | 0.30 | 0.39 |

### How accurate were the heuristic's predictions?

| Prediction | Cases | Within 20 pts | Took 20+ pts more | Took 20+ pts less | Mean error |
| --- | --- | --- | --- | --- | --- |
| Damage taken when staying in (turns the opponent attacked) | 20,512 | 76.0% | 10.7% | 13.3% | -0.5 pts |
| Damage taken by a voluntary switch-in on entry | 6,394 | 71.5% | 10.4% | 18.1% | -2.7 pts |

- Predicted to move first and both sides moved: 9,628 turns; the opponent moved first in **7.7%** of them.
- Voluntary switch-ins that fainted on entry: **3.1%**.

### Why did Pokémon faint when they were expected to survive?

1,756 knockouts where the heuristic stayed in judging the chance of being knocked out that turn below 25%. Each is counted once, under the first explanation that applies (critical hit, then Tera, then unseen move).

| Explanation | Share |
| --- | --- |
| Hit by an attack that Pokémon had not used before in the battle | 64.2% |
| An attack it had already shown did more than predicted | 19.0% |
| Critical hit | 12.0% |
| Opponent Terastallized that turn | 4.3% |
| No attack that turn (residual damage, recoil, hazards) | 0.6% |

## What separates `hybrid`'s losses from its wins

470 losses and 1,530 wins against `heuristic`, `max_base_power`, `ppo`, `simple_heuristics` (battles against `random` excluded). These are associations, not proven causes: a factor far more common in losses marks where to look.

| Happened in the battle | Share of losses | Share of wins | Difference |
| --- | --- | --- | --- |
| We trailed by 2+ Pokémon at some point | 73.0% | 3.4% | +69.6 pts |
| One opposing Pokémon knocked out 3 or more of ours | 60.0% | 5.0% | +55.0 pts |
| Decided by a single Pokémon | 32.3% | 9.6% | +22.7 pts |
| ...and that Pokémon had used a setup move | 0.6% | 0.0% | +0.6 pts |
| Opponent used a setup move at all | 2.1% | 3.2% | -1.1 pts |
| Opponent set entry hazards on our side | 15.1% | 16.7% | -1.6 pts |
| We led by 2+ Pokémon at some point | 28.1% | 95.2% | -67.1 pts |

| Average per battle | In losses | In wins |
| --- | --- | --- |
| Battle length in turns | 22.62 | 19.82 |
| Voluntary switches | 3.45 | 3.10 |
| Voluntary switches on consecutive decisions | 0.07 | 0.03 |
| Our moves that missed, failed or hit an immunity | 1.21 | 0.64 |
| Turns we could not act (sleep, paralysis, flinch, recharge) | 0.51 | 0.38 |
| Opponent's setup moves | 0.06 | 0.10 |
| Our setup moves | 0.17 | 0.13 |
| Our hazard moves | 0.12 | 0.06 |
| Our status-inflicting moves | 0.44 | 0.23 |
| Our recovery moves | 0.86 | 0.74 |

### Where the hybrid overrules the heuristic

The hybrid chose differently from the heuristic's top-scoring action in 7,261 of 46,783 real choices (15.5%).

| Heuristic's pick -> hybrid's pick | Count | Share of deviations |
| --- | --- | --- |
| move -> move | 3,678 | 50.7% |
| move -> move + tera | 1,523 | 21.0% |
| switch -> switch | 1,195 | 16.5% |
| move -> switch | 541 | 7.5% |
| switch -> move | 220 | 3.0% |
| switch -> move + tera | 59 | 0.8% |
| move + tera -> move + tera | 25 | 0.3% |
| move + tera -> move | 18 | 0.2% |
| move + tera -> switch | 2 | 0.0% |

## Limits of this evaluation

- All opponents are bots. Human players predict switches, bluff, and play around obvious moves; none of that is measured here.
- The loss analysis is observational and partly depends on the opponents chosen.
- Knockouts are attributed to the opposing Pokémon on the field, which includes faints from recoil, hazards and status.

<!-- interpretation -->

## Interpretation

This section is written by hand from the tables above (run
`final-20261112-195857`) and is kept when the report is regenerated.

### Which agent is strongest?

The heuristic agent and the hybrid are indistinguishable: 81.3% and 81.0%
against the other five participants, and 48.2% to 51.8% head to head with an
interval that spans 50%. Both beat every other participant clearly. Plain PPO
finishes behind poke-env's SimpleHeuristicsPlayer (52.3% vs 57.3% overall, 45%
head to head). No agent chose an illegal action in about 122,000 learned
decisions, and the heuristic engine never fell back in 58,141.

### The biggest strategic weaknesses

Ranked by how strongly each separates the heuristic agent's 466 losses from
its 1,534 wins. The hybrid's losses show the same pattern.

1. **It gets swept.** In 54% of losses a single opposing Pokémon knocked out
   three or more of ours; that happened in 5% of wins. The agent picks each
   replacement one turn at a time by its expected exchange, with no notion of
   keeping a specific answer to a specific threat, so it feeds Pokémon to
   whatever is winning. Only 2.6% of losses were sweeps after a setup move,
   because these bot opponents almost never set up (0.27 setup moves per
   loss). Setup sweepers are therefore an untested risk rather than a measured
   one; opponents did use setup three times as often in losses as in wins.

2. **It is blind-sided by moves it has not seen.** In 80% of losses it lost at
   least one Pokémon on a turn where it judged the knockout chance below 25%
   (52% of wins), 1.45 times per loss against 0.70 per win. Of those 1,756
   knockouts, 64% came from an attack that Pokémon had not yet used, 19% from
   a known attack doing more than predicted, 12% from critical hits and 4%
   from the opponent Terastallizing. Average damage predictions are
   well-centred (mean error -0.5 points), so the problem is the tail: unseen
   moves are discounted to 70% of their damage, which is reasonable for an
   average and too optimistic when deciding whether a Pokémon survives.

3. **It has no endgame.** A quarter of its losses (25.5%) came after leading
   by two or more Pokémon, and a third (33.9%) were decided by a single
   Pokémon, against 10% of wins. It optimises the next two turns only, so it
   does not protect a lead, preserve the Pokémon that wins the endgame, or
   notice when a simple trade would be enough.

4. **It wastes turns.** Its moves missed, failed or hit an immunity 1.20
   times per loss against 0.57 per win. It never anticipates a switch into an
   immune Pokémon, and it does not know ability-based immunities unless the
   species has only one possible ability.

Two things that looked like weaknesses were not, against these opponents:
misjudging turn order (wrong 7.7% of the time, equally common in wins and
losses) and entry hazards on our side (no difference).

### What training changed

The hybrid overrules the heuristic on 15.5% of real choices: half are a
different move, a fifth add Terastallization to the same move, a sixth pick a
different switch target, and 7.5% switch where the heuristic would attack.
None of this produced a measurable gain. The training opponents never punish
the weaknesses above (they do not set up or predict), so there was little for
the policy to learn from.

### Where to look next

- **Risk-aware survival checks** in `BattleContext._compute_threat`: keep the
  discounted estimate for expected damage, and use the undiscounted worst
  unseen move when computing the chance of being knocked out.
- **Sweep prevention**: when choosing a replacement after a faint, value
  outspeeding and knocking out the Pokémon that just scored, and penalise
  sending in something it also knocks out.
- **Endgame search**: with two or three Pokémon left per side the game tree is
  small enough to search a few turns ahead instead of scoring one.
- **Probable abilities** from the random-battle set data when a species has
  more than one, to stop attacking into likely immunities.
- **Opponents that exploit these gaps** (a setup-sweeper bot, self-play
  against earlier checkpoints) before spending more time on PPO.
- **Human opponents.** Everything here is against bots.