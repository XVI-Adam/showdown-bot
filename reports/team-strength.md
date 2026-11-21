# Team strength of the human battles

**Question:** when the bot loses to a person, was it dealt the weaker team, or was it outplayed?

**Method:** for each battle both teams were rebuilt. Ours is exact. Theirs keeps everything the battle revealed, with unrevealed moves, items, abilities, Tera types and unseen Pokémon sampled from the measured random-battle sets. The same bot then played both sides 300 times (10 samples of the hidden details x 30 battles). **Team strength** is the share our team won: 50% means an even draw. The 95% interval covers both the replay noise and the uncertainty about their hidden details. See `src/psbot/analytics/team_strength.py`; the method was first checked on bot battles where both teams are known.

## Verdict

- **A** (2 battles, bot won 2, expected 1.1): our team was the weaker one in 1 of 2; the bot won more than its teams predicted.
- **B** (5 battles, bot won 0, expected 1.2): our team was the weaker one in 5 of 5; bad draws are enough to explain it (26% chance from team luck alone); more battles are needed to tell whether skill played a part.
- **C** (2 battles, bot won 2, expected 1.2): our team was the weaker one in 0 of 2; the bot won more than its teams predicted.

## Battle by battle

| Opponent | Battle | Result | Our team strength (raw) | 95% interval | Calibrated win chance | Filled in per rebuild |
| --- | --- | --- | --- | --- | --- | --- |
| A | 2026-11-19 | won | 64% | 48% to 79% | 62% | 6 their moves sampled, 3 their ability sampled, 5 their item sampled, 6 their Tera type sampled |
| A | 2026-11-19 | won | 40% | 30% to 51% | 45% | 6 their moves sampled, 5 their ability sampled, 6 their Tera type sampled |
| B | 2026-11-19 | lost | 8% | 3% to 13% | 17% | 6 their moves sampled, 6 their ability sampled, 6 their Tera type sampled, 3 their item sampled |
| C | 2026-11-19 | won | 53% | 43% to 63% | 54% | 6 their moves sampled, 4 their ability sampled, 6 their Tera type sampled, 3 their item sampled |
| B | 2026-11-21 | lost | 13% | 7% to 19% | 23% | 5 their moves sampled, 5 their ability sampled, 5 their Tera type sampled, 1 their item sampled |
| C | 2026-11-21 | won | 73% | 63% to 82% | 68% | 6 their moves sampled, 5 their ability sampled, 5 their Tera type sampled, 3 their item sampled |
| B | 2026-11-21 | lost | 26% | 18% to 36% | 35% | 6 their moves sampled, 4 their item sampled, 5 their Tera type sampled, 4 their ability sampled |
| B | 2026-11-21 | lost | 7% | 3% to 11% | 15% | 5 their moves sampled, 5 their Tera type sampled, 4 their ability sampled, 3 their item sampled, 1 their unseen Pokémon sampled |
| B | 2026-11-21 | lost | 18% | 11% to 25% | 27% | 5 their moves sampled, 3 their item sampled, 4 their Tera type sampled, 3 their ability sampled, 1 their unseen Pokémon sampled |

## By opponent

| Opponent | Battles | Bot's wins | Expected from team strength | Chance of doing this badly or worse if teams alone decided |
| --- | --- | --- | --- | --- |
| A | 2 | 2 | 1.1 | 100.0% |
| B | 5 | 0 | 1.2 | 25.8% |
| C | 2 | 2 | 1.2 | 100.0% |

Expected wins and the last column use the *calibrated* win chance (see below). The last column is the chance of the bot winning this few if each battle went to the dice of its team strength alone. A small value (under about 5%) means the bot did worse than its teams deserved: the opponent's play explains it. A large value means bad draws are a sufficient explanation, though skill is not ruled out.

Two battles recorded in the older log format (before Phase 4) are not included: they lack our full team.

## How far to trust it

Checked first on 60 bot battles (`heuristic` vs itself), where both teams are known and the winner is known (6 x 20 replays each).

| Check | Result |
| --- | --- |
| Sampling their hidden details vs using their true team | correlation 0.82, mean difference +1.2% (no bias) |
| Does team strength predict the winner? | Brier score 0.211 (0.25 = no information) |
| Spread of team strength across battles | 13% to 86% (10th to 90th percentile) |
| Raw strength 0% to 30% | 15 battles, mean 16%, actually won 20% |
| Raw strength 30% to 70% | 29 battles, mean 48%, actually won 59% |
| Raw strength 70% to 100% | 16 battles, mean 84%, actually won 69% |

Raw strengths are **too extreme**: the calibration slope is 0.68 (standard error 0.24; 1 would mean raw values are right). The replays have the same bot on both sides, so they credit the team with what is really decided by play and luck. The calibrated win chance corrects for this. It is fitted on only 60 battles, so treat it as approximate.
