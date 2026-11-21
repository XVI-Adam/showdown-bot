# Phase 6: learning from games

Three methods were investigated. Only datasets and offline analysis were built, plus one trained model in a new run folder (`checkpoints/bc-v1`). No production agent was changed, and no existing checkpoint was touched.

## What the data allows

The 9 recorded human battles (3 opponents with 5, 2 and 2 battles; the heuristic agent went 4-5) hold two kinds of decision:

- **Ours:** full observation, legal-action mask and chosen action. These are valid imitation targets, but they are our agent's choices, not the human's.
- **The human's:** only what we saw them do. We have neither their observation (their own team, our hidden information) nor their legal actions. So they cannot be imitation targets and they are not RL trajectories. They are used to predict what an opponent will do (method 3).

A human's action is also not automatically good, and a battle's result says little about any single decision. That is why the outcome is stored per decision only as a value-function target, never as a judgement of the move.

## 1. Behaviour cloning

**Setup.** Teacher `heuristic_v2`, playing 4,000 local battles against five opponents (heuristic, SimpleHeuristics, MaxBasePower, random, PPO), produced 91,788 decisions. A separately generated test set has 1,000 battles and 23,000 decisions. The learner is the PPO agents' own masked policy (MLP 128-128, 112 features), so the result is an ordinary checkpoint.

**Leakage checks.** Splits are by battle and depend only on the battle id. There are 0 shared battles and 0 identical observations between the training set and each of validation, test and the human-battle states.

**Agreement with the teacher** (decisions with at least two legal actions):

| Held-out set | Choices | Agreement | Top-2 | When teacher switched |
| --- | --- | --- | --- | --- |
| Validation (391 battles) | 8,802 | 59.6% | 78.5% | 32% |
| Test (1,000 separate battles) | 22,586 | 60.0% | 79.3% | 33% |
| Human-battle states (9 battles) | 231 | 55.0% | 78.8% | 31% |

The human-battle states were played by the baseline `heuristic`, not by the teacher, so that row also measures a different teacher. Probability on illegal actions was 0 everywhere.

**Network size is not the limit.** 128-128, 256-256-256 and 512-512 all reach 60.0% / 59.8% / 60.1%. The teacher decides from damage calculations, set statistics and lookahead that the 112 basic features do not contain, and no network can recover information that is not in its input.

**In battle** (500 fresh battles each; 0 illegal actions in about 63,000 decisions):

| `bc-v1` vs | Win rate | 95% interval | PPO `ppo-v1` in the final evaluation |
| --- | --- | --- | --- |
| heuristic | 20.2% | 16.9% to 23.9% | 18.8% |
| heuristic_v2 (its teacher) | 16.4% | 13.4% to 19.9% | - |
| SimpleHeuristics | 58.6% | 54.2% to 62.8% | 45.0% |
| PPO `ppo-v1` | 53.0% | 48.6% to 57.3% | - |
| hybrid `hybrid-v2-sparse` | 24.8% | 21.2% to 28.8% | 18.6% |

Cloning reached roughly PPO's strength in seconds of training instead of 10 minutes of RL, and clearly beat it against SimpleHeuristics. It is far below its teacher and the heuristic baseline (81.3%). **Not promoted.**

## 2. Reward-based fine-tuning through self-play

Investigated, not run at scale:

- `train_ppo.py --resume checkpoints/bc-v1/final.zip` loads the cloned model and trains it (smoke-tested for 2,048 steps against the heuristic, 0 conversion errors). Its critic starts pre-fitted to discounted outcomes instead of random.
- **True self-play is not supported yet.** `train_ppo.py --opponents` accepts only scripted bots. Self-play would also need a schedule of frozen snapshots, so the learner is not chasing itself.
- What to expect: the earlier hybrid runs showed RL fine-tuning from a strong start held level at best (48.5% vs heuristic). Cloning a 112-feature model starts from a much weaker point, and RL faces the same missing-information ceiling as cloning. A cloned *hybrid* (165 features, heuristic scores as input) would start from a far better place, but its features come from the baseline engine. That keeps it tied to `heuristic`, not to `heuristic_v2`.

## 3. Opponent-specific adaptation

Predict each free choice of an opponent (attack / switch / support, given its active Pokémon's HP), comparing a population model with a per-opponent model shrunk toward it. Each battle is predicted from all other battles.

| Data | Choices | Population log-loss | Personal log-loss | Gain (95% interval) |
| --- | --- | --- | --- | --- |
| 3 bot types, 180 battles (method check) | 3,536 | 0.867 | 0.805 | +0.062 (+0.046 to +0.078) |
| 3 people, 9 battles | 192 | 0.921 | 0.926 | -0.006 (-0.022 to +0.013) |

On bots with genuinely different styles the method finds the difference, and needs about 5 battles of an opponent before it pays off: log-loss 0.938 with none seen, 0.916 after 1, 0.832 after 5. On the human data there is no detectable gain, which is what 2 to 5 battles per person should give. Acting on predictions (changing play per opponent) was not attempted. A real tendency is not necessarily exploitable, and that needs its own evaluation.

## Reproduce

```bash
scripts/start_server.sh   # in another terminal
uv run python scripts/generate_teacher_data.py --name v2-train -n 800
uv run python scripts/generate_teacher_data.py --name v2-test -n 200
uv run python scripts/train_bc.py --run-name bc-v2 --train data/synthetic/v2-train.npz --test data/synthetic/v2-test.npz
uv run python scripts/evaluate.py checkpoints/bc-v2/final.zip --opponents heuristic simple_heuristics -n 500
uv run python scripts/analyze_opponent_adaptation.py
```

## Next steps worth considering

- Clone into the hybrid observation (165 features), the one lever cloning lacked.
- Allow checkpoint opponents in `train_ppo.py` with frozen snapshots, then fine-tune `bc-v1` and measure against `heuristic_v2`.
- Collect at least 5 to 10 battles per human opponent before revisiting adaptation.
