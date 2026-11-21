# PS-Bot

A hybrid heuristic + PPO agent for Pokémon Showdown **Gen 9 Random Battles**,
built on [poke-env](https://poke-env.readthedocs.io/) and trained entirely
against a local simulator.

**Status:** all five phases complete. Full results: [reports/final-evaluation.md](reports/final-evaluation.md).

## Requirements

- macOS on Apple Silicon (developed on an M4 MacBook Air)
- [uv](https://docs.astral.sh/uv/) — manages Python 3.12 and the virtualenv
- Node.js ≥ 22.18 — runs the Pokémon Showdown server

## Setup

```bash
uv sync                      # Python 3.12 venv + pinned dependencies
scripts/setup_showdown.sh    # local Showdown server into vendor/
uv run python scripts/build_randbats_stats.py   # set statistics for the opponent move model (about 2 minutes)
```

## Run

Terminal 1 — the simulator (leave it running):

```bash
scripts/start_server.sh
```

Terminal 2 — battles and tests:

```bash
uv run python scripts/run_random_battle.py           # 1 battle, per-turn log
uv run python scripts/run_random_battle.py -n 20 -q  # 20 battles, summary only
uv run python scripts/run_heuristic_battles.py -n 100        # heuristic vs 3 baselines, saves logs + stats
uv run python scripts/run_heuristic_battles.py -n 1 --show   # print one battle's reasoning
uv run python scripts/train_ppo.py --run-name ppo-v1          # train PPO, saves checkpoints
uv run python scripts/train_ppo.py --run-name hybrid-v3 --agent hybrid --prior-scale 15 \
    --critic-warmup 50000 --ent-coef 0.001 --learning-rate 1e-4 --reward sparse   # hybrid, fine-tuning settings
uv run python scripts/evaluate.py checkpoints/hybrid-v2-sparse/final.zip      # measure any agent in fresh battles
uv run python scripts/evaluate.py heuristic --opponents checkpoints/ppo-v1/final.zip   # any agent vs any agent
uv run python scripts/final_evaluation.py            # round-robin of all agents + performance report
uv run python scripts/play_against_bot.py            # play against the bot yourself (see below)
uv run pytest                                        # integration tests skip if the server is down
```

## Play against the bot

Start the server (`scripts/start_server.sh`), then in another terminal:

```bash
uv run python scripts/play_against_bot.py                                        # heuristic agent
uv run python scripts/play_against_bot.py --bot checkpoints/hybrid-v2-sparse/final.zip   # the hybrid
```

1. Open <http://localhost:8000> in your browser. It loads the official
   Showdown client, which connects to your local server.
2. Pick any username (no password is needed on this server).
3. Click **Find a user**, type `ShowdownBot`, then Challenge it in the format
   **[Gen 9] Random Battle**. Challenges in other formats are ignored.

The bot answers instantly and accepts 10 battles by default (`-n` changes
that). Each finished battle is saved to `logs/human/` with the bot's reasoning
per turn, which `psbot.analysis` can study.

Things to know:

- **The bot does not learn from your games.** Training is `scripts/train_ppo.py`
  against scripted bots and earlier checkpoints; playing it is practice for you
  and a way to see its weaknesses first-hand (it is most beatable by setup
  sweepers and by Pokémon whose moves it has not seen yet; see the report).
- **Browser caveat.** The page is the public Showdown client talking to a
  server on your own machine. Recent Chrome versions may ask permission to
  connect to local devices; allow it. In the pane of the Claude desktop app I
  tested in, that connection was blocked, so this path is verified only up to
  the server (a WebSocket handshake with the client's origin succeeds, and a
  stand-in challenger plays the bot in the test suite).
- **Keep the server local.** It runs without login checks; do not port-forward
  it. To play friends' bots or friends themselves, use the official site, not
  this server.

## Official server

`psbot.battle_adapters.official` connects our agents, unchanged, to the
official Pokémon Showdown server and plays battles there against people you
name. It never searches the ladder.

**Before any login:** Showdown's bot policy of 2026-10-09
([announcement](https://www.smogon.com/forums/threads/3789528/)) makes bot
registration mandatory. In the Showdown client type `/ht`, then choose
*Something else* > *I want to register my bot*. The adapter refuses to log in
until you set `PSBOT_BOT_REGISTERED=1`.

Create a `.env` file in the project root (git-ignored; `chmod 600 .env`):

```
PSBOT_USERNAME=YourBotAccount
PSBOT_PASSWORD=...
PSBOT_BOT_REGISTERED=1
```

```bash
uv run python scripts/check_official_login.py --handshake-only   # connect only, no account needed
uv run python scripts/check_official_login.py                    # log in, confirm the name, log out
```

Play (the other person must have agreed to battle a bot):

```bash
uv run python scripts/play_official.py --accept-from FriendName      # wait for their challenge
uv run python scripts/play_official.py --challenge FriendName        # or challenge them once
uv run python scripts/play_official.py --bot checkpoints/hybrid-v2-sparse/final.zip \
    --accept-from FriendOne,FriendTwo -n 3                           # hybrid, up to 3 battles
```

It plays one battle (`-n` for more) and exits. Rules the session enforces:

- only **[Gen 9] Random Battle** challenges from the named users are accepted;
  anything else is declined, once per challenger per minute;
- one battle at a time; a challenge is accepted once even if announced twice;
- a battle room that is not in our format or not against the agreed opponent
  is forfeited and left;
- every request from the server is answered exactly once: choices carry the
  server's request number and repeats are dropped;
- a dropped connection is retried five times (2 to 30 s apart); the server
  puts the bot back in its battle and the state is rebuilt from the log;
- if nothing happens in a battle for 3 minutes the bot starts the timer;
- Ctrl-C withdraws any challenge, forfeits a battle in progress and logs out.

Battles are saved to `data/human/official-<time>.jsonl` (see the next section).

What the adapter adds to poke-env's client:

- login as documented in the simulator's `PROTOCOL.md`, with every failure
  reported as a clear error;
- the login assertion is redacted from all logs (poke-env's and the websocket
  library's), and the password is never stored where it can be printed;
- outgoing messages are spaced 650 ms apart, inside the server's rate limit;
- the password can only be sent to the official login server.

## Human battle dataset

Every finished battle against a person becomes one JSON line in `data/human/`
(git-ignored, and separate from the bot-versus-bot logs in `logs/`). The full
layout is documented at the top of `src/psbot/dataset/schema.py`. Per battle:
id, date, format, server, the agent and its checkpoint (path and SHA-256), the
opponent's username, the outcome, both teams as known, and the public battle
log without chat. Per decision:

- both sides' Pokémon with HP, status, boosts, and the field conditions;
- the legal actions and the 26-entry action mask the learned agents use;
- the selected action;
- the heuristic engine's score for every legal action, split into its named
  components, or the policy's probability for each action (plus, for the
  hybrid, the heuristic scores it was shown);
- the observation vector at that moment;
- what the opponent was then seen to do.

Battles are partially observable, so every fact about the opponent is labelled:

| Label | Meaning |
| --- | --- |
| `revealed` | The battle log showed it: a move it used, an ability or item the log named, its Tera type. |
| `inferred` | Not shown, but certain from public random-battle data (one possible ability, a forme's required item). The species' possible moves are listed once per battle under this label. |
| `unknown` | Neither. The value is null. |

"Revealed" is read from the log, not from poke-env's Pokémon objects, because
both poke-env and our damage code fill in likely abilities and items. When the
log is ambiguous about whose ability an effect belongs to, the fact is left
unrevealed. Each decision records only what had been shown by then.

```bash
uv run python scripts/validate_dataset.py        # check and summarise data/human/
```

The validator re-reads each battle's log and rejects, among other things, a
"revealed" fact that is not in the log at that point, an illegal selected
action, and scores that do not add up. Measured on the local simulator, where
the opponent's true team is known: across 450 battles every revealed ability
(600), item (1,154) and Tera type (259) was correct, and 3,428 of 3,430
revealed moves; the two misses were Illusions that never broke, which fool the
log itself.

`psbot.dataset.to_legacy_record` converts a record to the older log layout that
`psbot.analysis` reads. The two battles played before this recorder existed
are in `logs/human/` in that older layout.

## The heuristic agent

Every legal move and switch gets a score that is the sum of named components,
all in one unit: **fractions of an HP bar**. Dealing 40% is +0.40; expecting to
take 25% is -0.25. The agent plays the highest total and logs why:

```
turn 3 | ironvaliant 100% vs palkiaorigin 70% -> move moonblast (1.4)
    damage +0.70 (91%-108% of max HP, 100% accurate)
    ko +0.70 (100% to KO before they move)
```

Damage numbers come from poke-env's port of the Smogon damage calculator.
What we cannot see about the opponent is filled in from the simulator's own
random-battle data: the standard 85 EV / 31 IV spread, each species' possible
moves, and abilities/items that are certain for a species or forme.

Results on the pinned simulator (heuristic agent's win rate):

| Opponent | Battles | Win rate | 95% CI |
| --- | --- | --- | --- |
| RandomPlayer | 100 | 100% | 96-100% |
| MaxBasePowerPlayer | 100 | 96% | 90-98% |
| SimpleHeuristicsPlayer (poke-env) | 500 | 80% | 77-84% |

Known weaknesses, for later phases to address:

- **One-turn horizon.** It assumes the opponent attacks whatever is in front
  of it and never predicts a switch, so it cannot plan sacrifices or win
  conditions.
- **Rarely sets up or lays hazards** (under 1% of decisions). The weights make
  attacking almost always look better; they are untuned guesses.
- **Blind to utility moves** such as Protect, Substitute, Taunt, Trick, screens
  and pivoting (U-turn's switch): they score as zero or as plain attacks.
- **Hidden information is assumed away**: no Choice Scarf or opposing priority
  when judging who moves first, and uncertain abilities are ignored.
- **Terastallization is offensive-only**; it never Teras defensively.

## The Phase 5 engine (`heuristic_v2`)

`heuristic` is still the baseline above, unchanged, because the trained hybrids
take its scores as input. `heuristic_v2` is the same engine with four additions
switched on (`v2_weights` in `heuristics/context.py`):

- **Opponent move model.** The simulator's own team generator is sampled
  200,000 times to count what sets each species really gets. Given what a
  Pokémon has shown, each unrevealed move gets a probability, and the threat
  to us is weighted by them. Probabilities are well calibrated (a move
  predicted at 50% is there 50% of the time). An unrevealed move is never
  treated as confirmed, even at 100%.
- **Sweep check and endgame search.** The fight is played forward with both
  sides attacking. Early on this asks what their active Pokémon costs us
  before it is stopped; once everything left is known, it searches the rest of
  the battle by minimax over replacement choices.
- **Damage refinements.** Possible unseen abilities that nullify a move,
  accuracy and evasion stages, and a discount for attacks their revealed bench
  is immune to.

Measured against the baseline in 3,000-battle runs, the move model wins 53.7%
(95% interval 51.9-55.5%). The other additions have shown no measurable effect
on top of it, and the full `heuristic_v2` wins 52.8% (51.0-54.5%). Details,
calibration tables and what remains open: [reports/phase5-engine.md](reports/phase5-engine.md).

```bash
uv run python scripts/evaluate.py heuristic_v2 --opponents heuristic simple_heuristics -n 1000
uv run python scripts/play_official.py --bot heuristic_v2 --accept-from FriendName
```

## Opponent profiles

```bash
uv run python scripts/opponent_report.py                 # everyone you have played
uv run python scripts/opponent_report.py SomeUsername    # one person in detail
```

Built from the battle logs only: leads, move and switch frequencies,
aggression, setup use, repeated switches, how predictable their move choices
are, attacks into resistances and immunities, and which of their Pokémon
knocked ours out. Below five battles a profile shows counts and draws no
conclusions; above that, every rate carries its sample size and a 95%
interval, and a tendency is stated only when the interval clears its
threshold. Profiles name real people, so they are saved (with `--write`) under
the git-ignored `data/profiles/`.

## Phase 7 evaluation

28,000 battles: 8 agents in a round-robin, 1,000 battles per pairing, each
battle measured from both sides. Full report with every metric and interval:
[reports/phase7-evaluation.md](reports/phase7-evaluation.md).

| Agent | Win rate vs the baseline's pool | vs all others |
| --- | --- | --- |
| `heuristic_v2` | 83.9% (82.9-84.9%) | 78.9% |
| `heuristic` (baseline, 81.3% in the final evaluation) | 82.8% (81.7-83.8%) | 77.2% |
| `hybrid` | 2.3 points below `heuristic` on the same opponents | 74.5% |
| `bc` | 65.0% | 52.3% |

- **`heuristic_v2` vs the baseline:** 52.7% head to head over 4,000 battles
  (51.2-54.2%). It gets swept less (13.3% vs 16.2%), loses fewer Pokémon
  right after switching in, and misses fewer sure KOs. Immunity hits and
  endgames are unchanged.
- **Default agent:** `heuristic` stays the default until `heuristic_v2` holds
  up against people.
- **Human results:** 6-5 over 11 battles, too few to rank anything.
- **Bots vs people:** bot opponents almost never set up a sweeper (under 3% of
  battles); the people did in 5 of 11. Bot evaluation barely tests the
  weakness that matters most.

Metrics include win rate per opponent, net knockouts by stage, setup
sweepers, sweeps, knockouts by not-yet-seen moves, endgame win rate (from
ahead, level or behind), missed sure KOs, moves into immunities, switching
mistakes and remaining Pokémon. They are defined once in
`src/psbot/evaluation/metrics.py` and computed from the full battle log for
every agent, scripted bots included.

```bash
uv run python scripts/evaluate_all.py --run-dir logs/eval/phase7-mine -n 1000 --shard 0/4   # x4 in parallel: 0/4 .. 3/4
uv run python scripts/evaluate_all.py --run-dir logs/eval/phase7-mine --report
```

## Learning from games (Phase 6)

Offline only; no production agent changes. Full results: [reports/phase6-learning.md](reports/phase6-learning.md).

- **Behaviour cloning.** `heuristic_v2` played 4,000 local battles as teacher,
  and the PPO agents' masked network was trained to imitate it. Data is split
  by battle and checked for leakage, and masks are kept, so no illegal action
  is possible. Result `checkpoints/bc-v1`: 60% agreement with the teacher on
  separate test battles, about PPO-level strength (20% vs heuristic, 59% vs
  SimpleHeuristics). Not promoted. Larger networks do no better: the 112 basic
  features lack what the teacher uses.
- **Self-play fine-tuning.** The cloned model resumes in `train_ppo.py`.
  Self-play against frozen checkpoints is not supported yet.
- **Opponent adaptation.** Predicting opponents' choices per person versus
  population-wide works on bots with distinct styles, after about 5 battles.
  On the 9 human battles so far there is no detectable gain.

Human opponents' actions are never used as imitation targets: from our side
we see neither their observation nor their legal actions.

```bash
uv run python scripts/generate_teacher_data.py --name v2-train -n 800   # synthetic data -> data/synthetic/
uv run python scripts/train_bc.py --run-name bc-v2 --train data/synthetic/v2-train.npz --test data/synthetic/v2-test.npz
uv run python scripts/analyze_opponent_adaptation.py                    # on data/human/
```

## The PPO agent

Built from the official poke-env Stable-Baselines3 example: a `SinglesEnv`
subclass supplies the observation and reward, poke-env supplies the 26-action
space and the legal-action mask, and a custom policy adds `-inf` to the logits
of illegal actions so they can never be chosen.

- **Observation:** 112 floats in [-1, 1]: 4 move slots and 6 team slots in the
  same order as the action indices, both active Pokémon, and global state.
  Only observable facts and basic mechanics; none of the heuristic engine's
  judgement, so Phase 4 can measure what adding it is worth.
- **Reward:** change in (our HP and survivors minus theirs), plus +/-1 for the
  result; the official example's proportions scaled so a win is worth 1.
- **Network:** MLP 128-128, about 65k parameters, trained on CPU.

Run `ppo-v1`: 503,808 steps in 10.2 minutes on 8 environments, half against
SimpleHeuristicsPlayer and half against our heuristic agent. Evaluated
afterwards in separate battles, playing its most likely action:

| Opponent | Untrained (200 battles) | Trained (500 battles) | 95% CI (trained) |
| --- | --- | --- | --- |
| RandomPlayer | 78.0% | 97.6% | 95.9-98.6% |
| MaxBasePowerPlayer | 27.0% | 84.4% | 81.0-87.3% |
| SimpleHeuristicsPlayer | 4.0% | 49.6% | 45.2-54.0% |
| Heuristic agent (Phase 2) | 1.0% | 16.4% | 13.4-19.9% |

It also beats its own 100k-step checkpoint 67.2% of the time, and chose zero
illegal actions in about 63,000 evaluation decisions.

So training works and the agent clearly learned, but it is far weaker than
the hand-written heuristic and its training curve had flattened by about 350k
steps. With these features the network must rediscover from win/loss signals
what the damage calculator already knows.

## The hybrid agent

The hybrid sees everything the PPO agent sees plus what the heuristic engine
concluded this turn (165 inputs): its score for each of the 26 actions,
expected damage and KO chance per move, entry damage and expected exchange
per switch, and the threat to our active Pokémon.

Two ways of using that were compared:

- **Features only:** the analysis is just extra input to the same masked policy.
- **Features + prior:** the heuristic scores are also added to the action
  logits, `logits = network(obs) + prior_scale * score`. The network's output
  starts near zero, so an untrained hybrid plays the heuristic's choice 100% of
  the time, and training learns where to overrule it. Nothing is forbidden:
  learned logits can outweigh any score, and `prior_scale` is itself trained.

All runs: 503,808 steps on 8 environments against SimpleHeuristicsPlayer and
the heuristic agent. Win rates in separate evaluation battles, playing the
most likely action (battles in brackets):

| Agent | vs Heuristic | vs SimpleHeuristics | vs PPO `ppo-v1` | Agrees with heuristic |
| --- | --- | --- | --- | --- |
| Heuristic (Phase 2) | 52.4% (500, mirror) | 80.2% (500) | 80.8% (500) | - |
| PPO `ppo-v1` (Phase 3) | 16.4% (500) | 49.6% (500) | - | - |
| Hybrid, features only `hybridfeat-v1` | 33.3% (800) | 66.4% (500) | 67.2% (500) | 60% |
| Hybrid, untrained | 50.4% (700) | 82.2% (400) | - | 100% |
| Hybrid, default PPO settings `hybrid-v1` | 41.9% (800) | 74.0% (500) | 76.0% (500) | 68% |
| Hybrid, fine-tuned, shaped reward `hybrid-v2-shaped` | 47.7% (1800) | 79.0% (500) | 78.0% (500) | 85% |
| Hybrid, fine-tuned, sparse reward `hybrid-v2-sparse` | 48.5% (1500) | 83.6% (500) | 80.4% (500) | 84% |

95% intervals against the heuristic: features only 30-37%, `hybrid-v1`
39-45%, shaped 45-50%, sparse 46-51%.

What the evidence supports:

- **Heuristic knowledge as input doubles what PPO learns from scratch** in the
  same budget (33% vs 16% against the heuristic, 66% vs 50% against
  SimpleHeuristics).
- **Starting from the heuristic is far better than learning it**, but training
  has not improved on it. With default PPO settings the hybrid got worse
  (42% against the heuristic). With fine-tuning settings (a sharper prior,
  50k steps of critic warm-up, lower learning rate and entropy bonus) it held
  level, changing 15% of decisions without a measurable gain or loss.
- **Reward shaping made no measurable difference** under the fine-tuning
  settings: 47.7% (shaped) vs 48.5% (win/loss only). A first comparison at 500
  battles each suggested a 7-point gap; 1,000 more battles each erased it.
- **Action selection: play the most likely action.** Sampling cost 8 points
  for `hybrid-v1` (34% vs 43%) and 6 for the features-only model.
- **Deviation from the heuristic tracks losses** across all trained hybrids:
  60% agreement scores 33%, 68% scores 42%, 85% scores 48%.

The heuristic agent and the fine-tuned hybrid are therefore about equally
strong, and no learned agent beats the heuristic yet.

## Final evaluation

A round-robin of six participants, 500 fresh battles per pairing (7,500 in
total), with learned agents frozen and playing their most likely action.
Row participant's win rate against the column participant:

| | random | max_base_power | simple_heuristics | heuristic | ppo | hybrid | overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **random** | - | 9.6% | 1.2% | 0.4% | 2.4% | 1.0% | 2.9% |
| **heuristic** | 99.6% | 97.4% | 80.0% | - | 81.2% | 48.2% | 81.3% |
| **ppo** | 97.6% | 81.6% | 45.0% | 18.8% | - | 18.6% | 52.3% |
| **hybrid** | 99.0% | 95.6% | 77.2% | 51.8% | 81.4% | - | 81.0% |

The heuristic and the hybrid are equally strong; plain PPO is well behind. No
agent chose an illegal action (0 in about 122,000 learned decisions).

The biggest weaknesses, from 466 logged losses of the heuristic agent:

1. **Gets swept:** one opposing Pokémon took three or more knockouts in 54% of
   losses (5% of wins).
2. **Blind-sided by unseen moves:** it lost a Pokémon it expected to survive in
   80% of losses; 64% of those knockouts came from an attack not yet revealed.
3. **No endgame:** 26% of losses came after leading by two or more Pokémon.
4. **Wasted turns:** twice as many missed, failed or immune attacks in losses.

All opponents were bots, so play against people is still unmeasured. The
[full report](reports/final-evaluation.md) has every table and the reasoning.

## Layout

```
src/psbot/
  config.py          battle format, loopback-only server configuration
  server.py          fail-fast reachability check
  runner.py          run N battles between two players -> BattleSummary
  logging_setup.py   console logging
  recording.py       battle log: one JSON line per battle, with decision reasons
  heuristics/
    randbats.py      random-battle knowledge: stat spread, move pools, sure abilities/items
    damage.py        expected damage via poke-env's damage calculator
    mechanics.py     speed order, entry hazards, status immunities, Tera type
    context.py       per-turn analysis (threats, exchanges) + tunable weights
    scorers.py       one function per scoring component
    engine.py        sums components, ranks actions, decides Terastallization
    lookahead.py     forward projection: sweep check and minimax endgame (v2 only)
  opponent_model/    belief over an opposing Pokémon's set, from measured set statistics
  analytics/         per-opponent profiles from recorded human battles
  training/          decision datasets, behaviour cloning, opponent-adaptation analysis
  evaluation/        Phase 7: per-battle metrics, decision probe, aggregation, report
  rl/
    observation.py   battle -> 112 floats, laid out to match the action space
    env.py           Gymnasium environment (poke-env SinglesEnv) + reward
    hybrid.py        hybrid observation: basic features + the heuristic engine's analysis
    policy.py        masked PPO policy, and the hybrid policy with a learnable heuristic prior
    callbacks.py     win-rate logging, critic warm-up
  opponents.py       opponent registry: scripted players or saved checkpoints
  dataset/           human-battle records: reveal tracking, recorder, schema and validator
  hosting.py         a bot that accepts challenges from people on the local server
  battle_adapters/   official server: credentials, login, redaction, rate limit, reconnect,
                     and the consent-based challenge/battle session
  analysis.py        what separates losses from wins, prediction accuracy, hybrid deviations
  report.py          Markdown performance report from evaluation results
  players/           agents (poke-env Player subclasses): heuristic, ppo (also plays hybrids)
scripts/             server setup/start, runnable entry points
tests/               unit tests + end-to-end tests against the simulator
vendor/              Showdown checkout (git-ignored, pinned by setup script)
reports/             evaluation reports
checkpoints/         saved models (git-ignored)
logs/                battle logs, statistics, training curves (git-ignored)
data/                human battles (human/), synthetic training data (synthetic/), set statistics (git-ignored)
```

## Security

The server runs with `--no-security` (no rate limits, no login checks) so bots
can play thousands of battles quickly. `setup_showdown.sh` therefore binds it
to `127.0.0.1`, `start_server.sh` refuses to start otherwise, and
`psbot.config` rejects non-loopback hosts. Do not port-forward it.

The official-server adapter is separate: it never uses the local server's
password-less login, reads the bot account from the environment or `.env`
(never committed), and keeps the password and login assertion out of logs.

## Pinned versions

| Component | Version |
| --- | --- |
| Python | 3.12 |
| poke-env | 0.16.1 |
| gymnasium | 1.3.0 (pinned by poke-env) |
| stable-baselines3 | 2.9.2 |
| torch | 2.15.0 |
| numpy | 2.5.4 |
| Pokémon Showdown | commit `e5d26285ab` (see `scripts/setup_showdown.sh`) |
