"""Train a PPO agent (plain or hybrid) against scripted opponents.

Usage:
    uv run python scripts/train_ppo.py --run-name ppo-v1                  # plain PPO, 300k steps
    uv run python scripts/train_ppo.py --run-name hybrid-v1 --agent hybrid
    uv run python scripts/train_ppo.py --run-name smoke --timesteps 8192  # quick check
    uv run python scripts/train_ppo.py --run-name ppo-v1b --resume checkpoints/ppo-v1/final.zip

Agents:
    ppo              112 basic features, masked policy (Phase 3)
    hybrid-features  basic features + the heuristic engine's analysis as inputs
    hybrid           the same inputs, and heuristic scores added to the logits

Outputs:
    checkpoints/<run>/ppo_<steps>_steps.zip   periodic checkpoints (0 = untrained)
    checkpoints/<run>/final.zip               model at the end of training
    checkpoints/<run>/run.json                configuration and training summary
    logs/train/<run>/monitor_<i>.csv          per-episode reward, length, result
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import gymnasium
import poke_env
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from psbot.config import LOG_DIR, PROJECT_ROOT
from psbot.logging_setup import setup_logging
from psbot.opponents import SCRIPTED_OPPONENTS
from psbot.rl.callbacks import CriticWarmupCallback, WinRateCallback
from psbot.rl.env import REWARD_PRESETS, RewardConfig, env_factory
from psbot.rl.hybrid import ENCODERS
from psbot.rl.policy import NET_ARCH, HybridActorCriticPolicy, MaskedActorCriticPolicy
from psbot.server import ServerUnavailableError, require_server

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
ROLLOUT_STEPS = 4096  # environment steps collected (across all envs) per PPO update

# agent name -> (observation encoding, policy class)
AGENTS = {
    "ppo": ("basic", MaskedActorCriticPolicy),
    "hybrid-features": ("hybrid", MaskedActorCriticPolicy),
    "hybrid": ("hybrid", HybridActorCriticPolicy),
}


def read_conversion_errors(log_dir: Path) -> dict[str, int]:
    """Total action-conversion fallbacks, from each environment's last logged battle."""
    totals = {"agent": 0, "opponent": 0}
    for path in log_dir.glob("monitor_*.monitor.csv"):
        with path.open() as fh:
            fh.readline()  # JSON header
            rows = list(csv.DictReader(fh))
        if rows:
            totals["agent"] += int(rows[-1].get("agent_conversion_errors") or 0)
            totals["opponent"] += int(rows[-1].get("opponent_conversion_errors") or 0)
    return totals


def build_env(
    opponents: list[str], n_envs: int, log_dir: Path, reward: RewardConfig, observation: str
) -> VecEnv:
    """One subprocess per environment; opponents are assigned round-robin."""
    log_dir.mkdir(parents=True, exist_ok=True)
    factories = [
        env_factory(
            opponents[i % len(opponents)], reward, str(log_dir / f"monitor_{i}"), observation
        )
        for i in range(n_envs)
    ]
    return SubprocVecEnv(factories)


def main(args: argparse.Namespace) -> int:
    run_dir = CHECKPOINT_DIR / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        print(f"error: {run_dir} already has files; pick another --run-name or pass --overwrite", file=sys.stderr)
        return 1
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = LOG_DIR / "train" / args.run_name
    reward = REWARD_PRESETS[args.reward]
    observation, policy_class = AGENTS[args.agent]
    observation_size = ENCODERS[observation][0]
    policy_kwargs = {"prior_scale": args.prior_scale} if args.agent == "hybrid" else {}
    assigned = [args.opponents[i % len(args.opponents)] for i in range(args.n_envs)]

    env = build_env(args.opponents, args.n_envs, log_dir, reward, observation)
    try:
        obs_space, act_space = env.observation_space, env.action_space
        print(f"observation: {obs_space['observation'].shape} in [-1, 1] | action mask: {obs_space['action_mask'].shape} | actions: {act_space.n}")
        assert obs_space["observation"].shape == (observation_size,)
        assert obs_space["action_mask"].shape == (act_space.n,)
        print(f"agent: {args.agent} | reward: {args.reward} | envs: {args.n_envs} | opponents: {assigned}")

        n_steps = ROLLOUT_STEPS // args.n_envs
        if args.resume:
            model = PPO.load(args.resume, env=env, device="cpu")
            print(f"resumed from {args.resume} at {model.num_timesteps} timesteps")
        else:
            model = PPO(
                policy_class,
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=args.learning_rate,
                n_steps=n_steps,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=args.ent_coef,
                seed=args.seed,
                device="cpu",  # small MLPs train faster on CPU than on MPS
                verbose=1,
            )
            model.save(run_dir / "ppo_0_steps")  # untrained baseline for comparison

        n_params = sum(p.numel() for p in model.policy.parameters())
        print(f"policy: MLP {NET_ARCH}, {n_params:,} parameters | rollout {n_steps} x {args.n_envs} steps")

        win_rate = WinRateCallback()
        checkpoints = CheckpointCallback(
            save_freq=max(args.checkpoint_every // args.n_envs, 1),  # counted per env
            save_path=str(run_dir),
            name_prefix="ppo",
        )
        started = time.monotonic()
        interrupted = False
        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=CallbackList([checkpoints, win_rate, CriticWarmupCallback(args.critic_warmup)]),
                log_interval=args.log_interval,
                reset_num_timesteps=not args.resume,
            )
        except KeyboardInterrupt:
            interrupted = True
            print("\ninterrupted: saving what we have")
        elapsed = time.monotonic() - started
        conversion_errors = read_conversion_errors(log_dir)

        final_path = run_dir / ("interrupted.zip" if interrupted else "final.zip")
        model.save(final_path)
        summary = {
            "run_name": args.run_name,
            "completed": not interrupted,
            "timesteps": int(model.num_timesteps),
            "elapsed_s": round(elapsed, 1),
            "steps_per_second": round(model.num_timesteps / max(elapsed, 1e-9), 1) if not args.resume else None,
            "n_envs": args.n_envs,
            "opponents": assigned,
            "resumed_from": args.resume,
            "seed": args.seed,
            "agent": args.agent,
            "observation": observation,
            "observation_size": observation_size,
            "reward_preset": args.reward,
            "critic_warmup": args.critic_warmup,
            "conversion_errors": conversion_errors,
            "prior_scale_init": args.prior_scale if args.agent == "hybrid" else None,
            "prior_scale_final": (
                model.policy.prior_scale.item() if hasattr(model.policy, "prior_scale") else None
            ),
            "n_actions": int(act_space.n),
            "net_arch": NET_ARCH,
            "reward": asdict(reward),
            "hyperparameters": {
                "learning_rate": args.learning_rate,
                "n_steps": n_steps,
                "batch_size": 256,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": args.ent_coef,
            },
            "training_curve": win_rate.history,
            "versions": {
                "poke_env": getattr(poke_env, "__version__", "0.16.1"),
                "stable_baselines3": stable_baselines3.__version__,
                "torch": torch.__version__,
                "gymnasium": gymnasium.__version__,
            },
        }
        (run_dir / "run.json").write_text(json.dumps(summary, indent=2))

        print(f"\n{'INTERRUPTED' if interrupted else 'COMPLETED'}: {model.num_timesteps:,} timesteps in {elapsed / 60:.1f} min")
        if win_rate.history:
            first, last = win_rate.history[0], win_rate.history[-1]
            print(
                f"training win rate (last 100 battles, sampling actions, vs {sorted(set(assigned))}): "
                f"{first['win_rate']:.0%} at {first['timesteps']:,} steps -> {last['win_rate']:.0%} at {last['timesteps']:,} steps"
            )
        print(
            f"action-conversion fallbacks: agent {conversion_errors['agent']}, "
            f"opponent {conversion_errors['opponent']} (in {model.num_timesteps:,} steps)"
        )
        print(f"model: {final_path}\nsummary: {run_dir / 'run.json'}")
        print("This is a training-time number. Measure real strength with scripts/evaluate.py.")
        return 130 if interrupted else 0
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--agent", choices=sorted(AGENTS), default="ppo")
    parser.add_argument("--reward", choices=sorted(REWARD_PRESETS), default="shaped")
    parser.add_argument(
        "--prior-scale", type=float, default=5.0, help="hybrid only: initial weight of heuristic scores in the logits"
    )
    parser.add_argument("--ent-coef", type=float, default=0.01, help="entropy bonus (exploration)")
    parser.add_argument(
        "--critic-warmup", type=int, default=0, help="timesteps to train only the value function first"
    )
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=sorted(SCRIPTED_OPPONENTS),
        default=["simple_heuristics", "heuristic"],
        help="assigned to environments round-robin",
    )
    parser.add_argument("--checkpoint-every", type=int, default=50_000, help="timesteps between checkpoints")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=5, help="PPO updates between progress tables")
    parser.add_argument("--resume", help="checkpoint .zip to continue training from")
    parser.add_argument("--overwrite", action="store_true")
    cli_args = parser.parse_args()

    setup_logging(verbose=False)
    try:
        require_server()
    except ServerUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(cli_args))
