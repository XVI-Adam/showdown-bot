"""Behaviour cloning: train the masked PPO network to imitate a teacher.

Usage:
    uv run python scripts/train_bc.py --run-name bc-v1 --train data/synthetic/v2-train.npz \
        --test data/synthetic/v2-test.npz

Trains on --train (split 90/10 by battle for early stopping), then reports
agreement with the teacher on --test (separate battles, generated separately)
and on our agent's decisions in recorded human battles (data/human/), whose
states come from people rather than bots. Saves checkpoints/<run>/final.zip,
loadable by every script that takes a checkpoint, and run.json with the data
used, a leakage check and all metrics.

The new checkpoint is not used anywhere until you choose to: evaluate it first
(scripts/evaluate.py). Existing checkpoints are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from psbot.config import PROJECT_ROOT
from psbot.dataset.human import HUMAN_DIR
from psbot.dataset.schema import load_dataset
from psbot.training.bc import BCConfig, evaluate, new_model, train
from psbot.training.dataset import DecisionDataset, from_records, leakage_report, split_by_battle

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def show(name: str, metrics: dict) -> None:
    if not metrics.get("decisions"):
        print(f"{name}: no decisions")
        return
    stages = ", ".join(f"{k} {v['accuracy']:.0%}" for k, v in metrics["by_stage"].items() if v["n"])
    kinds = ", ".join(f"{k} {v['accuracy']:.0%} (n={v['n']})" for k, v in metrics["by_teacher_action"].items())
    print(
        f"{name}: {metrics['decisions']:,} choices in {metrics['battles']:,} battles | agreement "
        f"{metrics['accuracy']:.1%}, top-2 {metrics['top2_accuracy']:.1%}, nll {metrics['nll']:.3f}, "
        f"illegal probability {metrics['illegal_mass']:.1g}\n    by stage: {stages}\n    when the teacher chose: {kinds}"
    )


def main(args: argparse.Namespace) -> int:
    run_dir = CHECKPOINT_DIR / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"error: {run_dir} already has files; pick another --run-name", file=sys.stderr)
        return 1
    data = DecisionDataset.load(Path(args.train))
    train_data, val_data = split_by_battle(data, args.validation)
    test_data = DecisionDataset.load(Path(args.test)) if args.test else None
    human_records = [record for _, record in load_dataset(HUMAN_DIR)] if HUMAN_DIR.exists() else []
    human_data = from_records(human_records, "human") if human_records else None

    leaks = {"validation": leakage_report(train_data, val_data)}
    if test_data is not None:
        leaks["test"] = leakage_report(train_data, test_data)
    if human_data is not None:
        leaks["human"] = leakage_report(train_data, human_data)
    for name, report in leaks.items():
        print(f"leakage check, train vs {name}: {report['shared_battles']} shared battles, "
              f"{report['identical_share']:.2%} identical observations")  # fmt: skip
        if report["shared_battles"]:
            print("error: a battle is in two splits", file=sys.stderr)
            return 1
    print(f"train {len(train_data):,} decisions / {train_data.n_battles:,} battles, "
          f"validation {len(val_data):,} / {val_data.n_battles:,}")  # fmt: skip

    config = BCConfig(epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed)
    model = new_model(seed=args.seed)
    history = train(model, train_data, val_data, config)

    results = {"validation": evaluate(model, val_data, config.gamma)}
    if test_data is not None:
        results["test"] = evaluate(model, test_data, config.gamma)
    if human_data is not None:
        results["human_states"] = evaluate(model, human_data, config.gamma)
    for name, metrics in results.items():
        show(name, metrics)

    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "final")
    (run_dir / "run.json").write_text(json.dumps({
        "run_name": args.run_name,
        "method": "behaviour cloning (masked NLL) + critic fitted to discounted outcome",
        "teacher": data.meta.get("teacher"),
        "train_file": {"path": args.train, "sha256_16": digest(Path(args.train))},
        "test_file": {"path": args.test, "sha256_16": digest(Path(args.test))} if args.test else None,
        "human_battles_evaluated": len(human_records),
        "observation_size": int(train_data.obs.shape[1]),
        "n_actions": int(train_data.mask.shape[1]),
        "config": asdict(config),
        "leakage": leaks,
        "history": history,
        "results": results,
    }, indent=1))  # fmt: skip
    print(f"saved {run_dir / 'final.zip'}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--train", required=True, help="a .npz from generate_teacher_data.py")
    parser.add_argument("--test", help="a separately generated .npz for held-out evaluation")
    parser.add_argument("--validation", type=float, default=0.1, help="share of training battles for early stopping")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    sys.exit(main(parser.parse_args()))
