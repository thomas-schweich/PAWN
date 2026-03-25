#!/usr/bin/env python3
"""Run Optuna hyperparameter sweep for PAWN adapter or pretraining experiments.

Each trial runs a training script as a subprocess with suggested hyperparameters.
Results are stored in a SQLite database for persistence and visualization.

Usage:
    # LoRA sweep on Lichess data
    uv run python scripts/sweep.py \\
        --adapter lora \\
        --checkpoint checkpoints/pawn-base/model.safetensors \\
        --pgn data/lichess_1200_1400.pgn \\
        --n-trials 30

    # Bottleneck sweep with custom output
    uv run python scripts/sweep.py \\
        --adapter bottleneck \\
        --checkpoint checkpoints/pawn-base/model.safetensors \\
        --pgn data/lichess_1200_1400.pgn \\
        --n-trials 50 \\
        --output-dir sweeps/bottleneck_1200

    # View results with optuna-dashboard
    uv run optuna-dashboard sqlite:///sweeps/lora/study.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna

from pawn.sweep import (
    ADAPTER_SCRIPTS,
    AdapterObjective,
    create_study,
)


def main():
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter sweep for PAWN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="View results: uv run optuna-dashboard sqlite:///sweeps/<adapter>/study.db",
    )
    p.add_argument("--adapter", type=str, required=True,
                    choices=sorted(ADAPTER_SCRIPTS.keys()),
                    help="Adapter type to sweep")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to PAWN backbone checkpoint")
    p.add_argument("--pgn", type=str, default=None,
                    help="Path to Lichess PGN (required for adapter sweeps)")
    p.add_argument("--output-dir", type=str, default="sweeps",
                    help="Base output directory for trials")
    p.add_argument("--n-trials", type=int, default=30,
                    help="Number of trials to run")
    p.add_argument("--n-jobs", type=int, default=1,
                    help="Parallel trials (match --n-gpus for multi-GPU)")
    p.add_argument("--n-gpus", type=int, default=1,
                    help="Number of GPUs. Trials are pinned to GPUs round-robin.")
    p.add_argument("--epochs", type=int, default=30,
                    help="Max epochs per trial (adapter sweeps)")
    p.add_argument("--total-steps", type=int, default=20000,
                    help="Total steps per trial (architecture/pretrain sweeps)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--pruner", type=str, default="hyperband",
                    choices=["hyperband", "median", "none"])
    p.add_argument("--study-name", type=str, default=None,
                    help="Study name (default: adapter type)")
    p.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                    help="Extra args passed to training script (after --)")

    args = p.parse_args()

    if args.adapter not in ("pretrain", "architecture") and not args.pgn:
        p.error("--pgn is required for adapter sweeps")

    study_name = args.study_name or args.adapter
    db_dir = Path(args.output_dir) / args.adapter
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = f"sqlite:///{db_dir}/study.db"

    print(f"=== PAWN Hyperparameter Sweep ===")
    print(f"Adapter: {args.adapter}")
    print(f"Trials: {args.n_trials} (parallel: {args.n_jobs}, GPUs: {args.n_gpus})")
    if args.adapter in ("pretrain", "architecture"):
        print(f"Steps/trial: {args.total_steps}")
    else:
        print(f"Epochs/trial: {args.epochs}")
    print(f"Pruner: {args.pruner}")
    print(f"Storage: {db_path}")
    print(f"Dashboard: uv run optuna-dashboard {db_path}")
    print()

    study = create_study(
        name=study_name,
        storage=db_path,
        pruner=args.pruner,
    )

    # For architecture/pretrain sweeps, pass --total-steps via extra args
    extra = list(args.extra_args)
    if args.adapter in ("pretrain", "architecture"):
        extra.extend(["--total-steps", str(args.total_steps)])

    objective = AdapterObjective(
        adapter_type=args.adapter,
        checkpoint=args.checkpoint,
        pgn=args.pgn or "",
        device=args.device,
        output_base=args.output_dir,
        epochs=args.epochs,
        n_gpus=args.n_gpus,
        extra_args=extra,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE: {len(study.trials)} trials")
    print(f"{'='*60}")

    completed = [t for t in study.trials if t.value is not None]
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    failed = [t for t in study.trials if t.state.name == "FAIL"]

    print(f"  Completed: {len(completed)}  Pruned: {len(pruned)}  Failed: {len(failed)}")

    if completed:
        sorted_trials = sorted(completed, key=lambda t: t.value or float("inf"))
        best = sorted_trials[0]
        print(f"\nBest trial: #{best.number}")
        print(f"  Val loss: {best.value:.4f}")
        print(f"  Params:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")

        print(f"\nTop 5 trials:")
        for t in sorted_trials[:5]:
            params_str = ", ".join(f"{k}={v}" for k, v in t.params.items())
            print(f"  #{t.number}: val_loss={t.value:.4f}  {params_str}")
    else:
        print("\nNo completed trials. All were pruned or failed.")
        if failed:
            t = failed[0]
            print(f"  Example failure (trial #{t.number}):")
            for k, v in t.params.items():
                print(f"    {k}: {v}")

    print(f"\nResults: {db_path}")
    print(f"Dashboard: uv run optuna-dashboard {db_path}")


if __name__ == "__main__":
    main()
