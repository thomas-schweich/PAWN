#!/usr/bin/env python3
"""Optuna sweep driver — runs N trials of a chosen adapter strategy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna

from pawn.sweep import STRATEGY_SUGGESTERS, AdapterObjective


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sweep")
    ap.add_argument("--strategy", required=True, choices=list(STRATEGY_SUGGESTERS))
    ap.add_argument("--n-trials", type=int, default=3)
    ap.add_argument("--supernet", default="tiny")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--storage", default=None,
                    help="Optuna storage URL, e.g. sqlite:///./lora.db")
    ap.add_argument("--logs-dir", type=Path, default=Path("logs/sweep"))
    ap.add_argument("--total-steps", type=int, default=50)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        storage=args.storage,
        study_name=f"pawn-{args.strategy}",
        load_if_exists=True,
    )
    base_args = [
        "--supernet", args.supernet,
        "--variant", args.variant,
        "--total-steps", str(args.total_steps),
        "--no-pgn",  # random games for sweep smoke
        "--batch-size", "8",
        "--seq-len", "32",
        "--k", "5",
        "--local-checkpoints",
    ]
    obj = AdapterObjective(
        strategy=args.strategy,
        base_args=base_args,
        logs_dir=args.logs_dir,
    )
    study.optimize(obj, n_trials=args.n_trials)
    print(f"best_value: {study.best_value}")
    print(f"best_params: {study.best_params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
