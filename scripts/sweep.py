#!/usr/bin/env python3
"""Optuna sweep driver — runs N trials of a chosen adapter strategy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna

from pawn.sweep import STRATEGY_SUGGESTERS, AdapterObjective, make_pruner


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sweep")
    ap.add_argument("--strategy", required=True, choices=list(STRATEGY_SUGGESTERS))
    ap.add_argument("--n-trials", type=int, default=3)
    ap.add_argument("--supernet", default="tiny")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--checkpoint", default=None,
                    help="Backbone checkpoint each trial adapts (local dir "
                         "or HF repo ID). Defaults to the AdapterConfig "
                         "default (the published v2 base slice); point this "
                         "at a local tiny backbone for offline smoke sweeps.")
    # `--pgn` exposes the real Elo-stratified Lichess source v1 swept over
    # (v1 `scripts/sweep.py` `--pgn`); the prior v2 driver hardcoded
    # `--no-pgn` (random self-play games), so no realistic-data sweep was
    # reachable from the CLI. When set, each trial adapts on the parquet
    # source; when omitted, the driver falls back to `--no-pgn` random games
    # for the offline smoke path.
    ap.add_argument("--pgn", default=None,
                    help="Lichess parquet source (HF repo or local dir) each "
                         "trial adapts on. Omit for `--no-pgn` random-game "
                         "smoke sweeps.")
    ap.add_argument("--elo-min", type=int, default=None,
                    help="Lower Elo bound for the --pgn filter (ignored "
                         "without --pgn).")
    ap.add_argument("--elo-max", type=int, default=None,
                    help="Upper Elo bound for the --pgn filter (ignored "
                         "without --pgn).")
    ap.add_argument("--storage", default=None,
                    help="Optuna storage URL, e.g. sqlite:///./lora.db")
    ap.add_argument("--logs-dir", type=Path, default=Path("logs/sweep"))
    ap.add_argument("--total-steps", type=int, default=50)
    ap.add_argument("--pruner", default="median",
                    choices=("median", "hyperband", "none"),
                    help="Mid-trial pruner. The objective reports each "
                         "held-out val_loss (trial.report) and aborts the "
                         "subprocess on should_prune().")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        storage=args.storage,
        study_name=f"pawn-{args.strategy}",
        pruner=make_pruner(args.pruner),
        load_if_exists=True,
    )
    # When --pgn is given the trials adapt on the real parquet source;
    # otherwise fall back to random self-play games (`--no-pgn`). `batch_size`
    # is now a swept axis (suggest_common), so it is NOT hardcoded here — a
    # CLI `--batch-size` would clobber the per-trial suggested value in the
    # config merge.
    base_args = [
        "--supernet", args.supernet,
        "--variant", args.variant,
        *(["--checkpoint", args.checkpoint] if args.checkpoint else []),
        "--total-steps", str(args.total_steps),
        # Pick a log_interval well below the sweep's --total-steps so each
        # trial actually emits a val_loss row (the objective parses
        # metrics.jsonl for the best val_loss; no rows ⇒ TrialPruned).
        "--log-interval", str(max(1, args.total_steps // 5)),
        "--seq-len", "32",
        "--k", "5",
        "--local-checkpoints",
    ]
    if args.pgn is not None:
        base_args += ["--pgn", args.pgn]
        if args.elo_min is not None:
            base_args += ["--elo-min", str(args.elo_min)]
        if args.elo_max is not None:
            base_args += ["--elo-max", str(args.elo_max)]
    else:
        base_args += ["--no-pgn"]  # random games for sweep smoke
    # Resolve backbone depth so `suggest_unfreeze` doesn't propose
    # layer indices outside the variant's actual depth (tiny supernet
    # has 4 layers, production has 10).
    from pawn.config import SUPERNET, TINY_SUPERNET
    target = TINY_SUPERNET if args.supernet == "tiny" else SUPERNET
    obj = AdapterObjective(
        strategy=args.strategy,
        base_args=base_args,
        logs_dir=args.logs_dir,
        n_layers=target.n_layers,
    )
    study.optimize(obj, n_trials=args.n_trials)
    print(f"best_value: {study.best_value}")
    print(f"best_params: {study.best_params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
