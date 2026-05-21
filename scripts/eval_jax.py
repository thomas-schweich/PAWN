"""JAX move-prediction accuracy evaluator.

Phase-4 verification entry point. Loads a converted JAX checkpoint
(e.g. ``$HF_HOME/pawn-jax-converted/pawn-small``) or a freshly-init
TINY model, generates a finite Rust-engine corpus, and reports
overall + per-phase move accuracy.

The remaining ``pawn.eval_suite`` surface (probes, generation
diagnostics, Lichess Elo-stratified eval) sits on the same forward
path and ports incrementally on top of this entry point.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np

from pawn.checkpoint import load_model
from pawn.config import TINY_SUPERNET, TINY_VARIANTS, VARIANTS, SUPERNET
from pawn.corpus import generate_corpus
from pawn.eval import evaluate_accuracy
from pawn.model import PAWNModel, init_model, sliced


def _slug() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{ts}_{os.getpid()}"


def _load_or_init_model(args: argparse.Namespace) -> tuple[PAWNModel, str]:
    """Return (model, description)."""
    if args.checkpoint:
        path = Path(args.checkpoint).expanduser()
        if not path.exists():
            raise SystemExit(f"--checkpoint {path} does not exist")
        return load_model(path), f"checkpoint={path}"
    # Synthetic init for verification on a clean install.
    if args.supernet == "tiny":
        cfg = TINY_VARIANTS[args.variant] if args.variant != "supernet" else TINY_SUPERNET
    else:
        cfg = VARIANTS[args.variant] if args.variant != "supernet" else SUPERNET
    key = jax.random.PRNGKey(args.model_seed)
    model = init_model(cfg, key)
    return model, f"random-init supernet={args.supernet} variant={args.variant}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="path to a JAX checkpoint directory. If omitted, a "
        "freshly-initialised model is used (verification-only).",
    )
    parser.add_argument(
        "--supernet", choices=["tiny", "supernet"], default="tiny",
        help="only used when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--variant", default="base",
        help="variant slug from VARIANTS / TINY_VARIANTS, or 'supernet'.",
    )
    parser.add_argument("--n-games", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--corpus-seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument(
        "--logs-dir", default="logs",
        help="root directory for per-run output (eval_result.json).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.n_games <= 0:
        raise SystemExit(f"--n-games={args.n_games} must be positive")
    if args.seq_len <= 0:
        raise SystemExit(f"--seq-len={args.seq_len} must be positive")
    if args.batch_size <= 0:
        raise SystemExit(f"--batch-size={args.batch_size} must be positive")

    model, descr = _load_or_init_model(args)
    if not args.quiet:
        print(f"[setup] {descr} cfg=d{model.cfg.d_model}_L{model.cfg.n_layers}")

    if args.seq_len > model.cfg.max_seq_len:
        raise SystemExit(
            f"--seq-len={args.seq_len} exceeds model max_seq_len="
            f"{model.cfg.max_seq_len}. Lower --seq-len or pick a model "
            f"with a longer context window."
        )

    t0 = time.perf_counter()
    corpus = generate_corpus(
        n_games=args.n_games,
        max_ply=args.seq_len,
        seq_len=args.seq_len,
        seed=args.corpus_seed,
    )
    if not args.quiet:
        print(
            f"[corpus] {args.n_games} games, seq_len={args.seq_len}, "
            f"{time.perf_counter() - t0:.1f}s"
        )

    t1 = time.perf_counter()
    result = evaluate_accuracy(model, corpus, batch_size=args.batch_size)
    eval_dt = time.perf_counter() - t1

    run_dir = Path(args.logs_dir) / f"jax_eval_run_{_slug()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "checkpoint": args.checkpoint,
        "supernet": args.supernet,
        "variant": args.variant,
        "model_cfg": dataclasses.asdict(model.cfg),
        "n_games": args.n_games,
        "seq_len": args.seq_len,
        "corpus_seed": args.corpus_seed,
        "model_seed": args.model_seed,
        "overall_accuracy": result.overall,
        "n_supervised": result.n_supervised,
        "phase": {
            name: {"correct": c, "total": n, "accuracy": (c / n) if n > 0 else 0.0}
            for name, (c, n) in result.phase.items()
        },
        "eval_wall_s": eval_dt,
    }
    (run_dir / "eval_result.json").write_text(json.dumps(output, indent=2))

    print(
        f"[result] overall={result.overall:.4f} "
        f"(n_supervised={result.n_supervised}, eval={eval_dt:.1f}s)"
    )
    for name, (c, n) in result.phase.items():
        acc = (c / n) if n > 0 else 0.0
        print(f"  {name:<8} {c:>8d}/{n:<8d} = {acc:.4f}")
    print(f"[done] run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
