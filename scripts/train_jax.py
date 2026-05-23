#!/usr/bin/env python3
"""Supernet pretraining entry point — JAX/Optax + lax.scan K-step loop.

Acceptance criterion 6 verification:
    uv run --extra rocm python scripts/train_jax.py \
        --supernet tiny --total-steps 1000 --batch-size 16 --seq-len 64 \
        --k 50 --local-checkpoints

Reads a JSON run config (validated through `PretrainConfig`), drives
the trainer in `pawn.trainer`, and writes checkpoints via
`pawn.checkpoint`. Wires the SIGTERM / HF-push / `--resume` lifecycle
from `pawn.lifecycle`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pawn.checkpoint import save_model
from pawn.config import SUPERNET, TINY_SUPERNET, VARIANTS, TINY_VARIANTS
from pawn.corpus import generate_corpus
from pawn.lifecycle import (
    HFPushTracker,
    drain_push_queue,
    install_sigterm_handler,
    load_resume_state,
    push_checkpoint_async,
)
from pawn.logging import MetricsLogger
from pawn.model import init_model
from pawn.run_config import PretrainConfig
from pawn.trainer import (
    TrainState,
    VariantSpec,
    make_lr_schedule,
    make_optimizer,
    make_train_step,
    slice_batch,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="train_jax")
    ap.add_argument("--config", type=Path, default=None, help="JSON run config")
    ap.add_argument("--supernet", choices=("tiny", "production"), default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lr-schedule", default=None)
    ap.add_argument("--warmup-frac", type=float, default=None)
    ap.add_argument("--checkpoint-interval", type=int, default=None)
    # IO
    ap.add_argument("--local-checkpoints", action="store_true")
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--logs-dir", type=Path, default=Path("logs"))
    ap.add_argument("--wandb", action="store_true")
    return ap.parse_args(argv)


def _build_config(args: argparse.Namespace) -> PretrainConfig:
    """Build a PretrainConfig from --config + CLI overrides."""
    base: dict = {}
    if args.config is not None:
        base = json.loads(args.config.read_text())
        base.setdefault("run_type", "pretrain")
    # CLI flags override config-file values.
    for flag, val in (
        ("supernet", args.supernet),
        ("total_steps", args.total_steps),
        ("batch_size", args.batch_size),
        ("seq_len", args.seq_len),
        ("k", args.k),
        ("lr", args.lr),
        ("lr_schedule", args.lr_schedule),
        ("warmup_frac", args.warmup_frac),
        ("checkpoint_interval", args.checkpoint_interval),
        ("hf_repo", args.hf_repo),
        ("wandb", args.wandb),
        ("resume", str(args.resume) if args.resume else None),
    ):
        if val is not None:
            base[flag] = val
    if args.local_checkpoints:
        base["local_checkpoints"] = True
    base.setdefault("run_type", "pretrain")
    return PretrainConfig(**base)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = _build_config(args)
    if cfg.total_steps is None:
        print("error: --total-steps is required", file=sys.stderr)
        return 2

    # Pick supernet shape.
    supernet_cfg = TINY_SUPERNET if cfg.supernet == "tiny" else SUPERNET
    variants_dict = TINY_VARIANTS if cfg.supernet == "tiny" else VARIANTS
    variants = tuple(
        VariantSpec(name, variants_dict[name], is_supernet=(name == "large"))
        for name in ("small", "base", "large")
    )

    # Optimiser + state.
    schedule = make_lr_schedule(cfg, cfg.total_steps)
    optimizer = make_optimizer(cfg, schedule)
    if cfg.resume:
        state = load_resume_state(Path(cfg.resume), optimizer, jax.random.key(0))
    else:
        import equinox as eqx
        model = init_model(supernet_cfg, key=cfg.base_seed if hasattr(cfg, "base_seed") else 0)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        state = TrainState(
            model=model, opt_state=opt_state, step=jnp.int32(0),
            key=jax.random.key(0),
        )

    train_step = make_train_step(optimizer, variants)
    logger = MetricsLogger(
        log_dir=args.logs_dir, run_prefix="pretrain", device="cuda"
    )
    logger.log_config(run_type="pretrain", model=cfg.model_dump())

    # HF push tracker (optional).
    push_tracker = HFPushTracker(repo_id=cfg.hf_repo) if cfg.hf_repo else None

    # SIGTERM handler — flips a flag the loop polls between chunks.
    should_shutdown = install_sigterm_handler()

    # Pre-generate one chunk of training corpus per K steps (random
    # games are i.i.d. so no shuffle needed).
    chunk_size = cfg.k * cfg.batch_size
    rng = np.random.default_rng(0)

    start = int(state.step)
    last_save = start
    losses: list[float] = []
    t0 = time.time()
    for step in range(start, cfg.total_steps):
        # Fresh batch each step.
        seed = int(rng.integers(0, 2**31 - 1))
        corpus = generate_corpus(
            n_games=cfg.batch_size, max_ply=cfg.seq_len,
            seq_len=cfg.seq_len, seed=seed,
        )
        batch = slice_batch(corpus, np.arange(cfg.batch_size))
        state, loss = train_step(state, batch)
        losses.append(float(loss))

        if (step + 1) % cfg.log_interval == 0:
            logger.log_train(
                step=step + 1, loss=float(loss),
                lr=float(schedule(int(state.step))),
                step_time=(time.time() - t0) / max(1, step + 1 - start),
            )

        if (step + 1) % cfg.checkpoint_interval == 0 or (step + 1) == cfg.total_steps:
            if cfg.local_checkpoints or cfg.hf_repo:
                out = args.logs_dir / f"step_{step + 1:08d}"
                save_model(
                    state.model, out,
                    run_config=cfg.model_dump(),
                    training_state={"step": int(state.step)},
                )
                if push_tracker:
                    push_checkpoint_async(out, push_tracker)
                last_save = step + 1

        if should_shutdown():
            # Save once before exiting.
            out = args.logs_dir / f"step_{int(state.step):08d}"
            if not out.exists():
                save_model(
                    state.model, out,
                    run_config=cfg.model_dump(),
                    training_state={"step": int(state.step)},
                )
            break

    if push_tracker:
        drain_push_queue(push_tracker, timeout=300.0)
        push_tracker.shutdown()
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
