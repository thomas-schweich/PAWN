#!/usr/bin/env python3
"""Distillation entry point — distil a frozen teacher into a from-scratch student.

The canonical-ladder mechanism (plan §7/§7.1): load a frozen teacher
checkpoint, build a from-scratch student (specialized_clm shapes), and train
the student to match the teacher's soft targets via a temperature-scaled KL
(optionally mixed with ground-truth CE). Logit-only — no hidden-state
matching.

Validates the run config via pydantic (`DistillConfig`); reads Lichess data
via `pawn.lichess_data.load_lichess_corpus` (or random games with `--no-pgn`).
The teacher's conditioning / `C` is inherited from its own checkpoint and the
student corpus is built under it (load-time C-assert, Phase-A Chunk 4).
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

from pawn.checkpoint import load_model, resolve_checkpoint_source, save_model
from pawn.config import SUPERNET, TINY_SUPERNET, ModelConfig
from pawn.corpus import (
    conditioning_from_run_block,
    conditioning_to_C,
    generate_corpus,
)
from pawn.distill import (
    DistillTrainState,
    make_distill_scan_step,
    make_distill_train_step,
)
from pawn.jax_setup import require_accelerator, resolve_device, setup_jax_caching
from pawn.lichess_data import load_lichess_corpus
from pawn.lifecycle import (
    HFPushTracker,
    drain_push_queue,
    install_sigterm_handler,
    push_checkpoint_async,
)
from pawn.logging import MetricsLogger
from pawn.model import init_model
from pawn.run_config import DistillConfig
from pawn.trainer import (
    flatten_opt_state,
    make_lr_schedule,
    make_optimizer,
    slice_batch,
)


def _student_config(cfg: DistillConfig) -> ModelConfig:
    """Resolve the student's :class:`ModelConfig` from the run config.

    Either the ``student_supernet`` preset (tiny / production SUPERNET dims)
    or the four explicit dims — :class:`DistillConfig` already validated
    that exactly one is set. The student reuses the canonical
    vocab / seq / RoPE defaults (it is a standalone model, not a slice).
    """
    if cfg.student_supernet is not None:
        preset = TINY_SUPERNET if cfg.student_supernet == "tiny" else SUPERNET
        return ModelConfig(
            d_model=preset.d_model,
            n_layers=preset.n_layers,
            n_heads=preset.n_heads,
            d_ff=preset.d_ff,
            head_dim=preset.head_dim,
        )
    # Explicit dims (validated non-None + divisible by DistillConfig).
    assert (
        cfg.d_model is not None and cfg.n_layers is not None
        and cfg.n_heads is not None and cfg.d_ff is not None
    )
    return ModelConfig(
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        head_dim=cfg.d_model // cfg.n_heads,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="train_jax_distill")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--distill-from", default=None,
                    help="frozen teacher checkpoint: local v2 dir or HF repo ID")
    ap.add_argument("--objective", choices=("kl", "ce", "mix"), default=None)
    ap.add_argument("--distill-temp", type=float, default=None,
                    dest="temperature", help="KL softmax temperature")
    ap.add_argument("--distill-alpha", type=float, default=None,
                    dest="alpha", help="mix weight: alpha*ce + (1-alpha)*kl")
    ap.add_argument("--student-supernet", choices=("tiny", "production"),
                    default=None, dest="student_supernet")
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--d-ff", type=int, default=None)
    ap.add_argument("--pgn", default=None)
    ap.add_argument("--pgn-val-split", default=None)
    ap.add_argument("--elo-min", type=int, default=None)
    ap.add_argument("--elo-max", type=int, default=None)
    ap.add_argument("--min-ply", type=int, default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--k", type=int, default=None,
                    help="inner-scan length: K steps per lax.scan body")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no-pgn", action="store_true",
                    help="use random-game corpus instead of Lichess parquet")
    ap.add_argument("--local-checkpoints", action="store_true")
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--use-sdpa", action="store_true",
                    help="opt the attention block into "
                         "jax.nn.dot_product_attention (parity #43)")
    ap.add_argument("--no-flash", action="store_true",
                    help="disable the Pallas flash-attention kernel")
    ap.add_argument("--logs-dir", type=Path, default=Path("logs"))
    ap.add_argument("--log-interval", type=int, default=None)
    ap.add_argument("--checkpoint-interval", type=int, default=None,
                    dest="checkpoint_interval",
                    help="steps between student checkpoints (boundary-crossing "
                         "cadence; mirrors scripts/train_jax.py)")
    return ap.parse_args(argv)


def _build_config(args: argparse.Namespace) -> DistillConfig:
    base: dict = {"run_type": "distill"}
    if args.config is not None:
        base.update(json.loads(args.config.read_text()))
        base.setdefault("run_type", "distill")
    _CLI_STORE_TRUE_FLAGS = ("use_sdpa",)
    for flag, val in vars(args).items():
        if flag in (
            "config", "no_pgn", "local_checkpoints", "logs_dir",
            "no_flash", *_CLI_STORE_TRUE_FLAGS,
        ):
            continue
        if val is None:
            continue
        base[flag] = val
    for flag in _CLI_STORE_TRUE_FLAGS:
        if getattr(args, flag, False):
            base[flag] = True
    if args.no_flash:
        base["use_flash"] = False
    if args.local_checkpoints:
        base["local_checkpoints"] = True
    return DistillConfig(**base)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = _build_config(args)
    if cfg.total_steps is None:
        print("error: --total-steps is required", file=sys.stderr)
        return 2
    require_accelerator()
    cache_path = setup_jax_caching()
    if cache_path is not None:
        print(f"JAX compilation cache: {cache_path}")

    # Load the frozen teacher. Its conditioning / C is inherited and the
    # student corpus is built under it; a C-mismatch between the run config's
    # conditioning and the teacher's persisted conditioning would place every
    # move at a different absolute RoPE offset (silent drift), so cross-check
    # and fail loudly (Phase-A load-time C-assert).
    ckpt_dir = resolve_checkpoint_source(cfg.distill_from)
    teacher, teacher_run_block = load_model(ckpt_dir)
    teacher_conditioning = conditioning_from_run_block(teacher_run_block)
    teacher_C = conditioning_to_C(teacher_conditioning)
    run_C = conditioning_to_C(cfg.conditioning)
    if run_C != teacher_C:
        raise SystemExit(
            f"[train_jax_distill] conditioning mismatch: run config "
            f"conditioning {cfg.conditioning!r} implies C={run_C}, but the "
            f"teacher checkpoint was trained with C={teacher_C} "
            f"(conditioning {teacher_conditioning!r}). The student corpus is "
            f"built under the teacher's conditioning to avoid RoPE drift; "
            f"rerun with --conditioning matching the teacher."
        )
    # The student corpus is built under the teacher's own conditioning.
    conditioning = teacher_conditioning

    # Build the from-scratch student.
    student_cfg = _student_config(cfg)
    student = init_model(student_cfg, key=jax.random.key(0))

    # Optimizer over the student only.
    schedule = make_lr_schedule(cfg, cfg.total_steps)
    optimizer = make_optimizer(cfg, schedule)
    import equinox as eqx
    opt_state = optimizer.init(eqx.filter(student, eqx.is_inexact_array))

    state = DistillTrainState(
        student=student, teacher=teacher, opt_state=opt_state,
        step=jnp.int32(0), key=jax.random.key(0),
    )

    _DTYPE_MAP = {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": None,
    }
    compute_dtype = _DTYPE_MAP[cfg.amp_dtype]
    use_flash = cfg.use_flash and jax.default_backend() == "gpu"
    train_step = make_distill_train_step(
        optimizer,
        objective=cfg.objective, temperature=cfg.temperature, alpha=cfg.alpha,
        compute_dtype=compute_dtype, use_sdpa=cfg.use_sdpa, use_flash=use_flash,
    )
    # Drive the hot loop through the K-step `lax.scan` (mirrors
    # `scripts/train_jax.py`'s `make_scan_step`): the per-step body never
    # returns to the host inside a chunk, the donated buffers are amortised
    # across K steps, and metrics flush between chunks — one D→H per chunk,
    # not per step. `cfg.k` is clamped so the final chunk doesn't overrun
    # `total_steps`.
    scan_step = make_distill_scan_step(train_step)

    # Data: Lichess corpus or random games, under the teacher's conditioning.
    if args.no_pgn:
        corpus = generate_corpus(
            n_games=max(cfg.batch_size * 10, 1000),
            max_ply=cfg.seq_len, seq_len=cfg.seq_len, seed=0,
            conditioning=conditioning,
        )
        val_corpus = generate_corpus(
            n_games=max(cfg.batch_size * 4, 100),
            max_ply=cfg.seq_len, seq_len=cfg.seq_len, seed=1,
            conditioning=conditioning,
        )
    else:
        corpus = load_lichess_corpus(
            cfg.pgn, split="train",
            elo_min=cfg.elo_min, elo_max=cfg.elo_max, min_ply=cfg.min_ply,
            seq_len=cfg.seq_len, max_games=cfg.max_games,
            conditioning=conditioning,
        )
        val_corpus = load_lichess_corpus(
            cfg.pgn, split=cfg.pgn_val_split or "validation",
            elo_min=cfg.elo_min, elo_max=cfg.elo_max, min_ply=cfg.min_ply,
            seq_len=cfg.seq_len, max_games=cfg.val_games,
            conditioning=conditioning,
        )

    logger = MetricsLogger(
        log_dir=args.logs_dir, run_prefix=f"distill-{cfg.objective}",
        device=resolve_device(),
    )
    logger.log_config(run_type="distill", config=cfg.model_dump())

    push_tracker = HFPushTracker(repo_id=cfg.hf_repo) if cfg.hf_repo else None
    should_shutdown = install_sigterm_handler()

    def _save(step_int: int) -> None:
        """Save the trained student as an ordinary v2 checkpoint.

        The student is a standalone :class:`pawn.model.PAWNModel`, so the
        save path is the plain weight-folded one. The run config (carrying
        the teacher's inherited conditioning) + optimizer state + step land
        alongside so a downstream consumer treats it as any published
        checkpoint and an eval reads the right sequence layout.
        """
        run_cfg = cfg.model_dump()
        # Persist the conditioning the student was actually trained under
        # (inherited from the teacher) so eval rebuilds the right layout.
        run_cfg["conditioning"] = list(conditioning)
        out = logger.run_dir / f"distill_step_{step_int:08d}"
        save_model(
            state.student, out,
            run_config=run_cfg,
            optimizer_state=flatten_opt_state(state.opt_state),
            training_state={"step": int(state.step)},
        )
        if push_tracker:
            push_checkpoint_async(out, push_tracker)

    rng = np.random.default_rng(0)
    val_rng = np.random.default_rng(1)
    eval_interval = cfg.eval_interval or cfg.log_interval
    t0 = time.time()
    final_step = 0

    from pawn.trainer import Batch

    def _chunk_batch(k_inner: int) -> Batch:
        """Sample ``k_inner * batch_size`` games and stack them into a
        ``(K, B, T)`` :class:`Batch` for one `scan_step` chunk. Mirrors the
        ``(K, B, T)`` layout that `make_scan_step` consumes in
        `scripts/train_jax.py`."""
        idx = rng.integers(
            0, corpus.n_games, size=k_inner * cfg.batch_size
        )
        flat = slice_batch(corpus, idx)
        return Batch(
            tokens=flat.tokens.reshape(k_inner, cfg.batch_size, -1),
            targets=flat.targets.reshape(k_inner, cfg.batch_size, -1),
            attn_mask=flat.attn_mask.reshape(k_inner, cfg.batch_size, -1),
            loss_mask=flat.loss_mask.reshape(k_inner, cfg.batch_size, -1),
        )

    @eqx.filter_jit
    def val_step(
        student_m, teacher_m, batch
    ):
        from pawn.distill import distill_loss, frozen_teacher
        teacher_fn = frozen_teacher(
            teacher_m, compute_dtype=compute_dtype,
            use_sdpa=cfg.use_sdpa, use_flash=use_flash,
        )
        return distill_loss(
            student_m, teacher_fn, batch,
            objective=cfg.objective, temperature=cfg.temperature,
            alpha=cfg.alpha, compute_dtype=compute_dtype,
            use_sdpa=cfg.use_sdpa, use_flash=use_flash,
        )

    completed = 0
    while completed < cfg.total_steps:
        # Clamp the final chunk so the K-step scan never overruns
        # `total_steps`.
        this_k = min(cfg.k, cfg.total_steps - completed)
        chunk = _chunk_batch(this_k)
        state, chunk_losses = scan_step(state, chunk)
        # One D→H per chunk, not per step.
        chunk_losses_np = np.asarray(chunk_losses)
        chunk_start = completed
        completed += this_k
        final_step = completed

        # Replay the within-chunk loss curve at every log boundary the
        # chunk crossed, without per-step host syncs.
        for i in range(this_k):
            step = chunk_start + i + 1
            if step % cfg.log_interval == 0:
                logger.log_train(
                    step=step, loss=float(chunk_losses_np[i]),
                    lr=np.asarray(schedule(step)).item(),
                    step_time=(time.time() - t0) / max(1, step),
                )

        # Validate / checkpoint on division-based boundary crossing so
        # the chunk size doesn't have to divide the intervals.
        crossed_eval = (
            final_step // eval_interval != chunk_start // eval_interval
        )
        if crossed_eval:
            val_idx = val_rng.integers(
                0, val_corpus.n_games, size=cfg.batch_size
            )
            val_batch = slice_batch(val_corpus, val_idx)
            val_loss = float(val_step(state.student, state.teacher, val_batch))
            logger.log_val(
                step=final_step, val_loss=val_loss,
                val_source=cfg.pgn_val_split if not args.no_pgn else "random",
            )

        crossed_ckpt = (
            final_step // cfg.checkpoint_interval
            != chunk_start // cfg.checkpoint_interval
        )
        if crossed_ckpt:
            _save(final_step)
        if should_shutdown():
            break

    if final_step > 0 and final_step % cfg.checkpoint_interval != 0:
        _save(final_step)
    if push_tracker:
        timeouts, _errors = drain_push_queue(push_tracker, timeout=300.0)
        push_tracker.shutdown(drain_succeeded=(timeouts == 0))
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
