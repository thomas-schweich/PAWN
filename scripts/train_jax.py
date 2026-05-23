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
    Batch,
    TrainState,
    VariantSpec,
    flatten_opt_state,
    make_lr_schedule,
    make_optimizer,
    make_scan_step,
    make_train_step,
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


def _resolve_device() -> str:
    """Device label for MetricsLogger / GPU-stats source.

    Mirrors `scripts/train_jax_adapter._resolve_device` so the logger
    picks the right `*-smi` shell-out regardless of the JAX backend.
    """
    import jax

    backend = jax.default_backend()
    if backend == "gpu":
        dev_str = str(jax.devices()[0]).lower()
        if "rocm" in dev_str:
            return "rocm"
        return "cuda"
    if backend == "tpu":
        return "tpu"
    return "cpu"


def _require_accelerator() -> None:
    """Refuse to run training on CPU unless `PAWN_ALLOW_CPU=1` is set.

    Mirrors the v1 escape hatch pinned in plan §6 / CLAUDE.md.
    """
    import os

    import jax

    if jax.default_backend() == "cpu" and os.environ.get("PAWN_ALLOW_CPU") != "1":
        raise SystemExit(
            "JAX resolved to the CPU backend; refusing to run training. "
            "Install a GPU jaxlib plugin (--extra rocm or --extra cu128), "
            "or set PAWN_ALLOW_CPU=1 to override."
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    # `_build_config` raises a pydantic ValueError if --total-steps is
    # missing — no per-field runtime check needed here.
    cfg = _build_config(args)
    _require_accelerator()

    # Pick supernet shape.
    supernet_cfg = TINY_SUPERNET if cfg.supernet == "tiny" else SUPERNET
    variants_dict = TINY_VARIANTS if cfg.supernet == "tiny" else VARIANTS
    variants = tuple(
        VariantSpec(name, variants_dict[name], is_supernet=(name == "large"))
        for name in ("small", "base", "large")
    )

    # `PretrainConfig._check_pretrain` validates that `total_steps` is
    # not None — assert it for pyright (the model_validator constraint
    # doesn't narrow the optional type at the field level).
    assert cfg.total_steps is not None
    total_steps: int = cfg.total_steps

    # Optimiser + state.
    schedule = make_lr_schedule(cfg, total_steps)
    optimizer = make_optimizer(cfg, schedule)
    if cfg.resume:
        state = load_resume_state(Path(cfg.resume), optimizer, jax.random.key(0))
    else:
        import equinox as eqx
        # PretrainConfig doesn't carry a runtime seed field — the supernet
        # init key is fixed at 0 for reproducibility. The training stream's
        # randomness comes from the Rust engine's per-batch seed, not from
        # the model-init key.
        model = init_model(supernet_cfg, key=0)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        state = TrainState(
            model=model, opt_state=opt_state, step=jnp.int32(0),
            key=jax.random.key(0),
        )

    train_step = make_train_step(optimizer, variants)
    logger = MetricsLogger(
        log_dir=args.logs_dir, run_prefix="pretrain", device=_resolve_device()
    )
    logger.log_config(run_type="pretrain", model=cfg.model_dump())

    # HF push tracker (optional).
    push_tracker = HFPushTracker(repo_id=cfg.hf_repo) if cfg.hf_repo else None

    # SIGTERM handler — flips a flag the loop polls between chunks.
    should_shutdown = install_sigterm_handler()

    # K-step `lax.scan` body — plan §10 S6 + §4: "Whole training loop as
    # one compiled program — eliminate per-step launch and dispatch
    # overhead." The scan body never returns to the host inside a chunk;
    # per-chunk metrics flush between chunks.
    scan_step = make_scan_step(train_step)

    rng = np.random.default_rng(0)
    indices = np.arange(cfg.batch_size, dtype=np.int64)

    def _save_checkpoint(step_int: int) -> None:
        out = logger.run_dir / f"step_{step_int:08d}"
        if out.exists():
            return
        # Flatten the Optax PyTree to a name → ndarray dict so
        # save_model can drop it into `optimizer.safetensors`. The
        # resume path uses the matching `unflatten_opt_state` to rebuild
        # the PyTree against a freshly-initialised template — that's
        # what keeps Adam's first/second moment estimates + the clip
        # counter across the resume boundary.
        opt_tensors = flatten_opt_state(state.opt_state)
        save_model(
            state.model, out,
            run_config=cfg.model_dump(),
            optimizer_state=opt_tensors,
            training_state={"step": int(state.step)},
        )
        if push_tracker:
            push_checkpoint_async(out, push_tracker)

    start = int(state.step)
    t0 = time.time()
    next_step = start
    while next_step < total_steps:
        # Cap K at remaining steps so the final chunk lands exactly on
        # `total_steps`. Stack K batches on a leading axis; `scan_step`
        # runs them in one compiled program.
        chunk_k = min(cfg.k, total_steps - next_step)
        chunk_seed = int(rng.integers(0, 2**31 - 1))
        chunk_corpus = generate_corpus(
            n_games=cfg.batch_size * chunk_k,
            max_ply=cfg.seq_len, seq_len=cfg.seq_len, seed=chunk_seed,
        )
        tokens = chunk_corpus.tokens.reshape(chunk_k, cfg.batch_size, cfg.seq_len)
        targets = chunk_corpus.targets.reshape(chunk_k, cfg.batch_size, cfg.seq_len)
        attn = chunk_corpus.attn_mask.reshape(chunk_k, cfg.batch_size, cfg.seq_len)
        lmask = chunk_corpus.loss_mask.reshape(chunk_k, cfg.batch_size, cfg.seq_len)
        chunk_batches = Batch(
            tokens=jnp.asarray(tokens), targets=jnp.asarray(targets),
            attn_mask=jnp.asarray(attn), loss_mask=jnp.asarray(lmask),
        )
        state, chunk_losses = scan_step(state, chunk_batches)
        next_step += chunk_k

        # One D→H per chunk, not per step.
        chunk_losses_np = np.asarray(chunk_losses)

        # Log every step that crossed a log_interval boundary inside the
        # chunk — replays the within-chunk loss curve without per-step
        # syncs.
        chunk_start = next_step - chunk_k
        for i in range(chunk_k):
            step = chunk_start + i + 1
            if step % cfg.log_interval == 0:
                logger.log_train(
                    step=step, loss=float(chunk_losses_np[i]),
                    lr=np.asarray(schedule(step)).item(),
                    step_time=(time.time() - t0) / max(1, step - start),
                )

        if (
            next_step % cfg.checkpoint_interval == 0
            or next_step == total_steps
        ):
            if cfg.local_checkpoints or cfg.hf_repo:
                _save_checkpoint(next_step)

        if should_shutdown():
            _save_checkpoint(next_step)
            break

    if push_tracker:
        # `drain_push_queue` reports (timeouts, errors). Only `timeouts`
        # implies a worker is still running — that's the abandon-thread
        # case (`drain_succeeded=False`) so the daemon worker is killed
        # at interpreter exit. `errors > 0` means uploads raised but
        # the worker exited normally, which is safe to `wait=True` on.
        # Conflating the two (round-1 bug-detector finding) would
        # cause a transient network error to spuriously skip the
        # `wait=True` path that's still semantically correct.
        timeouts, _errors = drain_push_queue(push_tracker, timeout=300.0)
        push_tracker.shutdown(drain_succeeded=(timeouts == 0))
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
