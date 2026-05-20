"""JAX pretraining driver for the supernet (Phase 2).

Minimal entry point that ties the Phase-2 chunks together:

  1. Generate a Rust-engine corpus (or reuse a previously-generated one).
  2. Reshape the corpus into ``[N_chunks, K, B, T]`` chunks where each
     chunk is one ``make_scan_step`` invocation.
  3. Compose the optimizer with ``make_lr_schedule``; init TrainState.
  4. Train for ``--total-steps`` (a multiple of K is recommended).
  5. Log per-chunk metrics to ``logs/jax_run_<ts>_<slug>/metrics.jsonl``.

This is the Phase-2 verification driver, not the production pretraining
loop. Production runs need: HF-backed checkpoints, an LR schedule
shaped to a real total_steps budget, wandb / dashboard integration,
preemption-safe resume — those will land in Phase-3 polish or once
Phase-2 is otherwise green.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pawn.jax.config import (
    SUPERNET,
    TINY_SUPERNET,
    TINY_VARIANTS,
    VARIANTS,
    ModelConfig,
)
from pawn.jax.corpus import generate_corpus
from pawn.jax.model import init_model
from pawn.jax.trainer import (
    Batch,
    VariantSpec,
    init_train_state,
    make_lr_schedule,
    make_optimizer,
    make_scan_step,
    make_train_step,
)


def _slug() -> str:
    """A short timestamp-only slug — enough to disambiguate concurrent
    runs without needing a random word generator."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_supernet(name: str) -> tuple[ModelConfig, dict[str, ModelConfig]]:
    if name == "tiny":
        return TINY_SUPERNET, TINY_VARIANTS
    if name == "supernet":
        return SUPERNET, VARIANTS
    raise ValueError(
        f"unknown --supernet {name!r}; expected one of tiny|supernet"
    )


def _stage_chunk(
    np_tokens: np.ndarray,
    np_attn: np.ndarray,
    np_targets: np.ndarray,
    np_loss: np.ndarray,
    offset: int,
    k: int,
    batch_size: int,
) -> Batch:
    """Pull a [K, B, T] chunk out of the host NumPy corpus."""
    end = offset + k * batch_size
    # Reshape contiguous slice into (K, B, ...).
    def reshape(arr: np.ndarray) -> jax.Array:
        sl = arr[offset:end]
        return jnp.asarray(sl.reshape((k, batch_size) + arr.shape[1:]))
    return (
        reshape(np_tokens),
        reshape(np_attn),
        reshape(np_targets),
        reshape(np_loss),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supernet",
        choices=["tiny", "supernet"],
        default="tiny",
        help="which supernet to train (tiny for verification, supernet for production)",
    )
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="per-step batch size (B). chunk size on device is K * B.",
    )
    parser.add_argument(
        "--seq-len", type=int, default=128,
        help="sequence length (T). For TINY use a short value to keep "
        "the verification run fast.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="K = number of inner steps per lax.scan invocation. "
        "Larger K amortises host overhead, but consumes K * B games "
        "from the corpus per scan call.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--logs-dir", type=str, default="logs",
        help="root directory for per-run output (metrics.jsonl, config.json).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress per-chunk stdout prints (still writes to metrics.jsonl).",
    )
    args = parser.parse_args(argv)

    if args.total_steps % args.k != 0:
        raise SystemExit(
            f"--total-steps={args.total_steps} must be a multiple of --k={args.k}"
        )

    supernet_cfg, variants = _resolve_supernet(args.supernet)
    specs = tuple(VariantSpec(cfg=v) for v in variants.values())

    run_dir = Path(args.logs_dir) / f"jax_run_{_slug()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "supernet": args.supernet,
                "supernet_cfg": {
                    "d_model": supernet_cfg.d_model,
                    "n_layers": supernet_cfg.n_layers,
                    "n_heads": supernet_cfg.n_heads,
                    "d_ff": supernet_cfg.d_ff,
                },
                "total_steps": args.total_steps,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "k": args.k,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "seed": args.seed,
                "variants": list(variants.keys()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    n_games = args.total_steps * args.batch_size
    print(
        f"[setup] supernet={args.supernet} variants={list(variants.keys())} "
        f"total_steps={args.total_steps} K={args.k} B={args.batch_size} "
        f"T={args.seq_len} n_games={n_games}"
    )

    t0 = time.perf_counter()
    print(f"[corpus] generating {n_games} games (seed={args.seed}, max_ply={args.seq_len})")
    corpus = generate_corpus(
        n_games=n_games,
        max_ply=args.seq_len,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    print(f"[corpus] done in {time.perf_counter() - t0:.1f}s")

    key = jax.random.PRNGKey(args.seed)
    model = init_model(supernet_cfg, key)
    sched = make_lr_schedule(
        peak_lr=args.lr,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
    )
    optimizer = make_optimizer(sched)
    state = init_train_state(model, optimizer)
    train_step = make_train_step(optimizer, specs)
    scan = make_scan_step(train_step, args.k)

    n_chunks = args.total_steps // args.k
    print(f"[train] n_chunks={n_chunks} (K={args.k} steps each)")

    with metrics_path.open("w", encoding="utf-8") as mf:
        wall0 = time.perf_counter()
        for chunk_i in range(n_chunks):
            offset = chunk_i * args.k * args.batch_size
            chunk = _stage_chunk(
                corpus.tokens,
                corpus.attn_mask,
                corpus.targets,
                corpus.loss_mask,
                offset,
                args.k,
                args.batch_size,
            )
            t_chunk = time.perf_counter()
            state, metrics = scan(state, chunk)
            # Block on the first metric so wall-time is real.
            metrics["loss"].block_until_ready()
            dt = time.perf_counter() - t_chunk

            # Pull metrics off-device — they're [K]-shaped per key.
            host_metrics = {k: np.asarray(v) for k, v in metrics.items()}
            step_end = int(state.step)
            step_start = step_end - args.k
            # Aggregate per-chunk: mean loss, last grad_norm, etc.
            row = {
                "chunk": chunk_i,
                "step_start": step_start,
                "step_end": step_end,
                "wall_s": time.perf_counter() - wall0,
                "chunk_wall_s": dt,
                "loss_mean": float(host_metrics["loss"].mean()),
                "loss_last": float(host_metrics["loss"][-1]),
                "grad_norm_mean": float(host_metrics["grad_norm"].mean()),
                "n_supervised_mean": float(host_metrics["n_supervised"].mean()),
            }
            # Per-variant losses (last-in-chunk).
            for key_, vals in host_metrics.items():
                if key_.startswith("loss_d"):
                    row[f"{key_}_last"] = float(vals[-1])
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            if not args.quiet:
                print(
                    f"[chunk {chunk_i + 1}/{n_chunks}] step={step_end} "
                    f"loss={row['loss_mean']:.4f} "
                    f"grad_norm={row['grad_norm_mean']:.3f} "
                    f"dt={dt:.2f}s"
                )
    print(f"[done] total wall = {time.perf_counter() - wall0:.1f}s; run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
