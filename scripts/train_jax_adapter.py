"""JAX adapter training driver — Phase 3 verification.

Trains a LoRA adapter on a finite corpus (Rust-engine random games
serve as the verification proxy for the production Lichess
Elo-slice cache; the cache itself lands in Phase 4 cleanup).

Execution order:

  1. Parse args. Validates ``--rank > 0``, ``--total-steps > 0``,
     ``--total-steps % --k == 0``, ``--batch-size > 0``, ``--seq-len
     > 0``, ``--seq-len <= supernet.max_seq_len``.
  2. Build the LR schedule (catches misconfigs before any filesystem
     write).
  3. Estimate corpus memory; abort if it exceeds ``--max-corpus-gb``.
  4. Generate the Rust-engine corpus (treat as a finite dataset:
     train + val split).
  5. Create the run directory and write ``config.json``.
  6. Init the supernet, slice to ``--variant``, wrap with LoRA, init
     two-tier optimizer state, build the K-step scan + eval callable.
  7. Train for ``--total-steps`` chunks; eval every ``--val-every``
     chunks; log per-chunk train + val metrics to ``metrics.jsonl``.

The eval split is the last ``--val-frac`` of the generated corpus.
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
import jax.numpy as jnp
import numpy as np

from pawn.adapter_trainer import (
    init_adapter_state,
    make_adapter_scan_step,
    make_adapter_train_step,
    make_eval_step,
)
from pawn.adapters import LoRAConfig, LoRAModel, init_lora_model
from pawn.config import (
    SUPERNET,
    TINY_SUPERNET,
    TINY_VARIANTS,
    VARIANTS,
    ModelConfig,
)
from pawn.corpus import generate_corpus
from pawn.model import init_model, sliced
from pawn.trainer import Batch, make_lr_schedule, make_optimizer


def _slug() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{ts}_{os.getpid()}"


def _resolve_supernet(name: str) -> tuple[ModelConfig, dict[str, ModelConfig]]:
    if name == "tiny":
        return TINY_SUPERNET, TINY_VARIANTS
    if name == "supernet":
        return SUPERNET, VARIANTS
    raise ValueError(
        f"unknown --supernet {name!r}; expected one of tiny|supernet"
    )


def _stage(arr: np.ndarray) -> jax.Array:
    """One-shot host→device transfer."""
    return jnp.asarray(arr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supernet", choices=["tiny", "supernet"], default="tiny",
    )
    parser.add_argument(
        "--variant", default="base",
        help="which sliced variant to train the adapter on (one of "
        "the keys of the supernet's VARIANTS dict).",
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument(
        "--lora-targets", nargs="+", default=["q", "v"],
        choices=["q", "k", "v", "o"],
    )
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument(
        "--val-frac", type=float, default=0.1,
        help="fraction of the generated corpus held out for validation.",
    )
    parser.add_argument(
        "--val-every", type=int, default=5,
        help="run validation every N chunks (and at the final chunk).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpus-seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--max-corpus-gb", type=float, default=8.0)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    # Upfront guards — same shape as the pretrain driver.
    if args.k <= 0:
        raise SystemExit(f"--k={args.k} must be a positive integer")
    if args.batch_size <= 0:
        raise SystemExit(f"--batch-size={args.batch_size} must be positive")
    if args.seq_len <= 0:
        raise SystemExit(f"--seq-len={args.seq_len} must be positive")
    if args.total_steps <= 0:
        raise SystemExit(f"--total-steps={args.total_steps} must be positive")
    if args.total_steps % args.k != 0:
        raise SystemExit(
            f"--total-steps={args.total_steps} must be a multiple of --k={args.k}"
        )
    if args.rank <= 0:
        raise SystemExit(f"--rank={args.rank} must be positive")
    if not 0.0 < args.val_frac < 1.0:
        raise SystemExit(
            f"--val-frac={args.val_frac} must be in (0, 1)"
        )

    corpus_seed = args.seed if args.corpus_seed is None else args.corpus_seed
    model_seed = args.seed if args.model_seed is None else args.model_seed

    supernet_cfg, variants = _resolve_supernet(args.supernet)
    if args.variant not in variants:
        raise SystemExit(
            f"--variant={args.variant!r} not in {list(variants.keys())}"
        )
    variant_cfg = variants[args.variant]

    if args.seq_len > supernet_cfg.max_seq_len:
        raise SystemExit(
            f"--seq-len={args.seq_len} exceeds supernet.max_seq_len="
            f"{supernet_cfg.max_seq_len}"
        )

    try:
        sched = make_lr_schedule(
            peak_lr=args.lr,
            total_steps=args.total_steps,
            warmup_steps=args.warmup_steps,
        )
    except ValueError as exc:
        raise SystemExit(f"LR-schedule configuration error: {exc}") from exc

    # Train + val games. Round n_train_games so we get exactly
    # total_steps × batch_size, and add the val fraction on top.
    n_train_games = args.total_steps * args.batch_size
    n_val_games = max(1, int(n_train_games * args.val_frac))
    # Round n_val_games to a multiple of batch_size so the eval loop
    # processes whole batches.
    n_val_games = (n_val_games // args.batch_size) * args.batch_size
    if n_val_games == 0:
        raise SystemExit(
            f"--val-frac={args.val_frac} produces <1 batch of val games. "
            "Increase --val-frac or --total-steps."
        )
    n_total = n_train_games + n_val_games
    bytes_per_game = args.seq_len * 10 + 1
    estimated_gb = n_total * bytes_per_game / (1024 ** 3)
    if estimated_gb > args.max_corpus_gb:
        raise SystemExit(
            f"corpus would need ~{estimated_gb:.2f} GiB > "
            f"--max-corpus-gb={args.max_corpus_gb}"
        )

    print(
        f"[setup] supernet={args.supernet} variant={args.variant} "
        f"rank={args.rank} targets={args.lora_targets} "
        f"total_steps={args.total_steps} K={args.k} B={args.batch_size} "
        f"T={args.seq_len} n_train_games={n_train_games} "
        f"n_val_games={n_val_games}"
    )

    t0 = time.perf_counter()
    print(
        f"[corpus] generating {n_total} games (seed={corpus_seed})"
    )
    corpus = generate_corpus(
        n_games=n_total,
        max_ply=args.seq_len,
        seq_len=args.seq_len,
        seed=corpus_seed,
    )
    print(f"[corpus] done in {time.perf_counter() - t0:.1f}s")

    # Train / val split: first n_train, then n_val.
    train_tokens = corpus.tokens[:n_train_games]
    train_attn = corpus.attn_mask[:n_train_games]
    train_targets = corpus.targets[:n_train_games]
    train_loss = corpus.loss_mask[:n_train_games]
    val_tokens = corpus.tokens[n_train_games:]
    val_attn = corpus.attn_mask[n_train_games:]
    val_targets = corpus.targets[n_train_games:]
    val_loss = corpus.loss_mask[n_train_games:]

    run_dir = Path(args.logs_dir) / f"jax_adapter_run_{_slug()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "supernet": args.supernet,
                "variant": args.variant,
                "supernet_cfg": dataclasses.asdict(supernet_cfg),
                "variant_cfg": dataclasses.asdict(variant_cfg),
                "lora": {
                    "rank": args.rank,
                    "targets": list(args.lora_targets),
                    "alpha": args.lora_alpha,
                },
                "total_steps": args.total_steps,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "k": args.k,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "val_frac": args.val_frac,
                "val_every": args.val_every,
                "n_train_games": n_train_games,
                "n_val_games": n_val_games,
                "seed": args.seed,
                "corpus_seed": corpus_seed,
                "model_seed": model_seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Init: slice → wrap with LoRA → partition for adapter training.
    key = jax.random.PRNGKey(model_seed)
    supernet = init_model(supernet_cfg, key)
    backbone = sliced(supernet, variant_cfg)
    lora_cfg = LoRAConfig(
        rank=args.rank,
        targets=tuple(args.lora_targets),
        alpha=args.lora_alpha,
    )
    lora_model = init_lora_model(backbone, lora_cfg, jax.random.PRNGKey(model_seed + 1))
    optimizer = make_optimizer(sched)
    state, frozen = init_adapter_state(lora_model, optimizer)
    train_step = make_adapter_train_step(optimizer, frozen)
    scan = make_adapter_scan_step(train_step, args.k)
    eval_step = make_eval_step(frozen)

    n_chunks = args.total_steps // args.k
    print(f"[train] n_chunks={n_chunks} (K={args.k} steps each)")

    # Pre-shape train corpus into [N_chunks, K, B, T] views.
    chunk_shape = (n_chunks, args.k, args.batch_size, args.seq_len)
    train_tokens_chunks = train_tokens.reshape(chunk_shape)
    train_attn_chunks = train_attn.reshape(chunk_shape)
    train_targets_chunks = train_targets.reshape(chunk_shape)
    train_loss_chunks = train_loss.reshape(chunk_shape)

    def run_validation(trainable: LoRAModel) -> tuple[float, int]:
        """Iterate the val set in batches, return (mean_loss, total_n).
        We iterate on the host (eval is a one-shot diagnostic, not on
        the critical path) and accumulate weighted by n_supervised."""
        n = val_tokens.shape[0]
        total_loss = 0.0
        total_n = 0
        for s in range(0, n, args.batch_size):
            e = s + args.batch_size
            batch = (
                _stage(val_tokens[s:e]),
                _stage(val_attn[s:e]),
                _stage(val_targets[s:e]),
                _stage(val_loss[s:e]),
            )
            m = eval_step(trainable, batch)
            ns = int(m["n_supervised"])
            total_loss += float(m["loss"]) * ns
            total_n += ns
        return (total_loss / max(total_n, 1), total_n)

    best_val = float("inf")
    with metrics_path.open("w", encoding="utf-8") as mf:
        wall0 = time.perf_counter()
        step_start = 0
        for chunk_i in range(n_chunks):
            # Stage chunk to device.
            chunk: Batch = (
                _stage(train_tokens_chunks[chunk_i]),
                _stage(train_attn_chunks[chunk_i]),
                _stage(train_targets_chunks[chunk_i]),
                _stage(train_loss_chunks[chunk_i]),
            )
            t_chunk = time.perf_counter()
            # K-step ``lax.scan`` — one XLA dispatch per chunk instead
            # of K. Per-step metrics come back stacked on a leading
            # ``[K]`` axis.
            state, chunk_metrics = scan(state, chunk)
            jax.block_until_ready((state, chunk_metrics))
            dt = time.perf_counter() - t_chunk
            step_end = int(state.step)

            host_metrics = {
                mk: np.asarray(v) for mk, v in chunk_metrics.items()
            }
            row: dict[str, float | int | None] = {
                "chunk": chunk_i,
                "step_start": step_start,
                "step_end": step_end,
                "wall_s": time.perf_counter() - wall0,
                "chunk_wall_s": dt,
                "train_loss_mean": float(host_metrics["loss"].mean()),
                "train_loss_last": float(host_metrics["loss"][-1]),
                "grad_norm_mean": float(host_metrics["grad_norm"].mean()),
                "grad_norm_max": float(host_metrics["grad_norm"].max()),
                "val_loss": None,
                "val_n": None,
            }
            do_val = (
                (chunk_i + 1) % args.val_every == 0
                or chunk_i + 1 == n_chunks
            )
            if do_val:
                vloss, vn = run_validation(state.trainable)
                row["val_loss"] = vloss
                row["val_n"] = vn
                if vloss < best_val:
                    best_val = vloss
            mf.write(json.dumps(row) + "\n")
            if (chunk_i + 1) % 10 == 0 or chunk_i + 1 == n_chunks:
                mf.flush()
            if not args.quiet:
                vs = f" val={row['val_loss']:.4f}" if row["val_loss"] is not None else ""
                print(
                    f"[chunk {chunk_i + 1}/{n_chunks}] step={step_end} "
                    f"train_loss={row['train_loss_mean']:.4f}{vs} "
                    f"grad_norm={row['grad_norm_mean']:.3f} dt={dt:.2f}s"
                )
            step_start = step_end
    print(
        f"[done] total wall = {time.perf_counter() - wall0:.1f}s; "
        f"best_val={best_val:.4f}; run_dir={run_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
