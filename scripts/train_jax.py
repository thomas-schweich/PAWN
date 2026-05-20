"""JAX pretraining driver for the supernet (Phase 2).

Minimal entry point that ties the Phase-2 chunks together. Execution
order (notable because filesystem side-effects sit between validation
and the long-running steps):

  1. Parse args, resolve supernet config. Validates ``--k > 0``,
     ``--total-steps % --k == 0``, ``--batch-size > 0``,
     ``--seq-len > 0``, ``--seq-len <= supernet.max_seq_len``, and
     ``--total-steps > 0`` upfront — all SystemExit before any
     filesystem write.
  2. Build the LR schedule (catches ``warmup``/``total_steps`` /
     ``end_value`` misconfigs *before* any filesystem write).
  3. Estimate the corpus footprint and abort if it exceeds
     ``--max-corpus-gb`` (~122 GiB at SUPERNET / 100K-step defaults
     would OOM-kill a modest host; the guard makes that fail loudly).
  4. Generate the Rust-engine corpus (always — no cache reuse path)
     using ``--corpus-seed`` (defaults to ``--seed``).
  5. Create the run directory and write
     ``logs/jax_run_<YYYYMMDD_HHMMSS_µs>_<pid>/config.json`` (only
     after corpus generation succeeds, so a corpus-time failure does
     not leave an orphaned run dir). ``config.json`` includes every
     ``ModelConfig`` field for both the supernet and every variant.
  6. Init the model with ``--model-seed`` (defaults to ``--seed``),
     compose the optimizer + ``TrainState``, build the K-step
     ``scan`` callable.
  7. Slice the flat ``[N_games, T]`` corpus into ``[K, B, T]`` chunks
     per scan call; write one ``metrics.jsonl`` row per chunk.

This is the Phase-2 verification driver, not the production
pretraining loop. Production runs need: HF-backed checkpoints, an LR
schedule shaped to a real total_steps budget, wandb / dashboard
integration, preemption-safe resume, double-buffered chunk staging —
those will land in Phase-3 polish or once Phase-2 is otherwise green.
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
    """Collision-resistant run slug: timestamp at microsecond
    resolution plus PID. Two concurrent runs from the same user
    on the same host within the same second still get distinct
    directories (PIDs differ); the microsecond suffix also covers
    the rare same-PID sub-second loop case."""
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


class _ChunkedCorpus:
    """Pre-shaped corpus arrays: each field reshaped once from
    ``[N_games, T]`` to ``[N_chunks, K, B, T]`` (or ``[N_chunks, K,
    B]`` for per-game fields). Per-chunk staging is then a contiguous
    NumPy view + a single ``jnp.asarray`` per field, with no per-call
    bounds check needed (the reshape itself fails loudly if the
    corpus is mis-sized).

    Replaces the per-chunk slice+reshape path in ``_stage_chunk``;
    the reshape is one-time and zero-copy because ``generate_corpus``
    produces contiguous C-order arrays."""

    tokens: np.ndarray
    attn_mask: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray

    def __init__(
        self,
        tokens: np.ndarray,
        attn_mask: np.ndarray,
        targets: np.ndarray,
        loss_mask: np.ndarray,
        *,
        n_chunks: int,
        k: int,
        batch_size: int,
    ) -> None:
        new_shape = (n_chunks, k, batch_size, tokens.shape[1])
        self.tokens = tokens.reshape(new_shape)
        self.attn_mask = attn_mask.reshape(new_shape)
        self.targets = targets.reshape(new_shape)
        self.loss_mask = loss_mask.reshape(new_shape)

    def stage(self, chunk_i: int) -> Batch:
        """Return the [K, B, T] chunk for index ``chunk_i`` staged to
        device."""
        return (
            jnp.asarray(self.tokens[chunk_i]),
            jnp.asarray(self.attn_mask[chunk_i]),
            jnp.asarray(self.targets[chunk_i]),
            jnp.asarray(self.loss_mask[chunk_i]),
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
        help="K = number of inner steps per lax.scan invocation. Must "
        "be > 0 and divide --total-steps. Larger K amortises host "
        "overhead but consumes K * B games from the corpus per scan call.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--warmup-steps", type=int, default=100,
        help="linear-warmup span in steps. Must be >= 0 and < --total-steps.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="default seed for both the corpus RNG and the model init "
        "key. Override individually via --corpus-seed / --model-seed.",
    )
    parser.add_argument(
        "--corpus-seed", type=int, default=None,
        help="seed for the Rust corpus generator. Defaults to --seed.",
    )
    parser.add_argument(
        "--model-seed", type=int, default=None,
        help="seed for the JAX model init key. Defaults to --seed.",
    )
    parser.add_argument(
        "--max-corpus-gb", type=float, default=64.0,
        help="upper bound on host-RAM corpus footprint (GiB). The "
        "script aborts before generation if the requested corpus "
        "would exceed this — guards against an OOM kill on a "
        "misconfigured production run.",
    )
    parser.add_argument(
        "--logs-dir", type=str, default="logs",
        help="root directory for per-run output (metrics.jsonl, config.json).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress per-chunk stdout prints (still writes to metrics.jsonl).",
    )
    args = parser.parse_args(argv)

    if args.k <= 0:
        raise SystemExit(f"--k={args.k} must be a positive integer")
    if args.batch_size <= 0:
        raise SystemExit(
            f"--batch-size={args.batch_size} must be a positive integer; "
            "a zero or negative batch otherwise produces an entirely-padded "
            "no-op run with all-zero metrics."
        )
    if args.seq_len <= 0:
        raise SystemExit(
            f"--seq-len={args.seq_len} must be a positive integer; "
            "a zero sequence length crashes in JIT during RoPE reshape."
        )
    if args.total_steps <= 0:
        raise SystemExit(
            f"--total-steps={args.total_steps} must be a positive integer"
        )
    if args.total_steps % args.k != 0:
        raise SystemExit(
            f"--total-steps={args.total_steps} must be a multiple of --k={args.k}"
        )

    corpus_seed = args.seed if args.corpus_seed is None else args.corpus_seed
    model_seed = args.seed if args.model_seed is None else args.model_seed

    supernet_cfg, variants = _resolve_supernet(args.supernet)
    specs = tuple(VariantSpec(cfg=v) for v in variants.values())

    # Reject overlong sequence lengths upfront — otherwise the
    # corpus is generated and run files written before the first
    # JIT trace fails in PAWNModel.__call__ (Codex P2).
    if args.seq_len > supernet_cfg.max_seq_len:
        raise SystemExit(
            f"--seq-len={args.seq_len} exceeds supernet max_seq_len="
            f"{supernet_cfg.max_seq_len} ({args.supernet}); the model "
            "cannot attend beyond its RoPE table."
        )

    # Build the LR schedule first so any misconfiguration (warmup
    # >= total, end_value >= peak, non-positive arg) fails BEFORE
    # we create the run directory and write config.json. Otherwise
    # a bad-config invocation leaves an orphaned logs/ dir.
    # ``make_lr_schedule`` raises ``ValueError``; wrap into
    # ``SystemExit`` so all CLI-validation failures surface through
    # the same exception type (and ``test_validation_failures_do_not_create_run_dir``
    # catches every guard).
    try:
        sched = make_lr_schedule(
            peak_lr=args.lr,
            total_steps=args.total_steps,
            warmup_steps=args.warmup_steps,
        )
    except ValueError as exc:
        raise SystemExit(f"LR-schedule configuration error: {exc}") from exc

    n_games = args.total_steps * args.batch_size
    # Per-token byte cost is 10 bytes (int32 tokens + bool attn_mask +
    # int32 targets + bool loss_mask = 4+1+4+1). Per-game outcome
    # offset is a single uint8. The estimate is the FINAL Corpus
    # footprint; transient peak during ``_pack_clm`` is roughly
    # 1.5-2× this (int16 engine output + int32 cast + final pack
    # arrays coexist briefly). Size --max-corpus-gb with that
    # safety margin in mind (default 64 GiB ≈ 32-43 GiB estimate is
    # comfortable). At SUPERNET-class defaults (total_steps=100k,
    # batch_size=256, seq_len=512) the estimate is ~122 GiB.
    bytes_per_game = args.seq_len * 10 + 1
    estimated_gb = n_games * bytes_per_game / (1024 ** 3)
    if estimated_gb > args.max_corpus_gb:
        raise SystemExit(
            f"corpus would need ~{estimated_gb:.1f} GiB > "
            f"--max-corpus-gb={args.max_corpus_gb} (n_games={n_games}, "
            f"seq_len={args.seq_len}). Reduce --total-steps / --batch-size, "
            "or pass --max-corpus-gb to override after sizing the host."
        )

    print(
        f"[setup] supernet={args.supernet} variants={list(variants.keys())} "
        f"total_steps={args.total_steps} K={args.k} B={args.batch_size} "
        f"T={args.seq_len} n_games={n_games} estimated_corpus_gib={estimated_gb:.2f}"
    )

    t0 = time.perf_counter()
    print(
        f"[corpus] generating {n_games} games "
        f"(seed={corpus_seed}, max_ply={args.seq_len})"
    )
    corpus = generate_corpus(
        n_games=n_games,
        max_ply=args.seq_len,
        seq_len=args.seq_len,
        seed=corpus_seed,
    )
    print(f"[corpus] done in {time.perf_counter() - t0:.1f}s")

    # Run directory created AFTER corpus succeeds so a corpus-time
    # OOM / engine panic doesn't leave an orphaned ``jax_run_*``
    # directory containing only a config.json with no metrics.
    run_dir = Path(args.logs_dir) / f"jax_run_{_slug()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "supernet": args.supernet,
                # ``asdict`` captures every ``ModelConfig`` field — vocab_size,
                # max_seq_len, n_outcomes, rope_base — so the config file is
                # always a complete record without needing to hand-track the
                # field list.
                "supernet_cfg": dataclasses.asdict(supernet_cfg),
                # Serialise the full per-variant ModelConfig dicts, not
                # just the variant names: ``TINY_VARIANTS["small"]`` might
                # be redefined later, and a stale ``config.json`` referencing
                # only "small" would silently reflect the current definition.
                "variants": {
                    name: dataclasses.asdict(cfg)
                    for name, cfg in variants.items()
                },
                "total_steps": args.total_steps,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "k": args.k,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "seed": args.seed,
                "corpus_seed": corpus_seed,
                "model_seed": model_seed,
                "estimated_corpus_gib": estimated_gb,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    key = jax.random.PRNGKey(model_seed)
    model = init_model(supernet_cfg, key)
    optimizer = make_optimizer(sched)
    state = init_train_state(model, optimizer)
    train_step = make_train_step(optimizer, specs)
    scan = make_scan_step(train_step, args.k)

    n_chunks = args.total_steps // args.k
    print(f"[train] n_chunks={n_chunks} (K={args.k} steps each)")

    # One-time reshape of the flat corpus into per-chunk views. The
    # reshape is zero-copy on the contiguous C-order arrays the engine
    # produces; per-chunk staging then becomes a single contiguous
    # slice + jnp.asarray per field.
    chunked_corpus = _ChunkedCorpus(
        corpus.tokens,
        corpus.attn_mask,
        corpus.targets,
        corpus.loss_mask,
        n_chunks=n_chunks,
        k=args.k,
        batch_size=args.batch_size,
    )

    with metrics_path.open("w", encoding="utf-8") as mf:
        # Reset wall0 just before the chunk loop so ``wall_s`` in
        # metrics rows reflects training time only — corpus generation
        # + model init were already timed separately in their own
        # phases.
        wall0 = time.perf_counter()
        # ``step_start`` is carried across iterations on the host so we
        # never force a D2H read of ``state.step`` BEFORE dispatching
        # the next ``scan`` — that would serialise compute / staging.
        step_start = 0
        for chunk_i in range(n_chunks):
            chunk = chunked_corpus.stage(chunk_i)
            t_chunk = time.perf_counter()
            state, metrics = scan(state, chunk)
            # Block on both ``state`` and ``metrics`` — ``state.step``
            # is needed for step_end below and would otherwise force
            # an extra D2H sync if only ``metrics`` were waited on.
            jax.block_until_ready((state, metrics))
            dt = time.perf_counter() - t_chunk

            # Pull metrics off-device — they're [K]-shaped per key.
            # Use ``mk`` as the dict-comp variable so ``args.k`` is not
            # shadowed inside the loop body.
            host_metrics = {mk: np.asarray(v) for mk, v in metrics.items()}
            step_end = int(state.step)
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
            # Flush every 10 chunks (and on the final one) — at the
            # critical path between scan calls, frequent fsync stalls
            # can compound on networked filesystems. The buffered
            # write is still durable on graceful exit.
            if (chunk_i + 1) % 10 == 0 or chunk_i + 1 == n_chunks:
                mf.flush()
            if not args.quiet:
                print(
                    f"[chunk {chunk_i + 1}/{n_chunks}] step={step_end} "
                    f"loss={row['loss_mean']:.4f} "
                    f"grad_norm={row['grad_norm_mean']:.3f} "
                    f"dt={dt:.2f}s"
                )
            step_start = step_end
    print(f"[done] total wall = {time.perf_counter() - wall0:.1f}s; run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
