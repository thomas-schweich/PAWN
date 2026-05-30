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
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pawn.checkpoint import save_model
from pawn.config import (
    CONDITIONING_KINDS,
    PRETRAIN_BUCKETS,
    SUPERNET,
    TINY_SUPERNET,
    VARIANTS,
    TINY_VARIANTS,
)
from pawn.corpus import Corpus, generate_corpus
from pawn.jax_setup import setup_jax_caching
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
    ap.add_argument("--conditioning", nargs="*", default=None,
                    choices=sorted(CONDITIONING_KINDS),
                    help="ordered control-token kinds to prepend after BOS "
                         "(e.g. `outcome`). The sequence layout is "
                         "[BOS][cond...][ply...][PAD...]; prefix width is "
                         "C = 1 + len(conditioning). Default is BOS-only "
                         "(C=1). Persisted to config.json so eval rebuilds "
                         "the corpus at the same C.")
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
    ap.add_argument("--use-sdpa", action="store_true",
                    help="opt the attention block into "
                         "jax.nn.dot_product_attention (parity #43). "
                         "Off by default — superseded by --use-flash on GPU.")
    ap.add_argument("--no-flash", action="store_true",
                    help="disable the Pallas flash-attention kernel "
                         "(force the plain materialised QK^T path). "
                         "Flash is the default on GPU; CPU runs "
                         "auto-fall-back regardless of this flag.")
    # Stochastic supernet sampling is **on by default** (H.1 housekeeping).
    # Use --no-stochastic-variants to force the exhaustive 3-variant sum
    # (for parity tests / debug). --stochastic-variants is accepted as a
    # no-op so existing JSON configs and shell scripts that set it
    # explicitly still parse.
    ap.add_argument("--stochastic-variants", dest="stochastic_variants",
                    action="store_true", default=None,
                    help="(default) sandwich-sample one non-supernet variant "
                         "per step; supernet always runs. ~10%% throughput "
                         "improvement at production 3-variant loss.")
    ap.add_argument("--no-stochastic-variants", dest="stochastic_variants",
                    action="store_false",
                    help="force the deterministic exhaustive sum over all "
                         "variants (parity / debug only).")
    ap.add_argument("--no-bucketing", action="store_true",
                    help="disable A.1 length bucketing — pretrain at a "
                         "single seq_len bucket. Slower but useful for "
                         "parity / debug.")
    ap.add_argument("--bucket-outer-factor", type=int, default=3,
                    help="how many K-batches' worth of games to generate "
                         "per outer prefetch call (default 3 — covers the "
                         "natural bucket distribution).")
    ap.add_argument("--emit-grad-norms", action="store_true",
                    help="(C.5 spike) log per-step pre-clip grad norm "
                         "alongside loss. Slight per-step overhead from "
                         "emitting an extra scalar per body call.")
    ap.add_argument("--optimizer", choices=("adamw", "lion"), default=None,
                    help="(C.1) optimizer to use. Lion halves opt-state "
                         "memory + ~3-5%% step time but needs LR ~1/3 of "
                         "AdamW's. Default is adamw.")
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
        ("resume", str(args.resume) if args.resume else None),
        # `--conditioning` with nargs="*" yields a list when passed
        # (possibly empty for `--conditioning` with no args) and None
        # when omitted, so the `is not None` guard below distinguishes
        # "explicitly set to []" from "not passed".
        ("conditioning", args.conditioning),
    ):
        if val is not None:
            base[flag] = val
    # `store_true` flags: only merge when actually set so a CLI omit
    # doesn't override a JSON config that has the flag True (round-1
    # codex P2: `--use-sdpa` and `--wandb` defaulted to False, were
    # `not None`, and silently overrode the config). Mirrors the
    # `local_checkpoints` handling below.
    if args.use_sdpa:
        base["use_sdpa"] = True
    if args.optimizer is not None:
        base["optimizer"] = args.optimizer
    # `stochastic_variants` default in argparse is None — only override
    # the config value when the user explicitly passed --stochastic-variants
    # or --no-stochastic-variants on the CLI.
    if args.stochastic_variants is not None:
        base["stochastic_variants"] = args.stochastic_variants
    if args.no_flash:
        base["use_flash"] = False
    if args.wandb:
        base["wandb"] = True
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
    cache_path = setup_jax_caching()
    if cache_path is not None:
        print(f"JAX compilation cache: {cache_path}")

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

    # Resolve cfg.amp_dtype → jnp dtype. None ⇒ fp32 forward (back-
    # compat with existing tests that synthesise opt_state in fp32 and
    # parity-test the legacy converter bit-exact). The default is
    # bf16 per plan §5 + v1 parity.
    _DTYPE_MAP = {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": None,
    }
    compute_dtype = _DTYPE_MAP[cfg.amp_dtype]
    # Pallas flash requires the GPU backend; CPU smoke runs
    # (`PAWN_ALLOW_CPU=1`) auto-fall-back to the plain path regardless
    # of `cfg.use_flash`.
    use_flash = cfg.use_flash and jax.default_backend() == "gpu"
    if cfg.accumulation_steps != 1:
        # The trainer's batch shape would need to grow a leading
        # accumulation axis (K, N, B, T). The current bucketed prefetcher
        # produces (K, B, T). Wire-through is a follow-up; the trainer
        # itself (`make_train_step`) is C.4-complete and unit-tested.
        raise NotImplementedError(
            f"accumulation_steps={cfg.accumulation_steps} is not yet "
            f"wired through the data loop (use 1 for now). The "
            f"make_train_step path supports it; the prefetcher needs "
            f"to produce (K, N, B, T)-shaped batches."
        )
    train_step = make_train_step(
        optimizer, variants,
        compute_dtype=compute_dtype, use_sdpa=cfg.use_sdpa, use_flash=use_flash,
        stochastic_variants=cfg.stochastic_variants,
        accumulation_steps=cfg.accumulation_steps,
    )
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
    scan_step = make_scan_step(train_step, emit_grad_norms=args.emit_grad_norms)
    emit_grad_norms = args.emit_grad_norms

    rng = np.random.default_rng(0)
    indices = np.arange(cfg.batch_size, dtype=np.int64)

    # A.1 — Length bucketing. Read the schedule from `PRETRAIN_BUCKETS`
    # (the single source of truth). The top edge must equal cfg.seq_len
    # (so every game fits in some bucket). `--no-bucketing` ships the
    # legacy single-bucket path for parity / debug.
    if args.no_bucketing:
        bucket_edges: tuple[int, ...] = (cfg.seq_len,)
    else:
        bucket_edges = tuple(e for e in PRETRAIN_BUCKETS if e <= cfg.seq_len)
        if not bucket_edges or bucket_edges[-1] != cfg.seq_len:
            # Add the top edge if not already present
            bucket_edges = (*bucket_edges, cfg.seq_len) if bucket_edges else (cfg.seq_len,)

    # A.4 — Async corpus prefetch + bucketed batch production. The
    # background thread runs the Rust `generate_corpus`, bucketises into
    # per-T sub-corpora, slices each into (K, B, T_bucket) Batches, and
    # uploads via `jnp.asarray` (which is async). The GPU runs scan_step
    # over the resulting queue while the next outer corpus is being
    # generated.
    def _build_batch_from_corpus_slice(
        corp: Corpus, k_inner: int, batch_size: int, seq_len: int, start_idx: int
    ) -> Batch:
        sel = slice(start_idx, start_idx + k_inner * batch_size)
        tokens = corp.tokens[sel, :seq_len].reshape(k_inner, batch_size, seq_len)
        targets = corp.targets[sel, :seq_len].reshape(k_inner, batch_size, seq_len)
        attn = corp.attn_mask[sel, :seq_len].reshape(k_inner, batch_size, seq_len)
        lmask = corp.loss_mask[sel, :seq_len].reshape(k_inner, batch_size, seq_len)
        return Batch(
            tokens=jnp.asarray(tokens), targets=jnp.asarray(targets),
            attn_mask=jnp.asarray(attn), loss_mask=jnp.asarray(lmask),
        )

    def _prepare_outer_chunk(
        n_games: int, seed: int, edges: tuple[int, ...],
        batch_size: int, inner_k: int,
    ) -> list[tuple[int, Batch]]:
        """Generate `n_games` random games, bucketise, slice into
        K-batches per bucket, return as `[(edge, Batch), ...]` in
        ascending-edge order. Games that don't fill a complete K-batch
        in any bucket are dropped (acceptable when n_games >> B*K)."""
        corpus = generate_corpus(
            n_games=n_games, max_ply=cfg.seq_len, seq_len=cfg.seq_len, seed=seed,
            conditioning=cfg.conditioning,
        )
        if edges == (cfg.seq_len,):
            # Unbucketed path: one bucket at full seq_len.
            buckets = {cfg.seq_len: corpus}
        else:
            buckets = corpus.by_bucket(edges)
        out: list[tuple[int, Batch]] = []
        for edge in sorted(buckets):
            sub = buckets[edge]
            n_full_K = sub.n_games // (batch_size * inner_k)
            for j in range(n_full_K):
                start = j * batch_size * inner_k
                out.append((
                    edge,
                    _build_batch_from_corpus_slice(
                        sub, inner_k, batch_size, edge, start,
                    ),
                ))
        return out

    class BucketedPrefetcher:
        """Yields `(edge, Batch)` tuples with one outer-chunk lookahead.

        Each `next()` returns the next ready batch. When the queue is
        empty we wait on the pending future and immediately submit the
        next outer-chunk for the executor to start producing. The host
        cost (Rust gen + numpy reshape + H2D enqueue) is overlapped with
        the GPU work consuming earlier batches.
        """

        def __init__(self) -> None:
            self._queue: deque[tuple[int, Batch]] = deque()
            self._pending: Future[list[tuple[int, Batch]]] | None = None
            self._closed = False
            self.outer_factor = max(1, args.bucket_outer_factor)
            self._submit_next()

        def _next_seed(self) -> int:
            return int(rng.integers(0, 2**31 - 1))

        def _submit_next(self) -> None:
            if self._closed:
                return
            seed = self._next_seed()
            n_games = cfg.batch_size * cfg.k * self.outer_factor
            self._pending = executor.submit(
                _prepare_outer_chunk, n_games, seed, bucket_edges,
                cfg.batch_size, cfg.k,
            )

        def next(self) -> tuple[int, Batch] | None:
            while not self._queue:
                if self._pending is None:
                    return None
                outer = self._pending.result()
                # Immediately queue the next outer-chunk so its host work
                # overlaps with the GPU work consuming this one.
                self._submit_next()
                self._queue.extend(outer)
                if not self._queue:
                    # Pathological: a generated corpus has fewer than B*K
                    # games in ANY bucket. Try again rather than spinning.
                    if self._pending is None:
                        return None
            return self._queue.popleft()

        def close(self) -> None:
            self._closed = True
            if self._pending is not None:
                self._pending.cancel()
            self._pending = None
            self._queue.clear()

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

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pawn-prefetch")
    producer = BucketedPrefetcher()
    # Per-bucket throughput counters (visible in the JSONL stream).
    bucket_steps: dict[int, int] = {edge: 0 for edge in bucket_edges}
    try:
        while next_step < total_steps:
            nxt = producer.next()
            if nxt is None:
                # Should be unreachable in practice — generator always
                # produces at least one batch for a non-empty bucket.
                break
            edge, chunk_batches = nxt
            this_chunk_k = int(chunk_batches.tokens.shape[0])

            if emit_grad_norms:
                state, chunk_losses, chunk_gnorms = scan_step(state, chunk_batches)
                chunk_gnorms_np = np.asarray(chunk_gnorms)
            else:
                state, chunk_losses = scan_step(state, chunk_batches)
                chunk_gnorms_np = None
            next_step += this_chunk_k
            bucket_steps[edge] = bucket_steps.get(edge, 0) + this_chunk_k

            # One D→H per chunk, not per step.
            chunk_losses_np = np.asarray(chunk_losses)

            # Log every step that crossed a log_interval boundary inside
            # the chunk — replays the within-chunk loss curve without
            # per-step syncs.
            chunk_start = next_step - this_chunk_k
            for i in range(this_chunk_k):
                step = chunk_start + i + 1
                if step % cfg.log_interval == 0:
                    extra = {}
                    if chunk_gnorms_np is not None:
                        extra["grad_norm"] = float(chunk_gnorms_np[i])
                        extra["did_clip"] = bool(chunk_gnorms_np[i] > cfg.max_grad_norm)
                    logger.log_train(
                        step=step, loss=float(chunk_losses_np[i]),
                        lr=np.asarray(schedule(step)).item(),
                        step_time=(time.time() - t0) / max(1, step - start),
                        bucket=edge,
                        **extra,
                    )

            # Checkpoint when we cross a checkpoint boundary. Use
            # division-based crossing so chunk_k doesn't have to divide
            # checkpoint_interval. (Pre-A.1 chunks were exactly cfg.k
            # steps; bucketing keeps that, but the test is more robust.)
            crossed_checkpoint = (
                next_step // cfg.checkpoint_interval
                != (next_step - this_chunk_k) // cfg.checkpoint_interval
            )
            if crossed_checkpoint or next_step >= total_steps:
                if cfg.local_checkpoints or cfg.hf_repo:
                    _save_checkpoint(next_step)

            if should_shutdown():
                _save_checkpoint(next_step)
                break
    finally:
        producer.close()
        executor.shutdown(wait=False, cancel_futures=True)

    # Per-bucket step distribution at end of training.
    print(f"Per-bucket step counts: {bucket_steps}", flush=True)

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
