#!/usr/bin/env python3
"""Standardized performance benchmarks for PAWN (v2: JAX/Equinox/Optax).

Benchmarks three layers:
  1. Rust engine (CPU): game generation, validation, board extraction
  2. Backbone training steps (GPU): tiny/small/base/large models, eager
     vs jit-compiled
  3. Adapter training steps (GPU): LoRA, FiLM, Bottleneck on a frozen
     backbone via :func:`pawn.adapter_trainer.make_adapter_train_step`.

Defaults per platform:
  AMD/ROCm:    JAX-on-ROCm, plain attention (matches pawn/model.py)
  NVIDIA/CUDA: JAX-on-CUDA12, plain attention

Usage::

    uv run --extra rocm python scripts/benchmark.py
    uv run --extra rocm python scripts/benchmark.py --engine-only
    uv run --extra rocm python scripts/benchmark.py --gpu-only
    uv run --extra rocm python scripts/benchmark.py --variants tiny small
    uv run --extra rocm python scripts/benchmark.py --adapters lora film
    uv run --extra rocm python scripts/benchmark.py --no-backbone
    uv run --extra rocm python scripts/benchmark.py --no-adapters
    uv run --extra rocm python scripts/benchmark.py --no-jit
    uv run --extra rocm python scripts/benchmark.py --jit-only
    uv run --extra rocm python scripts/benchmark.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

import chess_engine as engine


# ── Result containers ────────────────────────────────────────────────────────

@dataclass
class TimingResult:
    name: str
    n_iterations: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    throughput: float | None = None
    throughput_unit: str = ""
    peak_memory_mb: float | None = None
    warmup_ms: float | None = None
    n_warmup: int | None = None

    def summary_line(self) -> str:
        parts = [
            f"{self.name:<50s}",
            f"{self.mean_ms:>9.2f}ms mean",
            f"{self.median_ms:>9.2f}ms median",
            f"({self.min_ms:.2f} - {self.max_ms:.2f}ms)",
        ]
        if self.throughput is not None:
            parts.append(f"  {self.throughput:>10.0f} {self.throughput_unit}")
        if self.peak_memory_mb is not None:
            parts.append(f"  {self.peak_memory_mb:>7.0f} MB")
        if self.warmup_ms is not None and self.n_warmup is not None:
            parts.append(f"  warmup: {self.warmup_ms:.0f}ms ({self.n_warmup} iters)")
        return "  ".join(parts)


@dataclass
class ConcurrencyResult:
    n_models: int
    step_ms: float
    per_model_ms: float
    total_throughput: float
    per_model_throughput: float
    total_vram_mb: float
    speedup: float


@dataclass
class BenchmarkReport:
    timestamp: str = ""
    platform_info: dict = field(default_factory=dict)
    engine_results: list[dict] = field(default_factory=list)
    backbone_results: list[dict] = field(default_factory=list)
    data_pipeline_results: list[dict] = field(default_factory=list)
    concurrency_results: list[dict] = field(default_factory=list)
    adapter_results: list[dict] = field(default_factory=list)


# ── Timing helpers ───────────────────────────────────────────────────────────

def time_cpu(fn, *, n_warmup: int = 2, n_iter: int = 10) -> list[float]:
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


@dataclass
class GPUTimingResult:
    times: list[float]
    warmup_secs: float
    n_warmup: int


def _sync():
    """JAX dispatch is async; block on any in-flight work to get honest
    wall-clock timings. Wraps `jax.block_until_ready` over a sentinel."""
    import jax
    import jax.numpy as jnp
    jax.block_until_ready(jnp.zeros(()))


def time_gpu(fn, *, n_warmup: int = 3, n_iter: int = 10) -> GPUTimingResult:
    """Time a JAX function with explicit sync points.

    Returns timed iteration durations plus total warmup wall time (which
    includes JIT compilation overhead — the first warmup call traces the
    function).
    """
    _sync()
    warmup_start = time.perf_counter()
    for _ in range(n_warmup):
        out = fn()
        if out is not None:
            import jax
            jax.block_until_ready(out)
    _sync()
    warmup_secs = time.perf_counter() - warmup_start

    times = []
    for _ in range(n_iter):
        _sync()
        t0 = time.perf_counter()
        out = fn()
        if out is not None:
            import jax
            jax.block_until_ready(out)
        else:
            _sync()
        times.append(time.perf_counter() - t0)
    return GPUTimingResult(times=times, warmup_secs=warmup_secs, n_warmup=n_warmup)


def _peak_memory_mb() -> float | None:
    """Best-effort GPU peak memory in MB via `jax.devices()[0].memory_stats()`.

    Returns None on platforms where the XLA runtime doesn't expose the
    `peak_bytes_in_use` field (some ROCm builds).
    """
    try:
        import jax
        dev = jax.devices()[0]
        stats = dev.memory_stats()
        if stats is None:
            return None
        peak = stats.get("peak_bytes_in_use")
        if peak is None:
            return None
        return peak / (1024**2)
    except Exception:
        return None


def make_result(
    name: str,
    times: list[float],
    *,
    throughput_count: int | None = None,
    throughput_unit: str = "",
    peak_memory_mb: float | None = None,
    warmup_secs: float | None = None,
    n_warmup: int | None = None,
) -> TimingResult:
    times_ms = [t * 1000 for t in times]
    mean = statistics.mean(times_ms)
    return TimingResult(
        name=name,
        n_iterations=len(times_ms),
        mean_ms=mean,
        median_ms=statistics.median(times_ms),
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        stdev_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        throughput=(throughput_count / mean * 1000) if throughput_count is not None else None,
        throughput_unit=throughput_unit,
        peak_memory_mb=peak_memory_mb,
        warmup_ms=warmup_secs * 1000 if warmup_secs is not None else None,
        n_warmup=n_warmup,
    )


# ── Engine benchmarks (CPU) ──────────────────────────────────────────────────

def bench_engine(n_games: int, n_iter: int, n_warmup: int) -> list[TimingResult]:
    """Benchmark the Rust chess engine in isolation."""
    results = []
    max_ply = 256
    seed = 42

    print("\n" + "=" * 72)
    print(" ENGINE BENCHMARKS (CPU)")
    print("=" * 72)
    print(f"  n_games={n_games}  max_ply={max_ply}  n_iter={n_iter}")

    print("\n  [1/5] generate_random_games (baseline) ...")
    times = time_cpu(
        lambda: engine.generate_random_games(n_games, max_ply, seed),
        n_warmup=n_warmup, n_iter=n_iter,
    )
    results.append(make_result(
        "engine/generate_random_games", times,
        throughput_count=n_games, throughput_unit="games/s",
    ))

    print("  [2/5] generate_random_games (mate_boost=1.0) ...")
    times = time_cpu(
        lambda: engine.generate_random_games(
            n_games, max_ply, seed, mate_boost=1.0),
        n_warmup=n_warmup, n_iter=n_iter,
    )
    results.append(make_result(
        "engine/generate_random_games [mate_boost=1.0]", times,
        throughput_count=n_games, throughput_unit="games/s",
    ))

    print("  [3/5] generate_random_games (discard_ply_limit) ...")
    times = time_cpu(
        lambda: engine.generate_random_games(
            n_games, max_ply, seed, discard_ply_limit=True),
        n_warmup=n_warmup, n_iter=n_iter,
    )
    results.append(make_result(
        "engine/generate_random_games [discard_ply_limit]", times,
        throughput_count=n_games, throughput_unit="games/s",
    ))

    # Pre-generate games for downstream benchmarks
    move_ids, game_lengths, _tc = engine.generate_random_games(
        n_games, max_ply, seed)

    print("  [4/5] validate_games ...")
    times = time_cpu(
        lambda: engine.validate_games(move_ids, game_lengths),
        n_warmup=n_warmup, n_iter=n_iter,
    )
    results.append(make_result(
        "engine/validate_games", times,
        throughput_count=n_games, throughput_unit="games/s",
    ))

    print("  [5/5] extract_board_states ...")
    times = time_cpu(
        lambda: engine.extract_board_states(move_ids, game_lengths),
        n_warmup=n_warmup, n_iter=n_iter,
    )
    results.append(make_result(
        "engine/extract_board_states", times,
        throughput_count=n_games, throughput_unit="games/s",
    ))

    print()
    for r in results:
        print(f"  {r.summary_line()}")

    return results


# ── Backbone benchmarks (GPU) ───────────────────────────────────────────────

_VARIANT_MAP = {
    "tiny": "tiny",   # → TINY_SUPERNET
    "small": "small",
    "base": "base",
    "large": "large",
}


def _make_corpus_batch(batch_size: int, max_ply: int = 256, seq_len: int = 512, seed: int = 42):
    """Generate one Corpus + slice into a Batch via the v2 trainer surface."""
    from pawn.corpus import generate_corpus
    from pawn.trainer import slice_batch

    corpus = generate_corpus(
        n_games=batch_size, max_ply=max_ply, seq_len=seq_len, seed=seed
    )
    indices = np.arange(batch_size, dtype=np.int64)
    return slice_batch(corpus, indices)


def _resolve_supernet(variant: str):
    """Return the (cfg, label) for a benchmarked backbone size."""
    from pawn.config import SUPERNET, TINY_SUPERNET, VARIANTS, TINY_VARIANTS

    if variant == "tiny":
        return TINY_SUPERNET, "tiny-supernet"
    if variant == "large":
        return SUPERNET, "large(supernet)"
    if variant in VARIANTS:
        return VARIANTS[variant], variant
    if variant in TINY_VARIANTS:
        return TINY_VARIANTS[variant], f"tiny-{variant}"
    raise ValueError(f"unknown variant {variant!r}")


def _build_train_state(cfg, lr: float = 3e-4, key: int = 42):
    """Build a v2 TrainState wrapping a freshly-initialised PAWNModel."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    from pawn.model import init_model
    from pawn.trainer import TrainState

    model = init_model(cfg, jax.random.key(key))
    optimizer = optax.adamw(lr, b1=0.9, b2=0.95, weight_decay=0.01)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    state = TrainState(
        model=model,
        opt_state=opt_state,
        step=jnp.int32(0),
        key=jax.random.key(key),
    )
    return state, optimizer, model


def _make_backbone_step(
    state, optimizer, batch, jit: bool, compute_dtype=None
):
    """Return a `() -> new_state` closure timing one backbone training step.

    `jit=True` returns the jitted train_step from `pawn.trainer.make_train_step`.
    `jit=False` builds an unjitted clone of the same loss + update path so
    eager mode measures the cost of running the autograd graph in Python.

    `compute_dtype` (None / jnp.bfloat16 / jnp.float16 / jnp.float32)
    controls AMP forward dtype. Threaded into both `train_step` paths
    so the perf benchmarks exercise the same precision settings the
    real training scripts use (otherwise this script silently runs
    fp32 even when the user asks for bf16).
    """
    import jax
    import equinox as eqx

    from pawn.trainer import (
        VariantSpec, make_train_step, cross_entropy_loss,
    )

    variants = (VariantSpec(name="bench", cfg=state.model.cfg, is_supernet=True),)

    if jit:
        train_step = make_train_step(
            optimizer, variants, compute_dtype=compute_dtype
        )
    else:
        def _eager_train_step(s, b):
            def loss_fn(model):
                return cross_entropy_loss(
                    model, b, compute_dtype=compute_dtype
                )
            loss, grads = eqx.filter_value_and_grad(loss_fn)(s.model)
            updates, new_opt = optimizer.update(grads, s.opt_state, s.model)
            new_model = eqx.apply_updates(s.model, updates)
            new_state = type(s)(
                model=new_model, opt_state=new_opt,
                step=s.step + 1, key=s.key,
            )
            return new_state, loss
        train_step = _eager_train_step

    cell = {"state": state}

    def step():
        new_state, loss = train_step(cell["state"], batch)
        cell["state"] = new_state
        return loss

    return step, cell


def bench_backbone(
    variants: list[str],
    batch_size: int,
    do_jit: bool,
    do_eager: bool,
    n_iter: int,
    n_warmup: int,
    compute_dtype=None,
) -> list[TimingResult]:
    """Benchmark backbone training steps."""
    import jax

    from pawn.model import init_model

    results: list[TimingResult] = []

    dtype_label = (
        compute_dtype.dtype.name
        if compute_dtype is not None and hasattr(compute_dtype, "dtype")
        else (str(compute_dtype) if compute_dtype is not None else "float32")
    )
    print("\n" + "=" * 72)
    print(" BACKBONE TRAINING BENCHMARKS (GPU)")
    print("=" * 72)
    print(f"  batch_size={batch_size}  device={jax.devices()[0]}  n_iter={n_iter}")
    print(
        f"  framework: JAX/Equinox/Optax  attention: plain (materialised QK^T)  "
        f"compute_dtype: {dtype_label}"
    )

    batch = _make_corpus_batch(batch_size)

    modes = []
    if do_eager:
        modes.append(("eager", False))
    if do_jit:
        modes.append(("jit", True))

    for variant_name in variants:
        cfg, label = _resolve_supernet(variant_name)
        n_params = None

        for mode_name, use_jit in modes:
            bench_label = f"backbone/{label} [{mode_name}]"
            print(f"\n  {bench_label} ...")

            state, optimizer, model = _build_train_state(cfg)
            if n_params is None:
                n_params = int(sum(
                    x.size for x in jax.tree_util.tree_leaves(
                        jax.tree_util.tree_map(
                            lambda v: v if hasattr(v, "size") else None,
                            model,
                        )
                    ) if x is not None
                ))
                print(f"    params: {n_params:,}")

            step_fn, _cell = _make_backbone_step(
                state, optimizer, batch, jit=use_jit,
                compute_dtype=compute_dtype,
            )

            try:
                gpu_timing = time_gpu(step_fn, n_warmup=n_warmup, n_iter=n_iter)
            except (RuntimeError, MemoryError) as exc:
                if "out of memory" in str(exc).lower() or "RESOURCE_EXHAUSTED" in str(exc):
                    print(f"    OOM — skipping (try smaller --batch-size)")
                    continue
                raise

            peak_mb = _peak_memory_mb()

            r = make_result(
                bench_label, gpu_timing.times,
                throughput_count=batch_size,
                throughput_unit="samples/s",
                peak_memory_mb=peak_mb,
                warmup_secs=gpu_timing.warmup_secs if use_jit else None,
                n_warmup=gpu_timing.n_warmup if use_jit else None,
            )
            results.append(r)
            print(f"    {r.summary_line()}")

    return results


# ── Data-pipeline-inclusive benchmarks (GPU) ─────────────────────────────────

def bench_data_pipeline(
    batch_size: int,
    n_iter: int,
    n_warmup: int,
) -> list[TimingResult]:
    """End-to-end step where the data is freshly generated by the Rust
    engine on every iteration, versus reusing a pre-staged batch.

    In v2 there is no PyTorch DataLoader / num_workers concept — fresh
    games come straight from `engine.generate_clm_batch` inside Rust,
    which is internally parallel via rayon. This bench measures the
    overhead of the host→device transfer + Rust call per step against a
    pre-staged batch baseline.
    """
    import jax

    from pawn.config import TINY_SUPERNET
    from pawn.trainer import VariantSpec, make_train_step, slice_batch
    from pawn.corpus import generate_corpus

    results: list[TimingResult] = []
    cfg = TINY_SUPERNET
    cfg_label = "tiny-supernet"

    print("\n" + "=" * 72)
    print(" DATA-PIPELINE-INCLUSIVE BENCHMARKS (GPU)")
    print("=" * 72)
    print(f"  model={cfg_label}  batch_size={batch_size}  device={jax.devices()[0]}")

    indices = np.arange(batch_size, dtype=np.int64)

    # 1) pre-staged batch (no per-step generation cost)
    state_a, optimizer, _ = _build_train_state(cfg)
    variants = (VariantSpec(name="bench", cfg=cfg, is_supernet=True),)
    train_step = make_train_step(optimizer, variants)
    pre_batch = _make_corpus_batch(batch_size)
    pre_cell = {"state": state_a}

    def pre_staged_step():
        new_state, loss = train_step(pre_cell["state"], pre_batch)
        pre_cell["state"] = new_state
        return loss

    print("\n  [1/2] pre-staged batch (reuse, no per-step generation) ...")
    timing = time_gpu(pre_staged_step, n_warmup=n_warmup, n_iter=n_iter)
    results.append(make_result(
        "data_pipeline/pre-staged",
        timing.times,
        throughput_count=batch_size,
        throughput_unit="samples/s",
        peak_memory_mb=_peak_memory_mb(),
        warmup_secs=timing.warmup_secs,
        n_warmup=timing.n_warmup,
    ))
    print(f"    {results[-1].summary_line()}")

    # 2) fresh corpus per step (Rust engine + host→device on every call).
    # Build a *separate* TrainState — `train_step` donates the previous
    # state buffer, so reusing `state_a` after `pre_staged_step` consumed
    # it would error with "Donation requested for invalid buffer".
    state_b, _, _ = _build_train_state(cfg, key=43)
    fresh_cell = {"state": state_b, "seed": 1000}

    def fresh_per_step():
        corpus = generate_corpus(
            n_games=batch_size, max_ply=256, seq_len=512,
            seed=fresh_cell["seed"],
        )
        fresh_cell["seed"] += 1
        b = slice_batch(corpus, indices)
        new_state, loss = train_step(fresh_cell["state"], b)
        fresh_cell["state"] = new_state
        return loss

    print("\n  [2/2] fresh corpus per step (engine + slice_batch every iter) ...")
    timing = time_gpu(fresh_per_step, n_warmup=n_warmup, n_iter=n_iter)
    results.append(make_result(
        "data_pipeline/fresh-per-step",
        timing.times,
        throughput_count=batch_size,
        throughput_unit="samples/s",
        peak_memory_mb=_peak_memory_mb(),
        warmup_secs=timing.warmup_secs,
        n_warmup=timing.n_warmup,
    ))
    print(f"    {results[-1].summary_line()}")

    return results


# ── Concurrency sweep (GPU) ──────────────────────────────────────────────────

_WORKER_SCRIPT = '''
"""JAX worker process for the concurrency benchmark. Runs `n_iter` jit-compiled
training steps on a (variant, optional-adapter) backbone and reports wall time
via a result file.

Barrier protocol mirrors the v1 PyTorch worker:
1. Each worker initialises + compiles + warms up independently
2. Touches a `ready_<id>` sentinel in the barrier dir
3. Spins until all `n_workers` sentinels exist
4. Runs `n_iter` timed iterations
5. Writes a JSON result blob
"""
import sys, time, json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

batch_size = int(sys.argv[1])
n_warmup = int(sys.argv[2])
n_iter = int(sys.argv[3])
variant = sys.argv[4]
adapter_kind = sys.argv[5]   # "none", "lora", "film", "bottleneck"
result_path = sys.argv[6]
worker_id = int(sys.argv[7])
n_workers = int(sys.argv[8])
barrier_dir = sys.argv[9]

from pawn.config import SUPERNET, TINY_SUPERNET, VARIANTS, TINY_VARIANTS
from pawn.model import init_model
from pawn.corpus import generate_corpus
from pawn.trainer import VariantSpec, make_train_step, slice_batch
import optax

if variant == "tiny":
    cfg = TINY_SUPERNET
elif variant == "large":
    cfg = SUPERNET
elif variant in VARIANTS:
    cfg = VARIANTS[variant]
else:
    cfg = TINY_VARIANTS[variant]

model = init_model(cfg, jax.random.key(42 + worker_id))

corpus = generate_corpus(
    n_games=batch_size, max_ply=256, seq_len=512, seed=42 + worker_id,
)
batch = slice_batch(corpus, np.arange(batch_size, dtype=np.int64))

if adapter_kind == "none":
    from pawn.trainer import TrainState
    import equinox as eqx
    optimizer = optax.adamw(3e-4, b1=0.9, b2=0.95, weight_decay=0.01)
    state = TrainState(
        model=model,
        opt_state=optimizer.init(eqx.filter(model, eqx.is_inexact_array)),
        step=jnp.int32(0), key=jax.random.key(42 + worker_id),
    )
    variants = (VariantSpec(name="bench", cfg=cfg, is_supernet=True),)
    train_step = make_train_step(optimizer, variants)
else:
    from pawn.adapter_trainer import (
        AdapterTrainState, dispatch_init, dispatch_filter,
        make_adapter_train_step,
    )
    if adapter_kind == "lora":
        from pawn.adapters.lora import LoRAConfig
        adapter_cfg = LoRAConfig(rank=4, targets="qkvo")
    elif adapter_kind == "film":
        from pawn.adapters.film import FiLMConfig
        adapter_cfg = FiLMConfig(use_output_film=True)
    elif adapter_kind == "bottleneck":
        from pawn.adapters.bottleneck import BottleneckConfig
        adapter_cfg = BottleneckConfig(dim=8)
    else:
        raise SystemExit(f"unknown adapter {adapter_kind!r}")
    adapter = dispatch_init(adapter_kind)(model, adapter_cfg, jax.random.key(99))
    filt = dispatch_filter(adapter_kind)
    trainable, _ = __import__("equinox").partition(adapter, filt)
    optimizer = optax.adamw(3e-4, b1=0.9, b2=0.95, weight_decay=0.01)
    state = AdapterTrainState(
        backbone=model, adapter=adapter,
        opt_state=optimizer.init(trainable),
        step=jnp.int32(0), key=jax.random.key(42 + worker_id),
    )
    train_step = make_adapter_train_step(adapter_kind, optimizer)

def step():
    global state
    state, loss = train_step(state, batch)
    return loss

for _ in range(n_warmup):
    jax.block_until_ready(step())

ready_file = Path(barrier_dir) / f"ready_{worker_id}"
ready_file.touch()

deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    ready_count = sum(1 for _ in Path(barrier_dir).glob("ready_*"))
    if ready_count >= n_workers:
        break
    time.sleep(0.05)

times = []
for _ in range(n_iter):
    t0 = time.perf_counter()
    out = step()
    jax.block_until_ready(out)
    times.append(time.perf_counter() - t0)

try:
    stats = jax.devices()[0].memory_stats() or {}
    peak_mb = (stats.get("peak_bytes_in_use") or 0) / (1024**2)
except Exception:
    peak_mb = 0.0

Path(result_path).write_text(json.dumps({"times": times, "peak_memory_mb": peak_mb}))
'''


def bench_concurrency(
    batch_size: int,
    n_iter: int,
    n_warmup: int,
    variant: str = "tiny",
    adapter: str = "none",
    max_n: int = 8,
) -> list[ConcurrencyResult]:
    """Sweep N concurrent JAX processes to find peak total throughput."""
    import shutil
    import subprocess
    import tempfile

    print("\n" + "=" * 72)
    print(" CONCURRENCY SWEEP (GPU)")
    print("=" * 72)
    config_str = f"  model={variant}"
    if adapter != "none":
        config_str += f"+{adapter}"
    config_str += f"  batch_size={batch_size}"
    print(config_str)
    print(f"  mode=jit  {n_warmup} warmup + {n_iter} timed iterations per process")

    worker_env = os.environ.copy()
    # JAX honours its own *_VISIBLE_DEVICES env var families; this pins
    # every worker to physical device 0 so the sweep stays on a single
    # GPU even on multi-GPU systems.
    worker_env["JAX_PLATFORMS"] = worker_env.get("JAX_PLATFORMS", "")
    worker_env["CUDA_VISIBLE_DEVICES"] = "0"
    worker_env["HIP_VISIBLE_DEVICES"] = "0"

    fd, worker_path = tempfile.mkstemp(suffix=".py", prefix="pawn_bench_worker_")
    os.close(fd)
    worker_file = Path(worker_path)
    worker_file.write_text(_WORKER_SCRIPT)

    results: list[ConcurrencyResult] = []
    single_throughput: float | None = None
    baseline_wall_secs: float | None = None

    try:
        for n_procs in range(1, max_n + 1):
            print(f"\n  N={n_procs} ...")
            result_files: list[Path] = []
            for i in range(n_procs):
                fd, rpath = tempfile.mkstemp(suffix=".json", prefix=f"pawn_bench_r{i}_")
                os.close(fd)
                result_files.append(Path(rpath))

            if baseline_wall_secs is not None:
                timeout_secs = max(baseline_wall_secs * n_procs * 3, 60)
            else:
                timeout_secs = 600

            sweep_start = time.perf_counter()
            barrier_dir = Path(tempfile.mkdtemp(prefix="pawn_bench_barrier_"))

            procs: list[subprocess.Popen[str]] = []
            for i, rf in enumerate(result_files):
                p = subprocess.Popen(
                    [sys.executable, str(worker_file),
                     str(batch_size), str(n_warmup), str(n_iter),
                     variant, adapter, str(rf),
                     str(i), str(n_procs), str(barrier_dir)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    env=worker_env,
                )
                procs.append(p)

            oom = False
            failed = False
            timed_out = False
            for p in procs:
                remaining = timeout_secs - (time.perf_counter() - sweep_start)
                try:
                    p.wait(timeout=max(remaining, 1))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    break
                if p.returncode != 0:
                    stderr = (p.stderr.read() if p.stderr else "")
                    if "out of memory" in stderr.lower() or "RESOURCE_EXHAUSTED" in stderr:
                        oom = True
                    else:
                        failed = True
                        print(f"    Worker failed (exit {p.returncode}):")
                        for line in stderr.strip().splitlines()[-3:]:
                            print(f"      {line}")

            if timed_out:
                print(f"    Timeout ({timeout_secs:.0f}s) — stopping sweep")
                for p in procs:
                    p.kill()
                    p.wait()
                for rf in result_files:
                    rf.unlink(missing_ok=True)
                shutil.rmtree(barrier_dir, ignore_errors=True)
                break

            if oom or failed:
                for rf in result_files:
                    rf.unlink(missing_ok=True)
                shutil.rmtree(barrier_dir, ignore_errors=True)
                if oom:
                    print(f"    OOM with N={n_procs} — stopping sweep")
                break

            shutil.rmtree(barrier_dir, ignore_errors=True)
            worker_results = []
            for rf in result_files:
                try:
                    data = json.loads(rf.read_text())
                    worker_results.append(data)
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                rf.unlink(missing_ok=True)

            if len(worker_results) != n_procs:
                print(f"    Only {len(worker_results)}/{n_procs} workers reported — stopping")
                break

            all_means = []
            total_vram_mb = 0.0
            for wr in worker_results:
                times_ms = [t * 1000 for t in wr["times"]]
                all_means.append(statistics.mean(times_ms))
                total_vram_mb += wr.get("peak_memory_mb") or 0.0

            wall_ms = max(all_means)
            total_throughput = n_procs * batch_size / wall_ms * 1000
            per_model_throughput = total_throughput / n_procs

            if single_throughput is None:
                single_throughput = per_model_throughput
                baseline_wall_secs = time.perf_counter() - sweep_start

            speedup = total_throughput / single_throughput

            cr = ConcurrencyResult(
                n_models=n_procs,
                step_ms=round(wall_ms, 1),
                per_model_ms=round(wall_ms / n_procs, 1),
                total_throughput=round(total_throughput),
                per_model_throughput=round(per_model_throughput),
                total_vram_mb=round(total_vram_mb),
                speedup=round(speedup, 2),
            )
            results.append(cr)

            print(f"    wall: {wall_ms:.0f}ms"
                  f"  total: {total_throughput:.0f} samples/s"
                  f"  per-job: {per_model_throughput:.0f} samples/s"
                  f"  speedup: {speedup:.2f}x"
                  f"  VRAM: {total_vram_mb:.0f} MB")

            if len(results) >= 2:
                prev_total = results[-2].total_throughput
                if total_throughput <= prev_total:
                    print(f"\n  Total throughput decreased: N={n_procs}"
                          f" ({total_throughput:.0f}) ≤ N={n_procs - 1}"
                          f" ({prev_total:.0f}) — stopping sweep")
                    best = max(results, key=lambda r: r.total_throughput)
                    print(f"  Peak total throughput: N={best.n_models}"
                          f" ({best.total_throughput} samples/s)")
                    break
        else:
            if results:
                best = max(results, key=lambda r: r.total_throughput)
                print(f"\n  Peak total throughput: N={best.n_models}"
                      f" ({best.total_throughput} samples/s)"
                      f" — max_n={max_n} reached without degradation")
    finally:
        worker_file.unlink(missing_ok=True)

    return results


# ── Adapter benchmarks (GPU) ─────────────────────────────────────────────────

def _build_adapter_state(model, adapter_kind: str, key: int = 99):
    """Build an `AdapterTrainState` + optimizer for the given strategy."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    from pawn.adapter_trainer import (
        AdapterTrainState, dispatch_init, dispatch_filter,
    )

    if adapter_kind == "lora":
        from pawn.adapters.lora import LoRAConfig
        cfg = LoRAConfig(rank=4, targets="qkvo")
    elif adapter_kind == "film":
        from pawn.adapters.film import FiLMConfig
        cfg = FiLMConfig(use_output_film=True)
    elif adapter_kind == "bottleneck":
        from pawn.adapters.bottleneck import BottleneckConfig
        cfg = BottleneckConfig(dim=8)
    else:
        raise ValueError(f"unknown adapter {adapter_kind!r}")

    adapter = dispatch_init(adapter_kind)(model, cfg, jax.random.key(key))
    filt = dispatch_filter(adapter_kind)
    trainable, _ = eqx.partition(adapter, filt)
    optimizer = optax.adamw(3e-4, b1=0.9, b2=0.95, weight_decay=0.01)
    state = AdapterTrainState(
        backbone=model, adapter=adapter,
        opt_state=optimizer.init(trainable),
        step=jnp.int32(0), key=jax.random.key(key),
    )
    n_trainable = int(sum(
        x.size for x in jax.tree_util.tree_leaves(trainable)
        if hasattr(x, "size")
    ))
    return state, optimizer, n_trainable


def bench_adapters(
    adapter_types: list[str],
    batch_size: int,
    do_jit: bool,
    do_eager: bool,
    n_iter: int,
    n_warmup: int,
) -> list[TimingResult]:
    """Benchmark adapter training steps on a frozen `base` backbone."""
    import jax

    from pawn.config import VARIANTS
    from pawn.adapter_trainer import make_adapter_train_step, dispatch_apply
    from pawn.trainer import cross_entropy_loss

    results: list[TimingResult] = []

    print("\n" + "=" * 72)
    print(" ADAPTER TRAINING BENCHMARKS (GPU)")
    print("=" * 72)
    print(f"  backbone=base  batch_size={batch_size}  device={jax.devices()[0]}  n_iter={n_iter}")

    batch = _make_corpus_batch(batch_size)

    modes = []
    if do_eager:
        modes.append(("eager", False))
    if do_jit:
        modes.append(("jit", True))

    base_cfg = VARIANTS["base"]
    _state, _opt, base_model = _build_train_state(base_cfg)
    n_total = int(sum(
        x.size for x in jax.tree_util.tree_leaves(base_model)
        if hasattr(x, "size")
    ))

    for adapter_name in adapter_types:
        for mode_name, use_jit in modes:
            label = f"adapter/{adapter_name} [{mode_name}]"
            print(f"\n  {label} ...")

            state, optimizer, n_adapter = _build_adapter_state(base_model, adapter_name)
            print(f"    adapter params: {n_adapter:,} / {n_total:,} total")

            if use_jit:
                train_step = make_adapter_train_step(adapter_name, optimizer)
            else:
                apply_fn = dispatch_apply(adapter_name)

                def _eager_adapter_step(s, b, _apply=apply_fn, _opt=optimizer):
                    import equinox as eqx
                    def loss_fn(adapter):
                        effective = _apply(s.backbone, adapter)
                        return cross_entropy_loss(effective, b)
                    loss, grads = eqx.filter_value_and_grad(loss_fn)(s.adapter)
                    updates, new_opt = _opt.update(grads, s.opt_state, s.adapter)
                    new_adapter = eqx.apply_updates(s.adapter, updates)
                    new_state = type(s)(
                        backbone=s.backbone, adapter=new_adapter,
                        opt_state=new_opt, step=s.step + 1, key=s.key,
                    )
                    return new_state, loss
                train_step = _eager_adapter_step

            cell = {"state": state}

            def step():
                new_state, loss = train_step(cell["state"], batch)
                cell["state"] = new_state
                return loss

            try:
                gpu_timing = time_gpu(step, n_warmup=n_warmup, n_iter=n_iter)
            except (RuntimeError, MemoryError) as exc:
                if "out of memory" in str(exc).lower() or "RESOURCE_EXHAUSTED" in str(exc):
                    print(f"    OOM — skipping (try smaller --batch-size)")
                    continue
                raise

            peak_mb = _peak_memory_mb()

            r = make_result(
                label, gpu_timing.times,
                throughput_count=batch_size,
                throughput_unit="samples/s",
                peak_memory_mb=peak_mb,
                warmup_secs=gpu_timing.warmup_secs if use_jit else None,
                n_warmup=gpu_timing.n_warmup if use_jit else None,
            )
            results.append(r)
            print(f"    {r.summary_line()}")

    return results


# ── Report ───────────────────────────────────────────────────────────────────

def print_summary(
    engine_results: list[TimingResult],
    backbone_results: list[TimingResult],
    data_pipeline_results: list[TimingResult],
    concurrency_results: list[ConcurrencyResult],
    adapter_results: list[TimingResult],
):
    print("\n" + "=" * 72)
    print(" BENCHMARK SUMMARY")
    print("=" * 72)

    if engine_results:
        print("\n  Engine (CPU):")
        for r in engine_results:
            print(f"    {r.summary_line()}")

    if backbone_results:
        print("\n  Backbone training (GPU):")
        for r in backbone_results:
            print(f"    {r.summary_line()}")

    if data_pipeline_results:
        print("\n  Data-pipeline-inclusive training (GPU):")
        for r in data_pipeline_results:
            print(f"    {r.summary_line()}")

    if concurrency_results:
        best_n = max(concurrency_results, key=lambda r: r.total_throughput).n_models
        print("\n  Concurrency sweep (GPU):")
        print(f"    {'':>1s} {'N':>3s}  {'round ms':>9s}  {'total samp/s':>12s}"
              f"  {'per-job samp/s':>14s}  {'speedup':>7s}  {'VRAM MB':>8s}")
        for cr in concurrency_results:
            marker = "*" if cr.n_models == best_n else " "
            print(f"    {marker} {cr.n_models:>3d}  {cr.step_ms:>9.0f}  {cr.total_throughput:>12.0f}"
                  f"  {cr.per_model_throughput:>14.0f}  {cr.speedup:>6.2f}x"
                  f"  {cr.total_vram_mb:>8.0f}")

    if adapter_results:
        print("\n  Adapter training (GPU):")
        for r in adapter_results:
            print(f"    {r.summary_line()}")

    print()


def save_json(
    path: str,
    engine_results: list[TimingResult],
    backbone_results: list[TimingResult],
    data_pipeline_results: list[TimingResult],
    concurrency_results: list[ConcurrencyResult],
    adapter_results: list[TimingResult],
    platform_info: dict,
):
    report = BenchmarkReport(
        timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
        platform_info=platform_info,
        engine_results=[asdict(r) for r in engine_results],
        backbone_results=[asdict(r) for r in backbone_results],
        data_pipeline_results=[asdict(r) for r in data_pipeline_results],
        concurrency_results=[asdict(r) for r in concurrency_results],
        adapter_results=[asdict(r) for r in adapter_results],
    )
    Path(path).write_text(json.dumps(asdict(report), indent=2))
    print(f"Results saved to {path}")


# ── System info collection ───────────────────────────────────────────────────

def _collect_cpu_cache() -> dict[str, str]:
    cache: dict[str, str] = {}
    cache_dir = Path("/sys/devices/system/cpu/cpu0/cache")
    if not cache_dir.exists():
        return cache
    for idx_dir in sorted(cache_dir.glob("index*")):
        try:
            level = (idx_dir / "level").read_text().strip()
            cache_type = (idx_dir / "type").read_text().strip()
            size = (idx_dir / "size").read_text().strip()
        except OSError:
            continue
        if cache_type == "Data":
            cache[f"l{level}d"] = size
        elif cache_type == "Instruction":
            cache[f"l{level}i"] = size
        elif cache_type == "Unified":
            cache[f"l{level}"] = size
    return cache


def _collect_system_info() -> dict:
    import multiprocessing

    cpu_name = ""
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    if not cpu_name:
        cpu_name = platform.processor() or platform.machine() or "unknown"

    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = multiprocessing.cpu_count() or 0

    ram_gb = 0.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram_gb = int(line.split()[1]) / (1024**2)
                    break
    except OSError:
        pass

    info: dict = {
        "python": sys.version.split()[0],
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu": cpu_name,
        "cpu_count": cpu_count,
        "ram_gb": round(ram_gb, 1),
    }

    cache = _collect_cpu_cache()
    if cache:
        info["cache"] = cache
    return info


def _collect_jax_info() -> dict:
    import jax

    devs = jax.devices()
    dev0 = devs[0]
    info: dict = {
        "jax": jax.__version__,
        "jax_platform": jax.default_backend(),
        "jax_device": str(dev0),
        "jax_device_count": len(devs),
    }
    stats = None
    try:
        stats = dev0.memory_stats()
    except Exception:
        pass
    if stats:
        limit = stats.get("bytes_limit") or stats.get("bytes_reservable_limit")
        if limit:
            info["vram_gb"] = round(limit / (1024**3), 1)
    return info


def _print_system_info(info: dict) -> None:
    print(f"CPU: {info['cpu']} ({info['cpu_count']} CPUs)")
    if info.get("ram_gb"):
        print(f"RAM: {info['ram_gb']:.1f} GB")
    cache = info.get("cache", {})
    if cache:
        parts = []
        if "l1d" in cache:
            parts.append(f"L1d: {cache['l1d']}")
        if "l2" in cache:
            parts.append(f"L2: {cache['l2']}")
        if "l3" in cache:
            parts.append(f"L3: {cache['l3']}")
        if parts:
            print(f"Cache: {', '.join(parts)}")


def _print_jax_info(info: dict) -> None:
    print(f"JAX: {info['jax']}  backend: {info['jax_platform']}"
          f"  device: {info['jax_device']}"
          f"  count: {info['jax_device_count']}")
    if "vram_gb" in info:
        print(f"VRAM: {info['vram_gb']:.1f} GB")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PAWN performance benchmarks (v2: JAX/Equinox/Optax)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--engine-only", action="store_true",
                       help="Only run engine (CPU) benchmarks")
    scope.add_argument("--gpu-only", action="store_true",
                       help="Only run GPU benchmarks (backbone + adapters)")

    parser.add_argument("--engine-games", type=int, default=10_000,
                        help="Number of games for engine benchmarks (default: 10000)")

    parser.add_argument("--variants", nargs="+", default=["tiny", "small"],
                        choices=["tiny", "small", "base", "large"],
                        help="Backbone variants to benchmark (default: tiny small)")
    parser.add_argument("--adapters", nargs="+", default=["lora", "film", "bottleneck"],
                        choices=["lora", "film", "bottleneck"],
                        help="Adapter types to benchmark (default: lora film bottleneck)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for GPU benchmarks (default: 64)")
    parser.add_argument("--no-backbone", action="store_true",
                        help="Skip backbone benchmarks")
    parser.add_argument("--no-data-pipeline", action="store_true",
                        help="Skip data-pipeline-inclusive benchmarks")
    parser.add_argument("--no-adapters", action="store_true",
                        help="Skip adapter benchmarks")
    parser.add_argument("--no-concurrency", action="store_true",
                        help="Skip concurrency sweep")

    parser.add_argument("--sweep-variant", type=str, default="tiny",
                        choices=["tiny", "small", "base", "large"],
                        help="Backbone variant for concurrency sweep (default: tiny)")
    parser.add_argument("--sweep-adapter", type=str, default="none",
                        choices=["none", "lora", "film", "bottleneck"],
                        help="Adapter for concurrency sweep (default: none)")

    jit_group = parser.add_mutually_exclusive_group()
    jit_group.add_argument("--no-jit", action="store_true",
                           help="Only run eager mode (skip JIT)")
    jit_group.add_argument("--jit-only", action="store_true",
                           help="Only run JIT mode (skip eager)")

    parser.add_argument("--n-iter", type=int, default=10,
                        help="Timed iterations per benchmark (default: 10)")
    parser.add_argument("--n-warmup", type=int, default=5,
                        help="Warmup iterations per benchmark (default: 5). "
                             "JIT compilation happens during warmup.")

    parser.add_argument(
        "--amp-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help=(
            "Forward-compute dtype (default: bfloat16). Mirrors the "
            "BaseRunConfig.amp_dtype field — bf16 is what the v1 PyTorch "
            "AMP path used and what the v2 training scripts default to."
        ),
    )

    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    do_engine = not args.gpu_only
    do_gpu = not args.engine_only
    do_backbone = do_gpu and not args.no_backbone
    do_adapters = do_gpu and not args.no_adapters
    do_data_pipeline = do_gpu and not args.no_data_pipeline
    do_concurrency = do_gpu and not args.no_concurrency
    do_jit = not args.no_jit
    do_eager = not args.jit_only

    info = _collect_system_info()
    _print_system_info(info)

    if do_gpu or do_concurrency:
        try:
            jax_info = _collect_jax_info()
            info.update(jax_info)
            _print_jax_info(info)
        except Exception as exc:
            if do_engine:
                print(f"JAX unavailable ({exc!r}) — running engine benchmarks only.")
                do_backbone = do_adapters = do_data_pipeline = do_concurrency = do_gpu = False
            else:
                print(f"ERROR: JAX unavailable: {exc!r}", file=sys.stderr)
                sys.exit(1)

    engine_results: list[TimingResult] = []
    backbone_results: list[TimingResult] = []
    data_pipeline_results: list[TimingResult] = []
    concurrency_results: list[ConcurrencyResult] = []
    adapter_results: list[TimingResult] = []

    # Resolve amp_dtype → jnp dtype.
    import jax.numpy as jnp_local
    _DTYPE_MAP = {
        "bfloat16": jnp_local.bfloat16,
        "float16": jnp_local.float16,
        "float32": None,
    }
    compute_dtype = _DTYPE_MAP[args.amp_dtype]

    if do_engine:
        engine_results = bench_engine(
            args.engine_games, args.n_iter, args.n_warmup,
        )

    if do_backbone:
        backbone_results = bench_backbone(
            args.variants, args.batch_size,
            do_jit, do_eager, args.n_iter, args.n_warmup,
            compute_dtype=compute_dtype,
        )
    if do_data_pipeline:
        data_pipeline_results = bench_data_pipeline(
            args.batch_size, args.n_iter, args.n_warmup,
        )
    if do_concurrency:
        concurrency_results = bench_concurrency(
            args.batch_size, args.n_iter, args.n_warmup,
            variant=args.sweep_variant, adapter=args.sweep_adapter,
        )
    if do_adapters:
        adapter_results = bench_adapters(
            args.adapters, args.batch_size,
            do_jit, do_eager, args.n_iter, args.n_warmup,
        )

    print_summary(engine_results, backbone_results, data_pipeline_results,
                  concurrency_results, adapter_results)

    if args.json:
        save_json(
            args.json, engine_results, backbone_results, data_pipeline_results,
            concurrency_results, adapter_results, info,
        )


if __name__ == "__main__":
    main()
