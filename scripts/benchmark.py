#!/usr/bin/env python3
"""Standardized performance benchmarks for PAWN.

Benchmarks three layers:
  1. Rust engine (CPU): game generation, validation, board extraction, legal masks
  2. Backbone training steps (GPU): small/base models, compiled vs eager
  3. Adapter training steps (GPU): LoRA, FiLM, Bottleneck on frozen backbone

Defaults per platform:
  AMD/ROCm:    SDPA MATH backend, bf16 AMP
  NVIDIA/CUDA: flash attention,   bf16 AMP

Usage:
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --engine-only
    uv run python scripts/benchmark.py --gpu-only
    uv run python scripts/benchmark.py --variants small base --batch-size 128
    uv run python scripts/benchmark.py --adapters lora film
    uv run python scripts/benchmark.py --no-backbone         # skip backbone, run adapters only
    uv run python scripts/benchmark.py --no-adapters         # skip adapters, run backbone only
    uv run python scripts/benchmark.py --no-compile          # eager only
    uv run python scripts/benchmark.py --compile-only        # compiled only
    uv run python scripts/benchmark.py --json results.json   # machine-readable output
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
    throughput: float | None = None      # items/sec (games, steps, etc.)
    throughput_unit: str = ""
    peak_memory_mb: float | None = None  # GPU peak memory
    warmup_ms: float | None = None       # total warmup time (compilation + first runs)
    n_warmup: int | None = None          # number of warmup iterations

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
    step_ms: float              # wall time for one round (all N models stepped)
    per_model_ms: float         # step_ms / n_models
    total_throughput: float     # total samples/s across all models
    per_model_throughput: float # samples/s per model
    total_vram_mb: float        # sum of peak VRAM across all processes
    speedup: float              # total_throughput / single_model_throughput


@dataclass
class BenchmarkReport:
    timestamp: str = ""
    platform_info: dict = field(default_factory=dict)
    engine_results: list[dict] = field(default_factory=list)
    backbone_results: list[dict] = field(default_factory=list)
    dataloader_results: list[dict] = field(default_factory=list)
    concurrency_results: list[dict] = field(default_factory=list)
    adapter_results: list[dict] = field(default_factory=list)


# ── Timing helpers ───────────────────────────────────────────────────────────

def time_cpu(fn, *, n_warmup: int = 2, n_iter: int = 10) -> list[float]:
    """Time a CPU function, returning wall-clock seconds per call."""
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
    """Raw timing data from time_gpu."""
    times: list[float]         # timed iteration durations (seconds)
    warmup_secs: float         # total warmup wall time (seconds)
    n_warmup: int              # number of warmup iterations


def time_gpu(fn, *, n_warmup: int = 3, n_iter: int = 10,
             reset_peak_memory: bool = False) -> GPUTimingResult:
    """Time a GPU function with CUDA synchronization.

    Returns timed iteration durations plus total warmup time (which includes
    compilation overhead for torch.compile'd functions).

    When reset_peak_memory is True, peak memory stats are reset after warmup
    so the caller gets steady-state memory usage, not compilation overhead.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warmup_start = time.perf_counter()
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warmup_secs = time.perf_counter() - warmup_start

    if torch.cuda.is_available() and reset_peak_memory:
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(n_iter):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return GPUTimingResult(times=times, warmup_secs=warmup_secs, n_warmup=n_warmup)


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

    # 1. Random game generation (baseline)
    print("\n  [1/7] generate_random_games (baseline) ...")
    times = time_cpu(
        lambda: engine.generate_random_games(n_games, max_ply, seed),
        n_warmup=n_warmup, n_iter=n_iter,
    )
    results.append(make_result(
        "engine/generate_random_games", times,
        throughput_count=n_games, throughput_unit="games/s",
    ))

    # 2. Random games with mate_boost=1.0 (always take mate-in-1)
    print("  [2/7] generate_random_games (mate_boost=1.0) ...")
    times = time_cpu(
        lambda: engine.generate_random_games(
            n_games, max_ply, seed, mate_boost=1.0),
        n_warmup=n_warmup, n_iter=n_iter,
    )
    results.append(make_result(
        "engine/generate_random_games [mate_boost=1.0]", times,
        throughput_count=n_games, throughput_unit="games/s",
    ))

    # 3. Random games with discard_ply_limit=True
    #    Engine discards ply-limit games and retries, so actual returned count
    #    equals n_games but wall time reflects extra generation work.
    print("  [3/7] generate_random_games (discard_ply_limit) ...")
    times = time_cpu(
        lambda: engine.generate_random_games(
            n_games, max_ply, seed, discard_ply_limit=True),
        n_warmup=n_warmup, n_iter=n_iter,
    )
    results.append(make_result(
        "engine/generate_random_games [discard_ply_limit]", times,
        throughput_count=n_games, throughput_unit="games/s",
    ))

    # 4. Full CLM batch generation (games + packing)
    print("  [4/7] generate_clm_batch ...")
    times = time_cpu(
        lambda: engine.generate_clm_batch(n_games, max_ply, seed),
        n_warmup=n_warmup, n_iter=n_iter,
    )
    results.append(make_result(
        "engine/generate_clm_batch", times,
        throughput_count=n_games, throughput_unit="games/s",
    ))

    # Pre-generate games for downstream benchmarks
    move_ids, game_lengths, _tc = engine.generate_random_games(
        n_games, max_ply, seed)

    # 5. validate_games
    print("  [5/7] validate_games ...")
    times = time_cpu(
        lambda: engine.validate_games(move_ids, game_lengths),
        n_warmup=n_warmup, n_iter=n_iter,
    )
    results.append(make_result(
        "engine/validate_games", times,
        throughput_count=n_games, throughput_unit="games/s",
    ))

    # 6. compute_edge_stats_per_game (validation + stat bits)
    print("  [6/7] compute_edge_stats_per_game (stat bits) ...")
    times = time_cpu(
        lambda: engine.compute_edge_stats_per_game(move_ids, game_lengths),
        n_warmup=n_warmup, n_iter=n_iter,
    )
    results.append(make_result(
        "engine/compute_edge_stats_per_game", times,
        throughput_count=n_games, throughput_unit="games/s",
    ))

    # 7. extract_board_states
    print("  [7/7] extract_board_states ...")
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


# ── GPU training step helper ─────────────────────────────────────────────────

def _make_backbone_step(model, optimizer, scaler, batch, device, forward_fn):
    """Build a closure for one full backbone training step."""
    import torch

    def step():
        model.train()
        with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=True):
            loss, _metrics = forward_fn(
                batch["input_ids"], batch["loss_mask"], batch["targets"],
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return step


def _make_adapter_step(adapter_model, optimizer, scaler, batch, device, forward_fn,
                       trainable):
    """Build a closure for one full adapter training step."""
    import torch
    from pawn.model import clm_loss

    def step():
        adapter_model.train()
        with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=True):
            logits = forward_fn(batch["input_ids"])
            loss, _metrics = clm_loss(
                logits, batch["targets"], batch["loss_mask"],
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return step


def _make_batch(batch_size: int, device: str):
    """Generate a CLM batch on the given device."""
    import torch

    input_ids, targets, loss_mask, _mid, _gl, _tc = \
        engine.generate_clm_batch(batch_size, 256, seed=42)
    return {
        "input_ids": torch.from_numpy(input_ids).long().to(device),
        "targets": torch.from_numpy(targets).long().to(device),
        "loss_mask": torch.from_numpy(loss_mask).to(device),
    }


# ── Backbone benchmarks (GPU) ───────────────────────────────────────────────

def bench_backbone(
    variants: list[str],
    batch_size: int,
    device: str,
    do_compile: bool,
    do_eager: bool,
    n_iter: int,
    n_warmup: int,
    sdpa_backend,
) -> list[TimingResult]:
    """Benchmark backbone training steps."""
    import torch
    import pawn.model as model_module
    from pawn.config import CLMConfig
    from pawn.model import PAWNCLM

    results = []
    variant_map = {
        "toy": CLMConfig.toy,
        "small": CLMConfig.small,
        "base": CLMConfig.base,
        "large": CLMConfig.large,
    }

    print("\n" + "=" * 72)
    print(" BACKBONE TRAINING BENCHMARKS (GPU)")
    print("=" * 72)
    print(f"  batch_size={batch_size}  device={device}  n_iter={n_iter}")
    print(f"  AMP: bf16  SDPA: {sdpa_backend.name if sdpa_backend else 'default (flash)'}")

    batch = _make_batch(batch_size, device)

    modes = []
    if do_eager:
        modes.append(("eager", False))
    if do_compile:
        modes.append(("compiled", True))

    for variant_name in variants:
        cfg = variant_map[variant_name]()
        n_params = None

        for mode_name, use_compile in modes:
            label = f"backbone/{variant_name} [{mode_name}]"
            print(f"\n  {label} ...")

            # Set SDPA backend before compile
            model_module.SDPA_BACKEND = sdpa_backend

            model = PAWNCLM(cfg).to(device)
            if n_params is None:
                n_params = sum(p.numel() for p in model.parameters())
                print(f"    params: {n_params:,}")

            forward_fn = model.forward_train
            if use_compile:
                forward_fn = torch.compile(forward_fn)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95),
            )
            scaler = torch.amp.GradScaler(device, enabled=True)

            step = _make_backbone_step(
                model, optimizer, scaler, batch, device, forward_fn,
            )

            try:
                gpu_timing = time_gpu(step, n_warmup=n_warmup, n_iter=n_iter,
                                      reset_peak_memory=True)
            except torch.cuda.OutOfMemoryError:
                print(f"    OOM — skipping (try smaller --batch-size)")
                del model, optimizer, scaler
                torch.cuda.empty_cache()
                continue

            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

            r = make_result(
                label, gpu_timing.times,
                throughput_count=batch_size,
                throughput_unit="samples/s",
                peak_memory_mb=peak_mb,
                warmup_secs=gpu_timing.warmup_secs if use_compile else None,
                n_warmup=gpu_timing.n_warmup if use_compile else None,
            )
            results.append(r)
            print(f"    {r.summary_line()}")

            # Free before next iteration
            del model, optimizer, scaler
            torch.cuda.empty_cache()

    return results


# ── Dataloader-inclusive benchmarks (GPU) ─────────────────────────────────────

def bench_dataloader(
    batch_size: int,
    device: str,
    n_iter: int,
    n_warmup: int,
    sdpa_backend,
) -> list[TimingResult]:
    """Benchmark end-to-end training steps with DataLoader data generation.

    Runs pawn-base in compiled mode with workers=0 and workers=2 to measure
    the impact of data loading on training throughput.
    """
    import torch
    import torch.utils.data
    import pawn.model as model_module
    from pawn.config import CLMConfig
    from pawn.data import CLMDataset
    from pawn.model import PAWNCLM

    results = []
    cfg = CLMConfig.base()

    print("\n" + "=" * 72)
    print(" DATALOADER-INCLUSIVE BENCHMARKS (GPU)")
    print("=" * 72)
    print(f"  model=base  batch_size={batch_size}  device={device}  n_iter={n_iter}")
    print(f"  AMP: bf16  SDPA: {sdpa_backend.name if sdpa_backend else 'default (flash)'}")

    for num_workers in [0, 2]:
        label = f"dataloader/base [compiled, workers={num_workers}]"
        print(f"\n  {label} ...")

        model_module.SDPA_BACKEND = sdpa_backend

        model = PAWNCLM(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    params: {n_params:,}")

        forward_fn = torch.compile(model.forward_train)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95),
        )
        scaler = torch.amp.GradScaler(device, enabled=True)

        dataset = CLMDataset(
            batch_size=batch_size, max_ply=256, base_seed=42,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=num_workers,
            multiprocessing_context="spawn" if num_workers > 0 else None,
            pin_memory=(num_workers > 0),
        )
        batch_iter = iter(loader)

        def step():
            batch = next(batch_iter)
            batch = {
                k: v.to(device, non_blocking=True) for k, v in batch.items()
            }
            model.train()
            with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=True):
                loss, _metrics = forward_fn(
                    batch["input_ids"], batch["loss_mask"], batch["targets"],
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        try:
            gpu_timing = time_gpu(step, n_warmup=n_warmup, n_iter=n_iter,
                                  reset_peak_memory=True)
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM — skipping (try smaller --batch-size)")
            del model, optimizer, scaler, loader, batch_iter
            torch.cuda.empty_cache()
            continue

        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

        r = make_result(
            label, gpu_timing.times,
            throughput_count=batch_size,
            throughput_unit="samples/s",
            peak_memory_mb=peak_mb,
            warmup_secs=gpu_timing.warmup_secs,
            n_warmup=gpu_timing.n_warmup,
        )
        results.append(r)
        print(f"    {r.summary_line()}")

        del model, optimizer, scaler, loader, batch_iter
        torch.cuda.empty_cache()

    return results


# ── Concurrency sweep (GPU) ──────────────────────────────────────────────────

_WORKER_SCRIPT = '''
"""Worker process for concurrency benchmark. Runs N training steps and
reports wall time back to the parent via a result file."""
import sys, time, json, torch
import torch.nn.functional as F
import pawn.model as model_module
from pawn.config import CLMConfig
from pawn.model import PAWNCLM, clm_loss
from torch.nn.attention import SDPBackend
import chess_engine as engine

batch_size = int(sys.argv[1])
n_warmup = int(sys.argv[2])
n_iter = int(sys.argv[3])
sdpa_backend_name = sys.argv[4]   # "MATH" or "NONE"
variant = sys.argv[5]
adapter_type = sys.argv[6]        # "none", "lora", "film", "bottleneck"
result_path = sys.argv[7]

device = "cuda"
cfg = getattr(CLMConfig, variant)()

if sdpa_backend_name != "NONE":
    model_module.SDPA_BACKEND = getattr(SDPBackend, sdpa_backend_name)

backbone = PAWNCLM(cfg).to(device)

# Optionally wrap in an adapter
if adapter_type == "lora":
    from pawn.adapters.lora import LoRACLM
    model = LoRACLM(backbone, rank=4, attn_targets="qkvo").to(device)
elif adapter_type == "film":
    from pawn.adapters.film import FiLMCLM
    model = FiLMCLM(backbone, use_output_film=True).to(device)
elif adapter_type == "bottleneck":
    from pawn.adapters.bottleneck import BottleneckCLM
    model = BottleneckCLM(backbone, bottleneck_dim=8).to(device)
else:
    model = backbone

is_adapter = adapter_type != "none"

if is_adapter:
    trainable = [p for p in model.parameters() if p.requires_grad]
    forward_fn = torch.compile(model.forward)
else:
    trainable = list(model.parameters())
    forward_fn = torch.compile(model.forward_train)

optimizer = torch.optim.AdamW(
    trainable, lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
scaler = torch.amp.GradScaler(device, enabled=True)

input_ids, targets, loss_mask, *_ = engine.generate_clm_batch(batch_size, 256, seed=42)
batch = {
    "input_ids": torch.from_numpy(input_ids).long().to(device),
    "targets": torch.from_numpy(targets).long().to(device),
    "loss_mask": torch.from_numpy(loss_mask).to(device),
}

def step():
    model.train()
    with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=True):
        if is_adapter:
            logits = forward_fn(batch["input_ids"])
            loss, _ = clm_loss(logits, batch["targets"], batch["loss_mask"])
        else:
            loss, _ = forward_fn(batch["input_ids"], batch["loss_mask"], batch["targets"])
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

# Warmup (includes compilation)
for _ in range(n_warmup):
    step()
torch.cuda.synchronize()

# Timed iterations
times = []
for _ in range(n_iter):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

json.dump({"times": times, "peak_memory_mb": peak_mb}, open(result_path, "w"))
'''


def bench_concurrency(
    batch_size: int,
    device: str,
    n_iter: int,
    n_warmup: int,
    sdpa_backend,
    variant: str = "small",
    adapter: str = "none",
    max_n: int = 10,
) -> list[ConcurrencyResult]:
    """Sweep N concurrent training processes to find peak total throughput.

    Spawns N independent Python processes, each running a compiled training
    loop on the same GPU. Measures aggregate throughput and per-process
    throughput. Stops when total throughput decreases or OOM is hit.

    Args:
        variant: backbone model size (toy/small/base/large)
        adapter: adapter type to wrap the backbone (none/lora/film/bottleneck)
    """
    import subprocess
    import tempfile

    sdpa_name = sdpa_backend.name if sdpa_backend else "NONE"

    print("\n" + "=" * 72)
    print(" CONCURRENCY SWEEP (GPU)")
    print("=" * 72)
    config_str = f"  model={variant}"
    if adapter != "none":
        config_str += f"+{adapter}"
    config_str += f"  batch_size={batch_size}  device={device}"
    print(config_str)
    print(f"  mode=compiled  {n_warmup} warmup + {n_iter} timed iterations per process")
    print(f"  AMP: bf16  SDPA: {sdpa_name}")

    # Detect CUDA MPS (changes concurrency dynamics)
    mps_active = False
    try:
        import subprocess as _sp
        _ps = _sp.run(["ps", "-eo", "comm"], capture_output=True, text=True, timeout=5)
        if "nvidia-cuda-mps" in _ps.stdout:
            mps_active = True
            print("  NOTE: CUDA MPS is active — results reflect MPS scheduling")
    except (FileNotFoundError, _sp.TimeoutExpired):
        pass

    # Pin all workers to GPU 0 so the sweep measures contention on a single
    # GPU, even on multi-GPU systems.
    worker_env = os.environ.copy()
    worker_env["CUDA_VISIBLE_DEVICES"] = "0"

    # Write worker script to a temp file
    worker_file = Path(tempfile.mktemp(suffix=".py", prefix="pawn_bench_worker_"))
    worker_file.write_text(_WORKER_SCRIPT)

    results: list[ConcurrencyResult] = []
    single_throughput: float | None = None
    baseline_wall_secs: float | None = None  # N=1 total wall time

    try:
        for n_procs in range(1, max_n + 1):
            print(f"\n  N={n_procs} ...")

            # Create result files for each worker
            result_files = [
                Path(tempfile.mktemp(suffix=".json", prefix=f"pawn_bench_r{i}_"))
                for i in range(n_procs)
            ]

            # Timeout: for N=1, be generous (10 min for compilation).
            # For N>1, allow N * baseline * 3 — if it takes longer than that,
            # the GPU is thrashing and we should stop.
            if baseline_wall_secs is not None:
                timeout_secs = max(baseline_wall_secs * n_procs * 3, 60)
            else:
                timeout_secs = 600  # 10 min for N=1 (includes compilation)

            sweep_start = time.perf_counter()

            # Launch all workers, pinned to GPU 0
            procs: list[subprocess.Popen[str]] = []
            for rf in result_files:
                p = subprocess.Popen(
                    [sys.executable, str(worker_file),
                     str(batch_size), str(n_warmup), str(n_iter),
                     sdpa_name, variant, adapter, str(rf)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    env=worker_env,
                )
                procs.append(p)

            # Wait for all workers with timeout
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
                    if "OutOfMemoryError" in stderr or "out of memory" in stderr.lower():
                        oom = True
                    else:
                        failed = True
                        print(f"    Worker failed (exit {p.returncode}):")
                        for line in stderr.strip().splitlines()[-3:]:
                            print(f"      {line}")

            if timed_out:
                print(f"    Timeout ({timeout_secs:.0f}s) — GPU thrashing, stopping sweep")
                for p in procs:
                    p.kill()
                    p.wait()
                for rf in result_files:
                    rf.unlink(missing_ok=True)
                break

            if oom:
                print(f"    OOM with N={n_procs} — stopping sweep")
                for rf in result_files:
                    rf.unlink(missing_ok=True)
                break

            if failed:
                for rf in result_files:
                    rf.unlink(missing_ok=True)
                break

            # Collect results
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

            # Each worker ran independently; wall time is the max across workers
            # (they ran in parallel, so total time = slowest worker)
            all_means = []
            total_vram_mb = 0.0
            for wr in worker_results:
                times_ms = [t * 1000 for t in wr["times"]]
                all_means.append(statistics.mean(times_ms))
                total_vram_mb += wr["peak_memory_mb"]

            # The effective wall time per "round" is the slowest worker
            wall_ms = max(all_means)
            # Total throughput: all N processes produce batch_size samples in wall_ms
            total_throughput = n_procs * batch_size / wall_ms * 1000
            per_model_throughput = total_throughput / n_procs

            if single_throughput is None:
                single_throughput = per_model_throughput
                # Record N=1 total wall time for timeout calculation
                baseline_wall_secs = time.perf_counter() - sweep_start

            speedup = total_throughput / single_throughput

            cr = ConcurrencyResult(
                n_models=n_procs,
                step_ms=round(wall_ms, 1),
                per_model_ms=round(wall_ms, 1),
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

            # Stop when total throughput decreases (adding a process hurt)
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

def bench_adapters(
    adapter_types: list[str],
    batch_size: int,
    device: str,
    do_compile: bool,
    do_eager: bool,
    n_iter: int,
    n_warmup: int,
    sdpa_backend,
) -> list[TimingResult]:
    """Benchmark adapter training steps on a frozen base backbone."""
    import torch
    import torch.nn as nn
    import pawn.model as model_module
    from pawn.config import CLMConfig
    from pawn.model import PAWNCLM

    results = []

    print("\n" + "=" * 72)
    print(" ADAPTER TRAINING BENCHMARKS (GPU)")
    print("=" * 72)
    print(f"  backbone=base  batch_size={batch_size}  device={device}  n_iter={n_iter}")
    print(f"  AMP: bf16  SDPA: {sdpa_backend.name if sdpa_backend else 'default (flash)'}")

    batch = _make_batch(batch_size, device)

    def _build_adapter(name: str, backbone: PAWNCLM) -> nn.Module:
        if name == "lora":
            from pawn.adapters.lora import LoRACLM
            return LoRACLM(backbone, rank=4, attn_targets="qkvo")
        elif name == "film":
            from pawn.adapters.film import FiLMCLM
            return FiLMCLM(backbone, use_output_film=True)
        elif name == "bottleneck":
            from pawn.adapters.bottleneck import BottleneckCLM
            return BottleneckCLM(backbone, bottleneck_dim=8)
        else:
            raise ValueError(f"Unknown adapter: {name}")

    modes = []
    if do_eager:
        modes.append(("eager", False))
    if do_compile:
        modes.append(("compiled", True))

    for adapter_name in adapter_types:
        for mode_name, use_compile in modes:
            label = f"adapter/{adapter_name} [{mode_name}]"
            print(f"\n  {label} ...")

            model_module.SDPA_BACKEND = sdpa_backend

            backbone = PAWNCLM(CLMConfig.base()).to(device)
            adapter_model = _build_adapter(adapter_name, backbone).to(device)

            trainable = [p for p in adapter_model.parameters() if p.requires_grad]
            n_adapter_params = sum(p.numel() for p in trainable)
            n_total_params = sum(p.numel() for p in adapter_model.parameters())
            print(f"    adapter params: {n_adapter_params:,} / {n_total_params:,} total")

            forward_fn = adapter_model.forward
            if use_compile:
                forward_fn = torch.compile(forward_fn)

            optimizer = torch.optim.AdamW(
                trainable, lr=3e-4, weight_decay=0.01,
            )
            scaler = torch.amp.GradScaler(device, enabled=True)

            step = _make_adapter_step(
                adapter_model, optimizer, scaler, batch, device, forward_fn,
                trainable,
            )

            try:
                gpu_timing = time_gpu(step, n_warmup=n_warmup, n_iter=n_iter,
                                      reset_peak_memory=True)
            except torch.cuda.OutOfMemoryError:
                print(f"    OOM — skipping (try smaller --batch-size)")
                del adapter_model, backbone, optimizer, scaler
                torch.cuda.empty_cache()
                continue

            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

            r = make_result(
                label, gpu_timing.times,
                throughput_count=batch_size,
                throughput_unit="samples/s",
                peak_memory_mb=peak_mb,
                warmup_secs=gpu_timing.warmup_secs if use_compile else None,
                n_warmup=gpu_timing.n_warmup if use_compile else None,
            )
            results.append(r)
            print(f"    {r.summary_line()}")

            del adapter_model, backbone, optimizer, scaler
            torch.cuda.empty_cache()

    return results


# ── Report ───────────────────────────────────────────────────────────────────

def print_summary(
    engine_results: list[TimingResult],
    backbone_results: list[TimingResult],
    dataloader_results: list[TimingResult],
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

    if dataloader_results:
        print("\n  Dataloader-inclusive training (GPU):")
        for r in dataloader_results:
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
    dataloader_results: list[TimingResult],
    concurrency_results: list[ConcurrencyResult],
    adapter_results: list[TimingResult],
    platform_info: dict,
):
    report = BenchmarkReport(
        timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
        platform_info=platform_info,
        engine_results=[asdict(r) for r in engine_results],
        backbone_results=[asdict(r) for r in backbone_results],
        dataloader_results=[asdict(r) for r in dataloader_results],
        concurrency_results=[asdict(r) for r in concurrency_results],
        adapter_results=[asdict(r) for r in adapter_results],
    )
    Path(path).write_text(json.dumps(asdict(report), indent=2))
    print(f"Results saved to {path}")


# ── System info collection ───────────────────────────────────────────────────

def _collect_cpu_cache() -> dict[str, str]:
    """Read CPU cache hierarchy from sysfs. Best-effort, Linux only."""
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
            # Check if L3 is shared across all assigned CPUs
            if level == "3":
                try:
                    shared = (idx_dir / "shared_cpu_list").read_text().strip()
                    cache["l3_shared_cpus"] = shared
                except OSError:
                    pass
            cache[f"l{level}"] = size

    return cache


def _collect_system_info() -> dict:
    """Collect CPU, RAM, and cache info."""
    import multiprocessing

    cpu_name = ""
    cpu_mhz = 0.0
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name") and not cpu_name:
                        cpu_name = line.split(":", 1)[1].strip()
                    elif line.startswith("cpu MHz") and not cpu_mhz:
                        cpu_mhz = float(line.split(":", 1)[1].strip())
                    if cpu_name and cpu_mhz:
                        break
        except OSError:
            pass
    if not cpu_name:
        cpu_name = platform.processor() or platform.machine() or "unknown"

    # Max frequency from cpufreq (more reliable than current MHz)
    try:
        max_khz = int(Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
                       .read_text().strip())
        cpu_mhz = max_khz / 1000
    except (OSError, ValueError):
        pass  # keep /proc/cpuinfo MHz if available

    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = multiprocessing.cpu_count() or 0

    # System RAM
    ram_gb = 0.0
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
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
    if cpu_mhz:
        info["cpu_mhz"] = round(cpu_mhz)

    cache = _collect_cpu_cache()
    if cache:
        info["cache"] = cache

    return info


def _collect_gpu_info_amdsmi(info: dict) -> None:
    """Collect AMD GPU clocks and VRAM bandwidth via the amdsmi Python library.

    Available on ROCm 6+. Native Python API — no subprocess needed.
    """
    try:
        import amdsmi  # type: ignore[import-untyped]
        amdsmi.amdsmi_init()
    except (ImportError, Exception):
        return

    try:
        handles = amdsmi.amdsmi_get_processor_handles()
        if not handles:
            return
        gpu = handles[0]

        # Max clocks: SYS (graphics) and MEM
        for clk_type, key in [
            (amdsmi.AmdSmiClkType.SYS, "gpu_clock_mhz"),
            (amdsmi.AmdSmiClkType.MEM, "gpu_mem_clock_mhz"),
        ]:
            try:
                clk = amdsmi.amdsmi_get_clock_info(gpu, clk_type)
                max_clk = clk.get("max_clk") or clk.get("max")
                if max_clk:
                    info[key] = int(max_clk)
            except Exception:
                pass

        # VRAM info (type, bus width, bandwidth)
        try:
            vram = amdsmi.amdsmi_get_gpu_vram_info(gpu)
            vram_type = vram.get("vram_type") or vram.get("type")
            if vram_type:
                info["vram_type"] = str(vram_type)
            vram_width = vram.get("vram_bit_width") or vram.get("bit_width")
            if vram_width:
                info["vram_bus_width"] = int(vram_width)
        except Exception:
            pass

        # PCIe info
        try:
            pcie = amdsmi.amdsmi_get_pcie_info(gpu)
            pcie_info = pcie.get("pcie_static", pcie)
            gen = pcie_info.get("max_pcie_speed") or pcie_info.get("pcie_generation")
            width = pcie_info.get("max_pcie_width") or pcie_info.get("pcie_width")
            if gen and width:
                info["pcie"] = f"Gen{gen} x{width}"
        except Exception:
            pass
    finally:
        try:
            amdsmi.amdsmi_shut_down()
        except Exception:
            pass


def _collect_gpu_info_nvidia_smi(info: dict) -> None:
    """Collect NVIDIA GPU clocks via nvidia-smi."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.max.graphics,clocks.max.mem,pcie.link.gen.max,pcie.link.width.max",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            if len(parts) >= 2:
                info["gpu_clock_mhz"] = int(parts[0])
                info["gpu_mem_clock_mhz"] = int(parts[1])
            if len(parts) >= 4:
                info["pcie"] = f"Gen{parts[2]} x{parts[3]}"
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass


def _collect_gpu_info() -> dict:
    """Collect GPU info from torch and system tools. Best-effort."""
    import subprocess
    import torch

    n_gpus = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    info: dict = {
        "torch": torch.__version__,
        "gpu": props.name,
        "gpu_count": n_gpus,
        "vram_gb": round(props.total_memory / (1024**3), 1),
        "gpu_sm_count": props.multi_processor_count,
    }

    # L2 cache (exposed by torch on some GPUs)
    l2 = getattr(props, "L2_cache_size", 0)
    if l2:
        info["gpu_l2_mb"] = round(l2 / (1024**2), 1)

    # Detect CUDA MPS
    try:
        ps = subprocess.run(
            ["ps", "-eo", "comm"], capture_output=True, text=True, timeout=5,
        )
        if "nvidia-cuda-mps" in ps.stdout:
            info["cuda_mps"] = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try platform-specific tools for clock speeds and VRAM details
    from pawn.gpu import is_rocm
    if is_rocm():
        _collect_gpu_info_amdsmi(info)
    else:
        _collect_gpu_info_nvidia_smi(info)

    return info


def _print_system_info(info: dict) -> None:
    """Print system info header."""
    cpu_line = f"CPU: {info['cpu']} ({info['cpu_count']} CPUs)"
    if info.get("cpu_mhz"):
        cpu_line += f" @ {info['cpu_mhz']} MHz"
    print(cpu_line)

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
            shared = cache.get("l3_shared_cpus", "")
            l3_str = f"L3: {cache['l3']}"
            # If L3 is shared with more CPUs than we're assigned, note it
            if shared:
                try:
                    assigned = len(os.sched_getaffinity(0))
                    # Parse "0-15" style ranges
                    shared_count = sum(
                        int(r.split("-")[1]) - int(r.split("-")[0]) + 1
                        if "-" in r else 1
                        for r in shared.split(",")
                    )
                    if shared_count > assigned:
                        l3_str += " (shared)"
                except (AttributeError, OSError, ValueError):
                    pass
            parts.append(l3_str)
        if parts:
            print(f"Cache: {', '.join(parts)}")


def _print_gpu_info(info: dict, sdpa_backend) -> None:
    """Print GPU info header."""
    n_gpus = info.get("gpu_count", 1)
    gpu_count_str = f" x{n_gpus}" if n_gpus > 1 else ""
    gpu_line = f"GPU: {info['gpu']}{gpu_count_str} ({info['platform']}, {info['vram_gb']:.1f} GB"
    vram_type = info.get("vram_type", "")
    if vram_type:
        gpu_line += f" {vram_type}"
    gpu_line += " VRAM"
    if info.get("gpu_clock_mhz"):
        gpu_line += f", {info['gpu_clock_mhz']} MHz"
    if info.get("gpu_mem_clock_mhz"):
        gpu_line += f", mem {info['gpu_mem_clock_mhz']} MHz"
    gpu_line += ")"
    print(gpu_line)

    extras = []
    if info.get("gpu_sm_count"):
        extras.append(f"{info['gpu_sm_count']} SMs")
    if info.get("gpu_l2_mb"):
        extras.append(f"L2: {info['gpu_l2_mb']} MB")
    if info.get("vram_bus_width"):
        extras.append(f"{info['vram_bus_width']}-bit bus")
    if info.get("pcie"):
        extras.append(f"PCIe {info['pcie']}")
    if info.get("cuda_mps"):
        extras.append("MPS active")
    if extras:
        print(f"      {', '.join(extras)}")
    if n_gpus > 1:
        print(f"      Benchmarking GPU 0 only")

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"SDPA backend: {sdpa_backend.name if sdpa_backend else 'default (flash)'}")
    print(f"AMP dtype: bfloat16")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PAWN performance benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Scope
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--engine-only", action="store_true",
                       help="Only run engine (CPU) benchmarks")
    scope.add_argument("--gpu-only", action="store_true",
                       help="Only run GPU benchmarks (backbone + adapters)")

    # Engine options
    parser.add_argument("--engine-games", type=int, default=10_000,
                        help="Number of games for engine benchmarks (default: 10000)")

    # GPU options
    parser.add_argument("--variants", nargs="+", default=["small", "base"],
                        choices=["toy", "small", "base", "large"],
                        help="Backbone variants to benchmark (default: small base)")
    parser.add_argument("--adapters", nargs="+", default=["lora", "film", "bottleneck"],
                        choices=["lora", "film", "bottleneck"],
                        help="Adapter types to benchmark (default: lora film bottleneck)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for GPU benchmarks (default: 256)")
    parser.add_argument("--no-backbone", action="store_true",
                        help="Skip backbone benchmarks")
    parser.add_argument("--no-adapters", action="store_true",
                        help="Skip adapter benchmarks")
    parser.add_argument("--device", type=str, default="cuda")

    # Concurrency sweep options
    parser.add_argument("--sweep-variant", type=str, default="small",
                        choices=["toy", "small", "base", "large"],
                        help="Backbone variant for concurrency sweep (default: small)")
    parser.add_argument("--sweep-adapter", type=str, default="none",
                        choices=["none", "lora", "film", "bottleneck"],
                        help="Adapter for concurrency sweep (default: none)")

    # Compile modes
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument("--no-compile", action="store_true",
                               help="Only run eager mode (skip torch.compile)")
    compile_group.add_argument("--compile-only", action="store_true",
                               help="Only run compiled mode (skip eager)")

    # Iteration control
    parser.add_argument("--n-iter", type=int, default=10,
                        help="Timed iterations per benchmark (default: 10)")
    parser.add_argument("--n-warmup", type=int, default=3,
                        help="Warmup iterations per benchmark (default: 3)")

    # Output
    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    do_engine = not args.gpu_only
    do_gpu = not args.engine_only
    do_backbone = do_gpu and not args.no_backbone
    do_adapters = do_gpu and not args.no_adapters
    do_compile = not args.no_compile
    do_eager = not args.compile_only

    # Platform info
    info = _collect_system_info()
    _print_system_info(info)

    # Detect GPU platform for SDPA backend selection
    sdpa_backend = None
    do_gpu_any = do_backbone or do_adapters or do_gpu
    if do_gpu_any:
        import torch
        from pawn.gpu import is_rocm

        if not torch.cuda.is_available():
            if do_engine:
                print("No GPU available — running engine benchmarks only.")
                do_backbone = do_adapters = do_gpu = False
            else:
                print("ERROR: No GPU available.", file=sys.stderr)
                sys.exit(1)
        else:
            gpu_info = _collect_gpu_info()
            info.update(gpu_info)
            from torch.nn.attention import SDPBackend

            if is_rocm():
                info["platform"] = "ROCm"
                info["hip"] = torch.version.hip
                sdpa_backend = SDPBackend.MATH
            else:
                info["platform"] = "CUDA"
                info["cuda"] = torch.version.cuda
                sdpa_backend = None

            _print_gpu_info(info, sdpa_backend)

    # Run benchmarks
    engine_results: list[TimingResult] = []
    backbone_results: list[TimingResult] = []
    dataloader_results: list[TimingResult] = []
    concurrency_results: list[ConcurrencyResult] = []
    adapter_results: list[TimingResult] = []

    if do_engine:
        engine_results = bench_engine(
            args.engine_games, args.n_iter, args.n_warmup,
        )

    if do_backbone:
        backbone_results = bench_backbone(
            args.variants, args.batch_size, args.device,
            do_compile, do_eager, args.n_iter, args.n_warmup,
            sdpa_backend,
        )
        dataloader_results = bench_dataloader(
            args.batch_size, args.device, args.n_iter, args.n_warmup,
            sdpa_backend,
        )
    if do_gpu:
        concurrency_results = bench_concurrency(
            args.batch_size, args.device, args.n_iter, args.n_warmup,
            sdpa_backend, variant=args.sweep_variant,
            adapter=args.sweep_adapter,
        )
    if do_adapters:
        adapter_results = bench_adapters(
            args.adapters, args.batch_size, args.device,
            do_compile, do_eager, args.n_iter, args.n_warmup,
            sdpa_backend,
        )

    # Summary
    print_summary(engine_results, backbone_results, dataloader_results,
                  concurrency_results, adapter_results)

    if args.json:
        save_json(
            args.json, engine_results, backbone_results, dataloader_results,
            concurrency_results, adapter_results, info,
        )


if __name__ == "__main__":
    main()
