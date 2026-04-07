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
class BenchmarkReport:
    timestamp: str = ""
    platform_info: dict = field(default_factory=dict)
    engine_results: list[dict] = field(default_factory=list)
    backbone_results: list[dict] = field(default_factory=list)
    dataloader_results: list[dict] = field(default_factory=list)
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
    adapter_results: list[TimingResult],
    platform_info: dict,
):
    report = BenchmarkReport(
        timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
        platform_info=platform_info,
        engine_results=[asdict(r) for r in engine_results],
        backbone_results=[asdict(r) for r in backbone_results],
        dataloader_results=[asdict(r) for r in dataloader_results],
        adapter_results=[asdict(r) for r in adapter_results],
    )
    Path(path).write_text(json.dumps(asdict(report), indent=2))
    print(f"Results saved to {path}")


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
    import multiprocessing
    cpu_name = ""
    # platform.processor() is unreliable on Linux (often empty or just arch);
    # read /proc/cpuinfo for the actual model name
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
    # os.sched_getaffinity respects cgroup cpuset limits (e.g. RunPod vCPUs),
    # unlike multiprocessing.cpu_count() which reports the host's total.
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        # sched_getaffinity is Linux-only; fall back on other platforms
        cpu_count = multiprocessing.cpu_count() or 0

    # System RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # psutil not available — fall back to /proc/meminfo on Linux
        ram_gb = 0.0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram_kb = int(line.split()[1])
                        ram_gb = ram_kb / (1024**2)
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

    print(f"CPU: {cpu_name} ({cpu_count} CPUs)")
    if ram_gb:
        print(f"RAM: {ram_gb:.1f} GB")

    # Detect GPU platform for SDPA backend selection
    sdpa_backend = None
    if do_backbone or do_adapters:
        import torch
        from pawn.gpu import is_rocm

        info["torch"] = torch.__version__

        if not torch.cuda.is_available():
            if do_engine:
                print("No GPU available — running engine benchmarks only.")
                do_backbone = do_adapters = False
            else:
                print("ERROR: No GPU available.", file=sys.stderr)
                sys.exit(1)
        else:
            info["gpu"] = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            info["vram_gb"] = round(vram_gb, 1)
            from torch.nn.attention import SDPBackend

            if is_rocm():
                info["platform"] = "ROCm"
                info["hip"] = torch.version.hip
                # ROCm: MATH backend avoids flash attn backward bug with compile
                sdpa_backend = SDPBackend.MATH
            else:
                info["platform"] = "CUDA"
                info["cuda"] = torch.version.cuda
                # NVIDIA: default SDPA (flash attention)
                sdpa_backend = None

            print(f"GPU: {info['gpu']} ({info['platform']}, {vram_gb:.1f} GB)")
            print(f"PyTorch: {torch.__version__}")
            print(f"SDPA backend: {sdpa_backend.name if sdpa_backend else 'default (flash)'}")
            print(f"AMP dtype: bfloat16")

    # Run benchmarks
    engine_results: list[TimingResult] = []
    backbone_results: list[TimingResult] = []
    dataloader_results: list[TimingResult] = []
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
    if do_adapters:
        adapter_results = bench_adapters(
            args.adapters, args.batch_size, args.device,
            do_compile, do_eager, args.n_iter, args.n_warmup,
            sdpa_backend,
        )

    # Summary
    print_summary(engine_results, backbone_results, dataloader_results,
                  adapter_results)

    if args.json:
        save_json(
            args.json, engine_results, backbone_results, dataloader_results,
            adapter_results, info,
        )


if __name__ == "__main__":
    main()
