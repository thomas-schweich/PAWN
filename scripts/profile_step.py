#!/usr/bin/env python3
"""Profile a single training iteration of PAWN.

Runs cProfile on the full train_step + optimizer_step pipeline,
prints comparative stats, and identifies bottlenecks.

Usage:
    uv run python scripts/profile_step.py [--device cuda] [--batch-size 256]
"""

import argparse
import cProfile
import pstats
import time
import io
import torch

# ── PAWN imports ─────────────────────────────────────────────────────────────
from pawn.config import CLMConfig
from pawn.model import PAWNCLM
from pawn.data import _to_clm_batch
import chess_engine as engine


def generate_clm_batch(batch_size: int, device: str):
    """Generate a CLM batch and move to device."""
    move_ids, game_lengths, term_codes = engine.generate_random_games(
        batch_size, 255, seed=42
    )
    batch = _to_clm_batch(move_ids, game_lengths, term_codes, 256)
    return {k: v.to(device) for k, v in batch.items()}



def clm_iteration(model, optimizer, scaler, batch, device, use_amp):
    """One full CLM train step: forward + loss + backward + optimizer."""
    model.train()
    with torch.amp.autocast(device, enabled=use_amp):
        loss, _metrics = model.forward_train(
            batch["input_ids"], batch["loss_mask"], batch["targets"]
        )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()



def profile_fn(fn, n_warmup=3, n_profile=10):
    """Run warmup iterations, then profile n_profile iterations.

    Returns (cProfile.Profile, wall_times_list).
    """
    # Warmup (let CUDA kernels JIT, allocator settle)
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs (wall clock)
    wall_times = []
    for _ in range(n_profile):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_times.append(time.perf_counter() - t0)

    # cProfile run (single iteration for call graph)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prof = cProfile.Profile()
    prof.enable()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prof.disable()

    return prof, wall_times


def print_profile(name: str, prof: cProfile.Profile, wall_times: list[float],
                  top_n: int = 25):
    """Print profile summary."""
    mean_ms = sum(wall_times) / len(wall_times) * 1000
    min_ms = min(wall_times) * 1000
    max_ms = max(wall_times) * 1000

    print(f"\n{'='*70}")
    print(f" {name}")
    print(f"{'='*70}")
    print(f" Wall time: {mean_ms:.1f}ms mean | {min_ms:.1f}ms min | {max_ms:.1f}ms max "
          f"({len(wall_times)} runs)")
    print(f"{'─'*70}")

    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s)
    ps.sort_stats("cumulative")
    ps.print_stats(top_n)
    print(s.getvalue())


def main():
    parser = argparse.ArgumentParser(description="Profile PAWN training iteration")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    device = args.device
    bs = args.batch_size
    use_amp = not args.no_amp

    print(f"Device: {device}")
    print(f"Batch size: {bs}")
    print(f"AMP: {use_amp}")
    print(f"Warmup: {args.warmup} | Profile: {args.iterations} iterations")

    # ── Profile PAWN ─────────────────────────────────────────────────────────
    print("\nGenerating data...")
    batch = generate_clm_batch(bs, device)

    print("Setting up PAWN model...")
    cfg = CLMConfig()
    model = PAWNCLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("Profiling...")
    prof, times = profile_fn(
        lambda: clm_iteration(model, opt, scaler, batch, device, use_amp),
        n_warmup=args.warmup,
        n_profile=args.iterations,
    )

    peak_mb = 0
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

    # ── Print results ─────────────────────────────────────────────────────────
    print_profile("PAWN (next-token softmax)", prof, times)

    mean_ms = sum(times) / len(times) * 1000
    print(f"\n{'='*70}")
    print(f" SUMMARY")
    print(f"{'='*70}")
    print(f" Parameters:          {n_params:,}")
    print(f" Mean step time (ms): {mean_ms:.1f}")
    print(f" Min step time (ms):  {min(times)*1000:.1f}")
    print(f" Peak GPU memory (MB):{peak_mb:.0f}")
    print(f" Throughput (games/s): {bs/mean_ms*1000:.0f}")
    print()


if __name__ == "__main__":
    main()
