#!/usr/bin/env python3
"""C.5 spike — measure how often gradient clipping triggers in real
pretraining. The hypothesis is that the branchless clip is pure
overhead once warmup completes. If clip almost never triggers, gating
to warmup-only (or removing entirely) cuts ~0.2-0.3 ms/step at LARGE.

How to use:
    # 1. Run a short pretraining job emitting grad_norm per step:
    uv run --extra rocm python scripts/train_jax.py --supernet tiny \
        --total-steps 2000 --batch-size 8 --seq-len 128 --k 50 \
        --log-interval 1 --local-checkpoints \
        --logs-dir logs/clip-trigger \
        --emit-grad-norms

    # 2. Analyse the resulting metrics.jsonl:
    uv run --extra rocm python scripts/bench/clip_trigger.py \
        logs/clip-trigger/<slug>/metrics.jsonl

Output:
    Step distribution of pre-clip grad norm, fraction of steps where
    grad_norm > max_grad_norm (=1.0 default), breakdown by phase
    (first 5% warmup vs rest), suggested action.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clip_trigger")
    ap.add_argument("path", type=Path, help="metrics.jsonl path")
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="clip threshold (matches BaseRunConfig.max_grad_norm)")
    ap.add_argument("--warmup-fraction", type=float, default=0.05,
                    help="fraction of total steps to call 'warmup' for breakdown")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    rows: list[dict] = []
    for line in args.path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("type") != "train" or "grad_norm" not in d:
            continue
        rows.append(d)
    if not rows:
        print("No train rows with grad_norm found.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["step"])
    last_step = rows[-1]["step"]
    warmup_cutoff = last_step * args.warmup_fraction

    g_norms = [r["grad_norm"] for r in rows]
    median = statistics.median(g_norms)
    mean = statistics.mean(g_norms)
    p95 = sorted(g_norms)[int(len(g_norms) * 0.95)]
    p99 = sorted(g_norms)[int(len(g_norms) * 0.99)]
    max_n = max(g_norms)
    min_n = min(g_norms)

    triggered = sum(1 for g in g_norms if g > args.max_grad_norm)
    trigger_rate = triggered / len(g_norms)

    warmup_rows = [r for r in rows if r["step"] < warmup_cutoff]
    steady_rows = [r for r in rows if r["step"] >= warmup_cutoff]
    warmup_trigger = (
        sum(1 for r in warmup_rows if r["grad_norm"] > args.max_grad_norm)
        / max(1, len(warmup_rows))
    )
    steady_trigger = (
        sum(1 for r in steady_rows if r["grad_norm"] > args.max_grad_norm)
        / max(1, len(steady_rows))
    )

    print(f"=== Clip-trigger analysis ({args.path}) ===")
    print(f"Steps emitted: {len(rows)} (last step {last_step}, "
          f"warmup cutoff {warmup_cutoff:.0f})")
    print(f"max_grad_norm threshold: {args.max_grad_norm}")
    print(f"")
    print(f"grad_norm stats:")
    print(f"  min={min_n:.4f}  median={median:.4f}  mean={mean:.4f}")
    print(f"  p95={p95:.4f}  p99={p99:.4f}  max={max_n:.4f}")
    print(f"")
    print(f"Clip triggers:")
    print(f"  Overall: {triggered}/{len(g_norms)} = {trigger_rate:.2%}")
    print(f"  Warmup ({len(warmup_rows)} steps): {warmup_trigger:.2%}")
    print(f"  Steady ({len(steady_rows)} steps): {steady_trigger:.2%}")
    print(f"")
    if steady_trigger < 0.01:
        print("VERDICT: clip almost never triggers post-warmup. Gating "
              "to warmup-only saves ~0.2-0.3 ms/step at LARGE for free.")
    elif steady_trigger < 0.05:
        print("VERDICT: clip occasionally triggers post-warmup. Worth "
              "doing the gate; the rare triggers don't justify always-on.")
    elif steady_trigger < 0.20:
        print("VERDICT: clip triggers regularly post-warmup. Keep it on.")
    else:
        print("VERDICT: clip triggers VERY often — investigate the LR "
              "schedule / model init / batch size.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
