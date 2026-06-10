#!/usr/bin/env python3
"""A.1 stage 1 — empirical cost(T) curve at SUPERNET / TINY.

Measures inner-step time at T ∈ {128, 256, 384, 512} (configurable),
fits cost(T) = a*T + b*T^2 via least squares, prints a table + the
fit coefficients. The fit then feeds into bucket_search.py to decide
the optimal K=2/3/4 edges.

Why this exists separately from bench/run.py: the steady-state matrix
sweeps shape combinations (variant, batch, mode). This script
specifically sweeps T at fixed (variant, batch, mode) to characterize
the per-T cost function used by bucket_search.

Output:
    bench/results/cost_curve-<sha>-<ts>.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from pawn.config import SUPERNET, TINY_SUPERNET, VARIANTS, TINY_VARIANTS
from pawn.corpus import generate_corpus
from pawn.jax_setup import setup_jax_caching
from pawn.model import init_model
from pawn.run_config import PretrainConfig
from pawn.trainer import (
    Batch,
    TrainState,
    VariantSpec,
    make_lr_schedule,
    make_optimizer,
    make_scan_step,
    make_train_step,
)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "no-git"


def _bench_one_T(
    *, supernet_name: str, batch_size: int, seq_len: int, k: int,
    warmup_outers: int, timed_outers: int,
    n_variants_mode: str,
) -> dict:
    cfg = TINY_SUPERNET if supernet_name == "tiny" else SUPERNET
    variants_dict = TINY_VARIANTS if supernet_name == "tiny" else VARIANTS
    if n_variants_mode == "1v":
        variants = (VariantSpec("large", variants_dict["large"], is_supernet=True),)
        stochastic = False
    elif n_variants_mode == "3v-stoch":
        variants = tuple(
            VariantSpec(name, variants_dict[name], is_supernet=(name == "large"))
            for name in ("small", "base", "large")
        )
        stochastic = True
    else:
        raise ValueError(n_variants_mode)

    fake_cfg = PretrainConfig(
        run_type="pretrain", local_checkpoints=True,
        total_steps=max(1, timed_outers * k), batch_size=batch_size,
        seq_len=seq_len, k=k,
        supernet=supernet_name,  # type: ignore[arg-type]
        variant="large", stochastic_variants=stochastic,
    )
    schedule = make_lr_schedule(fake_cfg, fake_cfg.total_steps or 1)
    optimizer = make_optimizer(fake_cfg, schedule)
    model = init_model(cfg, key=0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    state = TrainState(model=model, opt_state=opt_state, step=jnp.int32(0),
                       key=jax.random.key(0))
    train_step = make_train_step(
        optimizer, variants,
        compute_dtype=jnp.bfloat16, use_sdpa=False,
        use_flash=(jax.default_backend() == "gpu"),
        stochastic_variants=stochastic,
    )
    scan_step = make_scan_step(train_step)

    corpus = generate_corpus(
        n_games=batch_size * k, max_ply=seq_len, seq_len=seq_len, seed=42,
    )
    tokens = corpus.tokens.reshape(k, batch_size, seq_len)
    targets = corpus.targets.reshape(k, batch_size, seq_len)
    attn = corpus.attn_mask.reshape(k, batch_size, seq_len)
    lmask = corpus.loss_mask.reshape(k, batch_size, seq_len)
    batch = Batch(
        tokens=jnp.asarray(tokens), targets=jnp.asarray(targets),
        attn_mask=jnp.asarray(attn), loss_mask=jnp.asarray(lmask),
    )
    for _ in range(warmup_outers):
        state, losses = scan_step(state, batch)
        losses.block_until_ready()
    per_outer = []
    for _ in range(timed_outers):
        t0 = time.perf_counter()
        state, losses = scan_step(state, batch)
        losses.block_until_ready()
        per_outer.append((time.perf_counter() - t0) * 1000.0)
    per_inner = [x / k for x in per_outer]
    return {
        "seq_len": seq_len,
        "ms_per_inner_step_median": statistics.median(per_inner),
        "ms_per_inner_step_p95": float(np.percentile(per_inner, 95)),
        "ms_per_inner_step_mean": statistics.mean(per_inner),
        "samples_per_second": batch_size * 1000.0 / statistics.median(per_inner),
    }


def _fit_quadratic(seq_lens: list[int], ms_per_step: list[float]) -> tuple[float, float]:
    """Fit ms(T) = a*T + b*T^2 by least squares."""
    T = np.asarray(seq_lens, dtype=np.float64)
    Y = np.asarray(ms_per_step, dtype=np.float64)
    # Solve [a, b] @ [T, T^2] = Y
    X = np.stack([T, T**2], axis=1)  # (N, 2)
    coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cost_curve")
    ap.add_argument("--supernet", choices=("tiny", "production"), default="production")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--warmup-outers", type=int, default=2)
    ap.add_argument("--timed-outers", type=int, default=20)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 384, 512])
    ap.add_argument("--n-variants-mode", choices=("1v", "3v-stoch"), default="3v-stoch")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    setup_jax_caching()
    results = []
    for T in args.seq_lens:
        print(f"==> T={T}", flush=True)
        rec = _bench_one_T(
            supernet_name=args.supernet, batch_size=args.batch_size,
            seq_len=T, k=args.k, warmup_outers=args.warmup_outers,
            timed_outers=args.timed_outers, n_variants_mode=args.n_variants_mode,
        )
        results.append(rec)
        print(f"    {rec['ms_per_inner_step_median']:.2f} ms/step  "
              f"{rec['samples_per_second']:.1f} sam/s", flush=True)
    seq_lens = [r["seq_len"] for r in results]
    medians = [r["ms_per_inner_step_median"] for r in results]
    a, b = _fit_quadratic(seq_lens, medians)
    print(f"\nFit: cost(T) = {a:.4e} * T + {b:.4e} * T^2  (ms/step)")

    payload = {
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "label": args.label,
        "supernet": args.supernet, "batch_size": args.batch_size,
        "k": args.k, "n_variants_mode": args.n_variants_mode,
        "device": str(jax.devices()[0]) if jax.devices() else "unknown",
        "fit_coefficients": {"a": a, "b": b},
        "results": results,
    }
    if args.out is None:
        out_dir = Path(__file__).resolve().parents[2] / "bench" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"cost_curve-{payload['git_sha']}-{payload['timestamp_utc']}.json"
    else:
        out = args.out
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
