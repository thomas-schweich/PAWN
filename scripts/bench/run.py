#!/usr/bin/env python3
"""Steady-state pretraining bench harness — JSON output for perf tracking.

Replaces the throwaway ``/tmp/bench_*.py`` scripts that backed perf rounds
1-4. Every commit on the perf branch re-runs this and the JSON diff
against the prior commit is the perf claim.

What it measures:
    Per-shape: ``scan_step(state, batch)`` time over ``--outer-steps``
    outer × ``--k`` inner-step amortisation. After a ``--warmup-outers``
    warmup the steady-state median + p95 ms/inner-step is reported.

Why ``scan_step`` and not ``train_step``:
    Production runs amortise host overhead via ``make_scan_step(K)``;
    benching single ``train_step`` invocations pays per-call
    ``eqx.filter_jit`` block_until_ready overhead that production never
    pays. Round-1 Opus #2 flagged this and the throwaway benches never
    addressed it.

Default sweep matrix (override with --shapes-json):
    BASE/LARGE × B={64, 128, 256} × {1-variant, 3-det, 3-stochastic}

Output:
    ``bench/results/<short-sha>-<UTC-ts>.json`` — one record per shape
    with ms_per_step (median + p95), samples_per_second, peak memory
    (if available), GPU info, JAX version, git sha. ``--out`` overrides
    the default path.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from pawn.config import (
    ModelConfig,
    SUPERNET,
    TINY_SUPERNET,
    TINY_VARIANTS,
    VARIANTS,
)
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


@dataclass
class ShapeSpec:
    """One row of the bench matrix."""

    name: str  # "BASE-B64-1v" etc.
    supernet: str  # "tiny" | "production"
    variant: str  # "small" | "base" | "large"
    batch_size: int
    seq_len: int
    n_variants_mode: str  # "1v" | "3v-det" | "3v-stoch"


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


def _device_name() -> str:
    devs = jax.devices()
    if not devs:
        return "unknown"
    return str(devs[0])


def _peak_memory_mb() -> float | None:
    """Best-effort peak GPU memory (MB) via jax.live_arrays() summation.

    Not a hard guarantee — returns None on CPU and on backends where
    `memory_stats` is not available.
    """
    try:
        dev = jax.devices()[0]
        stats = dev.memory_stats()
    except Exception:
        return None
    if stats is None:
        return None
    val = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use")
    if val is None:
        return None
    return float(val) / (1024 * 1024)


def _supernet_and_variants(supernet_name: str) -> tuple[ModelConfig, dict[str, ModelConfig]]:
    if supernet_name == "tiny":
        return TINY_SUPERNET, dict(TINY_VARIANTS)
    return SUPERNET, dict(VARIANTS)


def _build_variants_for_mode(
    variants_dict: dict[str, ModelConfig],
    variant_focus: str,
    mode: str,
) -> tuple[tuple[VariantSpec, ...], bool]:
    """Return the (VariantSpec tuple, stochastic_variants flag) for a mode.

    - ``1v``: single variant matching ``variant_focus``; that variant
      *is* the supernet (so the joint loss is just one CE pass).
    - ``3v-det``: full 3-variant exhaustive sum.
    - ``3v-stoch``: full 3-variant set with sandwich sampling.
    """
    if mode == "1v":
        cfg = variants_dict[variant_focus]
        return (VariantSpec(variant_focus, cfg, is_supernet=True),), False
    triples = tuple(
        VariantSpec(name, variants_dict[name], is_supernet=(name == "large"))
        for name in ("small", "base", "large")
    )
    if mode == "3v-det":
        return triples, False
    if mode == "3v-stoch":
        return triples, True
    raise ValueError(f"unknown variant mode {mode!r}")


def _default_matrix() -> list[ShapeSpec]:
    rows: list[ShapeSpec] = []
    for variant in ("base", "large"):
        for B in (64, 128, 256):
            for mode in ("1v", "3v-det", "3v-stoch"):
                rows.append(ShapeSpec(
                    name=f"{variant.upper()}-B{B}-{mode}",
                    supernet="production",
                    variant=variant,
                    batch_size=B,
                    seq_len=512,
                    n_variants_mode=mode,
                ))
    return rows


def _tiny_matrix() -> list[ShapeSpec]:
    rows: list[ShapeSpec] = []
    for variant in ("base", "large"):
        for B in (8, 16):
            for mode in ("1v", "3v-stoch"):
                rows.append(ShapeSpec(
                    name=f"TINY-{variant}-B{B}-{mode}",
                    supernet="tiny",
                    variant=variant,
                    batch_size=B,
                    seq_len=128,
                    n_variants_mode=mode,
                ))
    return rows


def _build_state(
    supernet_cfg: ModelConfig, optimizer: Any
) -> TrainState:
    """Initialise a fresh train state. Optimizer.init on filtered params."""
    model = init_model(supernet_cfg, key=0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    return TrainState(
        model=model,
        opt_state=opt_state,
        step=jnp.int32(0),
        key=jax.random.key(0),
    )


def _make_chunk_batch(
    batch_size: int, seq_len: int, k: int, seed: int
) -> Batch:
    """Build one (K, B, T)-shaped Batch from random games."""
    chunk = generate_corpus(
        n_games=batch_size * k,
        max_ply=seq_len,
        seq_len=seq_len,
        seed=seed,
    )
    tokens = chunk.tokens.reshape(k, batch_size, seq_len)
    targets = chunk.targets.reshape(k, batch_size, seq_len)
    attn = chunk.attn_mask.reshape(k, batch_size, seq_len)
    lmask = chunk.loss_mask.reshape(k, batch_size, seq_len)
    return Batch(
        tokens=jnp.asarray(tokens),
        targets=jnp.asarray(targets),
        attn_mask=jnp.asarray(attn),
        loss_mask=jnp.asarray(lmask),
    )


def _bench_one_shape(
    shape: ShapeSpec,
    k: int,
    warmup_outers: int,
    timed_outers: int,
    compute_dtype: jnp.dtype | None,
    use_flash: bool,
) -> dict[str, Any]:
    supernet_cfg, variants_dict = _supernet_and_variants(shape.supernet)
    variants, stochastic = _build_variants_for_mode(
        variants_dict, shape.variant, shape.n_variants_mode
    )

    # Use a synthetic LR schedule matching the production shape (warmup +
    # cosine), but capped at a tiny step budget. The schedule only
    # affects the optimizer init and per-step LR scalar; it does not
    # alter step time.
    fake_cfg = PretrainConfig(
        run_type="pretrain",
        local_checkpoints=True,
        total_steps=max(1, timed_outers * k),
        batch_size=shape.batch_size,
        seq_len=shape.seq_len,
        k=k,
        supernet=shape.supernet,  # type: ignore[arg-type]
        variant=shape.variant,  # type: ignore[arg-type]
        stochastic_variants=stochastic,
    )
    schedule = make_lr_schedule(fake_cfg, fake_cfg.total_steps or 1)
    optimizer = make_optimizer(fake_cfg, schedule)
    state = _build_state(supernet_cfg, optimizer)
    train_step = make_train_step(
        optimizer, variants,
        compute_dtype=compute_dtype,
        use_sdpa=False,
        use_flash=use_flash,
        stochastic_variants=stochastic,
    )
    scan_step = make_scan_step(train_step)

    # One batch we reuse across all outers — avoids host-side corpus-gen
    # overhead masking the kernel time. The model output isn't read so
    # repeating the same data is fine for perf measurement.
    batch = _make_chunk_batch(shape.batch_size, shape.seq_len, k, seed=42)

    # Warmup
    for _ in range(warmup_outers):
        state, losses = scan_step(state, batch)
        losses.block_until_ready()

    # Reset peak-memory counter after warmup so we measure steady-state.
    try:
        dev = jax.devices()[0]
        if hasattr(dev, "memory_stats"):
            # No public API to clear peak; we just snapshot below and
            # report it as warmup-inclusive peak. That's still useful
            # for headroom; the JAX team has signaled an explicit
            # `reset_memory_stats()` is unlikely.
            pass
    except Exception:
        pass

    # Timed
    per_outer_ms: list[float] = []
    for _ in range(timed_outers):
        t0 = time.perf_counter()
        state, losses = scan_step(state, batch)
        losses.block_until_ready()
        per_outer_ms.append((time.perf_counter() - t0) * 1000.0)

    per_inner_ms = [x / k for x in per_outer_ms]
    median = statistics.median(per_inner_ms)
    p95 = float(np.percentile(per_inner_ms, 95))
    mean = statistics.mean(per_inner_ms)
    stdev = statistics.pstdev(per_inner_ms) if len(per_inner_ms) > 1 else 0.0
    samples_per_s = (shape.batch_size * 1000.0) / median if median > 0 else 0.0

    return {
        "name": shape.name,
        "shape": asdict(shape),
        "n_variants_evaluated": 1 if shape.n_variants_mode == "1v" else (2 if stochastic else 3),
        "stochastic_variants": stochastic,
        "ms_per_inner_step_median": median,
        "ms_per_inner_step_p95": p95,
        "ms_per_inner_step_mean": mean,
        "ms_per_inner_step_stdev": stdev,
        "samples_per_second": samples_per_s,
        "peak_memory_mb": _peak_memory_mb(),
        "k": k,
        "warmup_outers": warmup_outers,
        "timed_outers": timed_outers,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench")
    ap.add_argument("--k", type=int, default=50,
                    help="inner-scan steps per outer iteration (production K)")
    ap.add_argument("--warmup-outers", type=int, default=2,
                    help="outers discarded before timing (covers compile + 1 steady step)")
    ap.add_argument("--timed-outers", type=int, default=50,
                    help="outers timed for steady-state median/p95 (k×timed_outers inner steps)")
    ap.add_argument("--tiny", action="store_true",
                    help="run the tiny matrix instead of production")
    ap.add_argument("--shapes-json", type=Path, default=None,
                    help="optional JSON list of ShapeSpec dicts overriding the default matrix")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON path (default: bench/results/<sha>-<ts>.json)")
    ap.add_argument("--label", default=None,
                    help="optional tag baked into the JSON record (e.g. 'pre-A4')")
    ap.add_argument("--compute-dtype", choices=("bfloat16", "float16", "float32"),
                    default="bfloat16")
    ap.add_argument("--no-flash", action="store_true",
                    help="disable Pallas flash (force plain QK^T attention)")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    if args.compute_dtype == "float32":
        compute_dtype = None
    else:
        compute_dtype = jnp.bfloat16 if args.compute_dtype == "bfloat16" else jnp.float16
    use_flash = (not args.no_flash) and jax.default_backend() == "gpu"

    if args.shapes_json is not None:
        raw = json.loads(args.shapes_json.read_text())
        shapes = [ShapeSpec(**row) for row in raw]
    elif args.tiny:
        shapes = _tiny_matrix()
    else:
        shapes = _default_matrix()

    setup_jax_caching()

    results: list[dict[str, Any]] = []
    for shape in shapes:
        print(f"==> {shape.name}", flush=True)
        try:
            rec = _bench_one_shape(
                shape, k=args.k,
                warmup_outers=args.warmup_outers,
                timed_outers=args.timed_outers,
                compute_dtype=compute_dtype, use_flash=use_flash,
            )
            results.append(rec)
            print(
                f"    {rec['ms_per_inner_step_median']:.2f} ms/step  "
                f"{rec['samples_per_second']:.1f} sam/s  "
                f"peak {rec.get('peak_memory_mb') or 'n/a'}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"    FAILED: {e}", flush=True)
            results.append({"name": shape.name, "shape": asdict(shape), "error": str(e)})

    sha = _git_sha()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "git_sha": sha,
        "timestamp_utc": ts,
        "label": args.label,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "device": _device_name(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "k": args.k,
        "warmup_outers": args.warmup_outers,
        "timed_outers": args.timed_outers,
        "compute_dtype": args.compute_dtype,
        "use_flash": use_flash,
        "results": results,
    }

    if args.out is None:
        out_dir = Path(__file__).resolve().parents[2] / "bench" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{sha}-{ts}.json"
    else:
        out = args.out
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
