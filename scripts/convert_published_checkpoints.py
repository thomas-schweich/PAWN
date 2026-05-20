"""Convert published PyTorch ``pawn-{small,base,large}`` checkpoints to JAX format.

Phase-1 verification (retroactive: PR #101 merged before the
training-run-on-PR-open policy was in place). The script exercises the legacy
converter on the real published checkpoints — including the ``head_dim=80``
``pawn-large`` path that the toy-config tests don't cover — and produces
JAX-format artifacts under ``$HF_HOME/pawn-jax-converted/<variant>/`` for
later phases to reuse.

For each variant:

  1. ``huggingface_hub.snapshot_download`` (cache-aware; re-runs hit cache).
  2. ``pawn.jax.legacy.convert_legacy_checkpoint`` → JAX checkpoint dir.
  3. Build a PyTorch ``PAWNCLM`` reference from the same state_dict.
  4. Forward on a deterministic batch through both models; report
     ``max |Δlogit|`` and ``mean |Δlogit|``.

Exits non-zero if any variant fails the tolerance check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import TypedDict

import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
import torch
from safetensors.torch import load_file as torch_load_file

from pawn.config import CLMConfig
from pawn.jax.checkpoint import load_model
from pawn.jax.legacy import convert_legacy_checkpoint
from pawn.jax.model import PAWNModel
from pawn.model import PAWNCLM

VARIANTS = ("pawn-small", "pawn-base", "pawn-large")
HF_OWNER = "thomas-schweich"


class _VariantResult(TypedDict):
    """Per-variant outcome row. Typed so ``passed`` is a real bool, the
    numeric fields keep their float/int dtypes through format specs, and
    the summary loop's ``all(r["passed"] for r in results)`` is not
    truthiness-on-``object``."""

    variant: str
    d_model: int
    n_layers: int
    n_heads: int
    head_dim: int
    max_diff: float
    mean_diff: float
    passed: bool
    elapsed_s: float
    error: str | None


def _build_torch_reference(src: Path) -> tuple[PAWNCLM, CLMConfig]:
    """Construct the legacy PyTorch ``PAWNCLM`` from a HF snapshot directory."""
    cfg_dict = json.loads((src / "config.json").read_text())["model_config"]
    fields = CLMConfig.__dataclass_fields__
    cfg = CLMConfig(**{k: v for k, v in cfg_dict.items() if k in fields})
    model = PAWNCLM(cfg).eval()
    sd = torch_load_file(src / "model.safetensors")
    model.load_state_dict(sd)
    return model, cfg


def _forward_parity(
    torch_model: PAWNCLM,
    jax_model: PAWNModel,
    *,
    b: int = 4,
    t: int = 64,
    seed: int = 0,
) -> tuple[float, float]:
    """Run forward on both models with a deterministic batch and return
    ``(max_abs_diff, mean_abs_diff)`` over the logit tensor."""
    rng = np.random.default_rng(seed)
    tokens_np = rng.integers(0, 1968, size=(b, t), dtype=np.int64)
    mask_np = np.ones((b, t), dtype=bool)
    with torch.no_grad():
        torch_logits, _ = torch_model(
            torch.from_numpy(tokens_np), torch.from_numpy(mask_np)
        )
    torch_arr = torch_logits.cpu().numpy()
    jax_logits = jax.jit(lambda m, tk, am: m(tk, am))(
        jax_model, jnp.asarray(tokens_np), jnp.asarray(mask_np)
    )
    jax_arr = np.asarray(jax_logits)
    diff = np.abs(torch_arr - jax_arr)
    return float(diff.max()), float(diff.mean())


def _convert_one(
    variant: str,
    out_root: Path,
    *,
    batch_size: int,
    seq_len: int,
    tol: float,
) -> _VariantResult:
    """Convert + parity-check one variant. Failures are caught here so
    later variants in the loop still run (network blip on pawn-small
    must not skip pawn-base + pawn-large)."""
    repo_id = f"{HF_OWNER}/{variant}"
    dst = out_root / variant
    # A prior run may have left a JAX checkpoint at ``dst``. If this run
    # fails before reaching ``convert_legacy_checkpoint``, the stale one
    # would silently still be there — and downstream tools reading
    # ``$HF_HOME/pawn-jax-converted/<variant>/`` would load it. Snapshot
    # the pre-existing state so the failure handler can flag it.
    dst_preexisted = dst.exists()
    converted_this_run = False  # cleared once convert_legacy_checkpoint returns
    t0 = time.perf_counter()
    print(f"=== {variant} ===")
    try:
        print(f"  snapshot_download({repo_id!r}) ...")
        src = Path(
            huggingface_hub.snapshot_download(
                repo_id=repo_id,
                # The converter only reads these two files. ``allow_patterns``
                # filters which blobs are downloaded into the snapshot folder;
                # the cache key itself is ``(repo_id, commit_hash)``, so a
                # prior full-download of the same commit short-circuits here
                # immediately (the snapshot folder is reused as-is, extra
                # files and all). This is fine — the converter ignores
                # everything except ``model.safetensors`` + ``config.json``.
                allow_patterns=["model.safetensors", "config.json"],
            )
        )
        print(f"    cached at {src}")
        print(f"  convert_legacy_checkpoint -> {dst}")
        convert_legacy_checkpoint(src, dst)
        converted_this_run = True
        print("  build PyTorch reference ...")
        torch_model, torch_cfg = _build_torch_reference(src)
        print(f"  load_model({dst})")
        jax_model = load_model(dst)
        # Bare ``assert`` would be stripped under ``python -O``; use
        # explicit raises so a config mismatch always surfaces with a
        # readable message. Looped because the three checks differ only
        # in field name.
        for field in ("d_model", "n_layers", "n_heads"):
            jv = getattr(jax_model.cfg, field)
            tv = getattr(torch_cfg, field)
            if jv != tv:
                raise ValueError(
                    f"{field} mismatch after conversion: JAX={jv} PyTorch={tv}"
                )
        max_diff, mean_diff = _forward_parity(
            torch_model, jax_model, b=batch_size, t=seq_len
        )
    except Exception as exc:  # noqa: BLE001
        # Resilience: a network blip on one variant should not skip the
        # other two. We catch ``Exception``, not ``BaseException``, so
        # KeyboardInterrupt / SystemExit still propagate.
        elapsed = time.perf_counter() - t0
        error_str = f"{type(exc).__name__}: {exc}"
        # If ``dst`` exists, was there before this run, *and* this run
        # did not overwrite it (the failure happened before
        # ``convert_legacy_checkpoint`` returned), the bytes on disk are
        # stale. Note it in the error so an operator doesn't silently
        # load last-week's conversion thinking it's current.
        if dst_preexisted and not converted_this_run and dst.exists():
            error_str += " [stale dst on disk from prior run]"
        # Route every diagnostic line in this block to stderr so a CI
        # capture that split-tees stdout/stderr keeps the variant
        # context, traceback, and timing on the same stream.
        print(f"  FAILED: {error_str}", file=sys.stderr)
        traceback.print_exc()
        print(f"  elapsed: {elapsed:.1f}s\n", file=sys.stderr)
        return _VariantResult(
            variant=variant,
            d_model=0,
            n_layers=0,
            n_heads=0,
            head_dim=0,
            max_diff=float("nan"),
            mean_diff=float("nan"),
            passed=False,
            elapsed_s=elapsed,
            error=error_str,
        )
    elapsed = time.perf_counter() - t0
    passed = max_diff < tol
    status = "OK" if passed else "FAIL"
    head_dim = torch_cfg.d_model // torch_cfg.n_heads
    print(
        f"  d={torch_cfg.d_model} L={torch_cfg.n_layers} "
        f"heads={torch_cfg.n_heads} head_dim={head_dim}"
    )
    print(
        f"  parity: max |Δ| = {max_diff:.3e}, mean |Δ| = {mean_diff:.3e}, "
        f"tol = {tol:.0e} -> {status}"
    )
    print(f"  elapsed: {elapsed:.1f}s\n")
    return _VariantResult(
        variant=variant,
        d_model=torch_cfg.d_model,
        n_layers=torch_cfg.n_layers,
        n_heads=torch_cfg.n_heads,
        head_dim=head_dim,
        max_diff=max_diff,
        mean_diff=mean_diff,
        passed=passed,
        elapsed_s=elapsed,
        error=None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS)
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    # 1e-3 is the realistic fp32 cross-framework parity bar at 8-10 layer
    # scale: PyTorch's fused scaled_dot_product_attention and our JAX
    # einsum + softmax reduce QK^T in different orders, and the resulting
    # ~1e-7 per-op divergence accumulates through head_dim 64-80, 8-10
    # layers, and the residual stream. Observed mean |Δ| stays ~5e-6 (the
    # representative number); the max is the long tail.
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args(argv)

    hf_home = Path(
        os.environ.get("HF_HOME") or "~/.cache/huggingface"
    ).expanduser()
    out_root = hf_home / "pawn-jax-converted"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"HF cache:       {hf_home}")
    print(f"converted root: {out_root}\n")

    results: list[_VariantResult] = [
        _convert_one(
            variant,
            out_root,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            tol=args.tol,
        )
        for variant in args.variants
    ]

    print("=== summary ===")
    for r in results:
        status = "OK  " if r["passed"] else "FAIL"
        if r["error"] is not None:
            print(f"  [{status}] {r['variant']:<11} ERROR: {r['error']}")
            continue
        print(
            f"  [{status}] {r['variant']:<11} "
            f"d={r['d_model']:>4} L={r['n_layers']:>3} "
            f"heads={r['n_heads']:>3} head_dim={r['head_dim']:>3}  "
            f"max |Δ| = {r['max_diff']:.3e}"
        )

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
