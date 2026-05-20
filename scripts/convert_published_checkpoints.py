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
from pathlib import Path

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
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

    results: list[dict[str, object]] = []
    for variant in args.variants:
        repo_id = f"{HF_OWNER}/{variant}"
        dst = out_root / variant
        t0 = time.perf_counter()
        print(f"=== {variant} ===")
        print(f"  snapshot_download({repo_id!r}) ...")
        src = Path(
            huggingface_hub.snapshot_download(
                repo_id=repo_id,
                # The converter only reads these two files; pulling the whole
                # repo (~200 files / multi-GB of nested checkpoint dirs) is
                # wasteful. snapshot_download caches per-pattern so re-runs
                # never re-fetch.
                allow_patterns=["model.safetensors", "config.json"],
            )
        )
        print(f"    cached at {src}")
        print(f"  convert_legacy_checkpoint -> {dst}")
        convert_legacy_checkpoint(src, dst)
        print("  build PyTorch reference ...")
        torch_model, torch_cfg = _build_torch_reference(src)
        print(f"  load_model({dst})")
        jax_model = load_model(dst)
        assert jax_model.cfg.d_model == torch_cfg.d_model
        assert jax_model.cfg.n_layers == torch_cfg.n_layers
        assert jax_model.cfg.n_heads == torch_cfg.n_heads
        max_diff, mean_diff = _forward_parity(
            torch_model, jax_model, b=args.batch_size, t=args.seq_len
        )
        elapsed = time.perf_counter() - t0
        passed = max_diff < args.tol
        status = "OK" if passed else "FAIL"
        head_dim = torch_cfg.d_model // torch_cfg.n_heads
        print(
            f"  d={torch_cfg.d_model} L={torch_cfg.n_layers} "
            f"heads={torch_cfg.n_heads} head_dim={head_dim}"
        )
        print(
            f"  parity: max |Δ| = {max_diff:.3e}, mean |Δ| = {mean_diff:.3e}, "
            f"tol = {args.tol:.0e} -> {status}"
        )
        print(f"  elapsed: {elapsed:.1f}s\n")
        results.append(
            {
                "variant": variant,
                "d_model": torch_cfg.d_model,
                "n_layers": torch_cfg.n_layers,
                "n_heads": torch_cfg.n_heads,
                "head_dim": head_dim,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "passed": passed,
                "elapsed_s": elapsed,
            }
        )

    print("=== summary ===")
    for r in results:
        status = "OK  " if r["passed"] else "FAIL"
        print(
            f"  [{status}] {r['variant']!s:<11} "
            f"d={r['d_model']!s:>4} L={r['n_layers']!s:>3} "
            f"heads={r['n_heads']!s:>3} head_dim={r['head_dim']!s:>3}  "
            f"max |Δ| = {r['max_diff']:.3e}"  # type: ignore[str-format]
        )

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
