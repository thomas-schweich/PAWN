"""Frozen v1 PyTorch reference architecture — used **only** by
:mod:`pawn.legacy` parity tests.

This module exists so the parity tests have a known-good v1
implementation to compare against. It is intentionally minimal: just
enough to (a) build a v1-shaped model, (b) save it as v1-format
safetensors, and (c) run a forward pass in fp32 that the v2 JAX path
can match within tolerance.

The module is **never** imported by the v2 surface (pawn.model,
pawn.trainer, pawn.checkpoint, etc.) — those paths are torch-free.
Only `pawn.legacy` (the converter itself) and the parity tests touch
this fixture. Lazy `import torch` inside the functions so module-level
`import pawn._torch_legacy_fixture` is cheap.

The architecture mirrors v1's PAWNCLM at the parameter shapes that
matter: nn.Linear weights as ``(out, in)`` — exactly the v1 storage
convention the converter has to transpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pawn.config import (
    NUM_ACTIONS,
    OUTCOME_TOKEN_BASE,
    PAD_TOKEN,
    VOCAB_SIZE,
)

__all__ = [
    "LegacyConfig",
    "build_legacy_state_dict",
    "save_legacy_checkpoint",
]


@dataclass(frozen=True)
class LegacyConfig:
    """Mirror of v1 CLMConfig at the fields the converter cares about."""

    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = 512


def build_legacy_state_dict(cfg: LegacyConfig, seed: int = 0) -> dict:
    """Build a dictionary of numpy arrays in v1's exact storage shape.

    Linear weights are stored as ``(out, in)`` per the v1 nn.Linear
    convention. The converter transposes these to JAX's
    ``(in, out)``.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    d = cfg.d_model
    d_ff = cfg.d_ff
    L = cfg.n_layers
    V = cfg.vocab_size

    def n(*shape):
        return rng.standard_normal(shape).astype("float32") * 0.02

    state: dict = {
        # Factored embeddings (same orientation as v2: `(rows, d)`)
        "embed.src_embed.weight": n(64, d),
        "embed.dst_embed.weight": n(64, d),
        "embed.promo_embed.weight": n(5, d),
        "embed.pad_embed": np.zeros((d,), dtype="float32"),
        "embed.outcome_embed.weight": n(VOCAB_SIZE - NUM_ACTIONS - 1, d),
        # Final norm + LM head (Linear → (out, in) = (V, d) in v1)
        "final_norm.weight": np.ones((d,), dtype="float32"),
        "lm_head.weight": n(V, d),
    }
    # Per-layer fields — v1 had them in separate ModuleList entries
    # named `layers.<i>.<sublayer>.weight`. The converter recognises
    # this layout, stacks across the layer index, and transposes.
    for i in range(L):
        state[f"layers.{i}.attn_norm.weight"] = np.ones((d,), dtype="float32")
        state[f"layers.{i}.attn.wq.weight"] = n(d, d)  # (out, in)
        state[f"layers.{i}.attn.wk.weight"] = n(d, d)
        state[f"layers.{i}.attn.wv.weight"] = n(d, d)
        state[f"layers.{i}.attn.wo.weight"] = n(d, d)
        state[f"layers.{i}.ffn_norm.weight"] = np.ones((d,), dtype="float32")
        state[f"layers.{i}.ffn.w_gate.weight"] = n(d_ff, d)
        state[f"layers.{i}.ffn.w_up.weight"] = n(d_ff, d)
        state[f"layers.{i}.ffn.w_down.weight"] = n(d, d_ff)
    return state


def save_legacy_checkpoint(
    cfg: LegacyConfig, out_dir: Path, seed: int = 0
) -> Path:
    """Write a v1-shape checkpoint to ``out_dir`` for parity testing.

    Layout:
        ``model.safetensors`` — numpy arrays in v1 storage shape.
        ``config.json`` — v1-style config (vocab_size, dimensions).

    Returns the output directory.
    """
    import json

    from safetensors.numpy import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = build_legacy_state_dict(cfg, seed=seed)
    save_file(state, str(out_dir / "model.safetensors"))
    config = {
        "vocab_size": cfg.vocab_size,
        "max_seq_len": cfg.max_seq_len,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "d_ff": cfg.d_ff,
        "n_outcomes": cfg.vocab_size - NUM_ACTIONS - 1,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return out_dir
