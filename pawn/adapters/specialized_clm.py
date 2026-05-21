"""Specialized (from-scratch) chess language model.

A lightweight transformer trained directly on Lichess games without
a pretrained backbone. Used as a baseline comparison against adapter
strategies — how well can N parameters do trained from scratch vs.
adapting a frozen PAWN backbone?

Unlike the other ``pawn.adapters`` modules, ``specialized_clm`` does
not wrap a backbone — it *is* the model. The "adapter" terminology
is preserved for dispatch parity with the legacy unified
``train.py``'s ``--strategy`` table; in practice this module just
ships:

* ``SpecializedCLMConfig`` — small ``ModelConfig`` factory with the
  legacy defaults (d_model=84, n_layers=2, n_heads=2, d_ff=192).
* ``init_specialized_clm(cfg, key)`` — thin wrapper around
  ``pawn.model.init_model`` so dispatch from the adapter trainer
  can stay uniform.
* ``adapter_filter(model)`` — every backbone leaf is trainable
  (the whole model is "the adapter"). Provided so the two-tier
  partition contract works the same way as for the other strategies.

The architecture mirrors PAWN — RMSNorm, RoPE, SwiGLU FFN, factored
embedding — at a smaller scale. The legacy used GELU; SwiGLU is the
closest direct analogue in the current JAX model. For comparison
runs at strict parity with the legacy GELU baseline, set
``d_ff`` to ``int(2.667 × d_model)`` and accept the SwiGLU /
GELU divergence as a minor architectural change.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax

from pawn.config import ModelConfig
from pawn.model import PAWNModel, init_model


@dataclass(frozen=True)
class SpecializedCLMConfig:
    """Small standalone-CLM hyperparameters.

    Defaults match the legacy small baseline (d_model=84, 2 layers,
    2 heads, d_ff=192). ``head_dim = d_model / n_heads`` must be
    even (RoPE rotates pairs).
    """

    d_model: int = 84
    n_layers: int = 2
    n_heads: int = 2
    d_ff: int = 192
    max_seq_len: int = 512

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
        )


def init_specialized_clm(
    cfg: SpecializedCLMConfig, key: jax.Array,
) -> PAWNModel:
    """Build a fresh ``PAWNModel`` at the specialized-CLM scale."""
    return init_model(cfg.to_model_config(), key)


def adapter_filter(model: PAWNModel) -> PAWNModel:
    """Every array leaf is trainable — the whole model is the
    adapter. Mirrors the two-tier-partition contract the other
    strategies expose, even though there's no separate "backbone"
    to freeze.
    """
    return jax.tree_util.tree_map(eqx.is_inexact_array, model)
