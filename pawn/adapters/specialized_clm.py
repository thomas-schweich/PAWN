"""Specialized CLM — from-scratch standalone transformer adapter.

A small from-scratch :class:`pawn.model.PAWNModel` (no backbone). The
"adapter" is the model itself; the backbone parameter is ignored at
apply time.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax

from pawn.config import ModelConfig
from pawn.model import PAWNModel, init_model

__all__ = [
    "SpecializedCLMConfig",
    "SpecializedCLMAdapter",
    "init_specialized_clm_adapter",
    "apply_specialized_clm",
    "specialized_clm_filter",
]


@dataclass(frozen=True)
class SpecializedCLMConfig:
    """Bare arch dims per plan §10 S3 (no ``specialized_`` prefix)."""

    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int


class SpecializedCLMAdapter(eqx.Module):
    """The from-scratch model. Everything is trainable."""

    model: PAWNModel
    cfg: SpecializedCLMConfig = eqx.field(static=True)


def init_specialized_clm_adapter(
    backbone: PAWNModel, cfg: SpecializedCLMConfig, key: jax.Array | int
) -> SpecializedCLMAdapter:
    """Build a fresh small model. ``backbone`` is unused — this is
    standalone — but takes the same arg shape as the other adapters
    for unified dispatch."""
    del backbone
    if isinstance(key, int):
        key = jax.random.key(key)
    # Inherit vocab + seq settings from the canonical ModelConfig
    # defaults (they default to NUM_ACTIONS+PAD+outcomes and
    # MAX_SEQ_LEN respectively).
    model_cfg = ModelConfig(
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        head_dim=cfg.d_model // cfg.n_heads,
    )
    model = init_model(model_cfg, key)
    return SpecializedCLMAdapter(model=model, cfg=cfg)


def apply_specialized_clm(
    backbone: PAWNModel, adapter: SpecializedCLMAdapter
) -> PAWNModel:
    """Return the adapter's standalone model; backbone is ignored."""
    del backbone
    return adapter.model


def specialized_clm_filter(adapter: SpecializedCLMAdapter) -> SpecializedCLMAdapter:
    return jax.tree_util.tree_map(
        lambda leaf: True if eqx.is_inexact_array(leaf) else False, adapter
    )
