"""Unfreeze adapter — fine-tune explicit layer picks (v1 ``"5,6,7"`` form).

The adapter holds a per-layer boolean mask; the trainer's
gradient-mask hook zeros gradients for layers not in the unfreeze set
(this is the only strategy that needs a gradient mask vs the
partition alone, per plan §10 S7).
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool

from pawn.model import PAWNModel

__all__ = [
    "UnfreezeConfig",
    "UnfreezeAdapter",
    "init_unfreeze_adapter",
    "apply_unfreeze",
    "unfreeze_filter",
    "parse_unfreeze_layers",
]


@dataclass(frozen=True)
class UnfreezeConfig:
    """``layers`` is the explicit-pick string (e.g. ``"5,6,7"``) — the
    v1 contract per plan §10 S3."""

    layers: str  # e.g. "5,6,7"


def parse_unfreeze_layers(spec: str) -> list[int]:
    """Comma-separated → list[int]. Run config already normalises
    whitespace, but be defensive."""
    return [int(p.strip()) for p in spec.split(",") if p.strip()]


class UnfreezeAdapter(eqx.Module):
    """The "adapter" is a per-layer boolean mask flagging which layers
    are trainable. There are no new parameters — the trainer's
    gradient-mask hook uses this to zero gradients for frozen layers
    before the optimizer update.
    """

    layer_mask: Bool[Array, "n_layers"]
    cfg: UnfreezeConfig = eqx.field(static=True)


def init_unfreeze_adapter(
    backbone: PAWNModel, cfg: UnfreezeConfig, key: jax.Array | int
) -> UnfreezeAdapter:
    del key  # deterministic; no random init
    n_layers = backbone.cfg.n_layers
    picks = parse_unfreeze_layers(cfg.layers)
    for p in picks:
        if not 0 <= p < n_layers:
            raise ValueError(
                f"unfreeze_layers contains index {p} outside [0, {n_layers})"
            )
    mask = jnp.zeros((n_layers,), dtype=jnp.bool_).at[jnp.array(picks)].set(True)
    return UnfreezeAdapter(layer_mask=mask, cfg=cfg)


def apply_unfreeze(backbone: PAWNModel, adapter: UnfreezeAdapter) -> PAWNModel:
    """No-op — unfreeze doesn't modify the model; the trainer's
    gradient mask is what enforces the per-layer freezing."""
    del adapter
    return backbone


def unfreeze_filter(adapter: UnfreezeAdapter) -> UnfreezeAdapter:
    """No trainable params on UnfreezeAdapter itself — the trainable
    set lives in the backbone, gated by `layer_mask`. The trainer
    handles this with a custom gradient mask after computing the
    backbone's gradients."""
    return jax.tree_util.tree_map(lambda _: False, adapter)
