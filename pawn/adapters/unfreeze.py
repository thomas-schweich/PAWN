"""Unfreeze adapter — fine-tune explicit layer picks (v1 ``"5,6,7"`` form).

The adapter holds a per-layer-shaped copy of the backbone's transformer
slices plus a ``layer_mask`` selecting which slices are trainable.
:func:`apply_unfreeze` substitutes the adapter's values into the
backbone *only at unmasked layers* via ``jax.numpy.where`` — gradients
flow through the adapter's selected slots; masked slots get zero
gradient because they don't contribute to the forward pass.

This is the only strategy that trains the backbone itself rather than
a side-table of params; the substitute-via-where trick keeps the
two-tier ``eqx.partition`` model intact (the "adapter" is the
trainable surface; the backbone stays frozen).
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from pawn.model import PAWNModel, TransformerLayer

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
    """Trainable copies of the backbone's per-layer fields, gated by a
    boolean mask.

    Each field has the same ``(n_layers, ...)`` shape as the backbone's
    :class:`TransformerLayer`. Forward substitutes adapter values into
    the backbone via ``jnp.where(layer_mask, adapter, backbone)`` so
    the unmasked layers train while masked layers stay at the
    backbone's frozen values.
    """

    layers: TransformerLayer
    layer_mask: Bool[Array, "n_layers"]
    cfg: UnfreezeConfig = eqx.field(static=True)


def _broadcast_mask(
    mask: Bool[Array, "n_layers"], target_shape: tuple[int, ...]
) -> Bool[Array, "..."]:
    """Reshape `(n_layers,)` to `(n_layers, 1, 1, ...)` matching a
    target tensor's rank, so ``jnp.where`` broadcasts cleanly."""
    extra_dims = (1,) * (len(target_shape) - 1)
    return mask.reshape((mask.shape[0],) + extra_dims)


def init_unfreeze_adapter(
    backbone: PAWNModel, cfg: UnfreezeConfig, key: jax.Array | int
) -> UnfreezeAdapter:
    """Build an UnfreezeAdapter by cloning the backbone's per-layer
    fields and computing the layer mask from `cfg.layers`."""
    del key  # deterministic; no random init
    n_layers = backbone.cfg.n_layers
    picks = parse_unfreeze_layers(cfg.layers)
    for p in picks:
        if not 0 <= p < n_layers:
            raise ValueError(
                f"unfreeze_layers contains index {p} outside [0, {n_layers})"
            )
    mask = jnp.zeros((n_layers,), dtype=jnp.bool_).at[jnp.array(picks)].set(True)
    # The trainable adapter starts as an exact copy of the backbone's
    # transformer slices. Gradient updates accumulate there; the where
    # in `apply_unfreeze` blocks masked slots from contributing to
    # forward, so AdamW only effectively moves the unmasked entries.
    # Build a fresh `TransformerLayer` so the adapter's leaves are
    # distinct buffers from the backbone's — the JIT'd train step
    # donates both `state.backbone` and `state.adapter`; aliasing them
    # to the same underlying buffer triggers a donation collision.
    layers_copy = jax.tree_util.tree_map(lambda x: x.copy(), backbone.layers)
    return UnfreezeAdapter(
        layers=layers_copy, layer_mask=mask, cfg=cfg,
    )


def apply_unfreeze(
    backbone: PAWNModel, adapter: UnfreezeAdapter
) -> PAWNModel:
    """Substitute adapter values at unmasked layers; keep backbone at
    masked layers."""

    def _substitute(
        b: Float[Array, "n_layers ..."], a: Float[Array, "n_layers ..."]
    ) -> Float[Array, "n_layers ..."]:
        m = _broadcast_mask(adapter.layer_mask, b.shape)
        return jnp.where(m, a, b)

    new_layers = jax.tree_util.tree_map(
        _substitute, backbone.layers, adapter.layers
    )
    return eqx.tree_at(lambda m: m.layers, backbone, new_layers)


def unfreeze_filter(adapter: UnfreezeAdapter) -> UnfreezeAdapter:
    """Trainable: every inexact-float leaf in `adapter.layers`. Masked
    by `layer_mask` (which is bool, not an inexact array, so it's
    filtered out as non-trainable by default).

    The mask doesn't *prevent* AdamW from updating masked slots; it
    just ensures their forward contribution is zero, so their
    gradients are zero. AdamW's per-parameter state therefore sits
    idle for masked entries — no drift toward zero from weight decay
    because gradients are zero everywhere they're not used.
    """
    return jax.tree_util.tree_map(eqx.is_inexact_array, adapter)
