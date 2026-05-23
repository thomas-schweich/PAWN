"""Sparse adaptation — random sparse perturbation of frozen attention weights.

A density-`p` random binary mask selects which weight entries get a
trainable delta added on top of the frozen value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from pawn.model import PAWNModel, TransformerLayer

__all__ = [
    "SparseConfig",
    "SparseAdapter",
    "init_sparse_adapter",
    "apply_sparse",
    "sparse_filter",
]


SparseTargets = Literal["qkvo", "qv", "qkv"]


@dataclass(frozen=True)
class SparseConfig:
    density: float
    targets: SparseTargets = "qkvo"
    ffn: bool = False


class SparseAdapter(eqx.Module):
    """Per-projection (mask, delta) pairs.

    ``mask`` is a frozen binary tensor (sparse pattern fixed at init);
    ``delta`` is a trainable dense tensor that gets element-multiplied
    by the mask before adding to the frozen weight.
    """

    delta_q: Float[Array, "n_layers d d"] | None
    mask_q: Bool[Array, "n_layers d d"] | None
    delta_k: Float[Array, "n_layers d d"] | None
    mask_k: Bool[Array, "n_layers d d"] | None
    delta_v: Float[Array, "n_layers d d"] | None
    mask_v: Bool[Array, "n_layers d d"] | None
    delta_o: Float[Array, "n_layers d d"] | None
    mask_o: Bool[Array, "n_layers d d"] | None
    cfg: SparseConfig = eqx.field(static=True)


def _maybe_init_sparse(
    active: bool, k: jax.Array, shape: tuple[int, int, int], density: float
) -> tuple[jax.Array | None, jax.Array | None]:
    if not active:
        return None, None
    bern = jax.random.bernoulli(k, p=density, shape=shape)
    delta = jnp.zeros(shape, dtype=jnp.float32)
    return delta, bern


def init_sparse_adapter(
    backbone: PAWNModel, cfg: SparseConfig, key: jax.Array | int
) -> SparseAdapter:
    if isinstance(key, int):
        key = jax.random.key(key)
    d = backbone.cfg.d_model
    n_layers = backbone.cfg.n_layers
    keys = jax.random.split(key, 4)
    shape = (n_layers, d, d)
    flags = {c: c in cfg.targets for c in "qkvo"}
    dq, mq = _maybe_init_sparse(flags["q"], keys[0], shape, cfg.density)
    dk, mk = _maybe_init_sparse(flags["k"], keys[1], shape, cfg.density)
    dv, mv = _maybe_init_sparse(flags["v"], keys[2], shape, cfg.density)
    do, mo = _maybe_init_sparse(flags["o"], keys[3], shape, cfg.density)
    return SparseAdapter(
        delta_q=dq, mask_q=mq,
        delta_k=dk, mask_k=mk,
        delta_v=dv, mask_v=mv,
        delta_o=do, mask_o=mo,
        cfg=cfg,
    )


def _add_sparse(
    weight: jax.Array, delta: jax.Array | None, mask: jax.Array | None
) -> jax.Array:
    if delta is None or mask is None:
        return weight
    return weight + delta * mask.astype(weight.dtype)


def apply_sparse(backbone: PAWNModel, adapter: SparseAdapter) -> PAWNModel:
    layers = backbone.layers
    new_layers = TransformerLayer(
        attn_norm_w=layers.attn_norm_w,
        wq=_add_sparse(layers.wq, adapter.delta_q, adapter.mask_q),
        wk=_add_sparse(layers.wk, adapter.delta_k, adapter.mask_k),
        wv=_add_sparse(layers.wv, adapter.delta_v, adapter.mask_v),
        wo=_add_sparse(layers.wo, adapter.delta_o, adapter.mask_o),
        ffn_norm_w=layers.ffn_norm_w,
        w_gate=layers.w_gate,
        w_up=layers.w_up,
        w_down=layers.w_down,
    )
    return eqx.tree_at(lambda m: m.layers, backbone, new_layers)


def sparse_filter(adapter: SparseAdapter) -> SparseAdapter:
    """Only the `delta_*` arrays are trainable — `mask_*` are bool
    (filtered out by `eqx.is_inexact_array`) and treated as frozen."""
    return jax.tree_util.tree_map(
        lambda leaf: True if eqx.is_inexact_array(leaf) else False, adapter
    )
