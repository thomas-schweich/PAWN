"""Sparse adaptation — random sparse perturbation of frozen weights.

A density-`p` random binary mask selects which weight entries get a
trainable delta added on top of the frozen value.

Targets:

- Attention projections ``wq`` / ``wk`` / ``wv`` / ``wo`` per
  ``cfg.targets``.
- FFN projections ``w_gate`` / ``w_up`` / ``w_down`` when
  ``cfg.ffn`` is set. Per v1 parity (``pawn.adapters.sparse._FFN_TARGETS``
  is the full ``(w_gate, w_up, w_down)`` triple).
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
    ``delta`` is a trainable dense tensor element-multiplied by the mask
    before being added to the frozen weight.
    """

    delta_q: Float[Array, "n_layers d d"] | None
    mask_q: Bool[Array, "n_layers d d"] | None
    delta_k: Float[Array, "n_layers d d"] | None
    mask_k: Bool[Array, "n_layers d d"] | None
    delta_v: Float[Array, "n_layers d d"] | None
    mask_v: Bool[Array, "n_layers d d"] | None
    delta_o: Float[Array, "n_layers d d"] | None
    mask_o: Bool[Array, "n_layers d d"] | None
    # FFN sparse projections, populated when cfg.ffn=True. Shapes mirror
    # the backbone's TransformerLayer (gate/up: d→d_ff; down: d_ff→d).
    delta_gate: Float[Array, "n_layers d d_ff"] | None
    mask_gate: Bool[Array, "n_layers d d_ff"] | None
    delta_up: Float[Array, "n_layers d d_ff"] | None
    mask_up: Bool[Array, "n_layers d d_ff"] | None
    delta_down: Float[Array, "n_layers d_ff d"] | None
    mask_down: Bool[Array, "n_layers d_ff d"] | None
    cfg: SparseConfig = eqx.field(static=True)


def _maybe_init_sparse(
    active: bool, k: jax.Array, shape: tuple[int, ...], density: float
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
    d_ff = backbone.cfg.d_ff
    n_layers = backbone.cfg.n_layers
    keys = jax.random.split(key, 7)
    attn_shape = (n_layers, d, d)
    gate_shape = (n_layers, d, d_ff)
    up_shape = (n_layers, d, d_ff)
    down_shape = (n_layers, d_ff, d)
    flags = {c: c in cfg.targets for c in "qkvo"}
    dq, mq = _maybe_init_sparse(flags["q"], keys[0], attn_shape, cfg.density)
    dk, mk = _maybe_init_sparse(flags["k"], keys[1], attn_shape, cfg.density)
    dv, mv = _maybe_init_sparse(flags["v"], keys[2], attn_shape, cfg.density)
    do, mo = _maybe_init_sparse(flags["o"], keys[3], attn_shape, cfg.density)
    dg, mg = _maybe_init_sparse(cfg.ffn, keys[4], gate_shape, cfg.density)
    du, mu = _maybe_init_sparse(cfg.ffn, keys[5], up_shape, cfg.density)
    dd, md = _maybe_init_sparse(cfg.ffn, keys[6], down_shape, cfg.density)
    return SparseAdapter(
        delta_q=dq, mask_q=mq,
        delta_k=dk, mask_k=mk,
        delta_v=dv, mask_v=mv,
        delta_o=do, mask_o=mo,
        delta_gate=dg, mask_gate=mg,
        delta_up=du, mask_up=mu,
        delta_down=dd, mask_down=md,
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
        w_gate=_add_sparse(layers.w_gate, adapter.delta_gate, adapter.mask_gate),
        w_up=_add_sparse(layers.w_up, adapter.delta_up, adapter.mask_up),
        w_down=_add_sparse(layers.w_down, adapter.delta_down, adapter.mask_down),
    )
    return eqx.tree_at(lambda m: m.layers, backbone, new_layers)


def sparse_filter(adapter: SparseAdapter) -> SparseAdapter:
    """Only the `delta_*` arrays are trainable — `mask_*` are bool
    (filtered out by `eqx.is_inexact_array`) and treated as frozen."""
    return jax.tree_util.tree_map(
        lambda leaf: True if eqx.is_inexact_array(leaf) else False, adapter
    )
