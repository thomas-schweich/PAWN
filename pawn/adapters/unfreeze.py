"""Partial-unfreeze strategy: fine-tune the top ``n_unfreeze`` transformer
layers of a PAWN backbone directly, leave the rest frozen.

Unlike LoRA / FiLM, this strategy adds *no* new parameters — it
selects a subset of existing backbone parameters as trainable.

Two pieces fit together so per-layer "top N" slicing works without
violating Equinox's filter-spec contract ("leaves must be Python
bools or callables only"):

* ``adapter_filter(model)`` — returns a Python-bool filter spec.
  Layer-stacked weights are marked ``True`` *as a whole* iff
  ``cfg.n_unfreeze > 0``; ``final_norm`` + ``lm_head`` follow
  ``cfg.include_lm_head``; embedding tables follow
  ``cfg.include_embeddings``. ``eqx.partition`` consumes this to
  produce a fully-trainable layer-stack subtree.

* ``make_gradient_mask(model)`` — returns a PyTree of per-element
  bool *arrays* shaped like the corresponding backbone weights.
  Layer-stacked weights get a leading-axis mask True only on the
  unfrozen layer indices (``>= n_layers - n_unfreeze``); head +
  embedding leaves get full-True / full-False masks per the config
  flags. The adapter trainer multiplies gradients by this mask
  before the optimizer step, or wraps the optimizer with
  ``optax.masked(opt, mask=...)``. Either way the backbone PyTree
  shape stays identical to the model's.

Codex / bug-detector / type-correctness all flagged the prior
array-valued ``adapter_filter`` as breaking ``eqx.partition``:
filter-spec leaves must be ``bool`` / callable, never ``jax.Array``.
This split fixes that without losing the per-layer slicing the
strategy is named for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp

from pawn.model import PAWNModel


@dataclass(frozen=True)
class UnfreezeConfig:
    """Partial-unfreeze hyperparameters.

    Args:
        n_unfreeze: number of *top* transformer layers to unfreeze.
            Must be in ``[0, n_layers]``. ``0`` = freeze every layer
            (only lm_head / embeddings remain trainable, depending on
            the flags); ``n_layers`` = full fine-tune.
        include_lm_head: also unfreeze ``final_norm`` + ``lm_head``.
        include_embeddings: also unfreeze the input embedding tables.
    """

    n_unfreeze: int
    include_lm_head: bool = True
    include_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.n_unfreeze < 0:
            raise ValueError(
                f"n_unfreeze={self.n_unfreeze} must be >= 0"
            )


class UnfreezeModel(eqx.Module):
    """Trivial wrapper around a ``PAWNModel`` carrying the unfreeze
    config. Forward is identical to the bare backbone."""

    backbone: PAWNModel
    cfg: UnfreezeConfig = eqx.field(static=True)

    def __call__(self, tokens: jax.Array, attn_mask: jax.Array) -> jax.Array:
        return self.backbone(tokens, attn_mask)


def init_unfreeze_model(
    backbone: PAWNModel, cfg: UnfreezeConfig, key: jax.Array,
) -> UnfreezeModel:
    """Build an ``UnfreezeModel`` over ``backbone``. ``key`` is unused —
    partial unfreeze adds no parameters."""
    del key
    if cfg.n_unfreeze > backbone.cfg.n_layers:
        raise ValueError(
            f"n_unfreeze={cfg.n_unfreeze} exceeds backbone n_layers="
            f"{backbone.cfg.n_layers}"
        )
    return UnfreezeModel(backbone=backbone, cfg=cfg)


# Layer-stacked arrays — the leading axis is ``n_layers``.
_LAYER_STACKED_FIELDS = (
    "attn_norm", "wq", "wk", "wv", "wo",
    "ffn_norm", "w_gate", "w_up", "w_down",
)
# Embedding-table fields — toggled by ``include_embeddings``.
_EMBED_FIELDS = (
    "src_embed", "dst_embed", "promo_embed", "pad_embed", "outcome_embed",
)
# Head fields — toggled by ``include_lm_head``.
_HEAD_FIELDS = ("final_norm", "lm_head")


def adapter_filter(model: UnfreezeModel) -> UnfreezeModel:
    """Python-bool filter spec for ``eqx.partition``.

    Coarse granularity: each backbone field is fully True or fully
    False. Per-layer slicing (top-N layers only) is layered on top
    via ``make_gradient_mask`` + ``optax.masked``.

    Layer-stacked weights are True iff ``cfg.n_unfreeze > 0`` (i.e.
    at least one layer is unfrozen). ``final_norm`` + ``lm_head``
    follow ``cfg.include_lm_head``; embedding tables follow
    ``cfg.include_embeddings``.
    """
    cfg = model.cfg
    layers_trainable = cfg.n_unfreeze > 0
    false_tree = jax.tree_util.tree_map(lambda _: False, model)

    def _set(tree: UnfreezeModel, name: str, value: bool) -> UnfreezeModel:
        return eqx.tree_at(
            lambda t, n=name: getattr(t.backbone, n),
            tree, replace=value,
        )

    tree = false_tree
    for name in _LAYER_STACKED_FIELDS:
        tree = _set(tree, name, layers_trainable)
    if cfg.include_lm_head:
        for name in _HEAD_FIELDS:
            tree = _set(tree, name, True)
    if cfg.include_embeddings:
        for name in _EMBED_FIELDS:
            tree = _set(tree, name, True)
    return tree


def make_gradient_mask(model: UnfreezeModel) -> UnfreezeModel:
    """Per-element bool-mask PyTree for ``optax.masked``.

    Returns a tree shaped like ``model`` whose array leaves are bool
    arrays of the *same shape* as the corresponding backbone
    weight. Layer-stacked weights get a leading-axis bool mask True
    only on layer indices ``>= n_layers - n_unfreeze``. Head +
    embedding leaves get full-True / full-False masks per the
    config flags. Non-array leaves (the config object, ``None``
    leaves on the trainable partition, etc.) are left as-is.

    The trainer then either multiplies gradients element-wise by
    this mask, or wraps the optimizer with ``optax.masked(opt,
    mask)`` — both produce the same result: backbone PyTree shape
    stays identical, but only the unfrozen layer slices get
    optimizer updates.
    """
    bb = model.backbone
    cfg = model.cfg
    n_layers = bb.cfg.n_layers
    first_unfrozen = n_layers - cfg.n_unfreeze
    layer_mask = (jnp.arange(n_layers) >= first_unfrozen)  # (n_layers,) bool

    # Start from a tree of zeros-shaped-like-leaves for arrays,
    # passthrough for everything else. ``eqx.is_array`` covers the
    # weight leaves; we only ever overwrite under the backbone
    # subtree below.
    def _zero_leaf(leaf: object) -> object:
        # The ``eqx.is_array`` runtime guard narrows ``leaf`` to a
        # ``jax.Array``, but pyright doesn't see ``eqx.is_array`` as a
        # ``TypeGuard``; ``typing.cast`` makes the narrowing explicit
        # without a noqa.
        if eqx.is_array(leaf):
            return jnp.zeros_like(cast(jax.Array, leaf), dtype=jnp.bool_)
        return leaf
    false_tree = jax.tree_util.tree_map(_zero_leaf, model)

    def _set_array(tree: UnfreezeModel, name: str, mask: jax.Array) -> UnfreezeModel:
        return eqx.tree_at(
            lambda t, n=name: getattr(t.backbone, n),
            tree, replace=mask,
        )

    tree = false_tree
    for name in _LAYER_STACKED_FIELDS:
        weight = getattr(bb, name)
        broadcast_shape = (n_layers,) + (1,) * (weight.ndim - 1)
        mask = jnp.broadcast_to(
            layer_mask.reshape(broadcast_shape), weight.shape,
        )
        tree = _set_array(tree, name, mask)

    if cfg.include_lm_head:
        for name in _HEAD_FIELDS:
            weight = getattr(bb, name)
            tree = _set_array(
                tree, name, jnp.ones(weight.shape, dtype=jnp.bool_),
            )
    if cfg.include_embeddings:
        for name in _EMBED_FIELDS:
            weight = getattr(bb, name)
            tree = _set_array(
                tree, name, jnp.ones(weight.shape, dtype=jnp.bool_),
            )
    return tree
