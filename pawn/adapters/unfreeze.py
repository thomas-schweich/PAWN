"""Partial-unfreeze strategy: fine-tune the top ``n_unfreeze`` transformer
layers of a PAWN backbone directly, leave the rest frozen.

Unlike LoRA / FiLM / bottleneck, this strategy adds *no* new parameters —
it just changes which existing backbone parameters get gradients. The
"adapter" PyTree is therefore just the backbone itself wrapped in a
config-carrying module; ``adapter_filter`` builds the True-only-on-
top-N-layer-slices spec.

Filter semantics:

* For each stacked layer-axis array (``wq``, ``wk``, ``wv``, ``wo``,
  ``attn_norm``, ``ffn_norm``, ``w_gate``, ``w_up``, ``w_down``), the
  per-layer leaves are *all* True (i.e. the full array is trainable).
  Per-layer fine-grained masking (e.g. only layers 6..8 of an 8-layer
  model) is handled by passing a 0-gradient mask through the per-layer
  axis in the trainer; the lift here just builds a stacked-True array
  the optimizer chains can multiply by a per-layer mask.
* ``final_norm`` + ``lm_head`` follow ``cfg.include_lm_head``.
* Embedding tables (``src_embed`` / ``dst_embed`` / ``promo_embed`` /
  ``pad_embed`` / ``outcome_embed``) follow ``cfg.include_embeddings``.

Because the per-layer mask is the active dimension, ``UnfreezeConfig``
also carries the integer range ``(first_unfrozen_layer..n_layers - 1)``;
``adapter_filter`` builds a stacked True/False array for the leading
n_layers axis of every layer-stacked weight, set True for layer indices
``>= first_unfrozen_layer``. Per-layer linear-projection arrays therefore
get gradients only on the unfrozen layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from pawn.model import PAWNModel


@dataclass(frozen=True)
class UnfreezeConfig:
    """Partial-unfreeze hyperparameters.

    Args:
        n_unfreeze: number of *top* transformer layers to unfreeze.
            Must be in ``[0, n_layers]``. ``0`` = freeze everything
            (only lm_head / embeddings depending on the flags);
            ``n_layers`` = full fine-tune.
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
    """Build the True-on-trainable filter spec for partial unfreeze.

    For each layer-stacked weight, the returned tree replaces the
    array leaf with a *bool array* of shape ``(n_layers, ...)`` set
    True on layer indices ``>= first_unfrozen_layer``. Equinox's
    partition machinery treats arbitrary leaves matching the original
    shape as the filter spec, so the optimizer + gradient
    transformations see per-layer slicing without any special-case
    code in the trainer.

    Note: this means downstream code must handle the case where some
    leaves are bool arrays (the trainable axis) and others are bare
    Python bools (the always-frozen / always-trainable scalars). The
    standard ``eqx.partition`` call handles this correctly.
    """
    bb = model.backbone
    cfg = model.cfg
    n_layers = bb.cfg.n_layers
    first_unfrozen = n_layers - cfg.n_unfreeze

    # Bool mask along the leading layer axis.
    layer_mask = (jnp.arange(n_layers) >= first_unfrozen)  # (n_layers,) bool

    # Start with everything = False.
    false_tree = jax.tree_util.tree_map(lambda _: False, model)

    # Replace each layer-stacked weight with a per-layer broadcast of
    # ``layer_mask``. The shape matches the weight so partition can
    # split element-wise.
    def _replace_layer_stacked(tree: UnfreezeModel) -> UnfreezeModel:
        for name in _LAYER_STACKED_FIELDS:
            weight = getattr(bb, name)
            broadcast_shape = (n_layers,) + (1,) * (weight.ndim - 1)
            mask = jnp.broadcast_to(
                layer_mask.reshape(broadcast_shape), weight.shape,
            )
            tree = eqx.tree_at(
                lambda t, n=name: getattr(t.backbone, n),
                tree, mask,
            )
        return tree

    tree = _replace_layer_stacked(false_tree)

    if cfg.include_lm_head:
        for name in _HEAD_FIELDS:
            weight = getattr(bb, name)
            tree = eqx.tree_at(
                lambda t, n=name: getattr(t.backbone, n),
                tree, jnp.ones(weight.shape, dtype=bool),
            )

    if cfg.include_embeddings:
        for name in _EMBED_FIELDS:
            weight = getattr(bb, name)
            tree = eqx.tree_at(
                lambda t, n=name: getattr(t.backbone, n),
                tree, jnp.ones(weight.shape, dtype=bool),
            )

    return tree
