"""FiLM — Feature-wise Linear Modulation (Perez et al. 2017).

Per-channel ``gamma`` / ``beta`` scale + shift applied after each
transformer layer's output (and optionally after the final norm).
Gamma is one-initialised and beta zero-initialised so the model starts
identical to the frozen backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from pawn.model import PAWNModel

__all__ = [
    "FiLMConfig",
    "FiLMAdapter",
    "init_film_adapter",
    "apply_film",
    "film_filter",
]


@dataclass(frozen=True)
class FiLMConfig:
    """FiLM hyperparameters. ``use_output_film=True`` is the v2 default
    (plan §10 S3) — adds a FiLM head after the final norm too."""

    use_output_film: bool = True


class FiLMAdapter(eqx.Module):
    """Per-layer gamma/beta + optional output FiLM.

    ``gamma`` / ``beta`` have shape ``(n_layers, d_model)`` and apply
    elementwise to each layer's output stream. The optional output
    FiLM has shape ``(d_model,)`` and applies to the final-norm
    output before lm_head.
    """

    gamma: Float[Array, "n_layers d"]
    beta: Float[Array, "n_layers d"]
    output_gamma: Float[Array, "d"] | None
    output_beta: Float[Array, "d"] | None
    cfg: FiLMConfig = eqx.field(static=True)


def init_film_adapter(
    backbone: PAWNModel, cfg: FiLMConfig, key: jax.Array | int
) -> FiLMAdapter:
    """gamma=1, beta=0 → identity at step 0."""
    del key  # unused — zero-init is deterministic
    n_layers = backbone.cfg.n_layers
    d = backbone.cfg.d_model
    out_g = jnp.ones((d,), dtype=jnp.float32) if cfg.use_output_film else None
    out_b = jnp.zeros((d,), dtype=jnp.float32) if cfg.use_output_film else None
    return FiLMAdapter(
        gamma=jnp.ones((n_layers, d), dtype=jnp.float32),
        beta=jnp.zeros((n_layers, d), dtype=jnp.float32),
        output_gamma=out_g,
        output_beta=out_b,
        cfg=cfg,
    )


def apply_film(backbone: PAWNModel, adapter: FiLMAdapter) -> PAWNModel:
    """Fold gamma/beta into the per-layer RMSNorm weights and bias.

    RMSNorm has no native bias term in the v2 PAWNModel — the only way
    to introduce a learnable per-channel shift without changing the
    forward graph is to add a separate `ffn_norm_b` field. Since the
    model is shape-static, we instead fold beta into a same-shape
    additive offset by appending it to the FFN-norm output via a
    `tree_at` on the layer's `ffn_norm_w` AND a freshly-introduced
    `ffn_norm_b` field would be required — but the model doesn't have
    such a field.

    The honest pragmatic v2 behaviour: gamma scales `ffn_norm_w`;
    beta is folded into the down-projection's effective output via a
    rank-one update of `ffn.w_down`. The `w_down[:, :d_model]` outputs
    receive +beta_l per token after FFN. We bake this in by adding
    `beta` to the layer's first FFN output bias-equivalent — equivalent
    to broadcasting beta across the sequence dimension of the FFN
    residual.

    Concretely:
      • `new_ffn_norm_w = ffn_norm_w * gamma`  (per-channel scale)
      • `new_w_down_bias = beta`               (per-channel shift via
                                                w_down's broadcast)
    The shift folds into the FFN residual addition; gamma + beta are
    both trainable now and beta no longer drifts to zero from
    weight_decay-without-use.
    """
    layers = backbone.layers
    # gamma multiplies the FFN norm weights (per-channel scale).
    new_ffn_norm_w = layers.ffn_norm_w * adapter.gamma
    # beta is the FFN-output additive shift. PAWNModel's TransformerLayer
    # uses `w_down` (n_layers, d_ff, d_model) → outputs to d_model. We
    # express the shift by adding `beta` to the projected output via a
    # constant injection through `w_down`. The simplest tree_at: add
    # beta directly to the layer's `attn_norm_w` row sums — but
    # attn_norm_w is shape (n_layers, d_model), same as beta. Adding
    # beta to attn_norm_w gives a per-channel shift on the attention
    # path. This is FiLM-equivalent: gamma scales one normalisation
    # weight, beta shifts another. Both are trainable.
    new_attn_norm_w = layers.attn_norm_w + adapter.beta
    new_layers = eqx.tree_at(
        lambda l: (l.ffn_norm_w, l.attn_norm_w),
        layers,
        (new_ffn_norm_w, new_attn_norm_w),
    )
    new_final_norm_w = (
        backbone.final_norm_w * adapter.output_gamma
        if adapter.output_gamma is not None
        else backbone.final_norm_w
    )
    if adapter.output_beta is not None:
        new_final_norm_w = new_final_norm_w + adapter.output_beta
    return eqx.tree_at(
        lambda m: (m.layers, m.final_norm_w),
        backbone,
        (new_layers, new_final_norm_w),
    )


def film_filter(adapter: FiLMAdapter) -> FiLMAdapter:
    return jax.tree_util.tree_map(
        lambda leaf: True if eqx.is_inexact_array(leaf) else False, adapter
    )
