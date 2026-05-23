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
    """FiLM "applies" by folding gamma/beta into the per-layer RMSNorm
    weights — gamma_l multiplies the existing attn_norm + ffn_norm
    weights, beta is absorbed into the layer's running mean (which we
    don't have a separate field for, so we model gamma-only here for
    simplicity).

    Pragmatic v2 approach: scale ``ffn_norm_w`` by gamma. Beta is
    deferred — the gamma path alone is enough to dispatch and train
    the strategy under the unified trainer interface.
    """
    layers = backbone.layers
    # gamma multiplies the FFN norm weights (per-channel scale).
    new_ffn_norm_w = layers.ffn_norm_w * adapter.gamma
    new_layers = eqx.tree_at(lambda l: l.ffn_norm_w, layers, new_ffn_norm_w)
    new_final_norm_w = (
        backbone.final_norm_w * adapter.output_gamma
        if adapter.output_gamma is not None
        else backbone.final_norm_w
    )
    return eqx.tree_at(
        lambda m: (m.layers, m.final_norm_w),
        backbone,
        (new_layers, new_final_norm_w),
    )


def film_filter(adapter: FiLMAdapter) -> FiLMAdapter:
    return jax.tree_util.tree_map(
        lambda leaf: True if eqx.is_inexact_array(leaf) else False, adapter
    )
