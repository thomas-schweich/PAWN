"""Hybrid adapter: LoRA on attention projections + FiLM after each layer.

Composes the existing LoRA and FiLM modules: at forward time the
backbone first has the LoRA Δ folded into its attention projections
(via ``LoRAModel.effective_backbone()``), then the FiLM γ ⊙ h + β
post-layer affine is applied via the FiLM-style scan body. The
identity-at-init contract holds: LoRA's ``B == 0`` and FiLM's
γ=1, β=0 both make the wrapped forward bit-identical to the bare
backbone.

PyTree layout::

  HybridModel
  ├── backbone: PAWNModel  (frozen)
  ├── lora:     LoRAParams
  └── film:     FiLMParams

The adapter filter is True only on the ``lora.*`` + ``film.*``
array leaves; the backbone subtree is False.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from pawn.adapters.film import FiLMConfig, FiLMParams, init_film_model
from pawn.adapters.lora import (
    LoRAConfig,
    LoRAModel,
    LoRAParams,
    init_lora_model,
)
from pawn.model import (
    LayerWeights,
    PAWNModel,
    _attention,
    _build_attn_mask,
    _ffn,
    _rmsnorm,
    _rope_tables,
)


@dataclass(frozen=True)
class HybridConfig:
    """Hybrid (LoRA + FiLM) hyperparameters.

    Args:
        lora: LoRA config (rank, targets, alpha).
        film: FiLM config (use_output_film flag).
    """

    lora: LoRAConfig
    film: FiLMConfig


class HybridModel(eqx.Module):
    """Composes LoRA (attention-projection delta) + FiLM (per-layer
    affine). The forward folds the LoRA Δ in *before* the scan so each
    layer's attention projections see the adapted weights; the FiLM
    post-layer modulation then transforms the layer outputs."""

    backbone: PAWNModel
    lora: LoRAParams
    film: FiLMParams

    def __call__(self, tokens: jax.Array, attn_mask: jax.Array) -> jax.Array:
        # Materialise the LoRA-folded backbone once at the start of
        # the forward; each scan iteration then reads the adapted
        # projections without re-computing the delta.
        lora_wrapped = LoRAModel(backbone=self.backbone, lora=self.lora)
        effective = lora_wrapped.effective_backbone()

        cfg = effective.cfg
        seq_len = tokens.shape[1]
        if seq_len > cfg.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {cfg.max_seq_len}"
            )

        x = effective._embed(tokens)
        cos, sin = _rope_tables(cfg.head_dim, seq_len, cfg.rope_base)
        mask = _build_attn_mask(attn_mask, seq_len)
        real_so_far = jnp.cumsum(attn_mask, axis=1)
        all_masked_query = (real_so_far == 0)[:, None, :, None]

        layers = LayerWeights(
            attn_norm=effective.attn_norm,
            wq=effective.wq,
            wk=effective.wk,
            wv=effective.wv,
            wo=effective.wo,
            ffn_norm=effective.ffn_norm,
            w_gate=effective.w_gate,
            w_up=effective.w_up,
            w_down=effective.w_down,
        )

        def run_layer(
            h: jax.Array, step: tuple[LayerWeights, jax.Array, jax.Array],
        ) -> tuple[jax.Array, None]:
            lw, gamma, beta = step
            h = h + _attention(
                _rmsnorm(h, lw.attn_norm),
                lw.wq, lw.wk, lw.wv, lw.wo,
                cos, sin, mask, all_masked_query, cfg.n_heads,
            )
            h = h + _ffn(_rmsnorm(h, lw.ffn_norm), lw.w_gate, lw.w_up, lw.w_down)
            gamma = gamma.astype(h.dtype)
            beta = beta.astype(h.dtype)
            h = gamma[None, None, :] * h + beta[None, None, :]
            return h, None

        x, _ = jax.lax.scan(
            jax.checkpoint(run_layer, prevent_cse=False),
            x, (layers, self.film.gammas, self.film.betas),
        )
        x = _rmsnorm(x, effective.final_norm)
        logits = x @ effective.lm_head
        if self.film.cfg.use_output_film:
            logits = (
                self.film.gamma_out[None, None, :] * logits
                + self.film.beta_out[None, None, :]
            )
        return logits


def init_hybrid_model(
    backbone: PAWNModel, cfg: HybridConfig, key: jax.Array,
) -> HybridModel:
    """Compose LoRA + FiLM init. ``key`` is split into separate streams
    so each adapter's seed is independent."""
    klora, kfilm = jax.random.split(key, 2)
    lora_wrapped = init_lora_model(backbone, cfg.lora, klora)
    film_wrapped = init_film_model(backbone, cfg.film, kfilm)
    return HybridModel(
        backbone=backbone,
        lora=lora_wrapped.lora,
        film=film_wrapped.film,
    )


def adapter_filter(model: HybridModel) -> HybridModel:
    """``True`` on every ``model.lora.*`` + ``model.film.*`` array leaf,
    ``False`` everywhere else."""
    false_tree = jax.tree_util.tree_map(lambda _: False, model)
    lora_true = jax.tree_util.tree_map(eqx.is_inexact_array, model.lora)
    film_true = jax.tree_util.tree_map(eqx.is_inexact_array, model.film)
    tree = eqx.tree_at(lambda m: m.lora, false_tree, lora_true)
    tree = eqx.tree_at(lambda m: m.film, tree, film_true)
    return tree
