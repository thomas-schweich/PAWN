"""FiLM (Feature-wise Linear Modulation) adapter for the JAX PAWN backbone.

Implements Perez et al., 2017 (`arXiv:1709.07871`_), "FiLM: Visual
Reasoning with a General Conditioning Layer".

Inserts a per-channel affine transform ``h ← γ_l ⊙ h + β_l`` after each
transformer layer's output, and optionally a final FiLM on the lm-head
output (``logits ← γ_out ⊙ logits + β_out``).

Identity-initialised: ``γ`` = 1, ``β`` = 0 → the wrapped model is
bit-identical to the frozen backbone at step 0.

PyTree layout::

  FiLMModel
  ├── backbone: PAWNModel    (frozen)
  └── film: FiLMParams
       ├── gammas: [n_layers, d_model]
       ├── betas:  [n_layers, d_model]
       ├── gamma_out: [vocab_size]  (optional — present iff cfg.use_output_film)
       └── beta_out:  [vocab_size]

Param count (PAWN-base d=512, vocab=1980, n_layers=8, output FiLM on):
``2 × 8 × 512 + 2 × 1980 = 8,192 + 3,960 = 12,152`` — the smallest
adapter strategy by ~1-2 orders of magnitude.

.. _arXiv:1709.07871: https://arxiv.org/abs/1709.07871
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

# Re-use the model-internal helpers — FiLM's per-layer hook means the
# forward can't just call backbone(...) and apply post-hoc; it has to
# walk the layers with FiLM inserted between them.
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
class FiLMConfig:
    """FiLM hyperparameters.

    Args:
        use_output_film: also apply a FiLM transform to the logits.
            Default True (matches the legacy ``--no-output-film`` flag's
            False = "use it" semantics).
    """

    use_output_film: bool = True


class FiLMParams(eqx.Module):
    """Per-layer hidden-state FiLM + optional output-logit FiLM.

    Zero-width sentinel buffers stand in for the output gamma/beta when
    ``cfg.use_output_film`` is False so the PyTree shape is invariant —
    the partition machinery can then key on ``True`` array leaves
    unconditionally.
    """

    gammas: jax.Array        # (n_layers, d_model)
    betas: jax.Array         # (n_layers, d_model)
    gamma_out: jax.Array     # (vocab_size,) or (0,) sentinel
    beta_out: jax.Array      # (vocab_size,) or (0,) sentinel
    cfg: FiLMConfig = eqx.field(static=True)


class FiLMModel(eqx.Module):
    """Frozen PAWN backbone wrapped with per-layer + optional output FiLM."""

    backbone: PAWNModel
    film: FiLMParams

    def __call__(self, tokens: jax.Array, attn_mask: jax.Array) -> jax.Array:
        cfg = self.backbone.cfg
        seq_len = tokens.shape[1]
        if seq_len > cfg.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {cfg.max_seq_len}"
            )

        x = self.backbone._embed(tokens)
        cos, sin = _rope_tables(cfg.head_dim, seq_len, cfg.rope_base)
        mask = _build_attn_mask(attn_mask, seq_len)
        real_so_far = jnp.cumsum(attn_mask, axis=1)
        all_masked_query = (real_so_far == 0)[:, None, :, None]

        layers = LayerWeights(
            attn_norm=self.backbone.attn_norm,
            wq=self.backbone.wq,
            wk=self.backbone.wk,
            wv=self.backbone.wv,
            wo=self.backbone.wo,
            ffn_norm=self.backbone.ffn_norm,
            w_gate=self.backbone.w_gate,
            w_up=self.backbone.w_up,
            w_down=self.backbone.w_down,
        )

        # Pack the per-layer FiLM parameters alongside the layer
        # weights so a single scan walks them together — no Python
        # loop over n_layers and no separate post-hoc multiply on a
        # stacked-output tensor.
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
            # FiLM is per-channel, broadcast over (B, T):
            #   h_b_t_d ← γ_l_d · h_b_t_d + β_l_d
            # Cast γ/β to the carry dtype before the multiply so a bf16
            # backbone doesn't have its carry silently promoted to fp32
            # — ``lax.scan`` rejects the carry-dtype mismatch on the
            # next iteration. (Same dtype-promotion fix the Phase-3
            # LoRA chunk applied to its delta cast.) (Codex P2)
            gamma = gamma.astype(h.dtype)
            beta = beta.astype(h.dtype)
            h = gamma[None, None, :] * h + beta[None, None, :]
            return h, None

        x, _ = jax.lax.scan(
            jax.checkpoint(run_layer, prevent_cse=False),
            x, (layers, self.film.gammas, self.film.betas),
        )
        x = _rmsnorm(x, self.backbone.final_norm)
        logits = x @ self.backbone.lm_head
        if self.film.cfg.use_output_film:
            logits = (
                self.film.gamma_out[None, None, :] * logits
                + self.film.beta_out[None, None, :]
            )
        return logits


def init_film_model(
    backbone: PAWNModel, cfg: FiLMConfig, key: jax.Array,
) -> FiLMModel:
    """Identity-initialised FiLM (γ=1, β=0). ``key`` is kept in the
    signature for parity with other adapter constructors but unused
    here — FiLM has no random init."""
    del key  # explicit unused
    n_layers = backbone.cfg.n_layers
    d = backbone.cfg.d_model
    v = backbone.cfg.vocab_size
    gammas = jnp.ones((n_layers, d), dtype=jnp.float32)
    betas = jnp.zeros((n_layers, d), dtype=jnp.float32)
    if cfg.use_output_film:
        gamma_out = jnp.ones((v,), dtype=jnp.float32)
        beta_out = jnp.zeros((v,), dtype=jnp.float32)
    else:
        # Zero-width sentinels so the PyTree shape is invariant.
        gamma_out = jnp.zeros((0,), dtype=jnp.float32)
        beta_out = jnp.zeros((0,), dtype=jnp.float32)
    return FiLMModel(
        backbone=backbone,
        film=FiLMParams(
            gammas=gammas, betas=betas,
            gamma_out=gamma_out, beta_out=beta_out, cfg=cfg,
        ),
    )


def adapter_filter(model: FiLMModel) -> FiLMModel:
    """``True`` on every ``model.film.*`` array leaf, ``False`` elsewhere.

    Same convention as ``pawn.adapters.lora.adapter_filter`` — pair
    with ``eqx.partition(model, adapter_filter(model))`` to get the
    trainable / frozen subtrees.
    """
    false_tree = jax.tree_util.tree_map(lambda _: False, model)
    film_true = jax.tree_util.tree_map(eqx.is_inexact_array, model.film)
    return eqx.tree_at(lambda m: m.film, false_tree, film_true)
