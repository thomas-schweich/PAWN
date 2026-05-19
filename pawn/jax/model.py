"""Equinox implementation of the PAWN causal language model.

One ``PAWNModel`` class serves as the shared-weight supernet, any sliced
variant, and any standalone model (e.g. a converted legacy checkpoint).
Transformer layers are stored stacked on a leading axis of size
``cfg.n_layers`` and applied with ``jax.lax.scan``; variant extraction
(``sliced``) is a pure slice of the weight arrays.

Mirrors the PyTorch ``pawn.model.PAWNCLM`` architecture exactly: factored
input embeddings, RMSNorm, RoPE, SwiGLU FFN, untied LM head, pre-norm
residual blocks. Attention is plain (materialized scores) rather than a fused
kernel — at seq 512 it is a small fraction of FLOPs and this keeps the code
backend-portable (see ``docs/jax-migration.md`` §3.1).

Linear weights use the JAX ``(in, out)`` convention (``x @ W``), which is the
transpose of PyTorch's ``nn.Linear`` ``(out, in)`` layout.
"""

from __future__ import annotations

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pawn.jax.config import (
    NUM_ACTIONS,
    OUTCOME_TOKEN_BASE,
    PAD_TOKEN,
    ModelConfig,
    validate_nested,
)

RMSNORM_EPS = 1e-6


@functools.cache
def _decomp_table() -> np.ndarray:
    """token -> (src_sq, dst_sq, promo_type) lookup, built from the Rust engine.

    Mirrors ``pawn.model._build_decomposition_table``. Cached as a host-side
    numpy array (it is identical for every model, so it lives outside the
    parameter PyTree); callers wrap it with ``jnp.asarray`` so it is embedded
    as a constant rather than a traced value. promo encoding: 0=none, 1=q,
    2=r, 3=b, 4=n.
    """
    from chess_engine import export_move_vocabulary

    vocab = export_move_vocabulary()
    square_names: list[str] = vocab["square_names"]
    promo_map = {"q": 1, "r": 2, "b": 3, "n": 4}
    table = np.zeros((NUM_ACTIONS, 3), dtype=np.int32)
    for token_idx, uci in vocab["token_to_move"].items():
        src = square_names.index(uci[:2])
        dst = square_names.index(uci[2:4])
        promo = promo_map.get(uci[4:], 0)
        table[int(token_idx)] = (src, dst, promo)
    return table


# ---------------------------------------------------------------------------
# Stateless building blocks
# ---------------------------------------------------------------------------


def _rmsnorm(x: jax.Array, weight: jax.Array) -> jax.Array:
    """RMSNorm with an fp32 reduction (mirrors ``pawn.model.RMSNorm``)."""
    x32 = x.astype(jnp.float32)
    norm = jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + RMSNORM_EPS)
    return (x32 * norm).astype(x.dtype) * weight


def _rope_tables(
    head_dim: int, seq_len: int, base: float
) -> tuple[jax.Array, jax.Array]:
    """Precompute RoPE cos/sin tables of shape ``(seq_len, head_dim // 2)``."""
    inv_freq = 1.0 / (
        base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(pos, inv_freq)
    return jnp.cos(freqs), jnp.sin(freqs)


def _apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Apply RoPE to ``x`` of shape ``(B, H, T, head_dim)``.

    ``cos`` / ``sin`` have shape ``(T, head_dim // 2)``.
    """
    x32 = x.astype(jnp.float32)
    pairs = x32.reshape(*x.shape[:-1], -1, 2)
    x0 = pairs[..., 0]
    x1 = pairs[..., 1]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    out0 = x0 * c - x1 * s
    out1 = x0 * s + x1 * c
    out = jnp.stack([out0, out1], axis=-1).reshape(x.shape)
    return out.astype(x.dtype)


def _build_attn_mask(attn_mask: jax.Array, seq_len: int) -> jax.Array:
    """Additive (0 / -inf) attention mask of shape ``(B, 1, T, T)``.

    Combines a causal lower-triangular mask with the per-position padding mask
    (``attn_mask`` is True at real, non-padding tokens).
    """
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    keep = causal[None, None, :, :] & attn_mask[:, None, None, :]
    return jnp.where(keep, 0.0, -jnp.inf).astype(jnp.float32)


def _attention(
    x: jax.Array,
    wq: jax.Array,
    wk: jax.Array,
    wv: jax.Array,
    wo: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    mask: jax.Array,
    n_heads: int,
) -> jax.Array:
    """Multi-head self-attention with plain (materialized) scores."""
    b, t, d = x.shape
    head_dim = d // n_heads

    def split(proj: jax.Array) -> jax.Array:
        return (x @ proj).reshape(b, t, n_heads, head_dim).transpose(0, 2, 1, 3)

    q = _apply_rope(split(wq), cos, sin)
    k = _apply_rope(split(wk), cos, sin)
    v = split(wv)

    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (head_dim**-0.5)
    weights = jax.nn.softmax(scores + mask, axis=-1)
    ctx = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
    ctx = ctx.transpose(0, 2, 1, 3).reshape(b, t, d)
    return ctx @ wo


def _ffn(
    x: jax.Array, w_gate: jax.Array, w_up: jax.Array, w_down: jax.Array
) -> jax.Array:
    """SwiGLU feed-forward network (mirrors ``pawn.model.SwiGLUFFN``)."""
    return (jax.nn.silu(x @ w_gate) * (x @ w_up)) @ w_down


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PAWNModel(eqx.Module):
    """PAWN causal LM — the supernet, a sliced variant, or a standalone model.

    Transformer-layer weights are stacked on a leading axis of size
    ``cfg.n_layers``. The fields are exactly the differentiable parameters;
    the token-decomposition table is a global constant (see ``_decomp_table``).
    """

    src_embed: jax.Array
    dst_embed: jax.Array
    promo_embed: jax.Array
    pad_embed: jax.Array
    outcome_embed: jax.Array
    attn_norm: jax.Array
    wq: jax.Array
    wk: jax.Array
    wv: jax.Array
    wo: jax.Array
    ffn_norm: jax.Array
    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array
    final_norm: jax.Array
    lm_head: jax.Array
    cfg: ModelConfig = eqx.field(static=True)

    def __call__(self, tokens: jax.Array, attn_mask: jax.Array) -> jax.Array:
        """Compute logits.

        Args:
            tokens: ``(B, T)`` int token ids.
            attn_mask: ``(B, T)`` bool — True at real (non-padding) positions.

        Returns:
            ``(B, T, vocab_size)`` logits.
        """
        cfg = self.cfg
        seq_len = tokens.shape[1]
        x = self._embed(tokens)
        cos, sin = _rope_tables(cfg.head_dim, seq_len, cfg.rope_base)
        mask = _build_attn_mask(attn_mask, seq_len)

        layers = (
            self.attn_norm,
            self.wq,
            self.wk,
            self.wv,
            self.wo,
            self.ffn_norm,
            self.w_gate,
            self.w_up,
            self.w_down,
        )

        def run_layer(
            h: jax.Array, lw: tuple[jax.Array, ...]
        ) -> tuple[jax.Array, None]:
            attn_norm, wq, wk, wv, wo, ffn_norm, w_gate, w_up, w_down = lw
            h = h + _attention(
                _rmsnorm(h, attn_norm),
                wq,
                wk,
                wv,
                wo,
                cos,
                sin,
                mask,
                cfg.n_heads,
            )
            h = h + _ffn(_rmsnorm(h, ffn_norm), w_gate, w_up, w_down)
            return h, None

        x, _ = jax.lax.scan(run_layer, x, layers)
        x = _rmsnorm(x, self.final_norm)
        return x @ self.lm_head

    def _embed(self, tokens: jax.Array) -> jax.Array:
        """Factored input embedding (mirrors ``pawn.model.CLMEmbedding``)."""
        table = jnp.asarray(_decomp_table())
        safe = jnp.clip(tokens, 0, NUM_ACTIONS - 1)
        decomp = table[safe]
        emb = (
            self.src_embed[decomp[..., 0]]
            + self.dst_embed[decomp[..., 1]]
            + self.promo_embed[decomp[..., 2]]
        )
        is_pad = (tokens == PAD_TOKEN)[..., None]
        emb = jnp.where(is_pad, self.pad_embed, emb)
        outcome_idx = jnp.clip(
            tokens - OUTCOME_TOKEN_BASE, 0, self.cfg.n_outcomes - 1
        )
        is_outcome = (tokens >= OUTCOME_TOKEN_BASE)[..., None]
        return jnp.where(is_outcome, self.outcome_embed[outcome_idx], emb)


def init_model(cfg: ModelConfig, key: jax.Array) -> PAWNModel:
    """Construct a model with fresh weights.

    Matches the PyTorch init (``pawn.model.PAWNCLM._init_weights``): every
    matrix (ndim > 1) is ``N(0, 0.02)``, RMSNorm weights are ones, and
    ``pad_embed`` is zeros.
    """
    d, n_layers, d_ff = cfg.d_model, cfg.n_layers, cfg.d_ff
    keys = iter(jax.random.split(key, 12))

    def normal(shape: tuple[int, ...]) -> jax.Array:
        return jax.random.normal(next(keys), shape, dtype=jnp.float32) * 0.02

    return PAWNModel(
        src_embed=normal((64, d)),
        dst_embed=normal((64, d)),
        promo_embed=normal((5, d)),
        pad_embed=jnp.zeros((d,), dtype=jnp.float32),
        outcome_embed=normal((cfg.n_outcomes, d)),
        attn_norm=jnp.ones((n_layers, d), dtype=jnp.float32),
        wq=normal((n_layers, d, d)),
        wk=normal((n_layers, d, d)),
        wv=normal((n_layers, d, d)),
        wo=normal((n_layers, d, d)),
        ffn_norm=jnp.ones((n_layers, d), dtype=jnp.float32),
        w_gate=normal((n_layers, d, d_ff)),
        w_up=normal((n_layers, d, d_ff)),
        w_down=normal((n_layers, d_ff, d)),
        final_norm=jnp.ones((d,), dtype=jnp.float32),
        lm_head=normal((d, cfg.vocab_size)),
        cfg=cfg,
    )


def sliced(supernet: PAWNModel, variant: ModelConfig) -> PAWNModel:
    """Extract ``variant`` as a nested slice of ``supernet``.

    Width is sliced as a prefix of ``d_model`` / ``d_ff``; depth as a prefix
    of the stacked layer axis. See ``docs/jax-migration.md`` §5.2.
    """
    validate_nested(variant, supernet.cfg)
    d, n_layers, d_ff = variant.d_model, variant.n_layers, variant.d_ff
    return PAWNModel(
        src_embed=supernet.src_embed[:, :d],
        dst_embed=supernet.dst_embed[:, :d],
        promo_embed=supernet.promo_embed[:, :d],
        pad_embed=supernet.pad_embed[:d],
        outcome_embed=supernet.outcome_embed[:, :d],
        attn_norm=supernet.attn_norm[:n_layers, :d],
        wq=supernet.wq[:n_layers, :d, :d],
        wk=supernet.wk[:n_layers, :d, :d],
        wv=supernet.wv[:n_layers, :d, :d],
        wo=supernet.wo[:n_layers, :d, :d],
        ffn_norm=supernet.ffn_norm[:n_layers, :d],
        w_gate=supernet.w_gate[:n_layers, :d, :d_ff],
        w_up=supernet.w_up[:n_layers, :d, :d_ff],
        w_down=supernet.w_down[:n_layers, :d_ff, :d],
        final_norm=supernet.final_norm[:d],
        lm_head=supernet.lm_head[:d, :],
        cfg=variant,
    )
