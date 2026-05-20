"""Equinox implementation of the PAWN causal language model.

One ``PAWNModel`` class serves as the shared-weight supernet, any sliced
variant, and any standalone model (e.g. a converted legacy checkpoint).
Transformer layers are stored stacked on a leading axis of size
``cfg.n_layers`` and applied with ``jax.lax.scan``; variant extraction
(``sliced``) is a pure slice of the weight arrays.

Mirrors the PyTorch ``pawn.model.PAWNCLM`` architecture closely: factored
input embeddings, RMSNorm, RoPE, SwiGLU FFN, untied LM head, pre-norm
residual blocks. The single intentional divergence is the RMSNorm cast
order (weight multiply runs in fp32 then downcasts once, vs legacy's
downcast-then-multiply) so a bf16 residual stream stays bf16 — see
``_rmsnorm``. Attention is plain (materialized scores) rather than a
fused kernel — at seq 512 it is a small fraction of FLOPs and this
keeps the code backend-portable (see ``docs/jax-migration.md`` §3.1).

Linear weights use the JAX ``(in, out)`` convention (``x @ W``), which is the
transpose of PyTorch's ``nn.Linear`` ``(out, in)`` layout.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from pawn.config import (
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
    """RMSNorm — close to ``pawn.model.RMSNorm`` with one intentional change.

    The reduction, normalization, and weight multiply all run in fp32; the
    result is downcast once to ``x.dtype``. Multiplying by the fp32 ``weight``
    before the downcast (rather than after, as legacy PyTorch does) keeps a
    bf16 residual stream in bf16 instead of letting the fp32 ``weight``
    silently promote it. This is load-bearing for bf16 parity between the
    JAX model and the thin PyTorch loader.
    """
    x32 = x.astype(jnp.float32)
    norm = jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + RMSNORM_EPS)
    return (x32 * norm * weight).astype(x.dtype)


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
    (``attn_mask`` is True at real, non-padding tokens). Padding masks *keys*,
    so a query row keeps every causally-visible real key.
    """
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    keep = causal[None, None, :, :] & attn_mask[:, None, None, :]
    return jnp.where(keep, 0.0, -jnp.inf)


def _attention(
    x: jax.Array,
    wq: jax.Array,
    wk: jax.Array,
    wv: jax.Array,
    wo: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    mask: jax.Array,
    all_masked_query: jax.Array,
    n_heads: int,
) -> jax.Array:
    """Multi-head self-attention with plain (materialized) scores.

    ``all_masked_query`` is a precomputed ``(B, 1, T, 1)`` bool array marking
    query positions whose entire key row is masked — passed in instead of
    derived from ``jnp.isnan(weights)`` so that ``jax.checkpoint`` over the
    scan-of-layers doesn't rematerialise the ``(B, H, T, T)`` isnan pass per
    layer during the backward sweep.
    """
    b, t, d = x.shape[0], x.shape[1], x.shape[2]
    head_dim = d // n_heads

    def split(proj: jax.Array) -> jax.Array:
        return (x @ proj).reshape(b, t, n_heads, head_dim).transpose(0, 2, 1, 3)

    q = _apply_rope(split(wq), cos, sin)
    k = _apply_rope(split(wk), cos, sin)
    v = split(wv)

    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (head_dim**-0.5)
    # For fully-masked query rows, route the raw ``scores`` into softmax (a
    # finite, differentiable input) so autodiff cannot observe
    # ``softmax(scores + -inf-row) == 0/0 → NaN`` — a single all-padding
    # batch element would otherwise poison every upstream gradient. For
    # normal rows, add the additive mask in the scores' dtype to keep a
    # bf16 forward in bf16. The single ``where`` over the (B, H, T, T)
    # scores avoids materialising a separate cast_mask + safe_mask pair
    # inside the scan body (which would be rematerialised per layer under
    # ``jax.checkpoint`` during the backward sweep). The row's weights are
    # zeroed below, so the all-masked output stays semantically "no
    # context" while gradients remain well-defined.
    masked_scores = jnp.where(
        all_masked_query,
        scores,
        scores + mask.astype(scores.dtype),
    )
    weights = jax.nn.softmax(masked_scores, axis=-1)
    weights = jnp.where(all_masked_query, 0.0, weights)
    ctx = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
    ctx = ctx.transpose(0, 2, 1, 3).reshape(b, t, d)
    return ctx @ wo


def _attention_prefill(
    x: jax.Array,
    wq: jax.Array,
    wk: jax.Array,
    wv: jax.Array,
    wo: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    mask: jax.Array,
    all_masked_query: jax.Array,
    n_heads: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Like ``_attention`` but also returns post-RoPE ``K`` and raw ``V``
    so the caller can stash them in a KV cache for later incremental
    decoding. Computed values are bitwise-identical to ``_attention``
    on the same inputs — the function is a pure superset (no
    extra ops on the attention output itself).
    """
    b, t, d = x.shape[0], x.shape[1], x.shape[2]
    head_dim = d // n_heads

    def split(proj: jax.Array) -> jax.Array:
        return (x @ proj).reshape(b, t, n_heads, head_dim).transpose(0, 2, 1, 3)

    q = _apply_rope(split(wq), cos, sin)
    k = _apply_rope(split(wk), cos, sin)
    v = split(wv)

    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (head_dim**-0.5)
    masked_scores = jnp.where(
        all_masked_query,
        scores,
        scores + mask.astype(scores.dtype),
    )
    weights = jax.nn.softmax(masked_scores, axis=-1)
    weights = jnp.where(all_masked_query, 0.0, weights)
    ctx = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
    ctx = ctx.transpose(0, 2, 1, 3).reshape(b, t, d)
    out = ctx @ wo
    # Cache K, V in (B, H, T, head_dim) layout — the same layout the
    # attention einsum consumes. Storing K/V in (B, T, H, head_dim)
    # would force a per-decode-step transpose to (B, H, T_max, head_dim)
    # for each of the n_layers layers (bytes-moved ~ 2 × n_layers × B
    # × T_max × H × head_dim × sizeof(dtype) per step); keeping the
    # cache pre-transposed eliminates that cost. The prefill pays one
    # transpose at startup (during `_apply_rope` + `split`); the
    # increment runs entirely in the cached layout.
    return out, k, v


def _attention_incremental(
    x_single: jax.Array,
    wq: jax.Array,
    wk: jax.Array,
    wv: jax.Array,
    wo: jax.Array,
    cos_at_pos: jax.Array,
    sin_at_pos: jax.Array,
    cached_k: jax.Array,
    cached_v: jax.Array,
    pos: jax.Array,
    n_heads: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single-token incremental attention.

    Args:
        x_single: ``(B, 1, d)`` — the new position's input.
        cached_k / cached_v: ``(B, n_heads, T_max, head_dim)`` — post-RoPE K
            and raw V from prior positions; positions ``[0, pos)`` are
            populated. The function writes the new K, V at slot ``pos``
            along the T axis (axis 2). The layout matches the attention
            einsum so no per-step transpose is needed.
        cos_at_pos / sin_at_pos: ``(1, head_dim // 2)`` — RoPE tables
            sliced to the new position.
        pos: scalar int — the new position. Must be a *traced* scalar
            so the JIT cache contains one trace for all positions.

    Returns ``(output, updated_cached_k, updated_cached_v)``. ``output``
    has shape ``(B, 1, d)``.
    """
    b, _, d = x_single.shape[0], x_single.shape[1], x_single.shape[2]
    head_dim = d // n_heads

    # Project + reshape directly into (B, H, 1, head_dim) — the einsum-
    # ready layout — so the cache write below is a single slice (no
    # transpose).
    q_rope = _apply_rope(
        (x_single @ wq).reshape(b, 1, n_heads, head_dim).transpose(0, 2, 1, 3),
        cos_at_pos, sin_at_pos,
    )
    k_rope = _apply_rope(
        (x_single @ wk).reshape(b, 1, n_heads, head_dim).transpose(0, 2, 1, 3),
        cos_at_pos, sin_at_pos,
    )
    v_new = (x_single @ wv).reshape(b, 1, n_heads, head_dim).transpose(0, 2, 1, 3)
    # Write at slot ``pos`` along the T axis (axis 2). ``k_rope[:, :, 0, :]``
    # drops the T=1 axis to give (B, H, head_dim) — the rank-3 update
    # ``dynamic_update_index_in_dim`` expects.
    new_cached_k = jax.lax.dynamic_update_index_in_dim(
        cached_k, k_rope[:, :, 0, :], pos, axis=2,
    )
    new_cached_v = jax.lax.dynamic_update_index_in_dim(
        cached_v, v_new[:, :, 0, :], pos, axis=2,
    )

    # Attend the single new query against the full cache + new K, V.
    # Cache is already (B, H, T_max, head_dim) — no transpose needed.
    scores = jnp.einsum(
        "bhqd,bhkd->bhqk", q_rope, new_cached_k,
    ) * (head_dim**-0.5)

    # Mask positions > pos. Positions [0, pos] are real; the rest are
    # the unfilled tail of the pre-allocated cache.
    T_max = cached_k.shape[2]
    positions = jnp.arange(T_max)
    is_real = positions <= pos
    additive_mask = jnp.where(is_real, 0.0, -jnp.inf).astype(scores.dtype)
    scores = scores + additive_mask[None, None, None, :]
    weights = jax.nn.softmax(scores, axis=-1)
    ctx = jnp.einsum("bhqk,bhkd->bhqd", weights, new_cached_v)
    ctx = ctx.transpose(0, 2, 1, 3).reshape(b, 1, d)
    return ctx @ wo, new_cached_k, new_cached_v


def _ffn(
    x: jax.Array, w_gate: jax.Array, w_up: jax.Array, w_down: jax.Array
) -> jax.Array:
    """SwiGLU feed-forward network (mirrors ``pawn.model.SwiGLUFFN``)."""
    return (jax.nn.silu(x @ w_gate) * (x @ w_up)) @ w_down


class KVCache(NamedTuple):
    """Per-layer K, V cache for incremental decoding.

    Pre-allocated at ``T_max = cfg.max_seq_len`` so that the JIT trace
    for ``PAWNModel.forward_incremental`` is shape-stable across every
    decode step (no recompile per position). K is stored post-RoPE so
    the increment doesn't have to re-RoPE cached entries; V is raw.

    Shape contract::

        k: (n_layers, B, n_heads, T_max, head_dim)
        v: (n_layers, B, n_heads, T_max, head_dim)

    The H-before-T layout matches the attention einsum's expected
    operand layout, so ``_attention_incremental`` reads the cache
    without a transpose — at production shape (n_layers=10, B=64,
    T_max=512, n_heads=10, head_dim=64) the savings are ~860 GB of
    avoided buffer movement across a 512-step decode.

    Positions ``[0, current_pos]`` are populated; positions
    ``(current_pos, T_max)`` are filler — the attention mask in
    ``_attention_incremental`` excludes them.
    """

    k: jax.Array
    v: jax.Array


class LayerWeights(NamedTuple):
    """One transformer layer's weights.

    Used as the per-step ``xs`` element of the ``lax.scan`` over the stacked
    layer axis in ``PAWNModel.__call__``; as a stacked instance (each field
    carrying a leading ``n_layers`` axis) it is the scan's full ``xs``.
    """

    attn_norm: jax.Array
    wq: jax.Array
    wk: jax.Array
    wv: jax.Array
    wo: jax.Array
    ffn_norm: jax.Array
    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PAWNModel(eqx.Module):
    """PAWN causal LM — the supernet, a sliced variant, or a standalone model.

    Transformer-layer weights are stacked on a leading axis of size
    ``cfg.n_layers``. The fields are exactly the differentiable parameters;
    the token-decomposition table is a global constant (see ``_decomp_table``).
    Shapes use d = ``cfg.d_model``, L = ``cfg.n_layers``, F = ``cfg.d_ff``.
    """

    src_embed: jax.Array  # (64, d)
    dst_embed: jax.Array  # (64, d)
    promo_embed: jax.Array  # (5, d)
    pad_embed: jax.Array  # (d,)
    outcome_embed: jax.Array  # (n_outcomes, d)
    attn_norm: jax.Array  # (L, d)
    wq: jax.Array  # (L, d, d)
    wk: jax.Array  # (L, d, d)
    wv: jax.Array  # (L, d, d)
    wo: jax.Array  # (L, d, d)
    ffn_norm: jax.Array  # (L, d)
    w_gate: jax.Array  # (L, d, F)
    w_up: jax.Array  # (L, d, F)
    w_down: jax.Array  # (L, F, d)
    final_norm: jax.Array  # (d,)
    lm_head: jax.Array  # (d, vocab_size)
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
        if seq_len > cfg.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {cfg.max_seq_len}"
            )
        x = self._embed(tokens)
        cos, sin = _rope_tables(cfg.head_dim, seq_len, cfg.rope_base)
        mask = _build_attn_mask(attn_mask, seq_len)
        # Precompute the all-masked query rows once; closing it over
        # ``run_layer`` keeps it out of ``jax.checkpoint``'s rematerialisation.
        # Derive from the (B, T) attn_mask directly via a cumulative count —
        # a query at position q is fully masked iff its causal window
        # [0..q] contains no real tokens. O(B·T) work and a (B, T) buffer
        # vs O(B·T²) on the materialised (B, 1, T, T) additive mask.
        # ``jnp.cumsum`` of a bool auto-promotes to an integer accumulator,
        # so no explicit ``astype`` is needed.
        real_so_far = jnp.cumsum(attn_mask, axis=1)
        all_masked_query = (real_so_far == 0)[:, None, :, None]

        layers = LayerWeights(
            attn_norm=self.attn_norm,
            wq=self.wq,
            wk=self.wk,
            wv=self.wv,
            wo=self.wo,
            ffn_norm=self.ffn_norm,
            w_gate=self.w_gate,
            w_up=self.w_up,
            w_down=self.w_down,
        )

        def run_layer(h: jax.Array, lw: LayerWeights) -> tuple[jax.Array, None]:
            h = h + _attention(
                _rmsnorm(h, lw.attn_norm),
                lw.wq,
                lw.wk,
                lw.wv,
                lw.wo,
                cos,
                sin,
                mask,
                all_masked_query,
                cfg.n_heads,
            )
            h = h + _ffn(_rmsnorm(h, lw.ffn_norm), lw.w_gate, lw.w_up, lw.w_down)
            return h, None

        # Wrap the scan body in ``jax.checkpoint`` so the backward sweep
        # rematerialises layer activations instead of storing them — without
        # this, scan stores the full (B, H, T, T) scores and ~(B, T, d_ff)
        # MLP intermediates per layer (~30–50 GB at B=256/T=512 for the
        # supernet), which would OOM the moment Phase 2 takes a gradient.
        # ``prevent_cse=False`` is safe (and recommended) inside ``lax.scan``:
        # scan already provides sequencing, so the CSE-prevention barriers
        # are wasted HLO ops. Under plain forward (no grad), remat is a
        # no-op semantically.
        x, _ = jax.lax.scan(
            jax.checkpoint(run_layer, prevent_cse=False), x, layers
        )
        x = _rmsnorm(x, self.final_norm)
        return x @ self.lm_head

    def hidden_all_layers(
        self, tokens: jax.Array, attn_mask: jax.Array
    ) -> jax.Array:
        """Per-layer hidden states for linear-probe and diagnostic use.

        Mirrors ``__call__``'s forward up to (but not including) the
        ``final_norm`` + ``lm_head`` projection, and records the hidden
        state at every layer boundary. The scan now accumulates per-step
        outputs via its ``ys`` channel; under ``jit`` XLA fuses the
        accumulation back into the same kernel sequence as ``__call__``,
        so this costs no extra forward pass when both are called on the
        same inputs (the typical probe loop calls only this method).

        Args:
            tokens: ``(B, T)`` int token ids.
            attn_mask: ``(B, T)`` bool — True at real (non-padding) positions.

        Returns:
            ``(n_layers + 1, B, T, d_model)`` — index 0 is the embedding
            output (pre-attention); indices ``1..n_layers`` are the
            outputs of transformer layers 0..n_layers-1.
        """
        cfg = self.cfg
        seq_len = tokens.shape[1]
        if seq_len > cfg.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {cfg.max_seq_len}"
            )
        x = self._embed(tokens)
        cos, sin = _rope_tables(cfg.head_dim, seq_len, cfg.rope_base)
        mask = _build_attn_mask(attn_mask, seq_len)
        real_so_far = jnp.cumsum(attn_mask, axis=1)
        all_masked_query = (real_so_far == 0)[:, None, :, None]

        layers = LayerWeights(
            attn_norm=self.attn_norm,
            wq=self.wq,
            wk=self.wk,
            wv=self.wv,
            wo=self.wo,
            ffn_norm=self.ffn_norm,
            w_gate=self.w_gate,
            w_up=self.w_up,
            w_down=self.w_down,
        )

        def run_layer(
            h: jax.Array, lw: LayerWeights
        ) -> tuple[jax.Array, jax.Array]:
            h = h + _attention(
                _rmsnorm(h, lw.attn_norm),
                lw.wq,
                lw.wk,
                lw.wv,
                lw.wo,
                cos,
                sin,
                mask,
                all_masked_query,
                cfg.n_heads,
            )
            h = h + _ffn(_rmsnorm(h, lw.ffn_norm), lw.w_gate, lw.w_up, lw.w_down)
            return h, h

        # ``run_layer`` returns ``(carry, h_after_layer)`` so ``lax.scan``'s
        # ``ys`` is the stacked per-layer outputs. No ``jax.checkpoint``
        # here — this method is for eval and probe training (the latter
        # only takes gradients on the probe head, not the backbone), so
        # remat's memory-for-compute trade is a pure loss.
        _carry, layer_outs = jax.lax.scan(run_layer, x, layers)
        # ``layer_outs`` is ``(n_layers, B, T, d)``; prepend the embed
        # output to match the legacy convention.
        return jnp.concatenate([x[None], layer_outs], axis=0)

    def forward_with_cache(
        self, tokens: jax.Array, attn_mask: jax.Array,
    ) -> tuple[jax.Array, KVCache]:
        """Prefill forward — same logits as ``__call__`` plus a populated
        ``KVCache`` for incremental decoding.

        The returned cache is shape-stable at ``cfg.max_seq_len`` even if
        ``tokens.shape[1] < cfg.max_seq_len`` (positions past the input
        are zero-filled; subsequent ``forward_incremental`` calls write
        to those slots in order and the attention mask excludes the
        unfilled tail). ``__call__`` and ``forward_with_cache`` produce
        bitwise-identical logits on the same input — the only
        difference is the cache return.

        Args:
            tokens: ``(B, T)`` int. ``T <= cfg.max_seq_len``.
            attn_mask: ``(B, T)`` bool.

        Returns:
            ``(logits, cache)``: ``(B, T, vocab_size)`` and a populated
            cache shaped ``(n_layers, B, T_max, n_heads, head_dim)``.
        """
        cfg = self.cfg
        seq_len = tokens.shape[1]
        if seq_len > cfg.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {cfg.max_seq_len}"
            )
        x = self._embed(tokens)
        cos, sin = _rope_tables(cfg.head_dim, seq_len, cfg.rope_base)
        mask = _build_attn_mask(attn_mask, seq_len)
        real_so_far = jnp.cumsum(attn_mask, axis=1)
        all_masked_query = (real_so_far == 0)[:, None, :, None]

        layers = LayerWeights(
            attn_norm=self.attn_norm, wq=self.wq, wk=self.wk, wv=self.wv,
            wo=self.wo, ffn_norm=self.ffn_norm, w_gate=self.w_gate,
            w_up=self.w_up, w_down=self.w_down,
        )

        def run_layer(
            h: jax.Array, lw: LayerWeights,
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            attn_out, k_layer, v_layer = _attention_prefill(
                _rmsnorm(h, lw.attn_norm),
                lw.wq, lw.wk, lw.wv, lw.wo,
                cos, sin, mask, all_masked_query, cfg.n_heads,
            )
            h = h + attn_out
            h = h + _ffn(_rmsnorm(h, lw.ffn_norm), lw.w_gate, lw.w_up, lw.w_down)
            return h, (k_layer, v_layer)

        # No ``jax.checkpoint`` — generation only does forward, no
        # backward; remat's trade is a pure loss here. Bitwise-identical
        # to ``__call__`` thanks to plain-attention determinism (the
        # remat in ``__call__`` is also semantically a no-op under
        # forward-only).
        x, (k_per_layer, v_per_layer) = jax.lax.scan(
            run_layer, x, layers,
        )
        x = _rmsnorm(x, self.final_norm)
        logits = x @ self.lm_head
        # Pad the cache out to (n_layers, B, H, T_max, head_dim) so
        # incremental decoding has one stable shape. The T axis is
        # axis 3 in the post-scan stacked tensor.
        T_max = cfg.max_seq_len
        if seq_len == T_max:
            k_full = k_per_layer
            v_full = v_per_layer
        else:
            pad_shape = (
                cfg.n_layers, tokens.shape[0], cfg.n_heads,
                T_max - seq_len, cfg.head_dim,
            )
            k_full = jnp.concatenate(
                [k_per_layer, jnp.zeros(pad_shape, dtype=k_per_layer.dtype)],
                axis=3,
            )
            v_full = jnp.concatenate(
                [v_per_layer, jnp.zeros(pad_shape, dtype=v_per_layer.dtype)],
                axis=3,
            )
        return logits, KVCache(k=k_full, v=v_full)

    def forward_incremental(
        self, token: jax.Array, pos: jax.Array, cache: KVCache,
    ) -> tuple[jax.Array, KVCache]:
        """Single-token incremental forward.

        Computes logits at position ``pos`` using the K, V cache for
        positions ``[0, pos)`` plus the new K, V it derives from
        ``token``. Returns ``(logits, updated_cache)`` — see the
        ``Returns:`` block below for the exact ``logits`` shape.

        ``pos`` is a *traced* scalar — one JIT trace covers every
        decode step.

        Args:
            token: ``(B, 1)`` int — the new position's token id.
            pos: scalar int — the new position. Must be
                ``0 <= pos < cfg.max_seq_len``; the cache must have
                positions ``[0, pos)`` populated.
            cache: ``KVCache`` from a prior ``forward_with_cache`` or
                ``forward_incremental`` call.

        Returns:
            ``(logits, cache)`` — logits shape ``(B, 1, vocab_size)``,
            cache updated with K, V at slot ``pos``.
        """
        cfg = self.cfg
        cos_full, sin_full = _rope_tables(
            cfg.head_dim, cfg.max_seq_len, cfg.rope_base,
        )
        # RoPE slice at the new position.
        cos_at = jax.lax.dynamic_slice_in_dim(cos_full, pos, 1, axis=0)
        sin_at = jax.lax.dynamic_slice_in_dim(sin_full, pos, 1, axis=0)

        x = self._embed(token)

        layers = LayerWeights(
            attn_norm=self.attn_norm, wq=self.wq, wk=self.wk, wv=self.wv,
            wo=self.wo, ffn_norm=self.ffn_norm, w_gate=self.w_gate,
            w_up=self.w_up, w_down=self.w_down,
        )

        def run_layer(
            h: jax.Array,
            inputs: tuple[LayerWeights, jax.Array, jax.Array],
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            lw, k_cache_layer, v_cache_layer = inputs
            attn_out, new_k, new_v = _attention_incremental(
                _rmsnorm(h, lw.attn_norm),
                lw.wq, lw.wk, lw.wv, lw.wo,
                cos_at, sin_at,
                k_cache_layer, v_cache_layer, pos, cfg.n_heads,
            )
            h = h + attn_out
            h = h + _ffn(_rmsnorm(h, lw.ffn_norm), lw.w_gate, lw.w_up, lw.w_down)
            return h, (new_k, new_v)

        x, (new_k_cache, new_v_cache) = jax.lax.scan(
            run_layer, x, (layers, cache.k, cache.v),
        )
        x = _rmsnorm(x, self.final_norm)
        return x @ self.lm_head, KVCache(k=new_k_cache, v=new_v_cache)

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
    counter = 0

    def normal(shape: tuple[int, ...]) -> jax.Array:
        # Derive a fresh key per matrix via fold_in, so there is no fixed
        # split count to keep in sync with the number of initialised arrays.
        nonlocal counter
        subkey = jax.random.fold_in(key, counter)
        counter += 1
        return jax.random.normal(subkey, shape, dtype=jnp.float32) * 0.02

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
    of the stacked layer axis. ``validate_nested`` enforces the nesting
    invariant. See ``docs/jax-migration.md`` §5.2.
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
