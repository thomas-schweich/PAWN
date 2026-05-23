"""PAWN: Equinox decoder-only transformer for chess move prediction.

A single :class:`PAWNModel` covers the supernet, every nested variant,
and any standalone (converted-legacy) model. The full module is built
at the supernet's dimensions; :func:`sliced` returns a new
:class:`PAWNModel` with every parameter tensor sliced to the variant's
shape (the inner ``[:d_V, :d_V]`` block of every weight matrix).

Architectural choices:

- **Stacked-layer ``lax.scan``.** Per-layer weights are stored with a
  leading ``n_layers`` axis (e.g. ``wq`` is shape
  ``(n_layers, d_model, d_model)``). The forward pass applies the
  layers via :func:`jax.lax.scan`, so XLA sees one layer body unrolled
  into a loop rather than N separate trace bodies — far less HLO and a
  much faster compile.
- **Plain attention** (materialised ``QK^T``). At seq=512, attention
  is roughly 12% of step FLOPs; plain attention sidesteps the
  fused-kernel maturity issues we'd hit with JAX-on-ROCm.
- **RMSNorm in the v2 cast order** — compute the norm in fp32, multiply
  by the weight in fp32, downcast at the very end. The v1 PyTorch
  layout downcast *between* the norm and the weight multiply. The two
  paths are bit-identical in fp32 (the difference vanishes when the
  intermediate dtype is already float32), so the legacy converter's
  fp32 parity test agrees on both; the v2 layout is one fewer cast and
  is the form the plan §5 calls out. See :func:`_rmsnorm` for details.
- **RoPE applied in fp32** to Q and K, then downcast — same reason.
- **SwiGLU FFN:** ``down(silu(gate(x)) * up(x))``.
- **Factored input embeddings:** every move token decomposes into
  ``src_embed[s] + dst_embed[d] + promo_embed[p]``. PAD positions get
  ``pad_embed``; outcome-token positions get
  ``outcome_embed[token - OUTCOME_TOKEN_BASE]``. Overrides are
  branchless (:func:`jnp.where`) so the compiler can fuse them.
- **Output head:** a single linear over the full vocabulary. Argmax
  callers restrict to ``[0, NUM_ACTIONS)`` so PAD and outcome tokens
  can't be sampled — that restriction lives in :mod:`pawn.eval`, not
  here.

Save schema: the 16 trainable arrays in declaration order — the
:data:`SAVED_FIELDS` tuple is the canonical ordering, and this module
asserts ``len(SAVED_FIELDS) == 16`` at import (see below). The
non-trainable :attr:`PAWNModel.decomp_table` buffer is rebuilt at load
time from the engine vocabulary; RoPE phase tables are recomputed
inside the forward pass and never stored. Both stay out of the
safetensors payload.
"""

from __future__ import annotations

from typing import Final

import equinox as eqx
import jax
import jax.numpy as jnp
from chess_engine import export_move_vocabulary
from jaxtyping import Array, Bool, Float, Int

from pawn.config import (
    NUM_ACTIONS,
    OUTCOME_TOKEN_BASE,
    PAD_TOKEN,
    ModelConfig,
    validate_nested,
)

__all__ = [
    "PAWNModel",
    "TransformerLayer",
    "SAVED_FIELDS",
    "init_model",
    "sliced",
]


# Names of the 16 trainable arrays in safetensors declaration order.
# The `assert` below pins the count at import; `pawn.checkpoint` imports
# this module, so the guard fires before any save/load can run.
SAVED_FIELDS: Final[tuple[str, ...]] = (
    "embed_src",
    "embed_dst",
    "embed_promo",
    "embed_pad",
    "embed_outcome",
    "layers.attn_norm_w",
    "layers.wq",
    "layers.wk",
    "layers.wv",
    "layers.wo",
    "layers.ffn_norm_w",
    "layers.w_gate",
    "layers.w_up",
    "layers.w_down",
    "final_norm_w",
    "lm_head",
)
assert len(SAVED_FIELDS) == 16, "model save schema must be 16 fields"


_RMSNORM_EPS: Final[float] = 1e-6


# ---------------------------------------------------------------------------
# Math helpers (free functions — easier to unit-test, easier to JIT-trace)
# ---------------------------------------------------------------------------


def _rmsnorm(
    x: Float[Array, "... d"],
    weight: Float[Array, "d"],
) -> Float[Array, "... d"]:
    """RMSNorm with the v2 cast order: norm AND weight multiply in fp32,
    downcast once at the very end.

    Concretely: upcast ``x`` and ``weight`` to fp32, compute
    ``(x_f * rms_f) * w_f`` in fp32, then ``.astype(x.dtype)``. The
    plan §5 phrase is "weight-multiply-in-fp32-then-downcast" — i.e.
    this v2 order.

    Note the v1 PyTorch path was subtly different: ``(x_f * norm).to(
    x.dtype) * self.weight`` — downcasting *between* the norm and the
    weight multiply. The two orders are bit-identical in fp32 (a
    no-op ``.to(float32)`` between) and diverge only in bf16/fp16
    activations. The legacy converter's parity test runs in fp32, so
    both orders agree on its tolerance; this v2 layout is the one the
    plan asks for and is one fewer cast.
    """
    x_f = x.astype(jnp.float32)
    w_f = weight.astype(jnp.float32)
    rms = jax.lax.rsqrt(jnp.mean(x_f * x_f, axis=-1, keepdims=True) + _RMSNORM_EPS)
    return ((x_f * rms) * w_f).astype(x.dtype)


def _build_rope(
    head_dim: int, max_seq_len: int, base: float
) -> tuple[Float[Array, "T half"], Float[Array, "T half"]]:
    """Precompute RoPE phase tables.

    Returns ``(cos, sin)`` each of shape ``(max_seq_len, head_dim // 2)``,
    in fp32. The per-step RoPE application upcasts Q/K to fp32 before
    multiplying through, so storing the tables in fp32 is the canonical
    form.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.einsum("t,d->td", t, inv_freq)  # (T, half)
    assert freqs.shape == (max_seq_len, half)
    return jnp.cos(freqs), jnp.sin(freqs)


def _apply_rope(
    x: Float[Array, "B H T d_head"],
    rope_cos: Float[Array, "T half"],
    rope_sin: Float[Array, "T half"],
) -> Float[Array, "B H T d_head"]:
    """Apply RoPE to Q or K.

    Split the last axis (``head_dim``) into adjacent ``(even, odd)``
    pairs and rotate each pair by the angle ``freqs[t, i]``. Compute in
    fp32 then downcast to the input dtype.
    """
    orig_dtype = x.dtype
    x_f = x.astype(jnp.float32)
    pairs = x_f.reshape(*x_f.shape[:-1], -1, 2)
    x0 = pairs[..., 0]
    x1 = pairs[..., 1]
    out0 = x0 * rope_cos - x1 * rope_sin
    out1 = x0 * rope_sin + x1 * rope_cos
    out = jnp.stack([out0, out1], axis=-1).reshape(x_f.shape)
    return out.astype(orig_dtype)


# ---------------------------------------------------------------------------
# Static buffers
# ---------------------------------------------------------------------------


def _build_decomp_table() -> Int[Array, "n_actions 3"]:
    """Build the (src, dst, promo) lookup table from the engine vocab.

    Each move token in ``[0, NUM_ACTIONS)`` decomposes into a source
    square (0–63), destination square (0–63), and promotion type
    (0=none, 1=q, 2=r, 3=b, 4=n). PAD and outcome tokens are clamped to
    a safe row by :meth:`PAWNModel._embed` and then overwritten
    branchlessly.
    """
    vocab = export_move_vocabulary()
    sq_names: list[str] = list(vocab["square_names"])
    sq_index = {name: i for i, name in enumerate(sq_names)}  # O(1) lookup
    promo_map = {"q": 1, "r": 2, "b": 3, "n": 4}
    rows: list[list[int]] = []
    for token_idx in range(NUM_ACTIONS):
        uci: str = vocab["token_to_move"][token_idx]
        src = sq_index[uci[:2]]
        dst = sq_index[uci[2:4]]
        promo = promo_map.get(uci[4:], 0)
        rows.append([src, dst, promo])
    return jnp.array(rows, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class TransformerLayer(eqx.Module):
    """One transformer block's parameters, packed with a leading
    ``n_layers`` axis so the whole stack can be applied with
    :func:`jax.lax.scan`.

    For a single instantiation of :class:`PAWNModel`, ``model.layers`` is
    *one* :class:`TransformerLayer` whose leaves each have a leading
    ``n_layers`` dimension (e.g. ``model.layers.wq`` is
    ``(n_layers, d_model, d_model)``). The forward pass uses
    :func:`jax.lax.scan` to iterate over that leading axis.
    """

    attn_norm_w: Float[Array, "n_layers d"]
    wq: Float[Array, "n_layers d d"]
    wk: Float[Array, "n_layers d d"]
    wv: Float[Array, "n_layers d d"]
    wo: Float[Array, "n_layers d d"]
    ffn_norm_w: Float[Array, "n_layers d"]
    w_gate: Float[Array, "n_layers d d_ff"]
    w_up: Float[Array, "n_layers d d_ff"]
    w_down: Float[Array, "n_layers d_ff d"]


class PAWNModel(eqx.Module):
    """Decoder-only transformer over the move + outcome vocabulary.

    See module docstring for the architectural choices and the save
    schema. The class lays out 16 trainable array fields (declaration
    order matching :data:`SAVED_FIELDS`), one non-trainable
    int32 buffer (:attr:`decomp_table` — filtered out automatically by
    ``eqx.is_inexact_array``), and one static ``cfg`` reference. RoPE
    phase tables are not stored on the model; they're recomputed
    inside :meth:`__call__` per forward call and constant-folded by
    JIT when ``cfg`` is static.
    """

    # 16 trainable fields — declaration order = save order.
    embed_src: Float[Array, "64 d"]
    embed_dst: Float[Array, "64 d"]
    embed_promo: Float[Array, "5 d"]
    embed_pad: Float[Array, "d"]
    embed_outcome: Float[Array, "n_out d"]
    layers: TransformerLayer
    final_norm_w: Float[Array, "d"]
    lm_head: Float[Array, "d V"]

    # Non-trainable lookup table (int32 — `eqx.is_inexact_array` filters it
    # out automatically so the trainer's optimizer never touches it). Not
    # in `SAVED_FIELDS`; `pawn.checkpoint.load_model` rebuilds it from the
    # engine vocab at load time via `_build_decomp_table()`.
    decomp_table: Int[Array, "n_actions 3"]

    # Static config: kept out of the PyTree so JIT keys on it directly.
    # RoPE phase tables are NOT stored — they're a function of cfg only and
    # are recomputed inside `__call__`. Storing them as float PyTree leaves
    # would put them in the path of `eqx.filter(model, eqx.is_inexact_array)`
    # (the standard equinox trainable filter), letting the optimizer update
    # the positional encoding. Under JIT with static cfg, the recomputation
    # is constant-folded and lives in the compiled program exactly once.
    cfg: ModelConfig = eqx.field(static=True)

    def __call__(
        self,
        input_ids: Int[Array, "B T"],
        attention_mask: Int[Array, "B T"] | None = None,
    ) -> Float[Array, "B T V"]:
        """Forward pass.

        ``input_ids`` is ``(batch, seq)`` int32 token IDs (move tokens,
        PAD, or outcome tokens). ``attention_mask`` is ``1`` for real
        tokens and ``0`` for PAD; if omitted, a fully-real mask is
        assumed.

        Returns logits of shape ``(batch, seq, vocab_size)``. Callers
        that sample argmax over the move vocabulary should restrict to
        ``[:, :, :NUM_ACTIONS]`` so PAD and outcome tokens can't be
        chosen.
        """
        T = input_ids.shape[-1]
        if T > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds cfg.max_seq_len "
                f"{self.cfg.max_seq_len}"
            )
        x = self._embed(input_ids)
        # RoPE tables are recomputed per call — constant-folded by JIT
        # under a static cfg, so the cost is one trace-time build.
        rope_cos, rope_sin = _build_rope(self.cfg.head_dim, T, self.cfg.rope_base)
        causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        if attention_mask is None:
            mask = causal[None, None, :, :]  # (1, 1, T, T)
        else:
            pad = attention_mask.astype(jnp.bool_)[:, None, None, :]  # (B, 1, 1, T)
            mask = causal[None, None, :, :] & pad
        x = self._run_layers(x, rope_cos, rope_sin, mask)
        x = _rmsnorm(x, self.final_norm_w)
        return jnp.einsum("btd,dv->btv", x, self.lm_head)

    # -----------------------------------------------------------------------
    # Forward-pass internals
    # -----------------------------------------------------------------------

    def _embed(self, input_ids: Int[Array, "B T"]) -> Float[Array, "B T d"]:
        """Factored move embeddings + PAD/outcome overrides.

        Tokens in ``[0, NUM_ACTIONS)``: ``src + dst + promo`` lookup.
        Tokens equal to ``PAD_TOKEN``: replaced with ``embed_pad``.
        Tokens ``>= OUTCOME_TOKEN_BASE``: replaced with
        ``embed_outcome[token - OUTCOME_TOKEN_BASE]``.

        Overrides are branchless (:func:`jnp.where`) so the layout is
        fusion-friendly.
        """
        ids = input_ids.astype(jnp.int32)
        # Clamp to a safe range so the decomp table lookup doesn't OOB
        # on PAD / outcome positions; those positions are overwritten below.
        safe_ids = jnp.clip(ids, 0, NUM_ACTIONS - 1)
        decomp = self.decomp_table[safe_ids]  # (B, T, 3)
        src_idx = decomp[..., 0]
        dst_idx = decomp[..., 1]
        promo_idx = decomp[..., 2]
        emb = (
            self.embed_src[src_idx]
            + self.embed_dst[dst_idx]
            + self.embed_promo[promo_idx]
        )

        pad_mask = (ids == PAD_TOKEN)[..., None]
        emb = jnp.where(pad_mask, self.embed_pad, emb)

        n_outcomes = self.embed_outcome.shape[0]
        outcome_idx = jnp.clip(ids - OUTCOME_TOKEN_BASE, 0, n_outcomes - 1)
        outcome_emb = self.embed_outcome[outcome_idx]
        outcome_mask = (ids >= OUTCOME_TOKEN_BASE)[..., None]
        emb = jnp.where(outcome_mask, outcome_emb, emb)
        return emb

    def _run_layers(
        self,
        x: Float[Array, "B T d"],
        rope_cos: Float[Array, "T half"],
        rope_sin: Float[Array, "T half"],
        mask: Bool[Array, "B 1 T T"],
    ) -> Float[Array, "B T d"]:
        """Apply all ``n_layers`` transformer blocks via :func:`jax.lax.scan`.

        ``self.layers`` is one :class:`TransformerLayer` whose leaves
        have a leading ``n_layers`` axis. :func:`jax.lax.scan` iterates
        layer-by-layer; each iteration sees a per-layer slice of every
        weight tensor.
        """
        head_dim = self.cfg.head_dim
        n_heads = self.cfg.n_heads
        # Compile-time constant — hoist out of the scan body so we don't
        # re-allocate a 0-d scalar and dispatch a sqrt kernel on every layer.
        inv_scale = head_dim ** -0.5
        # Mask sentinel: hoist `jnp.finfo` out of the scan body (same reason).
        # Use the working-dtype's minimum so softmax sees -inf-equivalent.
        attn_neg_inf = jnp.finfo(x.dtype).min

        def step(
            carry: Float[Array, "B T d"],
            layer: TransformerLayer,
        ) -> tuple[Float[Array, "B T d"], None]:
            h = carry
            # ---- attention block (pre-norm + residual) ----
            normed = _rmsnorm(h, layer.attn_norm_w)
            B, T, D = normed.shape  # noqa: N806
            q = jnp.einsum("btd,de->bte", normed, layer.wq)
            k = jnp.einsum("btd,de->bte", normed, layer.wk)
            v = jnp.einsum("btd,de->bte", normed, layer.wv)
            q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
            q = _apply_rope(q, rope_cos, rope_sin)
            k = _apply_rope(k, rope_cos, rope_sin)
            scores = jnp.einsum("bhid,bhjd->bhij", q, k) * inv_scale
            scores = jnp.where(mask, scores, attn_neg_inf)
            attn = jax.nn.softmax(scores, axis=-1)
            attn_out = jnp.einsum("bhij,bhjd->bhid", attn, v)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, D)
            h = h + jnp.einsum("btd,de->bte", attn_out, layer.wo)

            # ---- ffn block (pre-norm + residual) ----
            normed = _rmsnorm(h, layer.ffn_norm_w)
            gate = jnp.einsum("btd,df->btf", normed, layer.w_gate)
            up = jnp.einsum("btd,df->btf", normed, layer.w_up)
            ffn_out = jnp.einsum("btf,fd->btd", jax.nn.silu(gate) * up, layer.w_down)
            h = h + ffn_out
            return h, None

        x, _ = jax.lax.scan(step, x, self.layers)
        return x


# ---------------------------------------------------------------------------
# Construction + slicing
# ---------------------------------------------------------------------------


def _normal_init(
    key: jax.Array, shape: tuple[int, ...], std: float = 0.02
) -> jax.Array:
    """Normal(0, std) init.

    Matches the v1 ``nn.init.normal_(p, std=0.02)`` convention for any
    parameter with dim > 1.
    """
    return jax.random.normal(key, shape, dtype=jnp.float32) * std


def init_model(cfg: ModelConfig, key: jax.Array | int) -> PAWNModel:
    """Build a freshly-initialised :class:`PAWNModel` at ``cfg``'s shape.

    ``key`` is either a :func:`jax.random.PRNGKey` / :func:`jax.random.key`
    output or a Python ``int`` (in which case ``jax.random.key(int)`` is
    called) — the int form is a convenience for tests and scripts.

    All weight tensors are drawn from ``Normal(0, 0.02)`` (the v1
    convention for ``dim > 1`` params). RMSNorm weights are
    one-initialised so the norm acts as identity at step 0;
    ``embed_pad`` is zero-initialised. ``decomp_table`` is built from
    the engine vocab. RoPE phase tables are *not* stored on the
    model — :meth:`PAWNModel.__call__` recomputes them per forward
    pass from ``cfg.head_dim`` / ``cfg.max_seq_len`` / ``cfg.rope_base``,
    and JIT constant-folds them when ``cfg`` is static.
    """
    if isinstance(key, int):
        key = jax.random.key(key)
    sub = jax.random.split(key, 12)
    d = cfg.d_model
    d_ff = cfg.d_ff
    n_layers = cfg.n_layers
    V = cfg.vocab_size  # noqa: N806

    layers = TransformerLayer(
        attn_norm_w=jnp.ones((n_layers, d), dtype=jnp.float32),
        wq=_normal_init(sub[0], (n_layers, d, d)),
        wk=_normal_init(sub[1], (n_layers, d, d)),
        wv=_normal_init(sub[2], (n_layers, d, d)),
        wo=_normal_init(sub[3], (n_layers, d, d)),
        ffn_norm_w=jnp.ones((n_layers, d), dtype=jnp.float32),
        w_gate=_normal_init(sub[4], (n_layers, d, d_ff)),
        w_up=_normal_init(sub[5], (n_layers, d, d_ff)),
        w_down=_normal_init(sub[6], (n_layers, d_ff, d)),
    )
    return PAWNModel(
        embed_src=_normal_init(sub[7], (64, d)),
        embed_dst=_normal_init(sub[8], (64, d)),
        embed_promo=_normal_init(sub[9], (5, d)),
        embed_pad=jnp.zeros((d,), dtype=jnp.float32),
        embed_outcome=_normal_init(sub[10], (cfg.n_outcomes, d)),
        layers=layers,
        final_norm_w=jnp.ones((d,), dtype=jnp.float32),
        lm_head=_normal_init(sub[11], (d, V)),
        decomp_table=_build_decomp_table(),
        cfg=cfg,
    )


def sliced(supernet_model: PAWNModel, variant_cfg: ModelConfig) -> PAWNModel:
    """Return a new :class:`PAWNModel` at the variant shape, taking the
    inner ``[:d_V, :d_V]`` block of every weight tensor.

    Raises :class:`pawn.config.NestingError` if ``variant_cfg`` doesn't
    nest under the supernet's config (so the slice would be undefined).

    The supernet is not mutated; this returns a fresh
    :class:`PAWNModel`. The decomposition table is reused as-is — it
    doesn't depend on width. RoPE phase tables are recomputed inside
    ``__call__`` per forward, so they don't need to be carried by the
    variant model.
    """
    validate_nested(variant_cfg, supernet_model.cfg)
    dv = variant_cfg.d_model
    dv_ff = variant_cfg.d_ff

    sup_layers = supernet_model.layers
    layers = TransformerLayer(
        attn_norm_w=sup_layers.attn_norm_w[:, :dv],
        wq=sup_layers.wq[:, :dv, :dv],
        wk=sup_layers.wk[:, :dv, :dv],
        wv=sup_layers.wv[:, :dv, :dv],
        wo=sup_layers.wo[:, :dv, :dv],
        ffn_norm_w=sup_layers.ffn_norm_w[:, :dv],
        w_gate=sup_layers.w_gate[:, :dv, :dv_ff],
        w_up=sup_layers.w_up[:, :dv, :dv_ff],
        w_down=sup_layers.w_down[:, :dv_ff, :dv],
    )
    return PAWNModel(
        embed_src=supernet_model.embed_src[:, :dv],
        embed_dst=supernet_model.embed_dst[:, :dv],
        embed_promo=supernet_model.embed_promo[:, :dv],
        embed_pad=supernet_model.embed_pad[:dv],
        embed_outcome=supernet_model.embed_outcome[:, :dv],
        layers=layers,
        final_norm_w=supernet_model.final_norm_w[:dv],
        lm_head=supernet_model.lm_head[:dv, :],
        decomp_table=supernet_model.decomp_table,
        cfg=variant_cfg,
    )
