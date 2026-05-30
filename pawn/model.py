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
- **Attention paths.** Three implementations share the per-layer
  ``_run_layers`` scan body, selected by call kwargs:
    * ``use_flash=True`` — :func:`jax.experimental.pallas.ops.gpu.attention.mha`,
      a Triton fused kernel. ~6× faster than the plain path at BASE
      T=512 on RDNA3 (gfx1100) and the default for training. PAD
      handling rides on Pallas ``segment_ids``; causal is
      unconditional.
    * ``use_sdpa=True`` — :func:`jax.nn.dot_product_attention` (XLA
      impl). Legacy fast path; superseded by Pallas on GPU.
    * Neither — plain materialised ``QK^T``. Bit-stable baseline used
      by the legacy converter's fp32 parity test and by CPU smoke
      runs (scripts auto-fall-back when ``jax.default_backend() !=
      "gpu"``).
  At seq=512, attention is roughly 12% of step FLOPs on plain — and
  the cliff that ate v2's win against v1 PyTorch before Pallas
  landed.
- **RMSNorm in the v2 cast order** — compute the norm in fp32, multiply
  by the weight in fp32, downcast at the very end. The v1 PyTorch
  layout downcast *between* the norm and the weight multiply. The two
  paths are bit-identical in fp32 (the difference vanishes when the
  intermediate dtype is already float32), so the legacy converter's
  fp32 parity test agrees on both; the v2 layout is one fewer cast and
  is the form the plan §5 calls out. See :func:`_rmsnorm` for details.
- **RoPE applied in fp32** to Q and K, then downcast — same reason.
- **SwiGLU FFN:** ``down(silu(gate(x)) * up(x))``.
- **Uniform token embeddings:** a single ``embed_tokens[V, d]`` table is
  gathered per token id — moves, PAD, outcomes, BOS, NULL, and the
  reserved control columns all index the same table. (v1 factored the
  move embedding into ``src + dst + promo`` lookups with PAD/outcome
  overrides; that aliased the Python-side control tokens onto the last
  outcome row, so v2 un-factors to one gather.)
- **Output head:** when ``cfg.tie_embeddings`` (the default) the model
  has **no** separate ``lm_head`` array — logits are
  ``x @ embed_tokens.T`` over the full vocabulary. Untied configs keep
  a standalone ``lm_head[d, V]``. Either way, argmax callers restrict to
  ``[0, NUM_ACTIONS)`` so PAD and outcome tokens can't be sampled — that
  restriction lives in :mod:`pawn.eval`, not here.

Save schema: the trainable arrays in declaration order — the
:func:`saved_fields` helper returns the canonical ordering for a given
``tie_embeddings`` (the tied schema omits ``lm_head``). The
non-trainable :attr:`PAWNModel.decomp_table` buffer is rebuilt at load
time from the engine vocabulary; RoPE phase tables are recomputed
inside the forward pass and never stored. Both stay out of the
safetensors payload.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Final, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from chess_engine import export_move_vocabulary
from jaxtyping import Array, Bool, Float, Int

from pawn.config import (
    NUM_ACTIONS,
    ModelConfig,
    validate_nested,
)

# Pallas flash attention. `jax.experimental.pallas.ops.gpu.attention.mha`
# is the Triton-flavoured fused attention bundled with JAX; it targets
# CUDA via Triton-CUDA and ROCm via Triton-ROCm. ~6× faster than the
# plain materialised QK^T path at B=16 T=512 H=8 D=64 bf16 on RDNA3
# (gfx1100). Backward is custom_vjp'd. Imported lazily inside
# :func:`_pallas_attn` so CPU-only environments (tests, parity) can
# still import :mod:`pawn.model` even when the Pallas GPU backend
# isn't loadable.

__all__ = [
    "PAWNModel",
    "TransformerLayer",
    "EffectiveCallable",
    "KVCacheCallable",
    "KVCache",
    "SAVED_FIELDS",
    "saved_fields",
    "init_model",
    "init_kv_cache",
    "sliced",
]


@runtime_checkable
class EffectiveCallable(Protocol):
    """The minimal call-signature the trainer and eval need from any
    "effective model" — :class:`PAWNModel` itself for weight-folding
    adapters (LoRA, sparse, FiLM, ...) or a wrapper module like
    :class:`pawn.adapters.bottleneck.BottleneckEffective` for adapters
    that inject post-sublayer residuals.

    Anything that exposes ``__call__(input_ids, attention_mask, *,
    compute_dtype) -> logits`` plus ``cfg`` satisfies this Protocol —
    runtime isinstance checks work because of ``@runtime_checkable``.
    The ``cfg`` property is what eval/generation paths read to size
    KV caches and choose seq lengths; both :class:`PAWNModel` (field)
    and :class:`BottleneckEffective` (property forwarding to backbone)
    expose it.
    """

    @property
    def cfg(self) -> "ModelConfig":
        # Property (read-only) lets implementations satisfy the
        # Protocol with either a dataclass-style field
        # (:class:`PAWNModel`, where ``cfg`` is an `eqx.field(static=True)`)
        # or a forwarded ``@property`` (:class:`BottleneckEffective`,
        # which exposes its backbone's cfg). A bare `cfg: ModelConfig`
        # attribute declaration would be mutable/invariant and reject
        # the property form under pyright.
        ...

    def __call__(
        self,
        input_ids: Int[Array, "B T"],
        attention_mask: Int[Array, "B T"] | None = None,
        *,
        compute_dtype: jnp.dtype | None = None,
        use_sdpa: bool = False,
        use_flash: bool = False,
    ) -> Float[Array, "B T V"]: ...


@runtime_checkable
class KVCacheCallable(EffectiveCallable, Protocol):
    """An :class:`EffectiveCallable` that also exposes the cached decode
    path. :class:`PAWNModel` and :class:`BottleneckEffective` qualify;
    weight-folded adapters whose ``apply_fn`` returns a fresh
    :class:`PAWNModel` qualify by inheritance. Callers use a
    ``hasattr`` check to detect at runtime — this Protocol gives the
    static-type surface for the branch where we know the cached path
    is available.
    """

    def forward_with_cache(
        self,
        input_ids: Int[Array, "B T_new"],
        cache: "KVCache",
        pos_start: Int[Array, ""] | int,
        *,
        compute_dtype: jnp.dtype | None = None,
    ) -> tuple[Float[Array, "B T_new V"], "KVCache"]: ...


# Names of the trainable arrays in safetensors declaration order. The
# tuple below is the untied superset (12 fields, ``lm_head`` last); the
# tied schema (11 fields) drops ``lm_head`` because the logits reuse
# ``embed_tokens`` via the transpose. Use :func:`saved_fields` to get the
# right list for a given ``tie_embeddings``; :data:`SAVED_FIELDS` is kept
# as the canonical declaration order for tooling that needs every field
# name. ``pawn.checkpoint`` imports both.
SAVED_FIELDS: Final[tuple[str, ...]] = (
    "embed_tokens",
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
assert len(SAVED_FIELDS) == 12, "model save schema (untied superset) must be 12 fields"


def saved_fields(tie_embeddings: bool) -> tuple[str, ...]:
    """Return the per-field save schema for a given ``tie_embeddings``.

    Tied models have no standalone ``lm_head`` array (logits reuse
    ``embed_tokens`` via the transpose), so the tied schema omits it.
    Declaration order is preserved so the safetensors payload stays
    deterministic.
    """
    if tie_embeddings:
        return tuple(name for name in SAVED_FIELDS if name != "lm_head")
    return SAVED_FIELDS


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
    pairs and rotate each pair by the angle ``freqs[t, i]``. Computes
    in the input dtype — the rope tables (which are built in fp32)
    are downcast to ``x.dtype`` at function entry. Earlier versions
    upcast ``x`` to fp32, did the rotation in fp32, then downcast at
    the end; the explicit upcast cost a full ``(B, H, T, d_head)``
    fp32 materialisation per Q + K call (×n_layers in the scan).
    Standard Llama-1/2/3 reference implementations apply RoPE in
    compute-dtype directly — the rotation is a unit-norm operation so
    the only precision concern is per-element accumulation, which at
    ``T<=512`` is well below bf16's noise floor.

    ``rope_cos`` / ``rope_sin`` must have the same ``T`` as ``x`` —
    callers that need a positional offset (KV-cached decode) should
    slice the full ``max_seq_len`` tables down to the active window
    before calling. See :meth:`PAWNModel.forward_with_cache`.

    fp32-mode callers (the legacy converter's parity test, eval, the
    KV-cached generation path when ``compute_dtype`` is ``None``)
    still pay no precision cost — ``x.dtype`` is fp32 there, so the
    cast on the rope tables is a no-op and the rotation stays fp32
    end-to-end.
    """
    cos = rope_cos.astype(x.dtype)
    sin = rope_sin.astype(x.dtype)
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    x0 = pairs[..., 0]
    x1 = pairs[..., 1]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return jnp.stack([out0, out1], axis=-1).reshape(x.shape)


def _pallas_attn(
    q_bhtd: Float[Array, "B H T d"],
    k_bhtd: Float[Array, "B H T d"],
    v_bhtd: Float[Array, "B H T d"],
    attention_mask: Int[Array, "B T"] | None,
    inv_scale: float,
) -> Float[Array, "B T HD"]:
    """Pallas flash attention.

    ``q``/``k``/``v`` come in the ``(B, H, T, d_head)`` layout used by
    the rest of the transformer block; this helper transposes to
    ``(B, T, H, d_head)`` for the Pallas kernel and flattens the
    head/dim axes on the way out so the caller can pass straight into
    the output projection.

    PAD handling: deliberately **not** passed to the kernel. The Rust
    engine emits strictly right-padded sequences (PAD tokens only at
    the tail), so causal attention already prevents real tokens from
    attending to PAD positions — every real query at position ``p``
    only sees positions ``[0, p]``, all of which are real. PAD queries
    *do* attend to real keys, but their outputs are weighted to zero
    by ``batch.loss_mask`` downstream and contribute nothing to the
    gradient. Passing ``segment_ids`` here would force Pallas onto a
    masked-attention codepath that does extra per-block segment-mask
    compute every kv block for no behavioural benefit; dropping it is
    a free ~0.5-3% win on the production-shape benches. Non-
    right-padded callers (rare, only some future inference paths)
    must use ``use_flash=False`` to opt into the explicit-mask plain
    path.
    """
    del attention_mask  # see docstring — causal+right-pad makes this redundant
    from jax.experimental.pallas.ops.gpu.attention import mha as _pl_mha

    q_bthd = q_bhtd.transpose(0, 2, 1, 3)
    k_bthd = k_bhtd.transpose(0, 2, 1, 3)
    v_bthd = v_bhtd.transpose(0, 2, 1, 3)
    # `mha` is a `jax.custom_vjp` wrapper, which pyright surfaces as an
    # opaque `object` — annotate so downstream `.shape` / `.reshape`
    # type-check. The runtime contract is well-defined: same dtype and
    # leading dims as the inputs, with the H/D axes preserved.
    out_bthd: jax.Array = _pl_mha(
        q_bthd, k_bthd, v_bthd,
        segment_ids=None,
        sm_scale=float(inv_scale),
        causal=True,
    )
    B, T, H, D = out_bthd.shape  # noqa: N806
    return out_bthd.reshape(B, T, H * D)


# ---------------------------------------------------------------------------
# Static buffers
# ---------------------------------------------------------------------------


def _build_decomp_table() -> Int[Array, "n_actions 3"]:
    """Build the (src, dst, promo) lookup table from the engine vocab.

    Each move token in ``[0, NUM_ACTIONS)`` decomposes into a source
    square (0–63), destination square (0–63), and promotion type
    (0=none, 1=q, 2=r, 3=b, 4=n).

    This table is **not** used by the (un-factored) embedding path —
    :meth:`PAWNModel._embed` gathers ``embed_tokens`` directly. It's
    retained as a non-trainable buffer for downstream consumers (eval /
    diagnostics) that decode a move token back into its squares.
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


class KVCache(eqx.Module):
    """Per-layer Key/Value cache for autoregressive decoding.

    Both arrays share the layout ``(n_layers, B, n_heads, T_max,
    head_dim)`` — leading ``n_layers`` axis so :func:`jax.lax.scan`
    over the transformer stack can read one layer's slice from each
    leaf. ``T_max`` is the cache capacity (typically
    ``cfg.max_seq_len``); the active prefix length is tracked by the
    caller (a Python ``int`` or a ``jnp`` scalar) so the JIT key
    doesn't depend on the position.

    The cache lives outside the model — :func:`init_kv_cache` allocates
    a fresh one, and :meth:`PAWNModel.forward_with_cache` returns a
    new ``KVCache`` with the new K/V slices written in.
    """

    k: Float[Array, "n_layers B H T_max d"]
    v: Float[Array, "n_layers B H T_max d"]


def init_kv_cache(
    cfg: ModelConfig,
    batch_size: int,
    max_seq_len: int | None = None,
    dtype: jnp.dtype | None = None,
) -> KVCache:
    """Allocate a zero-initialised :class:`KVCache` for ``cfg``.

    ``max_seq_len`` defaults to ``cfg.max_seq_len``; pass a smaller
    value for diagnostics that only generate short games (the cache
    allocation is ``2 * n_layers * B * n_heads * T_max * head_dim``
    floats — at the production supernet shape and ``T_max=512`` that's
    ~2.6 GB per game-batch in fp32 vs ~1.3 GB in bf16).

    ``dtype`` defaults to ``float32`` because the precision contract
    is: *the cache must hold at least as much precision as the
    surrounding forward's compute_dtype*. With a fp32 forward and a
    bf16 cache, the K/V values get downcast on write and lose
    precision; the bit-stable parity test
    (`tests/test_jax_model.py::test_forward_with_cache_matches_full_forward_one_shot`)
    would fail. Production callers running a bf16 forward should pass
    ``dtype=jnp.bfloat16`` explicitly — `autoregressive_generate` and
    similar paths do this when they're driving inference at
    ``n_per_outcome=1000`` scale. Round-1 perf review flagged the
    cache size as the dominant memory consumer at production scale,
    so the bf16 path is what unlocks the long-prefix decode budget.
    """
    if max_seq_len is None:
        max_seq_len = cfg.max_seq_len
    if dtype is None:
        dtype = jnp.float32
    shape = (cfg.n_layers, batch_size, cfg.n_heads, max_seq_len, cfg.head_dim)
    return KVCache(
        k=jnp.zeros(shape, dtype=dtype),
        v=jnp.zeros(shape, dtype=dtype),
    )


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
    schema. The class lays out the trainable array fields (declaration
    order matching :func:`saved_fields`), one non-trainable
    int32 buffer (:attr:`decomp_table` — filtered out automatically by
    ``eqx.is_inexact_array``), and one static ``cfg`` reference. RoPE
    phase tables are not stored on the model; they're recomputed
    inside :meth:`__call__` per forward call and constant-folded by
    JIT when ``cfg`` is static.

    ``lm_head`` is ``None`` when ``cfg.tie_embeddings`` (the default) —
    logits then reuse ``embed_tokens`` via the transpose. ``None`` is an
    empty PyTree subtree, so a tied model carries one fewer trainable
    leaf than an untied one and the optimizer never sees a phantom head.
    """

    # Trainable fields — declaration order = save order (see
    # :func:`saved_fields`). ``lm_head`` is ``None`` for tied models.
    embed_tokens: Float[Array, "V d"]
    layers: TransformerLayer
    final_norm_w: Float[Array, "d"]
    lm_head: Float[Array, "d V"] | None

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
        *,
        compute_dtype: jnp.dtype | None = None,
        attn_hook: Callable[[Float[Array, "B T d"], Any], Float[Array, "B T d"]]
        | None = None,
        ffn_hook: Callable[[Float[Array, "B T d"], Any], Float[Array, "B T d"]]
        | None = None,
        hook_data: Any = None,
        use_sdpa: bool = False,
        use_flash: bool = False,
    ) -> Float[Array, "B T V"]:
        """Forward pass.

        ``input_ids`` is ``(batch, seq)`` int32 token IDs (move tokens,
        PAD, or outcome tokens). ``attention_mask`` is ``1`` for real
        tokens and ``0`` for PAD; if omitted, a fully-real mask is
        assumed.

        ``compute_dtype`` selects the mixed-precision *forward-activation*
        dtype (plan §5). The master parameters stay fp32 — they're cast
        per einsum just before the matmul, then XLA folds the cast into
        the kernel where it can. Reductions (RMSNorm, softmax) upcast
        to fp32 internally and downcast back, matching the standard
        "weights fp32 / compute bf16 / accumulate fp32" recipe. The
        final logits are returned in fp32 for downstream loss
        stability. ``None`` (default) keeps the whole forward in fp32 —
        used by tests, eval, and the parity test that compares
        bit-exact against v1 in fp32.

        ``attn_hook`` / ``ffn_hook`` / ``hook_data`` are the adapter
        injection points. When non-None, the scan body calls
        ``h = attn_hook(h, hook_slice)`` after the attention sublayer's
        residual addition and ``h = ffn_hook(h, hook_slice)`` after the
        FFN sublayer's residual. ``hook_data`` is a PyTree (or None)
        with a leading ``n_layers`` axis on every leaf; the scan
        zip-iterates ``(layer, hook_slice)`` so the hooks see the
        per-layer slice. Used by :mod:`pawn.adapters.bottleneck` to add
        Houlsby residual MLPs after each sublayer; other adapters
        (LoRA, sparse) fold corrections into the weight tensors instead
        and leave the hooks unset.

        ``use_sdpa`` switches the attention block from the plain
        materialised-``QK^T`` path to :func:`jax.nn.dot_product_attention`
        (XLA implementation), which fuses Q@K, scale, mask, softmax,
        and attn@V into one kernel. The plan §5 marked SDPA out of
        scope for the framework swap citing fused-kernel maturity on
        JAX-on-ROCm; in practice ``implementation='xla'`` works on
        recent ROCm + jaxlib (verified bit-identical to the plain
        path within fp32 noise — max diff 2.4e-7 on (B=2, H=4,
        T=32, D=16) random q/k/v). Off by default to preserve the
        established bit-stable baseline; opt in for perf-sensitive
        runs.

        ``use_flash`` routes the attention block through
        :func:`jax.experimental.pallas.ops.gpu.attention.mha` — a
        Triton-flavoured fused attention kernel. ~6× faster than the
        plain path at BASE T=512 on RDNA3 (gfx1100). Requires a
        GPU backend; trainers set this from the run config after
        resolving the backend (CPU smoke runs auto-fall-back). Wins
        over ``use_sdpa`` when both are eligible; the XLA SDPA path on
        ROCm is slower than plain for our shapes.

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
        with jax.named_scope("embed"):
            x = self._embed(input_ids)
            if compute_dtype is not None:
                x = x.astype(compute_dtype)
        # RoPE tables are recomputed per call — constant-folded by JIT
        # under a static cfg, so the cost is one trace-time build.
        rope_cos, rope_sin = _build_rope(self.cfg.head_dim, T, self.cfg.rope_base)
        # The materialised ``(B, 1, T, T)`` mask is only consumed by the
        # plain and SDPA paths; the Pallas-flash path uses
        # ``segment_ids`` derived directly from ``attention_mask`` and
        # ignores ``mask``. Build it lazily so XLA can DCE it for free
        # (a 16 MB bool tensor at B=64 T=512 that would otherwise be
        # threaded through the scan as a loop-invariant capture).
        mask: Bool[Array, "B 1 T T"] | None
        if use_flash:
            mask = None
        else:
            causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
            if attention_mask is None:
                mask = causal[None, None, :, :]  # (1, 1, T, T)
            else:
                pad = attention_mask.astype(jnp.bool_)[:, None, None, :]  # (B, 1, 1, T)
                mask = causal[None, None, :, :] & pad
        with jax.named_scope("transformer_layers"):
            x = self._run_layers(
                x, rope_cos, rope_sin, mask, attention_mask, compute_dtype,
                attn_hook=attn_hook, ffn_hook=ffn_hook, hook_data=hook_data,
                use_sdpa=use_sdpa, use_flash=use_flash,
            )
        with jax.named_scope("final_norm"):
            x = _rmsnorm(x, self.final_norm_w)
        # `_rmsnorm` returns in `x.dtype` (compute dtype if set). Cast the
        # head to match. The trailing fp32 cast on `logits` was a
        # ~520 MB/step HBM bandwidth tax in the bf16 training path
        # (materialised a full ``(B, T, V)`` fp32 tensor, then
        # ``log_softmax`` materialised another). The training loss
        # (``pawn.trainer.cross_entropy_loss``) handles the fp32 cast
        # inside the fused ``optax.softmax_cross_entropy_with_integer_labels``
        # call, so we only upcast here for fp32-mode callers (legacy
        # parity test, eval, probes) — i.e. when ``compute_dtype is
        # None``, the einsum result is already fp32 and the cast is a
        # no-op.
        with jax.named_scope("lm_head"):
            logits = jnp.einsum("btd,dv->btv", x, self._head_weight(compute_dtype))
        if compute_dtype is None:
            return logits.astype(jnp.float32)
        return logits

    def hidden_states(
        self,
        input_ids: Int[Array, "B T"],
        attention_mask: Int[Array, "B T"] | None = None,
    ) -> Float[Array, "L1 B T d"]:
        """Per-layer residual-stream hidden states for linear probing.

        Returns a stack of shape ``(n_layers + 1, B, T, d)`` in fp32: index
        ``0`` is the post-embedding residual stream and index ``i`` (for
        ``i in 1..n_layers``) is the output of transformer block ``i`` (its
        post-FFN residual). The final entry is *not* run through
        :attr:`final_norm_w` — probes read the raw per-layer residual
        stream, which is the standard probing target.

        The forward runs in fp32 (``compute_dtype=None``) and through the
        plain materialised-``QK^T`` attention path so the extracted states
        are precision-stable and backend-agnostic; the probe trainer keeps
        the backbone frozen, so no gradient flows here.
        """
        T = input_ids.shape[-1]
        if T > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds cfg.max_seq_len {self.cfg.max_seq_len}"
            )
        x = self._embed(input_ids)
        rope_cos, rope_sin = _build_rope(self.cfg.head_dim, T, self.cfg.rope_base)
        causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        if attention_mask is None:
            mask = causal[None, None, :, :]
        else:
            pad = attention_mask.astype(jnp.bool_)[:, None, None, :]
            mask = causal[None, None, :, :] & pad
        per_layer = self._run_layers_collect(
            x, rope_cos, rope_sin, mask, attention_mask,
        )
        # Prepend the post-embedding stream so callers can probe the input
        # representation (layer 0) alongside every block output.
        return jnp.concatenate([x[None], per_layer], axis=0)

    # -----------------------------------------------------------------------
    # Forward-pass internals
    # -----------------------------------------------------------------------

    def _embed(self, input_ids: Int[Array, "B T"]) -> Float[Array, "B T d"]:
        """Uniform token embedding: a single gather into ``embed_tokens``.

        Every token id — moves, PAD, outcomes, BOS, NULL, and the
        reserved control columns — indexes the same ``[V, d]`` table.
        No clamping or override branches: ids are in ``[0, V)`` by
        construction (the prefix assembler + engine emission stay within
        the vocab), so the gather is in-bounds for all of them.
        """
        ids = input_ids.astype(jnp.int32)
        return self.embed_tokens[ids]

    def _head_weight(
        self, compute_dtype: jnp.dtype | None
    ) -> Float[Array, "d V"]:
        """Resolve the output-projection weight, honouring weight tying.

        Tied (``lm_head is None``): the head is ``embed_tokens.T`` —
        logits = ``x @ embed_tokens.T``. Untied: the standalone
        ``lm_head[d, V]``. Cast to ``compute_dtype`` when AMP is on so
        the einsum runs in compute dtype (XLA fuses the cast into the
        matmul); ``None`` keeps it fp32 for the parity / eval / probe
        callers.
        """
        head = self.embed_tokens.T if self.lm_head is None else self.lm_head
        if compute_dtype is not None:
            return head.astype(compute_dtype)
        return head

    def _run_layers(
        self,
        x: Float[Array, "B T d"],
        rope_cos: Float[Array, "T half"],
        rope_sin: Float[Array, "T half"],
        mask: Bool[Array, "B 1 T T"] | None,
        attention_mask: Int[Array, "B T"] | None,
        compute_dtype: jnp.dtype | None = None,
        *,
        attn_hook: Callable[[Float[Array, "B T d"], Any], Float[Array, "B T d"]]
        | None = None,
        ffn_hook: Callable[[Float[Array, "B T d"], Any], Float[Array, "B T d"]]
        | None = None,
        hook_data: Any = None,
        use_sdpa: bool = False,
        use_flash: bool = False,
    ) -> Float[Array, "B T d"]:
        """Apply all ``n_layers`` transformer blocks via :func:`jax.lax.scan`.

        ``self.layers`` is one :class:`TransformerLayer` whose leaves
        have a leading ``n_layers`` axis. :func:`jax.lax.scan` iterates
        layer-by-layer; each iteration sees a per-layer slice of every
        weight tensor.

        ``compute_dtype`` controls activation precision. When set, each
        per-layer weight is cast to ``compute_dtype`` just before its
        einsum — XLA fuses the cast into the matmul kernel where it
        can. `_rmsnorm` and `softmax` upcast to fp32 internally for
        numerical stability and downcast back.

        ``attn_hook`` / ``ffn_hook`` / ``hook_data`` inject adapter
        residuals after each sublayer; see :meth:`__call__` for the
        contract.
        """
        head_dim = self.cfg.head_dim
        n_heads = self.cfg.n_heads
        # Compile-time constant — hoist out of the scan body so we don't
        # re-allocate a 0-d scalar and dispatch a sqrt kernel on every layer.
        inv_scale = head_dim ** -0.5
        has_hooks = (
            attn_hook is not None or ffn_hook is not None
        ) and hook_data is not None

        def step(
            carry: Float[Array, "B T d"],
            layer_and_hook: Any,
        ) -> tuple[Float[Array, "B T d"], None]:
            if has_hooks:
                layer, hook_slice = layer_and_hook
            else:
                layer = layer_and_hook
                hook_slice = None
            h = carry
            # ---- attention block (pre-norm + residual) ----
            # `_rmsnorm` upcasts to fp32 internally and downcasts to
            # `h.dtype` — so `normed` is the compute dtype when AMP is on.
            with jax.named_scope("attn_norm"):
                normed = _rmsnorm(h, layer.attn_norm_w)
            B, T, D = normed.shape  # noqa: N806
            # Cast Q/K/V/O weights to compute_dtype just before each
            # einsum. XLA fuses the cast into the kernel; the master
            # weight stays fp32 in `self.layers.wq` etc., so backward
            # accumulates in fp32 via standard JAX autograd.
            #
            # Note: a QKV-pack variant (concat W along the out axis,
            # one bigger matmul, jnp.split after) was tested empirically
            # and is shape-dependent — wins ~2% at BASE B=64 but **loses
            # ~6% at LARGE B=64** on RTX 5090. The supernet trains at
            # LARGE shape, so the unpacked 3-matmul form ships.
            wq = layer.wq if compute_dtype is None else layer.wq.astype(compute_dtype)
            wk = layer.wk if compute_dtype is None else layer.wk.astype(compute_dtype)
            wv = layer.wv if compute_dtype is None else layer.wv.astype(compute_dtype)
            wo = layer.wo if compute_dtype is None else layer.wo.astype(compute_dtype)
            with jax.named_scope("qkv_proj"):
                q = jnp.einsum("btd,de->bte", normed, wq)
                k = jnp.einsum("btd,de->bte", normed, wk)
                v = jnp.einsum("btd,de->bte", normed, wv)
                q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
                k = k.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
                v = v.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
            with jax.named_scope("rope"):
                q = _apply_rope(q, rope_cos, rope_sin)
                k = _apply_rope(k, rope_cos, rope_sin)
            if use_flash:
                # Pallas Triton-flavoured fused attention; consumes the
                # (B, H, T, D) tensors after RoPE and emits (B, T, D)
                # ready for the output projection. PAD-masking comes
                # from `attention_mask` via segment_ids inside
                # `_pallas_attn`; causal is unconditional.
                with jax.named_scope("attn_pallas"):
                    attn_out = _pallas_attn(q, k, v, attention_mask, inv_scale)
            elif use_sdpa:
                # `jax.nn.dot_product_attention` expects (B, T, H, D)
                # layout, not the (B, H, T, D) we computed above. The
                # transpose is free under XLA fusion. SDPA's `mask`
                # argument is broadcastable to (B, H, T, T) — the
                # existing `mask` already has shape (B, 1, T, T) which
                # broadcasts cleanly. `is_causal=False` because we pass
                # an explicit mask that already encodes causality + the
                # per-batch PAD mask.
                q_bthd = q.transpose(0, 2, 1, 3)
                k_bthd = k.transpose(0, 2, 1, 3)
                v_bthd = v.transpose(0, 2, 1, 3)
                attn_out = jax.nn.dot_product_attention(
                    q_bthd, k_bthd, v_bthd,
                    mask=mask, scale=inv_scale,
                    implementation="xla",
                )
                # SDPA returns (B, T, H, D); flatten back to (B, T, D).
                attn_out = attn_out.reshape(B, T, D)
            else:
                # Attention scores: matmul in compute dtype, then upcast to
                # fp32 for the softmax (the fp32 score tensor is the
                # numerically-sensitive intermediate). Downcast attn weights
                # back to compute dtype for the value matmul. ``mask`` is
                # only ``None`` on the ``use_flash`` path (handled above) —
                # narrow it for pyright.
                assert mask is not None
                scores = jnp.einsum("bhid,bhjd->bhij", q, k) * inv_scale
                scores_f32 = scores.astype(jnp.float32)
                mask_neg_inf = jnp.finfo(jnp.float32).min
                scores_f32 = jnp.where(mask, scores_f32, mask_neg_inf)
                attn = jax.nn.softmax(scores_f32, axis=-1)
                if compute_dtype is not None:
                    attn = attn.astype(compute_dtype)
                attn_out = jnp.einsum("bhij,bhjd->bhid", attn, v)
                attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, D)
            with jax.named_scope("attn_out_proj"):
                h = h + jnp.einsum("btd,de->bte", attn_out, wo)
            if attn_hook is not None:
                h = attn_hook(h, hook_slice)

            # ---- ffn block (pre-norm + residual) ----
            with jax.named_scope("ffn_norm"):
                normed = _rmsnorm(h, layer.ffn_norm_w)
            # Gate+up pack variant (concat W along out axis, one bigger
            # matmul, jnp.split after) was tested and behaves like the
            # QKV-pack experiment above — wins at BASE but loses at
            # LARGE. Keeping unpacked for production parity.
            w_gate = (
                layer.w_gate
                if compute_dtype is None
                else layer.w_gate.astype(compute_dtype)
            )
            w_up = (
                layer.w_up
                if compute_dtype is None
                else layer.w_up.astype(compute_dtype)
            )
            w_down = (
                layer.w_down
                if compute_dtype is None
                else layer.w_down.astype(compute_dtype)
            )
            with jax.named_scope("ffn_gate_up"):
                gate = jnp.einsum("btd,df->btf", normed, w_gate)
                up = jnp.einsum("btd,df->btf", normed, w_up)
            with jax.named_scope("ffn_down"):
                ffn_out = jnp.einsum("btf,fd->btd", jax.nn.silu(gate) * up, w_down)
            h = h + ffn_out
            if ffn_hook is not None:
                h = ffn_hook(h, hook_slice)
            return h, None

        scan_input: Any = (self.layers, hook_data) if has_hooks else self.layers
        # Unrolling the layer loop trades HLO size for cross-layer
        # fusion (e.g., layer-i residual add into layer-i+1 RMSNorm
        # read) — the same kind of cross-iteration fusion that
        # `torch.compile`'s Inductor gets from an explicit Python
        # for-loop in v1. Full ``unroll=n_layers`` is best at
        # compile-bound shapes; partial unrolls (2 or 4) leave more
        # opportunity for XLA to schedule activation memory tighter,
        # which helps at high batch / long seq where activation
        # checkpoints dominate. ``PAWN_SCAN_UNROLL`` overrides for
        # benching; leaving it unset keeps the default (full unroll).
        unroll_str = os.environ.get("PAWN_SCAN_UNROLL")
        unroll = (
            int(unroll_str) if unroll_str else self.cfg.n_layers
        )
        # ``PAWN_USE_REMAT=1`` wraps the per-layer ``step`` body in
        # ``jax.checkpoint`` (with the dot-no-batch-dims policy that
        # saves matmul outputs and recomputes RMSNorm / RoPE /
        # residuals on backward). Trades ~18% backward FLOPs for ~3-5×
        # activation memory headroom — the only way to fit B=256 at
        # LARGE inside the 5090's 32 GB VRAM. Validated empirically:
        # round-3 review (Sonnet conv + Opus conv + Opus OOB) flagged
        # it as the largest-impact remaining lever specifically
        # because it unlocks larger batches, not because it makes
        # B=64 faster.
        if os.environ.get("PAWN_USE_REMAT"):
            step = jax.checkpoint(  # type: ignore[assignment]
                step,
                policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
            )
        x, _ = jax.lax.scan(step, x, scan_input, unroll=unroll)
        return x

    def _run_layers_collect(
        self,
        x: Float[Array, "B T d"],
        rope_cos: Float[Array, "T half"],
        rope_sin: Float[Array, "T half"],
        mask: Bool[Array, "B 1 T T"],
        attention_mask: Int[Array, "B T"] | None,
    ) -> Float[Array, "L B T d"]:
        """Per-layer output stack — the probe-only sibling of
        :meth:`_run_layers`.

        Runs the plain fp32 materialised-``QK^T`` attention path (no AMP,
        no adapter hooks, no SDPA/Pallas) and emits the post-FFN residual
        of every block via :func:`jax.lax.scan`'s ``ys`` channel, giving a
        ``(n_layers, B, T, d)`` stack. Kept separate from
        :meth:`_run_layers` so the hot training/eval path never threads an
        extra output through its scan; the precision and masking exactly
        mirror the ``compute_dtype is None`` branch of :meth:`_run_layers`,
        so probe states match what the final-logit forward sees.
        """
        head_dim = self.cfg.head_dim
        n_heads = self.cfg.n_heads
        inv_scale = head_dim ** -0.5

        def step(
            carry: Float[Array, "B T d"],
            layer: TransformerLayer,
        ) -> tuple[Float[Array, "B T d"], Float[Array, "B T d"]]:
            h = carry
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
            mask_neg_inf = jnp.finfo(jnp.float32).min
            scores = jnp.where(mask, scores.astype(jnp.float32), mask_neg_inf)
            attn = jax.nn.softmax(scores, axis=-1)
            attn_out = jnp.einsum("bhij,bhjd->bhid", attn, v)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, D)
            h = h + jnp.einsum("btd,de->bte", attn_out, layer.wo)
            normed = _rmsnorm(h, layer.ffn_norm_w)
            gate = jnp.einsum("btd,df->btf", normed, layer.w_gate)
            up = jnp.einsum("btd,df->btf", normed, layer.w_up)
            ffn_out = jnp.einsum("btf,fd->btd", jax.nn.silu(gate) * up, layer.w_down)
            h = h + ffn_out
            return h, h

        _, per_layer = jax.lax.scan(step, x, self.layers)
        return per_layer

    # -----------------------------------------------------------------------
    # KV-cached generation path
    # -----------------------------------------------------------------------

    def forward_with_cache(
        self,
        input_ids: Int[Array, "B T_new"],
        cache: KVCache,
        pos_start: Int[Array, ""] | int,
        *,
        compute_dtype: jnp.dtype | None = None,
        attn_hook: Callable[[Float[Array, "B T d"], Any], Float[Array, "B T d"]]
        | None = None,
        ffn_hook: Callable[[Float[Array, "B T d"], Any], Float[Array, "B T d"]]
        | None = None,
        hook_data: Any = None,
        use_sdpa: bool = False,
    ) -> tuple[Float[Array, "B T_new V"], KVCache]:
        """Cached forward pass for autoregressive decoding.

        Mirrors :meth:`__call__` but writes the freshly-computed K/V
        into ``cache`` at slots ``[pos_start, pos_start + T_new)`` and
        attends over the full cache window
        ``[0, pos_start + T_new)``. ``T_new`` can be ``1`` for the
        common decode step or larger for a prefill chunk; ``pos_start``
        is a Python ``int`` or a ``jnp`` scalar (use the latter when
        you want one JIT trace across many decode positions —
        :func:`jax.lax.dynamic_update_slice` handles both).

        Returns ``(logits, new_cache)``. ``logits`` is ``(B, T_new,
        V)``; callers that only want the next-token prediction take
        ``logits[:, -1, :]``. The returned cache is a fresh
        :class:`KVCache` (functional update) with the new slices
        written; the input cache is unchanged.

        Math invariant: the logits returned by ``forward_with_cache``
        at position ``pos_start`` are bit-identical (modulo XLA
        kernel-ordering noise) to
        ``model(input_ids[:, :pos_start + T_new])[:, pos_start:]``.
        The KV-cache test in ``tests/test_jax_kv_cache.py`` pins this.

        The cached path uses plain attention always (no SDPA fallback)
        because :func:`jax.nn.dot_product_attention` requires symmetric
        ``T_q == T_kv`` layouts; ``use_sdpa`` is accepted but ignored.
        Adapter hooks (``attn_hook`` / ``ffn_hook`` / ``hook_data``)
        are threaded through exactly as in the full-forward path.
        """
        del use_sdpa  # cached path is plain-attention only; flag accepted
        # for caller API parity but not honoured.
        T_new = input_ids.shape[-1]  # noqa: N806
        T_max = cache.k.shape[3]  # noqa: N806
        if T_new > T_max:
            raise ValueError(
                f"forward_with_cache input length {T_new} exceeds cache "
                f"capacity {T_max}"
            )
        # `lax.dynamic_update_slice` *silently clamps* the write start
        # so the slice fits inside the destination buffer —
        # `pos_start + T_new > T_max` would corrupt the cache instead
        # of raising. Use `eqx.error_if` so the guard fires uniformly
        # for Python int, numpy integer, and JAX-traced scalar
        # `pos_start` (round-1 + round-2 codex P2 / bug-detector
        # IMPORTANT). The error is raised at runtime — for jitted
        # callers that's after compile, for eager callers it's
        # immediate.
        pos_start_arr = jnp.asarray(pos_start, dtype=jnp.int32)
        end = pos_start_arr + jnp.int32(T_new)
        input_ids = eqx.error_if(
            input_ids,
            end > jnp.int32(T_max),
            f"forward_with_cache write window exceeds cache capacity "
            f"{T_max} (input length {T_new} starting at pos_start)",
        )

        x = self._embed(input_ids)
        if compute_dtype is not None:
            x = x.astype(compute_dtype)

        head_dim = self.cfg.head_dim
        n_heads = self.cfg.n_heads
        inv_scale = head_dim ** -0.5
        has_hooks = (
            attn_hook is not None or ffn_hook is not None
        ) and hook_data is not None

        # RoPE: build the full table once and slice the active window.
        rope_cos_full, rope_sin_full = _build_rope(
            head_dim, T_max, self.cfg.rope_base
        )

        # Absolute query positions for the new chunk; reused for the
        # RoPE slice and the causal mask. Round-1 simplification flagged
        # the prior code as recomputing the same array under two names
        # (pos_arr / absolute_q).
        absolute_q = pos_start + jnp.arange(T_new, dtype=jnp.int32)
        rope_cos = rope_cos_full[absolute_q]  # (T_new, half)
        rope_sin = rope_sin_full[absolute_q]

        # Causal + cache-window mask: q at offset i (absolute position
        # pos_start + i) attends to absolute positions [0, pos_start +
        # i + 1). Built explicitly so it works with both Python int and
        # jnp scalar pos_start.
        kv_positions = jnp.arange(T_max, dtype=jnp.int32)  # (T_max,)
        # mask[i, j] = (kv_positions[j] <= absolute_q[i])
        attn_mask_2d = kv_positions[None, :] <= absolute_q[:, None]
        # Broadcast to (1, 1, T_new, T_max) so it can be combined with
        # the per-batch PAD mask if one is added later. We don't take a
        # PAD mask here because autoregressive decode emits one token
        # per step and feeds it back — there are no PAD positions in
        # the cache window during decode.
        attn_mask = attn_mask_2d[None, None, :, :]

        def step(
            carry: Float[Array, "B T_new d"],
            layer_kv_hook: Any,
        ) -> tuple[
            Float[Array, "B T_new d"],
            tuple[
                Float[Array, "B H T_max d"],
                Float[Array, "B H T_max d"],
            ],
        ]:
            if has_hooks:
                layer, layer_k, layer_v, hook_slice = layer_kv_hook
            else:
                layer, layer_k, layer_v = layer_kv_hook
                hook_slice = None
            h = carry
            normed = _rmsnorm(h, layer.attn_norm_w)
            B, T_n, D = normed.shape  # noqa: N806
            wq = layer.wq if compute_dtype is None else layer.wq.astype(compute_dtype)
            wk = layer.wk if compute_dtype is None else layer.wk.astype(compute_dtype)
            wv = layer.wv if compute_dtype is None else layer.wv.astype(compute_dtype)
            wo = layer.wo if compute_dtype is None else layer.wo.astype(compute_dtype)
            q = jnp.einsum("btd,de->bte", normed, wq)
            k_new = jnp.einsum("btd,de->bte", normed, wk)
            v_new = jnp.einsum("btd,de->bte", normed, wv)
            q = q.reshape(B, T_n, n_heads, head_dim).transpose(0, 2, 1, 3)
            k_new = k_new.reshape(B, T_n, n_heads, head_dim).transpose(0, 2, 1, 3)
            v_new = v_new.reshape(B, T_n, n_heads, head_dim).transpose(0, 2, 1, 3)
            q = _apply_rope(q, rope_cos, rope_sin)
            k_new = _apply_rope(k_new, rope_cos, rope_sin)

            # Write k_new / v_new into the layer cache at pos_start.
            # The cache leaves are (B, H, T_max, D); the start indices
            # are (0, 0, pos_start, 0). `dynamic_update_slice` accepts
            # either Python int or jnp scalar at any axis.
            #
            # Match the cache dtype so the update doesn't widen the
            # cache: this is only relevant when compute_dtype != cache
            # dtype (e.g. cache fp32, compute bf16) — without the
            # explicit cast we'd hit a dtype-mismatch error inside
            # dynamic_update_slice.
            cache_dtype = layer_k.dtype
            k_to_write = k_new.astype(cache_dtype)
            v_to_write = v_new.astype(cache_dtype)
            zero = jnp.int32(0)
            pos_int = jnp.asarray(pos_start, dtype=jnp.int32)
            new_layer_k = jax.lax.dynamic_update_slice(
                layer_k, k_to_write, (zero, zero, pos_int, zero),
            )
            new_layer_v = jax.lax.dynamic_update_slice(
                layer_v, v_to_write, (zero, zero, pos_int, zero),
            )

            # Attention: q (B, H, T_new, d) against new_layer_{k,v} (B,
            # H, T_max, d). Cast cache to compute dtype before the
            # matmul so the einsum kernel runs in compute dtype.
            k_for_attn = (
                new_layer_k
                if compute_dtype is None
                else new_layer_k.astype(compute_dtype)
            )
            v_for_attn = (
                new_layer_v
                if compute_dtype is None
                else new_layer_v.astype(compute_dtype)
            )
            scores = jnp.einsum("bhid,bhjd->bhij", q, k_for_attn) * inv_scale
            scores_f32 = scores.astype(jnp.float32)
            mask_neg_inf = jnp.finfo(jnp.float32).min
            scores_f32 = jnp.where(attn_mask, scores_f32, mask_neg_inf)
            attn = jax.nn.softmax(scores_f32, axis=-1)
            if compute_dtype is not None:
                attn = attn.astype(compute_dtype)
            attn_out = jnp.einsum("bhij,bhjd->bhid", attn, v_for_attn)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T_n, D)
            h = h + jnp.einsum("btd,de->bte", attn_out, wo)
            if attn_hook is not None:
                h = attn_hook(h, hook_slice)

            normed = _rmsnorm(h, layer.ffn_norm_w)
            w_gate = (
                layer.w_gate
                if compute_dtype is None
                else layer.w_gate.astype(compute_dtype)
            )
            w_up = (
                layer.w_up
                if compute_dtype is None
                else layer.w_up.astype(compute_dtype)
            )
            w_down = (
                layer.w_down
                if compute_dtype is None
                else layer.w_down.astype(compute_dtype)
            )
            gate = jnp.einsum("btd,df->btf", normed, w_gate)
            up = jnp.einsum("btd,df->btf", normed, w_up)
            ffn_out = jnp.einsum("btf,fd->btd", jax.nn.silu(gate) * up, w_down)
            h = h + ffn_out
            if ffn_hook is not None:
                h = ffn_hook(h, hook_slice)
            return h, (new_layer_k, new_layer_v)

        scan_input: Any
        if has_hooks:
            scan_input = (self.layers, cache.k, cache.v, hook_data)
        else:
            scan_input = (self.layers, cache.k, cache.v)
        x, (new_k_stack, new_v_stack) = jax.lax.scan(step, x, scan_input)

        x = _rmsnorm(x, self.final_norm_w)
        logits = jnp.einsum("btd,dv->btv", x, self._head_weight(compute_dtype))
        new_cache = KVCache(k=new_k_stack, v=new_v_stack)
        return logits.astype(jnp.float32), new_cache


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
    one-initialised so the norm acts as identity at step 0.
    ``embed_tokens`` is the single ``[V, d]`` token table. ``lm_head``
    is ``None`` when ``cfg.tie_embeddings`` (the default) and a separate
    ``Normal(0, 0.02)`` ``[d, V]`` head otherwise. ``decomp_table`` is
    built from the engine vocab. RoPE phase tables are *not* stored on
    the model — :meth:`PAWNModel.__call__` recomputes them per forward
    pass from ``cfg.head_dim`` / ``cfg.max_seq_len`` / ``cfg.rope_base``,
    and JIT constant-folds them when ``cfg`` is static.
    """
    if isinstance(key, int):
        key = jax.random.key(key)
    sub = jax.random.split(key, 9)
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
    lm_head = None if cfg.tie_embeddings else _normal_init(sub[8], (d, V))
    return PAWNModel(
        embed_tokens=_normal_init(sub[7], (V, d)),
        layers=layers,
        final_norm_w=jnp.ones((d,), dtype=jnp.float32),
        lm_head=lm_head,
        decomp_table=_build_decomp_table(),
        cfg=cfg,
    )


def sliced(supernet_model: PAWNModel, variant_cfg: ModelConfig) -> PAWNModel:
    """Return a new :class:`PAWNModel` at the variant shape, taking the
    inner ``[:d_V, :d_V]`` block of every weight tensor.

    B.5: when ``variant_cfg.n_layers < supernet.n_layers`` we take the
    first ``variant_cfg.n_layers`` layers of the supernet (a depth +
    width slice). The slice direction is "outer" axis first → "inner"
    axes after, so the per-layer leading-axis stacks shrink from
    ``(supernet.n_layers, ...)`` to ``(variant.n_layers, ...)``.

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
    nv = variant_cfg.n_layers

    sup_layers = supernet_model.layers
    layers = TransformerLayer(
        attn_norm_w=sup_layers.attn_norm_w[:nv, :dv],
        wq=sup_layers.wq[:nv, :dv, :dv],
        wk=sup_layers.wk[:nv, :dv, :dv],
        wv=sup_layers.wv[:nv, :dv, :dv],
        wo=sup_layers.wo[:nv, :dv, :dv],
        ffn_norm_w=sup_layers.ffn_norm_w[:nv, :dv],
        w_gate=sup_layers.w_gate[:nv, :dv, :dv_ff],
        w_up=sup_layers.w_up[:nv, :dv, :dv_ff],
        w_down=sup_layers.w_down[:nv, :dv_ff, :dv],
    )
    # Width-slice the token table to the variant's d_V. The tied head
    # follows for free (logits reuse ``embed_tokens.T``); only the untied
    # standalone ``lm_head`` needs its own ``[:dv, :]`` slice.
    lm_head = (
        None
        if supernet_model.lm_head is None
        else supernet_model.lm_head[:dv, :]
    )
    return PAWNModel(
        embed_tokens=supernet_model.embed_tokens[:, :dv],
        layers=layers,
        final_norm_w=supernet_model.final_norm_w[:dv],
        lm_head=lm_head,
        decomp_table=supernet_model.decomp_table,
        cfg=variant_cfg,
    )
