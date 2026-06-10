"""Factored-embedding (v1-architecture) PAWN model — trainable in v2.

:class:`FactoredPAWNModel` reproduces the **v1 architecture** exactly:
factored move embeddings (``src_embed[64] + dst_embed[64] + promo_embed[5]``
summed per move token, with standalone ``pad_embed`` / ``outcome_embed``
overrides), an always-untied ``lm_head``, and v1's native dimensions
(``FACTORED_V1_LARGE``: d=640, 10 layers, 8 heads × head_dim 80, d_ff=2560,
vocab 1980 — no BOS/NULL/reserved rows). It exists to answer one question:
is the v1↔v2 compound-legality gap caused by the architecture (factored
embeddings + dims) or by the training recipe (LR schedule etc.)? Train this
model under the *exact* v2 recipe and compare.

**The trunk is shared with v2, deliberately.** Everything except the
embedding lookup and the head is imported from :mod:`pawn.model` —
:class:`~pawn.model.TransformerLayer`, the :func:`~pawn.model._run_layers_impl`
scan body, fp32 RoPE, fp32-softmax attention, the custom-VJP flash kernel,
and the fp32 output-head matmul. This is load-bearing for the experiment:
an earlier port of this class (pre-``cda3f8a``) carried the OLD numerical
recipe (compute-dtype RoPE, compute-dtype head, stock Pallas backward), and
training it under bf16 AMP would have re-introduced the exact instabilities
v2 already fixed — confounding the architecture comparison. With the shared
trunk, the ONLY differences from :class:`pawn.model.PAWNModel` are:

    1. ``_embed`` — factored gather vs. uniform ``embed_tokens[V, d]``.
    2. ``lm_head`` — always a standalone tensor (v1 never tied).
    3. The config dims (``head_dim=80``, ``d_ff=2560``, ``vocab_size=1980``).

Sequence contract: v1's native bare-moves layout — no BOS (1980 is
out-of-vocab here). Corpora built by the v2 pipeline are adapted via
:func:`pawn.corpus.to_v1_contract` (slot 0 becomes a masked, unsupervised
PAD), the transform validated against the published v1 checkpoints
(``scripts/eval_v1_legality.py`` through it measures the converted v1-large
at 99.9997% per-move legal / 99.90% game-completion, matching its published
99.9990% / 99.76%).

Save schema: :data:`FACTORED_SAVED_FIELDS` (16 tensors).
``pawn.checkpoint`` dispatches on ``ModelConfig.factored_embeddings``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from pawn.config import (
    NUM_ACTIONS,
    OUTCOME_TOKEN_BASE,
    PAD_TOKEN,
    ModelConfig,
)
from pawn.model import (
    KVCache,
    TransformerLayer,
    _apply_rope,
    _build_decomp_table,
    _build_rope,
    _normal_init,
    _rmsnorm,
    _run_layers_collect_impl,
    _run_layers_impl,
)

__all__ = [
    "FACTORED_SAVED_FIELDS",
    "FactoredPAWNModel",
    "init_factored_model",
]


# Names of the 16 trainable arrays in safetensors declaration order.
# ``layers.*`` paths are shared with the uniform model's schema (the
# TransformerLayer class itself is shared); only the embedding fields and
# the always-present ``lm_head`` differ. ``pawn.checkpoint`` imports this.
FACTORED_SAVED_FIELDS: Final[tuple[str, ...]] = (
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
assert len(FACTORED_SAVED_FIELDS) == 16, "factored save schema must be 16 fields"


class FactoredPAWNModel(eqx.Module):
    """Decoder-only transformer with v1's factored move embeddings.

    Mirrors :class:`pawn.model.PAWNModel`'s forward contract exactly —
    same ``__call__`` / ``hidden_states`` / ``forward_with_cache``
    signatures, same shared trunk — so the trainer, eval suite, probes,
    and generation harness drive it interchangeably (it satisfies
    :class:`pawn.model.EffectiveCallable` and
    :class:`pawn.model.KVCacheCallable` structurally).

    ``lm_head`` is always present (v1 never tied embeddings — and the
    factored input table has no ``[V, d]`` matrix to tie against).
    """

    # Trainable fields — declaration order = save order
    # (:data:`FACTORED_SAVED_FIELDS`).
    embed_src: Float[Array, "64 d"]
    embed_dst: Float[Array, "64 d"]
    embed_promo: Float[Array, "5 d"]
    embed_pad: Float[Array, "d"]
    embed_outcome: Float[Array, "n_out d"]
    layers: TransformerLayer
    final_norm_w: Float[Array, "d"]
    lm_head: Float[Array, "d V"]

    # Non-trainable int32 lookup (filtered out by ``eqx.is_inexact_array``);
    # rebuilt from the engine vocab at load time, never saved.
    decomp_table: Int[Array, "n_actions 3"]

    # Static config — JIT keys on it; RoPE tables recomputed per forward.
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
        """Forward pass — see :meth:`pawn.model.PAWNModel.__call__` for the
        full kwarg contract (this mirrors it line-for-line; only the
        embedding gather and the head-weight resolution differ).

        Returns logits of shape ``(batch, seq, vocab_size)`` — fp32, with
        the head matmul ALWAYS accumulated in fp32 (the v2 stability
        recipe), even under bf16 AMP.
        """
        T = input_ids.shape[-1]  # noqa: N806
        if T > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds cfg.max_seq_len "
                f"{self.cfg.max_seq_len}"
            )
        if use_flash:
            # Pallas flash drops the PAD mask entirely (safe only under the
            # strict right-pad invariant — see pawn.model._pallas_attn). The
            # factored model's v1 bare-moves contract places a masked PAD at
            # slot 0 (a LEADING pad), which flash would let every real
            # position attend to — a silent contract violation. Hard-reject
            # rather than compute the wrong thing (round-1 review, codex P2;
            # PretrainConfig neutralises the flag upstream too).
            raise ValueError(
                "FactoredPAWNModel does not support use_flash=True: the v1 "
                "bare-moves contract's masked slot-0 PAD violates the "
                "right-pad invariant the Pallas path relies on. Use the "
                "plain path (use_flash=False)."
            )
        with jax.named_scope("embed"):
            x = self._embed(input_ids)
            if compute_dtype is not None:
                x = x.astype(compute_dtype)
        rope_cos, rope_sin = _build_rope(self.cfg.head_dim, T, self.cfg.rope_base)
        mask: Bool[Array, "B 1 T T"] | None
        if use_flash:
            mask = None
        else:
            causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
            if attention_mask is None:
                mask = causal[None, None, :, :]
            else:
                pad = attention_mask.astype(jnp.bool_)[:, None, None, :]
                mask = causal[None, None, :, :] & pad
        with jax.named_scope("transformer_layers"):
            x = _run_layers_impl(
                self.layers, self.cfg, x, rope_cos, rope_sin, mask,
                attention_mask, compute_dtype,
                attn_hook=attn_hook, ffn_hook=ffn_hook, hook_data=hook_data,
                use_sdpa=use_sdpa, use_flash=use_flash,
            )
        with jax.named_scope("final_norm"):
            x = _rmsnorm(x, self.final_norm_w)
        # fp32 head matmul, always — same rationale as the uniform model:
        # a bf16 head matmul can accumulate a surviving logit column to
        # ``inf`` (→ softmax NaN → one AdamW step poisons the projection).
        # ``self.lm_head`` is the fp32 master weight.
        with jax.named_scope("lm_head"):
            logits = jnp.einsum(
                "btd,dv->btv", x.astype(jnp.float32), self.lm_head
            )
        return logits

    def hidden_states(
        self,
        input_ids: Int[Array, "B T"],
        attention_mask: Int[Array, "B T"] | None = None,
    ) -> Float[Array, "L1 B T d"]:
        """Per-layer residual-stream hidden states for linear probing —
        mirrors :meth:`pawn.model.PAWNModel.hidden_states` (shape
        ``(n_layers + 1, B, T, d)``, index 0 = post-embedding stream)."""
        T = input_ids.shape[-1]  # noqa: N806
        if T > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds cfg.max_seq_len "
                f"{self.cfg.max_seq_len}"
            )
        x = self._embed(input_ids)
        rope_cos, rope_sin = _build_rope(self.cfg.head_dim, T, self.cfg.rope_base)
        causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        if attention_mask is None:
            mask = causal[None, None, :, :]
        else:
            pad = attention_mask.astype(jnp.bool_)[:, None, None, :]
            mask = causal[None, None, :, :] & pad
        per_layer = _run_layers_collect_impl(
            self.layers, self.cfg, x, rope_cos, rope_sin, mask,
            attention_mask,
        )
        return jnp.concatenate([x[None], per_layer], axis=0)

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
        """Cached forward pass for autoregressive decoding — mirrors
        :meth:`pawn.model.PAWNModel.forward_with_cache` (same math
        invariant vs. the full forward, same plain-attention-only cache
        path); only ``_embed`` and the head weight differ.
        """
        del use_sdpa  # cached path is plain-attention only (API parity).
        T_new = input_ids.shape[-1]  # noqa: N806
        T_max = cache.k.shape[3]  # noqa: N806
        if T_new > T_max:
            raise ValueError(
                f"forward_with_cache input length {T_new} exceeds cache "
                f"capacity {T_max}"
            )
        # Guard the write window — `lax.dynamic_update_slice` silently
        # clamps, which would corrupt the cache instead of raising.
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

        rope_cos_full, rope_sin_full = _build_rope(
            head_dim, T_max, self.cfg.rope_base
        )
        absolute_q = pos_start + jnp.arange(T_new, dtype=jnp.int32)
        rope_cos = rope_cos_full[absolute_q]  # (T_new, half)
        rope_sin = rope_sin_full[absolute_q]

        kv_positions = jnp.arange(T_max, dtype=jnp.int32)
        attn_mask_2d = kv_positions[None, :] <= absolute_q[:, None]
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
        lm_head = (
            self.lm_head.astype(compute_dtype)
            if compute_dtype is not None
            else self.lm_head
        )
        logits = jnp.einsum("btd,dv->btv", x, lm_head)
        new_cache = KVCache(k=new_k_stack, v=new_v_stack)
        return logits.astype(jnp.float32), new_cache


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def init_factored_model(
    cfg: ModelConfig, key: jax.Array | int
) -> FactoredPAWNModel:
    """Build a freshly-initialised :class:`FactoredPAWNModel` at ``cfg``.

    Requires ``cfg.factored_embeddings`` (and therefore
    ``tie_embeddings=False`` — enforced by ``ModelConfig.__post_init__``).
    All weight tensors are drawn from ``Normal(0, 0.02)`` (the v1
    convention for dim > 1 params, identical to the uniform model's
    :func:`pawn.model.init_model`); RMSNorm weights are one-initialised;
    ``embed_pad`` is zero-initialised (v1 parity). ``decomp_table`` is
    rebuilt from the engine vocab; RoPE phase tables are recomputed
    inside the forward pass.
    """
    if not cfg.factored_embeddings:
        raise ValueError(
            "init_factored_model requires cfg.factored_embeddings=True "
            "(use pawn.model.init_model for the uniform architecture)"
        )
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
    return FactoredPAWNModel(
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
