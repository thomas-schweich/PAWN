"""Bottleneck (Houlsby et al. 2019) adapter for the JAX PAWN backbone.

Inserts a small residual MLP after the attention sublayer and/or the
FFN sublayer within each transformer block::

    x = x + up(gelu(... hidden(gelu(down(x))) ...))

``up.weight`` is zero-initialised so the wrapped model is bit-identical
to the frozen backbone at step 0. ``cfg.n_hidden`` adds extra
``Linear(bn, bn)`` stages with GELU between ``down`` and ``up`` so the
adapter MLP can be deeper than the two-layer Houlsby block; identity-
at-init still holds (``up.weight == 0``).

Per-adapter param count (no bias anywhere):
``2 · d_model · bn + n_hidden · bn²``.

PyTree layout::

  BottleneckModel
  ├── backbone: PAWNModel    (frozen)
  └── bot: BottleneckParams
       ├── attn_down  / attn_up  / attn_hidden   ([n_layers, d, bn] / etc.)
       └── ffn_down   / ffn_up   / ffn_hidden     (same per-layer stack)

Per-layer / per-position selectivity (only adapt layers 6..8; only the
FFN position; etc.) is achieved by setting the corresponding stacked
slice's ``up`` weights to zero at init — the residual then resolves
back to the bare backbone delta. This avoids the per-layer ragged-set
bookkeeping the legacy needed and keeps the scan shape-uniform.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

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
class BottleneckConfig:
    """Bottleneck hyperparameters.

    Args:
        bottleneck_dim: inner width of the down/up projection.
        n_hidden: number of extra ``Linear(bn, bn) + gelu`` stages
            between ``down`` and ``up``. ``0`` reproduces the
            standard Houlsby two-layer adapter.
        adapt_attn: insert an adapter after the attention residual.
        adapt_ffn: insert an adapter after the FFN residual.
        layers: per-layer indices to adapt; defaults to *all*
            layers when ``None``. Per-layer / per-position
            selectivity is realised by zero-initialising the
            ``up`` weights of inactive (layer, position) pairs
            — the residual then resolves back to the bare
            backbone delta.
    """

    bottleneck_dim: int
    n_hidden: int = 0
    adapt_attn: bool = True
    adapt_ffn: bool = True
    layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.bottleneck_dim <= 0:
            raise ValueError(
                f"bottleneck_dim={self.bottleneck_dim} must be > 0"
            )
        if self.n_hidden < 0:
            raise ValueError(
                f"n_hidden={self.n_hidden} must be >= 0"
            )
        if not self.adapt_attn and not self.adapt_ffn:
            raise ValueError(
                "at least one of adapt_attn / adapt_ffn must be True"
            )


class BottleneckParams(eqx.Module):
    """Per-layer down/hidden/up matrices for attention + FFN positions.

    Shapes (L = n_layers, d = d_model, bn = bottleneck_dim, H = n_hidden):
        attn_down:   (L, d, bn)
        attn_hidden: (L, H, bn, bn)
        attn_up:     (L, bn, d)
        ffn_down / ffn_hidden / ffn_up: same shape, FFN position.

    When ``H == 0``, ``attn_hidden`` / ``ffn_hidden`` are
    ``(L, 0, bn, bn)`` sentinels; the scan body skips them.
    """

    attn_down: jax.Array
    attn_hidden: jax.Array
    attn_up: jax.Array
    ffn_down: jax.Array
    ffn_hidden: jax.Array
    ffn_up: jax.Array
    cfg: BottleneckConfig = eqx.field(static=True)


class BottleneckModel(eqx.Module):
    """Frozen PAWN backbone wrapped with per-layer Houlsby bottleneck
    adapters at the attention + FFN positions."""

    backbone: PAWNModel
    bot: BottleneckParams

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

        # Per-layer adapter step composes the legacy two-stage forward
        # (attn → adapter → ffn → adapter) with the same residual
        # convention as the bare backbone.
        def adapter(
            h: jax.Array, down: jax.Array, hidden_stack: jax.Array, up: jax.Array,
        ) -> jax.Array:
            """``x + up(gelu(... hidden_l(gelu(down(x))) ...))``."""
            # Cast adapter weights to h.dtype so the carry dtype stays
            # stable under bf16 backbones (same fix as FiLM / LoRA).
            d = down.astype(h.dtype)
            u = up.astype(h.dtype)
            inner = h @ d
            # Hidden stages: ``H`` extra Linear(bn, bn) layers with
            # GELU before each. When H == 0 the scan range is empty.
            if hidden_stack.shape[0] > 0:
                def hidden_step(z: jax.Array, w: jax.Array) -> tuple[jax.Array, None]:
                    w = w.astype(z.dtype)
                    return jax.nn.gelu(z) @ w, None
                inner, _ = jax.lax.scan(hidden_step, inner, hidden_stack)
            return h + jax.nn.gelu(inner) @ u

        # The scan's ``xs`` packs (LayerWeights, attn_down, attn_hidden,
        # attn_up, ffn_down, ffn_hidden, ffn_up) per layer; the local
        # alias keeps the closure's type annotation legible.
        def run_layer(
            h: jax.Array,
            step: tuple[
                LayerWeights, jax.Array, jax.Array, jax.Array,
                jax.Array, jax.Array, jax.Array,
            ],
        ) -> tuple[jax.Array, None]:
            (lw, a_down, a_hid, a_up, f_down, f_hid, f_up) = step
            h = h + _attention(
                _rmsnorm(h, lw.attn_norm),
                lw.wq, lw.wk, lw.wv, lw.wo,
                cos, sin, mask, all_masked_query, cfg.n_heads,
            )
            h = adapter(h, a_down, a_hid, a_up)
            h = h + _ffn(_rmsnorm(h, lw.ffn_norm), lw.w_gate, lw.w_up, lw.w_down)
            h = adapter(h, f_down, f_hid, f_up)
            return h, None

        x, _ = jax.lax.scan(
            jax.checkpoint(run_layer, prevent_cse=False),
            x,
            (
                layers,
                self.bot.attn_down, self.bot.attn_hidden, self.bot.attn_up,
                self.bot.ffn_down, self.bot.ffn_hidden, self.bot.ffn_up,
            ),
        )
        x = _rmsnorm(x, self.backbone.final_norm)
        return x @ self.backbone.lm_head


def init_bottleneck_model(
    backbone: PAWNModel, cfg: BottleneckConfig, key: jax.Array,
) -> BottleneckModel:
    """Initialise a ``BottleneckModel``. ``down`` / ``hidden`` are
    Xavier-style Gaussians; ``up`` is zero so identity-at-init holds.

    For layers / positions not in ``cfg.layers`` (or with
    ``adapt_attn=False`` / ``adapt_ffn=False``), the ``up`` weights are
    zeroed individually so the residual resolves back to the bare
    backbone delta. The shapes stay layer-uniform, which keeps the
    ``lax.scan`` shape-uniform — no ragged sentinels.
    """
    n_layers = backbone.cfg.n_layers
    d = backbone.cfg.d_model
    bn = cfg.bottleneck_dim
    H = cfg.n_hidden

    active = (
        set(cfg.layers) if cfg.layers is not None else set(range(n_layers))
    )

    def per_layer_init(
        position_active: bool, k: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        kd, kh, _ku = jax.random.split(k, 3)
        # Xavier-style normal scaled by 1/sqrt(fan_in).
        down = jax.random.normal(kd, (d, bn), dtype=jnp.float32) * (
            1.0 / d ** 0.5
        )
        hidden = (
            jax.random.normal(kh, (H, bn, bn), dtype=jnp.float32)
            * (1.0 / bn ** 0.5)
        ) if H > 0 else jnp.zeros((0, bn, bn), dtype=jnp.float32)
        up = jnp.zeros((bn, d), dtype=jnp.float32)
        # If this (layer, position) is inactive the rest of the call
        # zeros every weight — keeps the scan shape uniform without
        # needing a position-mask tensor.
        if not position_active:
            down = jnp.zeros_like(down)
            hidden = jnp.zeros_like(hidden)
            # up is already zero
        return down, hidden, up

    keys = jax.random.split(key, n_layers * 2)
    attn_packs, ffn_packs = [], []
    for li in range(n_layers):
        attn_active = (li in active) and cfg.adapt_attn
        ffn_active = (li in active) and cfg.adapt_ffn
        attn_packs.append(per_layer_init(attn_active, keys[2 * li]))
        ffn_packs.append(per_layer_init(ffn_active, keys[2 * li + 1]))

    attn_down = jnp.stack([p[0] for p in attn_packs])
    attn_hidden = jnp.stack([p[1] for p in attn_packs])
    attn_up = jnp.stack([p[2] for p in attn_packs])
    ffn_down = jnp.stack([p[0] for p in ffn_packs])
    ffn_hidden = jnp.stack([p[1] for p in ffn_packs])
    ffn_up = jnp.stack([p[2] for p in ffn_packs])

    return BottleneckModel(
        backbone=backbone,
        bot=BottleneckParams(
            attn_down=attn_down, attn_hidden=attn_hidden, attn_up=attn_up,
            ffn_down=ffn_down, ffn_hidden=ffn_hidden, ffn_up=ffn_up,
            cfg=cfg,
        ),
    )


def adapter_filter(model: BottleneckModel) -> BottleneckModel:
    """``True`` on every ``model.bot.*`` array leaf, ``False`` elsewhere."""
    false_tree = jax.tree_util.tree_map(lambda _: False, model)
    bot_true = jax.tree_util.tree_map(eqx.is_inexact_array, model.bot)
    return eqx.tree_at(lambda m: m.bot, false_tree, bot_true)
