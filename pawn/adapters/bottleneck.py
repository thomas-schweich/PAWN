"""Bottleneck adapters (Houlsby et al. 2019).

A small residual MLP inserted after each transformer sublayer:

    x = x + up(gelu(down(x)))

(with optional extra ``Linear+GELU`` stages in between when
``n_hidden > 0``). The up-projection is zero-initialised so the
adapter starts identical to the frozen backbone.

Per the v1 contract (and the migration plan §10 S7 adapter table),
both placement flags are honoured:

- ``no_adapt_attn=False`` → inject the residual MLP after the
  attention sublayer.
- ``no_adapt_ffn=False`` → inject after the FFN sublayer.

At least one of the two must remain enabled (an all-off config would
silently degenerate to "frozen backbone").

The forward injects the residuals via the ``attn_hook`` / ``ffn_hook``
parameters of :meth:`pawn.model.PAWNModel.__call__`, which the
:class:`BottleneckEffective` wrapper threads through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from pawn.config import ModelConfig
from pawn.model import PAWNModel

__all__ = [
    "BottleneckConfig",
    "BottleneckAdapter",
    "BottleneckEffective",
    "init_bottleneck_adapter",
    "apply_bottleneck",
    "bottleneck_filter",
]


@dataclass(frozen=True)
class BottleneckConfig:
    """Houlsby bottleneck size + placement.

    ``dim`` is the inner bottleneck dimension; ``n_hidden`` is the
    number of extra ``Linear+GELU`` stages between the down and up
    projections (0 = the standard two-layer Houlsby block);
    ``no_adapt_attn`` / ``no_adapt_ffn`` honour the v1 flag names per
    plan §10 S3.
    """

    dim: int
    n_hidden: int = 0
    no_adapt_attn: bool = False
    no_adapt_ffn: bool = False

    def __post_init__(self) -> None:
        # Both placement flags off ⇒ the adapter touches nothing and
        # would silently degenerate to "frozen backbone" at runtime.
        # Surface that as a config error rather than a no-op run.
        if self.no_adapt_attn and self.no_adapt_ffn:
            raise ValueError(
                "BottleneckConfig: no_adapt_attn and no_adapt_ffn are "
                "both set — the bottleneck would touch nothing. Enable "
                "at least one site."
            )
        if self.n_hidden < 0:
            raise ValueError(
                f"BottleneckConfig.n_hidden must be >= 0, got {self.n_hidden}"
            )


class BottleneckAdapter(eqx.Module):
    """Per-layer down/(hidden)/up bottleneck weights for both placements.

    Attention-side and FFN-side adapters are stored separately so
    ``no_adapt_attn`` / ``no_adapt_ffn`` can disable one without
    affecting the other. ``hidden_*`` is shape
    ``(n_layers, n_hidden, dim, dim)`` (``n_hidden = 0`` ⇒ a 0-length
    leading axis, which scans degenerate to a no-op).

    All arrays have a leading ``n_layers`` axis so the bottleneck slice
    threads through the backbone's :func:`jax.lax.scan` cleanly.
    """

    # Attention-side (None when no_adapt_attn=True).
    down_attn: Float[Array, "n_layers d dim"] | None
    hidden_attn: Float[Array, "n_layers n_hidden dim dim"] | None
    up_attn: Float[Array, "n_layers dim d"] | None
    # FFN-side (None when no_adapt_ffn=True).
    down_ffn: Float[Array, "n_layers d dim"] | None
    hidden_ffn: Float[Array, "n_layers n_hidden dim dim"] | None
    up_ffn: Float[Array, "n_layers dim d"] | None
    cfg: BottleneckConfig = eqx.field(static=True)


def _kaiming_uniform(
    key: jax.Array, shape: tuple[int, ...], fan_in: int
) -> jax.Array:
    """``kaiming_uniform_(a=sqrt(5))`` from PyTorch — matches v1's init.

    For ``a=sqrt(5)``, ``gain = sqrt(2 / 6) = sqrt(1/3)``, and
    ``bound = gain * sqrt(3 / fan_in) = sqrt(1/fan_in)``.
    """
    bound = math.sqrt(1.0 / fan_in)
    return jax.random.uniform(key, shape, minval=-bound, maxval=bound)


def init_bottleneck_adapter(
    backbone: PAWNModel, cfg: BottleneckConfig, key: jax.Array | int
) -> BottleneckAdapter:
    """Kaiming-uniform `down`/`hidden`; zero `up` (identity at step 0)."""
    if isinstance(key, int):
        key = jax.random.key(key)
    d = backbone.cfg.d_model
    n_layers = backbone.cfg.n_layers
    n_hidden = cfg.n_hidden
    dim = cfg.dim

    keys = jax.random.split(key, 4)

    def attn_branch() -> tuple[jax.Array, jax.Array, jax.Array]:
        down = _kaiming_uniform(keys[0], (n_layers, d, dim), fan_in=d)
        hidden = _kaiming_uniform(
            keys[1], (n_layers, n_hidden, dim, dim), fan_in=dim,
        )
        up = jnp.zeros((n_layers, dim, d), dtype=jnp.float32)
        return down, hidden, up

    def ffn_branch() -> tuple[jax.Array, jax.Array, jax.Array]:
        down = _kaiming_uniform(keys[2], (n_layers, d, dim), fan_in=d)
        hidden = _kaiming_uniform(
            keys[3], (n_layers, n_hidden, dim, dim), fan_in=dim,
        )
        up = jnp.zeros((n_layers, dim, d), dtype=jnp.float32)
        return down, hidden, up

    if cfg.no_adapt_attn:
        d_a, h_a, u_a = None, None, None
    else:
        d_a, h_a, u_a = attn_branch()
    if cfg.no_adapt_ffn:
        d_f, h_f, u_f = None, None, None
    else:
        d_f, h_f, u_f = ffn_branch()

    return BottleneckAdapter(
        down_attn=d_a, hidden_attn=h_a, up_attn=u_a,
        down_ffn=d_f, hidden_ffn=h_f, up_ffn=u_f,
        cfg=cfg,
    )


def _bottleneck_residual(
    h: jax.Array,
    down: jax.Array | None,
    hidden: jax.Array | None,
    up: jax.Array | None,
    compute_dtype: jnp.dtype | None,
) -> jax.Array:
    """``h + up(gelu(hidden_stages(gelu(down(h)))))``.

    Returns ``h`` unchanged when ``down`` is None (placement disabled).
    All matmuls run in ``compute_dtype`` when set, with the master
    weights cast just before each einsum (XLA fuses the cast).
    """
    if down is None or up is None:
        return h
    d_w = down if compute_dtype is None else down.astype(compute_dtype)
    u_w = up if compute_dtype is None else up.astype(compute_dtype)
    # "i" = batch, "j" = seq, "d" = d_model, "k"/"l" = bottleneck dim.
    z = jnp.einsum("ijd,dk->ijk", h, d_w)
    z = jax.nn.gelu(z)
    if hidden is not None and hidden.shape[0] > 0:
        # `hidden` here is the per-layer slice with leading n_hidden axis.
        def stage(carry: jax.Array, w: jax.Array) -> tuple[jax.Array, None]:
            w_c = w if compute_dtype is None else w.astype(compute_dtype)
            out = jnp.einsum("ijk,kl->ijl", carry, w_c)
            return jax.nn.gelu(out), None

        z, _ = jax.lax.scan(stage, z, hidden)
    out = jnp.einsum("ijk,kd->ijd", z, u_w)
    return h + out


class BottleneckEffective(eqx.Module):
    """Wrapper that exposes ``PAWNModel.__call__``'s signature but
    threads the bottleneck adapter's per-layer weights through the
    backbone's scan via the ``attn_hook`` / ``ffn_hook`` injection
    points.

    Looks like a ``PAWNModel`` to the trainer (same call signature,
    same logits shape); the trainer and eval scripts treat it
    interchangeably.
    """

    backbone: PAWNModel
    adapter: BottleneckAdapter

    @property
    def cfg(self) -> ModelConfig:
        return self.backbone.cfg

    @property
    def decomp_table(self) -> Int[Array, "n_actions 3"]:
        return self.backbone.decomp_table

    @property
    def lm_head(self) -> Float[Array, "d V"]:
        return self.backbone.lm_head

    @property
    def final_norm_w(self) -> Float[Array, "d"]:
        return self.backbone.final_norm_w

    @property
    def layers(self) -> Any:  # TransformerLayer; avoid circular type import
        return self.backbone.layers

    @property
    def embed_src(self) -> Float[Array, "64 d"]:
        return self.backbone.embed_src

    @property
    def embed_dst(self) -> Float[Array, "64 d"]:
        return self.backbone.embed_dst

    @property
    def embed_promo(self) -> Float[Array, "5 d"]:
        return self.backbone.embed_promo

    @property
    def embed_pad(self) -> Float[Array, "d"]:
        return self.backbone.embed_pad

    @property
    def embed_outcome(self) -> Float[Array, "n_out d"]:
        return self.backbone.embed_outcome

    def __call__(
        self,
        input_ids: Int[Array, "B T"],
        attention_mask: Int[Array, "B T"] | None = None,
        *,
        compute_dtype: jnp.dtype | None = None,
    ) -> Float[Array, "B T V"]:
        adapter = self.adapter
        # Per-layer hook slices: each leaf of `hook_data` has a leading
        # n_layers axis; the scan zips it with the backbone's layers.
        hook_data = adapter

        def attn_hook(
            h: Float[Array, "B T d"], slice_: BottleneckAdapter
        ) -> Float[Array, "B T d"]:
            return _bottleneck_residual(
                h, slice_.down_attn, slice_.hidden_attn, slice_.up_attn,
                compute_dtype,
            )

        def ffn_hook(
            h: Float[Array, "B T d"], slice_: BottleneckAdapter
        ) -> Float[Array, "B T d"]:
            return _bottleneck_residual(
                h, slice_.down_ffn, slice_.hidden_ffn, slice_.up_ffn,
                compute_dtype,
            )

        return self.backbone(
            input_ids,
            attention_mask,
            compute_dtype=compute_dtype,
            attn_hook=attn_hook if not adapter.cfg.no_adapt_attn else None,
            ffn_hook=ffn_hook if not adapter.cfg.no_adapt_ffn else None,
            hook_data=hook_data,
        )


def apply_bottleneck(
    backbone: PAWNModel, adapter: BottleneckAdapter
) -> BottleneckEffective:
    """Return a callable that runs the backbone with the bottleneck
    residual MLPs injected after each enabled sublayer."""
    return BottleneckEffective(backbone=backbone, adapter=adapter)


def bottleneck_filter(adapter: BottleneckAdapter) -> BottleneckAdapter:
    return jax.tree_util.tree_map(
        lambda leaf: True if eqx.is_inexact_array(leaf) else False, adapter
    )
