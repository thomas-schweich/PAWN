"""FiLM — Feature-wise Linear Modulation (Perez et al. 2018).

Per-channel ``gamma`` / ``beta`` scale + shift applied to the residual
stream after each transformer layer (``h = gamma * h + beta``) and,
optionally, to the output logits (``logits = gamma * logits + beta``
over the uniform ``V``-wide vocabulary). ``gamma`` is one-initialised
and ``beta`` zero-initialised so the model starts identical to the
frozen backbone.

This is *true* FiLM: the affine modulation lives in the residual stream
(injected via the backbone's post-residual ``ffn_hook``), **not** folded
into the RMSNorm weights. Folding ``gamma`` into a norm weight would
couple it to the norm's rescaling and folding ``beta`` into a norm
weight is impossible (RMSNorm has no additive term), so the earlier
fold-into-norm implementation was not FiLM at all.

Because the per-layer shift and the output-logit modulation cannot
collapse into the backbone's weight tensors, :func:`apply_film` returns
a :class:`FiLMEffective` wrapper (mirroring
:class:`pawn.adapters.bottleneck.BottleneckEffective`) that threads the
modulation through the backbone's forward pass rather than a folded
:class:`PAWNModel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from pawn.checkpoint import ADAPTER_SAFETENSORS
from pawn.config import ModelConfig
from pawn.model import KVCache, PAWNModel

__all__ = [
    "FiLMConfig",
    "FiLMAdapter",
    "FiLMEffective",
    "ADAPTER_SAFETENSORS",
    "init_film_adapter",
    "apply_film",
    "film_filter",
    "save_film_adapter",
    "load_film_adapter",
]


# ``ADAPTER_SAFETENSORS`` (the sidecar filename used by the trainer's save /
# resume path) is owned by :mod:`pawn.checkpoint` and re-exported here so the
# train + load sites share a single literal regardless of strategy.


@dataclass(frozen=True)
class FiLMConfig:
    """FiLM hyperparameters. ``use_output_film=True`` is the v2 default
    (plan §10 S3) — adds a FiLM head over the output logits too."""

    use_output_film: bool = True


class FiLMAdapter(eqx.Module):
    """Per-layer gamma/beta + optional output FiLM.

    ``gamma`` / ``beta`` have shape ``(n_layers, d_model)`` and apply
    elementwise to each layer's residual-stream output. The optional
    output FiLM has shape ``(vocab_size,)`` and applies to the output
    logits (post-Phase-A: the uniform ``V``-wide head, **not**
    ``d_model``).
    """

    gamma: Float[Array, "n_layers d"]
    beta: Float[Array, "n_layers d"]
    output_gamma: Float[Array, "V"] | None
    output_beta: Float[Array, "V"] | None
    cfg: FiLMConfig = eqx.field(static=True)


def init_film_adapter(
    backbone: PAWNModel, cfg: FiLMConfig, key: jax.Array | int
) -> FiLMAdapter:
    """gamma=1, beta=0 → identity at step 0.

    The per-layer slabs are sized ``(n_layers, d_model)``; the optional
    output-FiLM slabs are sized ``(vocab_size,)`` because they modulate
    the logits, not the ``d_model``-wide final-norm output.
    """
    del key  # unused — one/zero init is deterministic
    n_layers = backbone.cfg.n_layers
    d = backbone.cfg.d_model
    v = backbone.cfg.vocab_size
    out_g = jnp.ones((v,), dtype=jnp.float32) if cfg.use_output_film else None
    out_b = jnp.zeros((v,), dtype=jnp.float32) if cfg.use_output_film else None
    return FiLMAdapter(
        gamma=jnp.ones((n_layers, d), dtype=jnp.float32),
        beta=jnp.zeros((n_layers, d), dtype=jnp.float32),
        output_gamma=out_g,
        output_beta=out_b,
        cfg=cfg,
    )


class FiLMEffective(eqx.Module):
    """Wrapper exposing :meth:`PAWNModel.__call__`'s signature while
    applying true FiLM modulation.

    Per-layer affine ``h = gamma_l * h + beta_l`` is injected after each
    layer's FFN residual via the backbone's ``ffn_hook``; the optional
    output FiLM rescales the final ``V``-wide logits. Looks like a
    :class:`PAWNModel` to the trainer / eval (same call signature, same
    logits shape), so they treat it interchangeably with the bare
    backbone and the bottleneck wrapper.
    """

    backbone: PAWNModel
    adapter: FiLMAdapter

    @property
    def cfg(self) -> ModelConfig:
        return self.backbone.cfg

    @property
    def decomp_table(self) -> Int[Array, "n_actions 3"]:
        return self.backbone.decomp_table

    @property
    def lm_head(self) -> Float[Array, "d V"] | None:
        return self.backbone.lm_head

    @property
    def final_norm_w(self) -> Float[Array, "d"]:
        return self.backbone.final_norm_w

    @property
    def layers(self) -> Any:  # TransformerLayer; avoid circular type import
        return self.backbone.layers

    @property
    def embed_tokens(self) -> Float[Array, "V d"]:
        return self.backbone.embed_tokens

    def _ffn_hook(
        self,
        h: Float[Array, "B T d"],
        slice_: tuple[Float[Array, "d"], Float[Array, "d"]],
    ) -> Float[Array, "B T d"]:
        """``h = gamma_l * h + beta_l`` in the residual stream.

        ``slice_`` is the per-layer ``(gamma_l, beta_l)`` pair handed in
        by the backbone's scan (each ``(d_model,)``). Broadcast over the
        ``(B, T)`` axes; cast the params to ``h``'s dtype so the AMP
        forward stays in compute dtype.
        """
        gamma_l, beta_l = slice_
        gamma_l = gamma_l.astype(h.dtype)
        beta_l = beta_l.astype(h.dtype)
        return gamma_l * h + beta_l

    def _apply_output_film(
        self, logits: Float[Array, "B T V"]
    ) -> Float[Array, "B T V"]:
        """``logits = output_gamma * logits + output_beta`` over ``V``.

        No-op when output FiLM is disabled. The slabs are ``(vocab_size,)``
        and broadcast over the ``(B, T)`` axes.
        """
        adapter = self.adapter
        if adapter.output_gamma is None or adapter.output_beta is None:
            return logits
        og = adapter.output_gamma.astype(logits.dtype)
        ob = adapter.output_beta.astype(logits.dtype)
        return og * logits + ob

    def __call__(
        self,
        input_ids: Int[Array, "B T"],
        attention_mask: Int[Array, "B T"] | None = None,
        *,
        compute_dtype: jnp.dtype | None = None,
        use_sdpa: bool = False,
        use_flash: bool = False,
    ) -> Float[Array, "B T V"]:
        # `hook_data` leaves must each carry a leading n_layers axis so
        # the backbone's scan zips them with the per-layer weights. The
        # gamma/beta slabs are (n_layers, d_model); pass them as a tuple
        # so the scan hands the ffn_hook a per-layer (gamma_l, beta_l).
        hook_data = (self.adapter.gamma, self.adapter.beta)
        logits = self.backbone(
            input_ids,
            attention_mask,
            compute_dtype=compute_dtype,
            ffn_hook=self._ffn_hook,
            hook_data=hook_data,
            use_sdpa=use_sdpa,
            use_flash=use_flash,
        )
        return self._apply_output_film(logits)

    def forward_with_cache(
        self,
        input_ids: Int[Array, "B T_new"],
        cache: KVCache,
        pos_start: Int[Array, ""] | int,
        *,
        compute_dtype: jnp.dtype | None = None,
    ) -> tuple[Float[Array, "B T_new V"], KVCache]:
        """Cached forward — threads the per-layer FiLM hook through the
        backbone's KV-cached path and applies output FiLM to the logits.

        The affine modulation is position-local (operates pointwise on
        the hidden state / logits at each token), so injecting it at the
        same ``ffn_hook`` site in the cached path produces the same
        logits as the full forward — modulo XLA kernel-ordering noise.
        """
        hook_data = (self.adapter.gamma, self.adapter.beta)
        logits, new_cache = self.backbone.forward_with_cache(
            input_ids, cache, pos_start,
            compute_dtype=compute_dtype,
            ffn_hook=self._ffn_hook,
            hook_data=hook_data,
        )
        return self._apply_output_film(logits), new_cache


def apply_film(backbone: PAWNModel, adapter: FiLMAdapter) -> FiLMEffective:
    """Return a :class:`FiLMEffective` that runs ``backbone`` with the
    FiLM affine modulation injected into the residual stream (and,
    optionally, the output logits).

    Unlike weight-folding adapters (LoRA, sparse), FiLM's per-layer
    shift and output-logit modulation can't collapse into the backbone's
    weight tensors, so this returns a callable wrapper — the trainer and
    eval treat it interchangeably with a bare :class:`PAWNModel`.
    """
    return FiLMEffective(backbone=backbone, adapter=adapter)


def film_filter(adapter: FiLMAdapter) -> FiLMAdapter:
    return jax.tree_util.tree_map(
        lambda leaf: True if eqx.is_inexact_array(leaf) else False, adapter
    )


# ---------------------------------------------------------------------------
# Save / load — sidecar safetensors next to the (frozen) backbone checkpoint
# ---------------------------------------------------------------------------


# Fields persisted to / restored from the sidecar. ``output_gamma`` /
# ``output_beta`` are absent when ``use_output_film=False``; the config is
# replayed at load time so the load path takes the same branch as init.
_ADAPTER_FIELDS: tuple[str, ...] = (
    "gamma", "beta", "output_gamma", "output_beta",
)


def save_film_adapter(adapter: FiLMAdapter, out_dir: "Path | str") -> None:
    """Write the FiLM slabs to ``out_dir/adapter.safetensors``.

    Only the populated fields land on disk — the output-FiLM slabs stay
    absent when ``use_output_film=False``. The caller writes the frozen
    backbone (``model.safetensors``) and the config block separately;
    this just emits the FiLM sidecar, keeping the wrapper-adapter save
    layout in lockstep with :mod:`pawn.adapters.bottleneck`.
    """
    from safetensors.numpy import save_file as st_save

    out_path = Path(out_dir)
    arrays: dict[str, np.ndarray] = {}
    for name in _ADAPTER_FIELDS:
        leaf = getattr(adapter, name)
        if leaf is not None:
            arrays[f"film.{name}"] = np.asarray(leaf)
    st_save(arrays, str(out_path / ADAPTER_SAFETENSORS))


def load_film_adapter(
    ckpt_dir: "Path | str", cfg: FiLMConfig
) -> FiLMAdapter:
    """Restore a :class:`FiLMAdapter` from a checkpoint sidecar.

    Expects ``ckpt_dir/adapter.safetensors`` written by
    :func:`save_film_adapter`. ``cfg`` must match the save-time config —
    ``use_output_film`` determines whether the output-FiLM slabs are
    populated, and a mismatch between the sidecar's keys and ``cfg`` is
    rejected with :class:`ValueError` rather than silently loading a
    mismatched adapter.

    Raises :class:`FileNotFoundError` if the sidecar isn't present.
    """
    from safetensors.numpy import load_file as st_load

    path = Path(ckpt_dir) / ADAPTER_SAFETENSORS
    if not path.is_file():
        raise FileNotFoundError(
            f"no {ADAPTER_SAFETENSORS} in {ckpt_dir} — not a FiLM "
            "checkpoint, or saved without sidecar"
        )
    flat = st_load(str(path))

    has_output = "film.output_gamma" in flat
    if cfg.use_output_film != has_output:
        raise ValueError(
            f"FiLMConfig / sidecar mismatch at {path}: cfg "
            f"use_output_film={cfg.use_output_film} but sidecar "
            f"{'has' if has_output else 'lacks'} output-FiLM slabs. Was "
            "the run resumed with a different --use-output-film flag?"
        )

    def _maybe(key: str) -> jax.Array | None:
        full = f"film.{key}"
        if full not in flat:
            return None
        return jnp.asarray(flat[full])

    return FiLMAdapter(
        gamma=jnp.asarray(flat["film.gamma"]),
        beta=jnp.asarray(flat["film.beta"]),
        output_gamma=_maybe("output_gamma"),
        output_beta=_maybe("output_beta"),
        cfg=cfg,
    )
