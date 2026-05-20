"""Sparse (learnable binary mask) adapter for the JAX PAWN backbone.

Each targeted backbone weight ``W`` gets a real-valued learnable
mask ``s`` of the same shape. The effective weight is::

    W_eff = W ⊙ σ(s)             (soft mode, default)
    W_eff = W ⊙ ((s > threshold) - σ(s)).detach() + σ(s))  (hard mode w/ STE)

In *soft* mode the mask is continuous (sigmoid over the logits) — easy
to optimise but doesn't actually zero any weights. In *hard* mode a
straight-through estimator replaces the sigmoid with a hard 0/1 mask
on the forward but routes gradients through the sigmoid on the
backward (so the mask logits learn).

Identity-at-init: the mask logits are initialised so ``σ(s) ≈ 1``
everywhere — i.e. ``s = +5`` gives ``σ(5) ≈ 0.993`` — so the wrapped
forward is approximately identical to the backbone. (Exact identity-
at-init would require ``σ(s) == 1`` which has no finite logit, so we
settle for "approximately bit-identical" — the resulting Δlogit per
position is ≤ ~0.7% of the bare logit norm.)

Targets attention projections only (``wq`` / ``wk`` / ``wv`` / ``wo``)
— same surface as LoRA. The legacy supports broader targets but the
attention projections are by far the most common request and the
infrastructure for the others is identical if needed.

PyTree layout::

  SparseModel
  ├── backbone: PAWNModel       (frozen)
  └── sparse: SparseParams
       └── mask_logits_q/k/v/o  ([n_layers, d, d] each)
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from pawn.model import PAWNModel

_VALID_TARGETS = ("q", "k", "v", "o")


@dataclass(frozen=True)
class SparseConfig:
    """Sparse-mask hyperparameters.

    Args:
        targets: attention projections to mask. Subset of
            ``("q", "k", "v", "o")``.
        density: target fraction of active mask elements. Used only
            for reporting / regularisation hooks; the mask logits
            themselves are unconstrained — couple this with an L1
            penalty on ``σ(s)`` in the trainer to actually drive the
            mask sparse.
        hard: if True, forward uses ``(σ(s) > threshold)`` 0/1 mask
            with a straight-through estimator on the backward.
        threshold: 0/1 cutoff in hard mode. Default 0.5.
        init_logit: initial value for all mask logits. Default ``+5.0``
            so ``σ(5) ≈ 0.993`` and the wrapped forward starts ≈
            identical to the backbone.
    """

    targets: tuple[str, ...] = _VALID_TARGETS
    density: float = 0.01
    hard: bool = False
    threshold: float = 0.5
    init_logit: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.targets, str):
            raise ValueError(
                f"targets must be a tuple of strings, not a bare str "
                f"({self.targets!r}); use e.g. ('q', 'v')"
            )
        if not self.targets:
            raise ValueError("Sparse needs at least one target projection")
        for t in self.targets:
            if t not in _VALID_TARGETS:
                raise ValueError(
                    f"unknown sparse target {t!r}; expected subset of "
                    f"{_VALID_TARGETS}"
                )
        if not 0.0 <= self.density <= 1.0:
            raise ValueError(
                f"density={self.density} must be in [0, 1]"
            )
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(
                f"threshold={self.threshold} must be in (0, 1)"
            )


class SparseParams(eqx.Module):
    """Per-target stacked mask-logit arrays.

    For inactive targets (not in ``cfg.targets``) the corresponding
    logits tensor is a ``[n_layers, d, 0]`` sentinel so the
    effective-weight construction stays shape-uniform and the
    optimizer ignores empty leaves.
    """

    logits_q: jax.Array
    logits_k: jax.Array
    logits_v: jax.Array
    logits_o: jax.Array
    cfg: SparseConfig = eqx.field(static=True)


def _hard_with_ste(s: jax.Array, threshold: float) -> jax.Array:
    """Straight-through estimator: forward is ``(σ(s) > thr)``,
    backward is ``σ(s)``. Implemented via ``jax.lax.stop_gradient``
    on the difference, so the gradient flows through the sigmoid
    branch only."""
    soft = jax.nn.sigmoid(s)
    hard = (soft > threshold).astype(soft.dtype)
    return soft + jax.lax.stop_gradient(hard - soft)


def _effective_weight(
    w: jax.Array, logits: jax.Array, cfg: SparseConfig,
) -> jax.Array:
    """``W ⊙ σ(s)`` (soft) or ``W ⊙ STE(σ(s) > thr)`` (hard).

    Inactive targets (logits with a zero-width axis) skip the
    multiply entirely and return the bare ``w``.
    """
    if logits.shape[-1] == 0:
        return w
    mask = (
        _hard_with_ste(logits, cfg.threshold)
        if cfg.hard else jax.nn.sigmoid(logits)
    )
    return w * mask.astype(w.dtype)


class SparseModel(eqx.Module):
    """Frozen backbone with per-element learnable sparse masks on the
    attention projections."""

    backbone: PAWNModel
    sparse: SparseParams

    def __call__(self, tokens: jax.Array, attn_mask: jax.Array) -> jax.Array:
        cfg = self.sparse.cfg
        bb = self.backbone
        effective = eqx.tree_at(
            lambda m: (m.wq, m.wk, m.wv, m.wo),
            bb,
            (
                _effective_weight(bb.wq, self.sparse.logits_q, cfg),
                _effective_weight(bb.wk, self.sparse.logits_k, cfg),
                _effective_weight(bb.wv, self.sparse.logits_v, cfg),
                _effective_weight(bb.wo, self.sparse.logits_o, cfg),
            ),
        )
        return effective(tokens, attn_mask)


def init_sparse_model(
    backbone: PAWNModel, cfg: SparseConfig, key: jax.Array,
) -> SparseModel:
    """Initialise mask logits at ``cfg.init_logit`` (≈ identity at
    init in soft mode). Inactive targets get zero-width sentinels.
    ``key`` is unused — initialisation is deterministic."""
    del key
    d = backbone.cfg.d_model
    n_layers = backbone.cfg.n_layers
    is_t = {t: (t in cfg.targets) for t in _VALID_TARGETS}

    def _init_logit(active: bool) -> jax.Array:
        if not active:
            return jnp.zeros((n_layers, d, 0), dtype=jnp.float32)
        return jnp.full(
            (n_layers, d, d), cfg.init_logit, dtype=jnp.float32,
        )

    return SparseModel(
        backbone=backbone,
        sparse=SparseParams(
            logits_q=_init_logit(is_t["q"]),
            logits_k=_init_logit(is_t["k"]),
            logits_v=_init_logit(is_t["v"]),
            logits_o=_init_logit(is_t["o"]),
            cfg=cfg,
        ),
    )


def adapter_filter(model: SparseModel) -> SparseModel:
    """``True`` on every ``model.sparse.*`` array leaf, ``False``
    elsewhere."""
    false_tree = jax.tree_util.tree_map(lambda _: False, model)
    sparse_true = jax.tree_util.tree_map(eqx.is_inexact_array, model.sparse)
    return eqx.tree_at(lambda m: m.sparse, false_tree, sparse_true)
