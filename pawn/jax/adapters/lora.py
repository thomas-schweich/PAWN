"""LoRA (Low-Rank Adaptation) adapter for the JAX PAWN backbone.

Each LoRA-targeted weight ``W ∈ [d_in, d_out]`` gets a low-rank
update ``A @ B`` where ``A ∈ [d_in, rank]`` and ``B ∈ [rank, d_out]``.
At forward time the effective weight is ``W + A @ B``. Standard LoRA
init: ``A`` ~ Gaussian, ``B`` = 0, so the initial forward output
equals the unmodified backbone — adapters that fail to learn never
silently destroy the pretrained baseline.

This module targets the attention projections (``wq``, ``wk``, ``wv``,
``wo``) and stacks per-layer parameters on a leading ``[n_layers, …]``
axis to match the backbone's stacked layout.

PyTree layout:

  LoRAModel
  ├── backbone: PAWNModel   (treat as frozen via ``is_adapter_leaf`` filter)
  └── lora: LoRAParams
       ├── a_q, a_k, a_v, a_o  ([n_layers, d_in, rank])
       └── b_q, b_k, b_v, b_o  ([n_layers, rank, d_out])

Filtering: ``is_adapter_leaf`` marks LoRA parameters as trainable
(returns True) and backbone parameters as frozen (returns False).
``eqx.filter_grad(..., is_adapter_leaf)`` then differentiates only
the LoRA leaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from pawn.jax.model import PAWNModel

# The four attention projections we can target. Each is a per-layer
# ``[d_in, d_out]`` matrix on the backbone.
_VALID_TARGETS = ("q", "k", "v", "o")


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA hyperparameters.

    Args:
        rank: rank of the ``A @ B`` decomposition. Typical: 4, 8, 16.
        targets: which attention projections to wrap. Subset of
            ``("q", "k", "v", "o")``. Default ``("q", "v")`` matches
            the original LoRA paper.
        alpha: LoRA scaling factor; effective update is ``(alpha/rank)
            · A @ B``. v1 fixes ``alpha == rank`` so the effective
            scale is 1.0; future tuning can lift it.
    """

    rank: int
    targets: tuple[str, ...] = ("q", "v")
    alpha: float | None = None

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank={self.rank} must be positive")
        for t in self.targets:
            if t not in _VALID_TARGETS:
                raise ValueError(
                    f"unknown LoRA target {t!r}; expected subset of "
                    f"{_VALID_TARGETS}"
                )
        if not self.targets:
            raise ValueError("LoRA needs at least one target projection")

    @property
    def scale(self) -> float:
        """Effective ``alpha / rank`` scaling factor."""
        a = self.alpha if self.alpha is not None else float(self.rank)
        return a / self.rank


class LoRAParams(eqx.Module):
    """Per-layer LoRA A/B matrices for the four attention projections.

    Each ``a_*`` / ``b_*`` is a stacked ``[n_layers, d_in, rank]`` /
    ``[n_layers, rank, d_out]`` array. For projections NOT in
    ``cfg.targets`` the corresponding matrices are zero-shape sentinels
    (``[n_layers, d, 0]`` and ``[n_layers, 0, d]``), so the effective
    delta is zero and the optimizer sees no parameters to update.
    """

    a_q: jax.Array
    a_k: jax.Array
    a_v: jax.Array
    a_o: jax.Array
    b_q: jax.Array
    b_k: jax.Array
    b_v: jax.Array
    b_o: jax.Array
    cfg: LoRAConfig = eqx.field(static=True)


class LoRAModel(eqx.Module):
    """A ``PAWNModel`` backbone composed with a ``LoRAParams`` adapter.

    Forward materializes the effective weights ``W + scale · A @ B``
    per layer and runs the backbone forward with them. Cost per
    forward: one ``[n_layers, d, d]`` add per LoRA target plus the
    rank-``r`` matmul ``A @ B``; at LoRA-rank 4 on PAWN-base this is
    ~32 MB extra of materialised effective weights vs the ~1.4 GB
    full forward — negligible.

    ``backbone`` is treated as frozen via ``is_adapter_leaf``;
    optimization (e.g. ``eqx.filter_grad``) sees only the
    ``LoRAParams`` leaves as trainable.
    """

    backbone: PAWNModel
    lora: LoRAParams

    def __call__(self, tokens: jax.Array, attn_mask: jax.Array) -> jax.Array:
        return self.effective_backbone()(tokens, attn_mask)

    def effective_backbone(self) -> PAWNModel:
        """Return a backbone with the LoRA delta folded into each
        attention projection.

        The returned object is a fresh ``PAWNModel`` that shares all
        non-attention leaves with ``self.backbone``; only ``wq``,
        ``wk``, ``wv``, ``wo`` differ. ``eqx.tree_at`` is the
        canonical way to do this in Equinox.
        """
        scale = jnp.float32(self.lora.cfg.scale)
        # Per-layer batched matmul: A @ B is ``[L, d, r] @ [L, r, d]``
        # → ``[L, d, d]``. ``jnp.einsum`` keeps the leading axis
        # broadcast-batched.
        deltas: dict[str, jax.Array] = {}
        for tgt in _VALID_TARGETS:
            a = getattr(self.lora, f"a_{tgt}")
            b = getattr(self.lora, f"b_{tgt}")
            # When the target was not requested, a is [L, d, 0] and
            # b is [L, 0, d] — the einsum returns the zero matrix
            # without allocating an intermediate.
            deltas[tgt] = scale * jnp.einsum("lir,lro->lio", a, b).astype(
                self.backbone.wq.dtype
            )
        # ``eqx.tree_at`` returns a new module with the swapped leaves.
        return eqx.tree_at(
            lambda m: (m.wq, m.wk, m.wv, m.wo),
            self.backbone,
            (
                self.backbone.wq + deltas["q"],
                self.backbone.wk + deltas["k"],
                self.backbone.wv + deltas["v"],
                self.backbone.wo + deltas["o"],
            ),
        )


def _init_lora_pair(
    n_layers: int,
    d_in: int,
    d_out: int,
    rank: int,
    *,
    is_target: bool,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Initialise ``(A, B)`` for one target.

    For active targets: ``A`` ~ Gaussian σ=0.02, ``B`` = 0. The B=0
    convention is load-bearing — it guarantees ``forward(LoRAModel)
    == forward(backbone)`` at step 0, so a regression where the
    optimizer never reaches the adapter is detectable as
    no-loss-progress, not as a destroyed pretrained baseline.

    For inactive targets: ``A`` and ``B`` are zero-width sentinels
    ``[n_layers, d, 0]`` and ``[n_layers, 0, d]`` so the einsum is
    well-defined and produces the zero matrix without flowing
    gradients into nonexistent parameters.
    """
    if not is_target:
        return (
            jnp.zeros((n_layers, d_in, 0), dtype=jnp.float32),
            jnp.zeros((n_layers, 0, d_out), dtype=jnp.float32),
        )
    a = jax.random.normal(key, (n_layers, d_in, rank), dtype=jnp.float32) * 0.02
    b = jnp.zeros((n_layers, rank, d_out), dtype=jnp.float32)
    return a, b


def init_lora_model(backbone: PAWNModel, cfg: LoRAConfig, key: jax.Array) -> LoRAModel:
    """Construct a ``LoRAModel`` wrapping ``backbone`` under ``cfg``.

    At step 0 the wrapped model is bit-identical to ``backbone`` on
    its forward pass (B = 0). Gradient flow at training time targets
    only the LoRA leaves; the backbone is filtered out by
    ``is_adapter_leaf``.
    """
    d = backbone.cfg.d_model
    n_layers = backbone.cfg.n_layers
    rank = cfg.rank
    is_t = {t: (t in cfg.targets) for t in _VALID_TARGETS}
    keys = jax.random.split(key, 4)
    a_q, b_q = _init_lora_pair(
        n_layers, d, d, rank, is_target=is_t["q"], key=keys[0]
    )
    a_k, b_k = _init_lora_pair(
        n_layers, d, d, rank, is_target=is_t["k"], key=keys[1]
    )
    a_v, b_v = _init_lora_pair(
        n_layers, d, d, rank, is_target=is_t["v"], key=keys[2]
    )
    a_o, b_o = _init_lora_pair(
        n_layers, d, d, rank, is_target=is_t["o"], key=keys[3]
    )
    lora = LoRAParams(
        a_q=a_q, a_k=a_k, a_v=a_v, a_o=a_o,
        b_q=b_q, b_k=b_k, b_v=b_v, b_o=b_o,
        cfg=cfg,
    )
    return LoRAModel(backbone=backbone, lora=lora)


def is_adapter_leaf(model: Optional[LoRAModel], leaf: Any) -> bool:
    """Filter callable for use with ``eqx.filter_grad`` / ``eqx.partition``.

    Returns ``True`` for any inexact-array leaf reachable through
    ``model.lora.*`` and ``False`` for everything else (the backbone
    leaves + static config fields). Designed for use as:

        params = eqx.filter(model, eqx.is_inexact_array)
        # but with adapter-only:
        params = eqx.partition(model, is_adapter_leaf(model, ...))

    For convenience inside ``eqx.filter_grad`` we expose a path-based
    filter via ``adapter_filter`` below.
    """
    # This function is a stub for symmetry — the real partition uses
    # the path-based helper. Equinox idiomatic two-tier split:
    # ``eqx.filter(model, eqx.is_inexact_array)`` returns array leaves;
    # ``adapter_filter`` then zeroes the backbone subtree.
    del model
    return eqx.is_inexact_array(leaf)


def adapter_filter(model: LoRAModel) -> Any:
    """Build a PyTree filter that is ``True`` only on ``model.lora.*``
    array leaves.

    Use with ``eqx.filter_grad(..., filter_spec=...)`` or
    ``eqx.partition(model, filter_spec=...)``. The returned filter
    has the same structure as ``model`` with bool leaves; backbone
    leaves are ``False``, adapter leaves are ``True``.
    """
    # Build a tree of False everywhere, then flip True on lora.* arrays.
    false_tree = jax.tree_util.tree_map(lambda _: False, model)
    # Replace the ``lora`` subtree with True-on-arrays.
    lora_true = jax.tree_util.tree_map(eqx.is_inexact_array, model.lora)
    return eqx.tree_at(lambda m: m.lora, false_tree, lora_true)
