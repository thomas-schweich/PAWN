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
  ├── backbone: PAWNModel   (frozen — see ``adapter_filter``)
  └── lora: LoRAParams
       ├── a_q, a_k, a_v, a_o  ([n_layers, d_in, rank])
       └── b_q, b_k, b_v, b_o  ([n_layers, rank, d_out])

Filtering: ``adapter_filter(model)`` builds a PyTree-shaped filter
spec with ``True`` only on ``model.lora.*`` array leaves and
``False`` on the backbone subtree. The canonical two-tier
partition is::

    trainable, frozen = eqx.partition(model, adapter_filter(model))

``eqx.filter_grad`` over ``trainable`` then differentiates only the
adapter parameters; XLA dead-code-eliminates the backbone
weight-gradient computations.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        if isinstance(self.targets, str):
            # A bare ``str`` is technically iterable (yields chars)
            # and would silently slip past the validation loop —
            # ``targets="qv"`` would be treated as ``("q", "v")`` but
            # only by accident, and ``targets="qkvo"`` would be valid
            # while ``targets="ab"`` would raise on the second char
            # with a confusing error. Reject explicitly.
            raise ValueError(
                f"targets must be a tuple of strings, not a bare str "
                f"({self.targets!r}); use e.g. ('q', 'v')"
            )
        if not self.targets:
            raise ValueError("LoRA needs at least one target projection")
        for t in self.targets:
            if t not in _VALID_TARGETS:
                raise ValueError(
                    f"unknown LoRA target {t!r}; expected subset of "
                    f"{_VALID_TARGETS}"
                )
        if self.alpha is not None and self.alpha <= 0:
            # alpha == 0 → scale == 0 (LoRA path silently nullified);
            # negative → flips the update sign vs the paper.
            raise ValueError(f"alpha={self.alpha} must be positive (or None)")

    @property
    def scale(self) -> float:
        """Effective ``alpha / rank`` scaling factor. When ``alpha``
        is ``None`` (the default), the implementation uses
        ``alpha = rank`` so the effective scale is 1.0."""
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
    per layer (only for targets actually requested in
    ``LoRAConfig.targets``) and runs the backbone forward with them.
    Cost per forward: one ``[n_layers, d_in, d_out]`` add per targeted
    projection plus the rank-``r`` matmul ``A @ B``. At LoRA-rank 4
    on PAWN-base (d=512, L=8, 4 targets, fp32) the deltas occupy
    ~33 MB — small relative to the dominant scan-carry activation
    cost of the full forward.

    ``backbone`` is treated as frozen via ``adapter_filter`` —
    ``eqx.partition(model, adapter_filter(model))`` produces a
    trainable subtree containing only the ``LoRAParams`` leaves.
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
        backbone_dtype = self.backbone.wq.dtype
        deltas: dict[str, jax.Array | None] = {}
        for tgt in _VALID_TARGETS:
            a = getattr(self.lora, f"a_{tgt}")
            b = getattr(self.lora, f"b_{tgt}")
            if a.shape[-1] == 0:
                # Sentinel: this target wasn't requested. Skip the
                # einsum + add entirely — XLA's algebraic simplifier
                # may not always constant-fold a zero-width
                # contraction through a subsequent dense add, so the
                # explicit ``None`` skip-path is cheaper and clearer.
                deltas[tgt] = None
                continue
            # Cast AFTER the multiply so ``scale`` (fp32) doesn't
            # promote a bf16 backbone delta back to fp32. The earlier
            # ``scale * jnp.einsum(...).astype(dt)`` parsed as
            # ``scale * (einsum.astype(dt))`` (attribute access binds
            # tighter than ``*``), then float32 ``scale`` promoted the
            # product back to float32 — a bf16 backbone forward would
            # then fail at ``lax.scan``'s dtype contract.
            deltas[tgt] = (scale * jnp.einsum("lir,lro->lio", a, b)).astype(
                backbone_dtype
            )

        # ``eqx.tree_at`` returns a new module with the swapped leaves.
        def _maybe_add(w: jax.Array, d: jax.Array | None) -> jax.Array:
            return w if d is None else w + d

        return eqx.tree_at(
            lambda m: (m.wq, m.wk, m.wv, m.wo),
            self.backbone,
            (
                _maybe_add(self.backbone.wq, deltas["q"]),
                _maybe_add(self.backbone.wk, deltas["k"]),
                _maybe_add(self.backbone.wv, deltas["v"]),
                _maybe_add(self.backbone.wo, deltas["o"]),
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
    only the LoRA leaves; the backbone is filtered out via
    ``adapter_filter``.
    """
    d = backbone.cfg.d_model
    n_layers = backbone.cfg.n_layers
    rank = cfg.rank
    is_t = {t: (t in cfg.targets) for t in _VALID_TARGETS}
    # One key per target for reproducible per-target seeding regardless
    # of which subset is active (so a run with ``targets=("q",)`` and
    # a follow-up with ``targets=("q", "v")`` see the same ``a_q``).
    keys = jax.random.split(key, len(_VALID_TARGETS))
    pairs: dict[str, tuple[jax.Array, jax.Array]] = {
        t: _init_lora_pair(
            n_layers, d, d, rank, is_target=is_t[t], key=keys[i]
        )
        for i, t in enumerate(_VALID_TARGETS)
    }
    lora = LoRAParams(
        a_q=pairs["q"][0], a_k=pairs["k"][0], a_v=pairs["v"][0], a_o=pairs["o"][0],
        b_q=pairs["q"][1], b_k=pairs["k"][1], b_v=pairs["v"][1], b_o=pairs["o"][1],
        cfg=cfg,
    )
    return LoRAModel(backbone=backbone, lora=lora)


def adapter_filter(model: LoRAModel) -> LoRAModel:
    """Build a PyTree filter that is ``True`` only on ``model.lora.*``
    array leaves.

    Use with ``eqx.partition(model, adapter_filter(model))`` or
    ``eqx.filter_grad(..., filter_spec=...)``. The returned tree has
    the same structure as ``model`` with ``bool`` leaves; backbone
    leaves are ``False``, adapter leaves are ``True``. The ``bool``
    leaves take the place of ``jax.Array`` leaves in the original,
    which is exactly what Equinox's filter machinery expects — the
    return type annotation reuses ``LoRAModel`` to make the tree
    structure visible to callers.
    """
    # Build a tree of False everywhere, then flip True on lora.* arrays.
    false_tree = jax.tree_util.tree_map(lambda _: False, model)
    # Replace the ``lora`` subtree with True-on-arrays.
    lora_true = jax.tree_util.tree_map(eqx.is_inexact_array, model.lora)
    return eqx.tree_at(lambda m: m.lora, false_tree, lora_true)
