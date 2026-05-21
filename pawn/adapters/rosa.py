"""RoSA (Robust Sparse Adaptation) — JAX port of `Nikdan et al. 2024
<https://arxiv.org/abs/2401.04679>`_.

Combines a low-rank adapter (LoRA) and a gradient-informed sparse
adapter on each frozen attention projection::

    W_eff = W + (scale · A @ B) + (Δ ⊙ mask)

The legacy module ran a three-phase training procedure:

1. **LoRA warm-up.** Sparse delta is zero + mask is all-False;
   train only LoRA.
2. **Mask generation.** Compute per-parameter gradient magnitudes
   under the warmed-up LoRA, then take the top-k by absolute
   gradient and freeze that mask.
3. **Joint training.** Train LoRA + sparse delta together; the
   mask is fixed at this point.

This JAX port ships the *parameterisation* (RoSAModel +
RoSAParams + adapter_filter) and a deterministic mask-setting
helper (``set_mask(model, target → bool-mask)``). The three-phase
training schedule itself is a trainer-side concern — the adapter
trainer driver picks which subset of leaves to update at each
phase via ``adapter_filter`` + optax's masked-gradient surface.
A "RoSA training" follow-up PR wires up the phase scheduler.

PyTree layout::

  RoSAModel
  ├── backbone: PAWNModel    (frozen)
  └── rosa: RoSAParams
       ├── a_q, a_k, a_v, a_o            ([n_layers, d, rank])
       ├── b_q, b_k, b_v, b_o            ([n_layers, rank, d])
       ├── delta_q, delta_k, delta_v, delta_o   ([n_layers, d, d])
       └── mask_q, mask_k, mask_v, mask_o       ([n_layers, d, d] bool)

Identity-at-init: B=0, Δ=0, mask=all-False — the wrapped forward
is bit-identical to the bare backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from pawn.model import PAWNModel

_VALID_TARGETS = ("q", "k", "v", "o")


@dataclass(frozen=True)
class RoSAConfig:
    """RoSA hyperparameters.

    Args:
        rank: LoRA-leg rank.
        targets: attention projections to wrap. Subset of
            ``("q", "k", "v", "o")``.
        alpha: LoRA scaling; effective scale is ``alpha / rank``.
            ``None`` → ``alpha = rank`` (scale 1.0).
    """

    rank: int
    targets: tuple[str, ...] = ("q", "k", "v", "o")
    alpha: float | None = None

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank={self.rank} must be positive")
        if isinstance(self.targets, str):
            raise ValueError(
                f"targets must be a tuple of strings, not a bare str "
                f"({self.targets!r}); use e.g. ('q', 'v')"
            )
        if not self.targets:
            raise ValueError("RoSA needs at least one target projection")
        for t in self.targets:
            if t not in _VALID_TARGETS:
                raise ValueError(
                    f"unknown RoSA target {t!r}; expected subset of "
                    f"{_VALID_TARGETS}"
                )
        if self.alpha is not None and self.alpha <= 0:
            raise ValueError(
                f"alpha={self.alpha} must be positive (or None)"
            )

    @property
    def scale(self) -> float:
        a = self.alpha if self.alpha is not None else float(self.rank)
        return a / self.rank


class RoSAParams(eqx.Module):
    """Per-target stacked LoRA A/B + sparse Δ + boolean mask."""

    a_q: jax.Array
    a_k: jax.Array
    a_v: jax.Array
    a_o: jax.Array
    b_q: jax.Array
    b_k: jax.Array
    b_v: jax.Array
    b_o: jax.Array
    delta_q: jax.Array
    delta_k: jax.Array
    delta_v: jax.Array
    delta_o: jax.Array
    mask_q: jax.Array
    mask_k: jax.Array
    mask_v: jax.Array
    mask_o: jax.Array
    cfg: RoSAConfig = eqx.field(static=True)


class RoSAModel(eqx.Module):
    """Frozen PAWN backbone composed with LoRA + sparse RoSA adapter."""

    backbone: PAWNModel
    rosa: RoSAParams

    def __call__(self, tokens: jax.Array, attn_mask: jax.Array) -> jax.Array:
        return self.effective_backbone()(tokens, attn_mask)

    def effective_backbone(self) -> PAWNModel:
        """``W + (scale · A @ B) + (Δ ⊙ mask)`` folded into each target.

        Inactive targets (zero-width sentinel ``A`` / ``B``) skip the
        LoRA delta entirely; the sparse leg is similarly skipped when
        mask is all-zero.
        """
        scale = jnp.float32(self.rosa.cfg.scale)
        backbone_dtype = self.backbone.wq.dtype

        deltas: dict[str, jax.Array | None] = {}
        for tgt in _VALID_TARGETS:
            a = getattr(self.rosa, f"a_{tgt}")
            b = getattr(self.rosa, f"b_{tgt}")
            d = getattr(self.rosa, f"delta_{tgt}")
            m = getattr(self.rosa, f"mask_{tgt}")
            if a.shape[-1] == 0:
                # Inactive target: LoRA delta vanishes; the sparse
                # leg also has shape (L, d, d) but Δ = 0 + mask =
                # False at init → contributes 0. Compose anyway so
                # the forward shape stays uniform.
                lora_delta = jnp.zeros_like(d)
            else:
                lora_delta = (scale * jnp.einsum("lir,lro->lio", a, b)).astype(
                    backbone_dtype,
                )
            sparse_delta = (d * m.astype(d.dtype)).astype(backbone_dtype)
            deltas[tgt] = lora_delta + sparse_delta

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


def _init_pair(
    n_layers: int, d_in: int, d_out: int, rank: int,
    *, is_target: bool, key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """LoRA A/B init: A ~ N(0, 0.02²); B = 0. Inactive → zero-width."""
    if not is_target:
        return (
            jnp.zeros((n_layers, d_in, 0), dtype=jnp.float32),
            jnp.zeros((n_layers, 0, d_out), dtype=jnp.float32),
        )
    a = jax.random.normal(key, (n_layers, d_in, rank), dtype=jnp.float32) * 0.02
    b = jnp.zeros((n_layers, rank, d_out), dtype=jnp.float32)
    return a, b


def init_rosa_model(
    backbone: PAWNModel, cfg: RoSAConfig, key: jax.Array,
) -> RoSAModel:
    """Initialise RoSA. LoRA B = 0, sparse Δ = 0, mask = False — so
    the wrapped forward is bit-identical to the bare backbone at
    step 0. Per-target keys are split independently so a single-target
    run produces the same A as a corresponding subset run."""
    d = backbone.cfg.d_model
    n_layers = backbone.cfg.n_layers
    rank = cfg.rank
    is_t = {t: (t in cfg.targets) for t in _VALID_TARGETS}
    keys = jax.random.split(key, len(_VALID_TARGETS))
    pairs = {
        t: _init_pair(n_layers, d, d, rank, is_target=is_t[t], key=keys[i])
        for i, t in enumerate(_VALID_TARGETS)
    }
    zeros_dd = jnp.zeros((n_layers, d, d), dtype=jnp.float32)
    false_dd = jnp.zeros((n_layers, d, d), dtype=jnp.bool_)
    rosa = RoSAParams(
        a_q=pairs["q"][0], a_k=pairs["k"][0],
        a_v=pairs["v"][0], a_o=pairs["o"][0],
        b_q=pairs["q"][1], b_k=pairs["k"][1],
        b_v=pairs["v"][1], b_o=pairs["o"][1],
        delta_q=zeros_dd, delta_k=zeros_dd, delta_v=zeros_dd, delta_o=zeros_dd,
        mask_q=false_dd, mask_k=false_dd, mask_v=false_dd, mask_o=false_dd,
        cfg=cfg,
    )
    return RoSAModel(backbone=backbone, rosa=rosa)


def set_mask(
    model: RoSAModel, masks: dict[str, jax.Array],
) -> RoSAModel:
    """Replace the sparse-leg masks with caller-computed bool arrays.

    Phase-2 of the legacy training procedure (Algorithm 1 in Nikdan
    et al.) computes per-parameter gradient magnitudes under a
    warmed-up LoRA, then sets the mask to True on the top-k entries.
    This helper exposes the mask-swap API for the trainer to plug
    those masks in once they're computed. Targets absent from
    ``masks`` keep their previous mask.

    Args:
        masks: ``{"q": bool_array_of_shape_(n_layers, d, d), ...}``.
    """
    rosa = model.rosa
    for tgt in _VALID_TARGETS:
        if tgt in masks:
            new = masks[tgt]
            expected = getattr(rosa, f"mask_{tgt}").shape
            if new.shape != expected:
                raise ValueError(
                    f"mask for {tgt!r} has shape {new.shape}, "
                    f"expected {expected}"
                )
            rosa = eqx.tree_at(
                lambda r, n=f"mask_{tgt}": getattr(r, n),
                rosa, new.astype(jnp.bool_),
            )
    return eqx.tree_at(lambda m: m.rosa, model, rosa)


def _filter_with(
    model: RoSAModel,
    *,
    lora_trainable: bool,
    delta_trainable: bool,
) -> RoSAModel:
    """Build a Python-bool filter spec for one of the three RoSA
    training phases.

    Only ``model.rosa.cfg.targets`` get marked trainable — inactive
    targets' ``delta_*`` arrays (full-shape ``(L, d, d)`` zeros,
    structurally unused in the forward) stay frozen, so Adam doesn't
    allocate moment buffers for them. Inactive targets' ``a_*`` /
    ``b_*`` are zero-width sentinels (no buffer cost regardless) but
    we exclude them too for symmetry.

    Bool ``mask_*`` arrays are always False (they're not
    differentiable parameters; the trainer rewrites them via
    ``set_mask`` between phases).
    """
    false_tree = jax.tree_util.tree_map(lambda _: False, model)
    active_targets = model.rosa.cfg.targets

    def _set(tree: RoSAModel, field: str) -> RoSAModel:
        return eqx.tree_at(
            lambda t, f=field: getattr(t.rosa, f),
            tree, replace=True,
        )

    tree = false_tree
    for tgt in active_targets:
        if lora_trainable:
            tree = _set(tree, f"a_{tgt}")
            tree = _set(tree, f"b_{tgt}")
        if delta_trainable:
            tree = _set(tree, f"delta_{tgt}")
    return tree


def adapter_filter(model: RoSAModel) -> RoSAModel:
    """Phase-3 joint filter: ``True`` on every active-target
    ``model.rosa.{a,b,delta}_*`` inexact array, ``False`` on the
    bool ``mask_*`` arrays and on every inactive-target leaf.

    The phase-1 (LoRA warmup) and phase-2 (sparse mask gen) filters
    are exposed as ``lora_only_adapter_filter`` and
    ``sparse_only_adapter_filter`` — see the module docstring for the
    three-phase RoSA training schedule."""
    return _filter_with(model, lora_trainable=True, delta_trainable=True)


def lora_only_adapter_filter(model: RoSAModel) -> RoSAModel:
    """RoSA Phase 1 filter: ``True`` only on the active-target LoRA
    A, B leaves.

    Used during the LoRA warmup phase — sparse Δ stays at its init
    value (zeros) and doesn't accumulate gradient updates. ``mask_*``
    arrays are False (always not-trainable; rewritten via
    ``set_mask``)."""
    return _filter_with(model, lora_trainable=True, delta_trainable=False)


def sparse_only_adapter_filter(model: RoSAModel) -> RoSAModel:
    """RoSA Phase 2 filter: ``True`` only on the active-target sparse
    Δ leaves.

    The trainer never actually optimises with this filter — Phase 2 is
    a single forward+backward to extract ``|dL/dΔ|`` per target. The
    filter is exposed so the gradient is computed against only the Δ
    leaves (XLA dead-code-eliminates dL/dA, dL/dB for the §6.1
    backward-pass FLOP cut, even though we're not training)."""
    return _filter_with(model, lora_trainable=False, delta_trainable=True)


def compute_phase2_mask(
    model: RoSAModel, dl_ddelta: dict[str, jax.Array], top_k_frac: float,
) -> dict[str, jax.Array]:
    """Build Phase-2 sparse-leg masks from the per-target gradient
    magnitudes.

    Picks the top ``top_k_frac`` entries by ``|dL/dΔ_{ij}|`` per
    target and returns ``{target: bool_array_of_shape_(L, d, d)}`` for
    the active targets in ``model.rosa.cfg.targets``. Inactive targets
    aren't returned (the caller skips them in ``rosa_set_mask``).

    Args:
        model: the warmed-up RoSA model from Phase 1.
        dl_ddelta: ``{target: dL/dΔ_array}`` for each active target,
            same shape as the corresponding ``model.rosa.delta_*``
            array.
        top_k_frac: fraction of entries to keep (e.g. ``0.01`` for
            top 1% by magnitude).

    Returns:
        ``{target: bool_array}`` — pass to ``rosa_set_mask`` to apply.
    """
    if not 0.0 < top_k_frac <= 1.0:
        raise ValueError(f"top_k_frac={top_k_frac} must be in (0, 1]")
    masks: dict[str, jax.Array] = {}
    for tgt in model.rosa.cfg.targets:
        grad = dl_ddelta[tgt]
        # Magnitude-rank per layer (rather than globally across L *
        # d * d) so each layer contributes its own top-k. The legacy
        # implementation rank-ordered per-layer; preserving that.
        n_layers = grad.shape[0]
        flat = jnp.abs(grad).reshape(n_layers, -1)  # (L, d * d)
        n_entries = flat.shape[1]
        k = max(1, int(n_entries * top_k_frac))
        # Pick exactly the top ``k`` indices per layer via argsort
        # rather than a >= threshold comparison. A threshold-based
        # selection over-counts when many entries tie at the
        # boundary value — pathologically, all-zero gradients tie
        # at 0 and ``>= 0`` selects every entry, producing
        # density=1.0 and defeating the sparsity guarantee. The
        # argsort path gives exactly ``k`` entries per layer
        # regardless of ties (ties are broken by index order, which
        # is fine — the legacy behaviour was identical in this
        # respect).
        top_k_indices = jnp.argsort(flat, axis=1)[:, n_entries - k:]
        layer_idx = jnp.arange(n_layers)[:, None].repeat(k, axis=1)
        mask_flat = jnp.zeros_like(flat, dtype=jnp.bool_)
        mask_flat = mask_flat.at[layer_idx, top_k_indices].set(True)
        masks[tgt] = mask_flat.reshape(grad.shape)
    return masks
