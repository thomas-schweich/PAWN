"""Adapter trainer — two-tier PyTree partition + K-step lax.scan.

The trainer takes a frozen backbone + an adapter PyTree and trains
only the adapter's parameters. The two-tier partition uses
:func:`eqx.partition` to separate trainable (adapter) from frozen
(backbone) — :func:`jax.grad` then differentiates only the trainable
PyTree, and XLA dead-code-eliminates the backbone weight-gradients
(~33% backward-pass FLOP cut per plan §5).

Public surface:

- :class:`AdapterTrainState` — (backbone, adapter, opt_state, step,
  key). The backbone is held but never updated.
- :func:`make_adapter_train_step` — JIT'd single training step with
  the strategy's ``apply`` function baked in.
- :func:`make_adapter_scan_step` — K-step :func:`jax.lax.scan` wrapper.
- :data:`STRATEGIES` — the dispatch table mapping ``--strategy`` value
  to ``(init, apply, filter)``.
- :func:`forward_eval` — jitted forward-only eval (no gradients).

RoSA orchestration:

- :func:`rosa_phase1_to_phase3` swings the adapter PyTree from Phase 1
  (LoRA warmup) to Phase 3 (joint training under fixed mask): zeroes
  LoRA-B + sparse deltas, installs the supplied masks, and flips the
  ``lora_active`` / ``sparse_active`` toggles based on ``cfg.mode``.
- :func:`generate_rosa_masks` runs ``mask_samples`` gradient-only
  forward passes with all-True sparse masks active, accumulates
  ``|grad|^grad_alpha`` per delta position, and returns the top-k
  (by density) boolean masks. Algorithm 1 from the RoSA paper.

The schedule's ``state.step`` counter is preserved across Phase 2→3
re-init so the metrics log stays monotonic; Optax's internal moments
reset by design (the Phase 3 trainable surface is shape-different from
Phase 1's). The training script in
``scripts/train_jax_adapter.py`` is what strings the three phases
together — adapter trainers for the other 7 strategies stay on the
simple :func:`make_adapter_train_step` path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int

from pawn.adapters import (
    BottleneckAdapter,
    FiLMAdapter,
    HybridAdapter,
    LoRAAdapter,
    RoSAAdapter,
    SparseAdapter,
    SpecializedCLMAdapter,
    UnfreezeAdapter,
    bottleneck,
    film,
    hybrid,
    lora,
    rosa,
    sparse,
    specialized_clm,
    unfreeze,
)
from pawn.model import EffectiveCallable, PAWNModel
from pawn.trainer import Batch, cross_entropy_loss

__all__ = [
    "AdapterTrainState",
    "STRATEGIES",
    "dispatch_init",
    "dispatch_apply",
    "dispatch_filter",
    "make_adapter_train_step",
    "make_adapter_scan_step",
    "make_forward_eval",
    "forward_eval",
    "generate_rosa_masks",
    "rosa_phase1_to_phase3",
]


# ---------------------------------------------------------------------------
# Strategy dispatch table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyEntry:
    init: Callable[..., Any]
    # Returns "anything callable like a PAWNModel". Plain weight-fold
    # adapters return a PAWNModel; bottleneck-style adapters return a
    # wrapper module that satisfies :class:`EffectiveCallable` but is
    # not itself a PAWNModel.
    apply: Callable[[PAWNModel, Any], EffectiveCallable]
    filter: Callable[[Any], Any]


STRATEGIES: dict[str, StrategyEntry] = {
    "lora": StrategyEntry(lora.init_lora_adapter, lora.apply_lora, lora.lora_filter),
    "film": StrategyEntry(film.init_film_adapter, film.apply_film, film.film_filter),
    "bottleneck": StrategyEntry(
        bottleneck.init_bottleneck_adapter,
        bottleneck.apply_bottleneck,
        bottleneck.bottleneck_filter,
    ),
    "hybrid": StrategyEntry(
        hybrid.init_hybrid_adapter, hybrid.apply_hybrid, hybrid.hybrid_filter
    ),
    "sparse": StrategyEntry(
        sparse.init_sparse_adapter, sparse.apply_sparse, sparse.sparse_filter
    ),
    "rosa": StrategyEntry(
        rosa.init_rosa_adapter, rosa.apply_rosa, rosa.rosa_filter
    ),
    # Plan §10 S7 + CLAUDE.md adapter table treat the three RoSA modes as
    # distinct `--strategy` values. They share init/apply/filter — the
    # `mode` field on the config selects the variant. Whoever holds the
    # CLI surface (S13's scripts/train_jax_adapter.py) is expected to
    # default `rosa_mode` to match the strategy name when the user passes
    # `--strategy rosa-retro-sparse` etc.
    "rosa-retro-sparse": StrategyEntry(
        rosa.init_rosa_adapter, rosa.apply_rosa, rosa.rosa_filter
    ),
    "rosa-retro-bottleneck": StrategyEntry(
        rosa.init_rosa_adapter, rosa.apply_rosa, rosa.rosa_filter
    ),
    "unfreeze": StrategyEntry(
        unfreeze.init_unfreeze_adapter,
        unfreeze.apply_unfreeze,
        unfreeze.unfreeze_filter,
    ),
    "specialized_clm": StrategyEntry(
        specialized_clm.init_specialized_clm_adapter,
        specialized_clm.apply_specialized_clm,
        specialized_clm.specialized_clm_filter,
    ),
}


def dispatch_init(strategy: str) -> Callable[..., Any]:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; valid: {sorted(STRATEGIES)}"
        )
    return STRATEGIES[strategy].init


def dispatch_apply(strategy: str) -> Callable[[PAWNModel, Any], EffectiveCallable]:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; valid: {sorted(STRATEGIES)}"
        )
    return STRATEGIES[strategy].apply


def dispatch_filter(strategy: str) -> Callable[[Any], Any]:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; valid: {sorted(STRATEGIES)}"
        )
    return STRATEGIES[strategy].filter


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------


class AdapterTrainState(eqx.Module):
    """State for adapter training.

    ``backbone`` is frozen (never updated by the optimizer);
    ``adapter`` holds the trainable parameters. ``opt_state`` tracks
    AdamW moments + clip + lr schedule. ``step`` is the JAX scalar
    counter.
    """

    backbone: PAWNModel
    adapter: Any  # one of the *Adapter types — eqx.Module
    opt_state: optax.OptState
    step: Int[Array, ""]
    key: jax.Array


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def _freeze_masked_unfreeze_slots(
    adapter: UnfreezeAdapter, backbone: PAWNModel
) -> UnfreezeAdapter:
    """Snap masked layer slots back to the backbone values.

    Background: AdamW's decoupled weight decay applies
    ``params -= lr * wd * params`` even when ``grad == 0``. The
    forward-side ``where(mask, adapter, backbone)`` in
    :func:`pawn.adapters.unfreeze.apply_unfreeze` already blocks masked
    slots from contributing to the loss, so their gradients are zero,
    but their buffers would still drift toward zero under
    ``weight_decay > 0``. The drift is forward-invisible (the
    ``where`` always reads backbone values for masked slots), but it
    pollutes the saved adapter state.

    This helper enforces the drift-free invariant explicitly: after
    every update, masked slots in the adapter's per-layer fields are
    overwritten with the backbone's values. Backbone values themselves
    are constants across training, so this is idempotent.
    """
    from pawn.adapters.unfreeze import _broadcast_mask
    mask = adapter.layer_mask

    def snap(
        bb_leaf: Float[Array, "n_layers ..."],
        ad_leaf: Float[Array, "n_layers ..."],
    ) -> Float[Array, "n_layers ..."]:
        m = _broadcast_mask(mask, bb_leaf.shape)
        return jnp.where(m, ad_leaf, bb_leaf)

    new_layers = jax.tree_util.tree_map(snap, backbone.layers, adapter.layers)
    return eqx.tree_at(lambda a: a.layers, adapter, new_layers)


def make_adapter_train_step(
    strategy: str,
    optimizer: optax.GradientTransformation,
    *,
    compute_dtype: "jnp.dtype | None" = None,
    use_sdpa: bool = False,
    use_flash: bool = False,
) -> Callable[
    [AdapterTrainState, Batch], tuple[AdapterTrainState, Float[Array, ""]]
]:
    """Build the JIT'd train step for ``strategy``.

    The closure captures the strategy's ``apply`` function so XLA can
    inline it. Gradients flow only through the adapter PyTree;
    ``backbone`` is held outside the autograd path via
    :func:`eqx.partition`.

    ``compute_dtype`` selects the AMP forward dtype (plan §5).
    ``use_sdpa`` is the parity #43 fast-path opt-in (XLA SDPA);
    ``use_flash`` opts into the Pallas Triton fused attention kernel.
    Both flags are only honoured for strategies whose ``apply_fn``
    returns a bare :class:`PAWNModel` (LoRA, sparse, FiLM, unfreeze,
    ...); bottleneck-style wrappers silently fall back to plain
    attention because the wrapper's ``__call__`` doesn't take the
    flags. ``use_flash`` wins when both are set.
    """

    apply_fn = dispatch_apply(strategy)
    is_unfreeze = strategy == "unfreeze"

    @eqx.filter_jit(donate="all")
    def step(
        state: AdapterTrainState, batch: Batch
    ) -> tuple[AdapterTrainState, Float[Array, ""]]:
        def loss_fn(adapter: Any) -> Float[Array, ""]:
            effective = apply_fn(state.backbone, adapter)
            return cross_entropy_loss(
                effective, batch,
                compute_dtype=compute_dtype,
                use_sdpa=use_sdpa, use_flash=use_flash,
            )

        loss, grads = eqx.filter_value_and_grad(loss_fn)(state.adapter)

        # Empty-batch guard (mirror of pretrain trainer).
        do_update = batch.loss_mask.sum() > 0

        def apply_update(args):
            grads_, opt_state_, adapter_ = args
            updates, new_opt = optimizer.update(grads_, opt_state_, adapter_)
            new_adapter = eqx.apply_updates(adapter_, updates)
            if is_unfreeze:
                new_adapter = _freeze_masked_unfreeze_slots(
                    new_adapter, state.backbone
                )
            return new_adapter, new_opt

        def skip_update(args):
            _, opt_state_, adapter_ = args
            return adapter_, opt_state_

        new_adapter, new_opt_state = jax.lax.cond(
            do_update,
            apply_update,
            skip_update,
            (grads, state.opt_state, state.adapter),
        )

        new_state = AdapterTrainState(
            backbone=state.backbone,
            adapter=new_adapter,
            opt_state=new_opt_state,
            step=state.step + jnp.int32(1),
            key=state.key,
        )
        return new_state, loss

    return step


def make_adapter_scan_step(
    train_step: Callable[
        [AdapterTrainState, Batch],
        tuple[AdapterTrainState, Float[Array, ""]],
    ],
) -> Callable[
    [AdapterTrainState, Batch], tuple[AdapterTrainState, Float[Array, "K"]]
]:
    """Wrap a single adapter train step in a K-step :func:`jax.lax.scan`.

    Input ``batches`` has a leading K axis on every Batch field.
    """

    @eqx.filter_jit(donate="all")
    def scan_step(
        state: AdapterTrainState, batches: Batch
    ) -> tuple[AdapterTrainState, Float[Array, "K"]]:
        def body(
            carry: AdapterTrainState, batch: Batch
        ) -> tuple[AdapterTrainState, Float[Array, ""]]:
            new_carry, loss = train_step(carry, batch)
            return new_carry, loss

        final_state, losses = jax.lax.scan(body, state, batches)
        return final_state, losses

    return scan_step


# ---------------------------------------------------------------------------
# Forward-only eval
# ---------------------------------------------------------------------------


def make_forward_eval(
    strategy: str,
) -> Callable[[PAWNModel, Any, Batch], Float[Array, "B T V"]]:
    """Build a JIT'd forward pass for eval — backbone + adapter → logits.

    Resolves ``dispatch_apply(strategy)`` *outside* the JIT closure so
    each (strategy, backbone-structure, adapter-structure) tuple
    compiles once, not per call. Earlier versions took ``strategy`` as
    an inner argument and called ``dispatch_apply(strategy)`` inside
    the traced body — that retraced per distinct ``strategy`` value
    and hid a runtime dispatch in the JIT signature.
    """
    apply_fn = dispatch_apply(strategy)

    @eqx.filter_jit
    def forward_eval(
        backbone: PAWNModel, adapter: Any, batch: Batch
    ) -> Float[Array, "B T V"]:
        effective = apply_fn(backbone, adapter)
        return effective(batch.tokens, batch.attn_mask)

    return forward_eval


def forward_eval(
    backbone: PAWNModel, adapter: Any, batch: Batch, strategy: str
) -> Float[Array, "B T V"]:
    """Back-compat shim: resolve the JIT'd forward eval per call.

    Prefer :func:`make_forward_eval` and reuse the returned closure —
    repeated invocations of this shim re-resolve ``dispatch_apply``
    every time, which is exactly the JIT-keying problem
    :func:`make_forward_eval` solves.
    """
    return make_forward_eval(strategy)(backbone, adapter, batch)


# ---------------------------------------------------------------------------
# RoSA three-phase orchestration (Nikdan et al. 2024, Algorithm 1)
# ---------------------------------------------------------------------------


def _mask_gen_loss(
    backbone: PAWNModel,
    adapter: RoSAAdapter,
    batch: Batch,
    compute_dtype: "jnp.dtype | None" = None,
) -> Float[Array, ""]:
    """Forward + cross-entropy with the RoSA composition active.

    The mask-gen phase wants gradients on the sparse ``delta_*`` arrays
    while every sparse mask is forced to all-True. The caller
    constructs that adapter before invoking this; here we just run the
    composed forward and return the loss so :func:`jax.grad` can
    differentiate it.
    """
    effective = rosa.apply_rosa(backbone, adapter)
    return cross_entropy_loss(
        effective, batch, compute_dtype=compute_dtype
    )


def generate_rosa_masks(
    backbone: PAWNModel,
    adapter: RoSAAdapter,
    batches: list[Batch],
    *,
    compute_dtype: "jnp.dtype | None" = None,
) -> SparseAdapter:
    """Run Algorithm 1: accumulate ``|grad|^grad_alpha`` over
    ``mask_samples`` batches; top-k per delta_* by ``density``.

    Returns a new :class:`pawn.adapters.sparse.SparseAdapter` with the
    fresh masks installed and deltas zeroed. The caller should slot
    this into the ``adapter.sparse`` field via :func:`eqx.tree_at`
    before entering Phase 3.
    """
    if not batches:
        raise ValueError(
            "generate_rosa_masks: at least one batch required for "
            "mask generation"
        )

    cfg = adapter.cfg
    # Force sparse_active=True with all-True masks so the grad path
    # captures the full weight gradient for each delta_*.
    full_masks = jax.tree_util.tree_map(
        lambda m: jnp.ones_like(m, dtype=jnp.bool_) if m is not None else None,
        adapter.sparse,
        is_leaf=lambda x: x is None,
    )
    primed_sparse = eqx.tree_at(
        lambda s: (
            s.mask_q, s.mask_k, s.mask_v, s.mask_o,
            s.mask_gate, s.mask_up, s.mask_down,
        ),
        adapter.sparse,
        (
            full_masks.mask_q, full_masks.mask_k,
            full_masks.mask_v, full_masks.mask_o,
            full_masks.mask_gate, full_masks.mask_up,
            full_masks.mask_down,
        ),
        is_leaf=lambda x: x is None,
    )
    # `sparse_active` is a static eqx field (not a PyTree leaf) so we
    # rebuild the wrapper rather than using `tree_at` on it.
    primed = RoSAAdapter(
        lora=adapter.lora,
        sparse=primed_sparse,
        bottleneck=adapter.bottleneck,
        cfg=adapter.cfg,
        lora_active=adapter.lora_active,
        sparse_active=True,
    )

    @eqx.filter_jit
    def grad_step(adapter_: RoSAAdapter, batch: Batch) -> Any:
        def loss_fn(a: RoSAAdapter) -> Float[Array, ""]:
            effective = rosa.apply_rosa(backbone, a)
            return cross_entropy_loss(
                effective, batch, compute_dtype=compute_dtype
            )
        _, grads = eqx.filter_value_and_grad(loss_fn)(adapter_)
        return grads

    alpha = cfg.grad_alpha
    # Lazily initialise the accumulators using the first batch's grad
    # shapes; this avoids needing a separate "shape probe" pass.
    accum: SparseAdapter | None = None
    for batch in batches:
        grads = grad_step(primed, batch)
        # Pull out the sparse-delta grads — these are what we accumulate.
        delta_grads = (
            grads.sparse.delta_q, grads.sparse.delta_k,
            grads.sparse.delta_v, grads.sparse.delta_o,
            grads.sparse.delta_gate, grads.sparse.delta_up,
            grads.sparse.delta_down,
        )
        accum_entries = (
            jax.tree_util.tree_map(
                lambda g: jnp.power(jnp.abs(g), alpha) if g is not None else None,
                d,
                is_leaf=lambda x: x is None,
            )
            for d in delta_grads
        )
        new_dq, new_dk, new_dv, new_do, new_dg, new_du, new_dd = list(accum_entries)
        if accum is None:
            accum = eqx.tree_at(
                lambda s: (
                    s.delta_q, s.delta_k, s.delta_v, s.delta_o,
                    s.delta_gate, s.delta_up, s.delta_down,
                ),
                adapter.sparse,
                (new_dq, new_dk, new_dv, new_do, new_dg, new_du, new_dd),
                is_leaf=lambda x: x is None,
            )
        else:
            def _add(a: jax.Array | None, b: jax.Array | None) -> jax.Array | None:
                if a is None or b is None:
                    return a if b is None else b
                return a + b
            new_deltas = jax.tree_util.tree_map(
                _add,
                (
                    accum.delta_q, accum.delta_k, accum.delta_v, accum.delta_o,
                    accum.delta_gate, accum.delta_up, accum.delta_down,
                ),
                (new_dq, new_dk, new_dv, new_do, new_dg, new_du, new_dd),
                is_leaf=lambda x: x is None,
            )
            accum = eqx.tree_at(
                lambda s: (
                    s.delta_q, s.delta_k, s.delta_v, s.delta_o,
                    s.delta_gate, s.delta_up, s.delta_down,
                ),
                accum,
                tuple(new_deltas),
                is_leaf=lambda x: x is None,
            )

    assert accum is not None  # protected by the empty-batches guard above

    # Top-k per-delta to construct the actual masks. Done per
    # delta_* rather than globally so each projection has the requested
    # density irrespective of relative grad magnitudes (matches v1
    # generate_gradient_masks). The top-k itself runs on CPU: the
    # JAX-on-ROCm `top_k` kernel returns degenerate output for
    # ``k > ~1000`` on arrays larger than ~100K, which would silently
    # produce an all-False mask. Mask generation is a one-shot
    # off-hot-path computation so the device-to-host cost is fine; we
    # use ``jax.device_put`` to land the score tensor on CPU and a
    # plain ``np.argpartition`` for the actual top-k selection.
    import numpy as _np
    cpu_dev = jax.devices("cpu")[0]

    def topk_mask(
        score: jax.Array | None, density: float
    ) -> jax.Array | None:
        if score is None:
            return None
        flat_host = _np.asarray(jax.device_put(score.reshape(-1), cpu_dev))
        k = max(1, int(density * flat_host.size))
        # `argpartition` is O(n) and only guarantees the partition
        # point; we don't need a sorted order, just the index set.
        top_idx = _np.argpartition(-flat_host, k - 1)[:k]
        flat_mask = _np.zeros(flat_host.shape, dtype=_np.bool_)
        flat_mask[top_idx] = True
        return jnp.asarray(flat_mask).reshape(score.shape)

    new_masks = tuple(
        topk_mask(s, cfg.density)
        for s in (
            accum.delta_q, accum.delta_k, accum.delta_v, accum.delta_o,
            accum.delta_gate, accum.delta_up, accum.delta_down,
        )
    )
    # Zero the deltas back out — Phase 3 starts from "identity"
    # sparse contribution and the mask alone determines which
    # positions can train.
    zeros = tuple(
        jnp.zeros_like(d) if d is not None else None
        for d in (
            adapter.sparse.delta_q, adapter.sparse.delta_k,
            adapter.sparse.delta_v, adapter.sparse.delta_o,
            adapter.sparse.delta_gate, adapter.sparse.delta_up,
            adapter.sparse.delta_down,
        )
    )
    return eqx.tree_at(
        lambda s: (
            s.delta_q, s.delta_k, s.delta_v, s.delta_o,
            s.delta_gate, s.delta_up, s.delta_down,
            s.mask_q, s.mask_k, s.mask_v, s.mask_o,
            s.mask_gate, s.mask_up, s.mask_down,
        ),
        adapter.sparse,
        zeros + new_masks,
        is_leaf=lambda x: x is None,
    )


def _reinit_lora(lora_adapter: LoRAAdapter, key: jax.Array) -> LoRAAdapter:
    """Re-draw LoRA A matrices Kaiming-uniform; zero B (v1 reinit_lora).

    The shape of each existing leaf is reused so the trainer doesn't
    need to thread the backbone through. Returns a fresh adapter with
    identity contribution at step 0 of Phase 3.
    """
    import math

    keys_iter = iter(jax.random.split(key, 16))

    def fresh_a(leaf: jax.Array | None) -> jax.Array | None:
        if leaf is None:
            return None
        fan_in = leaf.shape[-1]  # A has shape (..., fan_in, rank)
        bound = math.sqrt(1.0 / fan_in)
        k = next(keys_iter)
        return jax.random.uniform(k, leaf.shape, minval=-bound, maxval=bound)

    def fresh_b(leaf: jax.Array | None) -> jax.Array | None:
        if leaf is None:
            return None
        return jnp.zeros_like(leaf)

    a_leaves = (
        lora_adapter.A_q, lora_adapter.A_k, lora_adapter.A_v, lora_adapter.A_o,
        lora_adapter.A_gate, lora_adapter.A_up, lora_adapter.A_down,
    )
    b_leaves = (
        lora_adapter.B_q, lora_adapter.B_k, lora_adapter.B_v, lora_adapter.B_o,
        lora_adapter.B_gate, lora_adapter.B_up, lora_adapter.B_down,
    )
    new_a = tuple(fresh_a(leaf) for leaf in a_leaves)
    new_b = tuple(fresh_b(leaf) for leaf in b_leaves)
    return eqx.tree_at(
        lambda l: (
            l.A_q, l.A_k, l.A_v, l.A_o,
            l.A_gate, l.A_up, l.A_down,
            l.B_q, l.B_k, l.B_v, l.B_o,
            l.B_gate, l.B_up, l.B_down,
        ),
        lora_adapter,
        new_a + new_b,
        is_leaf=lambda x: x is None,
    )


def rosa_phase1_to_phase3(
    adapter: RoSAAdapter,
    new_sparse: SparseAdapter,
    key: jax.Array,
) -> RoSAAdapter:
    """Re-init the LoRA branch + install the new sparse adapter, then
    flip the active-branch toggles per ``cfg.mode``.

    - ``rosa`` (standard): LoRA on, sparse on, bottleneck off.
    - ``retro-sparse``: LoRA off, sparse on, bottleneck off.
    - ``retro-bottleneck``: LoRA off, sparse on, bottleneck on.

    The LoRA A matrices are re-drawn from Kaiming-uniform and B is
    zeroed (matches v1's ``reinit_lora``). Phase 3 starts from an
    identity contribution everywhere except the sparse mask carrier.
    """
    cfg = adapter.cfg
    fresh_lora = _reinit_lora(adapter.lora, key)
    new_lora_active = cfg.mode == "rosa"
    return RoSAAdapter(
        lora=fresh_lora,
        sparse=new_sparse,
        bottleneck=adapter.bottleneck,
        cfg=adapter.cfg,
        lora_active=new_lora_active,
        sparse_active=True,
    )
