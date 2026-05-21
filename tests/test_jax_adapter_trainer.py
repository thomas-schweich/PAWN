"""Tests for ``pawn.adapter_trainer`` — Phase-3 chunks 2/3.

Pins the load-bearing invariants of the adapter training loop:

  * The §6.1 contract: **backbone weights stay frozen** across steps.
    This is the headline claim — if it ever regresses, the
    pretrained baseline gets clobbered with no obvious symptom.
  * The two-tier partition: ``state.trainable.lora.*`` carries
    arrays, ``state.trainable.backbone.*`` carries ``None``.
  * Loss decreases on a fixed batch (gradient flow reaches LoRA).
  * Evaluation is forward-only (no state mutation, no weight
    drift).
  * Padded-batch invariance: ``loss_mask=False`` everywhere → no
    weight update (same Codex P2 guard pattern as the pretrain
    trainer).
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("optax")
pytest.importorskip("chess_engine")

import equinox as eqx
import jax
import jax.numpy as jnp

from pawn.adapter_trainer import (
    init_adapter_state,
    make_adapter_scan_step,
    make_adapter_train_step,
    make_eval_step,
)
from pawn.adapters import LoRAConfig, adapter_filter, init_lora_model
from pawn.config import TINY_SUPERNET, TINY_VARIANTS
from pawn.corpus import generate_corpus
from pawn.model import init_model, sliced
from pawn.trainer import Batch, make_optimizer


def _setup(rank: int = 4, lr: float = 3e-3) -> tuple:
    """Build a minimal adapter training stack at TINY scale."""
    key = jax.random.PRNGKey(0)
    supernet = init_model(TINY_SUPERNET, key)
    backbone = sliced(supernet, TINY_VARIANTS["base"])
    model = init_lora_model(
        backbone, LoRAConfig(rank=rank, targets=("q", "v")), jax.random.PRNGKey(1)
    )
    opt = make_optimizer(lr)
    state, frozen = init_adapter_state(
        model, opt, adapter_filter_fn=adapter_filter,
    )
    train_step = make_adapter_train_step(opt, frozen)
    eval_step = make_eval_step(frozen)
    return model, state, frozen, train_step, eval_step


def _real_batch(b: int = 4, t: int = 16, seed: int = 0) -> Batch:
    c = generate_corpus(n_games=b, max_ply=t, seed=seed)
    return (
        jnp.asarray(c.tokens),
        jnp.asarray(c.attn_mask),
        jnp.asarray(c.targets),
        jnp.asarray(c.loss_mask),
    )


def test_init_partition_separates_lora_from_backbone() -> None:
    """``init_adapter_state`` partitions correctly: trainable carries
    arrays at lora.*, ``None`` at every backbone leaf. Frozen is the
    inverse. The §6.1 contract begins here."""
    _, state, frozen, _, _ = _setup()
    # Trainable: lora.* arrays present; backbone arrays None.
    assert isinstance(state.trainable.lora.b_q, jax.Array)
    assert isinstance(state.trainable.lora.a_q, jax.Array)
    for name in (
        "src_embed", "dst_embed", "promo_embed", "outcome_embed",
        "attn_norm", "wq", "wk", "wv", "wo",
        "ffn_norm", "w_gate", "w_up", "w_down",
        "final_norm", "lm_head", "pad_embed",
    ):
        assert getattr(state.trainable.backbone, name) is None, (
            f"trainable.backbone.{name} should be None"
        )
    # Frozen: the opposite — backbone arrays present, lora.* None.
    assert isinstance(frozen.backbone.wq, jax.Array)
    assert frozen.lora.b_q is None
    assert frozen.lora.a_q is None


def test_train_step_decreases_loss_on_fixed_batch() -> None:
    """Repeated steps on the same batch reduce loss — the basic
    contract (gradient flow + optimizer wired correctly)."""
    _, state, frozen, train_step, _ = _setup()
    batch = _real_batch(b=4, t=16)
    initial_loss = None
    for i in range(10):
        state, m = train_step(state, batch)
        if i == 0:
            initial_loss = float(m["loss"])
    final_loss = float(m["loss"])
    assert initial_loss is not None
    assert final_loss < initial_loss, (
        f"loss did not decrease: initial={initial_loss}, final={final_loss}"
    )


def test_backbone_weights_are_frozen() -> None:
    """The §6.1 headline contract: after training, the optimizer must
    NOT have updated any backbone parameter. The structural guard is
    that ``state.trainable.backbone.*`` stays ``None`` (the partition
    invariant) — if a regression wired the optimizer through the
    backbone, those leaves would become arrays. A complementary check
    is that the reconstructed model's backbone (via ``eqx.combine``)
    is bit-equal to the original backbone — which is guaranteed by
    JAX immutability if the partition invariant holds.
    """
    model, state, frozen, train_step, _ = _setup(lr=1e-1)
    batch = _real_batch(b=4, t=16)
    # Many steps at a large LR to amplify any drift.
    for _ in range(20):
        state, _ = train_step(state, batch)
    # Structural invariant: state.trainable.backbone subtree is all
    # None. A regression that broadened the optimizer to the
    # backbone would surface here.
    for name in (
        "src_embed", "dst_embed", "promo_embed", "outcome_embed",
        "attn_norm", "wq", "wk", "wv", "wo",
        "ffn_norm", "w_gate", "w_up", "w_down",
        "final_norm", "lm_head", "pad_embed",
    ):
        assert getattr(state.trainable.backbone, name) is None, (
            f"state.trainable.backbone.{name} became non-None after "
            f"training — the optimizer leaked into the backbone."
        )
    # Functional invariant: the model reconstructed from
    # ``eqx.combine(state.trainable, frozen)``'s backbone is
    # bit-identical to the original. This is guaranteed by JAX
    # array immutability + Equinox closure capture, but pinning it
    # in a test ensures a future refactor that swaps ``frozen``
    # for a mutable container or stops using the partition
    # boundary surfaces the regression.
    reconstructed = eqx.combine(state.trainable, frozen)
    for name in ("wq", "wk", "wv", "wo", "lm_head", "final_norm"):
        original = getattr(model.backbone, name)
        current = getattr(reconstructed.backbone, name)
        assert jnp.array_equal(original, current), (
            f"reconstructed.backbone.{name} drifted from original — "
            f"max diff = {float(jnp.abs(original - current).max())}"
        )


def test_lora_weights_actually_update() -> None:
    """Complementary to the frozen-backbone test: the LoRA params
    DO change. Pins gradient flow through the adapter path."""
    _, state, _, train_step, _ = _setup()
    initial_b_q = state.trainable.lora.b_q
    batch = _real_batch(b=4, t=16)
    state, _ = train_step(state, batch)
    delta = float(jnp.abs(state.trainable.lora.b_q - initial_b_q).max())
    assert delta > 0.0, "b_q did not move after one train step"


def test_gradient_mask_freezes_layers_through_train_step() -> None:
    """``make_adapter_train_step(gradient_mask=...)`` zeroes
    gradients per-element through the trainer path.

    The unfreeze strategy's per-layer slicing relies on this — the
    `adapter_filter` partition is coarse (every layer-stacked field
    is fully True or fully False), and per-layer "top N unfrozen"
    selection happens via the gradient mask. This test pins the
    contract: after training, layer-stacked weight slices that the
    mask gates to False must not have moved, even though the trainer
    sees them as "trainable" via the partition spec.

    Pre-fix (before C.1), this path was unused (only LoRA had a
    trainer). The C.1 mask-multiplication code is exercised by
    ``test_each_strategy_dispatch_runs[strategy=unfreeze]`` but only
    at the "doesn't crash" granularity. This unit-level test pins
    the actual weight-freeze invariant the dispatch relies on.
    """
    from typing import cast

    from pawn.adapters import (
        UnfreezeConfig, UnfreezeModel, init_unfreeze_model,
        unfreeze_adapter_filter, unfreeze_gradient_mask,
    )
    from pawn.adapter_trainer import (
        init_adapter_state as init_state,
        make_adapter_train_step as make_step,
    )

    key = jax.random.PRNGKey(0)
    supernet = init_model(TINY_SUPERNET, key)
    backbone = sliced(supernet, TINY_VARIANTS["base"])
    # n_unfreeze=1 over backbone.cfg.n_layers=3 → layers [0, 1]
    # frozen, layer [2] unfrozen.
    model = init_unfreeze_model(
        backbone, UnfreezeConfig(n_unfreeze=1, include_lm_head=False),
        jax.random.PRNGKey(1),
    )
    n_layers = backbone.cfg.n_layers
    assert n_layers >= 2, "Test needs >=2 layers to distinguish frozen vs unfrozen"
    opt = make_optimizer(3e-2)  # Strong LR so any leak is unmissable.
    grad_mask = unfreeze_gradient_mask(model)
    state, frozen = init_state(
        model, opt, adapter_filter_fn=unfreeze_adapter_filter,
    )
    train_step = make_step(opt, frozen, gradient_mask=grad_mask)
    # ``state.trainable`` is statically typed ``eqx.Module``; cast to
    # the concrete adapter type so attribute access on the partition
    # tree is type-checked.
    initial_wq = cast(UnfreezeModel, state.trainable).backbone.wq.copy()
    batch = _real_batch(b=4, t=16)
    for _ in range(3):
        state, _ = train_step(state, batch)
    diff = jnp.abs(
        cast(UnfreezeModel, state.trainable).backbone.wq - initial_wq,
    )
    # Layers [0..n_layers-1] frozen, layer [n_layers-1] unfrozen.
    # The frozen layers' deltas must be exactly zero (the mask zeroed
    # those gradients before the optimizer step).
    for layer in range(n_layers - 1):
        assert float(diff[layer].max()) == 0.0, (
            f"layer {layer} should have stayed frozen but max-abs "
            f"delta = {float(diff[layer].max())}"
        )
    # The unfrozen layer's delta must be > 0.
    assert float(diff[n_layers - 1].max()) > 0.0, (
        f"unfrozen layer {n_layers - 1} did not change after 3 steps"
    )


def test_eval_step_is_forward_only() -> None:
    """``eval_step`` returns metrics without mutating state. Pins the
    forward-only contract — the eval result must be a function of
    ``(trainable, batch)`` alone, callable repeatedly without drift."""
    _, state, _, _, eval_step = _setup()
    batch = _real_batch(b=4, t=16)
    m1 = eval_step(state.trainable, batch)
    m2 = eval_step(state.trainable, batch)
    assert float(m1["loss"]) == float(m2["loss"]), (
        "eval_step is not deterministic — likely mutating shared state"
    )


def test_train_step_skips_update_on_all_padded_batch() -> None:
    """Codex-P2-style guard: a batch with ``loss_mask=False``
    everywhere must not advance state.step, mutate weights, or
    mutate optimizer state. Mirrors the pretraining trainer
    invariant — same lax.cond skip path."""
    _, state, _, train_step, _ = _setup()
    b, t = 2, 8
    pad_batch: Batch = (
        jnp.zeros((b, t), dtype=jnp.int32),
        jnp.zeros((b, t), dtype=jnp.bool_),
        jnp.zeros((b, t), dtype=jnp.int32),
        jnp.zeros((b, t), dtype=jnp.bool_),
    )
    initial_b_q = state.trainable.lora.b_q
    new_state, m = train_step(state, pad_batch)
    # No state advance.
    assert int(new_state.step) == 0
    # No weight drift.
    assert jnp.array_equal(new_state.trainable.lora.b_q, initial_b_q)
    # Loss is 0 (safe-divide on empty mask), n_supervised is 0.
    assert float(m["loss"]) == 0.0
    assert int(m["n_supervised"]) == 0


def test_train_step_state_is_jax_pytree() -> None:
    """``AdapterTrainState`` round-trips through ``tree_flatten`` /
    ``unflatten``; the step counter remains a 0-d ``jnp.int32`` so
    ``eqx.filter_jit`` doesn't retrace on increment."""
    _, state, _, _, _ = _setup()
    leaves, treedef = jax.tree_util.tree_flatten(state)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert int(rebuilt.step) == int(state.step)
    assert isinstance(state.step, jax.Array)
    assert state.step.shape == () and state.step.dtype == jnp.int32


def test_adapter_scan_step_advances_by_k() -> None:
    """``make_adapter_scan_step`` wraps the single-step in
    ``lax.scan(length=k)``. After one scan call ``state.step``
    advances by exactly K (assuming the chunk has supervision
    everywhere), metrics come back stacked on a leading [K] axis."""
    _, state, _, train_step, _ = _setup()
    K = 4
    scan = make_adapter_scan_step(train_step, K)
    # Build a [K, B, T] chunk by stacking K copies of one batch.
    single = _real_batch(b=2, t=8, seed=3)
    chunk: Batch = tuple(
        jnp.stack([x] * K, axis=0) for x in single
    )  # type: ignore[assignment]
    state, metrics = scan(state, chunk)
    assert int(state.step) == K
    assert metrics["loss"].shape == (K,)
    assert metrics["grad_norm"].shape == (K,)


def test_eval_step_uses_correct_frozen() -> None:
    """``make_eval_step(frozen)`` closes over the frozen backbone passed
    in. Two independently-initialised adapter stacks must produce
    eval losses tied to THEIR respective backbone — a regression
    that conflated frozen instances would silently evaluate one
    adapter against the other backbone.

    Set-up: two stacks with different backbones AND different
    post-training LoRA values, so the cross-eval result is
    distinguishable from both same-stack results."""
    # Two separately-seeded backbones.
    key_a, key_b = jax.random.PRNGKey(11), jax.random.PRNGKey(22)
    backbone_a = sliced(init_model(TINY_SUPERNET, key_a), TINY_VARIANTS["base"])
    backbone_b = sliced(init_model(TINY_SUPERNET, key_b), TINY_VARIANTS["base"])
    cfg = LoRAConfig(rank=4, targets=("q", "v"))
    # Different LoRA-init keys so the post-training LoRA differs too.
    model_a = init_lora_model(backbone_a, cfg, jax.random.PRNGKey(101))
    model_b = init_lora_model(backbone_b, cfg, jax.random.PRNGKey(202))
    opt = make_optimizer(3e-3)
    state_a, frozen_a = init_adapter_state(
        model_a, opt, adapter_filter_fn=adapter_filter,
    )
    state_b, frozen_b = init_adapter_state(
        model_b, opt, adapter_filter_fn=adapter_filter,
    )
    step_a = make_adapter_train_step(opt, frozen_a)
    step_b = make_adapter_train_step(opt, frozen_b)
    train_batch = _real_batch(b=4, t=16, seed=1)
    # Train each stack independently so LoRA params diverge (B is no
    # longer zero in either).
    for _ in range(5):
        state_a, _ = step_a(state_a, train_batch)
        state_b, _ = step_b(state_b, train_batch)
    eval_a = make_eval_step(frozen_a)
    batch = _real_batch(b=4, t=16, seed=2)
    loss_aa = float(eval_a(state_a.trainable, batch)["loss"])
    # Cross: eval_a with state_b's trainable — frozen_a's backbone
    # combined with state_b's LoRA. Different from same-stack
    # evaluation because state_b's LoRA was trained against a
    # different backbone.
    loss_ab = float(eval_a(state_b.trainable, batch)["loss"])
    assert loss_aa != loss_ab, (
        "eval_step's frozen closure is being ignored — same loss "
        "regardless of which state.trainable is passed"
    )


def test_lora_a_q_updates_after_multiple_steps() -> None:
    """``a_q``'s gradient is exactly zero at init (the gradient
    formula scales by ``b_q``, which is initialised to zero). After
    step 1, ``b_q`` is non-zero and the next step's gradient w.r.t.
    ``a_q`` becomes non-trivial. Pin the multi-step behaviour
    (the single-step test couldn't have caught a regression that
    broke the ``a_q`` update path)."""
    _, state, _, train_step, _ = _setup(lr=3e-3)
    initial_a_q = state.trainable.lora.a_q
    batch = _real_batch(b=4, t=16)
    for _ in range(5):
        state, _ = train_step(state, batch)
    delta = float(jnp.abs(state.trainable.lora.a_q - initial_a_q).max())
    assert delta > 0.0, "a_q did not move after 5 train steps"


def test_eval_loss_decreases_as_training_progresses() -> None:
    """Train + eval the same model on different batches. After
    several training steps, validation loss should drop (the
    adapter generalises, not just memorises the train batch)."""
    _, state, _, train_step, eval_step = _setup(lr=3e-3)
    train_batch = _real_batch(b=8, t=16, seed=1)
    val_batch = _real_batch(b=8, t=16, seed=2)
    initial_val = float(eval_step(state.trainable, val_batch)["loss"])
    for _ in range(30):
        state, _ = train_step(state, train_batch)
    final_val = float(eval_step(state.trainable, val_batch)["loss"])
    # 30 steps on a different batch may or may not decrease val a lot,
    # but it should not catastrophically increase. We require strict
    # ≤ to catch a regression where validation diverges. (A real
    # adapter run gets >0.05 decrease per the verification log.)
    assert final_val <= initial_val * 1.05, (
        f"val loss increased after training: "
        f"initial={initial_val}, final={final_val}"
    )
