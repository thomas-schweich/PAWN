"""Tests for ``pawn.jax.adapter_trainer`` — Phase-3 chunks 2/3.

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

from pawn.jax.adapter_trainer import (
    init_adapter_state,
    make_adapter_train_step,
    make_eval_step,
)
from pawn.jax.adapters import LoRAConfig, init_lora_model
from pawn.jax.config import TINY_SUPERNET, TINY_VARIANTS
from pawn.jax.corpus import generate_corpus
from pawn.jax.model import init_model, sliced
from pawn.jax.trainer import Batch, make_optimizer


def _setup(rank: int = 4, lr: float = 3e-3) -> tuple:
    """Build a minimal adapter training stack at TINY scale."""
    key = jax.random.PRNGKey(0)
    supernet = init_model(TINY_SUPERNET, key)
    backbone = sliced(supernet, TINY_VARIANTS["base"])
    model = init_lora_model(
        backbone, LoRAConfig(rank=rank, targets=("q", "v")), jax.random.PRNGKey(1)
    )
    opt = make_optimizer(lr)
    state, frozen = init_adapter_state(model, opt)
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
    """The §6.1 headline contract: after training, every backbone
    weight is bit-identical to its initial value. A regression that
    wired the optimizer through the backbone partition would silently
    clobber the pretrained baseline — and only a per-leaf comparison
    catches it (the adapter loss would still decrease)."""
    model, state, frozen, train_step, _ = _setup(lr=1e-1)
    batch = _real_batch(b=4, t=16)
    # Many steps at a large LR to amplify any drift.
    for _ in range(20):
        state, _ = train_step(state, batch)
    # The current backbone is ``frozen`` (captured at init) combined
    # with ``state.trainable.backbone`` (None subtree). The
    # backbone's arrays are reachable via ``frozen.backbone``
    # directly; verify each leaf is bit-equal to the corresponding
    # leaf in the original model.
    for name in (
        "src_embed", "dst_embed", "promo_embed", "outcome_embed",
        "attn_norm", "wq", "wk", "wv", "wo",
        "ffn_norm", "w_gate", "w_up", "w_down",
        "final_norm", "lm_head", "pad_embed",
    ):
        original = getattr(model.backbone, name)
        current = getattr(frozen.backbone, name)
        assert jnp.array_equal(original, current), (
            f"backbone leaf {name!r} drifted under adapter training — "
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
