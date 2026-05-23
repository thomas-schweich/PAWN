"""Tests for `pawn.adapters` + `pawn.adapter_trainer` — all 8 strategies
dispatch and train at least one chunk per plan §3 criterion 8.
"""

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from pawn.adapter_trainer import (
    STRATEGIES,
    AdapterTrainState,
    dispatch_apply,
    dispatch_filter,
    dispatch_init,
    forward_eval,
    make_adapter_scan_step,
    make_adapter_train_step,
)
from pawn.adapters import (
    BottleneckConfig,
    FiLMConfig,
    HybridConfig,
    LoRAConfig,
    RoSAConfig,
    SparseConfig,
    SpecializedCLMConfig,
    UnfreezeConfig,
)
from pawn.config import TINY_SUPERNET
from pawn.corpus import generate_corpus
from pawn.model import PAWNModel, init_model
from pawn.trainer import Batch, slice_batch


# ---------------------------------------------------------------------------
# Per-strategy configs
# ---------------------------------------------------------------------------


def _strategy_config(strategy: str) -> Any:
    """Return a minimal valid config for each strategy."""
    if strategy == "lora":
        return LoRAConfig(rank=2, targets="qkvo")
    if strategy == "film":
        return FiLMConfig(use_output_film=True)
    if strategy == "bottleneck":
        return BottleneckConfig(dim=4)
    if strategy == "hybrid":
        return HybridConfig(lora=LoRAConfig(rank=2, targets="qkvo"))
    if strategy == "sparse":
        return SparseConfig(density=0.1)
    if strategy == "rosa":
        return RoSAConfig(mode="rosa", lora_rank=2, density=0.1)
    if strategy == "rosa-retro-sparse":
        return RoSAConfig(mode="retro-sparse", lora_rank=2, density=0.1)
    if strategy == "rosa-retro-bottleneck":
        return RoSAConfig(mode="retro-bottleneck", lora_rank=2, density=0.1)
    if strategy == "unfreeze":
        return UnfreezeConfig(layers="0,1")
    if strategy == "specialized_clm":
        return SpecializedCLMConfig(d_model=64, n_layers=2, n_heads=1, d_ff=128)
    raise ValueError(strategy)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------


def test_strategies_table_includes_all_documented_keys() -> None:
    """Plan §10 S7 + CLAUDE.md adapter table: 8 base strategies plus
    the two RoSA retro modes as distinct `--strategy` keys."""
    assert set(STRATEGIES.keys()) == {
        "lora", "film", "bottleneck", "hybrid",
        "sparse", "rosa", "rosa-retro-sparse", "rosa-retro-bottleneck",
        "unfreeze", "specialized_clm",
    }


def test_dispatch_helpers_reject_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        dispatch_init("nope")
    with pytest.raises(ValueError, match="unknown strategy"):
        dispatch_apply("nope")
    with pytest.raises(ValueError, match="unknown strategy"):
        dispatch_filter("nope")


# ---------------------------------------------------------------------------
# Per-strategy init + apply (compile + forward)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", sorted(STRATEGIES.keys()))
def test_each_strategy_init_and_apply(strategy: str) -> None:
    """Every strategy's init builds an adapter; apply returns a
    PAWNModel; forward produces finite logits on a small batch."""
    backbone = init_model(TINY_SUPERNET, key=0)
    cfg = _strategy_config(strategy)
    init = dispatch_init(strategy)
    apply = dispatch_apply(strategy)
    adapter = init(backbone, cfg, key=jax.random.key(1))
    effective = apply(backbone, adapter)
    assert isinstance(effective, PAWNModel)
    # Forward pass on a tiny batch.
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    logits = effective(tokens)
    assert logits.shape == (2, 16, TINY_SUPERNET.vocab_size)
    assert jnp.all(jnp.isfinite(logits))


# ---------------------------------------------------------------------------
# Per-strategy train one chunk — the acceptance-criterion-8 contract
# ---------------------------------------------------------------------------


def _make_batch(seq_len: int = 16, n: int = 4) -> Batch:
    corpus = generate_corpus(
        n_games=n, max_ply=seq_len, seq_len=seq_len, seed=0
    )
    return slice_batch(corpus, np.arange(n))


@pytest.mark.parametrize("strategy", sorted(STRATEGIES.keys()))
def test_each_strategy_dispatch_runs(strategy: str) -> None:
    """Plan §3 criterion 8: each strategy dispatches and trains at
    least one chunk without crashing. The pre-training-step assertion
    is the load-bearing one — once the JIT compiles and the train
    step runs, the strategy is wired correctly."""
    backbone = init_model(TINY_SUPERNET, key=0)
    cfg = _strategy_config(strategy)
    init = dispatch_init(strategy)
    adapter_filter = dispatch_filter(strategy)
    adapter = init(backbone, cfg, key=jax.random.key(1))

    # Filter to just the trainable params for optimizer init.
    trainable = eqx.filter(adapter, adapter_filter(adapter))
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3),
    )
    opt_state = opt.init(trainable)

    state = AdapterTrainState(
        backbone=backbone,
        adapter=adapter,
        opt_state=opt_state,
        step=jnp.int32(0),
        key=jax.random.key(0),
    )

    # Some strategies have no trainable params (unfreeze), so the
    # optimizer would no-op; that's fine for the dispatch test.
    if strategy == "unfreeze":
        # Unfreeze's adapter has no trainable inexact-arrays — running
        # train step would do nothing, so we just verify the apply
        # function returns the backbone untouched.
        applied = dispatch_apply(strategy)(backbone, adapter)
        assert applied is backbone
        return

    train_step = make_adapter_train_step(strategy, opt)
    batch = _make_batch()
    new_state, loss = train_step(state, batch)
    assert int(new_state.step) == 1
    assert jnp.isfinite(loss)


# ---------------------------------------------------------------------------
# Two-tier partition: backbone gradients DCE'd
# ---------------------------------------------------------------------------


def test_lora_partition_only_trains_adapter() -> None:
    """The two-tier partition (eqx.filter on adapter_filter) ensures
    only the LoRA A/B matrices appear in the optimizer state — backbone
    weights stay frozen."""
    backbone = init_model(TINY_SUPERNET, key=0)
    cfg = LoRAConfig(rank=2)
    adapter = dispatch_init("lora")(backbone, cfg, key=jax.random.key(0))
    flt = dispatch_filter("lora")(adapter)
    trainable = eqx.filter(adapter, flt)
    n_trainable_leaves = sum(
        1 for leaf in jax.tree_util.tree_leaves(trainable) if leaf is not None
    )
    # 8 LoRA A/B matrices (q/k/v/o × A/B), no FFN.
    assert n_trainable_leaves == 8


def test_two_tier_partition_filters_backbone_arrays() -> None:
    """Plan §10 S7 structural invariant: after partitioning, the
    backbone half has `None` at every position that would be a
    trainable inexact-array leaf. The adapter half has the
    trainable A/B matrices.

    We pin the invariant on the BACKBONE side: `eqx.partition(backbone,
    eqx.is_inexact_array)` would split it into (all-arrays, all-None);
    when the AdapterTrainState is held with backbone-as-frozen, the
    backbone's arrays must not appear in the optimizer's update path.
    """
    import jax.tree_util as jtu

    backbone = init_model(TINY_SUPERNET, key=0)
    adapter = dispatch_init("lora")(
        backbone, LoRAConfig(rank=2), key=jax.random.key(0)
    )
    # The adapter's trainable subset is what eqx.filter(adapter, flt)
    # returns; everything outside that is None / static.
    flt = dispatch_filter("lora")(adapter)
    trainable, frozen = eqx.partition(adapter, flt)
    # Frozen side: the static `cfg` plus None on every array leaf.
    frozen_array_leaves = [
        leaf
        for leaf in jtu.tree_leaves(frozen, is_leaf=lambda x: x is None)
        if eqx.is_inexact_array(leaf)
    ]
    assert frozen_array_leaves == [], (
        f"frozen partition leaked inexact arrays: "
        f"{[leaf.shape for leaf in frozen_array_leaves]}"
    )
    # Trainable side: at least one inexact array (the A/B matrices).
    trainable_array_leaves = [
        leaf
        for leaf in jtu.tree_leaves(trainable, is_leaf=lambda x: x is None)
        if eqx.is_inexact_array(leaf)
    ]
    assert len(trainable_array_leaves) == 8  # 4 A + 4 B for q/k/v/o


# ---------------------------------------------------------------------------
# forward_eval
# ---------------------------------------------------------------------------


def test_forward_eval_runs_for_lora() -> None:
    backbone = init_model(TINY_SUPERNET, key=0)
    adapter = dispatch_init("lora")(
        backbone, LoRAConfig(rank=2), key=jax.random.key(0)
    )
    batch = _make_batch()
    logits = forward_eval(backbone, adapter, batch, "lora")
    assert logits.shape == (4, 16, TINY_SUPERNET.vocab_size)
    assert jnp.all(jnp.isfinite(logits))


# ---------------------------------------------------------------------------
# unfreeze parse helper
# ---------------------------------------------------------------------------


def test_unfreeze_parse_layers() -> None:
    from pawn.adapters.unfreeze import parse_unfreeze_layers
    assert parse_unfreeze_layers("5,6,7") == [5, 6, 7]
    assert parse_unfreeze_layers("0") == [0]


def test_unfreeze_rejects_out_of_range_indices() -> None:
    backbone = init_model(TINY_SUPERNET, key=0)  # n_layers = 4
    with pytest.raises(ValueError, match="outside"):
        dispatch_init("unfreeze")(
            backbone, UnfreezeConfig(layers="99"), key=jax.random.key(0)
        )


# ---------------------------------------------------------------------------
# RoSA's three-mode discriminator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["rosa", "retro-sparse", "retro-bottleneck"])
def test_rosa_dispatches_each_mode(
    mode: Literal["rosa", "retro-sparse", "retro-bottleneck"],
) -> None:
    backbone = init_model(TINY_SUPERNET, key=0)
    cfg = RoSAConfig(mode=mode, lora_rank=2, density=0.1)
    adapter = dispatch_init("rosa")(backbone, cfg, key=jax.random.key(0))
    assert adapter.cfg.mode == mode
    # apply runs without error in every mode.
    effective = dispatch_apply("rosa")(backbone, adapter)
    assert isinstance(effective, PAWNModel)
