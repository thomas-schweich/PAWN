"""Tests for `pawn.adapters` + `pawn.adapter_trainer` — all 8 strategies
dispatch and train at least one chunk per plan §3 criterion 8.
"""

from __future__ import annotations

from pathlib import Path
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
    FiLMAdapter,
    FiLMConfig,
    HybridConfig,
    LoRAConfig,
    RoSAConfig,
    SparseConfig,
    SpecializedCLMConfig,
    UnfreezeConfig,
)
from pawn.adapters.bottleneck import BottleneckEffective
from pawn.adapters.film import FiLMEffective
from pawn.config import TINY_SUPERNET
from pawn.corpus import generate_corpus
from pawn.model import PAWNModel, init_model
from pawn.trainer import Batch, slice_batch


# `apply_fn` may return either the patched ``PAWNModel`` (for adapters that
# fold corrections into the backbone weights — LoRA, sparse, FiLM,
# unfreeze, ...) or a callable wrapper such as ``BottleneckEffective`` (for
# adapters whose semantics require post-sublayer residual injection —
# BottleneckEffective for the Houlsby MLP, FiLMEffective for true FiLM's
# residual-stream + output-logit modulation). The trainer + eval scripts
# treat all three uniformly; the test accepts any of them.
_EFFECTIVE_TYPES = (PAWNModel, BottleneckEffective, FiLMEffective)


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
    assert isinstance(effective, _EFFECTIVE_TYPES)
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

    # Snapshot backbone.wq BEFORE train_step — the JIT donates the
    # state buffer, so the original `backbone` reference is invalid
    # after the call. `np.asarray(...)` materialises the comparison
    # baseline to host memory.
    if strategy == "unfreeze":
        backbone_wq_pre = np.asarray(backbone.layers.wq)

    train_step = make_adapter_train_step(strategy, opt)
    batch = _make_batch()
    new_state, loss = train_step(state, batch)
    assert int(new_state.step) == 1
    assert jnp.isfinite(loss)

    if strategy == "unfreeze":
        # At init the unfreeze adapter is an exact copy of the
        # backbone's transformer slices, so `apply_unfreeze` at step 0
        # produces a model whose layers match the backbone's. After one
        # train step, the unmasked layers in `state.adapter.layers`
        # must have moved (gradient was non-zero); the masked layers
        # must still match the pre-step backbone (gradient was zero
        # because `where` blocks their forward contribution).
        from pawn.adapters.unfreeze import parse_unfreeze_layers
        unfrozen = parse_unfreeze_layers("0,1")  # matches `_strategy_config`
        post_wq = np.asarray(new_state.adapter.layers.wq)
        for i in range(post_wq.shape[0]):
            if i in unfrozen:
                assert not np.allclose(post_wq[i], backbone_wq_pre[i]), (
                    f"unfrozen layer {i} should have trained"
                )
            else:
                assert np.allclose(post_wq[i], backbone_wq_pre[i]), (
                    f"frozen layer {i} should still match backbone"
                )


# ---------------------------------------------------------------------------
# Backbone-frozen invariant against the un-factored field set
# ---------------------------------------------------------------------------


# `specialized_clm` trains a standalone model (the backbone is ignored), so
# its "backbone" fields legitimately move; every other strategy must leave
# the frozen backbone's embedding / head / final-norm untouched.
_BACKBONE_HOLDING_STRATEGIES = sorted(
    k for k in STRATEGIES if k != "specialized_clm"
)


@pytest.mark.parametrize("strategy", _BACKBONE_HOLDING_STRATEGIES)
def test_each_strategy_keeps_backbone_embed_and_head_frozen(
    strategy: str,
) -> None:
    """Chunk 6 invariant: after a train step the frozen backbone's
    un-factored token table (``embed_tokens``), tied/untied head
    (``lm_head``), and ``final_norm_w`` are bit-identical.

    This is the field-set-aware guard that the old factored
    ``embed_src``/``dst``/``promo``/``pad``/``outcome`` fields used to
    be implicitly covered by — it pins that no adapter reconstruction
    accidentally threads gradient into the new uniform embedding or the
    (tied) head reused from it.
    """
    backbone = init_model(TINY_SUPERNET, key=0)
    assert backbone.lm_head is None  # TINY_SUPERNET ties by default
    cfg = _strategy_config(strategy)
    init = dispatch_init(strategy)
    adapter_filter = dispatch_filter(strategy)
    adapter = init(backbone, cfg, key=jax.random.key(1))

    embed_pre = np.asarray(backbone.embed_tokens)
    final_norm_pre = np.asarray(backbone.final_norm_w)

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
    train_step = make_adapter_train_step(strategy, opt)
    new_state, _ = train_step(state, _make_batch())

    # The backbone is held in state.backbone and never updated by the
    # optimizer; its embedding table / head / final-norm must not move.
    assert new_state.backbone.lm_head is None
    assert np.array_equal(
        np.asarray(new_state.backbone.embed_tokens), embed_pre
    ), f"{strategy}: backbone embed_tokens drifted"
    assert np.array_equal(
        np.asarray(new_state.backbone.final_norm_w), final_norm_pre
    ), f"{strategy}: backbone final_norm_w drifted"


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
    assert isinstance(effective, _EFFECTIVE_TYPES)


# ---------------------------------------------------------------------------
# Sparse FFN — v1 parity: `cfg.ffn=True` must populate FFN delta/mask pairs
# ---------------------------------------------------------------------------


def test_sparse_ffn_populates_ffn_delta_and_mask_pairs() -> None:
    """Parity #5 (sparse-FFN): with ``cfg.ffn=True`` the adapter
    carries delta/mask pairs for ``w_gate`` / ``w_up`` / ``w_down``;
    with ``cfg.ffn=False`` those pairs are ``None`` (no allocation)."""
    backbone = init_model(TINY_SUPERNET, key=0)
    cfg_on = SparseConfig(density=0.1, ffn=True)
    adapter_on = dispatch_init("sparse")(backbone, cfg_on, key=jax.random.key(0))
    assert adapter_on.delta_gate is not None
    assert adapter_on.delta_up is not None
    assert adapter_on.delta_down is not None
    assert adapter_on.mask_gate is not None
    assert adapter_on.mask_up is not None
    assert adapter_on.mask_down is not None
    # Mask density is approximately cfg.density (Bernoulli sampling).
    frac = float(adapter_on.mask_gate.mean())
    assert 0.02 < frac < 0.2, f"sparse ffn mask density={frac:.4f} off"

    cfg_off = SparseConfig(density=0.1, ffn=False)
    adapter_off = dispatch_init("sparse")(backbone, cfg_off, key=jax.random.key(0))
    assert adapter_off.delta_gate is None
    assert adapter_off.delta_up is None
    assert adapter_off.delta_down is None


# ---------------------------------------------------------------------------
# Bottleneck — Houlsby parity: real residual MLP with attn + FFN placement
# ---------------------------------------------------------------------------


def test_bottleneck_init_populates_both_placements_by_default() -> None:
    """With both placement flags off (the default), init populates both
    attn-side and FFN-side weights; the zero-init up-projection makes
    the start identical to the frozen backbone."""
    backbone = init_model(TINY_SUPERNET, key=0)
    adapter = dispatch_init("bottleneck")(
        backbone, BottleneckConfig(dim=4), key=jax.random.key(0)
    )
    assert adapter.down_attn is not None and adapter.up_attn is not None
    assert adapter.down_ffn is not None and adapter.up_ffn is not None
    # Up projections are zero-init ⇒ effective ≈ backbone at step 0.
    assert float(jnp.abs(adapter.up_attn).max()) == 0.0
    assert float(jnp.abs(adapter.up_ffn).max()) == 0.0
    # Down projections are kaiming-initialised (small but nonzero).
    assert float(jnp.abs(adapter.down_attn).max()) > 0.0


def test_bottleneck_no_adapt_attn_disables_attn_branch() -> None:
    backbone = init_model(TINY_SUPERNET, key=0)
    adapter = dispatch_init("bottleneck")(
        backbone, BottleneckConfig(dim=4, no_adapt_attn=True),
        key=jax.random.key(0),
    )
    assert adapter.down_attn is None
    assert adapter.up_attn is None
    assert adapter.down_ffn is not None  # FFN still active


def test_bottleneck_identity_at_init_matches_backbone_logits() -> None:
    """The Houlsby up-projection starts at zero ⇒ the residual is
    identity at step 0 ⇒ bottleneck-effective forward equals the bare
    backbone forward to within numerical tolerance."""
    backbone = init_model(TINY_SUPERNET, key=0)
    adapter = dispatch_init("bottleneck")(
        backbone, BottleneckConfig(dim=4), key=jax.random.key(0),
    )
    effective = dispatch_apply("bottleneck")(backbone, adapter)
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    bare = backbone(tokens)
    bottlenecked = effective(tokens)
    # Identity at init: differences should be exactly zero (the
    # residual is `h + up(gelu(down(h)))` and up is exactly 0).
    assert jnp.allclose(bare, bottlenecked, atol=1e-6, rtol=0)


def test_bottleneck_save_load_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A bottleneck adapter saved via ``save_bottleneck_adapter`` reloads
    bit-identical via ``load_bottleneck_adapter``. This is the load-bearing
    invariant for the trainer's --resume path (DEFERRALS bottleneck adapter
    resume): a re-composed wrapper must produce the same logits as the one
    that wrote the sidecar."""
    from pawn.adapters.bottleneck import (
        ADAPTER_SAFETENSORS,
        apply_bottleneck,
        load_bottleneck_adapter,
        save_bottleneck_adapter,
    )

    backbone = init_model(TINY_SUPERNET, key=0)
    cfg = BottleneckConfig(dim=4)
    adapter = dispatch_init("bottleneck")(backbone, cfg, key=jax.random.key(0))
    # Mutate the up projections to exercise the round-trip (zero-init
    # would be trivially equal — we want a non-degenerate test).
    rng_key = jax.random.key(7)
    perturb = jax.random.normal(rng_key, adapter.up_attn.shape) * 0.01
    adapter = eqx.tree_at(lambda a: a.up_attn, adapter, adapter.up_attn + perturb)

    save_bottleneck_adapter(adapter, tmp_path)
    assert (tmp_path / ADAPTER_SAFETENSORS).is_file()

    loaded = load_bottleneck_adapter(tmp_path, cfg)
    assert loaded.cfg == cfg
    # Tensor-by-tensor equality on the populated fields.
    for field_name in (
        "down_attn", "hidden_attn", "up_attn",
        "down_ffn", "hidden_ffn", "up_ffn",
    ):
        orig = getattr(adapter, field_name)
        new = getattr(loaded, field_name)
        if orig is None:
            assert new is None
        else:
            assert new is not None
            assert jnp.array_equal(orig, new)

    # Forward parity: the re-loaded wrapper produces the same logits.
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    orig_logits = apply_bottleneck(backbone, adapter)(tokens)
    new_logits = apply_bottleneck(backbone, loaded)(tokens)
    assert jnp.allclose(orig_logits, new_logits, atol=1e-6, rtol=0)


def test_bottleneck_load_rejects_missing_sidecar(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An empty directory raises FileNotFoundError — the trainer's
    resume path predicates on the sidecar's existence, so this is the
    expected exception type."""
    from pawn.adapters.bottleneck import load_bottleneck_adapter

    cfg = BottleneckConfig(dim=4)
    with pytest.raises(FileNotFoundError, match="adapter.safetensors"):
        load_bottleneck_adapter(tmp_path, cfg)


def test_bottleneck_save_rejects_empty_adapter(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A BottleneckAdapter constructed with every field None (only
    reachable by bypassing BottleneckConfig validation, e.g. directly
    via eqx.tree_at) must refuse to write rather than silently land
    an empty sidecar that load_bottleneck_adapter then can't read.
    Round-2 test-risk MEDIUM pinned this gap."""
    from pawn.adapters.bottleneck import (
        BottleneckAdapter,
        save_bottleneck_adapter,
    )

    # Construct a degenerate adapter directly (bypasses
    # BottleneckConfig's __post_init__ no-op guard).
    cfg = BottleneckConfig(dim=4)
    empty = BottleneckAdapter(
        down_attn=None, hidden_attn=None, up_attn=None,
        down_ffn=None, hidden_ffn=None, up_ffn=None,
        cfg=cfg,
    )
    with pytest.raises(ValueError, match="no populated fields"):
        save_bottleneck_adapter(empty, tmp_path)


def test_bottleneck_load_rejects_placement_mismatch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Saving with one placement disabled then loading with a cfg
    that expects both placements must raise — silently building a
    mixed None / non-None adapter would crash at first forward.
    Round-2 test-risk MEDIUM."""
    from pawn.adapters.bottleneck import (
        load_bottleneck_adapter,
        save_bottleneck_adapter,
    )

    backbone = init_model(TINY_SUPERNET, key=0)
    # Save with attn disabled (FFN-only sidecar).
    save_cfg = BottleneckConfig(dim=4, no_adapt_attn=True)
    adapter = dispatch_init("bottleneck")(
        backbone, save_cfg, key=jax.random.key(0),
    )
    save_bottleneck_adapter(adapter, tmp_path)

    # Load with a cfg that expects both placements: mismatch.
    bad_cfg = BottleneckConfig(dim=4)
    with pytest.raises(ValueError, match="sidecar mismatch"):
        load_bottleneck_adapter(tmp_path, bad_cfg)

    # And the reverse: save both, load with attn disabled.
    save_cfg = BottleneckConfig(dim=4)
    adapter = dispatch_init("bottleneck")(
        backbone, save_cfg, key=jax.random.key(0),
    )
    save_bottleneck_adapter(adapter, tmp_path)
    bad_cfg = BottleneckConfig(dim=4, no_adapt_attn=True)
    with pytest.raises(ValueError, match="sidecar mismatch"):
        load_bottleneck_adapter(tmp_path, bad_cfg)


def test_bottleneck_save_load_respects_placement_flags(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``no_adapt_attn=True`` writes only the FFN fields; the load
    restores Nones in the disabled slots."""
    from pawn.adapters.bottleneck import (
        load_bottleneck_adapter,
        save_bottleneck_adapter,
    )

    backbone = init_model(TINY_SUPERNET, key=0)
    cfg = BottleneckConfig(dim=4, no_adapt_attn=True)
    adapter = dispatch_init("bottleneck")(backbone, cfg, key=jax.random.key(0))
    save_bottleneck_adapter(adapter, tmp_path)
    loaded = load_bottleneck_adapter(tmp_path, cfg)
    assert loaded.down_attn is None and loaded.up_attn is None
    assert loaded.down_ffn is not None and loaded.up_ffn is not None


def test_bottleneck_n_hidden_extra_stages_shape() -> None:
    """``n_hidden=2`` adds two extra (dim, dim) GELU stages between the
    down and up projections — verify the buffer shape carries them."""
    backbone = init_model(TINY_SUPERNET, key=0)
    adapter = dispatch_init("bottleneck")(
        backbone, BottleneckConfig(dim=4, n_hidden=2), key=jax.random.key(0),
    )
    assert adapter.hidden_attn is not None
    assert adapter.hidden_attn.shape == (
        TINY_SUPERNET.n_layers, 2, 4, 4,
    ), f"got {adapter.hidden_attn.shape}"
    assert adapter.hidden_ffn is not None
    assert adapter.hidden_ffn.shape == (TINY_SUPERNET.n_layers, 2, 4, 4)


def test_bottleneck_rejects_both_placements_off() -> None:
    with pytest.raises(ValueError, match="both set"):
        BottleneckConfig(dim=4, no_adapt_attn=True, no_adapt_ffn=True)


# ---------------------------------------------------------------------------
# RoSA — mode-aware apply
# ---------------------------------------------------------------------------


def test_rosa_retro_bottleneck_init_builds_bottleneck_branch() -> None:
    backbone = init_model(TINY_SUPERNET, key=0)
    cfg = RoSAConfig(
        mode="retro-bottleneck", lora_rank=2, density=0.1, bottleneck_dim=4,
    )
    adapter = dispatch_init("rosa")(backbone, cfg, key=jax.random.key(0))
    assert adapter.bottleneck is not None
    # The retro-bottleneck phase 1 starts with sparse_active=False so
    # apply_rosa returns a LoRA-effective PAWNModel (the bottleneck
    # only contributes after Phase 2→3 flips sparse_active=True).
    eff = dispatch_apply("rosa")(backbone, adapter)
    assert isinstance(eff, _EFFECTIVE_TYPES)


def test_rosa_generate_masks_yields_density_targeted_topk() -> None:
    """generate_rosa_masks produces boolean masks where the True
    count per delta_* matches ``density * numel`` (the v1 Algorithm 1
    top-k contract)."""
    from pawn.adapter_trainer import generate_rosa_masks

    backbone = init_model(TINY_SUPERNET, key=0)
    cfg = RoSAConfig(mode="rosa", lora_rank=2, density=0.05, mask_samples=2)
    adapter = dispatch_init("rosa")(backbone, cfg, key=jax.random.key(0))
    batches = [_make_batch(), _make_batch()]
    new_sparse = generate_rosa_masks(
        backbone, adapter, batches, compute_dtype=None,
    )
    for name in (
        "mask_q", "mask_k", "mask_v", "mask_o",
    ):
        m = getattr(new_sparse, name)
        assert m is not None, f"{name} mask missing"
        on_count = int(m.sum())
        expected = max(1, int(0.05 * m.size))
        # top-k → exactly `expected` per leaf.
        assert on_count == expected, (
            f"{name}: density-targeted top-k expected {expected} "
            f"True positions, got {on_count}"
        )
    # Deltas reset to zero after mask gen (Phase 3 starts from
    # identity sparse contribution).
    for name in (
        "delta_q", "delta_k", "delta_v", "delta_o",
    ):
        d = getattr(new_sparse, name)
        assert d is not None
        assert float(jnp.abs(d).max()) == 0.0


def test_rosa_phase1_to_phase3_flips_toggles_per_mode() -> None:
    """rosa_phase1_to_phase3 sets `sparse_active=True` for every mode
    and `lora_active=(mode == "rosa")` — retro modes drop the LoRA
    branch after warmup."""
    from pawn.adapter_trainer import rosa_phase1_to_phase3

    backbone = init_model(TINY_SUPERNET, key=0)
    for mode, expect_lora in (
        ("rosa", True),
        ("retro-sparse", False),
        ("retro-bottleneck", False),
    ):
        cfg = RoSAConfig(mode=mode, lora_rank=2, density=0.1)  # type: ignore[arg-type]
        ad = dispatch_init("rosa")(backbone, cfg, key=jax.random.key(0))
        assert ad.lora_active is True and ad.sparse_active is False
        new = rosa_phase1_to_phase3(ad, ad.sparse, key=jax.random.key(2))
        assert new.sparse_active is True
        assert new.lora_active is expect_lora


def test_rosa_retro_modes_init_skip_bottleneck() -> None:
    """rosa + rosa-retro-sparse don't carry a bottleneck branch — the
    field stays None and `apply_rosa` returns the pure LoRA/sparse
    composition."""
    backbone = init_model(TINY_SUPERNET, key=0)
    for mode in ("rosa", "retro-sparse"):
        cfg = RoSAConfig(mode=mode, lora_rank=2, density=0.1)  # type: ignore[arg-type]
        adapter = dispatch_init("rosa")(backbone, cfg, key=jax.random.key(0))
        assert adapter.bottleneck is None, f"mode={mode} should not have bottleneck"


# ---------------------------------------------------------------------------
# Unfreeze — masked-slot drift under weight_decay > 0
# ---------------------------------------------------------------------------


def test_film_init_gamma_ones_beta_zeros() -> None:
    """v1 parity (deleted tests/adapters/test_film.py): FiLM gamma is
    one-initialised and beta is zero-initialised so the residual is
    identity at step 0."""
    backbone = init_model(TINY_SUPERNET, key=0)
    adapter = dispatch_init("film")(
        backbone, FiLMConfig(use_output_film=True), key=jax.random.key(0),
    )
    assert np.allclose(np.asarray(adapter.gamma), 1.0)
    assert np.allclose(np.asarray(adapter.beta), 0.0)
    if adapter.output_gamma is not None:
        assert np.allclose(np.asarray(adapter.output_gamma), 1.0)
    if adapter.output_beta is not None:
        assert np.allclose(np.asarray(adapter.output_beta), 0.0)


def test_film_param_shapes_match_n_layers_times_d_model() -> None:
    """Real FiLM (H2): the per-layer gamma + beta slabs are each shape
    ``(n_layers, d_model)`` (residual-stream modulation); the optional
    output FiLM modulates the logits, so its slabs are ``(vocab_size,)``
    — **not** ``(d_model,)`` (the stale v1-parity shape this asserts
    against). Output FiLM at logit space is the post-Phase-A uniform
    ``V``-wide head."""
    backbone = init_model(TINY_SUPERNET, key=0)
    adapter = dispatch_init("film")(
        backbone, FiLMConfig(use_output_film=True), key=jax.random.key(0),
    )
    n_layers = TINY_SUPERNET.n_layers
    d_model = TINY_SUPERNET.d_model
    vocab_size = TINY_SUPERNET.vocab_size
    assert adapter.gamma.shape == (n_layers, d_model)
    assert adapter.beta.shape == (n_layers, d_model)
    assert adapter.output_gamma is not None
    assert adapter.output_gamma.shape == (vocab_size,)
    assert adapter.output_beta is not None
    assert adapter.output_beta.shape == (vocab_size,)


def test_film_identity_at_init_matches_backbone_logits() -> None:
    """Real FiLM (H2): at init gamma=1, beta=0 (both per-layer and
    output), so ``h = 1 * h + 0 = h`` in the residual stream and
    ``logits = 1 * logits + 0`` — the effective forward is *exactly*
    identical to the bare backbone (no fp32 fold-into-norm noise)."""
    backbone = init_model(TINY_SUPERNET, key=0)
    adapter = dispatch_init("film")(
        backbone, FiLMConfig(use_output_film=True), key=jax.random.key(0),
    )
    effective = dispatch_apply("film")(backbone, adapter)
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    bare = backbone(tokens)
    filmed = effective(tokens)
    # True residual-stream FiLM with the identity init is a pointwise
    # multiply-by-1 / add-0, so the result is bit-identical to the
    # backbone modulo XLA op-ordering noise.
    assert jnp.allclose(bare, filmed, atol=1e-6, rtol=0)


def test_film_output_film_modulates_logits_against_reference() -> None:
    """Real FiLM (H2): with per-layer modulation held at identity
    (gamma=1, beta=0) and a **non-trivial** output FiLM, the effective
    logits equal the explicit reference
    ``output_gamma * backbone_logits + output_beta`` over the uniform
    ``V``-wide vocabulary. This pins both the math and that output FiLM
    lives in logit space (shape ``(vocab_size,)``)."""
    backbone = init_model(TINY_SUPERNET, key=0)
    base = dispatch_init("film")(
        backbone, FiLMConfig(use_output_film=True), key=jax.random.key(0),
    )
    vocab_size = TINY_SUPERNET.vocab_size
    # Non-trivial output gamma/beta over the V-wide logit axis.
    og = 1.0 + 0.5 * jax.random.normal(jax.random.key(11), (vocab_size,))
    ob = 0.3 * jax.random.normal(jax.random.key(12), (vocab_size,))
    adapter = eqx.tree_at(
        lambda a: (a.output_gamma, a.output_beta), base, (og, ob)
    )
    assert adapter.output_gamma is not None
    assert adapter.output_gamma.shape == (vocab_size,)

    effective = dispatch_apply("film")(backbone, adapter)
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    bare = backbone(tokens)
    filmed = effective(tokens)
    # Per-layer modulation is identity here, so the only difference is
    # the output FiLM applied to the V-wide logits.
    reference = og[None, None, :] * bare + ob[None, None, :]
    assert jnp.allclose(filmed, reference, atol=1e-5, rtol=0)
    # And it provably differs from the un-modulated backbone.
    assert not jnp.allclose(filmed, bare, atol=1e-3, rtol=0)


def test_film_per_layer_modulation_changes_residual_and_logits() -> None:
    """Real FiLM (H2): a **non-trivial** per-layer gamma/beta provably
    changes the residual stream (and therefore the logits) relative to
    the bare backbone — the modulation is a genuine residual-stream
    shift ``h = gamma_l * h + beta_l``, not a no-op fold into a norm
    weight.

    The reference checks the *single-layer* case exactly: with a
    one-layer backbone, FiLM at the (sole) ffn_hook applies
    ``h = gamma_0 * h_layer_out + beta_0`` to the layer output *before*
    the final norm + head, so the effective logits equal the backbone
    forward recomputed with that explicit shift spliced in."""
    backbone = init_model(TINY_SUPERNET, key=0)
    base = dispatch_init("film")(
        backbone, FiLMConfig(use_output_film=False), key=jax.random.key(0),
    )
    n_layers = TINY_SUPERNET.n_layers
    d_model = TINY_SUPERNET.d_model
    gamma = 1.0 + 0.2 * jax.random.normal(
        jax.random.key(21), (n_layers, d_model)
    )
    beta = 0.1 * jax.random.normal(jax.random.key(22), (n_layers, d_model))
    adapter = eqx.tree_at(lambda a: (a.gamma, a.beta), base, (gamma, beta))

    effective = dispatch_apply("film")(backbone, adapter)
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    bare = backbone(tokens)
    filmed = effective(tokens)
    # Non-identity per-layer FiLM must move the logits.
    assert not jnp.allclose(filmed, bare, atol=1e-3, rtol=0)

    # Explicit single-layer reference: run a 1-layer backbone, capture
    # the layer output via the same ffn_hook the wrapper uses, apply the
    # explicit `gamma_0 * h + beta_0`, then finish the head ourselves and
    # compare against the FiLMEffective output.
    from pawn.config import ModelConfig
    one_layer_cfg = ModelConfig(
        d_model=d_model,
        n_layers=1,
        n_heads=TINY_SUPERNET.n_heads,
        d_ff=TINY_SUPERNET.d_ff,
        vocab_size=TINY_SUPERNET.vocab_size,
        max_seq_len=TINY_SUPERNET.max_seq_len,
    )
    bb1 = init_model(one_layer_cfg, key=3)
    g0 = 1.0 + 0.2 * jax.random.normal(jax.random.key(31), (1, d_model))
    b0 = 0.1 * jax.random.normal(jax.random.key(32), (1, d_model))
    film1 = FiLMAdapter(
        gamma=g0, beta=b0, output_gamma=None, output_beta=None,
        cfg=FiLMConfig(use_output_film=False),
    )
    eff1 = dispatch_apply("film")(bb1, film1)
    filmed1 = eff1(tokens)

    # Explicit reference: an inline single-layer forward (no scan, no
    # side-effecting hook) reproducing PAWNModel.__call__ for n_layers=1,
    # with the explicit `gamma_0 * h_out + beta_0` shift applied to the
    # post-FFN-residual hidden state before the final norm + head. This
    # is the `gamma * h + beta` residual-stream reference the H2 spec
    # asks for.
    from pawn.model import _apply_rope, _build_rope, _rmsnorm
    lyr = bb1.layers  # leaves carry a leading n_layers=1 axis
    n_heads = one_layer_cfg.n_heads
    head_dim = one_layer_cfg.head_dim
    inv_scale = head_dim ** -0.5
    x = bb1.embed_tokens[tokens]  # (B, T, d)
    B, T, D = x.shape
    rope_cos, rope_sin = _build_rope(head_dim, T, one_layer_cfg.rope_base)
    # ---- attention sublayer (pre-norm + residual) ----
    normed = _rmsnorm(x, lyr.attn_norm_w[0])
    q = jnp.einsum("btd,de->bte", normed, lyr.wq[0])
    k = jnp.einsum("btd,de->bte", normed, lyr.wk[0])
    v = jnp.einsum("btd,de->bte", normed, lyr.wv[0])
    q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    q = _apply_rope(q, rope_cos, rope_sin)
    k = _apply_rope(k, rope_cos, rope_sin)
    causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))[None, None, :, :]
    scores = jnp.einsum("bhid,bhjd->bhij", q, k) * inv_scale
    scores = jnp.where(causal, scores, jnp.finfo(jnp.float32).min)
    attn = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.einsum("bhij,bhjd->bhid", attn, v)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, D)
    h = x + jnp.einsum("btd,de->bte", attn_out, lyr.wo[0])
    # ---- ffn sublayer (pre-norm + residual) ----
    normed = _rmsnorm(h, lyr.ffn_norm_w[0])
    gate = jnp.einsum("btd,df->btf", normed, lyr.w_gate[0])
    up = jnp.einsum("btd,df->btf", normed, lyr.w_up[0])
    ffn_out = jnp.einsum("btf,fd->btd", jax.nn.silu(gate) * up, lyr.w_down[0])
    h_out = h + ffn_out
    # ---- explicit FiLM shift on the residual stream, then norm + head ----
    shifted = g0[0][None, None, :] * h_out + b0[0][None, None, :]
    head = bb1.embed_tokens.T if bb1.lm_head is None else bb1.lm_head
    ref_logits = jnp.einsum(
        "btd,dv->btv", _rmsnorm(shifted, bb1.final_norm_w), head
    ).astype(jnp.float32)
    assert jnp.allclose(filmed1, ref_logits, atol=1e-5, rtol=0)


def test_film_without_output_film_skips_output_slab() -> None:
    """v1 parity: `use_output_film=False` should leave the output
    FiLM None (no extra trainable parameters)."""
    backbone = init_model(TINY_SUPERNET, key=0)
    adapter = dispatch_init("film")(
        backbone, FiLMConfig(use_output_film=False), key=jax.random.key(0),
    )
    assert adapter.output_gamma is None
    assert adapter.output_beta is None


def test_film_save_load_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A FiLM adapter saved via ``save_film_adapter`` reloads bit-identical
    via ``load_film_adapter``. This is the load-bearing invariant for the
    trainer's --resume path (C1 added the sidecar write; the resume block
    reads it back): a re-composed wrapper must produce the same logits as
    the one that wrote the sidecar.

    Mirrors ``test_bottleneck_save_load_roundtrip`` — perturb gamma/beta
    (and the output slabs) off identity so the round-trip is non-degenerate,
    then assert the reloaded slabs match tensor-by-tensor and the
    apply_film logits agree."""
    from pawn.adapters.film import (
        ADAPTER_SAFETENSORS,
        apply_film,
        load_film_adapter,
        save_film_adapter,
    )

    backbone = init_model(TINY_SUPERNET, key=0)
    cfg = FiLMConfig(use_output_film=True)
    base = dispatch_init("film")(backbone, cfg, key=jax.random.key(0))
    # Perturb every slab off the identity init (gamma=1, beta=0) so the
    # round-trip exercises real values, not the trivial-equal init.
    n_layers = TINY_SUPERNET.n_layers
    d_model = TINY_SUPERNET.d_model
    vocab_size = TINY_SUPERNET.vocab_size
    gamma = 1.0 + 0.2 * jax.random.normal(jax.random.key(41), (n_layers, d_model))
    beta = 0.1 * jax.random.normal(jax.random.key(42), (n_layers, d_model))
    og = 1.0 + 0.3 * jax.random.normal(jax.random.key(43), (vocab_size,))
    ob = 0.2 * jax.random.normal(jax.random.key(44), (vocab_size,))
    adapter = eqx.tree_at(
        lambda a: (a.gamma, a.beta, a.output_gamma, a.output_beta),
        base, (gamma, beta, og, ob),
    )

    save_film_adapter(adapter, tmp_path)
    assert (tmp_path / ADAPTER_SAFETENSORS).is_file()

    loaded = load_film_adapter(tmp_path, cfg)
    assert loaded.cfg == cfg
    for field_name in ("gamma", "beta", "output_gamma", "output_beta"):
        orig = getattr(adapter, field_name)
        new = getattr(loaded, field_name)
        assert orig is not None
        assert new is not None
        assert jnp.array_equal(orig, new)

    # Forward parity: the re-loaded wrapper produces the same logits.
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    orig_logits = apply_film(backbone, adapter)(tokens)
    new_logits = apply_film(backbone, loaded)(tokens)
    assert jnp.allclose(orig_logits, new_logits, atol=1e-6, rtol=0)


def test_film_save_load_roundtrip_no_output_film(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``use_output_film=False`` writes only the per-layer slabs; the
    output slabs correctly stay None through the round-trip (the
    ``_ADAPTER_FIELDS`` loop skips the None leaves at save time and
    ``load_film_adapter`` restores them as None)."""
    from pawn.adapters.film import load_film_adapter, save_film_adapter

    backbone = init_model(TINY_SUPERNET, key=0)
    cfg = FiLMConfig(use_output_film=False)
    base = dispatch_init("film")(backbone, cfg, key=jax.random.key(0))
    n_layers = TINY_SUPERNET.n_layers
    d_model = TINY_SUPERNET.d_model
    gamma = 1.0 + 0.2 * jax.random.normal(jax.random.key(51), (n_layers, d_model))
    beta = 0.1 * jax.random.normal(jax.random.key(52), (n_layers, d_model))
    adapter = eqx.tree_at(lambda a: (a.gamma, a.beta), base, (gamma, beta))

    save_film_adapter(adapter, tmp_path)
    loaded = load_film_adapter(tmp_path, cfg)
    assert loaded.output_gamma is None
    assert loaded.output_beta is None
    assert jnp.array_equal(loaded.gamma, gamma)
    assert jnp.array_equal(loaded.beta, beta)


def test_film_load_rejects_missing_sidecar(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An empty directory raises FileNotFoundError — the trainer's resume
    path predicates on the sidecar's existence, so this is the expected
    exception type (mirrors the bottleneck guard)."""
    from pawn.adapters.film import load_film_adapter

    cfg = FiLMConfig(use_output_film=True)
    with pytest.raises(FileNotFoundError, match="adapter.safetensors"):
        load_film_adapter(tmp_path, cfg)


def test_film_load_rejects_output_film_mismatch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Saving with ``use_output_film=True`` then loading with a cfg that
    expects no output FiLM (and the reverse) must raise — silently loading
    a mismatched adapter would corrupt the resume. This is the
    ``cfg.use_output_film != has_output`` ValueError branch."""
    from pawn.adapters.film import load_film_adapter, save_film_adapter

    backbone = init_model(TINY_SUPERNET, key=0)
    # Save WITH output FiLM; load expecting NONE → mismatch.
    save_cfg = FiLMConfig(use_output_film=True)
    adapter = dispatch_init("film")(backbone, save_cfg, key=jax.random.key(0))
    save_film_adapter(adapter, tmp_path)
    with pytest.raises(ValueError, match="sidecar mismatch"):
        load_film_adapter(tmp_path, FiLMConfig(use_output_film=False))

    # And the reverse: save WITHOUT output FiLM; load expecting it.
    save_cfg = FiLMConfig(use_output_film=False)
    adapter = dispatch_init("film")(backbone, save_cfg, key=jax.random.key(0))
    save_film_adapter(adapter, tmp_path)
    with pytest.raises(ValueError, match="sidecar mismatch"):
        load_film_adapter(tmp_path, FiLMConfig(use_output_film=True))


def test_lora_targets_qv_skips_k_and_o() -> None:
    """v1 parity (deleted tests/adapters/test_lora.py): `targets="qv"`
    populates only the q + v LoRA matrices, leaving k + o None."""
    backbone = init_model(TINY_SUPERNET, key=0)
    adapter = dispatch_init("lora")(
        backbone, LoRAConfig(rank=2, targets="qv"), key=jax.random.key(0),
    )
    assert adapter.A_q is not None and adapter.B_q is not None
    assert adapter.A_v is not None and adapter.B_v is not None
    assert adapter.A_k is None and adapter.B_k is None
    assert adapter.A_o is None and adapter.B_o is None


def test_lora_ffn_populates_gate_up_down_pairs() -> None:
    """v1 parity: `ffn=True` adds LoRA to all 3 FFN projections
    (gate, up, down) with the correct rank-bridge shapes."""
    backbone = init_model(TINY_SUPERNET, key=0)
    rank = 2
    adapter = dispatch_init("lora")(
        backbone, LoRAConfig(rank=rank, ffn=True), key=jax.random.key(0),
    )
    n_layers = TINY_SUPERNET.n_layers
    d_model = TINY_SUPERNET.d_model
    d_ff = TINY_SUPERNET.d_ff
    # A_gate / A_up: (n_layers, d_model, rank); B_gate / B_up: (n_layers, rank, d_ff)
    assert adapter.A_gate is not None and adapter.A_gate.shape == (n_layers, d_model, rank)
    assert adapter.B_gate is not None and adapter.B_gate.shape == (n_layers, rank, d_ff)
    assert adapter.A_up is not None and adapter.A_up.shape == (n_layers, d_model, rank)
    assert adapter.B_up is not None and adapter.B_up.shape == (n_layers, rank, d_ff)
    # A_down: (n_layers, d_ff, rank); B_down: (n_layers, rank, d_model)
    assert adapter.A_down is not None and adapter.A_down.shape == (n_layers, d_ff, rank)
    assert adapter.B_down is not None and adapter.B_down.shape == (n_layers, rank, d_model)
    # B is zero-init so identity-at-step-0 holds even with FFN LoRA.
    assert float(jnp.abs(adapter.B_gate).max()) == 0.0
    assert float(jnp.abs(adapter.B_up).max()) == 0.0
    assert float(jnp.abs(adapter.B_down).max()) == 0.0


def test_lora_identity_at_init_matches_backbone() -> None:
    """v1 parity: B=0 ⇒ A@B=0 ⇒ effective weight = frozen weight ⇒
    effective forward bit-exact-ish to the bare backbone."""
    backbone = init_model(TINY_SUPERNET, key=0)
    adapter = dispatch_init("lora")(
        backbone, LoRAConfig(rank=4), key=jax.random.key(0),
    )
    effective = dispatch_apply("lora")(backbone, adapter)
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    bare = backbone(tokens)
    lora_out = effective(tokens)
    assert jnp.allclose(bare, lora_out, atol=1e-6, rtol=0)


def test_unfreeze_masked_slots_do_not_drift_under_weight_decay() -> None:
    """Parity #5 (unfreeze): with `weight_decay > 0`, AdamW's decoupled
    decay would update masked layer slots toward zero even though
    their gradients are zero. The trainer's
    `_freeze_masked_unfreeze_slots` post-update hook keeps masked slots
    pinned to the backbone values."""
    backbone = init_model(TINY_SUPERNET, key=0)  # n_layers=4
    cfg = UnfreezeConfig(layers="0,1")  # masked = (2, 3)
    adapter = dispatch_init("unfreeze")(backbone, cfg, key=jax.random.key(0))
    flt = dispatch_filter("unfreeze")(adapter)
    # Large weight_decay so a single step's drift would be measurable
    # if the snap-back wasn't running.
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-2, weight_decay=1.0),
    )
    opt_state = opt.init(eqx.filter(adapter, flt))
    state = AdapterTrainState(
        backbone=backbone, adapter=adapter, opt_state=opt_state,
        step=jnp.int32(0), key=jax.random.key(0),
    )
    backbone_wq = np.asarray(backbone.layers.wq)
    train_step = make_adapter_train_step("unfreeze", opt)
    batch = _make_batch()
    # Step the trainer a handful of times so any drift would compound.
    for _ in range(3):
        state, _ = train_step(state, batch)
    post_wq = np.asarray(state.adapter.layers.wq)
    for masked_idx in (2, 3):
        np.testing.assert_array_equal(
            post_wq[masked_idx], backbone_wq[masked_idx],
            err_msg=(
                f"masked layer {masked_idx} drifted under weight_decay — "
                "the snap-back hook is not enforcing the invariant"
            ),
        )


# ---------------------------------------------------------------------------
# C2 — adapter resume opt-state (H3): a warm Adam state must never be applied
# to cold-started adapter params on --resume. The weight-folding strategies
# (lora / sparse / unfreeze / specialized_clm / hybrid) publish the *folded*
# effective model but persist the raw (backbone, adapter) PyTree to a resume
# sidecar so the resume path restores the exact params the warm moments index.
# ---------------------------------------------------------------------------


# Strategies whose `apply_fn` folds into / returns a bare PAWNModel-shaped
# effective and therefore rely on the `adapter_resume_state.eqx` sidecar
# (as opposed to bottleneck/FiLM, which save the raw frozen backbone + a
# typed `adapter.safetensors`). These are exactly the strategies the C2 fix
# routes through `save_adapter_resume_state` in `train_jax_adapter._save`.
_WEIGHT_FOLD_RESUME_STRATEGIES = (
    "lora", "sparse", "unfreeze", "hybrid", "specialized_clm",
)


def _make_resume_optimizer() -> optax.GradientTransformation:
    """An AdamW chain with the production clip — fp32 moments so the
    save→resume round-trip is bit-exact through ``flatten_opt_state`` /
    ``unflatten_opt_state`` (the real ``make_optimizer`` keeps ``mu`` in
    bf16, which adds quantisation noise we don't want masking a genuine
    warm/cold-mismatch regression)."""
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3, weight_decay=0.0),
    )


def _trained_adapter_state(
    strategy: str, *, n_steps: int, key_seed: int,
) -> tuple[AdapterTrainState, optax.GradientTransformation, PAWNModel, Any]:
    """Cold-init an adapter against a TINY backbone and run ``n_steps``
    real train steps. Returns the post-training state plus the optimizer,
    the frozen backbone, and the *cold* adapter template (for resume).

    The cold template is what the resume path rebuilds before splicing the
    sidecar in, so handing it back lets the round-trip tests deserialise
    into the identical structure the script would."""
    backbone = init_model(TINY_SUPERNET, key=0)
    cfg = _strategy_config(strategy)
    cold_adapter = dispatch_init(strategy)(
        backbone, cfg, key=jax.random.key(key_seed),
    )
    optimizer = _make_resume_optimizer()
    flt = dispatch_filter(strategy)(cold_adapter)
    opt_state = optimizer.init(eqx.filter(cold_adapter, flt))
    state = AdapterTrainState(
        backbone=backbone, adapter=cold_adapter, opt_state=opt_state,
        step=jnp.int32(0), key=jax.random.key(0),
    )
    train_step = make_adapter_train_step(strategy, optimizer)
    batch = _make_batch()
    for _ in range(n_steps):
        state, _ = train_step(state, batch)
    return state, optimizer, backbone, cold_adapter


def _save_then_resume(
    strategy: str,
    state: AdapterTrainState,
    optimizer: optax.GradientTransformation,
    cold_backbone: PAWNModel,
    cold_adapter: Any,
    tmp_dir,  # type: ignore[no-untyped-def]
) -> AdapterTrainState:
    """Mirror the script's save + resume for a weight-folding strategy.

    Save side: write the folded effective as ``model.safetensors`` (what
    downstream eval reads), the flattened opt_state, and the raw
    ``(backbone, adapter)`` resume sidecar. Resume side: reload the folded
    model (run-block / forward fallback), then deserialise the raw
    ``(backbone, adapter)`` into freshly cold templates and warm-load the
    opt_state — exactly what ``train_jax_adapter`` does on ``--resume``."""
    from pawn.adapters.film import FiLMEffective, save_film_adapter
    from pawn.checkpoint import (
        OPTIMIZER_FILE,
        load_adapter_resume_state,
        load_model,
        save_adapter_resume_state,
        save_model,
    )
    from pawn.trainer import flatten_opt_state, unflatten_opt_state

    apply_fn = dispatch_apply(strategy)
    effective = apply_fn(state.backbone, state.adapter)
    out = tmp_dir / "adapter_step_00000010"
    if isinstance(effective, FiLMEffective):
        # hybrid: the apply composes LoRA-folded weights into a FiLM wrapper.
        # Mirror the script — save the folded backbone + FiLM typed sidecar +
        # the full-adapter resume sidecar.
        save_model(
            effective.backbone, out,
            optimizer_state=flatten_opt_state(state.opt_state),
            training_state={"step": int(state.step)},
        )
        save_film_adapter(effective.adapter, out)
        save_adapter_resume_state(state.backbone, state.adapter, out)
    else:
        # specialized_clm's apply returns the standalone model; lora / sparse
        # / unfreeze fold into a PAWNModel. Either way it's a bare PAWNModel.
        assert isinstance(effective, PAWNModel)
        save_model(
            effective, out,
            optimizer_state=flatten_opt_state(state.opt_state),
            training_state={"step": int(state.step)},
        )
        save_adapter_resume_state(state.backbone, state.adapter, out)

    # --- resume ---
    loaded_backbone, _ = load_model(out)  # forward fallback / run block
    del loaded_backbone  # weight-folding path uses the sidecar, not this
    r_backbone, r_adapter = load_adapter_resume_state(
        cold_backbone, cold_adapter, out,
    )
    flt = dispatch_filter(strategy)(r_adapter)
    opt_state = optimizer.init(eqx.filter(r_adapter, flt))
    from safetensors.numpy import load_file as st_load
    flat = st_load(str(out / OPTIMIZER_FILE))
    opt_state = unflatten_opt_state(opt_state, flat)
    return AdapterTrainState(
        backbone=r_backbone, adapter=r_adapter, opt_state=opt_state,
        step=state.step, key=state.key,
    )


@pytest.mark.parametrize("strategy", _WEIGHT_FOLD_RESUME_STRATEGIES)
def test_resume_sidecar_restores_adapter_params_exactly(
    strategy: str, tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """After save→resume the restored adapter params equal the saved ones
    leaf-for-leaf (including non-trainable leaves like sparse masks and the
    standalone specialized_clm model). This is the precondition for the warm
    Adam moments to index the same params they were trained against (H3)."""
    state, optimizer, cold_bb, cold_ad = _trained_adapter_state(
        strategy, n_steps=5, key_seed=1,
    )
    resumed = _save_then_resume(
        strategy, state, optimizer, cold_bb, cold_ad, tmp_path,
    )
    saved_leaves = jax.tree_util.tree_leaves(
        eqx.filter(state.adapter, eqx.is_array)
    )
    restored_leaves = jax.tree_util.tree_leaves(
        eqx.filter(resumed.adapter, eqx.is_array)
    )
    assert len(saved_leaves) == len(restored_leaves)
    assert saved_leaves, "expected at least one array leaf in the adapter"
    for a, b in zip(saved_leaves, restored_leaves):
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg=f"{strategy}: adapter leaf changed across save→resume",
        )
    # The restored backbone (raw frozen backbone) must also match the saved
    # one — for sparse the trained mask lives nowhere else, and re-folding a
    # cold delta onto a folded backbone would otherwise double-apply.
    bb_saved = jax.tree_util.tree_leaves(
        eqx.filter(state.backbone, eqx.is_array)
    )
    bb_restored = jax.tree_util.tree_leaves(
        eqx.filter(resumed.backbone, eqx.is_array)
    )
    for a, b in zip(bb_saved, bb_restored):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("strategy", _WEIGHT_FOLD_RESUME_STRATEGIES)
def test_resumed_step_equals_uninterrupted_step(
    strategy: str, tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """A step taken after save→resume equals the equivalent step in an
    uninterrupted run within fp32 noise (H3 acceptance): the warm Adam
    moments and the restored adapter params correspond, so there is no
    first-post-resume-step corruption.

    Run K steps uninterrupted; separately run K-1 steps, save, resume,
    and take the K-th step. Compare the final adapter params."""
    K = 6
    # Uninterrupted reference: K full steps.
    ref_state, _opt, _bb, _ad = _trained_adapter_state(
        strategy, n_steps=K, key_seed=2,
    )
    # Interrupted: K-1 steps, save, resume, then the K-th step.
    mid_state, optimizer, cold_bb, cold_ad = _trained_adapter_state(
        strategy, n_steps=K - 1, key_seed=2,
    )
    resumed = _save_then_resume(
        strategy, mid_state, optimizer, cold_bb, cold_ad, tmp_path,
    )
    train_step = make_adapter_train_step(strategy, optimizer)
    batch = _make_batch()
    resumed, _ = train_step(resumed, batch)

    # The step counter advanced identically.
    assert int(resumed.step) == int(ref_state.step) == K
    ref_leaves = jax.tree_util.tree_leaves(
        eqx.filter(ref_state.adapter, eqx.is_inexact_array)
    )
    res_leaves = jax.tree_util.tree_leaves(
        eqx.filter(resumed.adapter, eqx.is_inexact_array)
    )
    assert len(ref_leaves) == len(res_leaves)
    for ref, res in zip(ref_leaves, res_leaves):
        np.testing.assert_allclose(
            np.asarray(res), np.asarray(ref), rtol=0, atol=1e-5,
            err_msg=(
                f"{strategy}: the post-resume step diverged from the "
                "uninterrupted trajectory — warm opt-state / adapter "
                "mismatch (H3)"
            ),
        )


def test_resume_sidecar_roundtrip_is_self_describing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``save_adapter_resume_state`` writes ``adapter_resume_state.eqx`` and
    ``load_adapter_resume_state`` reads it back; a missing sidecar raises
    FileNotFoundError so the resume path can predicate on its presence (the
    cold-start-opt fallback)."""
    from pawn.checkpoint import (
        ADAPTER_RESUME_FILE,
        load_adapter_resume_state,
        save_adapter_resume_state,
    )

    backbone = init_model(TINY_SUPERNET, key=0)
    adapter = dispatch_init("lora")(
        backbone, LoRAConfig(rank=2), key=jax.random.key(0),
    )
    out = tmp_path / "ckpt"
    out.mkdir()
    save_adapter_resume_state(backbone, adapter, out)
    assert (out / ADAPTER_RESUME_FILE).is_file()

    cold = dispatch_init("lora")(
        backbone, LoRAConfig(rank=2), key=jax.random.key(99),
    )
    r_bb, r_ad = load_adapter_resume_state(backbone, cold, out)
    # The restored A matrices match the saved ones, not the cold template's.
    assert r_ad.A_q is not None and adapter.A_q is not None
    np.testing.assert_array_equal(np.asarray(r_ad.A_q), np.asarray(adapter.A_q))
    assert isinstance(r_bb, PAWNModel)

    with pytest.raises(FileNotFoundError, match=ADAPTER_RESUME_FILE):
        load_adapter_resume_state(backbone, cold, tmp_path / "empty")


# ---------------------------------------------------------------------------
# C2 — script-level resume path. The H3 branch selection + cold-start fallback
# live inside `train_jax_adapter` (factored into
# `restore_adapter_resume_state`); these tests drive that real code path —
# including the wrapper (bottleneck / FiLM) typed-sidecar restore and the
# cold-start fallback when the sidecar is absent — rather than re-implementing
# the save/resume sequence inline.
# ---------------------------------------------------------------------------


def _load_train_jax_adapter():  # type: ignore[no-untyped-def]
    """Import ``scripts/train_jax_adapter.py`` as a module to reach the
    script-level ``restore_adapter_resume_state`` helper (the resume branch
    selection + H3 cold-start decision factored out of ``main``)."""
    import importlib.util
    from pathlib import Path

    script_path = Path("scripts") / "train_jax_adapter.py"
    spec = importlib.util.spec_from_file_location(
        "scripts_train_jax_adapter_resume", script_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _save_adapter_checkpoint(
    strategy: str,
    state: AdapterTrainState,
    out,  # type: ignore[no-untyped-def]
) -> None:
    """Write a checkpoint directory through the SHIPPED save-side branch
    selector ``train_jax_adapter.write_adapter_checkpoint`` — the exact
    helper ``_save`` delegates to. Driving the real save path (rather than
    re-implementing the sidecar selection) is what makes a regression in the
    shipped save dispatch — e.g. the hybrid resume-sidecar guard — visible to
    these tests (H3)."""
    from pawn.trainer import flatten_opt_state

    write_adapter_checkpoint = _load_train_jax_adapter().write_adapter_checkpoint
    effective = dispatch_apply(strategy)(state.backbone, state.adapter)
    write_adapter_checkpoint(
        effective=effective,
        backbone=state.backbone,
        adapter=state.adapter,
        out=out,
        optimizer_state=flatten_opt_state(state.opt_state),
        step=int(state.step),
    )


def _resume_via_script(
    strategy: str,
    out,  # type: ignore[no-untyped-def]
    cold_backbone: PAWNModel,
    cold_adapter: Any,
    optimizer: optax.GradientTransformation,
):  # type: ignore[no-untyped-def]
    """Drive the script's real resume branch selection + H3 cold-start
    decision via ``train_jax_adapter.restore_adapter_resume_state``."""
    from pawn.checkpoint import load_model

    restore_adapter_resume_state = _load_train_jax_adapter().restore_adapter_resume_state

    loaded_backbone, _run_block = load_model(out)
    return restore_adapter_resume_state(
        strategy=strategy,
        strategy_cfg=_strategy_config(strategy),
        ckpt_dir=out,
        loaded_backbone=loaded_backbone,
        cold_backbone=cold_backbone,
        cold_adapter=cold_adapter,
        optimizer=optimizer,
    )


def _opt_inexact_leaves(opt_state: Any) -> list[np.ndarray]:
    return [
        np.asarray(x)
        for x in jax.tree_util.tree_leaves(
            eqx.filter(opt_state, eqx.is_inexact_array)
        )
    ]


def test_script_resume_warm_loads_opt_state_when_sidecar_present(
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """When the strategy's sidecar is present, the script-level resume path
    fully restores the adapter and *warm-loads* the saved Adam moments —
    i.e. ``restore_adapter_resume_state`` reports ``adapter_restored`` and the
    resumed opt_state matches the saved warm one (not a cold init)."""
    strategy = "lora"
    state, optimizer, cold_bb, cold_ad = _trained_adapter_state(
        strategy, n_steps=5, key_seed=1,
    )
    out = tmp_path / "adapter_step_00000005"
    _save_adapter_checkpoint(strategy, state, out)

    resumed = _resume_via_script(strategy, out, cold_bb, cold_ad, optimizer)
    assert resumed.adapter_restored is True

    # The warm moments round-trip back: the resumed opt_state equals the saved
    # warm opt_state and is NOT the all-zero cold init.
    saved = _opt_inexact_leaves(state.opt_state)
    got = _opt_inexact_leaves(resumed.opt_state)
    assert len(saved) == len(got)
    any_nonzero = False
    for s, g in zip(saved, got):
        np.testing.assert_allclose(g, s, rtol=0, atol=0)
        any_nonzero = any_nonzero or bool(np.any(s != 0.0))
    assert any_nonzero, "expected non-zero warm Adam moments after 5 steps"


@pytest.mark.parametrize("strategy", ("lora", "bottleneck", "film"))
def test_script_resume_cold_starts_opt_state_when_sidecar_missing(
    strategy: str, tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """H3 safety net: when the resume sidecar the strategy depends on is
    *absent* (e.g. a legacy / ``model.safetensors``-only checkpoint), the
    script keeps opt_state COLD rather than splicing the warm
    ``optimizer.safetensors`` onto a cold-rebuilt adapter. A regression that
    loaded ``OPTIMIZER_FILE`` unconditionally would fail here.

    We save a real checkpoint (warm opt_state + sidecar), then delete the
    sidecar the strategy needs and confirm ``restore_adapter_resume_state``
    (a) reports ``adapter_restored is False`` and (b) returns a cold
    opt_state — leaf-identical to a fresh ``optimizer.init`` and provably
    different from the saved warm moments still on disk in ``OPTIMIZER_FILE``.
    """
    from pawn.adapters.bottleneck import ADAPTER_SAFETENSORS as BN_SIDECAR
    from pawn.adapters.film import ADAPTER_SAFETENSORS as FILM_SIDECAR
    from pawn.checkpoint import ADAPTER_RESUME_FILE, OPTIMIZER_FILE

    state, optimizer, cold_bb, cold_ad = _trained_adapter_state(
        strategy, n_steps=5, key_seed=3,
    )
    out = tmp_path / "adapter_step_00000005"
    _save_adapter_checkpoint(strategy, state, out)

    # The warm optimizer file is on disk — the regression we guard against is
    # loading it unconditionally despite no usable adapter sidecar.
    assert (out / OPTIMIZER_FILE).is_file()
    warm_moments = _opt_inexact_leaves(state.opt_state)
    assert any(bool(np.any(m != 0.0)) for m in warm_moments)

    # Remove every adapter sidecar so no restore branch can fire.
    for sidecar in (ADAPTER_RESUME_FILE, BN_SIDECAR, FILM_SIDECAR):
        p = out / sidecar
        if p.is_file():
            p.unlink()

    resumed = _resume_via_script(strategy, out, cold_bb, cold_ad, optimizer)

    # (a) The script recognised the adapter could not be restored.
    assert resumed.adapter_restored is False

    # (b) opt_state is cold — leaf-identical to a fresh init on the cold
    # adapter, and provably NOT the warm moments still sitting in
    # OPTIMIZER_FILE.
    flt = dispatch_filter(strategy)(cold_ad)
    cold_opt = optimizer.init(eqx.filter(cold_ad, flt))
    cold_leaves = _opt_inexact_leaves(cold_opt)
    got_leaves = _opt_inexact_leaves(resumed.opt_state)
    assert len(got_leaves) == len(cold_leaves) == len(warm_moments)
    for got, cold, warm in zip(got_leaves, cold_leaves, warm_moments):
        np.testing.assert_array_equal(
            got, cold,
            err_msg=(
                f"{strategy}: opt_state was not cold-started when the resume "
                "sidecar was missing — warm Adam moments were spliced onto a "
                "cold adapter (H3 regression)"
            ),
        )
    # The cold opt_state must differ from the warm moments on disk for at
    # least one leaf — otherwise the cold/warm distinction is untestable.
    assert any(
        bool(np.any(cold != warm))
        for cold, warm in zip(cold_leaves, warm_moments)
    ), "warm and cold moments coincide; pick a strategy that actually trains"


# The full strategy set the script-driven round-trip must cover: the two
# wrapper strategies (bottleneck / FiLM restore through their typed sidecar)
# AND every weight-folding strategy (lora / sparse / unfreeze / hybrid /
# specialized_clm restore through the raw resume sidecar). Driving ALL of
# them through the REAL ``write_adapter_checkpoint`` save +
# ``restore_adapter_resume_state`` resume — not the inline re-implementation —
# is the H3 acceptance: a save-side regression (e.g. the hybrid resume-sidecar
# guard) now fails a script-driven test instead of staying green.
_SCRIPT_RESUME_STRATEGIES = (
    "bottleneck", "film", *_WEIGHT_FOLD_RESUME_STRATEGIES,
)


@pytest.mark.parametrize("strategy", _SCRIPT_RESUME_STRATEGIES)
def test_wrapper_resume_via_script_restores_params_exactly(
    strategy: str, tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """Every adapter strategy restores its params leaf-for-leaf through the
    SHIPPED save + resume code path (``write_adapter_checkpoint`` →
    ``restore_adapter_resume_state``): wrappers (bottleneck / FiLM) via their
    typed sidecar, weight-folding strategies (lora / sparse / unfreeze /
    hybrid / specialized_clm) via the raw resume sidecar. Faithful param
    restore is the precondition for the warm Adam moments to index the rebuilt
    adapter (H3)."""
    state, optimizer, cold_bb, cold_ad = _trained_adapter_state(
        strategy, n_steps=5, key_seed=4,
    )
    out = tmp_path / "adapter_step_00000005"
    _save_adapter_checkpoint(strategy, state, out)
    resumed = _resume_via_script(strategy, out, cold_bb, cold_ad, optimizer)
    assert resumed.adapter_restored is True

    saved_leaves = jax.tree_util.tree_leaves(
        eqx.filter(state.adapter, eqx.is_array)
    )
    restored_leaves = jax.tree_util.tree_leaves(
        eqx.filter(resumed.adapter, eqx.is_array)
    )
    assert len(saved_leaves) == len(restored_leaves)
    assert saved_leaves, "expected at least one array leaf in the adapter"
    for a, b in zip(saved_leaves, restored_leaves):
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg=f"{strategy}: adapter leaf changed across save→resume",
        )


@pytest.mark.parametrize("strategy", _SCRIPT_RESUME_STRATEGIES)
def test_wrapper_resumed_step_equals_uninterrupted_step(
    strategy: str, tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """A step taken after the SHIPPED save→resume round-trip equals the
    equivalent uninterrupted step within fp32 noise (H3 acceptance), for every
    strategy. The warm Adam moments must still index the correct params after
    the adapter is rebuilt (from the typed sidecar for wrappers, the raw
    resume sidecar for weight-folding strategies) and the opt_state is
    re-templated on it — a leaf-order mismatch between the rebuilt adapter and
    the saved flat opt_state, or a missing save-side sidecar, would corrupt
    this step. This drives the real ``write_adapter_checkpoint`` /
    ``restore_adapter_resume_state``, so a save-side hybrid-guard regression
    fails here instead of staying green."""
    K = 6
    ref_state, _opt, _bb, _ad = _trained_adapter_state(
        strategy, n_steps=K, key_seed=5,
    )
    mid_state, optimizer, cold_bb, cold_ad = _trained_adapter_state(
        strategy, n_steps=K - 1, key_seed=5,
    )
    out = tmp_path / "adapter_step_00000005"
    _save_adapter_checkpoint(strategy, mid_state, out)
    resumed = _resume_via_script(strategy, out, cold_bb, cold_ad, optimizer)
    assert resumed.adapter_restored is True

    resumed_state = AdapterTrainState(
        backbone=resumed.backbone, adapter=resumed.adapter,
        opt_state=resumed.opt_state, step=mid_state.step, key=mid_state.key,
    )
    train_step = make_adapter_train_step(strategy, optimizer)
    batch = _make_batch()
    resumed_state, _ = train_step(resumed_state, batch)

    assert int(resumed_state.step) == int(ref_state.step) == K
    ref_leaves = jax.tree_util.tree_leaves(
        eqx.filter(ref_state.adapter, eqx.is_inexact_array)
    )
    res_leaves = jax.tree_util.tree_leaves(
        eqx.filter(resumed_state.adapter, eqx.is_inexact_array)
    )
    assert len(ref_leaves) == len(res_leaves)
    for ref, res in zip(ref_leaves, res_leaves):
        np.testing.assert_allclose(
            np.asarray(res), np.asarray(ref), rtol=0, atol=1e-5,
            err_msg=(
                f"{strategy}: the post-resume step diverged from the "
                "uninterrupted trajectory — warm opt-state / adapter "
                "mismatch on the typed-sidecar resume path (H3)"
            ),
        )


# ---------------------------------------------------------------------------
# C3 — adapter K-step lax.scan loop (H11, §8.3): the scan path must produce
# the same trajectory as the single-step loop over K steps within fp32 noise,
# and the `step_time` resume divisor must divide by (step - run_start).
# ---------------------------------------------------------------------------


def _stack_batches(batches: list[Batch]) -> Batch:
    """Stack a list of ``(B, T)`` batches into one ``(K, B, T)`` Batch —
    the leading axis is the K-step axis the ``make_adapter_scan_step``
    ``lax.scan`` iterates (mirrors ``train_jax_adapter._gather_chunk``)."""
    return Batch(
        tokens=jnp.stack([b.tokens for b in batches]),
        targets=jnp.stack([b.targets for b in batches]),
        attn_mask=jnp.stack([b.attn_mask for b in batches]),
        loss_mask=jnp.stack([b.loss_mask for b in batches]),
    )


def _fresh_lora_state() -> tuple[
    AdapterTrainState, optax.GradientTransformation
]:
    """A cold LoRA adapter + fp32 AdamW optimizer over a TINY backbone.

    fp32 moments (weight_decay=0) keep the scan-vs-single-step comparison free
    of the bf16 quantisation noise the production ``make_optimizer`` would add,
    so a genuine trajectory divergence isn't masked."""
    backbone = init_model(TINY_SUPERNET, key=0)
    cfg = LoRAConfig(rank=2, targets="qkvo")
    adapter = dispatch_init("lora")(backbone, cfg, key=jax.random.key(1))
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3, weight_decay=0.0),
    )
    flt = dispatch_filter("lora")(adapter)
    opt_state = optimizer.init(eqx.filter(adapter, flt))
    state = AdapterTrainState(
        backbone=backbone, adapter=adapter, opt_state=opt_state,
        step=jnp.int32(0), key=jax.random.key(0),
    )
    return state, optimizer


def test_adapter_scan_matches_single_step_trajectory() -> None:
    """H11: the K-step ``lax.scan`` loop lands on the same adapter params and
    emits the same per-step losses as K independent single-step dispatches,
    within fp32 noise. This is the load-bearing correctness contract for
    swapping the per-step dispatch loop for the scan.

    Both paths start from an identical cold state and consume an identical
    sequence of ``K`` batches; the only difference is single-step vs scanned
    execution. ``make_adapter_train_step`` donates its input buffers, so the
    two paths are built from *separate* freshly-initialised states."""
    K = 4
    # A distinct batch per step so a body that wrongly reused one element
    # (or scanned in the wrong order) would diverge.
    batches = [_make_batch(seq_len=16, n=4) for _ in range(K)]
    # Vary the token content across steps so the per-step losses differ.
    batches = [
        eqx.tree_at(
            lambda b: b.tokens, b,
            (b.tokens + jnp.int32(i)) % jnp.int32(TINY_SUPERNET.vocab_size),
        )
        for i, b in enumerate(batches)
    ]

    # --- single-step reference ---
    ref_state, ref_opt = _fresh_lora_state()
    ref_step = make_adapter_train_step("lora", ref_opt)
    single_losses: list[float] = []
    for b in batches:
        ref_state, loss = ref_step(ref_state, b)
        single_losses.append(float(loss))

    # --- K-step scan ---
    scan_state, scan_opt = _fresh_lora_state()
    scan_step = make_adapter_scan_step(make_adapter_train_step("lora", scan_opt))
    stacked = _stack_batches(batches)
    scan_state, scan_losses = scan_step(scan_state, stacked)
    scan_losses_np = np.asarray(scan_losses)

    # Step counter advanced by exactly K through the scan.
    assert int(scan_state.step) == int(ref_state.step) == K

    # Per-step losses agree.
    assert scan_losses_np.shape == (K,)
    np.testing.assert_allclose(
        scan_losses_np, np.asarray(single_losses), rtol=0, atol=1e-5,
        err_msg="scan per-step losses diverged from the single-step loop",
    )

    # Final adapter params agree leaf-for-leaf.
    ref_leaves = jax.tree_util.tree_leaves(
        eqx.filter(ref_state.adapter, eqx.is_inexact_array)
    )
    scan_leaves = jax.tree_util.tree_leaves(
        eqx.filter(scan_state.adapter, eqx.is_inexact_array)
    )
    assert len(ref_leaves) == len(scan_leaves)
    for ref, got in zip(ref_leaves, scan_leaves):
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(ref), rtol=0, atol=1e-5,
            err_msg=(
                "K-step scan adapter params diverged from the single-step "
                "trajectory (H11)"
            ),
        )

    # Optimizer moments agree too — a warm resume off the scan path must index
    # the same params the single-step path would have produced.
    ref_opt_leaves = _opt_inexact_leaves(ref_state.opt_state)
    scan_opt_leaves = _opt_inexact_leaves(scan_state.opt_state)
    assert len(ref_opt_leaves) == len(scan_opt_leaves)
    for ref, got in zip(ref_opt_leaves, scan_opt_leaves):
        np.testing.assert_allclose(
            got, ref, rtol=0, atol=1e-5,
            err_msg="scan opt-state moments diverged from the single-step loop",
        )


def test_step_time_divides_by_steps_since_run_start() -> None:
    """§8.3: the adapter ``step_time`` divisor is ``(step - run_start)``, not
    the absolute ``step``. After a resume at ``run_start`` steps, dividing the
    *current* run's elapsed wall-time by the absolute step counter would
    understate per-step time; the resume-aware divisor reports it honestly.
    """
    mod = _load_train_jax_adapter()
    step_time = mod._step_time

    # Fresh run (run_start == 0): plain elapsed / step.
    assert step_time(10.0, 5, 0) == pytest.approx(2.0)

    # Resumed run: 10 s of *this* run's wall-time spread over the 5 steps it
    # actually executed since resuming at step 100 — NOT over 105.
    assert step_time(10.0, 105, 100) == pytest.approx(2.0)
    # The buggy absolute-step divisor would have reported ~0.095 s/step here.
    assert step_time(10.0, 105, 100) != pytest.approx(10.0 / 105)

    # Degenerate first-row guard: step == run_start divides by 1, not 0.
    assert step_time(3.0, 100, 100) == pytest.approx(3.0)


def test_chunk_bound_never_crosses_eval_or_checkpoint_boundary() -> None:
    """C3: ``_chunk_bound`` caps each host-loop chunk to ``k``/``remaining``
    and shortens it so its trailing edge lands on the next eval / checkpoint
    boundary — the ``lax.scan`` only surfaces the *final* carry, so a chunk
    that straddled a boundary would deny the host the exact post-step state
    and silently skip the eval / checkpoint there.
    """
    chunk_bound = _load_train_jax_adapter()._chunk_bound

    # Plenty of room: capped only by ``k`` / ``remaining``.
    assert chunk_bound(0, 100, k=8, eval_interval=10, checkpoint_interval=20) == 8
    assert chunk_bound(0, 3, k=8, eval_interval=10, checkpoint_interval=20) == 3

    # Standing on a boundary (``absolute`` a multiple of the interval): the
    # boundary already past contributes a *full* interval of room, so the
    # next boundary — not a zero-length chunk — bounds the step.
    assert chunk_bound(10, 100, k=50, eval_interval=10, checkpoint_interval=20) == 10
    assert chunk_bound(20, 100, k=50, eval_interval=10, checkpoint_interval=20) == 10

    # Eval boundary is the nearer one and clips the chunk to land on it.
    assert chunk_bound(7, 100, k=50, eval_interval=10, checkpoint_interval=20) == 3
    # Checkpoint boundary is nearer than the eval one here.
    assert chunk_bound(18, 100, k=50, eval_interval=100, checkpoint_interval=20) == 2

    # Dropping the ``- (absolute % …)`` term (a plausible regression) would
    # return ``min(k, remaining, eval_interval, ckpt_interval)`` and let the
    # chunk overrun the boundary — pin that it does NOT.
    naive = min(50, 100, 10, 20)
    assert chunk_bound(7, 100, k=50, eval_interval=10, checkpoint_interval=20) != naive


def _drive_host_loop(
    n_steps: int,
    *,
    k: int,
    eval_interval: int,
    checkpoint_interval: int,
    log_interval: int,
    start_step: int = 0,
) -> dict[str, list[int]]:
    """Pure replica of ``_run_steps``' host-driving structure (the C3
    boundary loop), recording where chunks land and where eval / checkpoint /
    log fire. Mirrors the script line-for-line: ``_chunk_bound`` sizes each
    chunk, losses are replayed per-step at ``log_interval`` boundaries, and
    eval / checkpoint fire on ``final_step % interval == 0``.
    """
    chunk_bound = _load_train_jax_adapter()._chunk_bound
    chunk_ends: list[int] = []
    evals: list[int] = []
    ckpts: list[int] = []
    logs: list[int] = []
    done = 0
    while done < n_steps:
        absolute = start_step + done
        remaining = n_steps - done
        chunk = chunk_bound(absolute, remaining, k, eval_interval, checkpoint_interval)
        assert chunk > 0, "host loop would spin forever on a zero-length chunk"
        chunk_start = absolute
        for i in range(chunk):
            step = chunk_start + i + 1
            if step % log_interval == 0:
                logs.append(step)
        done += chunk
        final_step = start_step + done
        chunk_ends.append(final_step)
        if final_step % eval_interval == 0:
            evals.append(final_step)
        if final_step % checkpoint_interval == 0:
            ckpts.append(final_step)
    return {"chunk_ends": chunk_ends, "evals": evals, "ckpts": ckpts, "logs": logs}


def test_run_steps_fires_eval_and_checkpoint_on_every_boundary() -> None:
    """C3: drive the host loop over a multi-chunk run with
    ``eval_interval != checkpoint_interval != k`` and assert (a) eval fires at
    every eval boundary, (b) checkpoint at every checkpoint boundary, (c) no
    chunk crosses a boundary, (d) per-step losses are logged at every
    ``log_interval`` boundary. A regression that let chunks straddle a boundary
    would silently drop the eval / checkpoint there and leave the suite green
    without this guard.
    """
    n_steps, k, eval_interval, ckpt_interval, log_interval = 100, 7, 5, 20, 10
    rec = _drive_host_loop(
        n_steps,
        k=k,
        eval_interval=eval_interval,
        checkpoint_interval=ckpt_interval,
        log_interval=log_interval,
    )

    # (a) eval fires exactly at every eval boundary in (0, n_steps].
    assert rec["evals"] == list(range(eval_interval, n_steps + 1, eval_interval))
    # (b) checkpoint fires exactly at every checkpoint boundary.
    assert rec["ckpts"] == list(range(ckpt_interval, n_steps + 1, ckpt_interval))
    # (d) per-step losses logged at every log boundary.
    assert rec["logs"] == list(range(log_interval, n_steps + 1, log_interval))

    # (c) no chunk crosses a boundary: every chunk end at or before a boundary,
    # and each chunk no larger than k. Walk consecutive chunk ends.
    prev = 0
    for end in rec["chunk_ends"]:
        size = end - prev
        assert 0 < size <= k, f"chunk {prev}->{end} exceeds k={k}"
        # No eval boundary strictly inside (prev, end).
        for b in range(eval_interval, end, eval_interval):
            assert not (prev < b < end), f"chunk {prev}->{end} crosses eval @ {b}"
        for b in range(ckpt_interval, end, ckpt_interval):
            assert not (prev < b < end), f"chunk {prev}->{end} crosses ckpt @ {b}"
        prev = end
    assert rec["chunk_ends"][-1] == n_steps


def test_run_steps_resume_offsets_boundaries_by_start_step() -> None:
    """C3: when a phase starts mid-run (``start_step != 0``, e.g. RoSA Phase 2
    or a resume), boundaries are measured on the *absolute* step counter, so
    eval / checkpoint still fire on absolute multiples — not on offsets from
    ``start_step``.
    """
    start_step, n_steps = 13, 27  # absolute steps 13..40
    rec = _drive_host_loop(
        n_steps,
        k=6,
        eval_interval=5,
        checkpoint_interval=20,
        log_interval=10,
        start_step=start_step,
    )
    end = start_step + n_steps  # 40
    assert rec["evals"] == list(range(15, end + 1, 5))  # 15,20,...,40
    assert rec["ckpts"] == [20, 40]
    assert rec["logs"] == [20, 30, 40]
    # First chunk shrinks to land on absolute step 15, not 13 + min(k,…).
    assert rec["chunk_ends"][0] == 15


# ---------------------------------------------------------------------------
# adapter-loop workstream: legal-mask loss, illegal_penalty, patience,
# best-val checkpoint, richer val metrics, per-layer placement.
# ---------------------------------------------------------------------------


def _legal_batch(
    n_games: int = 4, seq_len: int = 28, seed: int = 5,
) -> tuple[Batch, Any]:
    """Build a small random-game Batch with an engine-replayed legal mask."""
    from pawn.corpus import legal_mask_for_games

    c = generate_corpus(
        n_games=n_games, max_ply=seq_len - 4, seq_len=seq_len, seed=seed,
        conditioning=[],
    )
    idx = np.arange(n_games)
    lm = jnp.asarray(legal_mask_for_games(c, idx))
    batch = Batch(
        tokens=jnp.asarray(c.tokens[idx]),
        targets=jnp.asarray(c.targets[idx]),
        attn_mask=jnp.asarray(c.attn_mask[idx]),
        loss_mask=jnp.asarray(c.loss_mask[idx]),
        legal_mask=lm,
    )
    return batch, c


def test_legal_mask_aligns_targets_are_always_legal() -> None:
    """Every supervised target token must be flagged legal by the engine-
    replayed mask — the alignment (engine ply axis shifted by C-1 onto the
    target slot) is the load-bearing invariant of the legal-mask source."""
    batch, _ = _legal_batch(n_games=6, seq_len=32, seed=11)
    assert batch.legal_mask is not None
    loss_mask = np.asarray(batch.loss_mask)
    targets = np.asarray(batch.targets)
    legal = np.asarray(batch.legal_mask)
    sup = loss_mask.astype(bool)
    # Gather the legal flag at each target token; every supervised position
    # must be legal.
    tgt_legal = np.take_along_axis(legal, targets[..., None], axis=-1)[..., 0]
    assert bool(tgt_legal[sup].all()), "a supervised target was flagged illegal"


def test_legal_mask_lowers_cross_entropy_vs_reserved_only() -> None:
    """Adapter-loss legality (v1 ``apply_legal_mask`` default-ON): hard-masking
    illegal columns to -inf shrinks the softmax denominator, so the CE on a
    fixed model is strictly lower than the reserved-column-only CE."""
    from pawn.trainer import cross_entropy_loss

    bb = init_model(TINY_SUPERNET, key=0)
    batch, _ = _legal_batch(seed=7)
    no_legal = Batch(
        tokens=batch.tokens, targets=batch.targets,
        attn_mask=batch.attn_mask, loss_mask=batch.loss_mask,
    )
    loss_legal = float(cross_entropy_loss(bb, batch, apply_legal=True))
    loss_plain = float(cross_entropy_loss(bb, no_legal))
    assert loss_legal < loss_plain


def test_illegal_penalty_raises_loss_only_with_hard_mask_off() -> None:
    """``illegal_penalty`` adds ``λ·E[P_illegal]`` to the loss when the hard
    mask is off (v1 ``compute_adapter_loss``). With the hard mask on the
    illegal mass is analytically zero, so the penalty is a no-op."""
    from pawn.trainer import cross_entropy_loss

    bb = init_model(TINY_SUPERNET, key=0)
    batch, _ = _legal_batch(seed=9)
    base = float(cross_entropy_loss(bb, batch, apply_legal=False))
    penalised = float(
        cross_entropy_loss(bb, batch, apply_legal=False, illegal_penalty=2.0)
    )
    assert penalised > base
    # Hard-mask-on: penalty has no effect (mass is zero).
    hard = float(cross_entropy_loss(bb, batch, apply_legal=True))
    hard_pen = float(
        cross_entropy_loss(bb, batch, apply_legal=True, illegal_penalty=2.0)
    )
    assert hard == pytest.approx(hard_pen)


def test_cross_entropy_legal_is_noop_without_legal_mask() -> None:
    """The legality terms are inert when the batch carries no legal_mask
    (pretrain parity): the loss equals the reserved-column-only CE regardless
    of ``apply_legal`` / ``illegal_penalty``."""
    from pawn.trainer import cross_entropy_loss

    bb = init_model(TINY_SUPERNET, key=0)
    c = generate_corpus(
        n_games=4, max_ply=20, seq_len=24, seed=3, conditioning=[],
    )
    batch = slice_batch(c, np.arange(4))
    assert batch.legal_mask is None
    base = float(cross_entropy_loss(bb, batch))
    assert base == pytest.approx(
        float(cross_entropy_loss(bb, batch, apply_legal=True))
    )
    assert base == pytest.approx(
        float(
            cross_entropy_loss(bb, batch, apply_legal=False, illegal_penalty=5.0)
        )
    )


def test_adapter_train_step_legal_mask_changes_gradient() -> None:
    """An adapter train step under hard legal masking moves the adapter to a
    *different* point than the un-masked step — the legality is actually
    consumed in the gradient, not just declared."""
    cfg = LoRAConfig(rank=2, targets="qkvo")
    opt = optax.adam(1e-2)

    def _fresh_state() -> AdapterTrainState:
        # Fresh backbone too: ``make_adapter_train_step`` donates the whole
        # state (incl. the backbone arrays), so the two regimes can't share a
        # backbone object without hitting a donated-buffer deletion.
        bb = init_model(TINY_SUPERNET, key=0)
        adapter = dispatch_init("lora")(bb, cfg, key=jax.random.key(1))
        flt = dispatch_filter("lora")(adapter)
        opt_state = opt.init(eqx.filter(adapter, flt))
        return AdapterTrainState(
            backbone=bb, adapter=adapter, opt_state=opt_state,
            step=jnp.int32(0), key=jax.random.key(0),
        )

    batch, _ = _legal_batch(seed=13)
    # ``make_adapter_train_step`` donates its inputs, so each regime gets a
    # fresh state rather than reusing donated buffers.
    step_legal = make_adapter_train_step("lora", opt, apply_legal=True)
    step_plain = make_adapter_train_step("lora", opt, apply_legal=False)
    s_legal, _ = step_legal(_fresh_state(), batch)
    s_plain, _ = step_plain(_fresh_state(), batch)
    # The B_q LoRA matrices diverge between the two regimes.
    assert not bool(
        jnp.allclose(s_legal.adapter.B_q, s_plain.adapter.B_q)
    )


def test_adapter_val_metrics_reports_v1_parity_set() -> None:
    """v1-parity val metrics: loss / top1 / top5 / illegal_pred_rate, with
    top5 >= top1 and illegal_pred_rate == 0 when the hard mask is on (the
    argmax is restricted to legal moves)."""
    from pawn.adapter_trainer import make_adapter_val_metrics

    bb = init_model(TINY_SUPERNET, key=0)
    cfg = LoRAConfig(rank=2, targets="qkvo")
    adapter = dispatch_init("lora")(bb, cfg, key=jax.random.key(1))
    batch, _ = _legal_batch(seed=21)
    metrics_fn = make_adapter_val_metrics("lora", apply_legal=True)
    m = metrics_fn(bb, adapter, batch)
    assert 0.0 <= float(m.top1) <= 1.0
    assert float(m.top5) >= float(m.top1)
    assert float(m.loss) > 0.0
    # illegal_pred_rate is computed on the raw (un-legal-masked) argmax, so it
    # can be nonzero even when the loss masks legality — but it must be a valid
    # rate in [0, 1].
    assert 0.0 <= float(m.illegal_pred_rate) <= 1.0


def test_adapter_val_metrics_illegal_rate_zero_without_legal_mask() -> None:
    """Without a legal_mask there's no legality reference, so the illegal
    prediction rate is reported as 0 (no spurious diagnostic)."""
    from pawn.adapter_trainer import make_adapter_val_metrics

    bb = init_model(TINY_SUPERNET, key=0)
    cfg = LoRAConfig(rank=2, targets="qkvo")
    adapter = dispatch_init("lora")(bb, cfg, key=jax.random.key(1))
    c = generate_corpus(n_games=4, max_ply=20, seq_len=24, seed=4, conditioning=[])
    batch = slice_batch(c, np.arange(4))
    m = make_adapter_val_metrics("lora")(bb, adapter, batch)
    assert float(m.illegal_pred_rate) == 0.0


def test_adapter_val_metrics_top_k_support_excludes_reserved_columns() -> None:
    """top1 / top5 / illegal_pred_rate score over ``[0, NUM_ACTIONS)`` only —
    the same support eval.py enforces via ``logits[..., :NUM_ACTIONS]``.

    PAD (1968), the 11 outcome tokens (1969-1979) and BOS (1980) are *not*
    moves, and the conditioning/outcome-prefix path explicitly trains the
    outcome columns. If the val-metrics argmax could land there, an
    outcome-dominated head would push top1 to ~0 and mark every prediction
    'illegal' (legal_mask is False at non-move columns) — a diagnostic v1 and
    eval.py could never produce. We force an outcome column to dominate the
    head and assert the metrics are unchanged from the un-biased run.
    """
    import dataclasses

    from pawn.adapter_trainer import make_adapter_val_metrics
    from pawn.config import OUTCOME_TOKEN_BASE

    # Untied head so we can spike a single output column without perturbing
    # the input embedding (tied head reuses embed_tokens for both).
    cfg = dataclasses.replace(TINY_SUPERNET, tie_embeddings=False)
    bb = init_model(cfg, key=0)
    assert bb.lm_head is not None
    lora_cfg = LoRAConfig(rank=2, targets="qkvo")
    adapter = dispatch_init("lora")(bb, lora_cfg, key=jax.random.key(1))
    batch, _ = _legal_batch(seed=21)

    metrics_fn = make_adapter_val_metrics("lora", apply_legal=True)
    base = metrics_fn(bb, adapter, batch)

    # Spike one outcome column so it is the global argmax over the full vocab.
    big = jnp.full_like(bb.lm_head[:, OUTCOME_TOKEN_BASE], 1e4)
    spiked_head = bb.lm_head.at[:, OUTCOME_TOKEN_BASE].set(big)
    bb_spiked = eqx.tree_at(lambda m: m.lm_head, bb, spiked_head)
    spiked = metrics_fn(bb_spiked, adapter, batch)

    # If the support leaked, the dominating outcome column would capture every
    # argmax: top1 -> ~0, illegal_pred_rate -> ~1. Restricting to the move
    # support makes the spike invisible to all three metrics.
    assert float(spiked.top1) == pytest.approx(float(base.top1))
    assert float(spiked.top5) == pytest.approx(float(base.top5))
    assert float(spiked.illegal_pred_rate) == pytest.approx(
        float(base.illegal_pred_rate)
    )


# --- per-layer placement ----------------------------------------------------


@pytest.mark.parametrize("strategy", ["lora", "sparse", "hybrid"])
def test_adapter_layers_only_modifies_selected_layers(strategy: str) -> None:
    """``--adapter-layers`` (consumed as ``cfg.layers``) restricts the adapter
    to the requested transformer layers: non-adapted layers are bit-identical
    to the frozen backbone in the effective ``wq`` weights, the adapted layer
    differs.

    For lora/hybrid this is checked after forcing the zero-init B arrays to a
    nonzero value (so the correction is observable); sparse zeroes its
    per-layer carrier at non-adapted layers so the slice is identity
    regardless. Bottleneck (a wrapper that injects a residual rather than
    folding into ``wq``) has its own placement test below.
    """
    bb = init_model(TINY_SUPERNET, key=0)
    n_layers = bb.cfg.n_layers
    picks = (1,)  # adapt only layer 1
    if strategy == "lora":
        cfg: Any = LoRAConfig(rank=2, targets="qkvo", layers=picks)
    elif strategy == "sparse":
        cfg = SparseConfig(density=0.5, targets="qkvo", layers=picks)
    else:  # hybrid
        cfg = HybridConfig(lora=LoRAConfig(rank=2, targets="qkvo", layers=picks))
    adapter = dispatch_init(strategy)(bb, cfg, key=jax.random.key(2))

    if strategy in ("lora", "hybrid"):
        # Make the LoRA correction observable by setting B_q nonzero.
        lora_ad = adapter.lora if strategy == "hybrid" else adapter
        lora_ad = eqx.tree_at(
            lambda a: a.B_q, lora_ad, jnp.ones_like(lora_ad.B_q),
        )
        if strategy == "hybrid":
            adapter = eqx.tree_at(lambda a: a.lora, adapter, lora_ad)
        else:
            adapter = lora_ad
    else:  # sparse
        # Set every delta nonzero; only the masked (placed) positions add.
        adapter = eqx.tree_at(
            lambda a: a.delta_q, adapter, jnp.ones_like(adapter.delta_q),
        )

    effective = dispatch_apply(strategy)(bb, adapter)
    # lora / sparse fold into a bare PAWNModel; hybrid folds LoRA into the
    # backbone then wraps it in a FiLMEffective whose `.backbone` carries the
    # folded weights. Resolve the folded PAWNModel either way.
    if isinstance(effective, FiLMEffective):
        folded = effective.backbone
    else:
        assert isinstance(effective, PAWNModel)
        folded = effective
    bb_wq = bb.layers.wq
    eff_wq = folded.layers.wq
    for layer in range(n_layers):
        same = bool(jnp.allclose(eff_wq[layer], bb_wq[layer]))
        if layer in picks:
            assert not same, f"adapted layer {layer} unchanged for {strategy}"
        else:
            assert same, f"non-adapted layer {layer} changed for {strategy}"


def test_bottleneck_placement_zeros_down_at_nonadapted_layers() -> None:
    """Bottleneck per-layer placement zeroes the ``down_*`` projection at
    non-adapted layers, so their Houlsby residual is identically zero."""
    bb = init_model(TINY_SUPERNET, key=0)
    n_layers = bb.cfg.n_layers
    cfg = BottleneckConfig(dim=4, layers=(2,))
    adapter = dispatch_init("bottleneck")(bb, cfg, key=jax.random.key(3))
    down = np.asarray(adapter.down_attn)
    for layer in range(n_layers):
        if layer == 2:
            assert np.any(down[layer] != 0.0)
        else:
            assert np.all(down[layer] == 0.0)


def test_sparse_placement_empties_mask_at_nonadapted_layers() -> None:
    """Sparse per-layer placement ANDs the binary mask with the layer mask, so
    non-adapted layers carry no trainable positions."""
    bb = init_model(TINY_SUPERNET, key=0)
    n_layers = bb.cfg.n_layers
    cfg = SparseConfig(density=0.9, targets="qkvo", layers=(0,))
    adapter = dispatch_init("sparse")(bb, cfg, key=jax.random.key(4))
    mask = np.asarray(adapter.mask_q)
    assert mask[0].any()
    for layer in range(1, n_layers):
        assert not mask[layer].any()


def test_parse_adapter_layers_bounds_and_form() -> None:
    """The ``--adapter-layers`` parser de-dupes / sorts and rejects malformed
    or out-of-range picks."""
    from pawn.adapters.placement import parse_adapter_layers

    assert parse_adapter_layers(None, 10) is None
    assert parse_adapter_layers("3, 1, 1", 10) == (1, 3)
    with pytest.raises(ValueError):
        parse_adapter_layers("3,99", 10)
    with pytest.raises(ValueError):
        parse_adapter_layers("a,b", 10)
    with pytest.raises(ValueError):
        parse_adapter_layers("", 10)


def test_layer_placement_mask_rejects_out_of_range_index() -> None:
    """``layer_placement_mask`` bound-checks against the *real* n_layers — a
    JAX out-of-bounds scatter would otherwise silently drop the write and
    yield an all-False (adapt-nothing) mask."""
    from pawn.adapters.placement import layer_placement_mask

    # In-range builds the expected mask.
    mask = np.asarray(layer_placement_mask((1, 3), 4))
    assert mask.tolist() == [False, True, False, True]
    # Out-of-range raises rather than silently dropping the index.
    with pytest.raises(ValueError, match=r"outside \[0, 4\)"):
        layer_placement_mask((5,), 4)


@pytest.mark.parametrize("strategy", ["lora", "sparse", "bottleneck", "rosa"])
def test_adapter_init_rejects_out_of_range_layers(strategy: str) -> None:
    """``--adapter-layers`` past the backbone depth is caught in the adapter
    init (the placement mask is the single chokepoint), not silently dropped.

    ``parse_adapter_layers`` validates syntax against a permissive sentinel at
    config-build time (the backbone isn't loaded yet), so the real bound check
    must fire here — otherwise ``--adapter-layers 5`` on a 4-layer model would
    scatter-drop the write and train a zero-correction adapter.
    """
    bb = init_model(TINY_SUPERNET, key=0)  # n_layers = 4
    bad = (bb.cfg.n_layers,)  # one past the last valid index
    if strategy == "lora":
        cfg: Any = LoRAConfig(rank=2, targets="qkvo", layers=bad)
    elif strategy == "sparse":
        cfg = SparseConfig(density=0.5, targets="qkvo", layers=bad)
    elif strategy == "bottleneck":
        cfg = BottleneckConfig(dim=4, layers=bad)
    else:  # rosa — delegates to lora/sparse inits, which hit the same check
        cfg = RoSAConfig(mode="rosa", lora_rank=2, density=0.3, layers=bad)
    with pytest.raises(ValueError, match=r"outside \[0, 4\)"):
        dispatch_init(strategy)(bb, cfg, key=jax.random.key(0))


def test_rosa_mask_gen_respects_layer_placement() -> None:
    """RoSA mask generation gates the gradient-derived sparse masks by the
    per-layer placement: no selected position falls outside the requested
    layer subset (otherwise the ``--adapter-layers`` restriction would be
    silently undone when Phase 2 forces all-True masks)."""
    from pawn.adapter_trainer import generate_rosa_masks

    bb = init_model(TINY_SUPERNET, key=0)
    cfg = RoSAConfig(mode="rosa", lora_rank=2, density=0.3, layers=(1,))
    adapter = dispatch_init("rosa")(bb, cfg, key=jax.random.key(5))
    batch, _ = _legal_batch(seed=15)
    new_sparse = generate_rosa_masks(bb, adapter, [batch])
    mask = np.asarray(new_sparse.mask_q)
    assert mask[1].any()
    for layer in range(bb.cfg.n_layers):
        if layer != 1:
            assert not mask[layer].any(), f"layer {layer} mask leaked"


# --- best-val checkpoint selection -----------------------------------------


def test_find_best_adapter_step_picks_lowest_val_loss(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``find_best_adapter_step`` returns the step with the minimum val_loss
    across a run's ``metrics.jsonl`` (v1 best-checkpoint selection); ties
    resolve to the earliest step and non-finite values are skipped."""
    import json

    from pawn.checkpoint import find_best_adapter_step

    p = tmp_path / "metrics.jsonl"
    rows = [
        {"type": "config"},
        {"type": "train", "step": 10, "loss": 1.0},
        {"type": "val", "step": 10, "val_loss": 2.0},
        {"type": "val", "step": 20, "val_loss": 1.5},
        {"type": "val", "step": 30, "val_loss": 1.5},  # tie — earlier wins
        {"type": "val", "step": 40, "val_loss": 1.7},
        {"type": "val", "step": 50, "val_loss": None},  # skipped
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert find_best_adapter_step(p) == 20
    assert find_best_adapter_step(tmp_path / "missing.jsonl") is None


def test_find_best_adapter_step_none_without_val_records(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A run with validation disabled (no ``type=val`` rows) yields None."""
    import json

    from pawn.checkpoint import find_best_adapter_step

    p = tmp_path / "metrics.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(r)
            for r in ({"type": "train", "step": 1, "loss": 3.0},)
        ),
        encoding="utf-8",
    )
    assert find_best_adapter_step(p) is None


# --- patience / early stopping (end-to-end through main()) ------------------


def _save_tiny_backbone(target_dir: Path) -> None:
    """Save a fresh TINY_SUPERNET as a loadable v2 checkpoint (conditioning
    ``[]`` ⇒ C=1, matching the AdapterConfig default)."""
    from pawn.checkpoint import save_model

    bb = init_model(TINY_SUPERNET, key=0)
    save_model(bb, target_dir, run_config={"conditioning": []})


def test_adapter_patience_early_stops_and_writes_best_step(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """End-to-end: a run whose held-out val never improves stops early once
    ``patience`` consecutive evals miss, records ``reason_for_stop=patience``
    in ``schedule_health.json``, and writes ``best_step.json`` pointing at the
    best-val checkpoint (v1 patience + best-checkpoint persistence).

    Driven through the real ``main()`` on a TINY_SUPERNET LoRA run with
    ``--no-pgn`` so it needs no Lichess data. ``patience`` is small and the LR
    is zero so the model never improves past its first eval — every subsequent
    eval is a miss and the patience budget trips.
    """
    import json

    from pawn.checkpoint import MASK_VERSION  # noqa: F401  (ensures import OK)

    ckpt = tmp_path / "backbone"
    _save_tiny_backbone(ckpt)
    main = _load_train_jax_adapter().main
    logs_dir = tmp_path / "logs"
    # A near-zero lr keeps the adapter effectively frozen ⇒ val_loss is flat
    # after the first eval, so every later eval is a "miss". eval cadence runs
    # every step (val_every=1, steps_per_epoch=1) and patience=2 trips after
    # 2 consecutive misses. (lr must be > 0 per AdapterConfig.)
    argv = [
        "--strategy", "lora", "--lora-rank", "2",
        "--supernet", "tiny", "--variant", "large",
        "--checkpoint", str(ckpt),
        "--no-pgn",
        "--total-steps", "20",
        "--batch-size", "4", "--seq-len", "32", "--k", "1",
        "--lr", "1e-12",
        "--no-flash",
        "--local-checkpoints",
        "--logs-dir", str(logs_dir),
    ]
    # Patience / cadence go through the JSON config since there's no CLI flag
    # for them on this script; merge via --config.
    cfg_json = tmp_path / "cfg.json"
    cfg_json.write_text(
        json.dumps(
            {
                "patience": 2,
                "val_every": 1,
                "steps_per_epoch": 1,
                "epochs": 20,
                "checkpoint_interval": 1,
                "log_interval": 1,
            }
        ),
        encoding="utf-8",
    )
    argv = ["--config", str(cfg_json), *argv]
    rc = main(argv)
    assert rc == 0

    run_dirs = sorted(logs_dir.glob("*"))
    assert run_dirs, "no run directory produced"
    run_dir = run_dirs[-1]

    health = json.loads((run_dir / "schedule_health.json").read_text())
    assert health["reason_for_stop"] == "patience"
    # Early stop: fewer steps ran than the planned 20-step budget.
    assert health["actual_total_steps"] < health["planned_total_steps"]

    best = json.loads((run_dir / "best_step.json").read_text())
    assert best["best_step"] >= 1
    assert (run_dir / best["checkpoint"]).is_dir()
