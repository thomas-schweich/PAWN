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
from pawn.adapters.bottleneck import BottleneckEffective
from pawn.config import TINY_SUPERNET
from pawn.corpus import generate_corpus
from pawn.model import PAWNModel, init_model
from pawn.trainer import Batch, slice_batch


# `apply_fn` may return either the patched ``PAWNModel`` (for adapters that
# fold corrections into the backbone weights — LoRA, sparse, FiLM,
# unfreeze, ...) or a callable wrapper such as ``BottleneckEffective`` (for
# adapters whose semantics require post-sublayer residual injection). The
# trainer + eval scripts treat both uniformly; the test accepts either.
_EFFECTIVE_TYPES = (PAWNModel, BottleneckEffective)


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
    """v1 parity: FiLM has 2 trainable param slabs per layer
    (gamma + beta), each shape ``(n_layers, d_model)``; optional
    output FiLM adds a third slab of shape ``(d_model,)``."""
    backbone = init_model(TINY_SUPERNET, key=0)
    adapter = dispatch_init("film")(
        backbone, FiLMConfig(use_output_film=True), key=jax.random.key(0),
    )
    n_layers = TINY_SUPERNET.n_layers
    d_model = TINY_SUPERNET.d_model
    assert adapter.gamma.shape == (n_layers, d_model)
    assert adapter.beta.shape == (n_layers, d_model)
    assert adapter.output_gamma is not None
    assert adapter.output_gamma.shape == (d_model,)
    assert adapter.output_beta is not None
    assert adapter.output_beta.shape == (d_model,)


def test_film_identity_at_init_matches_backbone_logits() -> None:
    """v1 parity: at init the residual is identity, so the effective
    forward should match the bare backbone within bf16 tolerance."""
    backbone = init_model(TINY_SUPERNET, key=0)
    adapter = dispatch_init("film")(
        backbone, FiLMConfig(use_output_film=True), key=jax.random.key(0),
    )
    effective = dispatch_apply("film")(backbone, adapter)
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    bare = backbone(tokens)
    filmed = effective(tokens)
    # FiLM folds beta into attn_norm_w (not as a residual) so the
    # identity is "approximately identity" — within fp32 noise on
    # untrained random weights.
    assert jnp.allclose(bare, filmed, atol=1e-5, rtol=0)


def test_film_without_output_film_skips_output_slab() -> None:
    """v1 parity: `use_output_film=False` should leave the output
    FiLM None (no extra trainable parameters)."""
    backbone = init_model(TINY_SUPERNET, key=0)
    adapter = dispatch_init("film")(
        backbone, FiLMConfig(use_output_film=False), key=jax.random.key(0),
    )
    assert adapter.output_gamma is None
    assert adapter.output_beta is None


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
