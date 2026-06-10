"""Tests for the factored (v1-architecture) training pipeline.

Covers the pieces the ``--arch factored-v1`` experiment adds:

- ``ModelConfig.factored_embeddings`` validation (incompatible with
  ``tie_embeddings``; nested-slice rejection across architectures).
- ``init_factored_model`` shapes + config guard.
- Forward contract: logits ``(B, T, 1980)``, fp32 out, finite under both
  fp32 and bf16 AMP; ``mask_reserved_columns`` is a no-op at V=1980.
- ``to_v1_contract``: slot-0 PAD/mask semantics, target preservation,
  C=1 requirement.
- Checkpoint roundtrip via the ``factored_embeddings`` dispatch +
  ``require_uniform`` guard.
- Trainer integration: a single ``is_supernet=True`` VariantSpec drives
  ``make_train_step`` end-to-end (loss finite and decreasing on a tiny
  real corpus), and the factored-model × sliceable-variants pairing is
  rejected loudly.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pawn.checkpoint import (
    CheckpointIntegrityError,
    load_model,
    require_uniform,
    save_model,
)
from pawn.config import (
    NUM_ACTIONS,
    PAD_TOKEN,
    TINY_FACTORED,
    TINY_SUPERNET,
    V1_VOCAB_SIZE,
    FACTORED_V1_LARGE,
    ModelConfig,
    NestingError,
    validate_nested,
)
from pawn.corpus import generate_corpus, to_v1_contract
from pawn.factored_model import FactoredPAWNModel, init_factored_model
from pawn.model import PAWNModel, init_model
from pawn.run_config import PretrainConfig
from pawn.trainer import (
    Batch,
    TrainState,
    VariantSpec,
    make_lr_schedule,
    make_optimizer,
    make_train_step,
    mask_reserved_columns,
    slice_batch,
    supernet_joint_loss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_factored() -> FactoredPAWNModel:
    return init_factored_model(TINY_FACTORED, key=0)


def _v1_batch(batch_size: int = 4, seq_len: int = 64) -> Batch:
    corpus = generate_corpus(
        n_games=batch_size, max_ply=seq_len, seq_len=seq_len, seed=0,
        conditioning=(),
    )
    return slice_batch(to_v1_contract(corpus), np.arange(batch_size))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_factored_config_constants_valid() -> None:
    assert FACTORED_V1_LARGE.factored_embeddings
    assert not FACTORED_V1_LARGE.tie_embeddings
    assert FACTORED_V1_LARGE.vocab_size == V1_VOCAB_SIZE == 1980
    assert FACTORED_V1_LARGE.head_dim == 80
    assert FACTORED_V1_LARGE.d_ff == 2560
    assert TINY_FACTORED.factored_embeddings


def test_factored_rejects_tied_embeddings() -> None:
    with pytest.raises(ValueError, match="incompatible with tie_embeddings"):
        ModelConfig(
            d_model=64, n_layers=1, n_heads=1, d_ff=128, head_dim=64,
            tie_embeddings=True, factored_embeddings=True,
        )


def test_validate_nested_rejects_cross_architecture() -> None:
    with pytest.raises(NestingError, match="factored_embeddings mismatch"):
        validate_nested(TINY_FACTORED, dataclasses.replace(
            TINY_FACTORED, factored_embeddings=False, tie_embeddings=False,
        ))


def test_pretrain_config_factored_guards() -> None:
    # conditioning must be empty (no BOS/control rows in the v1 vocab).
    with pytest.raises(ValueError, match="conditioning"):
        PretrainConfig(
            run_type="pretrain", total_steps=10, arch="factored-v1",
            local_checkpoints=True, conditioning=["outcome"],
        )
    # explicit variant selections are not applicable — including
    # ("large",), which names the UNIFORM supernet, not the factored
    # config (round-2 review: an earlier carve-out let it no-op silently).
    for variants in (("small",), ("large",)):
        with pytest.raises(ValueError, match="variants"):
            PretrainConfig(
                run_type="pretrain", total_steps=10, arch="factored-v1",
                local_checkpoints=True, variants=variants,
            )
    # stochastic_variants (the v2 default True) is neutralised, not fatal.
    cfg = PretrainConfig(
        run_type="pretrain", total_steps=10, arch="factored-v1",
        local_checkpoints=True,
    )
    assert cfg.stochastic_variants is False


# ---------------------------------------------------------------------------
# Init + forward
# ---------------------------------------------------------------------------


def test_init_factored_model_shapes() -> None:
    m = _tiny_factored()
    d = TINY_FACTORED.d_model
    assert m.embed_src.shape == (64, d)
    assert m.embed_dst.shape == (64, d)
    assert m.embed_promo.shape == (5, d)
    assert m.embed_pad.shape == (d,)
    assert m.embed_outcome.shape == (TINY_FACTORED.n_outcomes, d)
    assert m.lm_head.shape == (d, V1_VOCAB_SIZE)
    assert m.layers.wq.shape == (TINY_FACTORED.n_layers, d, d)
    # embed_pad is zero-initialised (v1 parity).
    assert float(jnp.abs(m.embed_pad).max()) == 0.0


def test_init_factored_model_rejects_uniform_cfg() -> None:
    with pytest.raises(ValueError, match="factored_embeddings"):
        init_factored_model(TINY_SUPERNET, key=0)


def test_factored_forward_shape_and_dtype() -> None:
    m = _tiny_factored()
    batch = _v1_batch()
    logits = m(batch.tokens, batch.attn_mask)
    assert logits.shape == (*batch.tokens.shape, V1_VOCAB_SIZE)
    assert logits.dtype == jnp.float32
    assert bool(jnp.isfinite(logits).all())


def test_factored_forward_bf16_amp_returns_finite_fp32() -> None:
    m = _tiny_factored()
    batch = _v1_batch()
    logits = m(batch.tokens, batch.attn_mask, compute_dtype=jnp.bfloat16)
    # The head matmul is ALWAYS fp32 (v2 stability recipe).
    assert logits.dtype == jnp.float32
    assert bool(jnp.isfinite(logits).all())


def test_factored_hidden_states_stack() -> None:
    m = _tiny_factored()
    batch = _v1_batch(batch_size=2, seq_len=32)
    hs = m.hidden_states(batch.tokens, batch.attn_mask)
    assert hs.shape == (
        TINY_FACTORED.n_layers + 1, 2, 32, TINY_FACTORED.d_model
    )


def test_mask_reserved_columns_noop_at_v1_vocab() -> None:
    # Reserved columns start at NULL_TOKEN=1981 > 1980, so the v1-width
    # logit table has nothing to mask: the CE path is byte-equivalent to
    # an unmasked softmax for the factored model.
    logits = jnp.zeros((2, 3, V1_VOCAB_SIZE))
    masked = mask_reserved_columns(logits)
    assert bool((masked == logits).all())


# ---------------------------------------------------------------------------
# to_v1_contract
# ---------------------------------------------------------------------------


def test_to_v1_contract_semantics() -> None:
    corpus = generate_corpus(
        n_games=8, max_ply=32, seq_len=64, seed=3, conditioning=(),
    )
    v1c = to_v1_contract(corpus)
    # Slot 0: masked, unsupervised PAD.
    assert (v1c.tokens[:, 0] == PAD_TOKEN).all()
    assert not v1c.attn_mask[:, 0].any()
    assert not v1c.loss_mask[:, 0].any()
    # Everything else byte-identical.
    np.testing.assert_array_equal(v1c.tokens[:, 1:], corpus.tokens[:, 1:])
    np.testing.assert_array_equal(v1c.targets, corpus.targets)
    np.testing.assert_array_equal(v1c.attn_mask[:, 1:], corpus.attn_mask[:, 1:])
    np.testing.assert_array_equal(v1c.loss_mask[:, 1:], corpus.loss_mask[:, 1:])
    np.testing.assert_array_equal(v1c.game_lengths, corpus.game_lengths)
    # All tokens in-vocab for the v1 model (no BOS=1980 anywhere).
    assert int(v1c.tokens.max()) < V1_VOCAB_SIZE
    # Source corpus untouched (copy, not view).
    assert corpus.attn_mask[:, 0].any()


def test_to_v1_contract_rejects_wider_prefix() -> None:
    corpus = generate_corpus(
        n_games=4, max_ply=16, seq_len=64, seed=0, conditioning=("outcome",),
    )
    with pytest.raises(ValueError, match="C=1"):
        to_v1_contract(corpus)


# ---------------------------------------------------------------------------
# Checkpoint roundtrip
# ---------------------------------------------------------------------------


def test_factored_checkpoint_roundtrip(tmp_path) -> None:
    m = _tiny_factored()
    out = save_model(m, tmp_path / "step_00000001", run_config={"arch": "factored-v1"})
    loaded, run_block = load_model(out)
    assert isinstance(loaded, FactoredPAWNModel)
    assert run_block == {"arch": "factored-v1"}
    assert loaded.cfg == TINY_FACTORED
    for leaf_a, leaf_b in zip(
        jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_inexact_array)),
        jax.tree_util.tree_leaves(eqx.filter(loaded, eqx.is_inexact_array)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(leaf_a), np.asarray(leaf_b))


def test_factored_checkpoint_cross_load_refused(tmp_path) -> None:
    # A uniform checkpoint's payload can't load into a factored config —
    # the schemas differ structurally (embed_tokens vs embed_src/...).
    uniform = init_model(TINY_SUPERNET, key=0)
    out = save_model(uniform, tmp_path / "step_00000001")
    import json
    cfg_path = out / "config.json"
    raw = json.loads(cfg_path.read_text())
    raw["model"]["factored_embeddings"] = True
    raw["model"]["tie_embeddings"] = False
    cfg_path.write_text(json.dumps(raw, indent=2) + "\n")
    # The sentinel guard fires first (config.json was mutated post-save) —
    # that's the right failure: a tampered checkpoint must not load.
    with pytest.raises(CheckpointIntegrityError):
        load_model(out)


def test_require_uniform_guard() -> None:
    m = _tiny_factored()
    with pytest.raises(TypeError, match="uniform PAWNModel"):
        require_uniform(m, "the test path")
    u = init_model(TINY_SUPERNET, key=0)
    assert require_uniform(u, "the test path") is u


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------


def _factored_spec() -> tuple[VariantSpec, ...]:
    return (VariantSpec("factored-v1-large", TINY_FACTORED, is_supernet=True),)


def test_supernet_joint_loss_rejects_factored_with_sliced_variants() -> None:
    m = _tiny_factored()
    batch = _v1_batch(batch_size=2, seq_len=32)
    bad = (
        VariantSpec("small", TINY_FACTORED, is_supernet=False),
    )
    with pytest.raises(TypeError, match="uniform"):
        supernet_joint_loss(m, batch, bad)


def test_factored_train_step_loss_decreases() -> None:
    cfg = PretrainConfig(
        run_type="pretrain", total_steps=60, batch_size=8, seq_len=64,
        lr=3e-3, lr_schedule="constant", warmup_frac=0.0,
        arch="factored-v1", supernet="tiny", local_checkpoints=True,
    )
    assert cfg.total_steps is not None
    schedule = make_lr_schedule(cfg, cfg.total_steps)
    optimizer = make_optimizer(cfg, schedule)
    model = _tiny_factored()
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    state = TrainState(
        model=model, opt_state=opt_state, step=jnp.int32(0),
        key=jax.random.key(0),
    )
    train_step = make_train_step(optimizer, _factored_spec())

    corpus = to_v1_contract(generate_corpus(
        n_games=8 * 60, max_ply=64, seq_len=64, seed=7, conditioning=(),
    ))
    losses = []
    for i in range(60):
        batch = slice_batch(corpus, np.arange(i * 8, (i + 1) * 8))
        state, loss = train_step(state, batch)
        losses.append(float(loss))
    assert all(np.isfinite(losses))
    # Loss should clearly decrease from the random-init plateau (~ln 1980
    # ≈ 7.6 over the reachable support) within 60 steps at lr 3e-3.
    assert np.mean(losses[-10:]) < np.mean(losses[:10]) - 0.5
    # The trained model is still the factored class with finite weights.
    assert isinstance(state.model, FactoredPAWNModel)
    leaves = jax.tree_util.tree_leaves(
        eqx.filter(state.model, eqx.is_inexact_array)
    )
    assert all(bool(jnp.isfinite(leaf).all()) for leaf in leaves)


def test_factored_resume_restores_opt_state(tmp_path) -> None:
    """save_model(optimizer_state=…) → load_resume_state roundtrip on the
    FACTORED PyTree: the leaf paths differ from the uniform model
    (embed_src/dst/promo/pad/outcome vs embed_tokens), so flatten/unflatten
    of the Optax state takes an untested path that must hold before the
    400k run's first real resume (round-1 review, test-risk)."""
    from pawn.lifecycle import load_resume_state
    from pawn.trainer import flatten_opt_state

    cfg = PretrainConfig(
        run_type="pretrain", total_steps=20, batch_size=4, seq_len=64,
        lr=1e-3, arch="factored-v1", supernet="tiny", local_checkpoints=True,
    )
    assert cfg.total_steps is not None
    optimizer = make_optimizer(cfg, make_lr_schedule(cfg, cfg.total_steps))
    model = _tiny_factored()
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    state = TrainState(
        model=model, opt_state=opt_state, step=jnp.int32(0),
        key=jax.random.key(0),
    )
    train_step = make_train_step(optimizer, _factored_spec())
    corpus = to_v1_contract(generate_corpus(
        n_games=8, max_ply=64, seq_len=64, seed=11, conditioning=(),
    ))
    for i in range(2):  # two steps so Adam moments are non-trivial
        state, _ = train_step(state, slice_batch(corpus, np.arange(4) + i * 4))

    out = save_model(
        state.model, tmp_path / "step_00000002",
        run_config=cfg.model_dump(),
        optimizer_state=flatten_opt_state(state.opt_state),
        training_state={"step": 2},
    )
    resumed = load_resume_state(out, optimizer, jax.random.key(0),
                                conditioning=cfg.conditioning)
    assert isinstance(resumed.model, FactoredPAWNModel)
    assert int(resumed.step) == 2
    for a, b in zip(
        jax.tree_util.tree_leaves(eqx.filter(state.opt_state, eqx.is_inexact_array)),
        jax.tree_util.tree_leaves(eqx.filter(resumed.opt_state, eqx.is_inexact_array)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_factored_train_step_with_accumulation() -> None:
    """accumulation_steps=2 (the production config's shape): batches gain a
    leading microbatch axis and the factored model runs under the nested
    accumulation scan (round-1 review, test-risk)."""
    cfg = PretrainConfig(
        run_type="pretrain", total_steps=10, batch_size=4, seq_len=64,
        lr=1e-3, arch="factored-v1", supernet="tiny", local_checkpoints=True,
        accumulation_steps=2,
    )
    assert cfg.total_steps is not None
    optimizer = make_optimizer(cfg, make_lr_schedule(cfg, cfg.total_steps))
    model = _tiny_factored()
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    state = TrainState(
        model=model, opt_state=opt_state, step=jnp.int32(0),
        key=jax.random.key(0),
    )
    train_step = make_train_step(
        optimizer, _factored_spec(), accumulation_steps=2,
    )
    corpus = to_v1_contract(generate_corpus(
        n_games=8, max_ply=64, seq_len=64, seed=13, conditioning=(),
    ))
    flat = slice_batch(corpus, np.arange(8))
    micro = jax.tree_util.tree_map(
        lambda x: x.reshape(2, 4, *x.shape[1:]), flat
    )
    # Pre-step reference: the accumulation scan computes both micros at the
    # PRE-update weights and reports their mean — pin the math so a scan
    # bug (e.g. last-micro-only, or sum-not-mean) can't pass as "finite"
    # (round-2 review, test-risk).
    micro_losses = [
        float(supernet_joint_loss(
            state.model,
            jax.tree_util.tree_map(lambda x, i=i: x[i], micro),
            _factored_spec(),
        ))
        for i in range(2)
    ]
    state, loss = train_step(state, micro)
    assert bool(jnp.isfinite(loss))
    assert int(state.step) == 1
    np.testing.assert_allclose(
        float(loss), float(np.mean(micro_losses)), rtol=1e-5,
    )


def test_factored_rejects_use_flash() -> None:
    """Pallas flash drops the PAD mask (right-pad invariant); the v1
    contract's masked slot-0 LEADING pad violates it, so the factored model
    must hard-reject rather than silently attend to it (round-1 review,
    codex P2). The config layer neutralises the default-True flag too."""
    m = _tiny_factored()
    batch = _v1_batch(batch_size=2, seq_len=32)
    with pytest.raises(ValueError, match="use_flash"):
        m(batch.tokens, batch.attn_mask, use_flash=True)
    cfg = PretrainConfig(
        run_type="pretrain", total_steps=10, arch="factored-v1",
        local_checkpoints=True,
    )
    assert cfg.use_flash is False  # neutralised from the default True


def test_converted_v1_model_roundtrips_through_v2_checkpoint(tmp_path) -> None:
    """End-to-end converter contract on a synthetic tiny v1 checkpoint: the
    loaded model's cfg must carry factored_embeddings=True so the v2
    checkpoint layer's cfg-driven dispatch can save AND re-load the
    converted object (round-1 review, codex P2 — previously the cfg said
    uniform, so save_model wrote factored tensors under a uniform config,
    an unloadable checkpoint)."""
    import json

    from safetensors.numpy import save_file as st_save_np

    from pawn._legacy.legacy import load_v1_factored_model

    d, d_ff, n_layers, V, n_out = 8, 16, 2, 1980, 11
    rng = np.random.default_rng(0)

    def t(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    state = {
        "embed.src_embed.weight": t(64, d),
        "embed.dst_embed.weight": t(64, d),
        "embed.promo_embed.weight": t(5, d),
        "embed.pad_embed": t(d),
        "embed.outcome_embed.weight": t(n_out, d),
        "final_norm.weight": t(d),
        "lm_head.weight": t(V, d),  # v1 stores (out, in)
    }
    for i in range(n_layers):
        state[f"layers.{i}.attn_norm.weight"] = t(d)
        state[f"layers.{i}.attn.wq.weight"] = t(d, d)
        state[f"layers.{i}.attn.wk.weight"] = t(d, d)
        state[f"layers.{i}.attn.wv.weight"] = t(d, d)
        state[f"layers.{i}.attn.wo.weight"] = t(d, d)
        state[f"layers.{i}.ffn_norm.weight"] = t(d)
        state[f"layers.{i}.ffn.w_gate.weight"] = t(d_ff, d)
        state[f"layers.{i}.ffn.w_up.weight"] = t(d_ff, d)
        state[f"layers.{i}.ffn.w_down.weight"] = t(d, d_ff)
    ckpt = tmp_path / "v1_ckpt"
    ckpt.mkdir()
    st_save_np(state, str(ckpt / "model.safetensors"))
    (ckpt / "config.json").write_text(json.dumps({
        "format_version": 1,
        "model_config": {
            "d_model": d, "n_layers": n_layers, "n_heads": 2, "d_ff": d_ff,
            "vocab_size": V, "max_seq_len": 64, "n_outcomes": n_out,
            "rope_base": 10000.0,
        },
    }))

    model, cfg = load_v1_factored_model(str(ckpt))
    assert isinstance(model, FactoredPAWNModel)
    assert cfg.factored_embeddings and not cfg.tie_embeddings

    # The whole point: a converted model must survive the v2 checkpoint
    # layer's cfg-driven save→load dispatch.
    out = save_model(model, tmp_path / "step_00000001")
    reloaded, _ = load_model(out)
    assert isinstance(reloaded, FactoredPAWNModel)
    np.testing.assert_array_equal(
        np.asarray(reloaded.lm_head), np.asarray(model.lm_head)
    )


def test_factored_embed_rejects_out_of_vocab_ids() -> None:
    """BOS=1980 (and NULL/reserved above it) are out-of-vocab for the
    factored model; `_embed` must raise loudly instead of silently
    clamping them onto the last outcome row (round-2 review, codex P2 —
    the systemic guard behind all the per-caller contract checks)."""
    from pawn.config import BOS_TOKEN

    m = _tiny_factored()
    bad = jnp.full((1, 4), BOS_TOKEN, dtype=jnp.int32)
    with pytest.raises(Exception, match="out-of-vocab"):
        jax.block_until_ready(m(bad))


def test_factored_probes_bare_moves_contract() -> None:
    """Probes on a factored model run under the bare-moves contract: the
    extractor rewrites slot 0 from BOS (out-of-vocab, would silently embed
    as the last outcome row) to a masked PAD, and rejects conditioning
    (round-2 review, test-risk: the branch was untested — removing the
    isinstance guard would corrupt every probe position silently)."""
    import chess_engine as engine

    from pawn.probes import run_layer_probes, side_to_move_labeler

    model = _tiny_factored()
    move_ids, game_lengths, _ = engine.generate_random_games(48, 24, 5)
    results = run_layer_probes(
        model, move_ids, game_lengths,
        n_classes=2, labeler=side_to_move_labeler,
        n_epochs=10, val_frac=0.25, key=0,
    )
    assert set(results.keys()) == set(range(TINY_FACTORED.n_layers + 1))
    best = max(r.accuracy for r in results.values())
    assert best > 0.6, f"side-to-move probe at chance: best={best}"

    # Conditioning is rejected for factored models (no control vocab rows).
    with pytest.raises(ValueError, match="bare-moves"):
        run_layer_probes(
            model, move_ids, game_lengths,
            n_classes=2, labeler=side_to_move_labeler,
            n_epochs=1, val_frac=0.25, key=0,
            conditioning=("outcome",),
        )


def test_factored_targets_never_out_of_vocab() -> None:
    batch = _v1_batch(batch_size=16, seq_len=128)
    sup = np.asarray(batch.loss_mask)
    tgt = np.asarray(batch.targets)
    assert tgt[sup].max() < NUM_ACTIONS  # supervised targets are moves only
    assert tgt.max() <= PAD_TOKEN  # nothing above PAD anywhere (no BOS)
