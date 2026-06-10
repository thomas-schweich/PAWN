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
    # explicit variant subsets are not applicable.
    with pytest.raises(ValueError, match="variants"):
        PretrainConfig(
            run_type="pretrain", total_steps=10, arch="factored-v1",
            local_checkpoints=True, variants=("small",),
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


def test_factored_targets_never_out_of_vocab() -> None:
    batch = _v1_batch(batch_size=16, seq_len=128)
    sup = np.asarray(batch.loss_mask)
    tgt = np.asarray(batch.targets)
    assert tgt[sup].max() < NUM_ACTIONS  # supervised targets are moves only
    assert tgt.max() <= PAD_TOKEN  # nothing above PAD anywhere (no BOS)
