"""Pin the post-swap public API surface.

After the JAX migration lands, third parties + downstream tooling
rely on a specific set of importable names. This test pins that
surface — a removal or rename surfaces here loudly rather than
silently breaking external integrations.

Per plan §10 S15: "tests/test_public_api.py pins the post-swap
public surface."
"""

from __future__ import annotations

import pytest


def test_pawn_init_exposes_v2_top_level_names() -> None:
    """The v2 public surface re-exports `ModelConfig`, `PAWNModel`, and
    the `RunConfig` discriminated-union members from `pawn`."""
    import pawn

    assert pawn.__doc__ is not None
    # v2 names — these must all be importable directly from `pawn`.
    from pawn import (
        AdapterConfig,
        BaseRunConfig,
        ModelConfig,
        PAWNModel,
        PretrainConfig,
        RunConfig,
        SpecializedCLMConfig,
    )

    # Sanity: each is a real type, not None / stub.
    for cls in (
        ModelConfig,
        PAWNModel,
        BaseRunConfig,
        PretrainConfig,
        AdapterConfig,
        SpecializedCLMConfig,
    ):
        assert cls is not None


def test_pawn_v1_names_raise_migration_error() -> None:
    """v1 import sites (`from pawn import CLMConfig`) must surface a
    precise migration error pointing at the v2 replacement, not a
    generic "no attribute" or, worse, silent success on a stale
    type."""
    import pawn

    for old_name in ("CLMConfig", "TrainingConfig", "PAWNCLM"):
        with pytest.raises(ImportError, match=old_name):
            getattr(pawn, old_name)


def test_config_public_surface() -> None:
    """`pawn.config` exports the architecture invariants + vocab
    constants downstream code reads."""
    from pawn import config

    expected = {
        "NUM_ACTIONS", "PAD_TOKEN", "OUTCOME_TOKEN_BASE",
        "N_PRETRAINING_OUTCOMES", "N_TOTAL_OUTCOMES", "VOCAB_SIZE",
        "WHITE_CHECKMATES", "BLACK_CHECKMATES", "STALEMATE",
        "DRAW_BY_RULE", "PLY_LIMIT",
        "WHITE_RESIGNS", "BLACK_RESIGNS", "DRAW_BY_AGREEMENT",
        "WHITE_WINS_ON_TIME", "BLACK_WINS_ON_TIME", "DRAW_BY_TIME",
        "MAX_SEQ_LEN", "ROPE_BASE", "HEAD_DIM",
        "ModelConfig", "NestingError",
        "SUPERNET", "VARIANTS", "TINY_SUPERNET", "TINY_VARIANTS",
        "validate_nested",
    }
    assert expected <= set(config.__all__)


def test_model_public_surface() -> None:
    from pawn import model

    assert {"PAWNModel", "TransformerLayer", "SAVED_FIELDS", "init_model", "sliced"} <= set(model.__all__)
    # Post un-factor/tie (spec §3): the canonical declaration order is the
    # 12-field untied superset (factored embed_src/dst/promo/pad/outcome are
    # gone, replaced by a single tied `embed_tokens`). The per-`tie_embeddings`
    # save schema drops `lm_head` when tied.
    assert len(model.SAVED_FIELDS) == 12
    assert "lm_head" in model.SAVED_FIELDS
    assert model.saved_fields(tie_embeddings=False) == model.SAVED_FIELDS
    assert model.saved_fields(tie_embeddings=True) == tuple(
        name for name in model.SAVED_FIELDS if name != "lm_head"
    )


def test_checkpoint_public_surface() -> None:
    from pawn import checkpoint

    assert {
        "save_model", "load_model", "load_model_config",
        "CheckpointIntegrityError", "IncompleteCheckpointError",
        "MODEL_FILE", "CONFIG_FILE", "OPTIMIZER_FILE", "TRAINING_STATE_FILE",
        "CHECKPOINT_FORMAT_VERSION",
    } <= set(checkpoint.__all__)


def test_run_config_public_surface() -> None:
    from pawn import run_config

    assert {
        "BaseRunConfig", "PretrainConfig", "AdapterConfig",
        "SpecializedCLMConfig", "RunConfig",
    } <= set(run_config.__all__)


def test_corpus_public_surface() -> None:
    from pawn import corpus

    assert {"Corpus", "generate_corpus", "pack_corpus"} <= set(corpus.__all__)


def test_lichess_data_public_surface() -> None:
    from pawn import lichess_data

    assert {"load_lichess_corpus", "make_epoch_schedule"} <= set(lichess_data.__all__)


def test_trainer_public_surface() -> None:
    from pawn import trainer

    assert {
        "Batch", "TrainState", "VariantSpec",
        "cross_entropy_loss", "supernet_joint_loss",
        "make_lr_schedule", "make_optimizer",
        "make_train_step", "make_scan_step", "slice_batch",
    } <= set(trainer.__all__)


def test_adapter_trainer_public_surface() -> None:
    from pawn import adapter_trainer

    assert {
        "AdapterTrainState", "STRATEGIES",
        "dispatch_init", "dispatch_apply", "dispatch_filter",
        "make_adapter_train_step", "make_adapter_scan_step", "forward_eval",
    } <= set(adapter_trainer.__all__)


def test_adapters_strategies_are_complete() -> None:
    """All 10 documented adapter strategy keys."""
    from pawn.adapters import STRATEGIES

    assert set(STRATEGIES) == {
        "lora", "film", "bottleneck", "hybrid", "sparse",
        "rosa", "rosa-retro-sparse", "rosa-retro-bottleneck",
        "unfreeze", "specialized_clm",
    }


def test_eval_public_surface() -> None:
    from pawn import eval as pawn_eval

    assert {
        "PhaseBoundaries", "AccuracyResult",
        "compute_move_accuracy", "compute_per_phase_accuracy",
    } <= set(pawn_eval.__all__)


def test_generation_public_surface_has_five_diagnostics() -> None:
    from pawn import generation

    assert set(generation.DIAGNOSTIC_NAMES) == {
        "outcome_signal_test", "prefix_continuation_test",
        "poisoned_prefix_test", "impossible_task_test",
        "improbable_task_test",
    }


def test_probes_public_surface() -> None:
    from pawn import probes

    assert {"ProbeConfig", "ProbeResult", "fit_probe"} <= set(probes.__all__)


def test_lichess_eval_public_surface() -> None:
    from pawn import lichess_eval

    assert {
        "EloBin", "EloBinResult",
        "default_elo_bins", "compute_elo_stratified_accuracy",
    } <= set(lichess_eval.__all__)


def test_lifecycle_public_surface() -> None:
    from pawn import lifecycle

    assert {
        "HFPushTracker", "push_checkpoint_async",
        "install_sigterm_handler", "drain_push_queue",
        "load_resume_state",
    } <= set(lifecycle.__all__)


def test_sweep_public_surface() -> None:
    from pawn import sweep

    assert {
        "AdapterObjective", "InProcessRoSAObjective",
        "STRATEGY_SUGGESTERS",
    } <= set(sweep.__all__)


def test_lab_public_surface() -> None:
    from pawn import lab

    assert {"lab_launch", "lab_schema", "validate_config"} <= set(lab.__all__)


def test_wandb_utils_public_surface() -> None:
    from pawn import wandb_utils

    assert {"init_wandb", "log_metrics", "finish_wandb"} <= set(wandb_utils.__all__)


def test_logging_public_surface() -> None:
    from pawn import logging as pawn_logging

    assert {"MetricsLogger", "get_git_info", "random_slug"} <= set(pawn_logging.__all__)


def test_no_cotrain_run_type_in_run_config() -> None:
    """`pawn.cotrain` was GONE BY DESIGN per plan §6 — the supernet's
    joint loss in `pawn.trainer.supernet_joint_loss` replaces it. The
    discriminated-union RunConfig should not accept run_type='cotrain'."""
    from pydantic import TypeAdapter, ValidationError

    from pawn.run_config import RunConfig

    adapter = TypeAdapter(RunConfig)
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {"run_type": "cotrain", "local_checkpoints": True}
        )
