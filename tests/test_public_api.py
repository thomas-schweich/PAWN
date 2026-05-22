"""Pin the public surface of the ``pawn`` package post-JAX-migration.

After the PyTorch removal in Phase 4 the package's public surface is
JAX-only. ``pawn`` re-exports nothing at the top level — JAX consumers
import from ``pawn.*`` directly; external PyTorch users use the
thin loader at ``pawn.torch_loader.load_pawn``.

Owned by the lead — workers should not edit.
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_pawn_top_level_has_no_torch_reexports() -> None:
    """``pawn`` should not surface any of the legacy torch-only symbols
    (``CLMConfig``, ``TrainingConfig``, ``PAWNCLM``). A regression that
    re-introduced one would silently revive the dual-framework era."""
    import pawn

    for legacy in ("CLMConfig", "TrainingConfig", "PAWNCLM"):
        assert not hasattr(pawn, legacy), (
            f"pawn.{legacy} re-introduced — the post-Phase-4 package "
            f"should not surface legacy torch symbols at the top level"
        )


@pytest.mark.unit
def test_pawn_jax_core_public_surface() -> None:
    """The JAX core surface (S3 of the migration) is reachable. The
    adapters / trainer surfaces are pinned by separate tests once
    S6 (trainer) and S7 (adapters) land — keep them out of this
    pin until the modules exist, otherwise the migration's S3
    section head ImportErrors at collection time."""
    from pawn.config import (
        MAX_SEQ_LEN,
        NUM_ACTIONS,
        PAD_TOKEN,
        SUPERNET,
        TINY_SUPERNET,
        TINY_VARIANTS,
        VARIANTS,
        ModelConfig,
        validate_nested,
    )
    from pawn.model import PAWNModel, init_model, sliced

    # Touch the imports so they don't get tree-shaken by a linter.
    assert NUM_ACTIONS == 1968
    assert PAD_TOKEN == 1968
    assert MAX_SEQ_LEN == 512
    assert SUPERNET.d_model == 640
    assert TINY_SUPERNET.d_model == 192
    assert set(VARIANTS) == {"small", "base", "large"}
    assert set(TINY_VARIANTS) == {"small", "base", "large"}
    _ = (
        ModelConfig, validate_nested, PAWNModel, init_model, sliced,
    )


@pytest.mark.unit
def test_pawn_jax_trainer_public_surface() -> None:
    """Pin the JAX trainer's public symbols (S6)."""
    from pawn.trainer import (
        Batch,
        VariantSpec,
        make_lr_schedule,
        make_optimizer,
        make_scan_step,
        make_train_step,
    )

    _ = (
        Batch,
        VariantSpec,
        make_lr_schedule,
        make_optimizer,
        make_scan_step,
        make_train_step,
    )


@pytest.mark.unit
def test_pawn_jax_adapters_public_surface() -> None:
    """Pin the JAX adapters' public symbols (S7)."""
    from pawn.adapters import (
        BottleneckConfig,
        FiLMConfig,
        HybridConfig,
        LoRAConfig,
        RoSAConfig,
        SparseConfig,
        SpecializedCLMConfig,
        UnfreezeConfig,
        adapter_filter,
        init_bottleneck_model,
        init_film_model,
        init_hybrid_model,
        init_lora_model,
        init_rosa_model,
        init_sparse_model,
        init_specialized_clm,
        init_unfreeze_model,
    )

    _ = (
        BottleneckConfig,
        FiLMConfig,
        HybridConfig,
        LoRAConfig,
        RoSAConfig,
        SparseConfig,
        SpecializedCLMConfig,
        UnfreezeConfig,
        adapter_filter,
        init_bottleneck_model,
        init_film_model,
        init_hybrid_model,
        init_lora_model,
        init_rosa_model,
        init_sparse_model,
        init_specialized_clm,
        init_unfreeze_model,
    )


@pytest.mark.unit
def test_pawn_run_config_public_surface() -> None:
    """Pin the pydantic config surface (S4)."""
    from pawn.run_config import (
        AdapterConfig,
        BaseRunConfig,
        PretrainConfig,
        RunConfig,
        SpecializedCLMConfig,
    )

    _ = (
        AdapterConfig,
        BaseRunConfig,
        PretrainConfig,
        RunConfig,
        SpecializedCLMConfig,
    )


@pytest.mark.unit
def test_pawn_logging_public_surface() -> None:
    """Pin the MetricsLogger surface (S4)."""
    from pawn.logging import MetricsLogger, get_git_info, random_slug

    _ = (MetricsLogger, get_git_info, random_slug)


@pytest.mark.unit
def test_pawn_sweep_public_surface() -> None:
    """Pin the Optuna sweep surface (S10)."""
    from pawn.sweep import AdapterObjective, SUGGEST_FNS, create_study

    _ = (AdapterObjective, SUGGEST_FNS, create_study)


@pytest.mark.unit
def test_pawn_lichess_data_public_surface() -> None:
    """Pin the Lichess adapter-training data surface (S17). PAWN is a
    finetuning testbed — the Elo-stratified Lichess path is the
    realistic adapter task, so its entry points stay pinned."""
    from pawn.corpus import pack_corpus
    from pawn.lichess_data import load_lichess_corpus, make_epoch_schedule

    _ = (pack_corpus, load_lichess_corpus, make_epoch_schedule)


@pytest.mark.unit
def test_chess_engine_importable() -> None:
    """The Rust extension must build before the Python test suite runs."""
    import chess_engine  # type: ignore[import-not-found]

    assert hasattr(chess_engine, "generate_random_games")
    assert hasattr(chess_engine, "generate_clm_batch")
    assert hasattr(chess_engine, "export_move_vocabulary")
