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
def test_pawn_jax_public_surface() -> None:
    """The documented JAX-side public surface is reachable from the
    ``pawn.jax`` namespace without any extra plumbing."""
    from pawn.adapters import LoRAConfig, LoRAModel, adapter_filter, init_lora_model
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
    from pawn.trainer import (
        Batch,
        TrainState,
        VariantSpec,
        make_lr_schedule,
        make_optimizer,
        make_scan_step,
        make_train_step,
    )

    # Touch the imports so they don't get tree-shaken by a linter.
    assert NUM_ACTIONS == 1968
    assert PAD_TOKEN == 1968
    assert MAX_SEQ_LEN == 512
    assert SUPERNET.d_model == 640
    assert TINY_SUPERNET.d_model == 192
    assert set(VARIANTS) == {"small", "base", "large"}
    assert set(TINY_VARIANTS) == {"small", "base", "large"}
    _ = (
        LoRAConfig, LoRAModel, adapter_filter, init_lora_model,
        ModelConfig, validate_nested, PAWNModel, init_model, sliced,
        Batch, TrainState, VariantSpec, make_lr_schedule,
        make_optimizer, make_scan_step, make_train_step,
    )


@pytest.mark.unit
def test_chess_engine_importable() -> None:
    """The Rust extension must build before the Python test suite runs."""
    import chess_engine  # type: ignore[import-not-found]

    assert hasattr(chess_engine, "generate_random_games")
    assert hasattr(chess_engine, "generate_clm_batch")
    assert hasattr(chess_engine, "export_move_vocabulary")
