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
def test_pawn_jax_trainer_surface_lands_in_S6() -> None:
    """Placeholder pin for the JAX trainer's public symbols. The
    actual import-and-touch test goes live when S6 lands
    ``pawn.trainer`` with ``Batch`` / ``TrainState`` / ``VariantSpec``
    / ``make_{lr_schedule,optimizer,scan_step,train_step}``."""
    pytest.skip("pawn.trainer JAX surface lands in S6 — see docs/jax-migration.md §13")


@pytest.mark.unit
def test_pawn_jax_adapters_surface_lands_in_S7() -> None:
    """Placeholder pin for the JAX adapters' public symbols. The
    actual import-and-touch test goes live when S7 lands
    ``pawn.adapters`` with ``LoRAConfig`` / ``LoRAModel`` /
    ``adapter_filter`` / ``init_lora_model`` (and the other 7
    strategies)."""
    pytest.skip("pawn.adapters JAX surface lands in S7 — see docs/jax-migration.md §13")


@pytest.mark.unit
def test_chess_engine_importable() -> None:
    """The Rust extension must build before the Python test suite runs."""
    import chess_engine  # type: ignore[import-not-found]

    assert hasattr(chess_engine, "generate_random_games")
    assert hasattr(chess_engine, "generate_clm_batch")
    assert hasattr(chess_engine, "export_move_vocabulary")
