"""Pin the public surface of the ``pawn`` package.

If any partition accidentally breaks a documented re-export, this test
fails loudly. Owned by the lead — workers should not edit.
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_pawn_reexports_stable() -> None:
    """``from pawn import CLMConfig, TrainingConfig, PAWNCLM`` must work
    when torch is installed.

    ``pawn/__init__.py`` gates the ``PAWNCLM`` re-export on torch being
    importable so JAX-only consumers can ``import pawn.jax.*`` without
    pulling torch (see the JAX migration). This test pins the stable
    surface under the torch-installed configuration that CI exercises;
    the torch-free configuration is covered by `test_pawn_jax_*` files
    importing pawn.jax.* without going through pawn.model.
    """
    pytest.importorskip("torch")
    import pawn

    assert hasattr(pawn, "CLMConfig")
    assert hasattr(pawn, "TrainingConfig")
    assert hasattr(pawn, "PAWNCLM")
    assert set(pawn.__all__) == {"CLMConfig", "TrainingConfig", "PAWNCLM"}


@pytest.mark.unit
def test_chess_engine_importable() -> None:
    """The Rust extension must build before the Python test suite runs."""
    import chess_engine  # type: ignore[import-not-found]

    assert hasattr(chess_engine, "generate_random_games")
    assert hasattr(chess_engine, "generate_clm_batch")
    assert hasattr(chess_engine, "export_move_vocabulary")
