"""Pin the public surface of the ``pawn`` package.

If any partition accidentally breaks a documented re-export, this test
fails loudly. Owned by the lead — workers should not edit.
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_pawn_reexports_stable() -> None:
    """``from pawn import CLMConfig, TrainingConfig, PAWNCLM`` must work."""
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
