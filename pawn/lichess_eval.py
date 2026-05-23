"""Maia-style Elo-stratified accuracy on held-out Lichess games.

Slices the val corpus by Elo band and reports per-bin move-prediction
accuracy. The trainer's per-band sampling is what gives this its
Maia-style flavour; this module just runs the accuracy computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from pawn.corpus import Corpus
from pawn.eval import compute_move_accuracy
from pawn.model import PAWNModel

__all__ = [
    "EloBin",
    "EloBinResult",
    "default_elo_bins",
    "compute_elo_stratified_accuracy",
]


@dataclass(frozen=True)
class EloBin:
    """Half-open Elo range `[lo, hi)`."""

    lo: int
    hi: int

    @property
    def label(self) -> str:
        return f"{self.lo}-{self.hi - 1}"


@dataclass(frozen=True)
class EloBinResult:
    bin: EloBin
    accuracy: float
    n_games: int


def default_elo_bins() -> tuple[EloBin, ...]:
    """Standard Maia-style 100-Elo-wide bins from 1100 to 2000."""
    return tuple(EloBin(lo, lo + 100) for lo in range(1100, 2000, 100))


def compute_elo_stratified_accuracy(
    model: PAWNModel,
    bins_corpora: dict[EloBin, Corpus],
    *,
    batch_size: int = 32,
) -> list[EloBinResult]:
    """For each bin's pre-filtered corpus, compute move accuracy.

    The caller is responsible for pre-filtering the dataset into one
    Corpus per bin (via :func:`pawn.lichess_data.load_lichess_corpus`
    with the bin's `elo_min` / `elo_max`).
    """
    results: list[EloBinResult] = []
    for bin_, corpus in bins_corpora.items():
        acc = compute_move_accuracy(model, corpus, batch_size=batch_size)
        results.append(
            EloBinResult(bin=bin_, accuracy=acc, n_games=corpus.n_games)
        )
    return results
