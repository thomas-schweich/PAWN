"""Maia-style Elo-stratified accuracy on held-out Lichess games.

Slices the val corpus by Elo band and reports per-bin move-prediction
metrics. The trainer's per-band sampling is what gives this its
Maia-style flavour; this module just runs the metric computation.

The per-bin schema matches v1's ``eval_suite/lichess.py`` (plan §3
criterion 13: "schema matches v1") — each bin reports ``loss`` /
``perplexity`` / ``top1_accuracy`` / ``top5_accuracy`` /
``legal_move_rate`` alongside the game count, all computed in one
:func:`pawn.eval.compute_val_metrics` pass per bin.
"""

from __future__ import annotations

from dataclasses import dataclass

from pawn.corpus import Corpus
from pawn.eval import compute_val_metrics
from pawn.model import EffectiveCallable

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
    """Per-Elo-bin metrics (v1 ``eval_suite/lichess.py`` schema).

    ``accuracy`` is the top-1 move-prediction accuracy (kept under that
    name for back-compat with callers that only read top-1); ``top1`` is
    its alias. ``loss`` / ``perplexity`` / ``top5`` / ``legal_move_rate``
    restore the full v1 per-bin schema that the prior v2 cut had dropped.
    """

    bin: EloBin
    accuracy: float
    n_games: int
    loss: float
    perplexity: float
    top5: float
    legal_move_rate: float

    @property
    def top1(self) -> float:
        return self.accuracy


def default_elo_bins() -> tuple[EloBin, ...]:
    """Standard Maia-style 100-Elo-wide bins from 1100 to 2000."""
    return tuple(EloBin(lo, lo + 100) for lo in range(1100, 2000, 100))


def compute_elo_stratified_accuracy(
    model: EffectiveCallable,
    bins_corpora: dict[EloBin, Corpus],
    *,
    batch_size: int = 32,
    min_eval_ply: int = 0,
    compute_legal: bool = True,
) -> list[EloBinResult]:
    """For each bin's pre-filtered corpus, compute the full v1 metric set.

    The caller is responsible for pre-filtering the dataset into one
    Corpus per bin (via :func:`pawn.lichess_data.load_lichess_corpus`
    with the bin's `elo_min` / `elo_max`).

    Each bin runs one :func:`pawn.eval.compute_val_metrics` pass, so the
    result carries ``loss`` / ``perplexity`` / top-1 / top-5 /
    ``legal_move_rate`` — the v1 per-bin schema. ``min_eval_ply`` applies
    the MAIA opening-skip to the headline metrics; ``compute_legal=False``
    skips the engine legality replay (legal rate reports 0.0) for callers
    that only need the loss / accuracy scalars.
    """
    results: list[EloBinResult] = []
    for bin_, corpus in bins_corpora.items():
        vm = compute_val_metrics(
            model, corpus,
            batch_size=batch_size,
            min_eval_ply=min_eval_ply,
            compute_legal=compute_legal,
        )
        results.append(
            EloBinResult(
                bin=bin_,
                accuracy=vm.top1,
                n_games=corpus.n_games,
                loss=vm.val_loss,
                perplexity=vm.perplexity,
                top5=vm.top5,
                legal_move_rate=vm.legal_move_rate,
            )
        )
    return results
