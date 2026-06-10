"""Parity #8: port from deleted v1 tests/eval/test_bounds.py.

`pawn.eval_suite.bounds.compute_theoretical_bounds` + ``format_bounds_report``
ship in v2 but had no test coverage. These tests port the v1 behavioral
assertions (keys present, n_positions positive, top1 in [0, 1], format
report contains expected sections) so a future refactor of the
accumulator can't silently regress the public output shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pawn.eval_suite.bounds import compute_theoretical_bounds, format_bounds_report
from pawn.eval_suite.corpus import generate_corpus, load_corpus


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped tiny corpus (shared between bounds tests) — the v1
    fixture pattern. Generate 8 games × 32 max_ply for a tight,
    deterministic fixture that doesn't multiply the test runtime."""
    d = tmp_path_factory.mktemp("bounds_corpus")
    generate_corpus(output_dir=d, n_games=8, max_ply=32, seed=42, batch_size=8)
    return d


@pytest.fixture(scope="module")
def bounds(corpus_dir: Path) -> dict:
    corpus = load_corpus(corpus_dir)
    return compute_theoretical_bounds(corpus)


def test_returns_expected_keys(bounds: dict) -> None:
    for key in (
        "n_positions", "top1_accuracy", "top5_accuracy",
        "loss_nats", "perplexity", "k_stats", "k_distribution",
        "k_histogram", "phase_bounds", "check_bounds",
    ):
        assert key in bounds, f"compute_theoretical_bounds missing key: {key!r}"


def test_n_positions_positive(bounds: dict) -> None:
    assert bounds["n_positions"] > 0


def test_top1_accuracy_in_unit_interval(bounds: dict) -> None:
    val = float(bounds["top1_accuracy"]["value"])
    se = float(bounds["top1_accuracy"]["se"])
    assert 0.0 <= val <= 1.0
    assert se >= 0.0


def test_top5_geq_top1(bounds: dict) -> None:
    """Top-5 accuracy must dominate top-1 by definition."""
    top1 = float(bounds["top1_accuracy"]["value"])
    top5 = float(bounds["top5_accuracy"]["value"])
    assert top5 >= top1 - 1e-9


def test_perplexity_matches_exp_loss(bounds: dict) -> None:
    """``perplexity = exp(loss_nats)`` is the load-bearing identity."""
    import math
    loss = float(bounds["loss_nats"]["value"])
    perp = float(bounds["perplexity"]["value"])
    assert math.isclose(perp, math.exp(loss), rel_tol=1e-9)


def test_k_stats_well_formed(bounds: dict) -> None:
    k = bounds["k_stats"]
    assert k["min"] >= 1  # non-terminal positions always have ≥1 legal move
    assert k["max"] >= k["min"]
    assert k["mean"] >= k["min"]


def test_format_bounds_report_contains_expected_sections(bounds: dict) -> None:
    out = format_bounds_report(bounds, seed=42, n_games=8)
    assert "Theoretical Bounds for Random Chess Next-Token Prediction" in out
    assert "Bounds (95% CI):" in out
    assert "Max top-1 accuracy:" in out
    assert "Min perplexity:" in out
    assert "Legal move count statistics:" in out
    assert "Distribution of K (most common):" in out
