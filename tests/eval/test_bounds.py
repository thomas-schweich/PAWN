"""Tests for pawn/eval_suite/bounds.py.

Covers compute_theoretical_bounds and format_bounds_report.
"""

from __future__ import annotations

from pathlib import Path

import math
import numpy as np
import pytest

from pawn.eval_suite.bounds import (
    compute_theoretical_bounds,
    format_bounds_report,
)
from pawn.eval_suite.corpus import generate_corpus, load_corpus


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory) -> Path:
    """Module-scoped tiny corpus (shared between bounds tests)."""
    d = tmp_path_factory.mktemp("bounds_corpus")
    generate_corpus(output_dir=d, n_games=8, max_ply=32, seed=42, batch_size=8)
    return d


@pytest.fixture(scope="module")
def bounds(corpus_dir: Path) -> dict:
    corpus = load_corpus(corpus_dir)
    return compute_theoretical_bounds(corpus)


# ---------------------------------------------------------------------------
# compute_theoretical_bounds
# ---------------------------------------------------------------------------


class TestComputeTheoreticalBounds:
    @staticmethod
    def _corpus_for_seed(seed: int):
        """Generate a small corpus with a given seed, returning the directory."""
        import tempfile
        d = Path(tempfile.mkdtemp())
        generate_corpus(output_dir=d, n_games=16, max_ply=32, seed=seed, batch_size=16)
        return d

    @pytest.mark.unit
    def test_returns_expected_keys(self, bounds: dict):
        for key in ("n_positions", "top1_accuracy", "top5_accuracy",
                    "loss_nats", "perplexity", "k_stats", "k_distribution",
                    "k_histogram", "phase_bounds", "check_bounds"):
            assert key in bounds

    @pytest.mark.unit
    def test_n_positions_positive(self, bounds: dict):
        assert bounds["n_positions"] > 0

    @pytest.mark.unit
    def test_top1_has_value_and_se(self, bounds: dict):
        assert "value" in bounds["top1_accuracy"]
        assert "se" in bounds["top1_accuracy"]
        # top-1 = E[1/K] should be in (0, 1]
        assert 0.0 < bounds["top1_accuracy"]["value"] <= 1.0
        assert bounds["top1_accuracy"]["se"] >= 0.0

    @pytest.mark.unit
    def test_top5_has_value_and_se(self, bounds: dict):
        assert "value" in bounds["top5_accuracy"]
        assert "se" in bounds["top5_accuracy"]
        assert 0.0 < bounds["top5_accuracy"]["value"] <= 1.0
        assert bounds["top5_accuracy"]["se"] >= 0.0

    @pytest.mark.unit
    def test_top5_dominates_top1(self, bounds: dict):
        # top-5 accuracy must be >= top-1 accuracy
        assert bounds["top5_accuracy"]["value"] >= bounds["top1_accuracy"]["value"]
        # With real chess positions, there are always positions with >1 legal move,
        # so top-5 should be strictly greater than top-1.
        assert bounds["top5_accuracy"]["value"] > bounds["top1_accuracy"]["value"], (
            "top-5 should strictly dominate top-1 when positions with >1 legal move exist"
        )

    @pytest.mark.unit
    def test_top5_dominates_top1_across_seeds(self):
        """Verify top5 >= top1 across multiple independently-seeded corpora."""
        for seed in (1, 17, 99):
            d = self._corpus_for_seed(seed)
            corpus = load_corpus(d)
            b = compute_theoretical_bounds(corpus)
            assert b["top5_accuracy"]["value"] > b["top1_accuracy"]["value"], (
                f"top5 should strictly dominate top1 for seed={seed}"
            )
            assert math.isfinite(b["top5_accuracy"]["value"])
            assert math.isfinite(b["top1_accuracy"]["value"])

    @pytest.mark.unit
    def test_loss_nats_positive(self, bounds: dict):
        assert bounds["loss_nats"]["value"] > 0.0
        assert bounds["loss_nats"]["se"] >= 0.0

    @pytest.mark.unit
    def test_perplexity_consistent_with_loss(self, bounds: dict):
        # perplexity = exp(loss)
        expected = math.exp(bounds["loss_nats"]["value"])
        assert bounds["perplexity"]["value"] == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_all_metrics_finite(self, bounds: dict):
        assert math.isfinite(bounds["top1_accuracy"]["value"])
        assert math.isfinite(bounds["top1_accuracy"]["se"])
        assert math.isfinite(bounds["top5_accuracy"]["value"])
        assert math.isfinite(bounds["top5_accuracy"]["se"])
        assert math.isfinite(bounds["loss_nats"]["value"])
        assert math.isfinite(bounds["perplexity"]["value"])

    @pytest.mark.unit
    def test_k_stats_keys(self, bounds: dict):
        for key in ("mean", "median", "std", "min", "max"):
            assert key in bounds["k_stats"]

    @pytest.mark.unit
    def test_k_stats_min_le_max(self, bounds: dict):
        assert bounds["k_stats"]["min"] <= bounds["k_stats"]["max"]

    @pytest.mark.unit
    def test_k_stats_mean_bounded(self, bounds: dict):
        stats = bounds["k_stats"]
        assert stats["min"] <= stats["mean"] <= stats["max"]

    @pytest.mark.unit
    def test_k_distribution_is_dict(self, bounds: dict):
        assert isinstance(bounds["k_distribution"], dict)
        assert len(bounds["k_distribution"]) > 0

    @pytest.mark.unit
    def test_k_histogram_has_lists(self, bounds: dict):
        hist = bounds["k_histogram"]
        assert "values" in hist and "counts" in hist and "total" in hist
        assert isinstance(hist["values"], list)
        assert isinstance(hist["counts"], list)
        assert len(hist["values"]) == len(hist["counts"])

    @pytest.mark.unit
    def test_phase_bounds_has_phase_entries(self, bounds: dict):
        # All positions in this corpus fall into ply_1_20/ply_21_80
        assert any(k in bounds["phase_bounds"] for k in
                   ("ply_1_20", "ply_21_80", "ply_81_150", "ply_150_plus"))

    @pytest.mark.unit
    def test_check_bounds_keys(self, bounds: dict):
        # Either or both of in_check/not_in_check may appear
        for label in bounds["check_bounds"]:
            assert label in ("in_check", "not_in_check")
            inner = bounds["check_bounds"][label]
            assert "mean_k" in inner
            assert "e_1_over_k" in inner
            assert "frequency" in inner


# ---------------------------------------------------------------------------
# format_bounds_report
# ---------------------------------------------------------------------------


class TestFormatBoundsReport:
    @pytest.mark.unit
    def test_returns_string(self, bounds: dict):
        report = format_bounds_report(bounds, seed=42, n_games=8)
        assert isinstance(report, str)
        assert len(report) > 0

    @pytest.mark.unit
    def test_contains_seed_and_n_games(self, bounds: dict):
        report = format_bounds_report(bounds, seed=1234, n_games=8)
        assert "1234" in report
        # n_games appears in the header
        assert "8" in report

    @pytest.mark.unit
    def test_contains_section_headers(self, bounds: dict):
        report = format_bounds_report(bounds, seed=42, n_games=8)
        assert "Theoretical Bounds" in report
        assert "Max top-1 accuracy" in report
        assert "Max top-5 accuracy" in report
        assert "Min loss (nats)" in report
        assert "Min perplexity" in report

    @pytest.mark.unit
    def test_contains_k_stats(self, bounds: dict):
        report = format_bounds_report(bounds, seed=42, n_games=8)
        assert "Mean K" in report
        assert "Median K" in report
        assert "Max K" in report

    @pytest.mark.unit
    def test_contains_phase_and_check_sections(self, bounds: dict):
        report = format_bounds_report(bounds, seed=42, n_games=8)
        assert "by game phase" in report
        assert "by check status" in report

    @pytest.mark.unit
    def test_report_has_k_distribution_entries(self, bounds: dict):
        report = format_bounds_report(bounds, seed=42, n_games=8)
        # Each reported K value is formatted as "K=<value>:"
        assert "K=" in report

    @pytest.mark.unit
    def test_report_truncates_if_many_k_values(self, bounds: dict):
        # Synthesize bounds with 40 distinct K values; report truncates >30
        synthetic = dict(bounds)
        synthetic["k_distribution"] = {i: 1 for i in range(40)}
        synthetic["n_positions"] = 40
        report = format_bounds_report(synthetic, seed=0, n_games=1)
        assert "distinct values total" in report
