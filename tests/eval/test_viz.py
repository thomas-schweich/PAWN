"""Tests for pawn/eval_suite/viz.py.

Smoke tests only — verify each plotting function returns a Figure and
doesn't crash on representative synthetic inputs. Uses matplotlib Agg
backend to avoid display requirement.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from pawn.eval_suite import viz
from pawn.eval_suite.viz import (
    GRID_PAWN_BASELINES,
    plot_diagnostic_results,
    plot_game_length_distribution,
    plot_k_by_phase,
    plot_legal_move_distribution,
    plot_lichess_comparison,
    plot_outcome_distributions,
    plot_outcome_rates,
    plot_outcome_signal_results,
    plot_prefix_continuation,
    plot_prefix_histogram,
    plot_probe_comparison,
    plot_probe_heatmap,
    plot_probe_layer_profile,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Sample data builders
# ---------------------------------------------------------------------------


def _sample_stats():
    return {
        "n_games": 100,
        "total_positions": 1000,
        "game_length": {
            "mean": 50.0,
            "median": 48,
            "std": 10.0,
            "min": 10,
            "max": 100,
            "histogram_counts": [10, 20, 30, 25, 15],
            "histogram_edges": [0, 20, 40, 60, 80, 100],
        },
        "outcome_rates": {
            "WHITE_CHECKMATES": 0.1,
            "BLACK_CHECKMATES": 0.1,
            "STALEMATE": 0.05,
            "DRAW_BY_RULE": 0.2,
            "PLY_LIMIT": 0.55,
        },
    }


def _sample_bounds():
    return {
        "n_positions": 1000,
        "top1_accuracy": {"value": 0.04, "se": 0.001},
        "top5_accuracy": {"value": 0.18, "se": 0.002},
        "loss_nats": {"value": 3.5, "se": 0.01},
        "perplexity": {"value": 33.0, "se": 0.3},
        "k_stats": {"mean": 25.0, "median": 25, "std": 5.0, "min": 1, "max": 80},
        "k_histogram": {
            "values": [1, 10, 20, 30, 50, 80],
            "counts": [5, 100, 300, 400, 150, 45],
            "total": 1000,
        },
        "k_distribution": {1: 5, 10: 100, 20: 300, 30: 400, 50: 150, 80: 45},
        "phase_bounds": {
            "ply_1_20": {"mean_k": 24.0, "e_1_over_k": 0.042, "e_ln_k": 3.1, "n_positions": 100},
            "ply_21_80": {"mean_k": 26.0, "e_1_over_k": 0.040, "e_ln_k": 3.2, "n_positions": 500},
        },
        "check_bounds": {
            "in_check": {"mean_k": 5.0, "e_1_over_k": 0.2, "frequency": 0.05},
            "not_in_check": {"mean_k": 26.5, "e_1_over_k": 0.038, "frequency": 0.95},
        },
    }


def _sample_probe_results():
    return {
        "piece_type": {
            "embed": {"best_accuracy": 0.5, "accuracy": 0.49, "loss": 0.7},
            "layer_0": {"best_accuracy": 0.75, "accuracy": 0.73, "loss": 0.4},
            "layer_1": {"best_accuracy": 0.9, "accuracy": 0.88, "loss": 0.2},
        },
        "side_to_move": {
            "embed": {"best_accuracy": 0.8, "accuracy": 0.78, "loss": 0.3},
            "layer_0": {"best_accuracy": 0.95, "accuracy": 0.94, "loss": 0.1},
            "layer_1": {"best_accuracy": 0.99, "accuracy": 0.99, "loss": 0.02},
        },
    }


def _sample_sanity():
    return {
        "duplicates": 0,
        "max_prefix_moves": 5,
        "prefix_length_histogram": {0: 100, 1: 50, 2: 10, 3: 3, 4: 1, 5: 1},
        "duplicate_pairs": [],
    }


def _sample_signal_results():
    base = {"outcome_match_rate": 0.5, "mean_game_length": 80.0,
            "outcome_distribution": {"WHITE_CHECKMATES": 0.5, "PLY_LIMIT": 0.5}}
    return {
        "WHITE_CHECKMATES": {"masked": base, "unmasked": base},
        "BLACK_CHECKMATES": {"masked": base, "unmasked": base},
        "PLY_LIMIT": {"masked": base, "unmasked": base},
    }


def _sample_diag_results():
    return {
        "in_check": {
            "mean_legal_rate": 0.8, "mean_pad_prob": 0.02, "mean_entropy": 3.0,
        },
        "double_check": {
            "mean_legal_rate": 0.7, "mean_pad_prob": 0.05, "mean_entropy": 2.5,
        },
        "checkmate": {
            "mean_legal_rate": 0.0, "mean_pad_prob": 0.8, "mean_entropy": 1.0,
        },
    }


def _sample_lichess_results():
    return {
        "elo_1000_1400": {
            "loss": 4.2, "top1_accuracy": 0.05, "top5_accuracy": 0.2,
            "legal_move_rate": 0.6, "perplexity": 60.0, "elo_range": (1000, 1400),
            "n_games": 10, "n_tokens": 500,
        },
        "elo_1400_1800": {
            "loss": 3.8, "top1_accuracy": 0.08, "top5_accuracy": 0.3,
            "legal_move_rate": 0.7, "perplexity": 45.0, "elo_range": (1400, 1800),
            "n_games": 10, "n_tokens": 500,
        },
    }


def _sample_prefix_results():
    bucket = {"outcome_match_rate": 0.5, "mean_game_length": 80.0,
              "outcome_distribution": {}}
    return {
        "WHITE_CHECKMATES": {
            "pct_50": {"WHITE_CHECKMATES": bucket, "BLACK_CHECKMATES": bucket},
            "ply_100": {"WHITE_CHECKMATES": bucket, "BLACK_CHECKMATES": bucket},
        },
    }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    @pytest.mark.unit
    def test_grid_pawn_baselines_is_dict(self):
        assert isinstance(GRID_PAWN_BASELINES, dict)
        for name, val in GRID_PAWN_BASELINES.items():
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0

    @pytest.mark.unit
    def test_baselines_contain_expected_probes(self):
        for key in ("piece_type", "side_to_move", "is_check",
                    "castling_rights", "ep_square"):
            assert key in GRID_PAWN_BASELINES

    @pytest.mark.unit
    def test_figsize_constants(self):
        assert isinstance(viz.FIGSIZE, tuple) and len(viz.FIGSIZE) == 2
        assert isinstance(viz.FIGSIZE_WIDE, tuple) and len(viz.FIGSIZE_WIDE) == 2
        assert isinstance(viz.FIGSIZE_TALL, tuple) and len(viz.FIGSIZE_TALL) == 2


# ---------------------------------------------------------------------------
# Corpus + bounds plots
# ---------------------------------------------------------------------------


class TestPlotGameLengthDistribution:
    @pytest.mark.unit
    def test_returns_figure(self):
        fig = plot_game_length_distribution(_sample_stats())
        assert isinstance(fig, plt.Figure)

    @pytest.mark.unit
    def test_with_provided_ax(self):
        _fig, ax = plt.subplots()
        fig = plot_game_length_distribution(_sample_stats(), ax=ax)
        assert isinstance(fig, plt.Figure)


class TestPlotLegalMoveDistribution:
    @pytest.mark.unit
    def test_returns_figure(self):
        fig = plot_legal_move_distribution(_sample_bounds())
        assert isinstance(fig, plt.Figure)


class TestPlotOutcomeRates:
    @pytest.mark.unit
    def test_returns_figure(self):
        fig = plot_outcome_rates(_sample_stats())
        assert isinstance(fig, plt.Figure)


class TestPlotKByPhase:
    @pytest.mark.unit
    def test_returns_figure(self):
        fig = plot_k_by_phase(_sample_bounds())
        assert isinstance(fig, plt.Figure)


class TestPlotPrefixHistogram:
    @pytest.mark.unit
    def test_returns_figure(self):
        fig = plot_prefix_histogram(_sample_sanity())
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# Probe plots
# ---------------------------------------------------------------------------


class TestPlotProbeHeatmap:
    @pytest.mark.unit
    def test_returns_figure(self):
        fig = plot_probe_heatmap(_sample_probe_results())
        assert isinstance(fig, plt.Figure)

    @pytest.mark.unit
    def test_accepts_custom_title(self):
        fig = plot_probe_heatmap(_sample_probe_results(), title="Custom")
        assert isinstance(fig, plt.Figure)

    @pytest.mark.unit
    def test_accepts_custom_metric_key(self):
        fig = plot_probe_heatmap(_sample_probe_results(), metric_key="accuracy")
        assert isinstance(fig, plt.Figure)


class TestPlotProbeComparison:
    @pytest.mark.unit
    def test_returns_figure_with_default_baselines(self):
        # Only side_to_move present in sample, baselines have more
        results = {
            "side_to_move": {"layer_1": {"best_accuracy": 0.99}},
            "is_check": {"layer_1": {"best_accuracy": 0.95}},
        }
        fig = plot_probe_comparison(results)
        assert isinstance(fig, plt.Figure)

    @pytest.mark.unit
    def test_returns_figure_with_custom_baselines(self):
        results = {"my_probe": {"layer_1": {"best_accuracy": 0.9}}}
        fig = plot_probe_comparison(results, pawn_baselines={"my_probe": 0.8})
        assert isinstance(fig, plt.Figure)


class TestPlotProbeLayerProfile:
    @pytest.mark.unit
    def test_returns_figure(self):
        fig = plot_probe_layer_profile(_sample_probe_results(), "piece_type")
        assert isinstance(fig, plt.Figure)

    @pytest.mark.unit
    def test_works_without_baseline(self):
        # Probe name not in GRID_PAWN_BASELINES
        results = {"novel_probe": {"layer_0": {"best_accuracy": 0.5}}}
        fig = plot_probe_layer_profile(results, "novel_probe")
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# Signal test plots
# ---------------------------------------------------------------------------


class TestPlotOutcomeSignalResults:
    @pytest.mark.unit
    def test_returns_figure(self):
        base_rates = {"WHITE_CHECKMATES": 0.1, "BLACK_CHECKMATES": 0.1,
                      "PLY_LIMIT": 0.5}
        fig = plot_outcome_signal_results(_sample_signal_results(), base_rates)
        assert isinstance(fig, plt.Figure)


class TestPlotOutcomeDistributions:
    @pytest.mark.unit
    def test_returns_figure_masked(self):
        fig = plot_outcome_distributions(_sample_signal_results(), condition="masked")
        assert isinstance(fig, plt.Figure)

    @pytest.mark.unit
    def test_returns_figure_single_outcome(self):
        single = {"WHITE_CHECKMATES": _sample_signal_results()["WHITE_CHECKMATES"]}
        fig = plot_outcome_distributions(single, condition="masked")
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------


class TestPlotDiagnosticResults:
    @pytest.mark.unit
    def test_returns_figure(self):
        fig = plot_diagnostic_results(_sample_diag_results())
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# Lichess plot
# ---------------------------------------------------------------------------


class TestPlotLichessComparison:
    @pytest.mark.unit
    def test_returns_figure(self):
        fig = plot_lichess_comparison(_sample_lichess_results())
        assert isinstance(fig, plt.Figure)

    @pytest.mark.unit
    def test_with_random_metrics(self):
        fig = plot_lichess_comparison(
            _sample_lichess_results(),
            random_metrics={"loss": 3.5, "top1_accuracy": 0.04,
                            "legal_move_rate": 0.5},
        )
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# Prefix continuation plot
# ---------------------------------------------------------------------------


class TestPlotPrefixContinuation:
    @pytest.mark.unit
    def test_returns_figure(self):
        fig = plot_prefix_continuation(
            _sample_prefix_results(),
            base_rates={"WHITE_CHECKMATES": 0.1},
        )
        assert isinstance(fig, plt.Figure)

    @pytest.mark.unit
    def test_empty_results_shows_no_data(self):
        fig = plot_prefix_continuation({}, base_rates={})
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# Module-level smoke tests: imports & callables
# ---------------------------------------------------------------------------


class TestViewModuleCallables:
    @pytest.mark.unit
    def test_all_documented_functions_callable(self):
        for name in ("plot_game_length_distribution", "plot_legal_move_distribution",
                     "plot_outcome_rates", "plot_k_by_phase", "plot_prefix_histogram",
                     "plot_probe_heatmap", "plot_probe_comparison",
                     "plot_probe_layer_profile", "plot_outcome_signal_results",
                     "plot_outcome_distributions", "plot_diagnostic_results",
                     "plot_lichess_comparison", "plot_prefix_continuation"):
            assert callable(getattr(viz, name))
