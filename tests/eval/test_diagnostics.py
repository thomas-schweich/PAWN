"""Tests for pawn/eval_suite/diagnostics.py.

Covers DIAGNOSTIC_CATEGORIES, EDGE_BITS, generate_diagnostic_corpus,
extract_diagnostic_positions, evaluate_diagnostic_positions,
_term_code_to_outcome_name.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pawn.eval_suite.diagnostics import (
    _CAT_BIT_INDEX,
    _TERMINAL_CATEGORIES,
    _term_code_to_outcome_name,
    DIAGNOSTIC_CATEGORIES,
    EDGE_BITS,
    evaluate_diagnostic_positions,
    extract_diagnostic_positions,
    generate_diagnostic_corpus,
)


# ---------------------------------------------------------------------------
# EDGE_BITS / DIAGNOSTIC_CATEGORIES constants
# ---------------------------------------------------------------------------


class TestEdgeBits:
    @pytest.mark.unit
    def test_edge_bits_is_dict(self):
        assert isinstance(EDGE_BITS, dict)
        assert len(EDGE_BITS) > 0

    @pytest.mark.unit
    def test_in_check_bit_is_1(self):
        assert EDGE_BITS["IN_CHECK"] == 1

    @pytest.mark.unit
    def test_known_categories_present(self):
        for key in ("IN_CHECK", "IN_DOUBLE_CHECK", "CHECKMATE", "STALEMATE",
                    "PIN_RESTRICTS_MOVEMENT", "EP_CAPTURE_AVAILABLE",
                    "PROMOTION_AVAILABLE", "CASTLE_LEGAL_KINGSIDE",
                    "CASTLE_LEGAL_QUEENSIDE", "CASTLE_BLOCKED_CHECK"):
            assert key in EDGE_BITS

    @pytest.mark.unit
    def test_all_values_are_powers_of_two(self):
        for name, val in EDGE_BITS.items():
            assert val > 0
            # Powers of 2: (v & (v-1)) == 0
            assert (val & (val - 1)) == 0, f"{name}={val} not a power of 2"


class TestDiagnosticCategories:
    @pytest.mark.unit
    def test_has_10_categories(self):
        assert len(DIAGNOSTIC_CATEGORIES) == 10

    @pytest.mark.unit
    def test_contains_expected_names(self):
        expected = {
            "in_check", "double_check", "pin_restricts", "ep_available",
            "castle_legal_k", "castle_legal_q", "castle_blocked_check",
            "promotion_available", "checkmate", "stalemate",
        }
        assert set(DIAGNOSTIC_CATEGORIES.keys()) == expected

    @pytest.mark.unit
    def test_all_values_are_ints(self):
        for name, val in DIAGNOSTIC_CATEGORIES.items():
            assert isinstance(val, int)
            assert val > 0

    @pytest.mark.unit
    def test_in_check_maps_to_bit_1(self):
        assert DIAGNOSTIC_CATEGORIES["in_check"] == 1


class TestCatBitIndex:
    @pytest.mark.unit
    def test_has_all_categories(self):
        assert set(_CAT_BIT_INDEX.keys()) == set(DIAGNOSTIC_CATEGORIES.keys())

    @pytest.mark.unit
    def test_bit_index_matches_bit_value(self):
        for name, bit_val in DIAGNOSTIC_CATEGORIES.items():
            idx = _CAT_BIT_INDEX[name]
            assert (1 << idx) == bit_val


class TestTerminalCategories:
    @pytest.mark.unit
    def test_terminal_categories_are_subset(self):
        for k in _TERMINAL_CATEGORIES:
            assert k in DIAGNOSTIC_CATEGORIES

    @pytest.mark.unit
    def test_checkmate_maps_to_tc_0(self):
        assert _TERMINAL_CATEGORIES["checkmate"] == 0

    @pytest.mark.unit
    def test_stalemate_maps_to_tc_1(self):
        assert _TERMINAL_CATEGORIES["stalemate"] == 1


# ---------------------------------------------------------------------------
# _term_code_to_outcome_name
# ---------------------------------------------------------------------------


class TestTermCodeToOutcomeName:
    @pytest.mark.unit
    def test_checkmate_white_wins(self):
        assert _term_code_to_outcome_name(0, 5) == "WHITE_CHECKMATES"

    @pytest.mark.unit
    def test_checkmate_black_wins(self):
        assert _term_code_to_outcome_name(0, 6) == "BLACK_CHECKMATES"

    @pytest.mark.unit
    def test_stalemate(self):
        assert _term_code_to_outcome_name(1, 100) == "STALEMATE"

    @pytest.mark.unit
    def test_draw_by_rule(self):
        assert _term_code_to_outcome_name(2, 100) == "DRAW_BY_RULE"
        assert _term_code_to_outcome_name(3, 100) == "DRAW_BY_RULE"
        assert _term_code_to_outcome_name(4, 100) == "DRAW_BY_RULE"

    @pytest.mark.unit
    def test_ply_limit(self):
        assert _term_code_to_outcome_name(5, 255) == "PLY_LIMIT"

    @pytest.mark.unit
    def test_unknown_fallthrough(self):
        assert _term_code_to_outcome_name(99, 100) == "UNKNOWN"


# ---------------------------------------------------------------------------
# generate_diagnostic_corpus (integration — uses Rust engine)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_diag_corpus() -> dict:
    """Small diagnostic corpus."""
    return generate_diagnostic_corpus(
        n_per_category=8, max_ply=64, seed=42, max_simulated_factor=50.0,
    )


class TestGenerateDiagnosticCorpus:
    @pytest.mark.unit
    def test_returns_expected_keys(self, tiny_diag_corpus: dict):
        for key in ("move_ids", "game_lengths", "termination_codes", "per_ply_stats"):
            assert key in tiny_diag_corpus

    @pytest.mark.unit
    def test_arrays_are_numpy(self, tiny_diag_corpus: dict):
        for key in ("move_ids", "game_lengths", "termination_codes", "per_ply_stats"):
            assert isinstance(tiny_diag_corpus[key], np.ndarray)

    @pytest.mark.unit
    def test_consistent_game_count(self, tiny_diag_corpus: dict):
        n = len(tiny_diag_corpus["game_lengths"])
        assert n > 0
        assert tiny_diag_corpus["termination_codes"].shape[0] == n
        assert tiny_diag_corpus["move_ids"].shape[0] == n
        assert tiny_diag_corpus["per_ply_stats"].shape[0] == n

    @pytest.mark.unit
    def test_game_lengths_positive(self, tiny_diag_corpus: dict):
        assert (tiny_diag_corpus["game_lengths"] > 0).all()


# ---------------------------------------------------------------------------
# extract_diagnostic_positions
# ---------------------------------------------------------------------------


class TestExtractDiagnosticPositions:
    @pytest.mark.unit
    def test_returns_dict_per_category(self, tiny_diag_corpus: dict):
        positions = extract_diagnostic_positions(tiny_diag_corpus, max_per_category=8)
        assert set(positions.keys()) == set(DIAGNOSTIC_CATEGORIES.keys())

    @pytest.mark.unit
    def test_values_are_lists(self, tiny_diag_corpus: dict):
        positions = extract_diagnostic_positions(tiny_diag_corpus, max_per_category=8)
        for cat, lst in positions.items():
            assert isinstance(lst, list)

    @pytest.mark.unit
    def test_position_entries_have_expected_keys(self, tiny_diag_corpus: dict):
        positions = extract_diagnostic_positions(tiny_diag_corpus, max_per_category=8)
        for cat_name, lst in positions.items():
            for entry in lst:
                for key in ("game_idx", "ply", "game_length", "term_code", "outcome_name"):
                    assert key in entry

    @pytest.mark.unit
    def test_max_per_category_enforced(self, tiny_diag_corpus: dict):
        positions = extract_diagnostic_positions(tiny_diag_corpus, max_per_category=2)
        for cat_name, lst in positions.items():
            assert len(lst) <= 2

    @pytest.mark.unit
    def test_works_without_precomputed_stats(self, tiny_diag_corpus: dict):
        # Remove per_ply_stats to force fallback compute path
        corpus_no_stats = {
            k: v for k, v in tiny_diag_corpus.items() if k != "per_ply_stats"
        }
        positions = extract_diagnostic_positions(corpus_no_stats, max_per_category=4)
        assert set(positions.keys()) == set(DIAGNOSTIC_CATEGORIES.keys())

    @pytest.mark.unit
    def test_terminal_category_matches_termination_code(self, tiny_diag_corpus: dict):
        positions = extract_diagnostic_positions(tiny_diag_corpus, max_per_category=32)
        tc = tiny_diag_corpus["termination_codes"]
        # Every checkmate entry should have term_code=0
        for entry in positions["checkmate"]:
            assert entry["term_code"] == 0
            assert tc[entry["game_idx"]] == 0
        # Every stalemate entry should have term_code=1
        for entry in positions["stalemate"]:
            assert entry["term_code"] == 1
            assert tc[entry["game_idx"]] == 1

    @pytest.mark.unit
    def test_terminal_position_ply_equals_game_length(self, tiny_diag_corpus: dict):
        positions = extract_diagnostic_positions(tiny_diag_corpus, max_per_category=32)
        for cat_name in ("checkmate", "stalemate"):
            for entry in positions[cat_name]:
                assert entry["ply"] == entry["game_length"]

    @pytest.mark.unit
    def test_non_terminal_bit_is_set(self, tiny_diag_corpus: dict):
        positions = extract_diagnostic_positions(tiny_diag_corpus, max_per_category=8)
        per_ply_stats = tiny_diag_corpus["per_ply_stats"]
        # For non-terminal categories, check the bit is indeed set in stats
        for cat_name in ("in_check", "pin_restricts", "promotion_available"):
            bit = DIAGNOSTIC_CATEGORIES[cat_name]
            for entry in positions[cat_name]:
                g = entry["game_idx"]
                t = entry["ply"]
                assert per_ply_stats[g, t] & bit, (
                    f"{cat_name} bit not set at game={g} ply={t}"
                )


# ---------------------------------------------------------------------------
# evaluate_diagnostic_positions (smoke with toy model)
# ---------------------------------------------------------------------------


class TestEvaluateDiagnosticPositions:
    @pytest.mark.unit
    def test_smoke_returns_dict(self, tiny_diag_corpus, toy_model, cpu_device):
        positions = extract_diagnostic_positions(tiny_diag_corpus, max_per_category=2)
        # Keep only categories that have positions, else evaluate skips
        results = evaluate_diagnostic_positions(
            toy_model, positions, tiny_diag_corpus, cpu_device,
            n_samples=8, batch_size=4,
        )
        assert isinstance(results, dict)

    @pytest.mark.unit
    def test_result_entries_have_metrics(self, tiny_diag_corpus, toy_model, cpu_device):
        positions = extract_diagnostic_positions(tiny_diag_corpus, max_per_category=2)
        results = evaluate_diagnostic_positions(
            toy_model, positions, tiny_diag_corpus, cpu_device,
            n_samples=8, batch_size=4,
        )
        for cat, metrics in results.items():
            assert "n_positions" in metrics
            assert "terminal" in metrics
            assert "mean_legal_rate" in metrics
            assert "std_legal_rate" in metrics
            assert "mean_pad_prob" in metrics
            assert "mean_entropy" in metrics
            assert "std_entropy" in metrics

    @pytest.mark.unit
    def test_terminal_flag_for_checkmate_stalemate(self, tiny_diag_corpus, toy_model, cpu_device):
        positions = extract_diagnostic_positions(tiny_diag_corpus, max_per_category=2)
        results = evaluate_diagnostic_positions(
            toy_model, positions, tiny_diag_corpus, cpu_device,
            n_samples=8, batch_size=4,
        )
        if "checkmate" in results:
            assert results["checkmate"]["terminal"] is True
        if "stalemate" in results:
            assert results["stalemate"]["terminal"] is True
        if "in_check" in results:
            assert results["in_check"]["terminal"] is False

    @pytest.mark.unit
    def test_metrics_in_valid_range(self, tiny_diag_corpus, toy_model, cpu_device):
        positions = extract_diagnostic_positions(tiny_diag_corpus, max_per_category=2)
        results = evaluate_diagnostic_positions(
            toy_model, positions, tiny_diag_corpus, cpu_device,
            n_samples=8, batch_size=4,
        )
        for cat, m in results.items():
            assert 0.0 <= m["mean_legal_rate"] <= 1.0
            assert 0.0 <= m["mean_pad_prob"] <= 1.0
            assert m["mean_entropy"] >= 0.0

    @pytest.mark.unit
    def test_skips_empty_categories(self, tiny_diag_corpus, toy_model, cpu_device):
        # Pass in all-empty positions
        empty_positions = {k: [] for k in DIAGNOSTIC_CATEGORIES}
        results = evaluate_diagnostic_positions(
            toy_model, empty_positions, tiny_diag_corpus, cpu_device,
            n_samples=8, batch_size=4,
        )
        assert results == {}
