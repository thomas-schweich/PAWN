"""Tests for pawn/eval_suite/corpus.py.

Covers popcount helper, _term_to_outcome mapping, _count_legal_moves,
generate_corpus, load_corpus, sanity_checks, summary_stats, and
accumulator helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pawn.eval_suite.corpus import (
    _accumulate,
    _count_legal_moves,
    _finalize_checks,
    _finalize_k_hist,
    _finalize_k_stats,
    _finalize_phases,
    _iter_position_parts,
    _new_accumulator,
    _PHASES,
    _popcount_u64,
    _POPCOUNT_LUT,
    _term_to_outcome,
    generate_corpus,
    load_corpus,
    sanity_checks,
    summary_stats,
)


# ---------------------------------------------------------------------------
# _POPCOUNT_LUT
# ---------------------------------------------------------------------------


class TestPopcountLUT:
    @pytest.mark.unit
    def test_lut_length_is_256(self):
        assert _POPCOUNT_LUT.shape == (256,)

    @pytest.mark.unit
    def test_lut_dtype_uint8(self):
        assert _POPCOUNT_LUT.dtype == np.uint8

    @pytest.mark.unit
    def test_lut_known_entries(self):
        assert _POPCOUNT_LUT[0] == 0
        assert _POPCOUNT_LUT[1] == 1
        assert _POPCOUNT_LUT[2] == 1
        assert _POPCOUNT_LUT[3] == 2
        assert _POPCOUNT_LUT[0xFF] == 8
        assert _POPCOUNT_LUT[0x55] == 4  # 01010101
        assert _POPCOUNT_LUT[0xAA] == 4  # 10101010


# ---------------------------------------------------------------------------
# _popcount_u64
# ---------------------------------------------------------------------------


class TestPopcountU64:
    @pytest.mark.unit
    def test_zero(self):
        arr = np.array([0], dtype=np.uint64)
        result = _popcount_u64(arr)
        assert result[0] == 0

    @pytest.mark.unit
    def test_all_ones(self):
        arr = np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
        result = _popcount_u64(arr)
        assert result[0] == 64

    @pytest.mark.unit
    def test_ff(self):
        arr = np.array([0xFF], dtype=np.uint64)
        result = _popcount_u64(arr)
        assert result[0] == 8

    @pytest.mark.unit
    def test_multiple_values(self):
        arr = np.array([0, 1, 2, 3, 0xFF, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
        result = _popcount_u64(arr)
        expected = np.array([0, 1, 1, 2, 8, 64])
        assert np.array_equal(result, expected)

    @pytest.mark.unit
    def test_returns_uint32(self):
        arr = np.array([0xFF], dtype=np.uint64)
        result = _popcount_u64(arr)
        assert result.dtype == np.uint32

    @pytest.mark.unit
    def test_2d_array(self):
        arr = np.array([[0, 1], [0xFF, 0x0F]], dtype=np.uint64)
        result = _popcount_u64(arr)
        assert result.shape == (2, 2)
        assert result[0, 0] == 0
        assert result[0, 1] == 1
        assert result[1, 0] == 8
        assert result[1, 1] == 4


# ---------------------------------------------------------------------------
# _term_to_outcome
# ---------------------------------------------------------------------------


class TestTermToOutcome:
    @pytest.mark.unit
    def test_white_checkmates_odd_length(self):
        assert _term_to_outcome(0, 1) == "WHITE_CHECKMATES"
        assert _term_to_outcome(0, 5) == "WHITE_CHECKMATES"
        assert _term_to_outcome(0, 101) == "WHITE_CHECKMATES"

    @pytest.mark.unit
    def test_black_checkmates_even_length(self):
        assert _term_to_outcome(0, 2) == "BLACK_CHECKMATES"
        assert _term_to_outcome(0, 4) == "BLACK_CHECKMATES"
        assert _term_to_outcome(0, 100) == "BLACK_CHECKMATES"

    @pytest.mark.unit
    def test_stalemate(self):
        assert _term_to_outcome(1, 10) == "STALEMATE"
        assert _term_to_outcome(1, 55) == "STALEMATE"

    @pytest.mark.unit
    def test_draw_by_rule_codes(self):
        assert _term_to_outcome(2, 50) == "DRAW_BY_RULE"
        assert _term_to_outcome(3, 50) == "DRAW_BY_RULE"
        assert _term_to_outcome(4, 50) == "DRAW_BY_RULE"

    @pytest.mark.unit
    def test_ply_limit_code(self):
        assert _term_to_outcome(5, 255) == "PLY_LIMIT"

    @pytest.mark.unit
    def test_unknown_code_maps_to_ply_limit(self):
        # Anything >= 5 or not in (0-4) falls through to PLY_LIMIT
        assert _term_to_outcome(99, 100) == "PLY_LIMIT"


# ---------------------------------------------------------------------------
# _count_legal_moves
# ---------------------------------------------------------------------------


class TestCountLegalMoves:
    @pytest.mark.unit
    def test_runs_on_tiny_corpus(self):
        import chess_engine
        move_ids, gl, _ = chess_engine.generate_random_games(4, 32, 42)
        counts = _count_legal_moves(move_ids, gl)
        assert counts.shape == (4, 32)
        assert counts.dtype == np.uint16

    @pytest.mark.unit
    def test_counts_positive_for_valid_positions(self):
        import chess_engine
        move_ids, gl, _ = chess_engine.generate_random_games(4, 32, 42)
        counts = _count_legal_moves(move_ids, gl)
        # At ply 0, initial position always has 20 legal moves
        assert counts[0, 0] == 20

    @pytest.mark.unit
    def test_counts_for_all_games_at_start(self):
        import chess_engine
        move_ids, gl, _ = chess_engine.generate_random_games(4, 32, 7)
        counts = _count_legal_moves(move_ids, gl)
        # Every game starts in the initial position
        assert (counts[:, 0] == 20).all()


# ---------------------------------------------------------------------------
# Accumulator helpers
# ---------------------------------------------------------------------------


class TestAccumulator:
    @pytest.mark.unit
    def test_new_accumulator_keys(self):
        acc = _new_accumulator()
        assert acc["n"] == 0
        assert acc["sum_k"] == 0.0
        assert acc["sum_k_sq"] == 0.0
        assert acc["k_min"] == 999
        assert acc["k_max"] == 0
        assert "k_hist" in acc
        assert acc["k_hist"].shape == (300,)

    @pytest.mark.unit
    def test_new_accumulator_has_phase_keys(self):
        acc = _new_accumulator()
        for phase_name, _, _ in _PHASES:
            for suffix in ("n", "sum_k", "sum_inv_k", "sum_ln_k"):
                assert f"{phase_name}_{suffix}" in acc

    @pytest.mark.unit
    def test_new_accumulator_has_check_keys(self):
        acc = _new_accumulator()
        assert "chk_n" in acc and "chk_sum_k" in acc and "chk_sum_inv_k" in acc
        assert "nochk_n" in acc and "nochk_sum_k" in acc and "nochk_sum_inv_k" in acc

    @pytest.mark.unit
    def test_accumulate_updates_counts(self):
        acc = _new_accumulator()
        df = pl.DataFrame({
            "k": np.array([5, 10, 15], dtype=np.uint16),
            "ply": np.array([1, 30, 100], dtype=np.uint16),
            "is_check": np.array([False, True, False]),
        })
        _accumulate(acc, df)
        assert acc["n"] == 3
        assert acc["sum_k"] == pytest.approx(30.0)
        assert acc["k_min"] == 5
        assert acc["k_max"] == 15

    @pytest.mark.unit
    def test_accumulate_empty_dataframe_noop(self):
        acc = _new_accumulator()
        df = pl.DataFrame({
            "k": np.array([], dtype=np.uint16),
            "ply": np.array([], dtype=np.uint16),
            "is_check": np.array([], dtype=bool),
        })
        _accumulate(acc, df)
        assert acc["n"] == 0

    @pytest.mark.unit
    def test_accumulate_partitions_check_status(self):
        acc = _new_accumulator()
        df = pl.DataFrame({
            "k": np.array([10, 20, 30], dtype=np.uint16),
            "ply": np.array([5, 10, 15], dtype=np.uint16),
            "is_check": np.array([True, False, True]),
        })
        _accumulate(acc, df)
        assert acc["chk_n"] == 2
        assert acc["nochk_n"] == 1

    @pytest.mark.unit
    def test_accumulate_partitions_phases(self):
        acc = _new_accumulator()
        # Plies: 0 -> ply_1_20, 30 -> ply_21_80, 100 -> ply_81_150, 200 -> ply_150_plus
        df = pl.DataFrame({
            "k": np.array([10, 20, 30, 40], dtype=np.uint16),
            "ply": np.array([5, 30, 100, 200], dtype=np.uint16),
            "is_check": np.array([False, False, False, False]),
        })
        _accumulate(acc, df)
        assert acc["ply_1_20_n"] == 1
        assert acc["ply_21_80_n"] == 1
        assert acc["ply_81_150_n"] == 1
        assert acc["ply_150_plus_n"] == 1


class TestFinalizers:
    @pytest.mark.unit
    def test_finalize_k_stats(self):
        acc = _new_accumulator()
        df = pl.DataFrame({
            "k": np.array([10, 20, 30], dtype=np.uint16),
            "ply": np.array([1, 2, 3], dtype=np.uint16),
            "is_check": np.array([False, False, False]),
        })
        _accumulate(acc, df)
        stats = _finalize_k_stats(acc)
        assert "mean" in stats and "median" in stats
        assert stats["mean"] == pytest.approx(20.0)
        assert stats["min"] == 10
        assert stats["max"] == 30

    @pytest.mark.unit
    def test_finalize_k_hist_returns_nonzero_only(self):
        acc = _new_accumulator()
        df = pl.DataFrame({
            "k": np.array([5, 5, 10], dtype=np.uint16),
            "ply": np.array([1, 2, 3], dtype=np.uint16),
            "is_check": np.array([False, False, False]),
        })
        _accumulate(acc, df)
        hist = _finalize_k_hist(acc)
        assert "values" in hist and "counts" in hist and "total" in hist
        # 5 and 10 should be present, other values skipped
        assert 5 in hist["values"]
        assert 10 in hist["values"]
        assert hist["total"] == 3

    @pytest.mark.unit
    def test_finalize_phases(self):
        acc = _new_accumulator()
        df = pl.DataFrame({
            "k": np.array([10, 20], dtype=np.uint16),
            "ply": np.array([5, 30], dtype=np.uint16),
            "is_check": np.array([False, False]),
        })
        _accumulate(acc, df)
        phases = _finalize_phases(acc)
        assert "ply_1_20" in phases
        assert "ply_21_80" in phases
        assert phases["ply_1_20"]["n_positions"] == 1
        assert phases["ply_21_80"]["n_positions"] == 1

    @pytest.mark.unit
    def test_finalize_phases_empty_skips(self):
        acc = _new_accumulator()
        df = pl.DataFrame({
            "k": np.array([10], dtype=np.uint16),
            "ply": np.array([5], dtype=np.uint16),
            "is_check": np.array([False]),
        })
        _accumulate(acc, df)
        phases = _finalize_phases(acc)
        # Only ply_1_20 should be present
        assert "ply_1_20" in phases
        assert "ply_21_80" not in phases

    @pytest.mark.unit
    def test_finalize_checks(self):
        acc = _new_accumulator()
        df = pl.DataFrame({
            "k": np.array([5, 10], dtype=np.uint16),
            "ply": np.array([1, 2], dtype=np.uint16),
            "is_check": np.array([True, False]),
        })
        _accumulate(acc, df)
        checks = _finalize_checks(acc)
        assert "in_check" in checks
        assert "not_in_check" in checks
        # Each has 1 sample
        assert checks["in_check"]["frequency"] == pytest.approx(0.5)
        assert checks["not_in_check"]["frequency"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# generate_corpus + load_corpus (end-to-end with tiny corpus)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    """Generate an 8-game corpus for tests."""
    corpus_dir = tmp_path / "corpus"
    generate_corpus(
        output_dir=corpus_dir,
        n_games=8,
        max_ply=32,
        seed=42,
        batch_size=8,
    )
    return corpus_dir


class TestGenerateCorpus:
    @pytest.mark.unit
    def test_writes_expected_files(self, tiny_corpus: Path):
        assert (tiny_corpus / "move_ids.npy").exists()
        assert (tiny_corpus / "games.parquet").exists()
        assert (tiny_corpus / "game_lengths.npy").exists()
        assert (tiny_corpus / "termination_codes.npy").exists()
        assert (tiny_corpus / "metadata.json").exists()
        assert (tiny_corpus / "positions").is_dir()

    @pytest.mark.unit
    def test_metadata_json_content(self, tiny_corpus: Path):
        with open(tiny_corpus / "metadata.json") as f:
            meta = json.load(f)
        assert meta["n_games"] == 8
        assert meta["max_ply"] == 32
        assert meta["seed"] == 42
        assert meta["format"] == "parquet"

    @pytest.mark.unit
    def test_games_parquet_has_expected_columns(self, tiny_corpus: Path):
        df = pl.read_parquet(tiny_corpus / "games.parquet")
        assert "game_idx" in df.columns
        assert "game_length" in df.columns
        assert "term_code" in df.columns
        assert "outcome" in df.columns
        assert len(df) == 8

    @pytest.mark.unit
    def test_move_ids_shape_matches_games(self, tiny_corpus: Path):
        move_ids = np.load(tiny_corpus / "move_ids.npy")
        assert move_ids.shape[0] == 8

    @pytest.mark.unit
    def test_positions_parquet_exists(self, tiny_corpus: Path):
        part_files = list((tiny_corpus / "positions").glob("*.parquet"))
        assert len(part_files) >= 1


class TestLoadCorpus:
    @pytest.mark.unit
    def test_load_returns_expected_keys(self, tiny_corpus: Path):
        corpus = load_corpus(tiny_corpus)
        assert "move_ids" in corpus
        assert "game_lengths" in corpus
        assert "termination_codes" in corpus
        assert "games" in corpus
        assert "positions" in corpus
        assert "metadata" in corpus

    @pytest.mark.unit
    def test_load_move_ids_is_mmap(self, tiny_corpus: Path):
        corpus = load_corpus(tiny_corpus)
        # mmap arrays expose filename
        assert hasattr(corpus["move_ids"], "filename")

    @pytest.mark.unit
    def test_load_games_is_lazyframe(self, tiny_corpus: Path):
        corpus = load_corpus(tiny_corpus)
        assert isinstance(corpus["games"], pl.LazyFrame)

    @pytest.mark.unit
    def test_load_metadata_is_dict(self, tiny_corpus: Path):
        corpus = load_corpus(tiny_corpus)
        assert corpus["metadata"] is not None
        assert corpus["metadata"]["n_games"] == 8

    @pytest.mark.unit
    def test_load_missing_metadata_returns_none(self, tiny_corpus: Path):
        (tiny_corpus / "metadata.json").unlink()
        corpus = load_corpus(tiny_corpus)
        assert corpus["metadata"] is None

    @pytest.mark.unit
    def test_load_arrays_have_correct_length(self, tiny_corpus: Path):
        corpus = load_corpus(tiny_corpus)
        assert len(corpus["game_lengths"]) == 8
        assert len(corpus["termination_codes"]) == 8


class TestIterPositionParts:
    @pytest.mark.unit
    def test_iter_yields_dataframes(self, tiny_corpus: Path):
        corpus = load_corpus(tiny_corpus)
        parts = list(_iter_position_parts(corpus))
        assert len(parts) >= 1
        for part in parts:
            assert isinstance(part, pl.DataFrame)
            # expected columns
            assert "k" in part.columns
            assert "ply" in part.columns
            assert "is_check" in part.columns


# ---------------------------------------------------------------------------
# sanity_checks
# ---------------------------------------------------------------------------


class TestSanityChecks:
    @pytest.mark.unit
    def test_returns_expected_keys(self, tiny_corpus: Path):
        corpus = load_corpus(tiny_corpus)
        result = sanity_checks(corpus)
        assert "duplicates" in result
        assert "max_prefix_moves" in result
        assert "prefix_length_histogram" in result
        assert "duplicate_pairs" in result

    @pytest.mark.unit
    def test_duplicates_is_int(self, tiny_corpus: Path):
        corpus = load_corpus(tiny_corpus)
        result = sanity_checks(corpus)
        assert isinstance(result["duplicates"], int)
        assert result["duplicates"] >= 0

    @pytest.mark.unit
    def test_duplicate_pairs_at_most_20(self, tiny_corpus: Path):
        corpus = load_corpus(tiny_corpus)
        result = sanity_checks(corpus)
        assert len(result["duplicate_pairs"]) <= 20


# ---------------------------------------------------------------------------
# summary_stats
# ---------------------------------------------------------------------------


class TestSummaryStats:
    @pytest.mark.unit
    def test_returns_expected_keys(self, tiny_corpus: Path):
        corpus = load_corpus(tiny_corpus)
        stats = summary_stats(corpus)
        assert "n_games" in stats
        assert "total_positions" in stats
        assert "game_length" in stats
        assert "legal_move_counts" in stats
        assert "k_histogram" in stats
        assert "outcome_rates" in stats
        assert "phase_stats" in stats
        assert "check_stats" in stats

    @pytest.mark.unit
    def test_n_games_matches(self, tiny_corpus: Path):
        corpus = load_corpus(tiny_corpus)
        stats = summary_stats(corpus)
        assert stats["n_games"] == 8

    @pytest.mark.unit
    def test_game_length_has_stats(self, tiny_corpus: Path):
        corpus = load_corpus(tiny_corpus)
        stats = summary_stats(corpus)
        gl = stats["game_length"]
        assert "mean" in gl and "median" in gl
        assert "std" in gl and "min" in gl and "max" in gl
        assert "histogram_counts" in gl and "histogram_edges" in gl

    @pytest.mark.unit
    def test_outcome_rates_sum_to_one(self, tiny_corpus: Path):
        corpus = load_corpus(tiny_corpus)
        stats = summary_stats(corpus)
        total = sum(stats["outcome_rates"].values())
        assert total == pytest.approx(1.0)

    @pytest.mark.unit
    def test_legal_move_counts_has_keys(self, tiny_corpus: Path):
        corpus = load_corpus(tiny_corpus)
        stats = summary_stats(corpus)
        lmc = stats["legal_move_counts"]
        assert "mean" in lmc and "median" in lmc
        assert "min" in lmc and "max" in lmc

    @pytest.mark.unit
    def test_total_positions_positive(self, tiny_corpus: Path):
        corpus = load_corpus(tiny_corpus)
        stats = summary_stats(corpus)
        assert stats["total_positions"] > 0
