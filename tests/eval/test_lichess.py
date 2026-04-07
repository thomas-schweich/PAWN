"""Tests for pawn/eval_suite/lichess.py.

Covers prepare_lichess_corpus, _extract_elos_from_pgn, evaluate_on_lichess.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pawn.eval_suite.lichess import (
    _extract_elos_from_pgn,
    evaluate_on_lichess,
    prepare_lichess_corpus,
)


# Inline Lichess-format PGN: 3 games covering different Elo bands
SAMPLE_PGN = """\
[Event "Rated Rapid game"]
[Site "https://lichess.org/game001"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[WhiteElo "1200"]
[BlackElo "1250"]
[TimeControl "600+0"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 1-0

[Event "Rated Blitz game"]
[Site "https://lichess.org/game002"]
[White "carol"]
[Black "dave"]
[Result "0-1"]
[WhiteElo "1600"]
[BlackElo "1650"]
[TimeControl "300+3"]
[Termination "Normal"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 0-1

[Event "Rated Bullet game"]
[Site "https://lichess.org/game003"]
[White "eve"]
[Black "frank"]
[Result "1/2-1/2"]
[WhiteElo "1900"]
[BlackElo "1950"]
[TimeControl "60+0"]
[Termination "Normal"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 1/2-1/2
"""


@pytest.fixture
def pgn_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.pgn"
    p.write_text(SAMPLE_PGN)
    return p


# ---------------------------------------------------------------------------
# _extract_elos_from_pgn
# ---------------------------------------------------------------------------


class TestExtractElosFromPGN:
    @pytest.mark.unit
    def test_returns_list_of_tuples(self, pgn_file: Path):
        elos = _extract_elos_from_pgn(pgn_file, max_games=10)
        assert isinstance(elos, list)
        for e in elos:
            assert isinstance(e, tuple)
            assert len(e) == 2

    @pytest.mark.unit
    def test_extracts_all_games(self, pgn_file: Path):
        elos = _extract_elos_from_pgn(pgn_file, max_games=10)
        assert len(elos) == 3

    @pytest.mark.unit
    def test_extracts_correct_white_elos(self, pgn_file: Path):
        elos = _extract_elos_from_pgn(pgn_file, max_games=10)
        white_elos = [w for w, _ in elos]
        assert 1200 in white_elos
        assert 1600 in white_elos
        assert 1900 in white_elos

    @pytest.mark.unit
    def test_extracts_correct_black_elos(self, pgn_file: Path):
        elos = _extract_elos_from_pgn(pgn_file, max_games=10)
        black_elos = [b for _, b in elos]
        assert 1250 in black_elos
        assert 1650 in black_elos
        assert 1950 in black_elos

    @pytest.mark.unit
    @pytest.mark.xfail(strict=True, reason="BUG-700: _extract_elos_from_pgn double-flushes when max_games reached")
    def test_max_games_respected(self, pgn_file: Path):
        elos = _extract_elos_from_pgn(pgn_file, max_games=2)
        # When max is reached, parser breaks early; may return <= max
        assert len(elos) <= 2

    @pytest.mark.unit
    def test_default_elo_when_missing(self, tmp_path: Path):
        pgn = """\
[Event "Game Without Elo"]
[White "x"]
[Black "y"]
[Result "1-0"]

1. e4 e5 1-0
"""
        p = tmp_path / "noelo.pgn"
        p.write_text(pgn)
        elos = _extract_elos_from_pgn(p, max_games=10)
        assert len(elos) == 1
        # Default is 1500
        assert elos[0] == (1500, 1500)


# ---------------------------------------------------------------------------
# prepare_lichess_corpus
# ---------------------------------------------------------------------------


class TestPrepareLichessCorpus:
    @pytest.mark.unit
    def test_returns_expected_keys(self, pgn_file: Path):
        corpus = prepare_lichess_corpus(pgn_file, max_ply=64)
        assert "bands" in corpus
        assert "all_move_ids" in corpus
        assert "all_game_lengths" in corpus

    @pytest.mark.unit
    def test_bands_keyed_by_elo_range(self, pgn_file: Path):
        corpus = prepare_lichess_corpus(
            pgn_file,
            elo_bands=[(1000, 1400), (1400, 1800), (1800, 2200)],
            max_games_per_band=10,
            max_ply=64,
        )
        # Each game should map to its band based on average Elo
        for band_name in corpus["bands"]:
            assert band_name.startswith("elo_")

    @pytest.mark.unit
    def test_band_entries_have_arrays(self, pgn_file: Path):
        corpus = prepare_lichess_corpus(
            pgn_file,
            elo_bands=[(1000, 1400), (1400, 1800), (1800, 2200)],
            max_games_per_band=10,
            max_ply=64,
        )
        for band, data in corpus["bands"].items():
            assert "move_ids" in data
            assert "game_lengths" in data
            assert "n_games" in data
            assert "elo_range" in data

    @pytest.mark.unit
    def test_game_correctly_stratified(self, pgn_file: Path):
        # avg elos: 1225 (alice/bob), 1625 (carol/dave), 1925 (eve/frank)
        corpus = prepare_lichess_corpus(
            pgn_file,
            elo_bands=[(1000, 1400), (1400, 1800), (1800, 2200)],
            max_games_per_band=10,
            max_ply=64,
        )
        # Each should be in distinct bands
        assert "elo_1000_1400" in corpus["bands"]
        assert "elo_1400_1800" in corpus["bands"]
        assert "elo_1800_2200" in corpus["bands"]
        assert corpus["bands"]["elo_1000_1400"]["n_games"] == 1
        assert corpus["bands"]["elo_1400_1800"]["n_games"] == 1
        assert corpus["bands"]["elo_1800_2200"]["n_games"] == 1

    @pytest.mark.unit
    def test_max_games_per_band(self, pgn_file: Path):
        corpus = prepare_lichess_corpus(
            pgn_file,
            elo_bands=[(1000, 2200)],
            max_games_per_band=2,
            max_ply=64,
        )
        band = corpus["bands"].get("elo_1000_2200")
        if band is not None:
            assert band["n_games"] <= 2

    @pytest.mark.unit
    def test_band_elo_range_preserved(self, pgn_file: Path):
        corpus = prepare_lichess_corpus(
            pgn_file,
            elo_bands=[(1400, 1800)],
            max_games_per_band=10,
            max_ply=64,
        )
        band = corpus["bands"]["elo_1400_1800"]
        assert band["elo_range"] == (1400, 1800)

    @pytest.mark.unit
    def test_band_boundaries_inclusive_exclusive(self, tmp_path: Path):
        # Game with avg Elo exactly 1400 should fall in (1400, 1800)
        pgn = """\
[Event "Boundary"]
[White "a"]
[Black "b"]
[Result "1-0"]
[WhiteElo "1400"]
[BlackElo "1400"]

1. e4 e5 1-0
"""
        p = tmp_path / "edge.pgn"
        p.write_text(pgn)
        corpus = prepare_lichess_corpus(
            p,
            elo_bands=[(1000, 1400), (1400, 1800)],
            max_games_per_band=10,
            max_ply=64,
        )
        # avg = 1400 → in [1400, 1800) band
        if "elo_1400_1800" in corpus["bands"]:
            assert corpus["bands"]["elo_1400_1800"]["n_games"] == 1


# ---------------------------------------------------------------------------
# evaluate_on_lichess (smoke with toy model)
# ---------------------------------------------------------------------------


class TestEvaluateOnLichess:
    @pytest.mark.unit
    def test_smoke_returns_dict(self, pgn_file, toy_model, cpu_device):
        corpus = prepare_lichess_corpus(
            pgn_file,
            elo_bands=[(1000, 2200)],
            max_games_per_band=10,
            max_ply=64,
        )
        results = evaluate_on_lichess(
            toy_model, corpus, cpu_device,
            max_seq_len=64, eval_batch_size=2,
        )
        assert isinstance(results, dict)

    @pytest.mark.unit
    def test_result_has_expected_metrics(self, pgn_file, toy_model, cpu_device):
        corpus = prepare_lichess_corpus(
            pgn_file,
            elo_bands=[(1000, 2200)],
            max_games_per_band=10,
            max_ply=64,
        )
        results = evaluate_on_lichess(
            toy_model, corpus, cpu_device,
            max_seq_len=64, eval_batch_size=2,
        )
        for band, metrics in results.items():
            assert "n_games" in metrics
            assert "n_tokens" in metrics
            assert "loss" in metrics
            assert "perplexity" in metrics
            assert "top1_accuracy" in metrics
            assert "top5_accuracy" in metrics
            assert "legal_move_rate" in metrics
            assert "elo_range" in metrics

    @pytest.mark.unit
    def test_metrics_ranges(self, pgn_file, toy_model, cpu_device):
        corpus = prepare_lichess_corpus(
            pgn_file,
            elo_bands=[(1000, 2200)],
            max_games_per_band=10,
            max_ply=64,
        )
        results = evaluate_on_lichess(
            toy_model, corpus, cpu_device,
            max_seq_len=64, eval_batch_size=2,
        )
        for band, m in results.items():
            assert m["loss"] > 0.0
            assert m["perplexity"] > 1.0
            assert 0.0 <= m["top1_accuracy"] <= 1.0
            assert 0.0 <= m["top5_accuracy"] <= 1.0
            assert 0.0 <= m["legal_move_rate"] <= 1.0
            # top5 >= top1
            assert m["top5_accuracy"] >= m["top1_accuracy"]

    @pytest.mark.unit
    def test_skips_empty_band(self, tmp_path, toy_model, cpu_device):
        # Create a corpus with an empty band
        corpus = {
            "bands": {},
            "all_move_ids": np.zeros((0, 1), dtype=np.int16),
            "all_game_lengths": np.zeros(0, dtype=np.int16),
        }
        results = evaluate_on_lichess(
            toy_model, corpus, cpu_device,
            max_seq_len=64, eval_batch_size=2,
        )
        assert results == {}
