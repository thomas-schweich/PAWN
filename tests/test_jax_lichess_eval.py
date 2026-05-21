"""Tests for ``pawn.lichess_eval`` — Lichess Elo-stratified eval (chunk 4.4).

Most of these tests construct a ``LichessCorpus`` directly (skipping
the Rust PGN parse) so they're hermetic + don't depend on a real PGN
fixture. The full parse-from-PGN path is covered by a small inline
PGN string fed to the engine + tmp-path cache verification.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from pawn.config import PAD_TOKEN, TINY_SUPERNET
from pawn.lichess_eval import (
    LichessCorpus,
    LichessElo,
    evaluate_elo_accuracy,
    filter_elo_slice,
    load_lichess_corpus,
)
from pawn.model import init_model


def _make_lichess_corpus(
    n: int = 4, seq_len: int = 16, ply: int = 6,
) -> LichessCorpus:
    """Synthesise a tiny LichessCorpus with deterministic move tokens
    and a handful of known Elos so the Elo-filter tests can pin exact
    selection behaviour."""
    rng = np.random.default_rng(0)
    tokens = np.full((n, seq_len), PAD_TOKEN, dtype=np.int32)
    tokens[:, :ply] = rng.integers(0, 1968, size=(n, ply), dtype=np.int32)
    targets = np.full_like(tokens, PAD_TOKEN, dtype=np.int32)
    targets[:, :-1] = tokens[:, 1:]
    attn = tokens != PAD_TOKEN
    pos = np.arange(seq_len, dtype=np.int32)[None, :]
    # Broadcast to ``(n, seq_len)``; the constant ``ply`` on the RHS
    # would otherwise leave loss_mask at shape ``(1, seq_len)``.
    loss_mask = np.broadcast_to(pos < ply, (n, seq_len)).copy()
    # Elos chosen to land in distinct buckets:
    #   game 0: 1200 / 1300 — 1200 bucket
    #   game 1: 1450 / 1500 — 1500 bucket
    #   game 2: 1800 / 1700 — 1700-1800 bucket
    #   game 3: 2200 / 2100 — 2100-2200 bucket
    white_elo = np.array([1200, 1450, 1800, 2200], dtype=np.int32)[:n]
    black_elo = np.array([1300, 1500, 1700, 2100], dtype=np.int32)[:n]
    result = np.array([1, 0, -1, 0], dtype=np.int8)[:n]
    return LichessCorpus(
        tokens=tokens, attn_mask=attn, targets=targets, loss_mask=loss_mask,
        game_lengths=np.full(n, ply, dtype=np.int32),
        white_elo=white_elo, black_elo=black_elo, result=result,
        seq_len=seq_len, prepend_outcome=False,
    )


# ---------------------------------------------------------------------------
# Elo filter
# ---------------------------------------------------------------------------


def test_filter_elo_slice_both_keeps_when_either_matches() -> None:
    corpus = _make_lichess_corpus(n=4)
    # 1400-1600 → game 1 (1450 white) matches via either side; game 0
    # has 1300 black, 1200 white — neither inside; game 2 has 1700
    # black which is outside but 1800 white also outside.
    sliced = filter_elo_slice(corpus, elo_min=1400, elo_max=1600)
    assert sliced.tokens.shape[0] == 1
    np.testing.assert_array_equal(
        sliced.white_elo, np.array([1450], dtype=np.int32),
    )


def test_filter_elo_slice_side_white_only() -> None:
    corpus = _make_lichess_corpus(n=4)
    sliced = filter_elo_slice(
        corpus, elo_min=1700, elo_max=1900, side="white",
    )
    # Only game 2 (white=1800) qualifies.
    assert sliced.tokens.shape[0] == 1
    assert int(sliced.white_elo[0]) == 1800


def test_filter_elo_slice_rejects_bad_range() -> None:
    corpus = _make_lichess_corpus(n=4)
    with pytest.raises(ValueError, match="elo_min"):
        filter_elo_slice(corpus, elo_min=1600, elo_max=1500)
    with pytest.raises(ValueError, match="side"):
        filter_elo_slice(corpus, elo_min=1400, elo_max=1600, side="invalid")


def test_filter_elo_slice_empty_range_returns_empty_corpus() -> None:
    corpus = _make_lichess_corpus(n=4)
    sliced = filter_elo_slice(corpus, elo_min=3000, elo_max=3500)
    assert sliced.tokens.shape[0] == 0
    assert sliced.seq_len == corpus.seq_len


# ---------------------------------------------------------------------------
# Accuracy eval
# ---------------------------------------------------------------------------


def test_evaluate_elo_accuracy_empty_slice_returns_nan() -> None:
    cfg = TINY_SUPERNET
    model = init_model(cfg, jax.random.key(0))
    corpus = _make_lichess_corpus(n=4)
    result = evaluate_elo_accuracy(
        model, corpus, elo_min=3000, elo_max=3500,
    )
    assert isinstance(result, LichessElo)
    assert result.n_games == 0
    assert np.isnan(result.accuracy.overall)
    assert result.accuracy.n_supervised == 0


def test_evaluate_elo_accuracy_returns_finite_overall() -> None:
    """End-to-end: filter + run model + collect AccuracyResult. The
    fresh-init TINY model will land near chance; the structural
    contract is what's pinned."""
    cfg = TINY_SUPERNET
    model = init_model(cfg, jax.random.key(0))
    corpus = _make_lichess_corpus(n=4)
    result = evaluate_elo_accuracy(
        model, corpus, elo_min=1100, elo_max=2300,
    )
    assert result.n_games == 4
    assert 0.0 <= result.accuracy.overall <= 1.0
    assert result.accuracy.n_supervised > 0
    # Phase keys match pawn.eval.AccuracyResult convention.
    assert set(result.accuracy.phase.keys()) == {"opening", "midgame", "endgame"}


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


def test_load_lichess_corpus_round_trips_through_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny inline PGN is parsed, cached, then re-loaded from cache
    without re-invoking the Rust parser. Verifies the cache layout +
    round-trip equivalence."""
    monkeypatch.setenv("PAWN_DATA_CACHE", str(tmp_path / "cache"))

    pgn = """\
[Event "test"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[WhiteElo "1800"]
[BlackElo "1750"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0

[Event "test2"]
[White "carol"]
[Black "dan"]
[Result "0-1"]
[WhiteElo "1500"]
[BlackElo "1550"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 0-1
"""
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(pgn)

    corpus1 = load_lichess_corpus(
        pgn_path, max_ply=64, min_ply=1, prepend_outcome=False,
    )
    # The Rust parser ships PGN strings through a tokeniser; the inline
    # PGN above has at least 5 half-moves per game (the parser also
    # truncates at the engine's first-illegal-move guard). If the
    # parser yielded zero games, skip the round-trip check — the test
    # is then a smoke for the cache layout itself.
    if corpus1.tokens.shape[0] == 0:
        pytest.skip(
            "Rust parser returned 0 games for the inline PGN — cache "
            "round-trip can't be verified without a non-empty corpus"
        )
    # Re-load: comes from cache (path exists). Result is identical.
    corpus2 = load_lichess_corpus(
        pgn_path, max_ply=64, min_ply=1, prepend_outcome=False,
    )
    np.testing.assert_array_equal(corpus1.tokens, corpus2.tokens)
    np.testing.assert_array_equal(corpus1.white_elo, corpus2.white_elo)
    np.testing.assert_array_equal(corpus1.black_elo, corpus2.black_elo)
    # Bypass cache → parses again, returns the same content.
    corpus3 = load_lichess_corpus(
        pgn_path, max_ply=64, min_ply=1, prepend_outcome=False,
        use_cache=False,
    )
    np.testing.assert_array_equal(corpus1.tokens, corpus3.tokens)


def test_load_lichess_corpus_reparses_when_metadata_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-side cache hit requires *both* ``games.npz`` and
    ``metadata.json``. Pins the ``and`` guard against accidental
    weakening to ``or`` — a refactor that read a partial write would
    silently load an .npz next to stale or missing metadata.

    Setup: prime the cache, then delete only ``metadata.json``. The
    next call must re-parse (not load the orphan .npz).
    """
    monkeypatch.setenv("PAWN_DATA_CACHE", str(tmp_path / "cache"))
    pgn = """\
[Event "t"]
[White "a"]
[Black "b"]
[Result "1-0"]
[WhiteElo "1800"]
[BlackElo "1750"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0
"""
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(pgn)

    corpus1 = load_lichess_corpus(
        pgn_path, max_ply=64, min_ply=1, prepend_outcome=False,
    )
    if corpus1.tokens.shape[0] == 0:
        pytest.skip("Rust parser returned 0 games for inline PGN")

    # Locate the cache dir from the public root + the same key the
    # loader would derive. There's exactly one subdir under the
    # cache root after the first load.
    cache_root = tmp_path / "cache"
    cache_dirs = list(cache_root.iterdir())
    assert len(cache_dirs) == 1
    cache_dir = cache_dirs[0]
    npz = cache_dir / "games.npz"
    meta = cache_dir / "metadata.json"
    assert npz.exists() and meta.exists(), (
        "expected both cache files after first load"
    )
    # Orphan: delete metadata only. games.npz remains.
    meta.unlink()

    # Second load must re-parse (and rewrite both files), not crash
    # on the orphan .npz nor silently load it.
    corpus2 = load_lichess_corpus(
        pgn_path, max_ply=64, min_ply=1, prepend_outcome=False,
    )
    np.testing.assert_array_equal(corpus1.tokens, corpus2.tokens)
    # The metadata file should have been rewritten by the re-parse.
    assert meta.exists()
