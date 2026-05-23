"""Tests for the JAX-native Lichess adapter-training data path.

Covers ``pawn.corpus.pack_corpus`` (pre-tokenized games → Corpus),
``pawn.lichess_data.make_epoch_schedule`` (finite-dataset multi-epoch
tiling), and ``pawn.lichess_data.load_lichess_corpus`` (parquet scan +
Elo / ply filter + pack + on-disk cache round-trip).

The Lichess load tests build a synthetic parquet in ``tmp_path``.
Polars is a base dep so no ``importorskip`` is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pawn.config import N_OUTCOMES, OUTCOME_TOKEN_BASE, PAD_TOKEN
from pawn.corpus import Corpus, pack_corpus
from pawn.lichess_data import make_epoch_schedule


# ---------------------------------------------------------------------------
# pack_corpus
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pack_corpus_shapes_and_masks() -> None:
    # 3 games, ply lengths 2 / 4 / 0, packed to width 6.
    move_ids = np.array(
        [
            [10, 11, 0, 0, 0, 0],
            [20, 21, 22, 23, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    game_lengths = np.array([2, 4, 0], dtype=np.int32)
    outcome_offset = np.array([0, 3, 4], dtype=np.uint8)
    corpus = pack_corpus(move_ids, game_lengths, outcome_offset, seq_len=6)

    assert isinstance(corpus, Corpus)
    assert corpus.tokens.shape == (3, 6)
    assert corpus.n_games == 3
    assert corpus.seq_len == 6
    # Real moves preserved; tail PAD-filled.
    assert corpus.tokens[0, 0] == 10
    assert corpus.tokens[0, 2] == PAD_TOKEN
    # attn_mask True only on real-move positions.
    assert corpus.attn_mask[0].tolist() == [
        True, True, False, False, False, False
    ]
    assert corpus.attn_mask[1].tolist() == [
        True, True, True, True, False, False
    ]
    assert not corpus.attn_mask[2].any()  # zero-move game
    # outcome offsets carried through.
    assert corpus.outcome_offset.tolist() == [0, 3, 4]


@pytest.mark.unit
def test_pack_corpus_rejects_length_mismatch() -> None:
    move_ids = np.zeros((2, 4), dtype=np.int32)
    game_lengths = np.array([1, 2], dtype=np.int32)
    bad_offsets = np.array([0], dtype=np.uint8)  # only 1, need 2
    with pytest.raises(ValueError, match="outcome_offset"):
        pack_corpus(move_ids, game_lengths, bad_offsets, seq_len=4)


# ---------------------------------------------------------------------------
# make_epoch_schedule
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_epoch_schedule_length_and_range() -> None:
    sched = make_epoch_schedule(10, 25, seed=0)
    assert sched.shape == (25,)
    assert sched.min() >= 0
    assert sched.max() < 10


@pytest.mark.unit
def test_epoch_schedule_each_epoch_is_a_permutation() -> None:
    sched = make_epoch_schedule(8, 24, seed=1)
    # Each consecutive block of n_pool is a full permutation.
    assert sorted(sched[0:8].tolist()) == list(range(8))
    assert sorted(sched[8:16].tolist()) == list(range(8))
    assert sorted(sched[16:24].tolist()) == list(range(8))


@pytest.mark.unit
def test_epoch_schedule_epochs_differ() -> None:
    sched = make_epoch_schedule(20, 40, seed=2)
    # Two independent permutations of the same pool should not be
    # identical (vanishingly unlikely for n_pool=20).
    assert not np.array_equal(sched[:20], sched[20:40])


@pytest.mark.unit
def test_epoch_schedule_deterministic_under_seed() -> None:
    a = make_epoch_schedule(15, 50, seed=7)
    b = make_epoch_schedule(15, 50, seed=7)
    assert np.array_equal(a, b)
    c = make_epoch_schedule(15, 50, seed=8)
    assert not np.array_equal(a, c)


@pytest.mark.unit
def test_epoch_schedule_exact_multiple() -> None:
    sched = make_epoch_schedule(10, 30, seed=0)
    assert sched.shape == (30,)


@pytest.mark.unit
def test_epoch_schedule_n_needed_smaller_than_pool() -> None:
    sched = make_epoch_schedule(100, 10, seed=0)
    assert sched.shape == (10,)
    # All distinct (first 10 of a single permutation).
    assert len(set(sched.tolist())) == 10


@pytest.mark.unit
def test_epoch_schedule_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="n_pool"):
        make_epoch_schedule(0, 10, seed=0)
    with pytest.raises(ValueError, match="n_needed"):
        make_epoch_schedule(10, -1, seed=0)


# ---------------------------------------------------------------------------
# load_lichess_corpus — synthetic parquet
# ---------------------------------------------------------------------------


def _write_synthetic_parquet(path: Path, rows: list[dict]) -> None:
    """Write a Lichess-schema parquet with the given rows."""
    import polars as pl
    pl.DataFrame(
        rows,
        schema={
            "tokens": pl.List(pl.Int32),
            "game_length": pl.Int32,
            "outcome_token": pl.Int32,
            "white_elo": pl.Int32,
            "black_elo": pl.Int32,
        },
    ).write_parquet(path)


def _game(
    tokens: list[int], outcome_off: int, white_elo: int, black_elo: int
) -> dict:
    return {
        "tokens": tokens,
        "game_length": len(tokens),
        "outcome_token": OUTCOME_TOKEN_BASE + outcome_off,
        "white_elo": white_elo,
        "black_elo": black_elo,
    }


@pytest.mark.unit
def test_load_lichess_corpus_basic(tmp_path: Path) -> None:
    from pawn.lichess_data import load_lichess_corpus

    pq = tmp_path / "lichess.parquet"
    _write_synthetic_parquet(
        pq,
        [
            _game([1, 2, 3, 4], 0, 1700, 1750),
            _game([5, 6, 7, 8, 9, 10], 3, 1820, 1810),
            _game([11, 12], 4, 1500, 1490),
        ],
    )
    corpus = load_lichess_corpus(
        str(pq), seq_len=12, min_ply=0, cache_dir=tmp_path / "cache"
    )
    assert corpus.n_games == 3
    assert corpus.seq_len == 12
    # First game: 4 real moves, then PAD.
    assert corpus.tokens[0, :4].tolist() == [1, 2, 3, 4]
    assert corpus.tokens[0, 4] == PAD_TOKEN
    assert corpus.attn_mask[0].sum() == 4
    # outcome offsets recovered from outcome_token.
    assert corpus.outcome_offset.tolist() == [0, 3, 4]


@pytest.mark.unit
def test_load_lichess_corpus_elo_filter(tmp_path: Path) -> None:
    from pawn.lichess_data import load_lichess_corpus

    pq = tmp_path / "lichess.parquet"
    _write_synthetic_parquet(
        pq,
        [
            _game([1, 2, 3], 0, 1500, 1500),   # both < 1800 → excluded
            _game([4, 5, 6], 0, 1850, 1820),   # both in [1800,2000) → in
            _game([7, 8, 9], 0, 1900, 1700),   # black < 1800 → excluded
            _game([1, 1, 1], 0, 1950, 1990),   # both in band → in
        ],
    )
    corpus = load_lichess_corpus(
        str(pq),
        elo_min=1800,
        elo_max=2000,
        seq_len=8,
        min_ply=0,
        cache_dir=tmp_path / "cache",
    )
    # Only the 2 games where BOTH players are in [1800, 2000).
    assert corpus.n_games == 2


@pytest.mark.unit
def test_load_lichess_corpus_min_ply_filter(tmp_path: Path) -> None:
    from pawn.lichess_data import load_lichess_corpus

    pq = tmp_path / "lichess.parquet"
    _write_synthetic_parquet(
        pq,
        [
            _game([1, 2], 0, 1700, 1700),               # 2 ply → dropped
            _game(list(range(1, 16)), 0, 1700, 1700),   # 15 ply → kept
            _game(list(range(1, 21)), 0, 1700, 1700),   # 20 ply → kept
        ],
    )
    corpus = load_lichess_corpus(
        str(pq), min_ply=10, seq_len=24, cache_dir=tmp_path / "cache"
    )
    assert corpus.n_games == 2


@pytest.mark.unit
def test_load_lichess_corpus_cache_round_trip(tmp_path: Path) -> None:
    from pawn.lichess_data import load_lichess_corpus

    pq = tmp_path / "lichess.parquet"
    cache = tmp_path / "cache"
    _write_synthetic_parquet(
        pq,
        [_game([1, 2, 3, 4, 5], 0, 1700, 1700) for _ in range(5)],
    )
    first = load_lichess_corpus(
        str(pq), seq_len=8, min_ply=0, cache_dir=cache
    )
    # The cache directory now holds exactly one entry with the
    # sentinel + corpus + meta.
    entries = [p for p in cache.iterdir() if p.is_dir()]
    assert len(entries) == 1
    assert (entries[0] / ".complete").exists()
    assert (entries[0] / "corpus.safetensors").exists()
    meta = json.loads((entries[0] / "meta.json").read_text())
    assert meta["n_games"] == 5

    # Second load hits the cache and returns identical arrays even
    # after the source parquet is deleted.
    pq.unlink()
    second = load_lichess_corpus(
        str(pq), seq_len=8, min_ply=0, cache_dir=cache
    )
    assert np.array_equal(first.tokens, second.tokens)
    assert np.array_equal(first.outcome_offset, second.outcome_offset)


@pytest.mark.unit
def test_load_lichess_corpus_cache_key_sensitive_to_filter(
    tmp_path: Path,
) -> None:
    """A different Elo band must produce a distinct cache entry — not
    silently reuse the first slice's cache."""
    from pawn.lichess_data import load_lichess_corpus

    pq = tmp_path / "lichess.parquet"
    cache = tmp_path / "cache"
    _write_synthetic_parquet(
        pq,
        [
            _game([1, 2, 3], 0, 1500, 1500),
            _game([4, 5, 6], 0, 1850, 1850),
        ],
    )
    load_lichess_corpus(
        str(pq), elo_min=1000, elo_max=2000, seq_len=8, min_ply=0,
        cache_dir=cache,
    )
    load_lichess_corpus(
        str(pq), elo_min=1800, elo_max=2000, seq_len=8, min_ply=0,
        cache_dir=cache,
    )
    # Two distinct filters → two distinct cache directories.
    entries = [p for p in cache.iterdir() if p.is_dir()]
    assert len(entries) == 2


@pytest.mark.unit
def test_load_lichess_corpus_empty_filter_raises(tmp_path: Path) -> None:
    from pawn.lichess_data import load_lichess_corpus

    pq = tmp_path / "lichess.parquet"
    _write_synthetic_parquet(
        pq, [_game([1, 2, 3], 0, 1500, 1500)]
    )
    with pytest.raises(ValueError, match="matched 0 games"):
        load_lichess_corpus(
            str(pq),
            elo_min=2500,
            elo_max=3000,
            seq_len=8,
            cache_dir=tmp_path / "cache",
        )


@pytest.mark.unit
def test_load_lichess_corpus_missing_columns_raises(tmp_path: Path) -> None:
    import polars as pl
    from pawn.lichess_data import load_lichess_corpus

    pq = tmp_path / "bad.parquet"
    # Missing ``outcome_token`` — not the canonical schema.
    pl.DataFrame(
        {"tokens": [[1, 2, 3]], "game_length": [3]},
        schema={"tokens": pl.List(pl.Int32), "game_length": pl.Int32},
    ).write_parquet(pq)
    with pytest.raises(ValueError, match="missing required columns"):
        load_lichess_corpus(
            str(pq), seq_len=8, cache_dir=tmp_path / "cache"
        )


@pytest.mark.unit
def test_load_lichess_corpus_max_games_cap(tmp_path: Path) -> None:
    from pawn.lichess_data import load_lichess_corpus

    pq = tmp_path / "lichess.parquet"
    _write_synthetic_parquet(
        pq,
        [_game([1, 2, 3, 4], 0, 1700, 1700) for _ in range(20)],
    )
    corpus = load_lichess_corpus(
        str(pq), seq_len=8, max_games=7, min_ply=0,
        cache_dir=tmp_path / "cache",
    )
    assert corpus.n_games == 7


@pytest.mark.unit
def test_load_lichess_corpus_split_prefixed_local_dir(tmp_path: Path) -> None:
    """A local dir with ``train-*.parquet`` / ``validation-*.parquet``
    shards is scanned by split — the HF ``data/{split}-*.parquet``
    convention also applies to local mirrors. The default
    ``--pgn-val-split=validation`` then loads only the held-out
    games."""
    from pawn.lichess_data import load_lichess_corpus

    pq_dir = tmp_path / "lichess"
    pq_dir.mkdir()
    # Two distinct splits with distinct outcome offsets so we can tell
    # them apart in the loaded Corpus.
    _write_synthetic_parquet(
        pq_dir / "train-00000-of-00001.parquet",
        [_game([1, 2, 3, 4, 5], 0, 1700, 1700) for _ in range(10)],
    )
    _write_synthetic_parquet(
        pq_dir / "validation-00000-of-00001.parquet",
        [_game([6, 7, 8, 9, 10], 2, 1700, 1700) for _ in range(4)],
    )
    train_corpus = load_lichess_corpus(
        str(pq_dir),
        split="train",
        seq_len=8,
        min_ply=0,
        cache_dir=tmp_path / "cache",
    )
    val_corpus = load_lichess_corpus(
        str(pq_dir),
        split="validation",
        seq_len=8,
        min_ply=0,
        cache_dir=tmp_path / "cache",
    )
    # Each split loaded its own shards — no cross-contamination.
    assert train_corpus.n_games == 10
    assert val_corpus.n_games == 4
    # outcome offsets match the split's writer (0 for train, 2 for val).
    assert (train_corpus.outcome_offset == 0).all()
    assert (val_corpus.outcome_offset == 2).all()
    # Two distinct cache entries (one per split).
    cache_entries = [p for p in (tmp_path / "cache").iterdir() if p.is_dir()]
    assert len(cache_entries) == 2


@pytest.mark.unit
def test_load_lichess_corpus_truncates_long_games(tmp_path: Path) -> None:
    """A game longer than seq_len is truncated to seq_len moves."""
    from pawn.lichess_data import load_lichess_corpus

    pq = tmp_path / "lichess.parquet"
    _write_synthetic_parquet(
        pq, [_game(list(range(1, 31)), 0, 1700, 1700)]  # 30 plies
    )
    corpus = load_lichess_corpus(
        str(pq), seq_len=10, cache_dir=tmp_path / "cache"
    )
    assert corpus.seq_len == 10
    # Every slot is a real move (game was longer than seq_len).
    assert corpus.attn_mask[0].all()
    assert corpus.outcome_offset[0] < N_OUTCOMES
