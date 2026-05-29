"""Tests for :mod:`pawn.lichess_data` — Polars filter + cache + epoch tiling.

Coverage per plan §10 S5 verification list:

- Synthetic-parquet tests for the Elo filter (both players in range,
  elo_max exclusive).
- min_ply filter.
- Cache round-trip: write → read → byte-equal arrays.
- Cache-key sensitivity: different filter → different cache entry.
- Local-dir split-prefixed scan.
- `make_epoch_schedule` tiles correctly.

Tests against the real HF dataset are not required (plan §10 S5
explicit) — those are a manual smoke for the verification command.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pawn._sentinel import SENTINEL_NAME, CheckpointIntegrityError, write_sentinel
from pawn.config import PAD_TOKEN, DRAW_BY_RULE, WHITE_CHECKMATES
from pawn.lichess_data import (
    _cache_key,
    _default_cache_root,
    _filter_lichess,
    _load_from_cache,
    _pack_dataframe,
    _save_to_cache,
    _scan_parquet,
    load_lichess_corpus,
    make_epoch_schedule,
)


# ---------------------------------------------------------------------------
# Synthetic parquet fixture
# ---------------------------------------------------------------------------


def _write_parquet(path: Path, rows: list[dict]) -> None:
    """Write a parquet shard at `path` with the canonical Lichess schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows, schema_overrides={"tokens": pl.List(pl.Int16)})
    df.write_parquet(path)


def _synthetic_rows(
    n: int,
    *,
    base_elo: int = 1500,
    game_length: int = 12,
    outcome: int = DRAW_BY_RULE,
) -> list[dict]:
    """Build n rows where row i has white_elo=base_elo+i, black_elo=base_elo+i,
    tokens=[1,2,...,game_length] (then PAD for safety)."""
    rows = []
    for i in range(n):
        toks = list(range(1, game_length + 1))
        rows.append({
            "tokens": toks,
            "game_length": game_length,
            "outcome_token": outcome,
            "white_elo": base_elo + i,
            "black_elo": base_elo + i,
        })
    return rows


# ---------------------------------------------------------------------------
# Elo filter
# ---------------------------------------------------------------------------


def test_elo_filter_includes_elo_min(tmp_path: Path) -> None:
    """`elo_min` is inclusive on both players. With base_elo=1800 + 10 rows
    (1800..1809), `elo_min=1805` should keep rows 5..9 (5 games)."""
    parquet = tmp_path / "train-0.parquet"
    _write_parquet(parquet, _synthetic_rows(10, base_elo=1800))
    lf = pl.scan_parquet(str(parquet))
    df = _filter_lichess(
        lf, elo_min=1805, elo_max=None, min_ply=0, max_games=None
    )
    assert len(df) == 5


def test_elo_filter_excludes_elo_max(tmp_path: Path) -> None:
    """`elo_max` is exclusive on both players. With 1800..1809 + elo_max=1805
    we keep rows whose elo < 1805 (i.e. 1800..1804) = 5 games."""
    parquet = tmp_path / "train-0.parquet"
    _write_parquet(parquet, _synthetic_rows(10, base_elo=1800))
    lf = pl.scan_parquet(str(parquet))
    df = _filter_lichess(
        lf, elo_min=None, elo_max=1805, min_ply=0, max_games=None
    )
    assert len(df) == 5


def test_elo_filter_range_combines_min_and_max(tmp_path: Path) -> None:
    """[elo_min, elo_max) → range of allowed Elos."""
    parquet = tmp_path / "train-0.parquet"
    _write_parquet(parquet, _synthetic_rows(20, base_elo=1800))
    lf = pl.scan_parquet(str(parquet))
    df = _filter_lichess(
        lf, elo_min=1810, elo_max=1815, min_ply=0, max_games=None
    )
    # 1810..1814 inclusive of 1810, exclusive of 1815 = 5 games.
    assert len(df) == 5


def test_elo_filter_both_players_must_be_in_range(tmp_path: Path) -> None:
    """A row where white_elo < elo_min OR black_elo < elo_min is dropped."""
    rows = [
        # White below range
        {"tokens": [1, 2, 3], "game_length": 3, "outcome_token": 0,
         "white_elo": 1500, "black_elo": 1900},
        # Black below range
        {"tokens": [1, 2, 3], "game_length": 3, "outcome_token": 0,
         "white_elo": 1900, "black_elo": 1500},
        # Both in range
        {"tokens": [1, 2, 3], "game_length": 3, "outcome_token": 0,
         "white_elo": 1900, "black_elo": 1900},
    ]
    parquet = tmp_path / "train-0.parquet"
    _write_parquet(parquet, rows)
    lf = pl.scan_parquet(str(parquet))
    df = _filter_lichess(lf, elo_min=1800, elo_max=None, min_ply=0, max_games=None)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# min_ply filter
# ---------------------------------------------------------------------------


def test_min_ply_filter_drops_short_games(tmp_path: Path) -> None:
    """Games with game_length < min_ply are dropped."""
    rows = [
        {"tokens": list(range(1, gl + 1)), "game_length": gl,
         "outcome_token": 0, "white_elo": 1500, "black_elo": 1500}
        for gl in (5, 8, 10, 12, 15)
    ]
    parquet = tmp_path / "train-0.parquet"
    _write_parquet(parquet, rows)
    lf = pl.scan_parquet(str(parquet))
    df = _filter_lichess(lf, elo_min=None, elo_max=None, min_ply=10, max_games=None)
    # game_length >= 10 → 10, 12, 15 → 3 games.
    assert len(df) == 3


# ---------------------------------------------------------------------------
# Schema enforcement
# ---------------------------------------------------------------------------


def test_filter_rejects_parquet_missing_required_columns(tmp_path: Path) -> None:
    """A parquet that lacks tokens/game_length/outcome_token surfaces a
    clear schema error before any rows are read."""
    parquet = tmp_path / "train-0.parquet"
    pl.DataFrame({"x": [1, 2, 3]}).write_parquet(parquet)
    lf = pl.scan_parquet(str(parquet))
    with pytest.raises(ValueError, match="missing required columns"):
        _filter_lichess(lf, elo_min=None, elo_max=None, min_ply=0, max_games=None)


def test_filter_rejects_elo_without_elo_columns(tmp_path: Path) -> None:
    """Elo filtering without white_elo/black_elo columns is rejected."""
    rows = [
        {"tokens": [1, 2, 3], "game_length": 3, "outcome_token": 0}
    ]
    parquet = tmp_path / "train-0.parquet"
    pl.DataFrame(rows, schema_overrides={"tokens": pl.List(pl.Int16)}).write_parquet(parquet)
    lf = pl.scan_parquet(str(parquet))
    with pytest.raises(ValueError, match="white_elo/black_elo"):
        _filter_lichess(
            lf, elo_min=1800, elo_max=None, min_ply=0, max_games=None
        )


# ---------------------------------------------------------------------------
# Split-prefixed scan (local dirs)
# ---------------------------------------------------------------------------


def test_scan_local_dir_matches_split_prefix(tmp_path: Path) -> None:
    """`<split>-*.parquet` shards in a local dir are picked up; other
    splits aren't."""
    _write_parquet(tmp_path / "train-0.parquet", _synthetic_rows(3, base_elo=1500))
    _write_parquet(tmp_path / "train-1.parquet", _synthetic_rows(2, base_elo=2000))
    _write_parquet(tmp_path / "validation-0.parquet", _synthetic_rows(4, base_elo=1800))

    train_df = _scan_parquet(str(tmp_path), "train").collect()
    val_df = _scan_parquet(str(tmp_path), "validation").collect()
    assert len(train_df) == 5
    assert len(val_df) == 4


def test_scan_local_dir_falls_back_to_all_parquet_when_no_split_match(
    tmp_path: Path,
) -> None:
    """When no `<split>-*.parquet` exists, fall back to all `*.parquet`
    in the dir (single-file local case the user opts into via
    `--pgn-val-split ""`)."""
    _write_parquet(tmp_path / "data.parquet", _synthetic_rows(3, base_elo=1500))
    df = _scan_parquet(str(tmp_path), "validation").collect()
    assert len(df) == 3


def test_scan_local_dir_rejects_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no parquet files"):
        _scan_parquet(str(tmp_path), "train")


def test_hf_scan_failure_does_not_bulk_download_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the native hf:// lazy scan fails, the bulk-download fallback
    (hf_hub_download of every shard — 262 GB for pawn-lichess-full) must
    NOT run implicitly. Default: raise a clear opt-in error and never
    touch huggingface_hub."""
    import pawn.lichess_data as ld

    def _boom(url: object, *a: object, **k: object) -> object:
        raise OSError("hf:// unsupported on this build")

    monkeypatch.setattr(ld.pl, "scan_parquet", _boom)
    monkeypatch.delenv("PAWN_ALLOW_BULK_DOWNLOAD", raising=False)

    # Guard: if the gate leaks through, importing/using huggingface_hub
    # would be the 262 GB mistake — make that loud rather than networked.
    def _must_not_call(*a: object, **k: object) -> object:  # pragma: no cover
        raise AssertionError("bulk-download fallback ran without opt-in")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _must_not_call)

    with pytest.raises(RuntimeError, match="PAWN_ALLOW_BULK_DOWNLOAD"):
        ld._scan_parquet("thomas-schweich/pawn-lichess-full", "train")


def test_hf_scan_failure_bulk_download_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With PAWN_ALLOW_BULK_DOWNLOAD=1, a failed native scan proceeds to
    the download fallback (gate opens)."""
    import pawn.lichess_data as ld

    calls: list[str] = []

    def _boom_on_hf(url: object, *a: object, **k: object) -> object:
        if isinstance(url, str) and url.startswith("hf://"):
            raise OSError("hf:// unsupported")
        return "scanned-local"  # the post-download scan

    monkeypatch.setattr(ld.pl, "scan_parquet", _boom_on_hf)
    monkeypatch.setenv("PAWN_ALLOW_BULK_DOWNLOAD", "1")

    import huggingface_hub

    class _FakeApi:
        def list_repo_files(self, *a: object, **k: object) -> list[str]:
            return ["data/train-0.parquet", "data/train-1.parquet"]

    def _fake_dl(repo: str, pf: str, *a: object, **k: object) -> str:
        calls.append(pf)
        return f"/tmp/{pf}"

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_dl)

    result = ld._scan_parquet("thomas-schweich/pawn-lichess-full", "train")
    assert result == "scanned-local"
    assert calls == ["data/train-0.parquet", "data/train-1.parquet"]


# ---------------------------------------------------------------------------
# Cache: round-trip + cache-key sensitivity
# ---------------------------------------------------------------------------


def test_cache_round_trip_arrays_byte_equal(tmp_path: Path) -> None:
    """A Corpus written to cache and read back has byte-identical
    arrays."""
    _write_parquet(
        tmp_path / "train-0.parquet",
        _synthetic_rows(5, base_elo=1800, game_length=10),
    )
    cache_root = tmp_path / "cache"
    orig = load_lichess_corpus(
        str(tmp_path),
        split="train",
        elo_min=1800,
        elo_max=None,
        min_ply=0,
        seq_len=16,
        cache_dir=cache_root,
    )
    # Second call should hit the cache — verify by reading the same
    # dir back manually.
    sha = _cache_key(
        str(tmp_path), "train", 1800, None, 0, 16, None, False
    )
    cached = _load_from_cache(cache_root / sha)
    assert np.array_equal(orig.tokens, cached.tokens)
    assert np.array_equal(orig.targets, cached.targets)
    assert np.array_equal(orig.attn_mask, cached.attn_mask)
    assert np.array_equal(orig.loss_mask, cached.loss_mask)
    assert np.array_equal(orig.outcome_offset, cached.outcome_offset)


def test_cache_key_sensitive_to_filter_params(tmp_path: Path) -> None:
    """Different filter params produce different SHA → different cache
    directories."""
    base = _cache_key("repo", "train", 1800, 2000, 10, 512, 100, False)
    # Each change produces a distinct key.
    assert _cache_key("repo", "train", 1801, 2000, 10, 512, 100, False) != base
    assert _cache_key("repo", "train", 1800, 2001, 10, 512, 100, False) != base
    assert _cache_key("repo", "train", 1800, 2000, 11, 512, 100, False) != base
    assert _cache_key("repo", "train", 1800, 2000, 10, 256, 100, False) != base
    assert _cache_key("repo", "train", 1800, 2000, 10, 512, 200, False) != base
    assert _cache_key("repo", "train", 1800, 2000, 10, 512, 100, True) != base
    assert _cache_key("repo", "validation", 1800, 2000, 10, 512, 100, False) != base
    assert _cache_key("other-repo", "train", 1800, 2000, 10, 512, 100, False) != base


def test_cache_recovers_from_corrupted_entry(tmp_path: Path) -> None:
    """If the cache has a stale/corrupt entry, the loader rebuilds
    silently (rather than crashing)."""
    _write_parquet(
        tmp_path / "train-0.parquet",
        _synthetic_rows(3, base_elo=1800, game_length=10),
    )
    cache_root = tmp_path / "cache"
    # Build the cache once.
    corpus = load_lichess_corpus(
        str(tmp_path),
        split="train",
        elo_min=1800,
        seq_len=16,
        cache_dir=cache_root,
    )
    # Find the cache entry and corrupt the sentinel.
    sha = _cache_key(str(tmp_path), "train", 1800, None, 10, 16, None, False)
    sentinel = cache_root / sha / SENTINEL_NAME
    sentinel.write_text("not valid json {")
    # Reload — the loader rebuilds.
    rebuilt = load_lichess_corpus(
        str(tmp_path),
        split="train",
        elo_min=1800,
        seq_len=16,
        cache_dir=cache_root,
    )
    assert np.array_equal(corpus.tokens, rebuilt.tokens)


def test_cache_detects_tampering_of_corpus_file(tmp_path: Path) -> None:
    """Direct corruption of corpus.safetensors causes the load helper
    to raise — this is the integrity contract the sentinel provides.
    The public `load_lichess_corpus` would rebuild on the same input;
    `_load_from_cache` raises so callers see the failure mode."""
    _write_parquet(
        tmp_path / "train-0.parquet",
        _synthetic_rows(3, base_elo=1800, game_length=10),
    )
    cache_root = tmp_path / "cache"
    load_lichess_corpus(
        str(tmp_path), split="train", elo_min=1800, seq_len=16,
        cache_dir=cache_root,
    )
    sha = _cache_key(str(tmp_path), "train", 1800, None, 10, 16, None, False)
    payload = cache_root / sha / "corpus.safetensors"
    data = payload.read_bytes()
    payload.write_bytes(data[:-32] + b"\x00" * 32)
    with pytest.raises(CheckpointIntegrityError):
        _load_from_cache(cache_root / sha)


def test_cache_dir_default_uses_hf_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`$HF_HOME/pawn-lichess-cache` is the default cache root."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    root = _default_cache_root()
    assert root == tmp_path / "pawn-lichess-cache"


def test_cache_dir_default_falls_back_to_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    root = _default_cache_root()
    assert root.name == "pawn-lichess-cache"
    assert "huggingface" in root.parts


# ---------------------------------------------------------------------------
# End-to-end load_lichess_corpus (synthetic)
# ---------------------------------------------------------------------------


def test_load_lichess_corpus_end_to_end(tmp_path: Path) -> None:
    """Full path: synthetic parquet → filter → pack → cache → reload."""
    _write_parquet(
        tmp_path / "train-0.parquet",
        _synthetic_rows(5, base_elo=1800, game_length=12),
    )
    cache_root = tmp_path / "cache"
    corpus = load_lichess_corpus(
        str(tmp_path),
        split="train",
        elo_min=1800,
        elo_max=2000,
        min_ply=10,
        seq_len=16,
        cache_dir=cache_root,
    )
    assert corpus.n_games == 5
    assert corpus.seq_len == 16
    # Each game has 12 moves so attn_mask covers 12 positions.
    assert int(corpus.attn_mask[0].sum()) == 12


def test_load_lichess_corpus_rejects_empty_filter_result(tmp_path: Path) -> None:
    """An over-restrictive filter that selects 0 games surfaces a clear
    error rather than crashing later in pack_corpus."""
    _write_parquet(
        tmp_path / "train-0.parquet",
        _synthetic_rows(3, base_elo=1500, game_length=12),
    )
    with pytest.raises(ValueError, match="0 games"):
        load_lichess_corpus(
            str(tmp_path),
            split="train",
            elo_min=9000,
            elo_max=10000,
            seq_len=16,
            cache_dir=tmp_path / "cache",
        )


def test_load_lichess_corpus_second_call_hits_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache-hit branch is the primary performance guarantee — a
    second call with the same params must skip `_filter_lichess`
    entirely. Verified by monkeypatching the filter to raise after the
    first call lands the cache."""
    _write_parquet(
        tmp_path / "train-0.parquet",
        _synthetic_rows(3, base_elo=1800, game_length=12),
    )
    cache_root = tmp_path / "cache"
    first = load_lichess_corpus(
        str(tmp_path),
        split="train",
        elo_min=1800,
        elo_max=None,
        min_ply=10,
        seq_len=16,
        cache_dir=cache_root,
    )

    # Any subsequent call to _filter_lichess MUST not happen.
    import pawn.lichess_data

    def boom(*args, **kwargs):
        raise AssertionError("filter ran on cache hit — perf regression")

    monkeypatch.setattr(pawn.lichess_data, "_filter_lichess", boom)
    second = load_lichess_corpus(
        str(tmp_path),
        split="train",
        elo_min=1800,
        elo_max=None,
        min_ply=10,
        seq_len=16,
        cache_dir=cache_root,
    )
    assert np.array_equal(first.tokens, second.tokens)
    assert second.n_games == first.n_games


def test_load_lichess_corpus_validation_excludes_train_shards(
    tmp_path: Path,
) -> None:
    """When BOTH train-*.parquet and validation-*.parquet exist in a
    dir, `split="validation"` reads only the validation shards (no
    train spillover)."""
    _write_parquet(tmp_path / "train-0.parquet", _synthetic_rows(5, base_elo=1500))
    _write_parquet(
        tmp_path / "validation-0.parquet", _synthetic_rows(3, base_elo=2000)
    )
    val_corpus = load_lichess_corpus(
        str(tmp_path),
        split="validation",
        elo_min=None,
        min_ply=0,
        seq_len=16,
        cache_dir=tmp_path / "cache",
    )
    # 3 validation rows, not 5+3=8.
    assert val_corpus.n_games == 3


def test_pack_corpus_truncates_long_games_with_outcome_prefix(tmp_path: Path) -> None:
    """outcome_prefixed truncation: gl=20, seq_len=8 → outcome at slot
    0, first 7 moves at slots 1..7, loss_mask covers all 8 positions."""
    rows = [
        {"tokens": list(range(1, 21)), "game_length": 20,
         "outcome_token": WHITE_CHECKMATES,
         "white_elo": 1800, "black_elo": 1800}
    ]
    _write_parquet(tmp_path / "train-0.parquet", rows)
    corpus = load_lichess_corpus(
        str(tmp_path),
        split="train",
        elo_min=1800,
        min_ply=0,
        seq_len=8,
        prepend_outcome=True,
        cache_dir=tmp_path / "cache",
    )
    assert corpus.tokens[0, 0] == WHITE_CHECKMATES
    assert corpus.tokens[0, 1] == 1
    assert corpus.tokens[0, 7] == 7
    # outcome + 7 moves = 8 supervised positions.
    assert int(corpus.loss_mask[0].sum()) == 8
    assert int(corpus.outcome_offset[0]) == 1


# ---------------------------------------------------------------------------
# make_epoch_schedule
# ---------------------------------------------------------------------------


def test_make_epoch_schedule_tiles_across_epochs() -> None:
    """A pool of 10 with 25 needed → 3 epochs (10+10+5) of permuted
    indices."""
    sched = make_epoch_schedule(n_pool=10, n_needed=25, seed=0)
    assert sched.shape == (25,)
    assert sched.dtype == np.int64
    # Each epoch chunk is a permutation of [0, 10).
    epoch0 = sched[:10]
    epoch1 = sched[10:20]
    last5 = sched[20:25]
    assert set(epoch0.tolist()) == set(range(10))
    assert set(epoch1.tolist()) == set(range(10))
    # Successive epochs use different seeds → different perms.
    assert not np.array_equal(epoch0, epoch1)
    # Last 5 are a prefix of epoch 2's permutation.
    rng = np.random.default_rng(2)
    expected = rng.permutation(10).astype(np.int64)[:5]
    assert np.array_equal(last5, expected)


def test_make_epoch_schedule_seed_reproducible() -> None:
    a = make_epoch_schedule(n_pool=20, n_needed=50, seed=42)
    b = make_epoch_schedule(n_pool=20, n_needed=50, seed=42)
    assert np.array_equal(a, b)


def test_make_epoch_schedule_zero_needed_returns_empty() -> None:
    sched = make_epoch_schedule(n_pool=10, n_needed=0, seed=0)
    assert sched.shape == (0,)
    assert sched.dtype == np.int64


def test_make_epoch_schedule_rejects_zero_pool() -> None:
    with pytest.raises(ValueError, match="n_pool"):
        make_epoch_schedule(n_pool=0, n_needed=5, seed=0)


def test_make_epoch_schedule_rejects_negative_needed() -> None:
    with pytest.raises(ValueError, match="n_needed"):
        make_epoch_schedule(n_pool=10, n_needed=-1, seed=0)


def test_make_epoch_schedule_exact_epoch_boundary() -> None:
    """When n_needed is a multiple of n_pool there's no truncation."""
    sched = make_epoch_schedule(n_pool=10, n_needed=20, seed=0)
    assert sched.shape == (20,)
    # Both 10-chunks are full permutations.
    assert set(sched[:10].tolist()) == set(range(10))
    assert set(sched[10:].tolist()) == set(range(10))


def test_make_epoch_schedule_n_needed_smaller_than_pool() -> None:
    """A schedule that needs less than one epoch gets a truncated
    permutation."""
    sched = make_epoch_schedule(n_pool=100, n_needed=10, seed=0)
    assert sched.shape == (10,)
    rng = np.random.default_rng(0)
    expected = rng.permutation(100).astype(np.int64)[:10]
    assert np.array_equal(sched, expected)
