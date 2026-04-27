"""Tests for the persistent tokenized Lichess cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from pawn.config import OUTCOME_TOKEN_BASE
from pawn.lichess_cache import (
    IndexedLichessDataset,
    LichessTokenCache,
    SeededEpochSampler,
    _derive_key,
    _load_cache,
    default_cache_root,
    prepare_lichess_cached,
    shuffled_indices,
)


WHITE_CHECKMATES = OUTCOME_TOKEN_BASE  # 1969


def _write_shard(
    path: Path,
    n_games: int,
    game_length: int = 8,
    white_elo: int = 1850,
    black_elo: int = 1850,
) -> None:
    """Mirror the schema produced by ``scripts/extract_lichess_parquet.py``.

    Tokens are pure-moves; the outcome lives in its own column.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    token_lists = [
        [1 + ((i + j) % 100) for j in range(game_length)] for i in range(n_games)
    ]
    df = pl.DataFrame(
        {
            "tokens": pl.Series("tokens", token_lists, dtype=pl.List(pl.Int16)),
            "game_length": pl.Series(
                "game_length", [game_length] * n_games, dtype=pl.UInt16
            ),
            "outcome_token": pl.Series(
                "outcome_token",
                [WHITE_CHECKMATES + (i % 2) for i in range(n_games)],
                dtype=pl.UInt16,
            ),
            "white_elo": pl.Series(
                "white_elo", [white_elo] * n_games, dtype=pl.UInt16
            ),
            "black_elo": pl.Series(
                "black_elo", [black_elo] * n_games, dtype=pl.UInt16
            ),
            "result": pl.Series("result", ["1-0"] * n_games),
        }
    )
    df.write_parquet(path)


class TestDeriveKey:
    @pytest.mark.unit
    def test_repo_slug_in_key(self) -> None:
        k = _derive_key("foo/bar", "train", 1800, 1900, 10)
        assert k.startswith("foo--bar_train_e1800-1900_p10_")

    @pytest.mark.unit
    def test_parquet_path_uses_stem(self) -> None:
        k = _derive_key("/tmp/data/games.parquet", "train", None, None, 10)
        assert k.startswith("games_train_eany-any_p10_")

    @pytest.mark.unit
    def test_keys_differ_by_param(self) -> None:
        a = _derive_key("foo/bar", "train", 1800, 1900, 10)
        b = _derive_key("foo/bar", "train", 1800, 1900, 12)
        assert a != b

    @pytest.mark.unit
    def test_keys_match_for_same_params(self) -> None:
        a = _derive_key("foo/bar", "train", 1800, 1900, 10)
        b = _derive_key("foo/bar", "train", 1800, 1900, 10)
        assert a == b


class TestDefaultCacheRoot:
    @pytest.mark.unit
    def test_pawn_data_cache_wins(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("PAWN_DATA_CACHE", str(tmp_path / "explicit"))
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        assert default_cache_root() == (tmp_path / "explicit")

    @pytest.mark.unit
    def test_falls_back_to_hf_home(self, monkeypatch, tmp_path) -> None:
        monkeypatch.delenv("PAWN_DATA_CACHE", raising=False)
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        assert default_cache_root() == (tmp_path / "hf" / "pawn-lichess-cache")


class TestPrepareLichessCached:
    @pytest.mark.integration
    def test_build_and_reload_round_trips(self, tmp_path) -> None:
        shard = tmp_path / "data" / "train-00000.parquet"
        _write_shard(shard, n_games=5, game_length=6)

        cache = prepare_lichess_cached(
            str(shard), min_ply=0, cache_dir=str(tmp_path / "cache")
        )
        assert isinstance(cache, LichessTokenCache)
        assert cache.n_games == 5
        # 5 games × 6 plies each.
        assert cache.tokens_flat.shape == (30,)
        assert cache.offsets.shape == (6,)
        assert cache.outcome_tokens.shape == (5,)
        # offsets are CSR → first is 0, last is total_tokens.
        assert int(cache.offsets[0].item()) == 0
        assert int(cache.offsets[-1].item()) == 30

        # Reload uses the same cache directory (no rebuild).
        cache2 = prepare_lichess_cached(
            str(shard), min_ply=0, cache_dir=str(tmp_path / "cache")
        )
        assert cache2.n_games == 5
        torch.testing.assert_close(cache.offsets, cache2.offsets)
        torch.testing.assert_close(cache.outcome_tokens, cache2.outcome_tokens)

    @pytest.mark.integration
    def test_elo_filter_excludes_out_of_band(self, tmp_path) -> None:
        in_band = tmp_path / "data" / "train-00000.parquet"
        out_band = tmp_path / "data" / "train-00001.parquet"
        _write_shard(in_band, n_games=4, white_elo=1850, black_elo=1850)
        _write_shard(out_band, n_games=3, white_elo=1750, black_elo=1750)

        cache = prepare_lichess_cached(
            str(tmp_path / "data" / "train-*.parquet"),
            elo_min=1800,
            elo_max=1900,
            min_ply=0,
            cache_dir=str(tmp_path / "cache"),
        )
        assert cache.n_games == 4

    @pytest.mark.integration
    def test_min_ply_filters_short_games(self, tmp_path) -> None:
        # Two shards: one with games of length 3, one with games of length 12.
        short = tmp_path / "data" / "train-00000.parquet"
        long_ = tmp_path / "data" / "train-00001.parquet"
        _write_shard(short, n_games=4, game_length=3)
        _write_shard(long_, n_games=2, game_length=12)

        cache = prepare_lichess_cached(
            str(tmp_path / "data" / "train-*.parquet"),
            min_ply=10,
            cache_dir=str(tmp_path / "cache"),
        )
        assert cache.n_games == 2

    @pytest.mark.integration
    def test_no_games_pass_filter_raises(self, tmp_path) -> None:
        shard = tmp_path / "data" / "train-00000.parquet"
        _write_shard(shard, n_games=2, white_elo=1500, black_elo=1500)

        with pytest.raises(ValueError, match="No games passed"):
            prepare_lichess_cached(
                str(shard),
                elo_min=1800,
                elo_max=1900,
                min_ply=0,
                cache_dir=str(tmp_path / "cache"),
            )


class TestIndexedLichessDataset:
    @pytest.mark.integration
    def test_pure_moves_packing_shapes(self, tmp_path) -> None:
        shard = tmp_path / "data" / "train-00000.parquet"
        _write_shard(shard, n_games=3, game_length=4)

        cache = prepare_lichess_cached(
            str(shard), min_ply=0, cache_dir=str(tmp_path / "cache")
        )
        ds = IndexedLichessDataset(
            cache_dir=tmp_path
            / "cache"
            / _derive_key(str(shard), "train", None, None, 0),
            indices=torch.arange(cache.n_games),
            max_ply=16,
            prepend_outcome=False,
        )

        assert len(ds) == 3
        sample = ds[0]
        assert sample["input_ids"].shape == (16,)
        assert sample["targets"].shape == (16,)
        assert sample["loss_mask"].shape == (16,)
        assert sample["move_ids"].shape == (16,)
        assert sample["game_length"] == 4

    @pytest.mark.integration
    def test_lazy_pack_returns_raw_inputs(self, tmp_path) -> None:
        """``lazy_pack=True`` skips per-sample packing and returns the
        raw move row + ``game_length`` + ``outcome_token`` for the
        bucketed collate to pack at a per-batch ``T``."""
        shard = tmp_path / "data" / "train-00000.parquet"
        _write_shard(shard, n_games=3, game_length=4)
        cache = prepare_lichess_cached(
            str(shard), min_ply=0, cache_dir=str(tmp_path / "cache")
        )
        cache_path = tmp_path / "cache" / _derive_key(
            str(shard), "train", None, None, 0
        )
        ds = IndexedLichessDataset(
            cache_dir=cache_path,
            indices=torch.arange(cache.n_games),
            max_ply=16,
            prepend_outcome=False,
            lazy_pack=True,
        )
        sample = ds[0]
        # No pre-packed tensors — the bucketed collate handles those.
        assert "input_ids" not in sample
        assert "targets" not in sample
        assert "loss_mask" not in sample
        # Raw fields the collate needs.
        assert sample["move_ids"].shape == (16,)
        assert sample["game_length"] == 4
        assert isinstance(sample["outcome_token"], int)
        # Outcome token in valid range (PAD or any of the 11 outcomes).
        assert (
            OUTCOME_TOKEN_BASE <= sample["outcome_token"] < OUTCOME_TOKEN_BASE + 11
        )

    @pytest.mark.integration
    def test_outcome_prepend_packing(self, tmp_path) -> None:
        shard = tmp_path / "data" / "train-00000.parquet"
        _write_shard(shard, n_games=2, game_length=3)
        cache = prepare_lichess_cached(
            str(shard), min_ply=0, cache_dir=str(tmp_path / "cache")
        )
        cache_path = tmp_path / "cache" / _derive_key(
            str(shard), "train", None, None, 0
        )
        ds = IndexedLichessDataset(
            cache_dir=cache_path,
            indices=torch.arange(cache.n_games),
            max_ply=8,
            prepend_outcome=True,
        )
        sample = ds[0]
        # Outcome at slot 0 (in [1969, 1979]).
        outcome = int(sample["input_ids"][0].item())
        assert OUTCOME_TOKEN_BASE <= outcome < OUTCOME_TOKEN_BASE + 11
        # move_ids width = max_ply - 1 in outcome-prefixed mode.
        assert sample["move_ids"].shape == (7,)

    @pytest.mark.integration
    def test_default_collate_compatible(self, tmp_path) -> None:
        """The dataset must produce items that ``default_collate`` can
        stack — same contract as the existing ``LichessDataset``.
        ``LegalMaskCollate`` further computes legal indices via the Rust
        replay; we don't exercise that here because the synthetic test
        data isn't a legal chess game (it would panic in the replayer).
        End-to-end collate compatibility is covered by
        ``LegalMaskCollate``'s own tests on real games."""
        shard = tmp_path / "data" / "train-00000.parquet"
        _write_shard(shard, n_games=4, game_length=6)
        cache = prepare_lichess_cached(
            str(shard), min_ply=0, cache_dir=str(tmp_path / "cache")
        )
        cache_path = tmp_path / "cache" / _derive_key(
            str(shard), "train", None, None, 0
        )
        ds = IndexedLichessDataset(
            cache_dir=cache_path,
            indices=torch.arange(cache.n_games),
            max_ply=16,
            prepend_outcome=False,
        )
        items = [ds[i] for i in range(len(ds))]
        batch = torch.utils.data.default_collate(items)
        assert batch["input_ids"].shape == (4, 16)
        assert batch["targets"].shape == (4, 16)
        assert batch["loss_mask"].shape == (4, 16)
        assert batch["move_ids"].shape == (4, 16)
        assert batch["game_length"].shape == (4,)

    @pytest.mark.integration
    def test_subset_indices(self, tmp_path) -> None:
        shard = tmp_path / "data" / "train-00000.parquet"
        _write_shard(shard, n_games=10, game_length=4)
        cache = prepare_lichess_cached(
            str(shard), min_ply=0, cache_dir=str(tmp_path / "cache")
        )
        cache_path = tmp_path / "cache" / _derive_key(
            str(shard), "train", None, None, 0
        )
        ds = IndexedLichessDataset(
            cache_dir=cache_path,
            indices=torch.tensor([5, 2, 9]),
            max_ply=8,
            prepend_outcome=False,
        )
        assert len(ds) == 3

    @pytest.mark.integration
    def test_indices_out_of_range_raises(self, tmp_path) -> None:
        shard = tmp_path / "data" / "train-00000.parquet"
        _write_shard(shard, n_games=3, game_length=4)
        prepare_lichess_cached(
            str(shard), min_ply=0, cache_dir=str(tmp_path / "cache")
        )
        cache_path = tmp_path / "cache" / _derive_key(
            str(shard), "train", None, None, 0
        )
        with pytest.raises(ValueError, match="out of range"):
            IndexedLichessDataset(
                cache_dir=cache_path,
                indices=torch.tensor([0, 5]),  # 5 ≥ 3
                max_ply=8,
            )

    @pytest.mark.integration
    def test_pickle_round_trip_resets_cache(self, tmp_path) -> None:
        """Spawn workers serialize the dataset; make sure the (potentially
        large) cache is dropped from the pickle stream and is reloaded
        on first access in the worker."""
        import pickle

        shard = tmp_path / "data" / "train-00000.parquet"
        _write_shard(shard, n_games=2, game_length=4)
        cache = prepare_lichess_cached(
            str(shard), min_ply=0, cache_dir=str(tmp_path / "cache")
        )
        cache_path = tmp_path / "cache" / _derive_key(
            str(shard), "train", None, None, 0
        )
        ds = IndexedLichessDataset(
            cache_dir=cache_path,
            indices=torch.arange(cache.n_games),
            max_ply=8,
        )
        _ = ds[0]  # force-load
        assert ds._cache is not None

        blob = pickle.dumps(ds)
        ds2 = pickle.loads(blob)
        assert ds2._cache is None  # not pickled
        # Still works after unpickling.
        sample = ds2[0]
        assert sample["input_ids"].shape == (8,)


class TestShuffledIndices:
    @pytest.mark.unit
    def test_deterministic_for_same_seed_epoch(self) -> None:
        base = torch.arange(100)
        a = shuffled_indices(base, epoch=3, base_seed=7)
        b = shuffled_indices(base, epoch=3, base_seed=7)
        torch.testing.assert_close(a, b)

    @pytest.mark.unit
    def test_different_epochs_give_different_orderings(self) -> None:
        base = torch.arange(100)
        a = shuffled_indices(base, epoch=0, base_seed=42)
        b = shuffled_indices(base, epoch=1, base_seed=42)
        assert not torch.equal(a, b)

    @pytest.mark.unit
    def test_permutation_preserves_set(self) -> None:
        base = torch.arange(50)
        out = shuffled_indices(base, epoch=2, base_seed=99)
        assert set(out.tolist()) == set(base.tolist())

    @pytest.mark.unit
    def test_works_with_non_identity_base(self) -> None:
        # Exercise the case where base_indices is a sub-range, e.g. the
        # train split before per-epoch shuffle.
        base = torch.tensor([10, 20, 30, 40, 50])
        out = shuffled_indices(base, epoch=0, base_seed=1)
        assert set(out.tolist()) == {10, 20, 30, 40, 50}


class TestSeededEpochSampler:
    @pytest.mark.unit
    def test_yields_full_permutation(self) -> None:
        s = SeededEpochSampler(n_samples=100, base_seed=42, batch_size=10)
        s.set_epoch(0)
        out = list(s)
        assert len(out) == 100
        assert sorted(out) == list(range(100))

    @pytest.mark.unit
    def test_drop_last_truncates_to_full_batches(self) -> None:
        s = SeededEpochSampler(
            n_samples=23, base_seed=0, batch_size=5, drop_last=True
        )
        s.set_epoch(0)
        # 23 samples → 4 full batches × 5 = 20 indices.
        assert len(s) == 20
        # The iterator itself yields whatever the permutation is; consumers
        # apply ``drop_last`` via the DataLoader. We just need __len__
        # to match the DataLoader's contract.

    @pytest.mark.unit
    def test_skip_batches_slices_off_prefix(self) -> None:
        s = SeededEpochSampler(n_samples=20, base_seed=7, batch_size=4)
        s.set_epoch(0, skip_batches=2)
        out = list(s)
        # 20 - 2*4 = 12 indices remain.
        assert len(out) == 12
        # __len__ matches.
        assert len(s) == 12  # 12 % 4 == 0, drop_last has no effect.

    @pytest.mark.unit
    def test_resume_reproduces_remainder_exactly(self) -> None:
        # Run epoch 3 from scratch, capture the full permutation. Then
        # simulate "resumed at batch 2" and confirm the resumed iterator
        # produces the exact tail of the original permutation.
        full = SeededEpochSampler(n_samples=20, base_seed=99, batch_size=4)
        full.set_epoch(3)
        original = list(full)

        resumed = SeededEpochSampler(n_samples=20, base_seed=99, batch_size=4)
        resumed.set_epoch(3, skip_batches=2)
        tail = list(resumed)

        assert tail == original[2 * 4:]

    @pytest.mark.unit
    def test_epoch_change_yields_different_permutation(self) -> None:
        s = SeededEpochSampler(n_samples=50, base_seed=1, batch_size=10)
        s.set_epoch(0)
        a = list(s)
        s.set_epoch(1)
        b = list(s)
        assert a != b

    @pytest.mark.unit
    def test_set_epoch_validates_skip_bound(self) -> None:
        s = SeededEpochSampler(n_samples=10, base_seed=0, batch_size=4)
        with pytest.raises(ValueError, match="exceeds n_samples"):
            s.set_epoch(0, skip_batches=10)
        with pytest.raises(ValueError, match=">= 0"):
            s.set_epoch(0, skip_batches=-1)
