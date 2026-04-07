"""Tests for pawn/shard_loader.py.

Covers _is_local_path, _hf_storage_options, _process_shard_tokens, and
ShardedLichessDataset iteration (with local parquet shards).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from pawn.config import WHITE_CHECKMATES, OUTCOME_TOKEN_BASE
from pawn.shard_loader import (
    ShardedLichessDataset,
    _hf_storage_options,
    _is_local_path,
    _list_shards,
    _process_shard_tokens,
)


# ---------------------------------------------------------------------------
# _is_local_path
# ---------------------------------------------------------------------------


class TestIsLocalPath:
    @pytest.mark.unit
    def test_existing_dir_is_local(self, tmp_path):
        assert _is_local_path(str(tmp_path))

    @pytest.mark.unit
    def test_nonexistent_is_not_local(self):
        assert not _is_local_path("/nonexistent/path/xyz")

    @pytest.mark.unit
    def test_hf_style_repo_not_local(self):
        assert not _is_local_path("user/dataset")

    @pytest.mark.unit
    def test_file_is_not_local_dir(self, tmp_path):
        """_is_local_path checks for directory, not file."""
        f = tmp_path / "foo.txt"
        f.write_text("hello")
        # It's a file, not a dir
        assert not _is_local_path(str(f))


# ---------------------------------------------------------------------------
# _hf_storage_options
# ---------------------------------------------------------------------------


class TestHfStorageOptions:
    @pytest.mark.unit
    def test_returns_token_when_env_set(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test_token_123")
        opts = _hf_storage_options()
        assert opts == {"token": "test_token_123"}

    @pytest.mark.unit
    def test_empty_dict_when_no_token(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        # Also redirect home so cached token file isn't found
        monkeypatch.setenv("HOME", str(tmp_path))
        opts = _hf_storage_options()
        assert opts == {}

    @pytest.mark.unit
    def test_falls_back_to_cached_token(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        token_path = tmp_path / ".cache" / "huggingface" / "token"
        token_path.parent.mkdir(parents=True)
        token_path.write_text("cached_tok\n")
        opts = _hf_storage_options()
        assert opts == {"token": "cached_tok"}


# ---------------------------------------------------------------------------
# _process_shard_tokens
# ---------------------------------------------------------------------------


class TestProcessShardTokens:
    @pytest.mark.unit
    def test_basic_shapes(self):
        token_lists = [
            [WHITE_CHECKMATES, 10, 20, 30],  # outcome + 3 moves
            [WHITE_CHECKMATES + 1, 11, 22],  # outcome + 2 moves
        ]
        out = _process_shard_tokens(token_lists, max_ply=8, seq_len=9)
        assert out["input_ids"].shape == (2, 9)
        assert out["targets"].shape == (2, 9)
        assert out["loss_mask"].shape == (2, 9)
        assert out["move_ids"].shape == (2, 8)
        assert out["game_lengths"].shape == (2,)

    @pytest.mark.unit
    def test_outcome_at_position_zero(self):
        token_lists = [[WHITE_CHECKMATES, 10, 20]]
        out = _process_shard_tokens(token_lists, max_ply=8, seq_len=9)
        assert out["input_ids"][0, 0].item() == WHITE_CHECKMATES
        assert out["input_ids"][0, 1].item() == 10
        assert out["input_ids"][0, 2].item() == 20
        assert out["input_ids"][0, 3].item() == 0  # PAD after game end

    @pytest.mark.unit
    def test_targets_shift_left(self):
        token_lists = [[WHITE_CHECKMATES, 10, 20, 30]]
        out = _process_shard_tokens(token_lists, max_ply=8, seq_len=9)
        T = out["targets"].shape[1]
        for t in range(T - 1):
            assert out["targets"][0, t] == out["input_ids"][0, t + 1]
        assert out["targets"][0, T - 1] == 0

    @pytest.mark.unit
    def test_game_lengths_correct(self):
        token_lists = [
            [WHITE_CHECKMATES, 10, 20, 30],  # 3 moves
            [WHITE_CHECKMATES, 1, 2, 3, 4, 5],  # 5 moves
        ]
        out = _process_shard_tokens(token_lists, max_ply=8, seq_len=9)
        assert out["game_lengths"][0] == 3
        assert out["game_lengths"][1] == 5

    @pytest.mark.unit
    def test_move_ids_excludes_outcome(self):
        token_lists = [[WHITE_CHECKMATES, 10, 20, 30]]
        out = _process_shard_tokens(token_lists, max_ply=8, seq_len=9)
        # move_ids[0] should start with [10, 20, 30, 0, 0, ...]
        assert out["move_ids"][0, 0] == 10
        assert out["move_ids"][0, 1] == 20
        assert out["move_ids"][0, 2] == 30
        assert out["move_ids"][0, 3] == 0

    @pytest.mark.unit
    def test_loss_mask_boundary(self):
        token_lists = [[WHITE_CHECKMATES, 10, 20, 30]]  # 3 moves → game_length=3
        out = _process_shard_tokens(token_lists, max_ply=8, seq_len=9)
        # loss_mask: positions 0..3 True (outcome + 3 moves)
        assert out["loss_mask"][0, :4].all()
        assert not out["loss_mask"][0, 4:].any()

    @pytest.mark.unit
    def test_max_ply_clipping(self):
        """Tokens beyond max_ply are clipped."""
        # 10 moves, but max_ply=4
        toks = [WHITE_CHECKMATES] + [i + 1 for i in range(10)]
        out = _process_shard_tokens([toks], max_ply=4, seq_len=5)
        # n_moves = min(10, 4) = 4
        assert out["game_lengths"][0] == 4
        # input_ids has seq_len=5 slots: [outcome, m1, m2, m3, m4]
        assert out["input_ids"][0, 0].item() == WHITE_CHECKMATES
        assert out["input_ids"][0, 1].item() == 1
        assert out["input_ids"][0, 4].item() == 4


# ---------------------------------------------------------------------------
# _list_shards (local path)
# ---------------------------------------------------------------------------


class TestListShards:
    @pytest.mark.unit
    def test_lists_local_train_shards(self, tmp_path):
        # Create fake parquet file layout
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "train-00000.parquet").touch()
        (tmp_path / "data" / "train-00001.parquet").touch()
        (tmp_path / "data" / "validation-00000.parquet").touch()

        shards = _list_shards(str(tmp_path), split="train")
        assert len(shards) == 2
        assert all("train-" in s for s in shards)

    @pytest.mark.unit
    def test_lists_local_validation_shards(self, tmp_path):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "train-00000.parquet").touch()
        (tmp_path / "data" / "validation-00000.parquet").touch()
        shards = _list_shards(str(tmp_path), split="validation")
        assert len(shards) == 1
        assert "validation-" in shards[0]

    @pytest.mark.unit
    def test_returns_sorted(self, tmp_path):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "train-00002.parquet").touch()
        (tmp_path / "data" / "train-00000.parquet").touch()
        (tmp_path / "data" / "train-00001.parquet").touch()
        shards = _list_shards(str(tmp_path), split="train")
        assert shards == sorted(shards)


# ---------------------------------------------------------------------------
# ShardedLichessDataset (local parquet path)
# ---------------------------------------------------------------------------


def _write_test_shard(path: Path, n_games: int, game_length: int = 10,
                     white_elo: int = 1800, black_elo: int = 1800) -> None:
    """Create a test parquet shard with v2-format tokens (outcome + moves)."""
    token_lists = []
    for i in range(n_games):
        # outcome + game_length moves
        tokens = [WHITE_CHECKMATES + (i % 2)] + [1 + ((i + j) % 100) for j in range(game_length)]
        token_lists.append(tokens)
    df = pl.DataFrame({
        "tokens": token_lists,
        "game_length": [game_length] * n_games,
        "white_elo": [white_elo] * n_games,
        "black_elo": [black_elo] * n_games,
        "result": ["1-0"] * n_games,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


class TestShardedDataset:
    @pytest.fixture
    def local_shard_dir(self, tmp_path):
        """Create a directory with 2 local train shards + 1 val shard."""
        data_dir = tmp_path / "data"
        _write_test_shard(data_dir / "train-00000.parquet", n_games=4, game_length=8)
        _write_test_shard(data_dir / "train-00001.parquet", n_games=3, game_length=8)
        _write_test_shard(data_dir / "validation-00000.parquet", n_games=2, game_length=8)
        return str(tmp_path)

    @pytest.mark.integration
    def test_iterates_over_all_shards(self, local_shard_dir):
        ds = ShardedLichessDataset(
            local_shard_dir, split="train",
            min_ply=0, max_ply=16, shuffle_shards=False,
            shuffle_buffer=100,
        )
        items = list(ds)
        assert len(items) == 7  # 4 + 3

    @pytest.mark.integration
    def test_yields_dict_items(self, local_shard_dir):
        ds = ShardedLichessDataset(
            local_shard_dir, split="train",
            min_ply=0, max_ply=16, shuffle_shards=False,
        )
        it = iter(ds)
        item = next(it)
        assert set(item.keys()) >= {
            "input_ids", "targets", "loss_mask", "move_ids", "game_length"
        }
        assert item["input_ids"].shape == (17,)  # max_ply + 1

    @pytest.mark.integration
    def test_max_games_limit(self, local_shard_dir):
        ds = ShardedLichessDataset(
            local_shard_dir, split="train",
            min_ply=0, max_ply=16, shuffle_shards=False,
            max_games=3,
        )
        items = list(ds)
        assert len(items) == 3

    @pytest.mark.integration
    def test_validation_split(self, local_shard_dir):
        ds = ShardedLichessDataset(
            local_shard_dir, split="validation",
            min_ply=0, max_ply=16, shuffle_shards=False,
        )
        items = list(ds)
        assert len(items) == 2

    @pytest.mark.integration
    def test_no_shards_raises(self, tmp_path):
        (tmp_path / "data").mkdir()
        # No parquet files
        with pytest.raises(FileNotFoundError, match="No train parquet shards"):
            ShardedLichessDataset(
                str(tmp_path), split="train",
                min_ply=0, max_ply=16,
            )

    @pytest.mark.integration
    def test_elo_filter(self, tmp_path):
        """Elo range filter should drop games outside the range."""
        data_dir = tmp_path / "data"
        _write_test_shard(data_dir / "train-00000.parquet", n_games=4,
                          game_length=8, white_elo=1500, black_elo=1500)
        _write_test_shard(data_dir / "train-00001.parquet", n_games=3,
                          game_length=8, white_elo=2000, black_elo=2000)
        ds = ShardedLichessDataset(
            str(tmp_path), split="train",
            min_ply=0, max_ply=16, shuffle_shards=False,
            elo_min=1800, elo_max=2100,
        )
        items = list(ds)
        # Only the 2000-elo shard (3 games) passes
        assert len(items) == 3

    @pytest.mark.integration
    def test_set_epoch_mutates_epoch(self, local_shard_dir):
        ds = ShardedLichessDataset(
            local_shard_dir, split="train",
            min_ply=0, max_ply=16,
        )
        ds.set_epoch(7)
        assert ds._epoch == 7

    @pytest.mark.integration
    def test_min_ply_filter(self, tmp_path):
        """Games shorter than min_ply should be dropped."""
        data_dir = tmp_path / "data"
        _write_test_shard(data_dir / "train-00000.parquet", n_games=4, game_length=5)
        _write_test_shard(data_dir / "train-00001.parquet", n_games=3, game_length=20)
        ds = ShardedLichessDataset(
            str(tmp_path), split="train",
            min_ply=10, max_ply=32, shuffle_shards=False,
        )
        items = list(ds)
        # Only the 20-ply shard passes
        assert len(items) == 3

    @pytest.mark.integration
    def test_tokens_are_v2_format(self, local_shard_dir):
        """Item[0] in input_ids should be an outcome token (>= 4273)."""
        ds = ShardedLichessDataset(
            local_shard_dir, split="train",
            min_ply=0, max_ply=16, shuffle_shards=False,
        )
        for item in ds:
            tok = int(item["input_ids"][0])
            assert tok >= OUTCOME_TOKEN_BASE


# ---------------------------------------------------------------------------
# prefetch_shards smoke test (local-only; no mocking HF)
# ---------------------------------------------------------------------------


class TestPrefetchShards:
    @pytest.mark.integration
    def test_local_path_bypasses_prefetch(self, tmp_path):
        """If hf_repo is already a local dir, dataset should load directly."""
        data_dir = tmp_path / "data"
        _write_test_shard(data_dir / "train-00000.parquet", n_games=2, game_length=8)
        # cache_dir provided but repo is local — should not try to prefetch
        ds = ShardedLichessDataset(
            str(tmp_path), split="train",
            min_ply=0, max_ply=16, shuffle_shards=False,
            cache_dir=str(tmp_path / "cache"),  # not used for local
        )
        items = list(ds)
        assert len(items) == 2
