"""Shard-parallel IterableDataset for HuggingFace Lichess data.

Each DataLoader worker scans individual parquet shards via Polars hf://
protocol, avoiding the rate limits caused by globbing all 289 shards at
once. Workers partition shards among themselves and lazily load one shard
at a time.

Usage:
    dataset = ShardedLichessDataset(
        "thomas-schweich/pawn-lichess-full",
        elo_min=1800, elo_max=1900,
    )
    loader = DataLoader(dataset, batch_size=64, num_workers=4)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.utils.data

from pawn.config import OUTCOME_TOKEN_BASE, PAD_TOKEN


def _list_shards(hf_repo: str, split: str = "train") -> list[str]:
    """List parquet shard filenames for a split in an HF dataset repo."""
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(hf_repo, repo_type="dataset")
    return sorted(
        f for f in files
        if f.endswith(".parquet") and f"/{split}-" in f"/{f}"
    )


def _hf_storage_options() -> dict[str, str]:
    """Get HF token for authenticated Polars scans."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()
    return {"token": token} if token else {}


def _process_shard_tokens(
    token_lists: list[list[int]],
    max_ply: int,
    seq_len: int,
) -> dict[str, torch.Tensor | np.ndarray]:
    """Convert a list of token sequences into training tensors.

    Handles v2 format (outcome token prepended). Returns a dict with
    input_ids, targets, loss_mask, move_ids, game_lengths for one shard.
    """
    N = len(token_lists)
    input_ids = torch.zeros(N, seq_len, dtype=torch.long)
    game_lengths = np.zeros(N, dtype=np.int16)
    move_ids = np.zeros((N, max_ply), dtype=np.int16)

    for i, toks in enumerate(token_lists):
        n_moves = min(len(toks) - 1, max_ply)
        game_lengths[i] = n_moves
        length = min(len(toks), seq_len)
        input_ids[i, :length] = torch.tensor(toks[:length], dtype=torch.long)
        move_ids[i, :n_moves] = toks[1:n_moves + 1]

    capped_lengths = torch.from_numpy(game_lengths).long().clamp(max=max_ply)
    targets = torch.zeros(N, seq_len, dtype=torch.long)
    targets[:, :-1] = input_ids[:, 1:]
    positions = torch.arange(seq_len).unsqueeze(0)
    loss_mask = positions <= capped_lengths.unsqueeze(1)

    return {
        "input_ids": input_ids,
        "targets": targets,
        "loss_mask": loss_mask,
        "move_ids": move_ids,
        "game_lengths": game_lengths,
    }


class ShardedLichessDataset(torch.utils.data.IterableDataset):
    """IterableDataset that scans HF parquet shards one at a time per worker.

    Each DataLoader worker gets a disjoint partition of shard URLs and
    lazily loads them via Polars hf:// — one shard per HTTP request,
    no glob, no rate limits.
    """

    def __init__(
        self,
        hf_repo: str,
        split: str = "train",
        elo_min: int | None = None,
        elo_max: int | None = None,
        min_ply: int = 10,
        max_ply: int = 255,
        max_games: int | None = None,
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        self.hf_repo = hf_repo
        self.split = split
        self.elo_min = elo_min
        self.elo_max = elo_max
        self.min_ply = min_ply
        self.max_ply = max_ply
        self.seq_len = max_ply + 1
        self.max_games = max_games
        self.seed = seed

        self.shard_files = _list_shards(hf_repo, split)
        if not self.shard_files:
            raise FileNotFoundError(
                f"No {split} parquet shards found in {hf_repo}"
            )

        if shuffle_shards:
            import random
            rng = random.Random(seed)
            rng.shuffle(self.shard_files)

        self._storage_options = _hf_storage_options()

    def _build_filter(self) -> pl.Expr | None:
        """Build a Polars filter expression for Elo + min_ply."""
        filters = []
        if self.elo_min is not None:
            filters.append(
                (pl.col("white_elo") >= self.elo_min)
                & (pl.col("black_elo") >= self.elo_min)
            )
        if self.elo_max is not None:
            filters.append(
                (pl.col("white_elo") < self.elo_max)
                & (pl.col("black_elo") < self.elo_max)
            )
        if self.min_ply > 0:
            filters.append(pl.col("game_length") >= self.min_ply)
        return pl.all_horizontal(filters) if filters else None

    def _worker_shards(self) -> list[str]:
        """Partition shards across DataLoader workers."""
        info = torch.utils.data.get_worker_info()
        if info is None:
            return self.shard_files
        # Round-robin partition
        return self.shard_files[info.id::info.num_workers]

    def __iter__(self):
        shards = self._worker_shards()
        filt = self._build_filter()
        games_yielded = 0

        for shard_file in shards:
            if self.max_games and games_yielded >= self.max_games:
                return

            url = f"hf://datasets/{self.hf_repo}/{shard_file}"
            try:
                lf = pl.scan_parquet(
                    url, storage_options=self._storage_options or None,
                )
                if filt is not None:
                    lf = lf.filter(filt)
                df = lf.select(["tokens", "game_length"]).collect()
            except Exception as e:
                print(f"  Warning: failed to load shard {shard_file}: {e}")
                continue

            if len(df) == 0:
                continue

            token_lists = df["tokens"].to_list()
            batch = _process_shard_tokens(
                token_lists, self.max_ply, self.seq_len,
            )

            # Yield individual games from this shard
            n = len(token_lists)
            for i in range(n):
                if self.max_games and games_yielded >= self.max_games:
                    return
                yield {
                    "input_ids": batch["input_ids"][i],
                    "targets": batch["targets"][i],
                    "loss_mask": batch["loss_mask"][i],
                    "move_ids": batch["move_ids"][i],
                    "game_length": int(batch["game_lengths"][i]),
                }
                games_yielded += 1


def load_val_shards(
    hf_repo: str,
    elo_min: int | None = None,
    elo_max: int | None = None,
    min_ply: int = 10,
    max_ply: int = 255,
    max_games: int = 50_000,
) -> dict:
    """Load validation data eagerly (small, needs to be stable across epochs).

    Returns a dict compatible with LichessDataset.
    """
    shard_files = _list_shards(hf_repo, "validation")
    if not shard_files:
        raise FileNotFoundError(f"No validation shards found in {hf_repo}")

    storage_opts = _hf_storage_options()
    seq_len = max_ply + 1

    all_tokens: list[list[int]] = []
    for shard_file in shard_files:
        if len(all_tokens) >= max_games:
            break
        url = f"hf://datasets/{hf_repo}/{shard_file}"
        lf = pl.scan_parquet(url, storage_options=storage_opts or None)

        filters = []
        if elo_min is not None:
            filters.append(
                (pl.col("white_elo") >= elo_min)
                & (pl.col("black_elo") >= elo_min)
            )
        if elo_max is not None:
            filters.append(
                (pl.col("white_elo") < elo_max)
                & (pl.col("black_elo") < elo_max)
            )
        if min_ply > 0:
            filters.append(pl.col("game_length") >= min_ply)
        if filters:
            lf = lf.filter(pl.all_horizontal(filters))

        remaining = max_games - len(all_tokens)
        df = lf.select(["tokens", "game_length"]).head(remaining).collect()
        all_tokens.extend(df["tokens"].to_list())

    print(f"  Validation: {len(all_tokens):,} games from {len(shard_files)} shards")

    result: dict = _process_shard_tokens(all_tokens, max_ply, seq_len)
    result["n_games"] = len(all_tokens)
    return result
