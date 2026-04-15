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
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.utils.data

from pawn.config import OUTCOME_TOKEN_BASE, PAD_TOKEN


def _is_local_path(path: str) -> bool:
    """Check if a path refers to a local directory rather than an HF repo."""
    return os.path.isdir(path)


def prefetch_shards(
    hf_repo: str,
    target_dir: str,
    elo_min: int | None = None,
    elo_max: int | None = None,
    min_ply: int = 10,
    splits: tuple[str, ...] = ("train", "validation"),
) -> str:
    """Download and filter HF parquet shards to a local directory.

    Builds a filter-specific subdirectory name so different Elo bands
    don't collide. Skips shards that already exist locally. Returns
    the path to use as --pgn.
    """
    # Build a cache key from filter params
    parts = [hf_repo.replace("/", "_")]
    if elo_min is not None:
        parts.append(f"elo{elo_min}")
    if elo_max is not None:
        parts.append(f"-{elo_max}")
    cache_dir = Path(target_dir) / "_".join(parts)

    storage_opts = _hf_storage_options()

    for split in splits:
        shards = _list_shards(hf_repo, split)
        print(f"Prefetch: filtering {len(shards)} {split} shards "
              f"(elo={elo_min}-{elo_max}, min_ply={min_ply})...", flush=True)

        for i, shard in enumerate(shards):
            out_path = cache_dir / shard
            if out_path.exists():
                continue

            url = f"hf://datasets/{hf_repo}/{shard}"
            try:
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
                df = lf.collect()

                if len(df) > 0:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    df.write_parquet(out_path)
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(shards)}] {split}", flush=True)
            except Exception as e:
                print(f"  Warning: {shard}: {e}", flush=True)

        print(f"  {split} done", flush=True)

    print(f"Prefetch complete: {cache_dir}", flush=True)
    return str(cache_dir)


def _list_shards(hf_repo: str, split: str = "train") -> list[str]:
    """List parquet shard filenames for a split in an HF dataset repo or local dir."""
    if _is_local_path(hf_repo):
        local = Path(hf_repo)
        return sorted(
            str(f.relative_to(local))
            for f in local.rglob("*.parquet")
            if f"/{split}-" in f"/{f.name}"
        )
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
    game_lengths_col: list[int] | np.ndarray,
    outcome_tokens_col: list[int] | np.ndarray,
    max_ply: int,
    prepend_outcome: bool,
) -> dict[str, torch.Tensor | np.ndarray]:
    """Convert parquet shard rows into training tensors.

    Expects the pure-moves parquet layout written by
    ``scripts/extract_lichess_parquet.py``: ``token_lists[i]`` contains
    only the move tokens for game ``i``, and ``game_lengths_col[i]`` /
    ``outcome_tokens_col[i]`` come directly from the parquet columns of
    the same name.

    ``max_ply`` is the total output tensor width — matches the model's
    ``max_seq_len`` and the convention used by ``pack_clm_sequences``. In
    both modes the output has shape ``(N, max_ply)``; the difference is
    whether slot 0 holds the outcome (``prepend_outcome=True``) or the
    first move (``prepend_outcome=False``).
    """
    from pawn.data import pack_clm_sequences

    N = len(token_lists)
    # ``max_ply`` is the total budget; effective move cap is one lower
    # when the outcome occupies slot 0.
    effective_max_moves = max_ply - 1 if prepend_outcome else max_ply
    game_lengths = np.asarray(game_lengths_col, dtype=np.int16).copy()
    np.minimum(game_lengths, effective_max_moves, out=game_lengths)
    outcomes = torch.tensor(outcome_tokens_col, dtype=torch.long)

    move_ids = np.zeros((N, effective_max_moves), dtype=np.int16)
    for i, toks in enumerate(token_lists):
        n_moves = int(game_lengths[i])
        if n_moves > 0:
            move_ids[i, :n_moves] = toks[:n_moves]

    batch = pack_clm_sequences(
        move_ids, game_lengths, outcomes, max_ply,
        prepend_outcome=prepend_outcome,
    )

    return {
        "input_ids": batch["input_ids"],
        "targets": batch["targets"],
        "loss_mask": batch["loss_mask"],
        "move_ids": move_ids,
        "game_lengths": game_lengths,
    }


class ShardedLichessDataset(torch.utils.data.IterableDataset):
    """IterableDataset that scans HF parquet shards one at a time per worker.

    Each DataLoader worker gets a disjoint partition of shard URLs and
    lazily loads them via Polars hf:// — one shard per HTTP request,
    no glob, no rate limits.

    ``max_games`` is a **global** limit across all DataLoader workers.
    Each worker yields ``max_games // num_workers`` games, with the
    remainder distributed across workers 0..R-1 (one extra game each).
    When there are no workers, the full ``max_games`` budget applies.

    Shard order is re-shuffled at the start of every iteration using a
    deterministic seed derived from ``seed + epoch``. Call
    ``set_epoch(n)`` from the training loop before each epoch — this is
    required because DataLoader workers get forked copies of the dataset
    object (same pattern as ``DistributedSampler``). When
    ``shuffle_shards=False``, no reshuffling occurs.

    Games are accumulated in a shuffle buffer (default 50K games) and
    yielded in random order, mixing games across shards within each batch.
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
        shuffle_buffer: int = 50_000,
        seed: int = 42,
        cache_dir: str | None = None,
        prepend_outcome: bool = False,
    ):
        # If cache_dir is set and repo is remote, prefetch filtered shards
        if cache_dir and not _is_local_path(hf_repo):
            hf_repo = prefetch_shards(
                hf_repo, cache_dir,
                elo_min=elo_min, elo_max=elo_max, min_ply=min_ply,
                splits=(split,),
            )

        self.hf_repo = hf_repo
        self.split = split
        self.elo_min = elo_min
        self.elo_max = elo_max
        self.min_ply = min_ply
        # ``max_ply`` here is the total tensor-width budget (matches the
        # model's ``max_seq_len``); the outcome slot, if any, lives
        # inside this budget.
        self.max_ply = max_ply
        self.prepend_outcome = prepend_outcome
        self.seq_len = max_ply
        self.max_games = max_games
        self.shuffle_buffer_size = shuffle_buffer
        self.seed = seed

        self._local = _is_local_path(hf_repo)
        self.shard_files = _list_shards(hf_repo, split)
        if not self.shard_files:
            raise FileNotFoundError(
                f"No {split} parquet shards found in {hf_repo}"
            )

        if shuffle_shards:
            rng = random.Random(seed)
            rng.shuffle(self.shard_files)

        self.shuffle_shards = shuffle_shards
        self._epoch = 0
        self._storage_options = None if self._local else _hf_storage_options()

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shard reshuffling.

        Call this from the training loop before each epoch. Required because
        DataLoader workers (with persistent_workers=False) get forked copies
        of the dataset — auto-incrementing ``_epoch`` in the worker doesn't
        propagate back to the main process. This follows the same pattern as
        ``DistributedSampler.set_epoch()``.
        """
        self._epoch = epoch

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
        shards = list(self._worker_shards())
        rng = random.Random(self.seed + self._epoch)

        # Re-shuffle shards with an epoch-dependent seed so each epoch sees
        # a different order. Only when shuffle_shards=True (the default).
        if self.shuffle_shards:
            rng.shuffle(shards)

        # Compute per-worker game limit from the global max_games.
        worker_limit: int | None = None
        if self.max_games is not None:
            info = torch.utils.data.get_worker_info()
            if info is not None:
                base = self.max_games // info.num_workers
                remainder = self.max_games % info.num_workers
                worker_limit = base + (1 if info.id < remainder else 0)
            else:
                worker_limit = self.max_games

        filt = self._build_filter()
        games_yielded = 0
        shuffle = self.shuffle_shards
        buf_size = self.shuffle_buffer_size
        np_rng = np.random.default_rng(self.seed + self._epoch)

        # Accumulate shard batches as columnar arrays for vectorized shuffle
        buf_ids: list[Any] = []
        buf_tgt: list[Any] = []
        buf_mask: list[Any] = []
        buf_moves: list[Any] = []
        buf_lengths: list[Any] = []
        buf_n = 0

        def _flush_buf() -> Iterator[dict[str, Any]]:
            nonlocal buf_n
            ids = torch.cat(buf_ids)
            tgt = torch.cat(buf_tgt)
            mask = torch.cat(buf_mask)
            moves = np.concatenate(buf_moves)
            lengths = np.concatenate(buf_lengths)
            perm = np_rng.permutation(len(ids))
            for j in perm:
                yield {
                    "input_ids": ids[j],
                    "targets": tgt[j],
                    "loss_mask": mask[j],
                    "move_ids": moves[j],
                    "game_length": int(lengths[j]),
                }
            buf_ids.clear(); buf_tgt.clear(); buf_mask.clear()
            buf_moves.clear(); buf_lengths.clear()
            buf_n = 0

        for shard_file in shards:
            if worker_limit is not None and games_yielded >= worker_limit:
                break

            if self._local:
                source = str(Path(self.hf_repo) / shard_file)
            else:
                source = f"hf://datasets/{self.hf_repo}/{shard_file}"
            try:
                lf = pl.scan_parquet(
                    source, storage_options=self._storage_options or None,
                )
                if filt is not None:
                    lf = lf.filter(filt)
                df = lf.select(
                    ["tokens", "game_length", "outcome_token"],
                ).collect()
            except Exception as e:
                print(f"  Warning: failed to load shard {shard_file}: {e}")
                continue

            if len(df) == 0:
                continue

            token_lists = df["tokens"].to_list()
            batch = _process_shard_tokens(
                token_lists,
                df["game_length"].to_list(),
                df["outcome_token"].to_list(),
                self.max_ply,
                self.prepend_outcome,
            )

            n = len(token_lists)
            # Trim to worker limit
            if worker_limit is not None:
                n = min(n, worker_limit - games_yielded)

            if shuffle:
                buf_ids.append(batch["input_ids"][:n])
                buf_tgt.append(batch["targets"][:n])
                buf_mask.append(batch["loss_mask"][:n])
                buf_moves.append(batch["move_ids"][:n])
                buf_lengths.append(batch["game_lengths"][:n])
                buf_n += n
                games_yielded += n
                if buf_n >= buf_size:
                    yield from _flush_buf()
            else:
                for i in range(n):
                    yield {
                        "input_ids": batch["input_ids"][i],
                        "targets": batch["targets"][i],
                        "loss_mask": batch["loss_mask"][i],
                        "move_ids": batch["move_ids"][i],
                        "game_length": int(batch["game_lengths"][i]),
                    }
                    games_yielded += 1

        if buf_n > 0:
            yield from _flush_buf()


def load_val_shards(
    hf_repo: str,
    elo_min: int | None = None,
    elo_max: int | None = None,
    min_ply: int = 10,
    max_ply: int = 255,
    max_games: int = 50_000,
    cache_dir: str | None = None,
    prepend_outcome: bool = False,
) -> dict:
    """Load validation data eagerly (small, needs to be stable across epochs).

    ``prepend_outcome`` selects the training-tensor layout — default False
    (pure moves) matches the loader and trainer defaults.

    Returns a dict compatible with LichessDataset.
    """
    if cache_dir and not _is_local_path(hf_repo):
        hf_repo = prefetch_shards(
            hf_repo, cache_dir,
            elo_min=elo_min, elo_max=elo_max, min_ply=min_ply,
            splits=("validation",),
        )
    local = _is_local_path(hf_repo)
    shard_files = _list_shards(hf_repo, "validation")
    if not shard_files:
        raise FileNotFoundError(f"No validation shards found in {hf_repo}")

    storage_opts = None if local else _hf_storage_options()

    all_tokens: list[list[int]] = []
    all_game_lengths: list[int] = []
    all_outcome_tokens: list[int] = []
    for shard_file in shard_files:
        if len(all_tokens) >= max_games:
            break
        if local:
            source = str(Path(hf_repo) / shard_file)
        else:
            source = f"hf://datasets/{hf_repo}/{shard_file}"
        lf = pl.scan_parquet(source, storage_options=storage_opts or None)

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
        df = lf.select(
            ["tokens", "game_length", "outcome_token"],
        ).head(remaining).collect()
        all_tokens.extend(df["tokens"].to_list())
        all_game_lengths.extend(df["game_length"].to_list())
        all_outcome_tokens.extend(df["outcome_token"].to_list())

    print(f"  Validation: {len(all_tokens):,} games from {len(shard_files)} shards")

    result: dict = _process_shard_tokens(
        all_tokens, all_game_lengths, all_outcome_tokens, max_ply,
        prepend_outcome,
    )
    result["n_games"] = len(all_tokens)
    return result
