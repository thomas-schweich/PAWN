"""Lichess data preparation for FiLM behavioral cloning.

Parses a PGN file, tokenizes via the Rust engine, and produces PyTorch
tensors ready for training.  Legal move grids are computed per-batch during
training (not precomputed) to keep memory independent of dataset size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.utils.data

import chess_engine as engine



# ---------------------------------------------------------------------------
# Legal token mask via fused Rust computation
# ---------------------------------------------------------------------------


def compute_legal_indices(
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    seq_len: int,
    vocab_size: int = 1968,
) -> np.ndarray:
    """Compute flat sparse indices for legal token masks (CPU only).

    Calls the Rust engine to replay games and returns flat i64 indices
    suitable for scattering into a (B, seq_len, vocab_size) bool mask.
    """
    move_ids = np.ascontiguousarray(move_ids, dtype=np.int16)
    game_lengths = np.asarray(game_lengths, dtype=np.int16)
    return engine.compute_legal_token_masks_sparse(
        move_ids, game_lengths, seq_len, vocab_size,
    )


class LegalMaskBuilder:
    """Legal token mask via sparse Rust computation + GPU scatter.

    Calls engine.compute_legal_token_masks_sparse which replays games and
    returns flat i64 indices (~2 MB) instead of a dense bool mask (~70 MB).
    Indices are transferred to GPU and scattered into a pre-allocated buffer.

    Two usage modes:
      1. ``scatter(indices, B)`` — fast GPU-only path for pre-computed indices
         (from ``LegalMaskCollate`` or precomputation).
      2. ``__call__(batch)`` — legacy path that computes indices inline.
    """

    def __init__(self, batch_size: int, max_ply: int, vocab_size: int = 1968,
                 device: str = "cpu", max_index_buf: int = 4_000_000,
                 prepend_outcome: bool = False):
        self.vocab_size = vocab_size
        self.max_ply = max_ply
        # When prepend_outcome=True we reserve one extra slot for the
        # outcome token at position 0; pure-moves mode fits in max_ply.
        self.T = max_ply + 1 if prepend_outcome else max_ply
        self.device = device

        # Pre-allocated GPU output buffer
        self._mask_gpu = torch.zeros(batch_size, self.T, vocab_size,
                                     dtype=torch.bool, device=device)
        # Pre-allocated GPU index buffer to avoid per-batch allocation
        self._idx_buf = torch.empty(max_index_buf, dtype=torch.long, device=device)

    def scatter(self, legal_indices: torch.Tensor, B: int) -> torch.Tensor:
        """Scatter pre-computed CPU indices into the GPU mask buffer.

        Uses a pre-allocated index buffer to avoid per-batch GPU allocation.
        Falls back to a fresh allocation if the buffer is too small.
        """
        if B > self._mask_gpu.shape[0]:
            raise ValueError(
                f"B={B} exceeds pre-allocated batch_size={self._mask_gpu.shape[0]}"
            )
        mask_view = self._mask_gpu[:B]
        mask_view.zero_()
        n = legal_indices.shape[0]
        if n > 0:
            if n <= self._idx_buf.shape[0]:
                self._idx_buf[:n].copy_(legal_indices)
                mask_view.view(-1).index_fill_(0, self._idx_buf[:n], True)
            else:
                idx_gpu = legal_indices.to(self.device)
                mask_view.view(-1).index_fill_(0, idx_gpu, True)
        return mask_view

    def __call__(self, batch: dict) -> torch.Tensor:
        """Build (B, T, V) legal mask from batch move_ids + game_lengths.

        Computes sparse indices via Rust and scatters to the GPU buffer.
        For better performance, use ``LegalMaskCollate`` with DataLoader
        workers to compute indices off the critical path, then call
        ``scatter()`` directly.
        """
        move_ids = batch["move_ids"]
        game_lengths_raw = batch["game_length"]
        B = move_ids.shape[0] if hasattr(move_ids, 'shape') else len(move_ids)

        if isinstance(move_ids, torch.Tensor):
            move_ids = move_ids.numpy()
        move_ids = np.ascontiguousarray(move_ids, dtype=np.int16)
        game_lengths = np.asarray(game_lengths_raw, dtype=np.int16)

        indices = engine.compute_legal_token_masks_sparse(
            move_ids, game_lengths, self.T, self.vocab_size,
        )

        return self.scatter(torch.from_numpy(indices), B)


class LegalMaskCollate:
    """Collate that computes legal mask indices in DataLoader workers.

    Wraps default collation and appends a ``legal_indices`` CPU tensor
    to each batch so the Rust replay runs in worker processes, off the
    GPU training critical path.
    """

    def __init__(self, seq_len: int, vocab_size: int = 1968):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __call__(self, items: list[dict]) -> dict:
        batch = torch.utils.data.default_collate(items)
        move_ids = batch["move_ids"].numpy()
        game_lengths = np.asarray(batch["game_length"], dtype=np.int16)
        indices = compute_legal_indices(
            move_ids, game_lengths, self.seq_len, self.vocab_size,
        )
        batch["legal_indices"] = torch.from_numpy(indices)
        return batch


# ---------------------------------------------------------------------------
# PGN → tokenized dataset with legal move masks
# ---------------------------------------------------------------------------


def prepare_lichess_dataset(
    pgn_path: str | Path,
    max_ply: int = 255,
    max_games: int = 50_000,
    min_ply: int = 10,
    elo_min: int | None = None,
    elo_max: int | None = None,
    prepend_outcome: bool = False,
) -> dict:
    """Load a pre-tokenized Lichess dataset and produce training-ready tensors.

    ``pgn_path`` accepts a local ``.parquet`` file path or a HuggingFace
    dataset repo ID (contains ``/``). Raw PGN files are not supported —
    convert them with ``scripts/extract_lichess_parquet.py`` first.

    Returns dict with:
        move_ids:       (N, max_ply) int16 — tokenized moves
        game_lengths:   (N,) int16
        input_ids:      (N, seq_len) long
        targets:        (N, seq_len) long — shifted left
        loss_mask:      (N, seq_len) bool
        n_games:        int
    """
    pgn_path_str = str(pgn_path)
    if pgn_path_str.endswith(".parquet"):
        return prepare_lichess_parquet(
            parquet_path=pgn_path_str, max_ply=max_ply,
            max_games=max_games, min_ply=min_ply,
            elo_min=elo_min, elo_max=elo_max,
            prepend_outcome=prepend_outcome,
        )
    # Check if it looks like a HF repo ID (e.g. "user/dataset")
    if "/" in pgn_path_str and not Path(pgn_path_str).exists():
        return prepare_lichess_parquet(
            hf_repo=pgn_path_str, max_ply=max_ply,
            max_games=max_games, min_ply=min_ply,
            elo_min=elo_min, elo_max=elo_max,
            prepend_outcome=prepend_outcome,
        )
    raise ValueError(
        f"prepare_lichess_dataset accepts only .parquet files or HF dataset "
        f"repo IDs; got {pgn_path_str!r}. Convert raw PGN with "
        "scripts/extract_lichess_parquet.py first."
    )


def _scan_parquet(
    parquet_path: str | Path | None = None,
    hf_repo: str | None = None,
    split: str = "train",
) -> "pl.LazyFrame":
    """Create a Polars LazyFrame from a local Parquet file or HF dataset repo."""
    if hf_repo is not None:
        # Use hf:// protocol for direct lazy scanning from HuggingFace
        hf_url = f"hf://datasets/{hf_repo}/data/{split}-*.parquet"
        try:
            return pl.scan_parquet(hf_url)
        except Exception:
            # Fallback: download files and scan locally
            from huggingface_hub import hf_hub_download, HfApi
            api = HfApi()
            files = api.list_repo_files(hf_repo, repo_type="dataset")
            parquet_files = [
                f for f in files
                if f.endswith(".parquet") and f"/{split}-" in f"/{f}"
            ]
            if not parquet_files:
                # No split-specific files — try all parquet files
                parquet_files = [f for f in files if f.endswith(".parquet")]
            local_files = [
                hf_hub_download(hf_repo, pf, repo_type="dataset")
                for pf in parquet_files
            ]
            return pl.scan_parquet(local_files)
    elif parquet_path is not None:
        return pl.scan_parquet(str(parquet_path))
    else:
        raise ValueError("Either parquet_path or hf_repo must be provided")


def prepare_lichess_parquet(
    parquet_path: str | Path | None = None,
    hf_repo: str | None = None,
    max_ply: int = 255,
    max_games: int = 50_000,
    min_ply: int = 10,
    split: str = "train",
    elo_min: int | None = None,
    elo_max: int | None = None,
    prepend_outcome: bool = False,
) -> dict:
    """Load a Parquet dataset and produce training-ready tensors.

    The extractor writes one canonical schema: pure-moves ``tokens``,
    ``game_length``, ``outcome_token`` (Rust-classified), and per-game
    metadata columns. Parquets missing any of those columns raise an error —
    legacy parquet layouts were removed along with the legacy vocabulary.

    ``prepend_outcome=False`` (default) produces pure-moves tensors;
    ``prepend_outcome=True`` prepends the outcome token at slot 0 for
    outcome-conditioned training.

    Reads from a local Parquet file or a HuggingFace dataset repo.
    Returns the same dict format as ``prepare_lichess_dataset``.
    """
    lf = _scan_parquet(parquet_path, hf_repo, split)
    schema = lf.collect_schema()

    # Elo filtering (both players must be within range)
    if elo_min is not None or elo_max is not None:
        if "white_elo" not in schema or "black_elo" not in schema:
            raise ValueError(
                "Elo filtering requires white_elo/black_elo columns in Parquet schema, "
                f"got: {list(schema.names())}"
            )
        if elo_min is not None:
            lf = lf.filter(
                (pl.col("white_elo") >= elo_min) & (pl.col("black_elo") >= elo_min)
            )
        if elo_max is not None:
            lf = lf.filter(
                (pl.col("white_elo") < elo_max) & (pl.col("black_elo") < elo_max)
            )

    return _prepare_from_tokens(lf, max_ply, max_games, min_ply, prepend_outcome)


def _prepare_from_tokens(
    lf: "pl.LazyFrame",
    max_ply: int,
    max_games: int,
    min_ply: int,
    prepend_outcome: bool,
) -> dict:
    """Load the canonical Lichess parquet schema into training tensors.

    Expects the schema written by ``scripts/extract_lichess_parquet.py``:
    pure-moves ``tokens``, ``game_length``, ``outcome_token`` — all three
    columns must be present or the caller gets a clear error.

    When ``prepend_outcome=True``, emits the legacy outcome-prefixed layout
    (outcome at slot 0, moves at slots 1..gl+1). When ``False`` (default),
    emits pure-moves sequences that match the extractor's on-disk layout.
    """
    schema = lf.collect_schema()
    required = {"tokens", "game_length", "outcome_token"}
    missing = required - set(schema.names())
    if missing:
        raise ValueError(
            f"Parquet schema is missing required columns {sorted(missing)}. "
            f"Got: {sorted(schema.names())}. Legacy v1/v2 layouts (coarse "
            "result-derived outcomes, outcome-prefixed tokens) were removed "
            "along with the legacy vocabulary; re-extract the dataset with "
            "`scripts/extract_lichess_parquet.py`."
        )

    df = lf.select(
        ["tokens", "game_length", "outcome_token", "result"],
    ).head(max_games).collect()
    print(f"Loaded {len(df):,} games from pre-tokenized Parquet")

    token_lists = df["tokens"].to_list()
    game_lengths_list = df["game_length"].to_list()
    outcome_tokens_list = df["outcome_token"].to_list()
    N = len(token_lists)

    seq_len = max_ply + 1 if prepend_outcome else max_ply

    if N == 0:
        return {"move_ids": np.zeros((0, max_ply), dtype=np.int16),
                "game_lengths": np.zeros(0, dtype=np.int16),
                "input_ids": torch.zeros(0, seq_len, dtype=torch.long),
                "targets": torch.zeros(0, seq_len, dtype=torch.long),
                "loss_mask": torch.zeros(0, seq_len, dtype=torch.bool),
                "outcome_tokens": torch.zeros(0, dtype=torch.long),
                "n_games": 0}

    move_ids = np.zeros((N, max_ply), dtype=np.int16)
    game_lengths = np.zeros(N, dtype=np.int16)
    for i, toks in enumerate(token_lists):
        gl = min(int(game_lengths_list[i]), max_ply)
        game_lengths[i] = gl
        if gl > 0:
            move_ids[i, :gl] = toks[:gl]

    if min_ply > 1:
        keep = game_lengths >= min_ply
        move_ids = move_ids[keep]
        game_lengths = game_lengths[keep]
        outcome_tokens_list = [
            o for o, k in zip(outcome_tokens_list, keep) if k
        ]
        N = len(outcome_tokens_list)
        print(f"  After min_ply={min_ply} filter: {N:,} games")

    outcome_tokens = torch.tensor(outcome_tokens_list, dtype=torch.long)

    from pawn.data import pack_clm_sequences
    batch = pack_clm_sequences(
        move_ids, game_lengths, outcome_tokens, seq_len,
        prepend_outcome=prepend_outcome,
    )

    return {
        "move_ids": move_ids,
        "game_lengths": game_lengths,
        "input_ids": batch["input_ids"],
        "targets": batch["targets"],
        "loss_mask": batch["loss_mask"],
        "outcome_tokens": outcome_tokens,
        "n_games": N,
    }


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class LichessDataset(torch.utils.data.Dataset):
    """Map-style dataset for Lichess behavioral cloning."""

    def __init__(self, data: dict, start: int = 0, end: int | None = None):
        end = end or data["n_games"]
        self.input_ids = data["input_ids"][start:end]
        self.targets = data["targets"][start:end]
        self.loss_mask = data["loss_mask"][start:end]
        self.move_ids = data["move_ids"][start:end]
        self.game_lengths = data["game_lengths"][start:end]

    def share_memory(self):
        """Move tensors to shared memory so spawn workers avoid copies."""
        self.input_ids = self.input_ids.share_memory_()
        self.targets = self.targets.share_memory_()
        self.loss_mask = self.loss_mask.share_memory_()
        self.move_ids = torch.from_numpy(np.array(self.move_ids)).share_memory_()
        self.game_lengths = torch.from_numpy(np.array(self.game_lengths)).share_memory_()
        return self

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        return {
            "input_ids": self.input_ids[idx],
            "targets": self.targets[idx],
            "loss_mask": self.loss_mask[idx],
            "move_ids": self.move_ids[idx],
            "game_length": int(self.game_lengths[idx]),
        }
