"""Lichess data preparation for FiLM behavioral cloning.

Parses a PGN file, tokenizes via the Rust engine, and produces PyTorch
tensors ready for training.  Legal move grids are computed per-batch during
training (not precomputed) to keep memory independent of dataset size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.utils.data

import chess_engine as engine

from pawn.config import (
    WHITE_CHECKMATES,
    BLACK_CHECKMATES,
    DRAW_BY_RULE,
    PLY_LIMIT,
)


# ---------------------------------------------------------------------------
# PGN result → outcome token
# ---------------------------------------------------------------------------

_RESULT_MAP = {
    "1-0": "white",
    "0-1": "black",
    "1/2-1/2": "draw",
}


def _result_to_outcome(results: list[str]) -> torch.Tensor:
    """Map PGN result strings to outcome token IDs.

    For decisive games we use the checkmate token even though the actual
    termination was likely resignation/time — the prefix of moves is still
    valid strategic play and the outcome token approximation is acceptable
    per the spec (§3.4).
    """
    outcomes = torch.full((len(results),), PLY_LIMIT, dtype=torch.long)
    for i, result in enumerate(results):
        mapped = _RESULT_MAP.get(result)
        if mapped == "white":
            outcomes[i] = WHITE_CHECKMATES
        elif mapped == "black":
            outcomes[i] = BLACK_CHECKMATES
        elif mapped == "draw":
            outcomes[i] = DRAW_BY_RULE
    return outcomes


# ---------------------------------------------------------------------------
# Legal token mask via fused Rust computation
# ---------------------------------------------------------------------------


def compute_legal_indices(
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    seq_len: int,
    vocab_size: int = 4278,
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

    def __init__(self, batch_size: int, max_ply: int, vocab_size: int = 4278,
                 device: str = "cpu", max_index_buf: int = 4_000_000):
        self.vocab_size = vocab_size
        self.max_ply = max_ply
        self.T = max_ply + 1  # seq_len = outcome token + max_ply move slots
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

    def __init__(self, seq_len: int, vocab_size: int = 4278):
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
) -> dict:
    """Parse a PGN or Parquet file and produce training-ready tensors.

    If pgn_path ends with .parquet, delegates to prepare_lichess_parquet().
    If pgn_path looks like a HuggingFace repo (contains '/'), loads from HF.

    Returns dict with:
        move_ids:       (N, max_ply) int16 — tokenized moves
        game_lengths:   (N,) int16
        input_ids:      (N, seq_len) long — [outcome, move_0, ..., PAD]
        targets:        (N, seq_len) long — shifted left
        loss_mask:      (N, seq_len) bool
        n_games:        int
    """
    pgn_path_str = str(pgn_path)
    if pgn_path_str.endswith(".parquet"):
        return prepare_lichess_parquet(
            parquet_path=pgn_path_str, max_ply=max_ply,
            max_games=max_games, min_ply=min_ply,
        )
    # Check if it looks like a HF repo ID (e.g. "user/dataset")
    if "/" in pgn_path_str and not Path(pgn_path_str).exists():
        return prepare_lichess_parquet(
            hf_repo=pgn_path_str, max_ply=max_ply,
            max_games=max_games, min_ply=min_ply,
        )
    pgn_path = Path(pgn_path)

    # Parse with min_ply=1 so every parseable game appears in the output,
    # keeping result extraction aligned.  We apply min_ply in Python below.
    print(f"Parsing PGN: {pgn_path}")
    move_ids, game_lengths, n_parsed = engine.parse_pgn_file(
        str(pgn_path), max_ply=max_ply, max_games=max_games, min_ply=1,
    )
    N = move_ids.shape[0]
    print(f"  Parsed {n_parsed} PGN games, {N} tokenized")

    move_ids = move_ids[:N]
    game_lengths = game_lengths[:N]

    # Extract results — aligned with engine output since min_ply=1
    results = _extract_results(pgn_path, n_parsed)[:N]

    # Apply min_ply filter in Python on aligned arrays
    if min_ply > 1:
        keep = game_lengths >= min_ply
        move_ids = move_ids[keep]
        game_lengths = game_lengths[keep]
        results = [r for r, k in zip(results, keep) if k]
        N = len(results)
        print(f"  After min_ply={min_ply} filter: {N} games")

    outcome_tokens = _result_to_outcome(results)

    seq_len = max_ply + 1  # outcome token + max_ply move slots

    from pawn.data import pack_clm_sequences
    batch = pack_clm_sequences(move_ids, game_lengths, outcome_tokens, seq_len)

    return {
        "move_ids": move_ids,
        "game_lengths": game_lengths,
        "input_ids": batch["input_ids"],
        "targets": batch["targets"],
        "loss_mask": batch["loss_mask"],
        "outcome_tokens": outcome_tokens,
        "n_games": N,
    }


def prepare_lichess_parquet(
    parquet_path: str | Path = None,
    hf_repo: str = None,
    max_ply: int = 255,
    max_games: int = 50_000,
    min_ply: int = 10,
) -> dict:
    """Load a Lichess Parquet dataset and produce training-ready tensors.

    Reads from a local Parquet file or a HuggingFace dataset repo.
    Expects columns: pgn (SAN move text), result (1-0/0-1/1/2-1/2).

    Returns the same dict format as prepare_lichess_dataset().
    """
    import pyarrow.parquet as pq

    if hf_repo is not None:
        from huggingface_hub import hf_hub_download, HfApi
        import pyarrow as pa
        api = HfApi()
        files = api.list_repo_files(hf_repo, repo_type="dataset")
        parquet_files = [f for f in files if f.endswith(".parquet")]
        tables = []
        remaining = max_games
        for pf in parquet_files:
            if remaining <= 0:
                break
            local = hf_hub_download(hf_repo, pf, repo_type="dataset")
            t = pq.read_table(local, columns=["pgn", "result"])
            if len(t) > remaining:
                t = t.slice(0, remaining)
            tables.append(t)
            remaining -= len(t)
        table = pa.concat_tables(tables)
    elif parquet_path is not None:
        table = pq.read_table(parquet_path, columns=["pgn", "result"])
    else:
        raise ValueError("Either parquet_path or hf_repo must be provided")

    n_available = len(table)
    n_to_use = min(max_games, n_available)
    table = table.slice(0, n_to_use)
    print(f"Loaded {n_to_use} games from Parquet ({n_available} available)")

    # Extract SAN move lists from the pgn column
    pgn_strings = table.column("pgn").to_pylist()
    results = table.column("result").to_pylist()

    # Split PGN text into move lists, stripping move numbers and results
    import re
    games: list[list[str]] = []
    for pgn_text in pgn_strings:
        # Remove move numbers (1. 2. etc) and result markers
        tokens = pgn_text.split()
        moves = []
        for tok in tokens:
            if tok in ("1-0", "0-1", "1/2-1/2", "*"):
                break
            # Skip move numbers
            stripped = tok.rstrip(".")
            if stripped and stripped.replace(".", "").isdigit():
                continue
            moves.append(tok)
        games.append(moves)

    # Tokenize via Rust engine (batch)
    print(f"  Tokenizing {len(games)} games...")
    move_ids, game_lengths = engine.pgn_to_tokens(games, max_ply=max_ply)
    N = move_ids.shape[0]

    # Apply min_ply filter
    if min_ply > 1:
        keep = game_lengths >= min_ply
        move_ids = move_ids[keep]
        game_lengths = game_lengths[keep]
        results = [r for r, k in zip(results, keep) if k]
        N = len(results)
        print(f"  After min_ply={min_ply} filter: {N} games")

    outcome_tokens = _result_to_outcome(results)

    seq_len = max_ply + 1
    from pawn.data import pack_clm_sequences
    batch = pack_clm_sequences(move_ids, game_lengths, outcome_tokens, seq_len)

    return {
        "move_ids": move_ids,
        "game_lengths": game_lengths,
        "input_ids": batch["input_ids"],
        "targets": batch["targets"],
        "loss_mask": batch["loss_mask"],
        "outcome_tokens": outcome_tokens,
        "n_games": N,
    }


def _extract_results(pgn_path: Path, max_games: int) -> list[str]:
    """Extract game results from PGN headers.

    Uses [Event header to delimit games, matching the Rust parser's
    game-boundary detection.  The previous approach (one result per
    [Result] header) could miscount when headers were malformed.
    """
    import re
    results: list[str] = []
    current_result = "*"
    in_game = False

    with open(pgn_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("[Event "):
                if in_game:
                    results.append(current_result)
                    if len(results) >= max_games:
                        break
                current_result = "*"
                in_game = True
            elif line.startswith('[Result "'):
                m = re.search(r'"([^"]+)"', line)
                if m:
                    current_result = m.group(1)
        # Flush last game
        if in_game and len(results) < max_games:
            results.append(current_result)

    return results


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
