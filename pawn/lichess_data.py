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

from pawn.config import (
    OUTCOME_TOKEN_BASE,
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
) -> dict:
    """Load a Parquet dataset and produce training-ready tensors.

    Supports two Parquet formats:
    1. **PAWN token format** (has ``tokens`` column): pre-tokenized list[int16].
       No parsing needed — just pad and pack. Used by pawn-lichess-full and
       stockfish-nodes1 datasets.
    2. **Legacy PGN format** (has ``pgn`` column): SAN move text that needs
       tokenization via the Rust engine.

    Reads from a local Parquet file or a HuggingFace dataset repo.
    Returns the same dict format as prepare_lichess_dataset().
    """
    lf = _scan_parquet(parquet_path, hf_repo, split)
    schema = lf.collect_schema()

    if "tokens" in schema:
        return _prepare_from_tokens(lf, max_ply, max_games, min_ply)
    elif "pgn" in schema:
        return _prepare_from_pgn(lf, max_ply, max_games, min_ply)
    else:
        raise ValueError(
            f"Parquet schema must have 'tokens' or 'pgn' column, "
            f"got: {list(schema.names())}"
        )


def _prepare_from_tokens(
    lf: "pl.LazyFrame",
    max_ply: int,
    max_games: int,
    min_ply: int,
) -> dict:
    """Load pre-tokenized PAWN format: tokens list[int16] + result str.

    Supports two token formats:
    - **v2 (outcome-prepended)**: tokens[0] is an outcome token (>= OUTCOME_TOKEN_BASE),
      followed by move tokens. Used by pawn-lichess-full v2+.
    - **v1 (moves only)**: tokens are pure move IDs. Outcome is derived from the
      ``result`` column. Used by stockfish-nodes1 and pawn-lichess-full v1.

    Format is auto-detected from the first token of the first game.
    """

    needed_cols = ["tokens", "result"]
    if "game_length" in lf.collect_schema():
        needed_cols.append("game_length")

    df = lf.select(needed_cols).head(max_games).collect()
    print(f"Loaded {len(df):,} games from pre-tokenized Parquet")

    token_lists = df["tokens"].to_list()
    N = len(token_lists)
    if N == 0:
        return {"move_ids": np.zeros((0, max_ply), dtype=np.int16),
                "game_lengths": np.zeros(0, dtype=np.int16),
                "input_ids": torch.zeros(0, max_ply + 1, dtype=torch.long),
                "targets": torch.zeros(0, max_ply + 1, dtype=torch.long),
                "loss_mask": torch.zeros(0, max_ply + 1, dtype=torch.bool),
                "outcome_tokens": torch.zeros(0, dtype=torch.long),
                "n_games": 0}

    # Auto-detect format: v2 has outcome token (>= 4273) at position 0
    first_token = token_lists[0][0] if token_lists[0] else 0
    has_outcome_token = first_token >= OUTCOME_TOKEN_BASE
    seq_len = max_ply + 1

    if has_outcome_token:
        print(f"  Detected v2 format (outcome token prepended)")
        return _prepare_v2_tokens(token_lists, df["result"].to_list(),
                                  max_ply, min_ply, seq_len)
    else:
        print(f"  Detected v1 format (moves only, deriving outcomes from result)")
        return _prepare_v1_tokens(token_lists, df["result"].to_list(),
                                  max_ply, min_ply, seq_len)


def _prepare_v2_tokens(
    token_lists: list[list[int]],
    results: list[str],
    max_ply: int,
    min_ply: int,
    seq_len: int,
) -> dict:
    """Prepare training data from v2 tokens (outcome already prepended).

    Uses the token sequence verbatim as input_ids — no disassembly.
    """
    N = len(token_lists)

    # Build input_ids directly: pad each token list to seq_len
    input_ids = torch.zeros(N, seq_len, dtype=torch.long)
    game_lengths = np.zeros(N, dtype=np.int16)
    move_ids = np.zeros((N, max_ply), dtype=np.int16)

    for i, toks in enumerate(token_lists):
        # toks = [outcome, move_1, ..., move_K]
        n_moves = min(len(toks) - 1, max_ply)
        game_lengths[i] = n_moves
        length = min(len(toks), seq_len)
        input_ids[i, :length] = torch.tensor(toks[:length], dtype=torch.long)
        # move_ids for legal mask computation (moves only, no outcome)
        move_ids[i, :n_moves] = toks[1 : n_moves + 1]

    # Apply min_ply filter
    if min_ply > 1:
        keep = game_lengths >= min_ply
        input_ids = input_ids[keep]
        move_ids = move_ids[keep]
        game_lengths = game_lengths[keep]
        results = [r for r, k in zip(results, keep) if k]
        N = len(results)
        print(f"  After min_ply={min_ply} filter: {N:,} games")

    outcome_tokens = input_ids[:, 0]
    capped_lengths = torch.from_numpy(game_lengths).long().clamp(max=max_ply)

    # Targets: input shifted left by 1
    targets = torch.zeros(N, seq_len, dtype=torch.long)
    targets[:, :-1] = input_ids[:, 1:]

    # Loss mask: True for positions 0 through game_length
    positions = torch.arange(seq_len).unsqueeze(0)
    loss_mask = positions <= capped_lengths.unsqueeze(1)

    return {
        "move_ids": move_ids,
        "game_lengths": game_lengths,
        "input_ids": input_ids,
        "targets": targets,
        "loss_mask": loss_mask,
        "outcome_tokens": outcome_tokens,
        "n_games": N,
    }


def _prepare_v1_tokens(
    token_lists: list[list[int]],
    results: list[str],
    max_ply: int,
    min_ply: int,
    seq_len: int,
) -> dict:
    """Prepare training data from v1 tokens (moves only, no outcome)."""
    N = len(token_lists)

    move_ids = np.zeros((N, max_ply), dtype=np.int16)
    game_lengths = np.zeros(N, dtype=np.int16)

    for i, toks in enumerate(token_lists):
        length = min(len(toks), max_ply)
        game_lengths[i] = length
        move_ids[i, :length] = toks[:length]

    if min_ply > 1:
        keep = game_lengths >= min_ply
        move_ids = move_ids[keep]
        game_lengths = game_lengths[keep]
        results = [r for r, k in zip(results, keep) if k]
        N = len(results)
        print(f"  After min_ply={min_ply} filter: {N:,} games")

    outcome_tokens = _result_to_outcome(results)

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


def _prepare_from_pgn(
    lf: "pl.LazyFrame",
    max_ply: int,
    max_games: int,
    min_ply: int,
) -> dict:
    """Load legacy PGN format: pgn str + result str, tokenize via Rust."""
    import polars as pl
    import re

    df = lf.select(["pgn", "result"]).head(max_games).collect()
    n_to_use = len(df)
    print(f"Loaded {n_to_use:,} games from PGN Parquet")

    pgn_strings = df["pgn"].to_list()
    results = df["result"].to_list()

    # Split PGN text into move lists, stripping comments, move numbers, results
    games: list[list[str]] = []
    for pgn_text in pgn_strings:
        cleaned = re.sub(r'\{[^}]*\}', '', pgn_text)
        tokens = cleaned.split()
        moves = []
        for tok in tokens:
            if tok in ("1-0", "0-1", "1/2-1/2", "*"):
                break
            stripped = tok.rstrip(".")
            if stripped and stripped.replace(".", "").isdigit():
                continue
            if not tok:
                continue
            moves.append(tok)
        games.append(moves)

    print(f"  Tokenizing {len(games):,} games...")
    move_ids, game_lengths = engine.pgn_to_tokens(games, max_ply=max_ply)
    N = move_ids.shape[0]

    if min_ply > 1:
        keep = game_lengths >= min_ply
        move_ids = move_ids[keep]
        game_lengths = game_lengths[keep]
        results = [r for r, k in zip(results, keep) if k]
        N = len(results)
        print(f"  After min_ply={min_ply} filter: {N:,} games")

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
