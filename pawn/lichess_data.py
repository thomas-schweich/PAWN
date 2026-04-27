"""Lichess data preparation for FiLM behavioral cloning.

Parses a PGN file, tokenizes via the Rust engine, and produces PyTorch
tensors ready for training.  Legal move grids are computed per-batch during
training (not precomputed) to keep memory independent of dataset size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from typing_extensions import NotRequired

import numpy as np
import polars as pl
import torch
import torch.utils.data

import chess_engine as engine

from pawn.config import PAD_TOKEN


class BucketedBatchDict(TypedDict):
    """Output schema of :class:`BucketedLegalMaskCollate`.

    All keys except ``legal_indices`` are always populated.
    ``legal_indices`` is omitted when the collate's
    ``skip_legal_indices`` flag is set (e.g. on the val loader after
    :func:`pawn.adapter_training.precompute_val_masks` has cached them
    upstream); consumers branch on ``"legal_indices" in batch``.
    """

    input_ids: torch.Tensor       # (B, T_actual) long
    targets: torch.Tensor         # (B, T_actual) long
    loss_mask: torch.Tensor       # (B, T_actual) bool
    move_ids: torch.Tensor        # (B, max_ply) int16
    game_length: torch.Tensor     # (B,) int64
    T_actual: int                 # bucketed per-batch sequence width
    legal_indices: NotRequired[torch.Tensor]  # (n,) int64 — flat sparse
                                              # indices into the
                                              # (B, T_actual, vocab_size)
                                              # mask buffer; absent when
                                              # ``skip_legal_indices`` is set


def round_up_to_bucket(n: int, bucket_size: int, cap: int) -> int:
    """Round ``n`` up to the next multiple of ``bucket_size``, clamped to ``cap``.

    With ``bucket_size=64`` and ``cap=512`` this produces T ∈
    {64, 128, 192, 256, 320, 384, 448, 512}. ``n <= 0`` rounds up to
    ``bucket_size`` (a degenerate empty batch still gets a valid T).
    """
    if bucket_size <= 0:
        return min(max(n, 1), cap)
    rounded = ((max(n, 1) + bucket_size - 1) // bucket_size) * bucket_size
    return min(rounded, cap)


def bucket_grid(bucket_size: int, cap: int) -> list[int]:
    """Enumerate the distinct ``T`` values :func:`round_up_to_bucket` can emit.

    With ``bucket_size=64, cap=512`` returns ``[64, 128, ..., 512]`` (8
    entries). When ``cap`` is not a multiple of ``bucket_size`` the
    final entry is ``cap`` itself (e.g. ``bucket_size=64, cap=300`` →
    ``[64, 128, 192, 256, 300]``). ``bucket_size <= 0`` collapses to
    ``[cap]``. Used by the compiled-step pre-warm loop so each shape's
    dynamo trace + cudagraph capture cost is paid before training
    starts rather than lazily on the first batch that hits each bucket.
    """
    if bucket_size <= 0:
        return [max(cap, 1)]
    grid = list(range(bucket_size, cap + 1, bucket_size))
    if not grid or grid[-1] != cap:
        grid.append(cap)
    return grid


# ---------------------------------------------------------------------------
# Legal token mask via fused Rust computation
# ---------------------------------------------------------------------------


def compute_legal_indices(
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    seq_len: int,
    vocab_size: int = 1968,
    prepend_outcome: bool = False,
) -> np.ndarray:
    """Compute flat sparse indices for legal token masks (CPU only).

    Calls the Rust engine to replay games and returns flat i64 indices
    suitable for scattering into a ``(B, seq_len, vocab_size)`` bool mask.

    The engine returns a raw mask indexed by game-ply:
    ``raw[p]`` = moves legal at game-ply ``p`` (i.e., the legal moves for
    choosing the ``(p+1)``-th move). In the **outcome-prefixed layout**
    (``prepend_outcome=True``), the model at token position ``t`` predicts
    ``m_{t+1}``, so the raw ``raw[t]`` mask aligns directly — no shift.

    In the **pure-moves layout** (``prepend_outcome=False``, the default),
    ``pack_clm_sequences`` makes ``targets[t] == input_ids[t+1] == m_{t+2}``,
    so the constraint at position ``t`` is ``raw[t+1]``. We need to shift
    the raw mask left by one ply, which we do sparsely: for each flat
    index ``(b, p, v)`` we replace ``p`` with ``p-1`` and drop the entries
    where the original ``p == 0`` (those have no pure-moves token position
    to align with).
    """
    move_ids = np.ascontiguousarray(move_ids, dtype=np.int16)
    game_lengths = np.asarray(game_lengths, dtype=np.int16)
    indices = engine.compute_legal_token_masks_sparse(
        move_ids, game_lengths, seq_len, vocab_size,
    )
    if not prepend_outcome:
        # Shift left by one ply. The flat index encodes (b, p, v) as
        #   idx = b * seq_len * vocab_size + p * vocab_size + v
        # so ``(idx % (seq_len * vocab_size)) < vocab_size`` is the
        # ``p == 0`` case — those entries have no corresponding pure-moves
        # slot and must be dropped. The rest shift down by ``vocab_size``
        # so ``new_p = old_p - 1``.
        per_batch_offset = indices % (seq_len * vocab_size)
        keep = per_batch_offset >= vocab_size
        indices = indices[keep] - vocab_size

        # Full-length games: when ``game_length == seq_len``, the Rust
        # engine skips the PAD entry at position ``length`` because
        # there's no room in the (B, seq_len, V) tensor. The shift
        # doesn't produce one either (it would come from ``raw[seq_len]``
        # which doesn't exist), so the last supervised position
        # (``seq_len - 1``, whose target is PAD in pure-moves mode)
        # ends up with an all-zero legal mask. Add PAD explicitly for
        # those games.
        full_length = np.where(np.asarray(game_lengths) == seq_len)[0]
        if full_length.size > 0:
            cell_size = seq_len * vocab_size
            last_slot_base = (seq_len - 1) * vocab_size
            pad_indices = np.fromiter(
                (int(b) * cell_size + last_slot_base + PAD_TOKEN
                 for b in full_length),
                dtype=np.int64,
                count=full_length.size,
            )
            indices = np.concatenate([indices, pad_indices])

    # Defensive bounds check after the shift / PAD-fix steps. By this
    # point every index must land inside the ``(B, seq_len, vocab_size)``
    # mask buffer; an out-of-bounds value would scatter into the wrong
    # row at runtime and silently corrupt the legality mask. Cheap to
    # verify and catches any future Rust-engine regression at the
    # source rather than as a CUDA scatter fault.
    if indices.size > 0:
        B = move_ids.shape[0]
        upper = B * seq_len * vocab_size
        if not (indices < upper).all():
            bad = int(indices.max())
            raise AssertionError(
                f"legal indices out of bounds after shift: max={bad}, "
                f"limit={upper} (B={B}, seq_len={seq_len}, V={vocab_size})"
            )
    return indices


class LegalMaskBuilder:
    """Legal token mask via sparse Rust computation + GPU scatter.

    Calls engine.compute_legal_token_masks_sparse which replays games and
    returns flat i64 indices (~2 MB) instead of a dense bool mask (~70 MB).
    Indices are transferred to GPU and scattered into a fresh ``(B, T, V)``
    bool tensor per call. The caching allocator reuses freed buffers
    across steps, so per-call allocation cost is dominated by the
    ``zero_()`` memset; pre-allocation no longer pulls its weight under
    bucketed collates where ``T`` varies per batch.

    The reusable ``_idx_buf`` (a long index staging buffer) is still
    pre-allocated — it's small (~32 MB at default size) and avoids a
    per-batch H2D copy in the common case.

    Two usage modes:
      1. ``scatter(indices, B, T=...)`` — fast GPU-only path for
         pre-computed indices (from ``LegalMaskCollate`` /
         ``BucketedLegalMaskCollate`` / precomputation).
      2. ``__call__(batch)`` — legacy path that computes indices inline.
    """

    def __init__(self, batch_size: int, seq_len: int, vocab_size: int = 1968,
                 device: str = "cpu", max_index_buf: int = 4_000_000,
                 prepend_outcome: bool = False):
        """
        Args:
            batch_size: maximum batch size accepted by ``scatter``. No
                tensor of this size is pre-allocated; the value is only
                used as a capacity check.
            seq_len: total tensor width matching the model's ``max_seq_len``.
                In outcome-prefixed mode the outcome slot lives inside this
                budget; in pure-moves mode all ``seq_len`` slots hold moves.
                This matches the convention used by ``pack_clm_sequences``
                and the Rust ``parse_pgn_lichess`` wrapper.
            prepend_outcome: required so the sparse legal mask can be shifted
                by one ply when the data layout is pure moves (see
                ``compute_legal_indices``).
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.prepend_outcome = prepend_outcome
        self.T = seq_len
        self.device = device
        self._max_batch = int(batch_size)

        # Pre-allocated GPU index buffer to avoid per-batch allocation
        # for the indices staging copy. The output mask buffer is *not*
        # pre-allocated — see class docstring.
        self._idx_buf = torch.empty(max_index_buf, dtype=torch.long, device=device)

    def scatter(
        self,
        legal_indices: torch.Tensor,
        B: int,
        T: int,
    ) -> torch.Tensor:
        """Scatter pre-computed CPU indices into a fresh GPU mask buffer.

        Allocates a ``(B, T, V)`` bool tensor and scatters the supplied
        flat indices into it. ``T`` is the per-batch tensor width that
        the indices were computed against — required, because under the
        bucketed collate every batch has its own ``T_actual`` and
        silently defaulting to ``self.T`` (the model's full seq_len)
        would over-allocate. Legacy fixed-T callers pass ``T=builder.T``
        explicitly. The caching allocator reuses freed buffers across
        steps, so per-call cost is dominated by the ``zero_()`` memset.
        """
        if B > self._max_batch:
            raise ValueError(
                f"B={B} exceeds builder capacity batch_size={self._max_batch}"
            )
        if T > self.T:
            raise ValueError(
                f"scatter T={T} exceeds builder seq_len={self.T}"
            )
        mask_view = torch.zeros(
            B, T, self.vocab_size,
            dtype=torch.bool, device=self.device,
        )
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

        indices = compute_legal_indices(
            move_ids, game_lengths, self.T, self.vocab_size,
            prepend_outcome=self.prepend_outcome,
        )

        return self.scatter(torch.from_numpy(indices), B, self.T)


class LegalMaskCollate:
    """Collate that computes legal mask indices in DataLoader workers.

    Wraps default collation and appends a ``legal_indices`` CPU tensor
    to each batch so the Rust replay runs in worker processes, off the
    GPU training critical path. ``prepend_outcome`` must match the
    data pipeline's layout — see ``compute_legal_indices`` for why the
    pure-moves path shifts the mask by one ply.
    """

    def __init__(self, seq_len: int, vocab_size: int = 1968,
                 prepend_outcome: bool = False):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.prepend_outcome = prepend_outcome

    def __call__(self, items: list[dict]) -> dict:
        batch = torch.utils.data.default_collate(items)
        move_ids = batch["move_ids"].numpy()
        game_lengths = np.asarray(batch["game_length"], dtype=np.int16)
        indices = compute_legal_indices(
            move_ids, game_lengths, self.seq_len, self.vocab_size,
            prepend_outcome=self.prepend_outcome,
        )
        batch["legal_indices"] = torch.from_numpy(indices)
        return batch


class BucketedLegalMaskCollate:
    """Collate that packs sequences and legal masks at a per-batch ``T``.

    Designed for ``IndexedLichessDataset(lazy_pack=True)`` — the dataset
    yields raw ``move_ids`` (an ``int16`` array of length ``max_ply``,
    zero-padded after the actual game) plus ``game_length`` and
    ``outcome_token``. This collate computes::

        T_eff    = max(game_lengths) + (1 if prepend_outcome else 0)
        T_padded = round_up(T_eff, bucket_size)  clamped to seq_len

    and runs both :func:`pawn.data.pack_clm_sequences` and
    :func:`compute_legal_indices` at ``T_padded``. With ``bucket_size``
    chosen so that only a small handful of distinct ``T`` values are
    ever emitted, the downstream compiled step graph cache stays
    bounded — the win is concentrated on the typical Lichess game
    (~80 ply) which previously paid attention cost at ``seq_len=512``
    and now pays it at ``T_padded=128``.
    """

    def __init__(
        self,
        seq_len: int,
        bucket_size: int,
        vocab_size: int = 1968,
        prepend_outcome: bool = False,
    ):
        if bucket_size <= 0:
            raise ValueError(f"bucket_size must be > 0, got {bucket_size}")
        # When ``seq_len`` is not a multiple of ``bucket_size`` the
        # cap-clamped top bucket is off-grid (e.g. seq_len=300,
        # bucket_size=64 → buckets {64, 128, 192, 256, 300} instead of
        # {64, 128, 192, 256}). torch.compile traces one extra graph
        # for that shape; that's a small cache-pressure cost, not a
        # correctness issue. Permitted so legacy ``max_seq_len`` values
        # like 255 / 300 keep working without a forced migration.
        self.seq_len = seq_len
        self.bucket_size = bucket_size
        self.vocab_size = vocab_size
        self.prepend_outcome = prepend_outcome
        # When the val loader's indices are precomputed and cached
        # upstream (see ``precompute_val_masks``), the per-batch
        # ``compute_legal_indices`` call is pure waste — the eval loop
        # ignores the collate's indices and uses the cache instead.
        # Callers flip this to ``True`` after precompute completes; on
        # the train loader it stays ``False`` (every batch's indices
        # are consumed exactly once).
        self.skip_legal_indices: bool = False

    def __call__(self, items: list[dict[str, Any]]) -> BucketedBatchDict:
        from pawn.data import pack_clm_sequences

        # Stack raw inputs. Keeping ``move_ids`` as int16 (its storage
        # dtype on disk) avoids an extra cast in the worker; it's
        # promoted inside ``pack_clm_sequences``.
        move_ids = np.stack(
            [np.asarray(it["move_ids"], dtype=np.int16) for it in items], axis=0
        )
        game_lengths = np.asarray(
            [int(it["game_length"]) for it in items], dtype=np.int16
        )
        outcome_tokens = torch.tensor(
            [int(it["outcome_token"]) for it in items], dtype=torch.long
        )

        max_len = int(game_lengths.max()) if len(game_lengths) > 0 else 0
        offset = 1 if self.prepend_outcome else 0
        T_padded = round_up_to_bucket(
            max_len + offset, self.bucket_size, self.seq_len
        )

        packed = pack_clm_sequences(
            move_ids, game_lengths, outcome_tokens, T_padded,
            prepend_outcome=self.prepend_outcome,
        )

        out: BucketedBatchDict = {
            "input_ids": packed["input_ids"],
            "targets": packed["targets"],
            "loss_mask": packed["loss_mask"],
            "move_ids": torch.from_numpy(move_ids),
            "game_length": torch.from_numpy(game_lengths.astype(np.int64)),
            "T_actual": T_padded,
        }
        if not self.skip_legal_indices:
            indices = compute_legal_indices(
                move_ids, game_lengths, T_padded, self.vocab_size,
                prepend_outcome=self.prepend_outcome,
            )
            out["legal_indices"] = torch.from_numpy(indices)
        return out


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
        ["tokens", "game_length", "outcome_token"],
    ).head(max_games).collect()
    print(f"Loaded {len(df):,} games from pre-tokenized Parquet")

    token_lists = df["tokens"].to_list()
    game_lengths_list = df["game_length"].to_list()
    outcome_tokens_list = df["outcome_token"].to_list()
    N = len(token_lists)

    # ``max_ply`` is the total tensor-width budget (matches the model's
    # ``max_seq_len`` and the convention used by ``pack_clm_sequences``).
    # In outcome-prefixed mode the outcome slot lives inside that budget,
    # so the effective move cap is ``max_ply - 1``.
    seq_len = max_ply
    effective_max_moves = max_ply - 1 if prepend_outcome else max_ply

    if N == 0:
        return {"move_ids": np.zeros((0, effective_max_moves), dtype=np.int16),
                "game_lengths": np.zeros(0, dtype=np.int16),
                "input_ids": torch.zeros(0, seq_len, dtype=torch.long),
                "targets": torch.zeros(0, seq_len, dtype=torch.long),
                "loss_mask": torch.zeros(0, seq_len, dtype=torch.bool),
                "outcome_tokens": torch.zeros(0, dtype=torch.long),
                "n_games": 0}

    move_ids = np.zeros((N, effective_max_moves), dtype=np.int16)
    game_lengths = np.zeros(N, dtype=np.int16)
    for i, toks in enumerate(token_lists):
        gl = min(int(game_lengths_list[i]), effective_max_moves)
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
