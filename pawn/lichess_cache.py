"""Persistent tokenized cache for Lichess datasets.

Stores the result of `prepare_lichess_dataset`'s filter + tokenize step at
the training-example level. Keyed on canonical filter parameters; the
cache contents are invariant to ``max_ply`` and ``prepend_outcome``,
which only affect packing and apply at access time.

On-disk layout under ``<cache_root>/<key>/``:
  - ``tokens_flat.bin``  — raw int16 buffer, ``(total_tokens,)``.
                           mmap-backed at load, never resident in RAM.
  - ``index.safetensors`` — ``offsets`` (int64, n_games+1) and
                            ``outcome_tokens`` (int16, n_games).
  - ``meta.json``         — filter params + counts.
  - ``.complete``         — SHA-256 sentinel (mirrors the checkpoint pattern).

The build path uses Polars' Arrow round-trip to extract the variable-length
``tokens`` column in CSR form (flat values + offsets) without a Python loop.
For a 16 M-game Elo-filtered slice this is dominated by Polars' parallel
filter and runs in under a minute on a fast disk.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from pawn.checkpoint import (
    CheckpointIntegrityError,
    IncompleteCheckpointError,
    _atomic_directory_write,
    _verify_complete_sentinel,
)

log = logging.getLogger("pawn.lichess_cache")


@dataclass
class LichessTokenCache:
    """A loaded tokenized cache.

    ``tokens_flat`` is mmap-backed: slices into it page-fault on first
    read and are shared via the OS page cache across spawn workers.

    Use :class:`IndexedLichessDataset` to wrap a cache for DataLoader use.
    """

    cache_dir: Path
    tokens_flat: torch.Tensor       # int16, (total_tokens,)
    offsets: torch.Tensor           # int64, (n_games + 1,)
    outcome_tokens: torch.Tensor    # int16, (n_games,)
    n_games: int
    meta: dict[str, Any]


# ---------------------------------------------------------------------------
# Cache root resolution
# ---------------------------------------------------------------------------

def default_cache_root() -> Path:
    """Default cache directory.

    Resolution order:
      1. ``PAWN_DATA_CACHE`` env var
      2. ``$HF_HOME/pawn-lichess-cache``
      3. ``~/.cache/huggingface/pawn-lichess-cache``
    """
    env = os.environ.get("PAWN_DATA_CACHE")
    if env:
        return Path(env).expanduser()
    hf_home = os.environ.get("HF_HOME")
    base = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    return base.expanduser() / "pawn-lichess-cache"


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _derive_key(
    repo_or_path: str,
    split: str,
    elo_min: int | None,
    elo_max: int | None,
    min_ply: int,
) -> str:
    """Stable cache key from canonical filter params.

    A short Blake2b digest disambiguates collisions in the human-readable
    prefix (e.g. two different parquet paths whose stems happen to match).
    """
    norm = {
        "repo_or_path": repo_or_path,
        "split": split,
        "elo_min": elo_min,
        "elo_max": elo_max,
        "min_ply": min_ply,
    }
    digest = hashlib.blake2b(
        json.dumps(norm, sort_keys=True).encode(), digest_size=6
    ).hexdigest()
    if "/" in repo_or_path and not repo_or_path.endswith(".parquet"):
        slug = repo_or_path.replace("/", "--")
    else:
        slug = Path(repo_or_path).stem
    elo_part = f"e{elo_min if elo_min is not None else 'any'}-{elo_max if elo_max is not None else 'any'}"
    return f"{slug}_{split}_{elo_part}_p{min_ply}_{digest}"


# ---------------------------------------------------------------------------
# Probe — does a split have any parquet shards we could load?
# ---------------------------------------------------------------------------

def split_has_files(repo_or_path: str | Path, split: str) -> bool:
    """Check whether ``split`` has parquet shards available.

    For HF repos this lists repo files once and checks for paths
    containing ``/<split>-`` ending in ``.parquet``. For a local single
    ``.parquet`` file there's no split semantics — return True only for
    ``split == "train"``. For local directories / globs we match the
    filesystem.

    Cheap probe (one HF ``list_repo_files`` call); the right guard
    before attempting to build a validation cache. ``_scan_parquet``'s
    own "fall back to all parquet files" branch would silently load
    the wrong shards if a split had no files.
    """
    s = str(repo_or_path)

    if s.endswith(".parquet") and Path(s).exists():
        # Single file has no split semantics; only "train" makes sense.
        return split == "train"

    if "/" in s and not Path(s).exists():
        try:
            from huggingface_hub import HfApi

            files = HfApi().list_repo_files(s, repo_type="dataset")
        except Exception:
            return False
        marker = f"/{split}-"
        return any(
            f.endswith(".parquet") and marker in f"/{f}"
            for f in files
        )

    # Local directory or glob pattern.
    p = Path(s)
    if p.is_dir():
        return any(p.glob(f"data/{split}-*.parquet")) or any(
            p.glob(f"{split}-*.parquet")
        )
    # Glob pattern: substitute the asterisk for ``<split>-*`` and check.
    if "*" in s:
        import glob as _glob

        return bool(_glob.glob(s.replace("*", f"{split}-*", 1)))
    return False


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _build_cache(
    cache_dir: Path,
    repo_or_path: str,
    split: str,
    elo_min: int | None,
    elo_max: int | None,
    min_ply: int,
) -> None:
    """Materialize the filtered, tokenized dataset to ``cache_dir``.

    Atomic via :func:`pawn.checkpoint._atomic_directory_write`: writes go
    to a sibling ``.tmp`` directory and the rename only happens after the
    ``.complete`` sentinel is written. Crashed builds leave a ``.tmp`` dir
    that the next call cleans up.
    """
    import polars as pl

    from pawn.lichess_data import _scan_parquet

    is_parquet_path = str(repo_or_path).endswith(".parquet")
    if is_parquet_path:
        lf = _scan_parquet(parquet_path=repo_or_path, split=split)
    else:
        lf = _scan_parquet(hf_repo=repo_or_path, split=split)

    schema = lf.collect_schema()
    required = {"tokens", "game_length", "outcome_token"}
    missing = required - set(schema.names())
    if missing:
        raise ValueError(
            f"Parquet schema is missing required columns {sorted(missing)}. "
            f"Got: {sorted(schema.names())}. Re-extract with "
            "scripts/extract_lichess_parquet.py."
        )

    if elo_min is not None or elo_max is not None:
        if "white_elo" not in schema or "black_elo" not in schema:
            raise ValueError(
                "Elo filter requires white_elo/black_elo columns; got: "
                f"{list(schema.names())}"
            )
        if elo_min is not None:
            lf = lf.filter(
                (pl.col("white_elo") >= elo_min)
                & (pl.col("black_elo") >= elo_min)
            )
        if elo_max is not None:
            lf = lf.filter(
                (pl.col("white_elo") < elo_max)
                & (pl.col("black_elo") < elo_max)
            )
    if min_ply > 1:
        lf = lf.filter(pl.col("game_length") >= min_ply)

    log.info("Building cache at %s ...", cache_dir)
    print(f"Building cache at {cache_dir} ...", flush=True)
    t0 = time.time()
    df = lf.select(["tokens", "game_length", "outcome_token"]).collect()
    n_games = df.height
    if n_games == 0:
        raise ValueError(
            "No games passed the filters: "
            f"repo={repo_or_path!r}, split={split!r}, "
            f"elo=[{elo_min}, {elo_max}), min_ply={min_ply}"
        )
    log.info("filter+collect: %d games in %.1fs", n_games, time.time() - t0)
    print(
        f"  filter+collect: {n_games:,} games in {time.time() - t0:.1f}s",
        flush=True,
    )

    # Arrow-level CSR extraction: polars stores List<Int16> as a
    # ListArray with contiguous values + offsets. ``Series.to_arrow``
    # returns either an ``Array`` (single-chunk, common case) or a
    # ``ChunkedArray`` (multi-chunk); combine the latter so we can grab
    # ``.values`` / ``.offsets`` directly.
    import pyarrow as pa

    def _as_array(s: pl.Series) -> pa.Array:
        a = s.to_arrow()
        return a.combine_chunks() if isinstance(a, pa.ChunkedArray) else a

    arrow = _as_array(df["tokens"])
    tokens_np = arrow.values.to_numpy(zero_copy_only=False).astype(
        np.int16, copy=False
    )
    offsets_np = arrow.offsets.to_numpy(zero_copy_only=False).astype(
        np.int64, copy=True
    )
    if offsets_np.shape[0] != n_games + 1:
        raise RuntimeError(
            f"offsets length {offsets_np.shape[0]} != n_games+1 {n_games + 1}"
        )

    outcome_np = _as_array(df["outcome_token"]).to_numpy(
        zero_copy_only=False
    ).astype(np.int16, copy=False)

    total_tokens = int(tokens_np.shape[0])

    with _atomic_directory_write(cache_dir) as tmp:
        # Tokens go into a raw binary file so the dataset can ``np.memmap``
        # directly without going through safetensors' tensor abstraction
        # (which doesn't expose mmap-backed reads at the Python level).
        tokens_path = tmp / "tokens_flat.bin"
        with open(tokens_path, "wb") as f:
            f.write(tokens_np.tobytes(order="C"))

        save_file(
            {
                "offsets": torch.from_numpy(offsets_np),
                "outcome_tokens": torch.from_numpy(outcome_np),
            },
            str(tmp / "index.safetensors"),
        )

        meta = {
            "format_version": 1,
            "repo_or_path": repo_or_path,
            "split": split,
            "elo_min": elo_min,
            "elo_max": elo_max,
            "min_ply": min_ply,
            "n_games": int(n_games),
            "total_tokens": total_tokens,
            "created_at_unix": int(time.time()),
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2))

    print(
        f"  cache built: n_games={n_games:,}, total_tokens={total_tokens:,}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _load_cache(cache_dir: Path) -> LichessTokenCache:
    """Load a complete cache.

    The token buffer is mmap-backed via numpy and exposed as a torch
    tensor that shares storage. Reads page-fault and the OS page cache
    is shared across DataLoader workers that re-open the same file.
    """
    _verify_complete_sentinel(cache_dir)

    meta = json.loads((cache_dir / "meta.json").read_text())
    total_tokens = int(meta["total_tokens"])

    # ``torch.from_file`` mmaps the underlying file as the tensor's
    # storage. ``shared=False`` is the read-only contract we want
    # (writes stay process-local). Spawn workers re-call ``_load_cache``
    # and re-mmap the file; the OS page cache shares the underlying
    # physical pages across them.
    tokens_flat = torch.from_file(
        str(cache_dir / "tokens_flat.bin"),
        shared=False,
        size=total_tokens,
        dtype=torch.int16,
    )

    # ``safetensors.torch.load_file`` is also mmap-backed on Linux/macOS
    # — the returned tensors share memory with the file rather than
    # copying it into RSS. A 16M-game cache produces a ~128 MB offsets
    # array and a ~32 MB outcomes array; neither pins resident memory,
    # and spawn workers share the OS page cache the same way as
    # ``tokens_flat.bin``.
    index = load_file(str(cache_dir / "index.safetensors"))
    offsets = index["offsets"]
    outcome_tokens = index["outcome_tokens"]

    n_games = int(meta["n_games"])
    if offsets.shape[0] != n_games + 1:
        raise CheckpointIntegrityError(
            f"offsets shape {tuple(offsets.shape)} disagrees with "
            f"meta.n_games={n_games}"
        )
    if outcome_tokens.shape[0] != n_games:
        raise CheckpointIntegrityError(
            f"outcome_tokens shape {tuple(outcome_tokens.shape)} disagrees "
            f"with meta.n_games={n_games}"
        )
    if tokens_flat.shape[0] != total_tokens:
        raise CheckpointIntegrityError(
            f"tokens_flat shape {tuple(tokens_flat.shape)} disagrees with "
            f"meta.total_tokens={total_tokens}"
        )

    return LichessTokenCache(
        cache_dir=cache_dir,
        tokens_flat=tokens_flat,
        offsets=offsets,
        outcome_tokens=outcome_tokens,
        n_games=n_games,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_lichess_cached(
    repo_or_path: str | Path,
    *,
    split: str = "train",
    elo_min: int | None = None,
    elo_max: int | None = None,
    min_ply: int = 10,
    cache_dir: str | Path | None = None,
) -> LichessTokenCache:
    """Build (or load from cache) a tokenized Lichess subset.

    Cache contents are invariant to ``max_ply`` / ``prepend_outcome``;
    those affect packing only and apply at access time inside
    :class:`IndexedLichessDataset`.
    """
    repo_str = str(repo_or_path)
    root = Path(cache_dir).expanduser() if cache_dir else default_cache_root()
    key = _derive_key(repo_str, split, elo_min, elo_max, min_ply)
    target = root / key

    if (target / ".complete").exists():
        try:
            return _load_cache(target)
        except (CheckpointIntegrityError, IncompleteCheckpointError) as e:
            log.warning(
                "Cache at %s is corrupted (%s); rebuilding.", target, e
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    _build_cache(target, repo_str, split, elo_min, elo_max, min_ply)
    return _load_cache(target)


# ---------------------------------------------------------------------------
# Map-style dataset over a cache
# ---------------------------------------------------------------------------

class IndexedLichessDataset(torch.utils.data.Dataset):
    """Map-style dataset that packs samples on demand from a cache.

    Holding a path-only reference (``cache_dir``) plus a small index
    tensor lets us hand the dataset to spawn workers cheaply: the actual
    cache (mmap'd token buffer + small index tensors) loads lazily inside
    each worker, and the OS shares physical pages across them. The
    parent process loads the cache as soon as it accesses a sample,
    which is fine for the precompute-val-masks pass.

    ``__getitem__`` mirrors :class:`pawn.lichess_data.LichessDataset`'s
    return shape (with ``lazy_pack=False``, the default) so the same
    :class:`pawn.lichess_data.LegalMaskCollate` works without
    modification.

    With ``lazy_pack=True`` the dataset returns only the raw token row
    plus ``game_length`` and ``outcome_token``; packing happens in
    :class:`pawn.lichess_data.BucketedLegalMaskCollate` at a per-batch
    ``T``. This keeps padded attention cost proportional to the actual
    games in each batch instead of the model's full context window.

    Performance per sample (typical adapter training, batch=128 on
    GPU): the per-batch bottleneck is ``compute_legal_indices`` in the
    collate (Rust replay of legal moves), which dominates the small
    cost of slicing the mmap'd token buffer plus a single
    :func:`pack_clm_sequences` call with ``B=1``.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        indices: torch.Tensor,
        max_ply: int,
        prepend_outcome: bool = False,
        lazy_pack: bool = False,
    ):
        from pawn.data import pack_clm_sequences  # local: avoids cycle

        self._pack = pack_clm_sequences
        self.cache_dir = Path(cache_dir)
        if indices.dtype != torch.int64:
            indices = indices.to(torch.int64)
        self.indices = indices
        self.max_ply = int(max_ply)
        self.prepend_outcome = bool(prepend_outcome)
        self.lazy_pack = bool(lazy_pack)
        self._effective_max_moves = (
            self.max_ply - 1 if self.prepend_outcome else self.max_ply
        )

        meta = json.loads((self.cache_dir / "meta.json").read_text())
        self._n_games_total = int(meta["n_games"])
        # ``indices.max()`` raises on an empty tensor; an empty index
        # set is a legitimate (if degenerate) case — e.g. a val carve
        # when the cache is too small — so just skip the bounds check.
        if indices.numel() > 0 and int(indices.max().item()) >= self._n_games_total:
            raise ValueError(
                f"index {int(indices.max().item())} out of range "
                f"for cache with {self._n_games_total} games"
            )

        self._cache: LichessTokenCache | None = None

    # The mmap'd cache is opened lazily so the dataset object pickles
    # cheaply when shipped to spawn workers. Each worker re-mmaps on
    # first access; the OS page cache shares physical pages.
    def __getstate__(self) -> dict[str, Any]:
        s = self.__dict__.copy()
        s["_cache"] = None
        return s

    def _ensure_cache(self) -> LichessTokenCache:
        if self._cache is None:
            self._cache = _load_cache(self.cache_dir)
        return self._cache

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cache = self._ensure_cache()
        gi = int(self.indices[idx].item())
        s = int(cache.offsets[gi].item())
        e = int(cache.offsets[gi + 1].item())
        gl_full = e - s
        gl = min(gl_full, self._effective_max_moves)

        # Raw move row buffer. ``np.zeros`` is allocated afresh per call
        # so the worker's PyTorch DataLoader can pin and transfer it
        # without aliasing into the mmap.
        move_ids_row = np.zeros(self._effective_max_moves, dtype=np.int16)
        if gl > 0:
            move_ids_row[:gl] = (
                cache.tokens_flat[s : s + gl].numpy().astype(np.int16, copy=False)
            )
        outcome = int(cache.outcome_tokens[gi].item())

        if self.lazy_pack:
            # BucketedLegalMaskCollate consumes these directly; packing
            # plus legal-mask Rust replay happen in the collate at a
            # per-batch ``T``.
            return {
                "move_ids": move_ids_row,
                "game_length": gl,
                "outcome_token": outcome,
            }

        outcome_t = torch.tensor([outcome], dtype=torch.long)
        packed = self._pack(
            move_ids_row[None, :],
            np.array([gl], dtype=np.int16),
            outcome_t,
            self.max_ply,
            prepend_outcome=self.prepend_outcome,
        )

        return {
            "input_ids": packed["input_ids"][0],
            "targets": packed["targets"][0],
            "loss_mask": packed["loss_mask"][0],
            "move_ids": move_ids_row,
            "game_length": gl,
        }


# ---------------------------------------------------------------------------
# Helpers used by the trainer for deterministic shuffles and resume
# ---------------------------------------------------------------------------

def shuffled_indices(
    base_indices: torch.Tensor,
    *,
    epoch: int,
    base_seed: int,
) -> torch.Tensor:
    """Return ``base_indices`` permuted by a seed derived from
    ``(base_seed, epoch)``.

    Reproducible across process restarts as long as ``base_seed`` and
    ``epoch`` are persisted. The resulting permutation can be sliced
    after a checkpoint-time ``batches_consumed`` to skip ahead exactly
    on resume — no fast-forward iteration through the DataLoader.
    """
    g = torch.Generator()
    # Mix epoch in so each epoch is a different shuffle; XOR-style mix
    # avoids the two scalars colliding for ``base_seed=0, epoch=k``.
    g.manual_seed((int(base_seed) ^ (int(epoch) * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF)
    perm = torch.randperm(base_indices.shape[0], generator=g)
    return base_indices[perm]


class SeededEpochSampler(torch.utils.data.Sampler[int]):
    """Per-epoch deterministic shuffle sampler with O(1) mid-epoch resume.

    Yields integer indices into the wrapped dataset's ``indices`` array.
    The shuffle for epoch ``E`` depends only on ``(base_seed, E)``, so
    the same shuffle is reproduced across restarts. ``set_epoch`` rotates
    to a new epoch (and an optional ``skip_batches`` slice for resume),
    triggering a fresh permutation on the next ``__iter__``.

    Resume contract: when the trainer checkpoints at ``global_step``, the
    sampler's ``epoch`` is ``global_step // steps_per_epoch`` and
    ``skip_batches`` is ``global_step % steps_per_epoch``. Restoring
    those two values is sufficient — no FF iteration needed.
    """

    def __init__(
        self,
        n_samples: int,
        *,
        base_seed: int,
        batch_size: int,
        drop_last: bool = True,
    ) -> None:
        self._n = int(n_samples)
        self._base_seed = int(base_seed)
        self._batch_size = int(batch_size)
        self._drop_last = bool(drop_last)
        self._epoch = 0
        self._skip_batches = 0

    def set_epoch(self, epoch: int, *, skip_batches: int = 0) -> None:
        if skip_batches < 0:
            raise ValueError(f"skip_batches must be >= 0, got {skip_batches}")
        if skip_batches * self._batch_size > self._n:
            raise ValueError(
                f"skip_batches={skip_batches} × batch_size={self._batch_size} "
                f"exceeds n_samples={self._n}"
            )
        self._epoch = int(epoch)
        self._skip_batches = int(skip_batches)

    def __iter__(self):
        base = torch.arange(self._n)
        perm = shuffled_indices(
            base, epoch=self._epoch, base_seed=self._base_seed
        )
        if self._skip_batches > 0:
            perm = perm[self._skip_batches * self._batch_size :]
        return iter(perm.tolist())

    def __len__(self) -> int:
        remaining = self._n - self._skip_batches * self._batch_size
        if self._drop_last:
            return remaining - (remaining % self._batch_size)
        return remaining

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def skip_batches(self) -> int:
        return self._skip_batches

    @property
    def base_seed(self) -> int:
        return self._base_seed
