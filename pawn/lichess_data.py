"""Lichess parquet → Corpus loader with cache + multi-epoch tiling.

The realistic adapter task is Elo-stratified human-move prediction
(plan §11). This module is the on-disk bridge from the Lichess parquet
shards (HF dataset ``thomas-schweich/pawn-lichess-full`` or local
directories produced by ``scripts/extract_lichess_parquet.py``) to a
:class:`pawn.corpus.Corpus` ready for JAX consumption.

The plan calls out:

- **Polars scan + Elo filter** (both players' ELOs in
  ``[elo_min, elo_max)`` — elo_min inclusive, elo_max exclusive).
- **min_ply filter** — drop games shorter than ``min_ply`` moves.
- **Split-aware** for HF repos and for local directories with
  ``train-*.parquet`` / ``validation-*.parquet`` shards.
- **Disk cache** under ``$HF_HOME/pawn-lichess-cache/<sha>`` where
  ``<sha>`` is the SHA-256 of the filter params. Subsequent runs with
  the same params mmap the cache; a different filter → different cache
  entry. Integrity guarded by :mod:`pawn._sentinel`.
- **``make_epoch_schedule(n_pool, n_needed, seed)``** for tiling the
  finite training pool across multiple epochs with a per-epoch
  permutation.

The parquet schema is the canonical one written by
``scripts/extract_lichess_parquet.py``:

- ``tokens``: ``list[int]`` — pre-tokenised moves (1968-token vocab).
- ``game_length``: ``int`` — number of moves.
- ``outcome_token``: ``int`` — pre-classified outcome token ID.
- ``white_elo`` / ``black_elo``: ``int`` — used for filtering.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray
from safetensors.numpy import load_file as st_load
from safetensors.numpy import save_file as st_save

from pawn._sentinel import (
    CheckpointIntegrityError,
    IncompleteCheckpointError,
    verify_sentinel,
    write_sentinel,
)
from pawn.config import MASK_VERSION, PAD_TOKEN
from pawn.corpus import Corpus, conditioning_to_C, pack_corpus

__all__ = [
    "load_lichess_corpus",
    "make_epoch_schedule",
]


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------

_CACHE_FILES = ("corpus.safetensors",)
# Bumped to 2 in Phase-A Chunk 4: the on-disk corpus layout changed to
# ``[BOS][cond…][ply…]`` with the first-move-supervised loss mask, and the
# cache key now folds in the conditioning-derived ``C`` + ``mask_version``.
# A v1 cache entry is byte-incompatible, so the version bump alone forces a
# rebuild (distinct key) even before the new params are considered.
_CACHE_VERSION = 2


def _default_cache_root() -> Path:
    """Resolve ``$HF_HOME/pawn-lichess-cache``, defaulting to
    ``~/.cache/huggingface/pawn-lichess-cache`` when ``HF_HOME`` is
    unset (matches the huggingface-hub default).
    """
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "pawn-lichess-cache"
    return Path.home() / ".cache" / "huggingface" / "pawn-lichess-cache"


def _cache_key(
    source: str,
    split: str,
    elo_min: int | None,
    elo_max: int | None,
    min_ply: int,
    seq_len: int,
    max_games: int | None,
    conditioning: Sequence[str],
) -> str:
    """SHA-256 of the filter params, hex-encoded. Same inputs → same hash
    → same cache entry. Any change in any of these params produces a
    different cache directory; the user pays the filter+pack cost once.

    The ordered ``conditioning`` list, its derived prefix width ``C``, and
    the global ``mask_version`` all fold into the key: a corpus packed
    under one prefix/loss-mask contract must never be served to a run
    that expects another (silent absolute-RoPE drift).
    """
    payload = json.dumps(
        {
            "version": _CACHE_VERSION,
            "mask_version": MASK_VERSION,
            "source": source,
            "split": split,
            "elo_min": elo_min,
            "elo_max": elo_max,
            "min_ply": min_ply,
            "seq_len": seq_len,
            "max_games": max_games,
            "conditioning": list(conditioning),
            "C": conditioning_to_C(conditioning),
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Parquet scan
# ---------------------------------------------------------------------------


def _scan_parquet(source: str, split: str) -> pl.LazyFrame:
    """Build a Polars LazyFrame over the parquet shards matching ``split``.

    Two source forms:

    - HF dataset repo ID (e.g. ``"thomas-schweich/pawn-lichess-full"``):
      try the native ``hf://`` lazy-scan; fall back to downloading the
      relevant shards via :mod:`huggingface_hub` if the native scan
      fails.

    - Local directory: scan ``<split>-*.parquet`` inside it. If no
      split-prefixed shards exist, fall back to all ``*.parquet``
      (single-file datasets where the user passes ``--pgn-val-split ""``
      for carve-from-train).
    """
    path = Path(source)
    if path.is_dir():
        candidates = sorted(path.glob(f"{split}-*.parquet"))
        if not candidates:
            candidates = sorted(path.glob("*.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"no parquet files under {path} matching split={split!r}"
            )
        return pl.scan_parquet([str(p) for p in candidates])

    hf_url = f"hf://datasets/{source}/data/{split}-*.parquet"
    try:
        return pl.scan_parquet(hf_url)
    # Narrow exception scope: only catch errors plausibly stemming from
    # the hf:// scheme being unsupported on this polars build / the
    # data/ prefix being wrong on this dataset layout. Network errors,
    # auth failures, malformed parquet bodies — those propagate.
    except (OSError, ValueError, FileNotFoundError, RuntimeError) as exc:
        # The native hf:// lazy scan failed. The fallback below downloads
        # EVERY shard for the split via huggingface_hub — for
        # `pawn-lichess-full` that is 262 GB. Unlike the lazy scan (which
        # range-streams only the projected columns + sliced row-groups it
        # needs), this pulls whole files, so it must never run implicitly:
        # a single failed scan would silently saturate a metered link.
        # Gate it behind an explicit opt-in; default to a clear error.
        if os.environ.get("PAWN_ALLOW_BULK_DOWNLOAD") != "1":
            raise RuntimeError(
                f"Native hf:// lazy scan of dataset {source!r} "
                f"(split={split!r}) failed: {type(exc).__name__}: {exc}.\n"
                "The fallback that downloads every shard via "
                "huggingface_hub is disabled by default — it can pull the "
                "entire dataset (pawn-lichess-full is 262 GB) over the "
                "network. To opt in, re-run with "
                "PAWN_ALLOW_BULK_DOWNLOAD=1. Better: pre-download the "
                "shard(s) you need and pass a local directory as --pgn, or "
                "fix the hf:// scan so it streams lazily."
            ) from exc

        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        files = api.list_repo_files(source, repo_type="dataset")
        parquet_files = [
            f for f in files
            if f.endswith(".parquet") and f"/{split}-" in f"/{f}"
        ]
        if not parquet_files:
            parquet_files = [f for f in files if f.endswith(".parquet")]
        if not parquet_files:
            raise FileNotFoundError(
                f"HF repo {source!r} has no parquet files for split={split!r}"
            ) from None
        local_files = [
            hf_hub_download(source, pf, repo_type="dataset")
            for pf in parquet_files
        ]
        return pl.scan_parquet(local_files)


_REQUIRED_COLS = ("tokens", "game_length", "outcome_token")


def _filter_lichess(
    lf: pl.LazyFrame,
    *,
    elo_min: int | None,
    elo_max: int | None,
    min_ply: int,
    max_games: int | None,
) -> pl.DataFrame:
    """Apply Elo / min_ply / max_games filters and collect to memory."""
    schema = lf.collect_schema()
    required: set[str] = set(_REQUIRED_COLS)
    missing = required - set(schema.names())
    if missing:
        raise ValueError(
            f"Lichess parquet schema is missing required columns "
            f"{sorted(missing)}. Got: {sorted(schema.names())}. Re-extract "
            f"the dataset with scripts/extract_lichess_parquet.py."
        )
    if elo_min is not None or elo_max is not None:
        if "white_elo" not in schema or "black_elo" not in schema:
            raise ValueError(
                "Elo filtering requires white_elo/black_elo columns in the "
                f"parquet schema, got: {sorted(schema.names())}"
            )
        if elo_min is not None:
            lf = lf.filter(
                (pl.col("white_elo") >= elo_min)
                & (pl.col("black_elo") >= elo_min)
            )
        if elo_max is not None:
            # elo_max is exclusive on both sides per the v1 contract.
            lf = lf.filter(
                (pl.col("white_elo") < elo_max)
                & (pl.col("black_elo") < elo_max)
            )

    if min_ply > 0:
        lf = lf.filter(pl.col("game_length") >= min_ply)

    lf = lf.select(list(_REQUIRED_COLS))
    if max_games is not None:
        lf = lf.head(max_games)
    return lf.collect()


def _pack_dataframe(
    df: pl.DataFrame,
    *,
    seq_len: int,
    conditioning: Sequence[str],
) -> Corpus:
    """Turn a filtered DataFrame into a :class:`Corpus` via
    :func:`pawn.corpus.pack_corpus`."""
    if df.is_empty():
        raise ValueError(
            "filtered Lichess parquet produced 0 games — check elo_min / "
            "elo_max / min_ply filters"
        )
    token_lists: Sequence[Sequence[int]] = df["tokens"].to_list()
    game_lengths_raw = np.asarray(df["game_length"].to_list(), dtype=np.int32)
    outcome_tokens = np.asarray(df["outcome_token"].to_list(), dtype=np.int32)
    n = len(token_lists)
    max_per_row = max((len(t) for t in token_lists), default=0)
    # Defensive: a malformed parquet row where `game_length > len(tokens)`
    # would make `_pack_clm` declare PAD slots as "valid moves"
    # (attn_mask True at PAD positions). Clip game_lengths to the
    # actual per-row token count so a schema violation degrades to a
    # shorter game rather than silent corruption.
    per_row_token_count = np.array(
        [len(t) for t in token_lists], dtype=np.int32
    )
    game_lengths = np.minimum(game_lengths_raw, per_row_token_count)
    max_ply = max(max_per_row, int(game_lengths.max(initial=0)))
    move_ids = np.full((n, max_ply), PAD_TOKEN, dtype=np.int16)
    for i, toks in enumerate(token_lists):
        if toks:
            move_ids[i, : len(toks)] = np.asarray(toks, dtype=np.int16)
    return pack_corpus(
        move_ids,
        game_lengths,
        outcome_tokens,
        seq_len=seq_len,
        conditioning=conditioning,
    )


# ---------------------------------------------------------------------------
# Cache write/read
# ---------------------------------------------------------------------------


def _save_to_cache(corpus: Corpus, cache_dir: Path) -> None:
    """Atomically write a Corpus to the cache directory.

    Mirror of :func:`pawn.checkpoint.save_model`'s atomic flow:
    ``<dir>.tmp`` staging, sentinel written into it, POSIX-atomic
    rename. On any exception the partial ``.tmp`` is cleaned up.
    """
    final = cache_dir
    tmp = final.with_name(final.name + ".tmp")
    if tmp.is_dir():
        shutil.rmtree(tmp)
    elif tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        tensors = {
            "tokens": np.asarray(corpus.tokens),
            "targets": np.asarray(corpus.targets),
            "attn_mask": np.asarray(corpus.attn_mask),
            "loss_mask": np.asarray(corpus.loss_mask),
            "outcome_offset": np.asarray(corpus.outcome_offset),
            "game_lengths": np.asarray(corpus.game_lengths),
        }
        st_save(tensors, str(tmp / "corpus.safetensors"))
        write_sentinel(tmp, _CACHE_FILES)
        os.rename(tmp, final)
    except BaseException:
        if tmp.is_dir():
            shutil.rmtree(tmp, ignore_errors=True)
        raise


def _load_from_cache(cache_dir: Path) -> Corpus:
    """Verify the sentinel, then materialise a Corpus from the cached
    safetensors. Raises if the cache directory is missing the
    sentinel or any payload is corrupt / incomplete."""
    verify_sentinel(cache_dir)
    raw = st_load(str(cache_dir / "corpus.safetensors"))
    needed = {"tokens", "targets", "attn_mask", "loss_mask", "outcome_offset"}
    missing = needed - set(raw.keys())
    if missing:
        raise CheckpointIntegrityError(
            f"cache at {cache_dir} missing tensors {sorted(missing)}"
        )
    # game_lengths is normally present (every cache written since the A.1
    # bucketing work saves it). Recovery path: under the Chunk-4 loss-mask
    # contract the supervised positions are exactly
    # ``[C-1 .. C-1 + game_length - 1]``, so ``loss_mask.sum(axis=-1) ==
    # game_length`` regardless of the prefix width — no ``outcome_offset``
    # correction needed. (Pre-Chunk-4 caches live under a different cache
    # key after the _CACHE_VERSION bump, so they're never read here.)
    if "game_lengths" in raw:
        game_lengths = raw["game_lengths"].astype(np.int32)
    else:
        loss_count = raw["loss_mask"].astype(np.int32).sum(axis=-1)
        game_lengths = loss_count.astype(np.int32)
    # Stay on host — Corpus is host-side numpy; the trainer transfers
    # per-batch.
    return Corpus(
        tokens=raw["tokens"],
        targets=raw["targets"],
        attn_mask=raw["attn_mask"],
        loss_mask=raw["loss_mask"],
        outcome_offset=raw["outcome_offset"],
        game_lengths=game_lengths,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_lichess_corpus(
    source: str,
    *,
    split: str = "train",
    elo_min: int | None = None,
    elo_max: int | None = None,
    min_ply: int = 10,
    seq_len: int = 512,
    max_games: int | None = None,
    conditioning: Sequence[str] = (),
    cache_dir: str | Path | None = None,
) -> Corpus:
    """Load a Lichess parquet slice into a :class:`Corpus`, cached on disk.

    ``source`` is either a HuggingFace dataset repo ID (e.g.
    ``"thomas-schweich/pawn-lichess-full"``) or a local directory
    holding ``<split>-*.parquet`` shards.

    ``split`` defaults to ``"train"``; pass ``"validation"`` for the
    held-out split (the plan §11 default for adapter val).

    Filters: ``elo_min`` (inclusive, both players) / ``elo_max``
    (exclusive, both players) / ``min_ply`` (drop short games) /
    ``max_games`` (take the first N after filtering).

    ``conditioning`` is the ordered list of control kinds to prepend
    (see :data:`pawn.config.CONDITIONING_KINDS`); ``["outcome"]`` is the
    layout the generation diagnostics in S8 require. The default ``()``
    prepends only BOS (``C = 1``).

    Caches the resulting :class:`Corpus` under
    ``$HF_HOME/pawn-lichess-cache/<sha>/`` (override via ``cache_dir``).
    The cache key is the SHA-256 of all filter params **plus** the
    conditioning-derived ``C`` and the global ``mask_version`` — change
    any one and you get a fresh cache entry. The first run does the
    filter+pack work; subsequent runs verify the sentinel and load
    directly.
    """
    cache_root = Path(cache_dir) if cache_dir is not None else _default_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    sha = _cache_key(
        source, split, elo_min, elo_max, min_ply, seq_len, max_games, conditioning
    )
    final_dir = cache_root / sha

    if final_dir.is_dir():
        try:
            return _load_from_cache(final_dir)
        except (IncompleteCheckpointError, CheckpointIntegrityError):
            # Corrupted entry (interrupted write, manual edit) — rebuild.
            shutil.rmtree(final_dir, ignore_errors=True)

    lf = _scan_parquet(source, split)
    df = _filter_lichess(
        lf, elo_min=elo_min, elo_max=elo_max, min_ply=min_ply, max_games=max_games
    )
    corpus = _pack_dataframe(df, seq_len=seq_len, conditioning=conditioning)
    _save_to_cache(corpus, final_dir)
    return corpus


# ---------------------------------------------------------------------------
# Multi-epoch tiling
# ---------------------------------------------------------------------------


def make_epoch_schedule(
    n_pool: int, n_needed: int, seed: int
) -> NDArray[np.int64]:
    """Tile a finite training pool of ``n_pool`` games across multiple
    epochs to produce ``n_needed`` indices.

    Each epoch is a fresh permutation of ``[0, n_pool)``. Successive
    epochs concatenate; if ``n_needed`` doesn't fall on an epoch
    boundary the final epoch is truncated. The deterministic
    permutation per epoch (seeded off ``seed + epoch_index``) means
    a resumed run that re-derives the schedule from the same seed
    sees the same indices.

    Returns an ``int64`` array of length ``n_needed``.
    """
    if n_pool <= 0:
        raise ValueError(f"n_pool must be positive, got {n_pool}")
    if n_needed < 0:
        raise ValueError(f"n_needed must be non-negative, got {n_needed}")
    if n_needed == 0:
        return np.array([], dtype=np.int64)
    n_epochs = (n_needed + n_pool - 1) // n_pool  # ceil
    chunks: list[NDArray[np.int64]] = []
    for epoch in range(n_epochs):
        rng = np.random.default_rng(seed + epoch)
        chunks.append(rng.permutation(n_pool).astype(np.int64))
    full = np.concatenate(chunks)
    return full[:n_needed]
