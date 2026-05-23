"""JAX-native Lichess data path for adapter training.

PAWN is a testbed for parameter-efficient finetuning; the realistic
adapter task is human-move prediction at a target Elo band. This
module is what makes that task runnable on the JAX adapter trainer —
it reads the canonical pre-tokenized Lichess parquet
(``thomas-schweich/pawn-lichess-full`` or a local file), filters by
Elo band + minimum ply count, packs the result into the same
``pawn.corpus.Corpus`` shape ``scripts/train_jax_adapter.py``
consumes, and caches the packed arrays on disk so the second run with
a given filter is instant.

It replaces the v1 ``pawn/lichess_data.py`` + ``pawn/lichess_cache.py``
pair, which produced PyTorch tensors via ``torch.utils.data`` Dataset
+ bucketed-collate machinery. The v2 path is simpler: the JAX trainer
is shape-static (no bucketed dynamic padding — ``docs/jax-migration.md``
§8.4 removed ``bucket_size``), so this produces fixed-width ``[N, T]``
arrays directly and the on-disk cache stores those arrays verbatim.

Cache layout under ``<cache_root>/<key>/``:
  - ``corpus.safetensors`` — the five packed ``Corpus`` arrays.
  - ``meta.json``          — filter params + game count.
  - ``.complete``          — SHA-256 integrity sentinel
                             (``pawn._sentinel``).
``<cache_root>`` defaults to ``$HF_HOME/pawn-lichess-cache`` (the v1
convention). ``<key>`` is a SHA-256 of the canonical filter params,
so a cache entry is reused iff every filter parameter matches.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from safetensors.numpy import load_file, save_file

from pawn.config import N_OUTCOMES, OUTCOME_TOKEN_BASE
from pawn.corpus import Corpus, pack_corpus

_DEFAULT_HF_REPO = "thomas-schweich/pawn-lichess-full"
_CACHE_FORMAT_VERSION = 1
_CORPUS_FILE = "corpus.safetensors"
_META_FILE = "meta.json"


# ---------------------------------------------------------------------------
# Parquet scan
# ---------------------------------------------------------------------------


def _scan_parquet(
    source: str | Path,
    split: str = "train",
) -> "pl.LazyFrame":
    """Return a polars ``LazyFrame`` over the Lichess parquet ``source``.

    ``source`` is either a local path (file or directory of parquet
    shards) or a HuggingFace dataset repo id (``namespace/name``).
    Local directories with split-prefixed shards
    (``train-XXXXX.parquet`` / ``validation-XXXXX.parquet`` / etc. —
    matching the HF datasets convention) are scanned by split; dirs
    without that structure (or single-file local sources) fall back
    to "scan all parquet files." HF repos are scanned via the
    ``hf://`` protocol; if that fails (older fsspec, auth quirks)
    the shards are downloaded with ``hf_hub_download`` and scanned
    locally.
    """
    src = str(source)

    local = Path(src)
    if local.exists():
        if local.is_dir():
            # Prefer split-prefixed shards when present.
            shards = sorted(local.glob(f"{split}-*.parquet"))
            if not shards:
                shards = sorted(local.glob("*.parquet"))
            if not shards:
                raise FileNotFoundError(
                    f"no parquet files matched {split!r} in {local}"
                )
            return pl.scan_parquet([str(s) for s in shards])
        return pl.scan_parquet(str(local))

    # Treat ``source`` as a HF dataset repo id.
    hf_url = f"hf://datasets/{src}/data/{split}-*.parquet"
    try:
        return pl.scan_parquet(hf_url)
    except Exception:  # pragma: no cover - network/fsspec dependent
        from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415

        api = HfApi()
        files = api.list_repo_files(src, repo_type="dataset")
        shards = [
            f
            for f in files
            if f.endswith(".parquet") and f"/{split}-" in f"/{f}"
        ]
        if not shards:
            shards = [f for f in files if f.endswith(".parquet")]
        if not shards:
            raise FileNotFoundError(
                f"no parquet shards found in HF dataset repo {src!r}"
            )
        local_files = [
            hf_hub_download(src, shard, repo_type="dataset")
            for shard in shards
        ]
        return pl.scan_parquet(local_files)


# ---------------------------------------------------------------------------
# Cache key + paths
# ---------------------------------------------------------------------------


def _default_cache_root() -> Path:
    """``$HF_HOME/pawn-lichess-cache`` — the v1 convention. Falls back
    to ``~/.cache/huggingface`` when ``HF_HOME`` is unset."""
    hf_home = os.environ.get("HF_HOME")
    base = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    return base / "pawn-lichess-cache"


def _cache_key(
    source: str,
    split: str,
    elo_min: int | None,
    elo_max: int | None,
    min_ply: int,
    seq_len: int,
    max_games: int | None,
) -> str:
    """SHA-256 over the canonical filter params. Any change to a
    parameter that affects the packed arrays changes the key — so a
    stale cache is never silently reused."""
    payload = json.dumps(
        {
            "v": _CACHE_FORMAT_VERSION,
            "source": source,
            "split": split,
            "elo_min": elo_min,
            "elo_max": elo_max,
            "min_ply": min_ply,
            "seq_len": seq_len,
            "max_games": max_games,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Corpus build
# ---------------------------------------------------------------------------


def _build_corpus_from_parquet(
    source: str,
    *,
    split: str,
    elo_min: int | None,
    elo_max: int | None,
    min_ply: int,
    seq_len: int,
    max_games: int | None,
) -> Corpus:
    """Scan, filter, and pack the Lichess parquet into a ``Corpus``."""
    lf = _scan_parquet(source, split)
    schema = lf.collect_schema()
    names = set(schema.names())

    required = {"tokens", "game_length", "outcome_token"}
    missing = required - names
    if missing:
        raise ValueError(
            f"Lichess parquet {source!r} is missing required columns "
            f"{sorted(missing)} (got {sorted(names)}). Re-extract with "
            "scripts/extract_lichess_parquet.py — the canonical schema "
            "is pure-moves `tokens` + `game_length` + `outcome_token` "
            "+ per-game metadata."
        )

    # Elo filter — both players within the band. ``elo_max`` is
    # exclusive so adjacent bands ([1600,1800), [1800,2000)) tile
    # without overlap.
    if elo_min is not None or elo_max is not None:
        if "white_elo" not in names or "black_elo" not in names:
            raise ValueError(
                "Elo filtering needs white_elo / black_elo columns; "
                f"parquet {source!r} has {sorted(names)}"
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

    # Minimum-ply filter — drop games too short to be informative.
    if min_ply > 0:
        lf = lf.filter(pl.col("game_length") >= min_ply)

    lf = lf.select(["tokens", "game_length", "outcome_token"])
    if max_games is not None:
        lf = lf.head(max_games)
    df = lf.collect()

    n = df.height
    if n == 0:
        raise ValueError(
            f"Lichess filter (elo_min={elo_min}, elo_max={elo_max}, "
            f"min_ply={min_ply}) matched 0 games in {source!r}"
        )

    token_lists = df["tokens"].to_list()
    game_lengths_list = df["game_length"].to_list()
    outcome_tokens_list = df["outcome_token"].to_list()

    # Pack the variable-length token lists into a fixed [N, seq_len]
    # int32 matrix. Games longer than seq_len are truncated to
    # seq_len moves (game_length is clamped to match).
    move_ids = np.zeros((n, seq_len), dtype=np.int32)
    game_lengths = np.zeros(n, dtype=np.int32)
    for i, toks in enumerate(token_lists):
        gl = min(int(game_lengths_list[i]), seq_len, len(toks))
        if gl > 0:
            move_ids[i, :gl] = np.asarray(toks[:gl], dtype=np.int32)
        game_lengths[i] = gl

    # outcome_token (absolute ID 1969..1979) → offset [0, N_OUTCOMES).
    # Clip defensively: a parquet row with an out-of-band outcome is a
    # data error, but clamping keeps the run alive (the offset is only
    # used for optional outcome-prefix conditioning).
    outcome_offset = np.clip(
        np.asarray(outcome_tokens_list, dtype=np.int64) - OUTCOME_TOKEN_BASE,
        0,
        N_OUTCOMES - 1,
    ).astype(np.uint8)

    return pack_corpus(
        move_ids, game_lengths, outcome_offset, seq_len=seq_len
    )


# ---------------------------------------------------------------------------
# Cache read / write
# ---------------------------------------------------------------------------


def _save_corpus(directory: Path, corpus: Corpus, meta: dict[str, Any]) -> None:
    """Write a packed ``Corpus`` + meta + ``.complete`` sentinel.

    The atomic-write contract mirrors ``pawn.checkpoint``: payload
    files land first, the sentinel last, so a sentinel-present
    directory is always integrity-checkable.
    """
    from pawn._sentinel import write_sentinel  # noqa: PLC0415

    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "tokens": corpus.tokens,
            "attn_mask": corpus.attn_mask,
            "targets": corpus.targets,
            "loss_mask": corpus.loss_mask,
            "outcome_offset": corpus.outcome_offset,
        },
        str(directory / _CORPUS_FILE),
    )
    (directory / _META_FILE).write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_sentinel(directory)


def _load_cached_corpus(directory: Path) -> Corpus:
    """Verify the sentinel and load the packed ``Corpus``."""
    from pawn._sentinel import verify_sentinel  # noqa: PLC0415

    verify_sentinel(directory)
    arrays = load_file(str(directory / _CORPUS_FILE))
    return Corpus(
        tokens=arrays["tokens"].astype(np.int32),
        attn_mask=arrays["attn_mask"].astype(np.bool_),
        targets=arrays["targets"].astype(np.int32),
        loss_mask=arrays["loss_mask"].astype(np.bool_),
        outcome_offset=arrays["outcome_offset"].astype(np.uint8),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_lichess_corpus(
    source: str = _DEFAULT_HF_REPO,
    *,
    split: str = "train",
    elo_min: int | None = None,
    elo_max: int | None = None,
    min_ply: int = 10,
    seq_len: int = 512,
    max_games: int | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> Corpus:
    """Load an Elo-filtered Lichess slice as a trainer-ready ``Corpus``.

    The first call with a given ``(source, split, elo_min, elo_max,
    min_ply, seq_len, max_games)`` tuple scans + filters + packs the
    parquet and writes the packed arrays to the on-disk cache; later
    calls with the same filter reload the cache (sentinel-verified)
    instead of re-scanning.

    Args:
        source: HF dataset repo id (default
            ``thomas-schweich/pawn-lichess-full``) or a local parquet
            file / directory.
        split: parquet split name (``train`` / ``test`` / ...).
        elo_min / elo_max: Elo band — both players must satisfy
            ``elo_min <= elo < elo_max`` (``elo_max`` exclusive so
            adjacent bands tile cleanly). ``None`` disables that bound.
        min_ply: drop games with fewer than ``min_ply`` moves.
        seq_len: packed tensor width; games longer than this are
            truncated.
        max_games: cap on games loaded (``None`` = no cap — the whole
            filtered slice).
        cache_dir: cache root; defaults to
            ``$HF_HOME/pawn-lichess-cache``.
        use_cache: set ``False`` to bypass the cache entirely (always
            rebuild, never write).

    Returns:
        ``Corpus`` — the finite filtered slice. The adapter trainer
        splits it train/val and tiles it across epochs with
        ``make_epoch_schedule``.
    """
    root = Path(cache_dir) if cache_dir is not None else _default_cache_root()
    key = _cache_key(
        str(source), split, elo_min, elo_max, min_ply, seq_len, max_games
    )
    cache_path = root / key

    if use_cache:
        from pawn._sentinel import (  # noqa: PLC0415
            CheckpointIntegrityError,
            IncompleteCheckpointError,
        )

        try:
            return _load_cached_corpus(cache_path)
        except (IncompleteCheckpointError, FileNotFoundError):
            pass  # no cache entry yet — build it below
        except CheckpointIntegrityError:
            # A corrupt cache entry shouldn't wedge the run — rebuild.
            pass

    corpus = _build_corpus_from_parquet(
        str(source),
        split=split,
        elo_min=elo_min,
        elo_max=elo_max,
        min_ply=min_ply,
        seq_len=seq_len,
        max_games=max_games,
    )

    if use_cache:
        meta = {
            "format_version": _CACHE_FORMAT_VERSION,
            "source": str(source),
            "split": split,
            "elo_min": elo_min,
            "elo_max": elo_max,
            "min_ply": min_ply,
            "seq_len": seq_len,
            "max_games": max_games,
            "n_games": corpus.n_games,
        }
        try:
            _save_corpus(cache_path, corpus, meta)
        except OSError:
            # A read-only / full cache disk is non-fatal — the run
            # proceeds with the freshly-built in-memory corpus.
            pass

    return corpus


def make_epoch_schedule(
    n_pool: int,
    n_needed: int,
    *,
    seed: int,
) -> NDArray[np.int64]:
    """Build an index schedule that tiles a finite game pool across
    epochs with a fresh permutation each epoch.

    The JAX adapter trainer consumes a fixed ``total_steps *
    batch_size`` games in a single ``lax.scan`` pass, but a Lichess
    Elo slice is finite — usually smaller than that. This repeats the
    pool, shuffling it independently each epoch (``docs/jax-migration.md``
    §6 "per-epoch index permutation"), and returns exactly
    ``n_needed`` indices into ``[0, n_pool)``.

    Args:
        n_pool: number of distinct games available.
        n_needed: number of (possibly repeated) game slots required.
        seed: base RNG seed; epoch ``e`` is permuted with ``seed + e``.

    Returns:
        ``int64[n_needed]`` — indices into the game pool.
    """
    if n_pool <= 0:
        raise ValueError(f"n_pool must be > 0, got {n_pool}")
    if n_needed < 0:
        raise ValueError(f"n_needed must be >= 0, got {n_needed}")
    chunks: list[NDArray[np.int64]] = []
    filled = 0
    epoch = 0
    while filled < n_needed:
        rng = np.random.default_rng(seed + epoch)
        perm = rng.permutation(n_pool).astype(np.int64)
        chunks.append(perm)
        filled += n_pool
        epoch += 1
    return np.concatenate(chunks)[:n_needed]


__all__ = [
    "load_lichess_corpus",
    "make_epoch_schedule",
]
