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
    "load_lichess_train_val",
    "make_epoch_schedule",
    "split_has_files",
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
    """Resolve the on-disk Lichess cache root.

    Resolution order (matches the v1 ``PAWN_DATA_CACHE`` escape hatch):

      1. ``$PAWN_DATA_CACHE`` — used verbatim (expanded), no subdir appended.
         This is the operator override for routing the cache onto a roomy
         scratch volume independent of ``$HF_HOME``.
      2. ``$HF_HOME/pawn-lichess-cache``.
      3. ``~/.cache/huggingface/pawn-lichess-cache`` (the huggingface-hub
         default) when neither env var is set.
    """
    env = os.environ.get("PAWN_DATA_CACHE")
    if env:
        return Path(env).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "pawn-lichess-cache"
    return Path.home() / ".cache" / "huggingface" / "pawn-lichess-cache"


def split_has_files(source: str, split: str) -> bool:
    """Probe whether ``split`` has parquet shards available under ``source``.

    This is the carve-vs-held-out decision gate. ``load_lichess_train_val``
    (and the adapter trainer) call it to decide whether a real held-out
    validation split exists or whether validation must be carved out of the
    training games instead.

    Three source forms:

    - **Single local ``.parquet`` file** — no split semantics, so only
      ``split == "train"`` is satisfiable; any other split returns ``False``
      (and the caller carves from train).
    - **HF dataset repo ID** (``"user/dataset"``) — list the repo's files
      once and look for any ``.parquet`` whose path contains ``/<split>-``.
      A network/listing failure conservatively returns ``False`` rather than
      letting :func:`_scan_parquet`'s "fall back to all parquet files" branch
      silently load the *train* shards as validation.
    - **Local directory / glob** — match ``<split>-*.parquet`` on disk.

    Cheap (one ``list_repo_files`` call for HF); the right guard before
    attempting to build a validation cache.
    """
    s = str(source)

    if s.endswith(".parquet") and Path(s).exists():
        # A single parquet file has no per-split structure.
        return split == "train"

    if "/" in s and not Path(s).exists():
        try:
            from huggingface_hub import HfApi

            files = HfApi().list_repo_files(s, repo_type="dataset")
        except Exception:
            return False
        marker = f"/{split}-"
        return any(
            f.endswith(".parquet") and marker in f"/{f}" for f in files
        )

    # Local directory or glob pattern.
    p = Path(s)
    if p.is_dir():
        return bool(list(p.glob(f"{split}-*.parquet")))
    import glob as _glob

    matches = _glob.glob(s)
    return any(split in Path(m).name for m in matches)


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

    - Local directory: scan ``<split>-*.parquet`` inside it. When the
      requested split is ``"train"`` and no ``train-*.parquet`` shards exist,
      fall back to all ``*.parquet`` (single-file / split-less datasets where
      the user carves validation out of train). A non-``train`` split with no
      matching shards raises ``FileNotFoundError`` rather than falling back to
      the train files — otherwise an explicit ``--pgn-val-split validation``
      against a split-less directory would silently load the *train* shards as
      validation (identical train/val leak).
    """
    path = Path(source)
    if path.is_dir():
        candidates = sorted(path.glob(f"{split}-*.parquet"))
        if not candidates and split == "train":
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
        # Same carve-from-train guard as the local-dir branch: only the
        # ``train`` split may fall back to "all parquet files" (a split-less
        # repo where validation is carved out of train). A non-``train`` split
        # with no matching shards must raise rather than silently serve the
        # train shards as validation.
        if not parquet_files and split == "train":
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


def _index_corpus(corpus: Corpus, idx: NDArray[np.int64]) -> Corpus:
    """Return the sub-Corpus selecting rows ``idx`` from every field array.

    ``Corpus`` is a frozen numpy-backed dataclass (not a pytree); every field
    is an ``(N, …)`` host array sharing the same leading game axis, so a single
    fancy-index over that axis carves a consistent subset across ``tokens`` /
    ``targets`` / masks / ``outcome_offset`` / ``game_lengths``.
    """
    return Corpus(
        tokens=np.asarray(corpus.tokens)[idx],
        targets=np.asarray(corpus.targets)[idx],
        attn_mask=np.asarray(corpus.attn_mask)[idx],
        loss_mask=np.asarray(corpus.loss_mask)[idx],
        outcome_offset=np.asarray(corpus.outcome_offset)[idx],
        game_lengths=np.asarray(corpus.game_lengths)[idx],
    )


def load_lichess_train_val(
    source: str,
    *,
    val_split: str | None = None,
    elo_min: int | None = None,
    elo_max: int | None = None,
    min_ply: int = 10,
    seq_len: int = 512,
    max_games: int | None = None,
    max_val_games: int | None = None,
    conditioning: Sequence[str] = (),
    cache_dir: str | Path | None = None,
    seed: int = 0,
) -> tuple[Corpus, Corpus]:
    """Load train + validation corpora, honouring the v1 carve/held-out gate.

    Validation source selection (mirrors v1 ``scripts/train.py``):

    1. If ``val_split`` names a non-empty split, it is **used directly** — the
       caller asked for a specific held-out split, so it is loaded via
       :func:`load_lichess_corpus` and a genuinely absent split raises
       ``FileNotFoundError`` rather than silently degrading to carve. (The
       :func:`split_has_files` probe is deliberately *not* consulted here: it
       returns ``False`` on any transient listing/auth failure or on a dataset
       whose shards don't match the ``/<split>-`` naming convention, which
       would turn a deliberate held-out request into a silent carve of
       validation out of the training set.)
    2. Else (``val_split`` is ``None`` — the implicit default), when the
       conventional ``"validation"`` split actually has files
       (:func:`split_has_files`), that held-out split is used.
    3. Otherwise — ``val_split`` is ``None`` with no ``validation`` shards, or
       ``val_split == ""`` (explicit carve-from-train) — a **deterministic,
       disjoint** slice is carved out of the training games. The carve uses a
       seeded permutation, so train and val never share a game and the split is
       reproducible across runs.

    Carve size matches the v1 cap formula (``scripts/train.py:427``):
    ``n_val = min(max_val_games, n_total // 5)`` — a 20% floor-division carve
    capped by ``max_val_games`` when given. (v1 took the trailing-window slice
    ``arange(n_train, n_total)``; v2 instead draws a seeded random permutation
    so the carve is reproducible without coupling to corpus row order, but the
    *count* is the same as v1.) A final ``min(n_val, n_total - 1)`` safety clamp
    keeps at least one training game.

    The carved validation slice and the returned training corpus are
    guaranteed disjoint, fixing the v1→v2 regression where ``--pgn-val-split
    ""`` on a single-file source leaked the entire dataset into val (identical
    train/val) via an ``or`` short-circuit that fell back to the ``train``
    shards.
    """
    train_corpus = load_lichess_corpus(
        source,
        split="train",
        elo_min=elo_min,
        elo_max=elo_max,
        min_ply=min_ply,
        seq_len=seq_len,
        max_games=max_games,
        conditioning=conditioning,
        cache_dir=cache_dir,
    )

    # Decide between a real held-out split and carve-from-train.
    #
    # * ``val_split == ""`` is the explicit carve-from-train request (v1
    #   ``--pgn-val-split ""``): never load a held-out split, always carve.
    # * A non-empty ``val_split`` is an explicit held-out-split request: load
    #   it directly. We do NOT route it through ``split_has_files`` — that probe
    #   returns ``False`` on a transient HF listing/auth failure or a dataset
    #   whose shards don't match the ``/<split>-`` convention, which would
    #   silently degrade a deliberate held-out request into a carve of
    #   validation out of the training set (the exact silent-contamination
    #   footgun). ``load_lichess_corpus`` raises ``FileNotFoundError`` when the
    #   named split genuinely has no shards, so a typo/absent split surfaces
    #   loudly instead.
    # * ``val_split is None`` is the implicit default. Here we *probe* the
    #   conventional "validation" split via ``split_has_files`` and fall through
    #   to carve on a miss — a single-file or split-less source has no held-out
    #   shards, so this is where carve-from-train is the right default.
    if val_split == "":
        chosen_split: str | None = None
    elif val_split:
        # Explicit named split — load directly, let an absent split raise.
        chosen_split = val_split
    else:
        chosen_split = (
            "validation" if split_has_files(source, "validation") else None
        )

    if chosen_split is not None:
        val_corpus = load_lichess_corpus(
            source,
            split=chosen_split,
            elo_min=elo_min,
            elo_max=elo_max,
            min_ply=min_ply,
            seq_len=seq_len,
            max_games=max_val_games,
            conditioning=conditioning,
            cache_dir=cache_dir,
        )
        return train_corpus, val_corpus

    # Carve a deterministic, disjoint validation tail out of train.
    n_total = int(train_corpus.tokens.shape[0])
    if n_total < 2:
        raise ValueError(
            "carve-from-train needs at least 2 games to produce a disjoint "
            f"train/val split, got {n_total}. Provide a held-out split via "
            "--pgn-val-split or loosen the Elo / min_ply filters."
        )
    # v1 cap formula (scripts/train.py:427): n_val = min(val_games, n_total//5).
    # The 20% floor-division is the uncapped ceiling; max_val_games (the v2
    # name for v1's val_games) caps it tighter when supplied.
    n_val = n_total // 5
    if max_val_games is not None:
        n_val = min(n_val, max_val_games)
    # Keep at least one val game (floor-division yields 0 for n_total < 5) and
    # never let val consume every game — keep at least one training game.
    n_val = max(1, min(n_val, n_total - 1))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    val_corpus = _index_corpus(train_corpus, val_idx)
    carved_train = _index_corpus(train_corpus, train_idx)
    return carved_train, val_corpus


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
