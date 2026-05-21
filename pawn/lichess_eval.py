"""Lichess Elo-stratified move-accuracy evaluator (chunk 4.4) — JAX port
of the legacy `pawn.eval_suite.lichess` + the lichess-data path it
consumed.

Loads a PGN file via the Rust engine's ``parse_pgn_lichess`` (which
also returns per-game ``white_elo`` / ``black_elo`` / ``result``
metadata), filters by Elo range, and computes
``pawn.eval.evaluate_accuracy`` on the filtered games.

The legacy module dragged in ``polars`` + ``torch.utils.data`` for the
on-disk cache + DataLoader wrappers. This port stays at the raw
numpy + Rust-engine level — the cache is a single ``games.npz`` file
+ ``metadata.json`` sibling, written under the cache root (see
``_cache_root``: defaults to ``$HF_HOME/pawn-lichess-cache/``; honour
``PAWN_DATA_CACHE`` as a direct override). Re-running the same
``(pgn_path, max_ply, min_ply, prepend_outcome, max_games)`` combination
re-uses the cache.

Per-Elo bucketing matches the legacy convention: an "Elo slice" of
the corpus is the set of games whose **side-to-move ELO** (the
player making the *predicted* move) lands in ``[lo, hi)``. For a
typical ``maia-1500`` evaluation, both 1450 ≤ Elo < 1550 buckets
contribute.

Public surface:

* ``load_lichess_corpus(...)`` — parse + cache. Returns ``LichessCorpus``.
* ``filter_elo_slice(...)``     — game-level filter by Elo range.
* ``evaluate_elo_accuracy(...)`` — runs ``pawn.eval.evaluate_accuracy``
  on a filtered slice. Returns a ``LichessElo`` result with per-Elo
  accuracy buckets + overall.

Out of scope (legacy features deferred to a future PR):

* Bucketed-length collation + GPU DataLoader (legacy
  ``LegalMaskBuilder`` + ``BucketedLegalMaskCollate``) — the JAX eval
  is single-batch with a static ``seq_len`` and doesn't need them.
* Per-ply accuracy curves (the legacy ``--per-ply`` flag). The
  underlying ``pawn.eval.AccuracyResult`` already carries per-phase
  counts (opening / middle / end); per-ply is a small extension.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess_engine as engine
import jax
import numpy as np

from pawn.config import N_OUTCOMES, PAD_TOKEN
from pawn.corpus import Corpus
from pawn.eval import AccuracyResult, evaluate_accuracy
from pawn.model import PAWNModel


@dataclass
class LichessCorpus:
    """One parsed + cached Lichess PGN slice.

    All arrays are NumPy on host. The trainer / eval slices into them
    per-batch.
    """

    tokens: np.ndarray            # (N, seq_len) int32 — left-aligned, PAD-padded
    attn_mask: np.ndarray         # (N, seq_len) bool
    targets: np.ndarray           # (N, seq_len) int32
    loss_mask: np.ndarray         # (N, seq_len) bool — supervised positions
    game_lengths: np.ndarray      # (N,) int32
    white_elo: np.ndarray         # (N,) int32 — 0 if missing
    black_elo: np.ndarray         # (N,) int32 — 0 if missing
    result: np.ndarray            # (N,) int8 — +1 white, -1 black, 0 draw
    seq_len: int = field(default=512)
    prepend_outcome: bool = field(default=False)


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


def _cache_root() -> Path:
    """Honour ``PAWN_DATA_CACHE`` and ``HF_HOME`` for the cache root —
    same precedence the legacy lichess cache observed."""
    base = (
        os.environ.get("PAWN_DATA_CACHE")
        or os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "pawn-lichess-cache",
        )
    )
    return Path(base)


def _cache_key(
    pgn_path: Path,
    max_ply: int,
    min_ply: int,
    prepend_outcome: bool,
    max_games: int,
) -> str:
    """Stable per-(pgn, params) cache key. Hashed so the path doesn't
    leak source basenames + isn't filesystem-sensitive.

    ``max_games`` is part of the key because the Rust parser truncates
    at that count — a smaller call would otherwise silently hit a
    larger-call cache (or vice versa).
    """
    payload = (
        f"{pgn_path.resolve()}|max_ply={max_ply}|min_ply={min_ply}"
        f"|prepend_outcome={prepend_outcome}|max_games={max_games}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _cache_dir(
    pgn_path: Path,
    max_ply: int,
    min_ply: int,
    prepend_outcome: bool,
    max_games: int,
) -> Path:
    key = _cache_key(pgn_path, max_ply, min_ply, prepend_outcome, max_games)
    return _cache_root() / key


# ---------------------------------------------------------------------------
# PGN parsing + cache materialisation
# ---------------------------------------------------------------------------


def load_lichess_corpus(
    pgn_path: str | Path,
    *,
    max_ply: int = 512,
    min_ply: int = 10,
    prepend_outcome: bool = False,
    max_games: int = 1_000_000,
    use_cache: bool = True,
) -> LichessCorpus:
    """Parse a Lichess PGN into a ``LichessCorpus``.

    On first call with a given ``(pgn_path, max_ply, min_ply,
    prepend_outcome, max_games)`` tuple, this parses + caches the
    corpus under the cache root (``_cache_root()``; see the module
    docstring for the env-var precedence). Subsequent calls mmap the
    cache.

    Args:
        max_ply: CLM sequence length. Capped at the model's max_seq_len
            by the trainer.
        min_ply: drop games shorter than this many plies.
        prepend_outcome: pack the outcome token at sequence position 0.
        max_games: hard cap on parsed games (the Rust parser also
            short-circuits at this count).
        use_cache: disable to force a fresh parse.
    """
    pgn_path = Path(pgn_path)
    if not pgn_path.exists():
        raise FileNotFoundError(pgn_path)

    cache_dir = _cache_dir(
        pgn_path, max_ply, min_ply, prepend_outcome, max_games,
    )
    npz_path = cache_dir / "games.npz"
    meta_path = cache_dir / "metadata.json"

    if use_cache and npz_path.exists() and meta_path.exists():
        with np.load(npz_path) as cache:
            return LichessCorpus(
                tokens=cache["tokens"].astype(np.int32),
                attn_mask=cache["attn_mask"].astype(np.bool_),
                targets=cache["targets"].astype(np.int32),
                loss_mask=cache["loss_mask"].astype(np.bool_),
                game_lengths=cache["game_lengths"].astype(np.int32),
                white_elo=cache["white_elo"].astype(np.int32),
                black_elo=cache["black_elo"].astype(np.int32),
                result=cache["result"].astype(np.int8),
                seq_len=int(max_ply),
                prepend_outcome=bool(prepend_outcome),
            )

    content = pgn_path.read_text(encoding="utf-8", errors="replace")
    parsed = engine.parse_pgn_lichess(
        content, max_ply, max_games, min_ply, prepend_outcome,
    )
    corpus = _materialise(parsed, max_ply, prepend_outcome)

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Atomic write: stage both files under per-process tmp
        # siblings, then atomically rename via os.replace. Order is
        # npz first, metadata last: the load-side check requires
        # *both* files to exist, so a kill between the renames leaves
        # the cache as a miss (which re-parses) — never a
        # corrupt-but-readable .npz next to stale metadata. The
        # tmp names embed pid + uuid so two concurrent writers (eg a
        # multi-pod eval) can't interleave their .savez writes into
        # one corrupt-but-syntactically-valid archive and then both
        # promote it via the same os.replace path. np.savez_compressed
        # appends ``.npz`` to filenames that don't already end in
        # ``.npz``, so the tmp name keeps the extension to avoid
        # landing as ``games.npz.tmp.npz``.
        suffix = f"{os.getpid()}.{uuid.uuid4().hex}"
        npz_tmp = cache_dir / f"games.tmp.{suffix}.npz"
        meta_tmp = cache_dir / f"metadata.json.tmp.{suffix}"
        # try/finally guarantees the per-process tmp files are cleaned
        # up even on exception (out-of-disk during savez, json.dump
        # failure, KeyboardInterrupt). Without this, every interrupted
        # write leaks a unique-named tmp file forever — the next run
        # can't find or reuse it because pid + uuid never repeats.
        # os.replace removes the source path on success, so unlink
        # below is a no-op on the happy path.
        try:
            np.savez_compressed(
                npz_tmp,
                tokens=corpus.tokens.astype(np.int32),
                attn_mask=corpus.attn_mask,
                targets=corpus.targets.astype(np.int32),
                loss_mask=corpus.loss_mask,
                game_lengths=corpus.game_lengths.astype(np.int32),
                white_elo=corpus.white_elo.astype(np.int32),
                black_elo=corpus.black_elo.astype(np.int32),
                result=corpus.result.astype(np.int8),
            )
            with open(meta_tmp, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "pgn_path": str(pgn_path.resolve()),
                        "max_ply": int(max_ply),
                        "min_ply": int(min_ply),
                        "prepend_outcome": bool(prepend_outcome),
                        "max_games": int(max_games),
                        "n_games": int(corpus.tokens.shape[0]),
                    },
                    fh, indent=2,
                )
            os.replace(npz_tmp, npz_path)
            os.replace(meta_tmp, meta_path)
        finally:
            npz_tmp.unlink(missing_ok=True)
            meta_tmp.unlink(missing_ok=True)
    return corpus


def _materialise(
    parsed: dict[str, Any], seq_len: int, prepend_outcome: bool,
) -> LichessCorpus:
    """Convert the Rust parser's output dict into a ``LichessCorpus``.

    The Rust parser returns:
      * ``tokens``: (N, seq_len) int16 — already PAD-padded
      * ``game_lengths``: (N,) int32 — number of move plies
      * ``orig_lengths``: untruncated lengths (unused here)
      * per-game metadata lists: ``white_elo`` / ``black_elo`` /
        ``result`` / ``site`` / ``datetime`` / etc.
    """
    tokens = np.asarray(parsed["tokens"], dtype=np.int32)
    n_games, sl = tokens.shape
    if sl != seq_len:
        raise ValueError(
            f"engine returned tokens of width {sl}, expected {seq_len}"
        )
    game_lengths = np.asarray(parsed["game_lengths"], dtype=np.int32)

    # attn_mask: True at every real (non-PAD) position. Derived from
    # tokens to be robust against either prepend_outcome mode.
    attn_mask = tokens != PAD_TOKEN

    # targets[t] = tokens[t + 1], with PAD at the terminal slot.
    targets = np.full_like(tokens, PAD_TOKEN, dtype=np.int32)
    targets[:, :-1] = tokens[:, 1:]

    # loss_mask spans every supervised next-token transition.
    pos = np.arange(seq_len, dtype=np.int32)[None, :]
    if prepend_outcome:
        # Layout: [outcome, m1, m2, ..., mN, PAD, PAD, ...]. Supervised
        # positions are 0..N inclusive: position 0 predicts m1 *from*
        # the outcome token (outcome conditioning's whole point —
        # tokens[0] is the predict-from input, targets[0] = m1), and
        # position N predicts the terminal PAD from mN. An earlier
        # cut at pos>=1 was wrong: it dropped the outcome→m1 step
        # silently, biasing first-move accuracy / loss for every
        # outcome-prefixed eval (Codex round 4 P2).
        loss_mask = pos <= game_lengths[:, None]
    else:
        # Layout: [m1, m2, ..., mN, PAD, ...]. Supervised positions
        # are 0..N-1 inclusive — N rows total — predicting m2..mN
        # then the terminal PAD.
        loss_mask = pos < game_lengths[:, None]

    # Coerce metadata into per-game numpy arrays.
    white_elo = np.asarray(parsed.get("white_elo", [0] * n_games), dtype=np.int32)
    black_elo = np.asarray(parsed.get("black_elo", [0] * n_games), dtype=np.int32)
    result_strs = parsed.get("result", ["*"] * n_games)
    result = np.zeros(n_games, dtype=np.int8)
    for i, r in enumerate(result_strs):
        if r == "1-0":
            result[i] = 1
        elif r == "0-1":
            result[i] = -1
        # Anything else (draws, "*", malformed) stays 0.

    return LichessCorpus(
        tokens=tokens,
        attn_mask=attn_mask,
        targets=targets,
        loss_mask=loss_mask,
        game_lengths=game_lengths,
        white_elo=white_elo,
        black_elo=black_elo,
        result=result,
        seq_len=seq_len,
        prepend_outcome=prepend_outcome,
    )


# ---------------------------------------------------------------------------
# Elo filtering
# ---------------------------------------------------------------------------


def filter_elo_slice(
    corpus: LichessCorpus,
    *,
    elo_min: int,
    elo_max: int,
    side: str = "both",
) -> LichessCorpus:
    """Filter games whose Elo lands in ``[elo_min, elo_max)``.

    Args:
        side: ``"white"`` / ``"black"`` / ``"both"``. With ``"both"``
            (default), games are kept iff *either* side's Elo falls in
            the range — matches the legacy Maia-eval default. With
            ``"white"`` / ``"black"``, only that side's Elo is checked.
    """
    if side not in ("white", "black", "both"):
        raise ValueError(f"side={side!r} must be 'white', 'black', or 'both'")
    if elo_min >= elo_max:
        raise ValueError(
            f"elo_min={elo_min} must be < elo_max={elo_max}"
        )

    w = (corpus.white_elo >= elo_min) & (corpus.white_elo < elo_max)
    b = (corpus.black_elo >= elo_min) & (corpus.black_elo < elo_max)
    if side == "white":
        keep = w
    elif side == "black":
        keep = b
    else:
        keep = w | b

    return LichessCorpus(
        tokens=corpus.tokens[keep],
        attn_mask=corpus.attn_mask[keep],
        targets=corpus.targets[keep],
        loss_mask=corpus.loss_mask[keep],
        game_lengths=corpus.game_lengths[keep],
        white_elo=corpus.white_elo[keep],
        black_elo=corpus.black_elo[keep],
        result=corpus.result[keep],
        seq_len=corpus.seq_len,
        prepend_outcome=corpus.prepend_outcome,
    )


# ---------------------------------------------------------------------------
# Move-accuracy eval
# ---------------------------------------------------------------------------


@dataclass
class LichessElo:
    """Per-bucket Lichess move-accuracy result."""

    elo_min: int
    elo_max: int
    side: str
    n_games: int
    accuracy: AccuracyResult


def _to_eval_corpus(sliced: LichessCorpus) -> Corpus:
    """Wrap a ``LichessCorpus`` slice in the ``pawn.corpus.Corpus``
    shape ``evaluate_accuracy`` expects. The outcome offsets are
    derived from the per-game ``result`` (engine doesn't expose
    finer-grained terminations for human PGN games):
        +1 result → WHITE_CHECKMATES bucket (offset 0)
        -1 result → BLACK_CHECKMATES bucket (offset 1)
         0 result → DRAW_BY_RULE bucket (offset 3)
    These offsets only matter when ``prepend_outcome=True`` for the
    training pipeline; the accuracy eval reads them only for the
    ``Corpus`` dataclass contract."""
    n = int(sliced.tokens.shape[0])
    outcome_offset = np.empty((n,), dtype=np.uint8)
    outcome_offset.fill(3)  # default DRAW_BY_RULE
    outcome_offset[sliced.result > 0] = 0  # WHITE_CHECKMATES
    outcome_offset[sliced.result < 0] = 1  # BLACK_CHECKMATES
    # Guard the contract: every value must land inside
    # ``[0, N_OUTCOMES)`` — if N_OUTCOMES drops below 4 the offsets
    # become invalid. Cheap.
    assert (outcome_offset < N_OUTCOMES).all()
    return Corpus(
        tokens=sliced.tokens,
        attn_mask=sliced.attn_mask,
        targets=sliced.targets,
        loss_mask=sliced.loss_mask,
        outcome_offset=outcome_offset,
    )


def evaluate_elo_accuracy(
    model: PAWNModel,
    corpus: LichessCorpus,
    *,
    elo_min: int,
    elo_max: int,
    side: str = "both",
    batch_size: int = 64,
) -> LichessElo:
    """Run ``pawn.eval.evaluate_accuracy`` on the Elo-filtered slice."""
    sliced = filter_elo_slice(corpus, elo_min=elo_min, elo_max=elo_max, side=side)
    n = int(sliced.tokens.shape[0])
    if n == 0:
        return LichessElo(
            elo_min=elo_min, elo_max=elo_max, side=side,
            n_games=0,
            accuracy=AccuracyResult(
                overall=float("nan"), n_supervised=0,
                phase={"opening": (0, 0), "midgame": (0, 0), "endgame": (0, 0)},
            ),
        )
    acc = evaluate_accuracy(model, _to_eval_corpus(sliced), batch_size=batch_size)
    return LichessElo(
        elo_min=elo_min, elo_max=elo_max, side=side,
        n_games=n, accuracy=acc,
    )
