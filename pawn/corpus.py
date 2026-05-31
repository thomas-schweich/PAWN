"""Random-game corpus + the shared CLM packing / prefix helpers.

A :class:`Corpus` is the v2 data unit consumed by every JAX-side
trainer (pretrain, adapter, specialized_clm). It is a tight bundle of
JAX-friendly arrays — fixed-width ``tokens`` / ``targets`` /
``loss_mask`` / ``attn_mask`` plus a per-game ``outcome_offset``
scalar — built once before training and reused across the
double-buffered streaming path.

Two entry points produce a :class:`Corpus`:

- :func:`generate_corpus` — calls into the Rust engine's
  ``generate_random_games`` to produce a fresh batch of random
  self-play games, then packs them with :func:`_pack_clm`.
- :func:`pack_corpus` — packs **pre-tokenised** game data (move IDs +
  game lengths + outcome tokens) into the same shape. The Lichess
  data path in :mod:`pawn.lichess_data` (S5.C2) uses this.

**Conditioning prefix (Phase-A Chunk 4).** Every sequence is assembled
as ``[BOS][cond…][ply…][PAD…]``: slot 0 is always :data:`BOS_TOKEN`,
slots ``1..C-1`` carry one control token per entry in the run's
``conditioning`` list (resolved per-kind, :data:`NULL_TOKEN` where a
game lacks the value), and the game's moves start at slot ``C`` where
``C = 1 + len(conditioning)``. The two shared helpers
:func:`build_prefix` and :func:`build_loss_mask` are the *single* owners
of this layout — both :func:`pack_corpus` here and
:mod:`pawn.lichess_data` route through them so there is exactly one
off-by-one to reason about.

The loss is supervised on positions ``[C-1 .. C-1 + game_length - 1]``:
the last prefix slot (``C-1``) predicts ``ply_1`` (the first move IS
supervised), and the predict-PAD slot (``C-1 + game_length``) is NOT.
This is a deliberate change from the pre-Chunk-4 contract, which
supervised the predict-PAD slot but never the first move — it shifts
which positions count toward per-move accuracy vs v1.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import chess_engine as engine
from pawn.config import (
    BLACK_CHECKMATES,
    BOS_TOKEN,
    CONDITIONING_KINDS,
    DRAW_BY_RULE,
    NULL_TOKEN,
    NUM_ACTIONS,
    PAD_TOKEN,
    PLY_LIMIT,
    STALEMATE,
    VOCAB_SIZE,
    WHITE_CHECKMATES,
)

__all__ = [
    "Corpus",
    "generate_corpus",
    "pack_corpus",
    "build_prefix",
    "build_loss_mask",
    "conditioning_to_C",
    "assert_conditioning_C",
    "conditioning_from_run_block",
    "legal_mask_for_games",
]


def conditioning_from_run_block(run_block: Mapping[str, object] | None) -> list[str]:
    """Extract a checkpoint's ``conditioning`` list from its persisted run block.

    ``run_block`` is the ``config.json`` ``run`` dict returned by
    :func:`pawn.checkpoint.load_model` — ``BaseRunConfig.model_dump()``,
    which carries the run's ``conditioning`` field. Eval / generation read
    it so they rebuild the exact sequence layout the checkpoint was trained
    under (plan §8.1) instead of assuming a hardcoded default.

    Returns ``[]`` (the BOS-only ``C=1`` layout) when the checkpoint has no
    run block (it predates the conditioning prefix) or the block omits the
    key. Every entry is validated against
    :data:`pawn.config.CONDITIONING_KINDS` so a corrupted block can't
    silently produce a bad prefix.
    """
    if not run_block:
        return []
    raw = run_block.get("conditioning")
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"run block `conditioning` must be a list of kinds, got "
            f"{type(raw).__name__}: {raw!r}"
        )
    conditioning = [str(k) for k in raw]
    # Validate via the shared helper (raises on an unknown kind).
    conditioning_to_C(conditioning)
    return conditioning


def conditioning_to_C(conditioning: Sequence[str]) -> int:
    """Prefix width ``C = 1 + len(conditioning)`` (BOS always present).

    Every entry must be a known kind in
    :data:`pawn.config.CONDITIONING_KINDS`; an unknown kind is a hard
    error so a typo'd ``--conditioning`` value can't silently produce a
    NULL-only prefix slot.
    """
    for kind in conditioning:
        if kind not in CONDITIONING_KINDS:
            raise ValueError(
                f"unknown conditioning kind {kind!r}; valid kinds are "
                f"{sorted(CONDITIONING_KINDS)}"
            )
    return 1 + len(conditioning)


def assert_conditioning_C(builder_conditioning: Sequence[str], checkpoint_C: int) -> None:
    """Assert a builder's conditioning-derived ``C`` matches a checkpoint's.

    Move positions sit at a fixed, run-constant offset ``C``; absolute
    RoPE phases therefore drift silently if a checkpoint trained with
    one ``C`` is evaluated against a corpus assembled with another. Call
    this at the eval/build boundary (with the checkpoint's persisted
    ``C``) to turn that silent drift into a loud failure.
    """
    builder_C = conditioning_to_C(builder_conditioning)
    if builder_C != checkpoint_C:
        raise ValueError(
            f"conditioning mismatch: builder conditioning "
            f"{list(builder_conditioning)!r} implies C={builder_C}, but the "
            f"checkpoint was trained with C={checkpoint_C}. Move positions "
            f"would land at a different absolute offset (RoPE drift). Rebuild "
            f"the corpus with the checkpoint's own conditioning."
        )


@dataclass(frozen=True, slots=True)
class Corpus:
    """A packed game corpus ready for batched JAX consumption.

    Every field has a leading game axis ``N`` and a trailing sequence
    axis ``T`` (= ``seq_len``). Storage dtype is int32 / bool — small
    enough to keep multi-million-game corpora resident-or-streamable
    on commodity hardware.

    **Arrays are stored as host numpy** (not JAX device arrays). For
    realistic Lichess slices the full corpus can be multi-GB; eagerly
    placing it on the default JAX device would exhaust accelerator
    memory before the double-buffered prefetch path can batch it. The
    trainer's prefetch loop slices per-batch and calls ``jnp.asarray``
    at the device-transfer boundary.

    Fields:
        tokens: ``(N, T)`` int32 — input token IDs. PAD where the game
            is shorter than seq_len.
        targets: ``(N, T)`` int32 — tokens shifted left by 1, with PAD
            in the trailing slot. Loss positions consume this.
        attn_mask: ``(N, T)`` bool — True where ``tokens`` is real
            (not PAD). Attention applies a causal × pad mask combo at
            forward time.
        loss_mask: ``(N, T)`` bool — True at positions where the loss
            is supervised: exactly ``[C-1 .. C-1 + game_length - 1]``.
            The last prefix slot (``C-1``) predicts ``ply_1`` (first
            move supervised); the predict-PAD slot is excluded.
        outcome_offset: ``(N,)`` int32 — the prefix width ``C`` (the
            slot where this game's first move lives), constant across
            all games in a corpus. ``C = 1`` with no conditioning (BOS
            only); ``C = 1 + len(conditioning)`` otherwise. Named
            ``outcome_offset`` for cache back-compat; it is the
            move-start offset, not a 0/1 outcome flag.
        game_lengths: ``(N,)`` int32 — number of real moves (does NOT
            include the BOS/conditioning prefix slots). Used by A.1
            bucketing to pick the right seq-length bucket per game.
    """

    tokens: NDArray[np.int32]
    targets: NDArray[np.int32]
    attn_mask: NDArray[np.bool_]
    loss_mask: NDArray[np.bool_]
    outcome_offset: NDArray[np.int32]
    game_lengths: NDArray[np.int32]

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def n_games(self) -> int:
        return len(self)

    @property
    def seq_len(self) -> int:
        return int(self.tokens.shape[1])

    def by_bucket(self, edges: tuple[int, ...]) -> dict[int, "Corpus"]:
        """Partition this Corpus into per-bucket sub-corpora.

        Each game's bucket = smallest edge ``e`` such that
        ``effective_length(game) <= e``, where ``effective_length`` is
        ``game_lengths + outcome_offset`` (the actual width of real
        tokens including any prefix). Each returned sub-corpus has its
        own truncated seq_len matching the bucket edge.

        ``edges`` must be sorted ascending. The top edge must be >=
        ``self.seq_len`` (every game must fit in some bucket).

        Returned dict maps ``edge -> Corpus`` (sliced + width-truncated).
        Buckets with zero games are omitted.
        """
        if not edges:
            return {self.seq_len: self}
        sorted_edges = tuple(sorted(set(edges)))
        if sorted_edges[-1] > self.seq_len:
            raise ValueError(
                f"top bucket edge {sorted_edges[-1]} > corpus seq_len "
                f"{self.seq_len}; nothing to truncate to"
            )
        if sorted_edges[-1] < self.seq_len:
            raise ValueError(
                f"top bucket edge {sorted_edges[-1]} < corpus seq_len "
                f"{self.seq_len}; some games may overflow"
            )

        eff = (self.game_lengths + self.outcome_offset).astype(np.int32)
        edges_arr = np.asarray(sorted_edges, dtype=np.int32)
        # bucket_idx[i] = smallest j s.t. eff[i] <= edges[j]
        bucket_idx = np.searchsorted(edges_arr, eff, side="left")
        bucket_idx = np.clip(bucket_idx, 0, len(sorted_edges) - 1)

        out: dict[int, Corpus] = {}
        for j, edge in enumerate(sorted_edges):
            mask = bucket_idx == j
            if not mask.any():
                continue
            sel = np.flatnonzero(mask)
            out[int(edge)] = Corpus(
                tokens=self.tokens[sel, :edge].copy(),
                targets=self.targets[sel, :edge].copy(),
                attn_mask=self.attn_mask[sel, :edge].copy(),
                loss_mask=self.loss_mask[sel, :edge].copy(),
                outcome_offset=self.outcome_offset[sel].copy(),
                game_lengths=self.game_lengths[sel].copy(),
            )
        return out

    def move_ids(self) -> NDArray[np.int16]:
        """Recover the per-game move-id matrix the engine replays from.

        Returns ``(N, max_ply)`` int16 where ``max_ply = seq_len -
        prefix_width`` (the prefix is BOS + conditioning, width
        ``C = outcome_offset``). Column ``t`` is the engine action token of
        the game's ``t``-th ply, taken straight off the packed sequence at
        slot ``C + t``; padding slots are :data:`PAD_TOKEN`. The
        prefix width is constant across the corpus
        (:attr:`outcome_offset`), so a single slice recovers every game's
        moves. Feeds :func:`legal_mask_for_games`.
        """
        if self.n_games == 0:
            return np.zeros((0, 0), dtype=np.int16)
        c = int(self.outcome_offset[0])
        return self.tokens[:, c:].astype(np.int16)


def legal_mask_for_games(
    corpus: Corpus, indices: NDArray[np.integer],
) -> NDArray[np.bool_]:
    """Dense ``(B, T, V)`` per-position legal-move mask for ``indices``.

    The v2 source for the adapter loss's legality term (parity with v1's
    :class:`LegalMaskBuilder` + sparse scatter, ``git show
    main:pawn/adapter_training.py``). For each selected game the Rust
    engine replays the moves and emits, per ply, the set of legal action
    tokens; we scatter those into a dense boolean grid aligned to the
    *target* positions of the packed sequence.

    Alignment: the engine indexes legal sets by game-relative ply ``t``
    (``0 = first move``). In the packed sequence the slot whose target is
    ply-``t`` move sits at ``(C-1) + t`` (``C = outcome_offset``): the last
    prefix slot ``C-1`` predicts ``ply_1``, matching
    :func:`build_loss_mask`. We therefore shift the engine's ply axis by
    ``C-1`` when scattering. The engine also marks PAD legal at the
    end-of-game slot (its ply ``length``), which lands at ``(C-1) +
    length`` — the predict-PAD position, which the loss mask excludes, so
    it is harmless.

    Returns a freshly-allocated host array; the caller uploads it via
    :func:`jax.numpy.asarray` at the batch boundary.
    """
    idx = np.asarray(indices).reshape(-1)
    b = idx.shape[0]
    t = corpus.seq_len
    v = VOCAB_SIZE
    if b == 0:
        return np.zeros((0, t, v), dtype=np.bool_)
    c = int(corpus.outcome_offset[0])
    # Select first, slice second: indexing `corpus.tokens[idx]` materialises
    # only the B selected games, then the `[:, c:]` slice + cast builds the
    # move-region copy for those B rows. Calling `corpus.move_ids()` here would
    # instead cast the move region of ALL N games before selecting `idx` — a
    # corpus-wide host allocation churned every chunk. `c` is the prefix width,
    # so the alignment math below is unchanged.
    move_ids = corpus.tokens[idx][:, c:].astype(np.int16)  # (B, max_ply) int16
    max_ply = move_ids.shape[1]
    # Clamp game_lengths to the move-id width: a game whose ply count would
    # overflow the packed sequence (prefix C + plies > seq_len) is stored
    # truncated, so only the moves that physically fit can be replayed. The
    # engine replays exactly `game_lengths` plies, so the clamp keeps it in
    # bounds and aligned with the tokens actually present.
    game_lengths = np.minimum(
        corpus.game_lengths[idx], max_ply
    ).astype(np.int16)
    # Engine returns flat indices into a (B, max_ply, V) grid: a legal
    # token `tok` at game `g`, ply `p` is encoded `g*max_ply*V + p*V + tok`.
    # We pass `seq_len=max_ply` so the engine's per-game stride matches the
    # `move_ids` width; the predict-PAD entry it appends at ply `length`
    # only fires when `length < max_ply` (always true for sub-max games).
    flat = engine.compute_legal_token_masks_sparse(
        np.ascontiguousarray(move_ids),
        np.ascontiguousarray(game_lengths),
        max_ply,
        v,
    )
    flat = np.asarray(flat, dtype=np.int64)
    out = np.zeros((b, t, v), dtype=np.bool_)
    if flat.size == 0:
        return out
    g_idx = flat // (max_ply * v)
    rem = flat % (max_ply * v)
    p_idx = rem // v
    tok_idx = rem % v
    # Shift the engine ply axis by (C-1) to land on the target slot, and
    # drop entries that fall outside the packed sequence width.
    seq_pos = p_idx + (c - 1)
    keep = seq_pos < t
    out[g_idx[keep], seq_pos[keep], tok_idx[keep]] = True
    return out


def _map_termination_to_outcome(
    term_codes: np.ndarray, game_lengths: np.ndarray
) -> np.ndarray:
    """Map engine termination codes to outcome token IDs.

    Engine termination codes:
        0 = Checkmate
        1 = Stalemate
        2 = SeventyFiveMoveRule
        3 = FivefoldRepetition
        4 = InsufficientMaterial
        5 = PlyLimit

    For checkmate, the winning side is inferred from game length —
    odd ``game_length`` (white played the last move) means white
    delivered the mate; even means black.
    """
    term = np.asarray(term_codes, dtype=np.int32)
    gl = np.asarray(game_lengths, dtype=np.int32)

    outcomes = np.full(len(term), PLY_LIMIT, dtype=np.int32)
    is_checkmate = term == 0
    outcomes[is_checkmate & (gl % 2 == 1)] = WHITE_CHECKMATES
    outcomes[is_checkmate & (gl % 2 == 0)] = BLACK_CHECKMATES
    outcomes[term == 1] = STALEMATE
    outcomes[(term == 2) | (term == 3) | (term == 4)] = DRAW_BY_RULE
    # PlyLimit (code 5) keeps the default PLY_LIMIT.
    return outcomes


# ---------------------------------------------------------------------------
# Shared prefix + loss-mask assembler (the single owner of the layout)
# ---------------------------------------------------------------------------


def _resolve_outcome_slot(
    outcome_tokens: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Resolve the ``"outcome"`` conditioning kind to a per-game token.

    Every game has an outcome token, so this is the identity in practice;
    a sentinel ``< 0`` (a game whose outcome is genuinely unknown) maps
    to :data:`NULL_TOKEN` so the prefix slot is filled rather than left
    holding an out-of-vocab value.
    """
    return np.where(outcome_tokens >= 0, outcome_tokens, NULL_TOKEN).astype(
        np.int32
    )


# Registry: conditioning kind → resolver(outcome_tokens) -> (N,) int32 of the
# per-game control token for that kind. The kind *names* are pinned in
# :data:`pawn.config.CONDITIONING_KINDS`; this dict supplies the resolution
# logic. Extending the registry (e.g. an Elo-bucket kind) means adding a name
# there and a resolver here — both must stay in lockstep.
_CONDITIONING_RESOLVERS: Mapping[
    str, Callable[[NDArray[np.int32]], NDArray[np.int32]]
] = {
    "outcome": _resolve_outcome_slot,
}


def build_prefix(
    conditioning: Sequence[str],
    outcome_tokens: NDArray[np.int32],
    n: int,
) -> NDArray[np.int32]:
    """Assemble the ``(n, C)`` control prefix for every game.

    Slot 0 is always :data:`BOS_TOKEN`; slots ``1..C-1`` hold the
    resolved control token for ``conditioning[0..]`` in order, or
    :data:`NULL_TOKEN` where a game lacks that kind's value. The prefix
    width is ``C = 1 + len(conditioning)``.

    The single owner of the prefix layout — both :func:`pack_corpus` and
    :mod:`pawn.lichess_data` route through it.
    """
    C = conditioning_to_C(conditioning)
    outcome_tokens = np.asarray(outcome_tokens, dtype=np.int32)
    if outcome_tokens.shape != (n,):
        raise ValueError(
            f"outcome_tokens must be ({n},) matching n={n}, got shape "
            f"{outcome_tokens.shape}"
        )
    prefix = np.empty((n, C), dtype=np.int32)
    prefix[:, 0] = BOS_TOKEN
    for slot, kind in enumerate(conditioning, start=1):
        resolver = _CONDITIONING_RESOLVERS.get(kind)
        if resolver is None:
            # Registered in CONDITIONING_KINDS (so conditioning_to_C accepted
            # it) but no resolver wired here — a developer-side bug, not user
            # input. Fail loud rather than silently NULL-filling.
            raise ValueError(
                f"conditioning kind {kind!r} is registered but has no "
                f"resolver in pawn.corpus._CONDITIONING_RESOLVERS"
            )
        prefix[:, slot] = resolver(outcome_tokens)
    return prefix


def build_loss_mask(
    C: int,
    game_lengths: NDArray[np.int32],
    seq_len: int,
) -> NDArray[np.bool_]:
    """``(n, seq_len)`` bool mask, True exactly on the supervised slots.

    Supervised positions are ``[C-1 .. C-1 + game_length - 1]``: the last
    prefix slot (``C-1``) predicts ``ply_1`` (the first move IS
    supervised) and the predict-PAD slot (``C-1 + game_length``) is NOT.
    ``game_length`` is clamped to the moves that actually fit in
    ``seq_len`` (``seq_len - C`` move slots), so a truncated game's mask
    never overruns. A zero-length game supervises no positions.

    This is a deliberate change from the pre-Chunk-4 mask, which
    supervised the predict-PAD slot and never the first move; it shifts
    which positions count toward per-move accuracy vs v1.
    """
    game_lengths = np.asarray(game_lengths, dtype=np.int32)
    n_move_slots = max(seq_len - C, 0)
    capped = np.minimum(game_lengths, n_move_slots)
    seq_positions = np.arange(seq_len, dtype=np.int32)[None, :]
    lo = C - 1
    # hi = last supervised slot = C-1 + capped - 1. For a zero-length game
    # capped==0 so hi == lo-1 < lo and the mask is all-False for that row.
    hi = (lo + capped - 1)[:, None]
    return (seq_positions >= lo) & (seq_positions <= hi)


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


def _pack_clm(
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    outcome_tokens: np.ndarray,
    *,
    seq_len: int,
    conditioning: Sequence[str] = (),
) -> Corpus:
    """Shared packing helper — turns engine / parquet output into a Corpus.

    ``move_ids`` is ``(N, max_ply)`` int16 with PAD past ``game_length``;
    ``game_lengths`` is ``(N,)`` int (number of moves, never including
    the prefix slots); ``outcome_tokens`` is ``(N,)`` int (the outcome
    token for the game, used to resolve the ``"outcome"`` conditioning
    kind when present).

    Output sequence width is always ``seq_len``. The sequence is laid
    out as ``[BOS][cond…][ply…][PAD…]`` via the shared
    :func:`build_prefix` / :func:`build_loss_mask` helpers, so moves
    start at slot ``C = 1 + len(conditioning)`` and the first move is
    supervised by the last prefix slot.
    """
    move_ids = np.asarray(move_ids, dtype=np.int32)
    game_lengths = np.asarray(game_lengths, dtype=np.int32)
    outcome_tokens = np.asarray(outcome_tokens, dtype=np.int32)

    if move_ids.ndim != 2:
        raise ValueError(f"move_ids must be 2-D (N, max_ply), got shape {move_ids.shape}")
    if game_lengths.ndim != 1 or game_lengths.shape[0] != move_ids.shape[0]:
        raise ValueError(
            f"game_lengths must be (N,) matching move_ids' N={move_ids.shape[0]}, "
            f"got shape {game_lengths.shape}"
        )
    if outcome_tokens.shape != (move_ids.shape[0],):
        raise ValueError(
            f"outcome_tokens must be (N,) matching move_ids' N={move_ids.shape[0]}, "
            f"got shape {outcome_tokens.shape}"
        )
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    n, max_ply = move_ids.shape
    C = conditioning_to_C(conditioning)
    if seq_len <= C:
        raise ValueError(
            f"seq_len ({seq_len}) must exceed the conditioning prefix width "
            f"C={C} (= 1 BOS + {C - 1} conditioning slot(s)) so at least one "
            f"move slot remains"
        )
    n_move_slots = seq_len - C

    # 1. Initial tokens buffer — all PAD, then lay the BOS+conditioning
    # prefix into slots 0..C-1 via the shared assembler.
    tokens = np.full((n, seq_len), PAD_TOKEN, dtype=np.int32)
    tokens[:, :C] = build_prefix(conditioning, outcome_tokens, n)

    # 2. Clean move IDs (PAD past game_length so trailing junk from the
    # engine doesn't leak into the sequence).
    positions = np.arange(max_ply, dtype=np.int32)[None, :]  # (1, max_ply)
    valid_move = positions < game_lengths[:, None]
    clean_moves = np.where(valid_move, move_ids, PAD_TOKEN)

    # 3. Copy the moves into slots C..C+n_to_copy-1.
    n_to_copy = min(max_ply, n_move_slots)
    tokens[:, C : C + n_to_copy] = clean_moves[:, :n_to_copy]

    # 4. attn_mask: True where tokens != PAD. The BOS/conditioning prefix
    # is real (no interior PAD — preserves the right-pad invariant).
    attn_mask = tokens != PAD_TOKEN

    # 5. targets: tokens shifted left by 1, with PAD on the trailing slot.
    # The output vocab includes the PAD column, so PAD-as-target is benign;
    # the loss_mask is what restricts supervision to real moves.
    targets = np.full_like(tokens, PAD_TOKEN)
    targets[:, :-1] = tokens[:, 1:]

    # 6. loss_mask: True on [C-1 .. C-1 + game_length - 1] via the shared
    # helper (first move supervised, predict-PAD slot excluded).
    loss_mask = build_loss_mask(C, game_lengths, seq_len)

    # outcome_offset carries the constant prefix width C (the slot where
    # moves start) — see the Corpus field docstring.
    outcome_offset = np.full(n, C, dtype=np.int32)

    # Stay on host. The trainer's prefetch loop is responsible for
    # batched device transfer; eagerly materialising a multi-GB Lichess
    # corpus on the JAX device would blow the accelerator's memory
    # budget before training even starts.
    return Corpus(
        tokens=tokens,
        targets=targets,
        attn_mask=attn_mask,
        loss_mask=loss_mask,
        outcome_offset=outcome_offset,
        game_lengths=game_lengths.astype(np.int32),
    )


def pack_corpus(
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    outcome_tokens: np.ndarray,
    *,
    seq_len: int,
    conditioning: Sequence[str] = (),
) -> Corpus:
    """Pack pre-tokenised games into a :class:`Corpus`.

    Use this for any source that already has tokenised moves and
    outcome tokens (e.g. Lichess parquet via
    :mod:`pawn.lichess_data`). ``outcome_tokens`` is the per-game
    outcome **token ID** (one of the :data:`pawn.config.WHITE_CHECKMATES`
    / etc. constants). ``conditioning`` is the ordered list of control
    kinds to prepend (see :data:`pawn.config.CONDITIONING_KINDS`);
    the default ``()`` prepends only BOS (``C = 1``).
    """
    return _pack_clm(
        move_ids,
        game_lengths,
        outcome_tokens,
        seq_len=seq_len,
        conditioning=conditioning,
    )


def generate_corpus(
    n_games: int,
    max_ply: int,
    seq_len: int,
    seed: int,
    *,
    conditioning: Sequence[str] = (),
    mate_boost: float = 0.0,
    discard_ply_limit: bool = False,
) -> Corpus:
    """Generate ``n_games`` random self-play games via the Rust engine and
    pack them into a :class:`Corpus`.

    ``max_ply`` is the per-game length cap inside the engine; the
    engine truncates with a ``PlyLimit`` termination code if a game
    runs past it. ``seq_len`` is the output sequence width; the
    packing helper truncates moves past ``seq_len - C`` where
    ``C = 1 + len(conditioning)``.

    ``seed`` is the engine RNG seed for reproducible runs.

    ``mate_boost`` (B2 / plan §8.3) biases the engine's random move
    selection toward checkmating lines — ``0.0`` (the default) is pure
    uniform self-play; a positive value upweights mate-delivering moves so
    the corpus carries a higher density of decisive terminations. It maps
    onto :data:`pawn.run_config.BaseRunConfig.mate_boost` and is the only
    consumer of that field; ``train_jax.py`` passes ``cfg.mate_boost``
    here. ``discard_ply_limit`` drops games that hit the ``max_ply`` cap
    (``PlyLimit`` termination) instead of keeping the truncated tail, so
    the corpus contains only naturally-terminated games — it maps onto
    :data:`pawn.run_config.BaseRunConfig.discard_ply_limit`.
    """
    if n_games <= 0:
        raise ValueError(f"n_games must be positive, got {n_games}")
    if max_ply <= 0:
        raise ValueError(f"max_ply must be positive, got {max_ply}")
    move_ids, game_lengths, term_codes = engine.generate_random_games(
        n_games, max_ply, seed,
        discard_ply_limit=discard_ply_limit, mate_boost=mate_boost,
    )
    outcome_tokens = _map_termination_to_outcome(term_codes, game_lengths)
    return _pack_clm(
        move_ids,
        game_lengths,
        outcome_tokens,
        seq_len=seq_len,
        conditioning=conditioning,
    )
