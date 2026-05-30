"""Tests for :mod:`pawn.corpus` — Corpus dataclass + generate/pack helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pawn.config import (
    BLACK_CHECKMATES,
    BOS_TOKEN,
    DRAW_BY_RULE,
    NULL_TOKEN,
    PAD_TOKEN,
    PLY_LIMIT,
    STALEMATE,
    WHITE_CHECKMATES,
)
from pawn.corpus import (
    Corpus,
    _map_termination_to_outcome,
    assert_conditioning_C,
    build_loss_mask,
    build_prefix,
    conditioning_to_C,
    generate_corpus,
    pack_corpus,
)


# ---------------------------------------------------------------------------
# generate_corpus (round-trip through the Rust engine)
# ---------------------------------------------------------------------------


def test_generate_corpus_runs_end_to_end() -> None:
    """The Rust engine produces a non-empty corpus with the right shapes."""
    corpus = generate_corpus(n_games=8, max_ply=32, seq_len=64, seed=42)
    assert corpus.n_games == 8
    assert corpus.seq_len == 64
    assert corpus.tokens.shape == (8, 64)
    assert corpus.targets.shape == (8, 64)
    assert corpus.attn_mask.shape == (8, 64)
    assert corpus.loss_mask.shape == (8, 64)
    assert corpus.outcome_offset.shape == (8,)


def test_generate_corpus_reproducible_via_seed() -> None:
    """Same seed → byte-identical tokens; reproducibility is the engine
    contract that the v2 data path inherits."""
    a = generate_corpus(n_games=4, max_ply=32, seq_len=64, seed=42)
    b = generate_corpus(n_games=4, max_ply=32, seq_len=64, seed=42)
    assert np.array_equal(a.tokens, b.tokens)


def test_generate_corpus_no_conditioning_layout() -> None:
    """Default empty conditioning → BOS at slot 0, moves at slot 1;
    outcome_offset == C == 1 per game."""
    corpus = generate_corpus(n_games=4, max_ply=32, seq_len=64, seed=1)
    assert np.all(corpus.outcome_offset == 1)
    # Slot 0 is BOS; slot 1 is the first real move.
    assert np.all(corpus.tokens[:, 0] == BOS_TOKEN)
    assert np.all(corpus.tokens[:, 1] != PAD_TOKEN)


def test_generate_corpus_outcome_conditioned_layout() -> None:
    """`conditioning=["outcome"]` → BOS at slot 0, outcome at slot 1,
    moves at slot 2; outcome_offset == C == 2."""
    corpus = generate_corpus(
        n_games=4, max_ply=32, seq_len=64, seed=1, conditioning=["outcome"]
    )
    assert np.all(corpus.outcome_offset == 2)
    assert np.all(corpus.tokens[:, 0] == BOS_TOKEN)
    # Slot 1 holds an outcome token (>= OUTCOME_TOKEN_BASE).
    from pawn.config import OUTCOME_TOKEN_BASE
    assert np.all(corpus.tokens[:, 1] >= OUTCOME_TOKEN_BASE)


def test_generate_corpus_rejects_zero_games() -> None:
    with pytest.raises(ValueError, match="n_games"):
        generate_corpus(n_games=0, max_ply=32, seq_len=64, seed=0)


def test_generate_corpus_rejects_zero_max_ply() -> None:
    with pytest.raises(ValueError, match="max_ply"):
        generate_corpus(n_games=4, max_ply=0, seq_len=64, seed=0)


# ---------------------------------------------------------------------------
# pack_corpus with synthetic data — control over edge cases
# ---------------------------------------------------------------------------


def _synthetic_games(
    n: int, lengths: list[int], max_ply: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (move_ids, game_lengths) pair where moves are 1, 2, 3, ...
    (so easy to spot in test failures)."""
    assert len(lengths) == n
    move_ids = np.full((n, max_ply), PAD_TOKEN, dtype=np.int16)
    for i, gl in enumerate(lengths):
        move_ids[i, :gl] = np.arange(1, gl + 1, dtype=np.int16)
    return move_ids, np.asarray(lengths, dtype=np.int16)


def test_pack_corpus_no_conditioning_basic() -> None:
    """A two-game corpus with lengths [5, 10] under empty conditioning:
    BOS at slot 0, moves at slots 1..gl, loss covers C-1..C-1+gl-1."""
    move_ids, game_lengths = _synthetic_games(2, [5, 10])
    outcome_tokens = np.array([WHITE_CHECKMATES, DRAW_BY_RULE], dtype=np.int32)
    corpus = pack_corpus(
        move_ids, game_lengths, outcome_tokens, seq_len=16, conditioning=()
    )
    # C == 1: slot 0 is BOS, moves occupy slots 1..5.
    assert int(corpus.tokens[0, 0]) == BOS_TOKEN
    assert corpus.tokens[0, 1] == 1
    assert corpus.tokens[0, 5] == 5
    assert corpus.tokens[0, 6] == PAD_TOKEN
    # attn_mask True on BOS + 5 real moves = 6 positions.
    expected_attn_0 = np.array([True] * 6 + [False] * 10)
    assert np.array_equal(corpus.attn_mask[0], expected_attn_0)
    # targets = tokens shifted left. Slot 0 (BOS) predicts ply_1 = move 1.
    assert corpus.targets[0, 0] == 1  # BOS -> first move (supervised)
    assert corpus.targets[0, 4] == 5  # predicts the 5th move
    assert corpus.targets[0, 5] == PAD_TOKEN  # predict-PAD slot
    # loss_mask: C-1=0 .. C-1+gl-1=4 → 5 positions (= game_length).
    assert int(corpus.loss_mask[0].sum()) == 5
    assert int(corpus.loss_mask[1].sum()) == 10
    # First-move BOS->ply_1 is supervised; predict-PAD slot is not.
    assert bool(corpus.loss_mask[0, 0])
    assert not bool(corpus.loss_mask[0, 5])
    # outcome_offset holds C == 1.
    assert int(corpus.outcome_offset[0]) == 1


def test_pack_corpus_outcome_conditioned_basic() -> None:
    """Outcome-conditioned layout (C=2): BOS at slot 0, outcome at slot
    1, moves at slots 2..gl+1. loss covers C-1..C-1+gl-1 (== game_length
    positions, same count as no-conditioning — the prefix slot count
    doesn't change how many moves get supervised)."""
    move_ids, game_lengths = _synthetic_games(2, [5, 10])
    outcome_tokens = np.array([WHITE_CHECKMATES, DRAW_BY_RULE], dtype=np.int32)
    corpus = pack_corpus(
        move_ids, game_lengths, outcome_tokens, seq_len=16,
        conditioning=["outcome"],
    )
    # Slot 0 BOS, slot 1 outcome.
    assert int(corpus.tokens[0, 0]) == BOS_TOKEN
    assert int(corpus.tokens[0, 1]) == WHITE_CHECKMATES
    assert int(corpus.tokens[1, 1]) == DRAW_BY_RULE
    # Moves start at slot 2.
    assert int(corpus.tokens[0, 2]) == 1  # first move
    assert int(corpus.tokens[0, 6]) == 5  # last move (5 moves, slots 2..6)
    assert int(corpus.tokens[0, 7]) == PAD_TOKEN
    # outcome_offset == C == 2.
    assert int(corpus.outcome_offset[0]) == 2
    # loss_mask: C-1=1 .. C-1+gl-1. game 0 gl=5 → slots 1..5 (5 positions);
    # the slot-1 (outcome) position predicts ply_1 (first move supervised).
    assert int(corpus.loss_mask[0].sum()) == 5
    assert int(corpus.loss_mask[1].sum()) == 10
    assert bool(corpus.loss_mask[0, 1])  # outcome-slot -> ply_1 supervised
    assert not bool(corpus.loss_mask[0, 0])  # BOS slot not supervised
    assert not bool(corpus.loss_mask[0, 6])  # predict-PAD slot not supervised


def test_pack_corpus_truncates_long_games_to_seq_len() -> None:
    """A game longer than seq_len gets truncated; loss_mask doesn't
    overflow. C=1 → 1 BOS slot + (seq_len-1) move slots."""
    move_ids, game_lengths = _synthetic_games(1, [20], max_ply=24)
    outcome_tokens = np.array([DRAW_BY_RULE], dtype=np.int32)
    corpus = pack_corpus(
        move_ids, game_lengths, outcome_tokens, seq_len=8, conditioning=()
    )
    # 8 slots: 1 BOS + 7 real moves (truncated from 20), no PAD.
    assert int(corpus.attn_mask[0].sum()) == 8
    # loss_mask: n_move_slots = 8 - 1 = 7, capped = min(20, 7) = 7.
    # Supervised slots C-1=0 .. C-1+7-1=6 = 7 positions; slot 7 (the last
    # move, with no next-token target in-window) is excluded.
    assert int(corpus.loss_mask[0].sum()) == 7


def test_pack_corpus_handles_pad_in_input() -> None:
    """The Rust engine may emit non-PAD junk past game_length in the
    raw move_ids buffer; pack_corpus zeroes those positions to PAD."""
    move_ids = np.full((1, 8), 999, dtype=np.int16)  # all "junk" = 999
    move_ids[0, :3] = [10, 20, 30]
    game_lengths = np.array([3], dtype=np.int16)
    outcome_tokens = np.array([WHITE_CHECKMATES], dtype=np.int32)
    corpus = pack_corpus(
        move_ids, game_lengths, outcome_tokens, seq_len=8, conditioning=()
    )
    # C=1: BOS at slot 0, moves [10,20,30] at slots 1..3, PAD at 4..7
    # (the junk 999s past game_length must be PAD, not leaked).
    assert int(corpus.tokens[0, 0]) == BOS_TOKEN
    assert [int(corpus.tokens[0, i]) for i in (1, 2, 3)] == [10, 20, 30]
    for i in range(4, 8):
        assert int(corpus.tokens[0, i]) == PAD_TOKEN


def test_pack_corpus_rejects_mismatched_shapes() -> None:
    move_ids = np.zeros((3, 8), dtype=np.int16)
    bad_lengths = np.zeros((4,), dtype=np.int16)  # mismatched N
    outcome = np.zeros((3,), dtype=np.int32)
    with pytest.raises(ValueError, match="game_lengths"):
        pack_corpus(move_ids, bad_lengths, outcome, seq_len=8)


def test_pack_corpus_rejects_zero_seq_len() -> None:
    move_ids = np.zeros((1, 8), dtype=np.int16)
    lengths = np.array([3], dtype=np.int16)
    outcome = np.array([0], dtype=np.int32)
    with pytest.raises(ValueError, match="seq_len"):
        pack_corpus(move_ids, lengths, outcome, seq_len=0)


def test_pack_corpus_rejects_bad_outcome_shape() -> None:
    move_ids = np.zeros((3, 8), dtype=np.int16)
    lengths = np.array([3, 4, 5], dtype=np.int16)
    bad_outcome = np.zeros((4,), dtype=np.int32)
    with pytest.raises(ValueError, match="outcome_tokens"):
        pack_corpus(move_ids, lengths, bad_outcome, seq_len=8)


def test_pack_corpus_zero_length_game_has_empty_loss_mask() -> None:
    """PR #115 review #4: a zero-length game previously got
    `loss_mask[:, 0]` = True (supervising PAD → PAD). The threshold
    should be -1 for the zero-length row so no positions are
    supervised."""
    move_ids = np.zeros((2, 8), dtype=np.int16)
    # Game 0 has 3 real moves; game 1 has 0 (all PAD).
    lengths = np.array([3, 0], dtype=np.int16)
    outcome = np.zeros((2,), dtype=np.int32)
    corpus = pack_corpus(
        move_ids, lengths, outcome, seq_len=8, conditioning=()
    )
    assert bool(corpus.loss_mask[0].any()), (
        "non-empty game should still supervise its real moves"
    )
    assert not bool(corpus.loss_mask[1].any()), (
        "zero-length game should have an empty loss mask"
    )
    # A zero-length game still has its BOS slot present (attn_mask True at
    # slot 0) but supervises nothing.
    assert bool(corpus.attn_mask[1, 0])
    assert int(corpus.tokens[1, 0]) == BOS_TOKEN


# ---------------------------------------------------------------------------
# Outcome-token mapping from termination codes
# ---------------------------------------------------------------------------


def test_map_termination_white_checkmate_on_odd_length() -> None:
    """Engine code 0 (Checkmate) + odd game_length → WHITE_CHECKMATES."""
    out = _map_termination_to_outcome(
        np.array([0]), np.array([7])
    )
    assert int(out[0]) == WHITE_CHECKMATES


def test_map_termination_black_checkmate_on_even_length() -> None:
    out = _map_termination_to_outcome(
        np.array([0]), np.array([8])
    )
    assert int(out[0]) == BLACK_CHECKMATES


def test_map_termination_stalemate() -> None:
    assert int(_map_termination_to_outcome(np.array([1]), np.array([20]))[0]) == STALEMATE


def test_map_termination_draws() -> None:
    """Codes 2 (75-move), 3 (fivefold), 4 (insufficient material) all
    map to DRAW_BY_RULE."""
    for code in (2, 3, 4):
        out = _map_termination_to_outcome(np.array([code]), np.array([50]))
        assert int(out[0]) == DRAW_BY_RULE


def test_map_termination_ply_limit() -> None:
    """Code 5 → PLY_LIMIT (the default fall-through)."""
    out = _map_termination_to_outcome(np.array([5]), np.array([100]))
    assert int(out[0]) == PLY_LIMIT


# ---------------------------------------------------------------------------
# Corpus dataclass invariants
# ---------------------------------------------------------------------------


def test_corpus_is_frozen() -> None:
    """Corpus is a frozen dataclass — accidentally reassigning a field
    raises."""
    corpus = generate_corpus(n_games=2, max_ply=16, seq_len=32, seed=0)
    with pytest.raises((AttributeError, TypeError)):
        corpus.tokens = np.zeros_like(corpus.tokens)  # type: ignore[misc]


def test_corpus_attn_and_loss_mask_consistency() -> None:
    """attn_mask is True over the BOS prefix + real moves; loss_mask is a
    subset of attn_mask. Layout (C=1): slot 0 BOS, slots 1..gl moves."""
    move_ids, game_lengths = _synthetic_games(3, [4, 8, 12])
    outcome_tokens = np.array(
        [WHITE_CHECKMATES, DRAW_BY_RULE, STALEMATE], dtype=np.int32
    )
    corpus = pack_corpus(
        move_ids, game_lengths, outcome_tokens, seq_len=16, conditioning=()
    )
    C = 1
    for g in range(3):
        gl = int(game_lengths[g])
        real_width = C + gl  # BOS + gl moves
        # attn_mask True over [0 .. C+gl-1], PAD afterwards.
        assert np.all(corpus.attn_mask[g, :real_width] == True)  # noqa: E712
        assert np.all(corpus.attn_mask[g, real_width:] == False)  # noqa: E712
        # loss_mask True over [C-1 .. C-1+gl-1]; everything there is real.
        lo, hi = C - 1, C - 1 + gl  # hi exclusive
        assert np.all(corpus.loss_mask[g, lo:hi] == True)  # noqa: E712
        assert np.all(corpus.loss_mask[g, :lo] == False)  # noqa: E712
        assert np.all(corpus.loss_mask[g, hi:] == False)  # noqa: E712
        # loss_mask ⊆ attn_mask.
        assert np.all(corpus.attn_mask[g][corpus.loss_mask[g]] == True)  # noqa: E712


# ---------------------------------------------------------------------------
# A.1 — Bucketing (Corpus.by_bucket)
# ---------------------------------------------------------------------------


def test_corpus_carries_game_lengths() -> None:
    """A.1 added a game_lengths field; populated from the Rust engine."""
    corpus = generate_corpus(n_games=4, max_ply=64, seq_len=128, seed=11)
    assert corpus.game_lengths.shape == (4,)
    assert corpus.game_lengths.dtype == np.int32
    # All lengths within [0, max_ply].
    assert int(corpus.game_lengths.min()) >= 0
    assert int(corpus.game_lengths.max()) <= 64


def test_by_bucket_partitions_by_length() -> None:
    """`by_bucket` assigns each game to the smallest edge >= its
    effective length (game_lengths + outcome_offset). Sub-corpora are
    truncated to the bucket's seq_len."""
    # Synthetic — control lengths exactly.
    move_ids, game_lengths = _synthetic_games(5, [3, 6, 10, 15, 30], max_ply=32)
    outcome_tokens = np.array(
        [WHITE_CHECKMATES] * 5, dtype=np.int32
    )
    corpus = pack_corpus(
        move_ids, game_lengths, outcome_tokens, seq_len=32, conditioning=()
    )
    buckets = corpus.by_bucket((8, 16, 32))
    # Effective length = game_lengths + outcome_offset; with conditioning=()
    # the prefix is BOS-only (C=1), so eff = gl+1: 4,7→bucket 8; 11,16→16; 31→32.
    assert set(buckets.keys()) == {8, 16, 32}
    assert buckets[8].n_games == 2
    assert buckets[16].n_games == 2
    assert buckets[32].n_games == 1
    # Width truncation
    assert buckets[8].tokens.shape == (2, 8)
    assert buckets[16].tokens.shape == (2, 16)
    assert buckets[32].tokens.shape == (1, 32)


def test_by_bucket_empty_dict_returns_self_at_full_len() -> None:
    """No edges → degenerate single-bucket pass-through at seq_len."""
    corpus = generate_corpus(n_games=4, max_ply=16, seq_len=32, seed=0)
    buckets = corpus.by_bucket(())
    assert set(buckets.keys()) == {32}
    assert buckets[32] is corpus


def test_by_bucket_rejects_top_edge_smaller_than_seq_len() -> None:
    """Top edge must == seq_len so every game fits."""
    corpus = generate_corpus(n_games=2, max_ply=16, seq_len=32, seed=0)
    with pytest.raises(ValueError, match="top bucket edge"):
        corpus.by_bucket((8, 16))


# ---------------------------------------------------------------------------
# Chunk 4 — direct unit tests for the shared layout helpers
# ---------------------------------------------------------------------------


def test_conditioning_to_C_counts_bos_plus_kinds() -> None:
    """C = 1 (BOS) + len(conditioning)."""
    assert conditioning_to_C(()) == 1
    assert conditioning_to_C(["outcome"]) == 2


def test_conditioning_to_C_rejects_unknown_kind() -> None:
    """A typo'd / unregistered kind is a hard error, not a silent
    NULL-only slot."""
    with pytest.raises(ValueError, match="unknown conditioning kind"):
        conditioning_to_C(["bogus"])


def test_build_loss_mask_c1_first_move_supervised_predict_pad_excluded() -> None:
    """C=1: supervised slots are exactly [C-1 .. C-1 + gl - 1] = [0 .. gl-1].
    The first move (slot 0 → ply_1) IS supervised; the predict-PAD slot
    (slot gl) is NOT."""
    game_lengths = np.array([3, 5], dtype=np.int32)
    mask = build_loss_mask(1, game_lengths, seq_len=8)
    assert mask.shape == (2, 8)
    # Row 0: gl=3 → slots 0,1,2 True; 3.. False.
    assert mask[0].tolist() == [True, True, True, False, False, False, False, False]
    # Row 1: gl=5 → slots 0..4 True; 5.. (incl. predict-PAD at 5) False.
    assert mask[1].tolist() == [True, True, True, True, True, False, False, False]


def test_build_loss_mask_c2_offset_by_one() -> None:
    """C=2: prefix occupies slots 0 (BOS) and 1 (cond); the last prefix
    slot (C-1 == 1) predicts ply_1, so supervised slots are
    [1 .. 1 + gl - 1]."""
    game_lengths = np.array([3], dtype=np.int32)
    mask = build_loss_mask(2, game_lengths, seq_len=8)
    # gl=3 → slots 1,2,3 True; slot 0 (BOS→cond, not a move) and slot 4
    # (predict-PAD) False.
    assert mask[0].tolist() == [False, True, True, True, False, False, False, False]


def test_build_loss_mask_truncates_game_to_fit_seq_len() -> None:
    """A game longer than the available move slots (seq_len - C) is capped
    so the mask never overruns."""
    game_lengths = np.array([20], dtype=np.int32)
    mask = build_loss_mask(2, game_lengths, seq_len=8)
    # n_move_slots = 8 - 2 = 6 → supervised [1 .. 6], the remaining slot 7
    # is the predict-PAD slot and stays False.
    assert int(mask[0].sum()) == 6
    assert mask[0].tolist() == [False, True, True, True, True, True, True, False]


def test_build_loss_mask_zero_length_game_is_all_false() -> None:
    """A zero-length game supervises no positions."""
    game_lengths = np.array([0], dtype=np.int32)
    mask = build_loss_mask(1, game_lengths, seq_len=8)
    assert not mask[0].any()


def test_build_prefix_bos_at_slot_zero_and_outcome_resolved() -> None:
    """Slot 0 is always BOS; the `outcome` kind resolves to the per-game
    outcome token in slot 1."""
    outcome_tokens = np.array(
        [WHITE_CHECKMATES, DRAW_BY_RULE], dtype=np.int32
    )
    prefix = build_prefix(["outcome"], outcome_tokens, n=2)
    assert prefix.shape == (2, 2)
    assert np.all(prefix[:, 0] == BOS_TOKEN)
    assert prefix[0, 1] == WHITE_CHECKMATES
    assert prefix[1, 1] == DRAW_BY_RULE


def test_build_prefix_bos_only_when_no_conditioning() -> None:
    """conditioning=() → C=1, a single BOS slot."""
    outcome_tokens = np.array([WHITE_CHECKMATES], dtype=np.int32)
    prefix = build_prefix((), outcome_tokens, n=1)
    assert prefix.shape == (1, 1)
    assert prefix[0, 0] == BOS_TOKEN


def test_build_prefix_null_fills_unknown_outcome() -> None:
    """A game whose outcome is genuinely unknown (sentinel < 0) resolves
    to NULL_TOKEN rather than holding an out-of-vocab value."""
    outcome_tokens = np.array([WHITE_CHECKMATES, -1], dtype=np.int32)
    prefix = build_prefix(["outcome"], outcome_tokens, n=2)
    assert prefix[0, 1] == WHITE_CHECKMATES
    assert prefix[1, 1] == NULL_TOKEN


def test_assert_conditioning_C_passes_on_match() -> None:
    """A builder whose conditioning implies the checkpoint's C is fine."""
    # conditioning=["outcome"] → C=2.
    assert_conditioning_C(["outcome"], checkpoint_C=2)
    # conditioning=() → C=1.
    assert_conditioning_C((), checkpoint_C=1)


def test_assert_conditioning_C_raises_on_mismatch() -> None:
    """A C mismatch is the silent-RoPE-drift guard — it must raise."""
    with pytest.raises(ValueError, match="conditioning mismatch"):
        # conditioning=["outcome"] → C=2, but the checkpoint was trained C=1.
        assert_conditioning_C(["outcome"], checkpoint_C=1)
