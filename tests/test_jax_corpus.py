"""Tests for :mod:`pawn.corpus` — Corpus dataclass + generate/pack helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pawn.config import (
    BLACK_CHECKMATES,
    DRAW_BY_RULE,
    PAD_TOKEN,
    PLY_LIMIT,
    STALEMATE,
    WHITE_CHECKMATES,
)
from pawn.corpus import (
    Corpus,
    _map_termination_to_outcome,
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


def test_generate_corpus_pure_moves_layout() -> None:
    """Default `prepend_outcome=False` puts moves at slot 0; outcome
    offsets are 0 per game."""
    corpus = generate_corpus(n_games=4, max_ply=32, seq_len=64, seed=1)
    assert np.all(corpus.outcome_offset == 0)
    # Slot 0 is a real move (not a PAD or outcome token).
    assert np.all(corpus.tokens[:, 0] != PAD_TOKEN)


def test_generate_corpus_outcome_prefixed_layout() -> None:
    """`prepend_outcome=True` writes the outcome token at slot 0;
    outcome_offset becomes 1."""
    corpus = generate_corpus(
        n_games=4, max_ply=32, seq_len=64, seed=1, prepend_outcome=True
    )
    assert np.all(corpus.outcome_offset == 1)
    # Slot 0 holds an outcome token (>= OUTCOME_TOKEN_BASE).
    from pawn.config import OUTCOME_TOKEN_BASE
    assert np.all(corpus.tokens[:, 0] >= OUTCOME_TOKEN_BASE)


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


def test_pack_corpus_pure_moves_basic() -> None:
    """A two-game corpus with lengths [5, 10] produces correct tokens,
    targets, attn_mask, loss_mask, and outcome_offset."""
    move_ids, game_lengths = _synthetic_games(2, [5, 10])
    outcome_tokens = np.array([WHITE_CHECKMATES, DRAW_BY_RULE], dtype=np.int32)
    corpus = pack_corpus(
        move_ids, game_lengths, outcome_tokens, seq_len=16, prepend_outcome=False
    )
    # Tokens: game 0 has 5 real moves then PAD.
    assert corpus.tokens[0, 0] == 1
    assert corpus.tokens[0, 4] == 5
    assert corpus.tokens[0, 5] == PAD_TOKEN
    # attn_mask True on real moves.
    expected_attn_0 = np.array([True] * 5 + [False] * 11)
    assert np.array_equal(corpus.attn_mask[0], expected_attn_0)
    # targets = tokens shifted left.
    assert corpus.targets[0, 0] == 2  # first target is the 2nd move
    assert corpus.targets[0, 3] == 5
    assert corpus.targets[0, 4] == PAD_TOKEN  # gl-th target is PAD
    # loss_mask: positions 0..gl-1 = 0..4 supervised (5 positions).
    assert int(corpus.loss_mask[0].sum()) == 5
    assert int(corpus.loss_mask[1].sum()) == 10
    # outcome_offset is 0 in pure-moves layout.
    assert int(corpus.outcome_offset[0]) == 0


def test_pack_corpus_outcome_prefixed_basic() -> None:
    """Outcome-prefixed layout: outcome at slot 0, moves shift right,
    loss_mask covers one MORE position than pure-moves (gl+1 vs gl)."""
    move_ids, game_lengths = _synthetic_games(2, [5, 10])
    outcome_tokens = np.array([WHITE_CHECKMATES, DRAW_BY_RULE], dtype=np.int32)
    corpus = pack_corpus(
        move_ids, game_lengths, outcome_tokens, seq_len=16, prepend_outcome=True
    )
    # Slot 0 is the outcome.
    assert int(corpus.tokens[0, 0]) == WHITE_CHECKMATES
    assert int(corpus.tokens[1, 0]) == DRAW_BY_RULE
    # Moves start at slot 1.
    assert int(corpus.tokens[0, 1]) == 1  # first move
    assert int(corpus.tokens[0, 5]) == 5  # last move (5 moves, slots 1..5)
    assert int(corpus.tokens[0, 6]) == PAD_TOKEN
    # outcome_offset == 1.
    assert int(corpus.outcome_offset[0]) == 1
    # loss_mask covers gl+1 positions.
    assert int(corpus.loss_mask[0].sum()) == 6  # gl=5, gl+1=6
    assert int(corpus.loss_mask[1].sum()) == 11  # gl=10, gl+1=11


def test_pack_corpus_truncates_long_games_to_seq_len() -> None:
    """A game longer than seq_len gets truncated; loss_mask doesn't
    overflow."""
    move_ids, game_lengths = _synthetic_games(1, [20], max_ply=24)
    outcome_tokens = np.array([DRAW_BY_RULE], dtype=np.int32)
    corpus = pack_corpus(
        move_ids, game_lengths, outcome_tokens, seq_len=8, prepend_outcome=False
    )
    # 8 slots, all real moves (truncated from 20).
    assert int(corpus.attn_mask[0].sum()) == 8
    # loss_mask: capped_lengths = min(20, 8) = 8. threshold = 8 - 1 = 7.
    # Positions 0..7 inclusive = 8 supervised positions.
    assert int(corpus.loss_mask[0].sum()) == 8


def test_pack_corpus_handles_pad_in_input() -> None:
    """The Rust engine may emit non-PAD junk past game_length in the
    raw move_ids buffer; pack_corpus zeroes those positions to PAD."""
    move_ids = np.full((1, 8), 999, dtype=np.int16)  # all "junk" = 999
    move_ids[0, :3] = [10, 20, 30]
    game_lengths = np.array([3], dtype=np.int16)
    outcome_tokens = np.array([WHITE_CHECKMATES], dtype=np.int32)
    corpus = pack_corpus(
        move_ids, game_lengths, outcome_tokens, seq_len=8, prepend_outcome=False
    )
    # Positions 3..7 should be PAD, NOT 999.
    for i in range(3, 8):
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
        move_ids, lengths, outcome, seq_len=8, prepend_outcome=False
    )
    assert bool(corpus.loss_mask[0].any()), (
        "non-empty game should still supervise its real moves"
    )
    assert not bool(corpus.loss_mask[1].any()), (
        "zero-length game should have an empty loss mask"
    )


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
    """attn_mask is True wherever tokens are real; loss_mask is a
    (non-strict) subset of attn_mask at the supervised positions."""
    move_ids, game_lengths = _synthetic_games(3, [4, 8, 12])
    outcome_tokens = np.array(
        [WHITE_CHECKMATES, DRAW_BY_RULE, STALEMATE], dtype=np.int32
    )
    corpus = pack_corpus(
        move_ids, game_lengths, outcome_tokens, seq_len=16, prepend_outcome=False
    )
    # Where loss_mask is True, the position has a real (non-PAD) token
    # (or the *target* is the PAD signal, which is still in the loss
    # via the legality penalty — but attn_mask says "real input").
    for g in range(3):
        gl = int(game_lengths[g])
        # loss_mask True for positions 0..gl-1, all of which are also
        # input-real (attn_mask True).
        assert np.all(corpus.loss_mask[g, :gl] == True)  # noqa: E712
        assert np.all(corpus.attn_mask[g, :gl] == True)  # noqa: E712
        # Positions past gl-1 are not in the loss; positions past gl-1
        # are PAD inputs.
        assert np.all(corpus.loss_mask[g, gl:] == False)  # noqa: E712
        assert np.all(corpus.attn_mask[g, gl:] == False)  # noqa: E712
