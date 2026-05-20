"""Tests for ``pawn.jax.corpus``."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
pytest.importorskip("chess_engine")

import numpy as np

from pawn.jax.config import (
    N_OUTCOMES,
    OUTCOME_TOKEN_BASE,
    PAD_TOKEN,
    TINY_SUPERNET,
    TINY_VARIANTS,
    VARIANTS,
    validate_nested,
)
from pawn.jax.corpus import (
    _OFFSET_BLACK_CHECKMATES,
    _OFFSET_DRAW_BY_RULE,
    _OFFSET_PLY_LIMIT,
    _OFFSET_STALEMATE,
    _OFFSET_WHITE_CHECKMATES,
    _map_term_to_outcome_offset,
    generate_corpus,
    outcome_tokens,
)


def test_corpus_shapes_and_dtypes() -> None:
    """Basic invariants: shape matches request, dtypes are the
    documented int32/bool/uint8."""
    c = generate_corpus(n_games=8, max_ply=64, seed=0)
    assert c.n_games == 8
    assert c.seq_len == 64
    assert c.tokens.shape == (8, 64)
    assert c.attn_mask.shape == (8, 64)
    assert c.targets.shape == (8, 64)
    assert c.loss_mask.shape == (8, 64)
    assert c.outcome_offset.shape == (8,)
    assert c.tokens.dtype == np.int32
    assert c.attn_mask.dtype == np.bool_
    assert c.targets.dtype == np.int32
    assert c.loss_mask.dtype == np.bool_
    assert c.outcome_offset.dtype == np.uint8


def test_corpus_token_range() -> None:
    """Every token is either a valid move id, PAD, or unused-PAD. No
    out-of-vocab values."""
    c = generate_corpus(n_games=16, max_ply=64, seed=1)
    valid = (c.tokens >= 0) & (c.tokens < 1968)
    pad = c.tokens == PAD_TOKEN
    assert (valid | pad).all(), (
        f"unexpected token values: min={c.tokens.min()}, max={c.tokens.max()}"
    )


def test_corpus_attn_mask_aligned_with_pad() -> None:
    """Where ``attn_mask`` is False the token must be PAD; where True
    the token must be a valid move id."""
    c = generate_corpus(n_games=16, max_ply=64, seed=2)
    pad_positions = ~c.attn_mask
    assert (c.tokens[pad_positions] == PAD_TOKEN).all()
    valid_positions = c.attn_mask
    valid_moves = c.tokens[valid_positions]
    assert ((valid_moves >= 0) & (valid_moves < 1968)).all()


def test_corpus_targets_are_input_shifted() -> None:
    """The trainer expects ``targets[t] == tokens[t+1]`` for ``t < T-1``
    and ``targets[T-1] == PAD``."""
    c = generate_corpus(n_games=4, max_ply=32, seed=3)
    assert np.array_equal(c.targets[:, :-1], c.tokens[:, 1:])
    assert (c.targets[:, -1] == PAD_TOKEN).all()


def test_corpus_loss_mask_in_attn_mask() -> None:
    """Every position where loss applies must also be a real input
    position — predicting from a PAD position is meaningless.

    Per the legacy ``prepend_outcome=False`` convention: loss covers
    positions ``0..game_length - 1`` inclusive — i.e. ``game_length``
    positions. The last covered position (``game_length - 1``) predicts
    PAD, which is the supervised game-over signal. So
    ``loss_count == game_length`` (= ``attn_mask`` count) except for
    zero-move games, where the ``& attn_mask`` zeros position 0 out
    of the raw threshold-derived mask."""
    c = generate_corpus(n_games=16, max_ply=64, seed=4)
    assert (c.loss_mask & ~c.attn_mask).sum() == 0, (
        "loss_mask must be a subset of attn_mask"
    )
    real_per_game = c.attn_mask.sum(axis=1)
    loss_per_game = c.loss_mask.sum(axis=1)
    # For game_length > 0: loss_count == game_length. For length 0
    # the ``& attn_mask`` step zeroed it, but ``real_per_game[i]`` is
    # also 0 in that case — so the identity ``expected == real_per_game``
    # holds either way.
    assert np.array_equal(loss_per_game, real_per_game), (
        f"loss-mask count {loss_per_game.tolist()} != real_per_game "
        f"{real_per_game.tolist()}"
    )


def test_corpus_seed_reproducible() -> None:
    """Same seed → identical corpus. The trainer's atomic-resume story
    depends on this."""
    a = generate_corpus(n_games=8, max_ply=64, seed=42)
    b = generate_corpus(n_games=8, max_ply=64, seed=42)
    assert np.array_equal(a.tokens, b.tokens)
    assert np.array_equal(a.attn_mask, b.attn_mask)
    assert np.array_equal(a.outcome_offset, b.outcome_offset)


def test_corpus_seed_differs() -> None:
    """Different seeds → different tokens (with extremely high probability)."""
    a = generate_corpus(n_games=16, max_ply=64, seed=1)
    b = generate_corpus(n_games=16, max_ply=64, seed=2)
    assert not np.array_equal(a.tokens, b.tokens)


def test_corpus_seq_len_larger_than_max_ply() -> None:
    """``seq_len > max_ply`` right-pads the trailing slots with PAD and
    keeps ``attn_mask`` aligned (moves are left-aligned at positions
    ``0..N-1``; PAD fills the right tail)."""
    c = generate_corpus(n_games=4, max_ply=32, seq_len=64, seed=5)
    assert c.seq_len == 64
    # Positions beyond max_ply must all be PAD + attn_mask=False.
    assert (c.tokens[:, 32:] == PAD_TOKEN).all()
    assert (c.attn_mask[:, 32:] == False).all()  # noqa: E712


def test_corpus_seq_len_smaller_than_max_ply_raises() -> None:
    """The packer refuses to truncate — caller would lose moves."""
    with pytest.raises(ValueError, match="seq_len=.*max_ply"):
        generate_corpus(n_games=4, max_ply=64, seq_len=32, seed=0)


def test_outcome_offsets_in_range() -> None:
    """Outcome offsets occupy ``[0, N_OUTCOMES)`` — required by the
    embedding table, which is sized exactly ``N_OUTCOMES``."""
    c = generate_corpus(n_games=32, max_ply=64, seed=6)
    assert ((c.outcome_offset >= 0) & (c.outcome_offset < N_OUTCOMES)).all()


def test_outcome_tokens_offset_by_base() -> None:
    """``outcome_tokens()`` adds ``OUTCOME_TOKEN_BASE`` so the result
    lands in the model's outcome-token vocab slot."""
    c = generate_corpus(n_games=8, max_ply=64, seed=7)
    toks = outcome_tokens(c)
    assert toks.dtype == np.int32
    assert np.array_equal(toks, c.outcome_offset.astype(np.int32) + OUTCOME_TOKEN_BASE)
    # And the resulting values are in the documented outcome-token band.
    assert ((toks >= OUTCOME_TOKEN_BASE) & (toks < OUTCOME_TOKEN_BASE + N_OUTCOMES)).all()


def test_tiny_supernet_variants_nest_cleanly() -> None:
    """All three TINY_VARIANTS must satisfy ``validate_nested`` against
    TINY_SUPERNET. This is the load-bearing invariant the Phase-2 run
    relies on."""
    for name, variant in TINY_VARIANTS.items():
        validate_nested(variant, TINY_SUPERNET)
        assert variant.head_dim == TINY_SUPERNET.head_dim, (
            f"{name} has head_dim={variant.head_dim} != "
            f"supernet head_dim={TINY_SUPERNET.head_dim}"
        )


def test_tiny_supernet_three_distinct_widths() -> None:
    """The §11 doc claim — TINY_SUPERNET admits three distinct nested
    slices — needs to actually hold. Pins against a future regression
    that collapses two variants into the same width."""
    widths = {v.d_model for v in TINY_VARIANTS.values()}
    assert len(widths) == 3, (
        f"expected 3 distinct widths in TINY_VARIANTS, got {widths}"
    )
    assert widths == {64, 128, 192}


def test_production_supernet_variants_still_nest() -> None:
    """Regression guard: the production ``VARIANTS`` must continue to
    nest under ``SUPERNET`` even as we add the tiny variants. The two
    nesting trees are independent but share validation logic."""
    from pawn.jax.config import SUPERNET
    for variant in VARIANTS.values():
        validate_nested(variant, SUPERNET)


def test_term_to_outcome_offset_all_codes_and_parities() -> None:
    """Direct unit test of ``_map_term_to_outcome_offset``. The
    white/black checkmate parity hinges on ``game_length % 2``; a
    silent inversion (or a future engine convention change) would
    corrupt outcome-conditional probes for an entire training run
    with no obvious loss-curve signal. Pin every code path explicitly.

    Engine convention (engine/src/types.rs::Termination):
        0 = Checkmate     (parity by game_length)
        1 = Stalemate     → STALEMATE
        2 = SeventyFiveMoveRule  → DRAW_BY_RULE
        3 = FivefoldRep   → DRAW_BY_RULE
        4 = InsufficientMaterial → DRAW_BY_RULE
        5 = PlyLimit      → PLY_LIMIT  (also: default fallback)
    """
    # Six synthetic games, one per term-code path (with both checkmate
    # parities). Game lengths chosen to be small and distinguishable.
    term_codes = np.array([0, 0, 1, 2, 3, 4, 5], dtype=np.int32)
    game_lengths = np.array([5, 4, 9, 9, 9, 9, 9], dtype=np.int32)
    # The first two games are checkmates with odd (5) vs even (4) game
    # length: legacy convention says odd → white delivered mate.
    expected = np.array(
        [
            _OFFSET_WHITE_CHECKMATES,  # mate, length=5 (odd → white)
            _OFFSET_BLACK_CHECKMATES,  # mate, length=4 (even → black)
            _OFFSET_STALEMATE,         # stalemate
            _OFFSET_DRAW_BY_RULE,      # 75-move rule
            _OFFSET_DRAW_BY_RULE,      # fivefold repetition
            _OFFSET_DRAW_BY_RULE,      # insufficient material
            _OFFSET_PLY_LIMIT,         # ply limit
        ],
        dtype=np.uint8,
    )
    got = _map_term_to_outcome_offset(term_codes, game_lengths)
    assert got.dtype == np.uint8
    assert np.array_equal(got, expected), (
        f"outcome-offset mapping diverged: got {got.tolist()}, "
        f"expected {expected.tolist()}"
    )


def test_term_to_outcome_offset_unknown_term_code_falls_to_ply_limit() -> None:
    """A term code outside the known set falls to the default
    ``PLY_LIMIT`` slot. Pins the safety net so a future engine that
    adds a new code (e.g. ``6 = StalemateByMoveLimit``) does not
    silently produce out-of-vocab offsets — they default to
    PLY_LIMIT until the table is updated."""
    term_codes = np.array([99], dtype=np.int32)
    game_lengths = np.array([10], dtype=np.int32)
    got = _map_term_to_outcome_offset(term_codes, game_lengths)
    assert got.tolist() == [_OFFSET_PLY_LIMIT]


def test_corpus_terminal_target_is_pad() -> None:
    """For every real game of known length, ``targets[i, gl - 1]``
    must equal PAD — that's the supervised game-end signal, the load-
    bearing detail behind the loss_mask convention. A future refactor
    that drops the right-shift's terminal PAD would silently change
    what the trainer learns on the final-move position."""
    c = generate_corpus(n_games=16, max_ply=64, seed=8)
    # Per-game last-supervised position = game_length - 1.
    real_per_game = c.attn_mask.sum(axis=1).astype(np.int32)
    for i in range(c.n_games):
        gl = int(real_per_game[i])
        if gl == 0:
            continue
        assert c.targets[i, gl - 1] == PAD_TOKEN, (
            f"game {i} (length {gl}): terminal target should be PAD "
            f"(={PAD_TOKEN}), got {int(c.targets[i, gl - 1])}"
        )
        # Sanity: the position just before that must NOT be PAD (it
        # predicts a real next move).
        if gl >= 2:
            assert c.targets[i, gl - 2] != PAD_TOKEN, (
                f"game {i} (length {gl}): pre-terminal target should "
                f"be a real move, got PAD"
            )
