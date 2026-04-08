"""Backward compatibility tests for the vocabulary transition.

Verifies that models and data pipelines work correctly with both the new
searchless_chess vocabulary (1980 tokens) and the legacy PAWN vocabulary
(4284 tokens). This is critical for loading old checkpoints trained with
the legacy vocab.
"""

from __future__ import annotations

import pytest
import torch

import chess_engine as engine

from pawn.config import (
    CLMConfig,
    LegacyVocab,
    NUM_ACTIONS,
    OUTCOME_TOKEN_BASE,
    PAD_TOKEN,
    PLY_LIMIT,
    WHITE_CHECKMATES,
)
from pawn.model import PAWNCLM


# ---------------------------------------------------------------------------
# Config round-trips
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_old_config_roundtrip():
    """CLMConfig(vocab_size=4284) constructs correctly with legacy constants."""
    cfg = CLMConfig(vocab_size=4284)
    assert cfg.vocab_size == 4284

    torch.manual_seed(0)
    model = PAWNCLM(cfg)

    # lm_head projects to old vocab size
    assert model.lm_head.out_features == 4284

    # Decomposition table has legacy n_actions rows
    assert model.embed.decomp_table.shape == (4272, 3)


@pytest.mark.unit
def test_new_config_roundtrip():
    """CLMConfig() default has the new searchless_chess vocab."""
    cfg = CLMConfig()
    assert cfg.vocab_size == 1980

    torch.manual_seed(0)
    model = PAWNCLM(cfg)

    # lm_head projects to new vocab size
    assert model.lm_head.out_features == 1980

    # Decomposition table has new n_actions rows
    assert model.embed.decomp_table.shape == (1968, 3)


# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_old_model_forward_pass():
    """Legacy-vocab model produces correct output shape on a forward pass."""
    cfg = CLMConfig(
        vocab_size=4284,
        max_seq_len=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
    )
    torch.manual_seed(0)
    model = PAWNCLM(cfg)
    model.eval()

    input_ids = torch.randint(1, 4272, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.bool)

    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask)

    assert logits.shape == (2, 16, 4284)


@pytest.mark.unit
def test_new_model_forward_pass():
    """New-vocab model produces correct output shape on a forward pass."""
    cfg = CLMConfig(
        max_seq_len=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
    )
    torch.manual_seed(0)
    model = PAWNCLM(cfg)
    model.eval()

    input_ids = torch.randint(0, 1968, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.bool)

    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask)

    assert logits.shape == (2, 16, 1980)


# ---------------------------------------------------------------------------
# Factored embedding equivalence
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_factored_embedding_equivalence():
    """e2e4 decomposes to the same (src, dst, promo) in both vocabs.

    New action index for e2e4 is 317; legacy token is 797 (src=12, dst=28,
    12*64+28+1=797). Both decomposition tables must map to (src=12, dst=28,
    promo=0), confirming factored embeddings are vocab-independent.
    """
    new_cfg = CLMConfig(
        max_seq_len=32, d_model=64, n_layers=2, n_heads=4, d_ff=128
    )
    old_cfg = CLMConfig(
        vocab_size=4284, max_seq_len=32, d_model=64, n_layers=2, n_heads=4, d_ff=128
    )

    torch.manual_seed(0)
    new_model = PAWNCLM(new_cfg)
    torch.manual_seed(0)
    old_model = PAWNCLM(old_cfg)

    new_decomp = new_model.embed.decomp_table[317]  # e2e4 in new vocab
    old_decomp = old_model.embed.decomp_table[797]  # e2e4 in legacy vocab

    assert new_decomp.tolist() == [12, 28, 0], f"new decomp: {new_decomp.tolist()}"
    assert old_decomp.tolist() == [12, 28, 0], f"old decomp: {old_decomp.tolist()}"
    assert new_decomp.tolist() == old_decomp.tolist()


# ---------------------------------------------------------------------------
# Conversion round-trips
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_conversion_roundtrip_all_actions():
    """pawn_to_searchless(searchless_to_pawn(action)) == action for all 1968 actions."""
    for action in range(1968):
        pawn_token = engine.searchless_to_pawn(action)
        back = engine.pawn_to_searchless(pawn_token)
        assert back == action, (
            f"Round-trip failed for action {action}: "
            f"searchless_to_pawn={pawn_token}, pawn_to_searchless={back}"
        )


@pytest.mark.unit
def test_conversion_roundtrip_valid_pawn_tokens():
    """searchless_to_pawn(pawn_to_searchless(token)) == token for all valid legacy tokens."""
    for action in range(1968):
        pawn_token = engine.searchless_to_pawn(action)
        back_action = engine.pawn_to_searchless(pawn_token)
        back_pawn = engine.searchless_to_pawn(back_action)
        assert back_pawn == pawn_token, (
            f"Round-trip failed for pawn_token {pawn_token}: "
            f"pawn_to_searchless={back_action}, searchless_to_pawn={back_pawn}"
        )


@pytest.mark.unit
def test_pawn_to_searchless_impossible_returns_minus_one():
    """Impossible legacy tokens map to -1 (no valid searchless action)."""
    # Old PAD token (0) has no corresponding action
    assert engine.pawn_to_searchless(0) == -1

    # a1a1 (token 1 in legacy = src*64+dst+1 = 0*64+0+1=1) is impossible
    assert engine.pawn_to_searchless(1) == -1


# ---------------------------------------------------------------------------
# CLM batch generation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_old_vocab_clm_batch():
    """generate_clm_batch with prepend_outcome=True produces correct token ranges."""
    input_ids, targets, loss_mask, move_ids, game_lengths, term_codes = (
        engine.generate_clm_batch(4, 32, 42, prepend_outcome=True)
    )

    ids = input_ids.ravel()

    # Outcome tokens should be in [1969, 1973] (pretraining outcomes)
    outcome_tokens = ids[ids >= OUTCOME_TOKEN_BASE]
    assert len(outcome_tokens) > 0, "Expected at least one outcome token"
    assert outcome_tokens.min() >= WHITE_CHECKMATES  # 1969
    assert outcome_tokens.max() <= PLY_LIMIT  # 1973

    # Move tokens should be in [0, 1967]
    non_pad_non_outcome = ids[(ids != PAD_TOKEN) & (ids < OUTCOME_TOKEN_BASE)]
    assert non_pad_non_outcome.min() >= 0
    assert non_pad_non_outcome.max() <= NUM_ACTIONS - 1  # 1967

    # Positions beyond game length should be PAD (1968)
    import numpy as np

    for i in range(4):
        gl = int(game_lengths[i])
        # With prepend_outcome, positions 0..gl are content (outcome + moves)
        padding_region = input_ids[i, gl + 1 :]
        if len(padding_region) > 0:
            assert np.all(padding_region == PAD_TOKEN), (
                f"Game {i}: expected PAD in tail, got {padding_region}"
            )


@pytest.mark.unit
def test_new_vocab_clm_batch_no_outcome():
    """generate_clm_batch default (no outcome) has no outcome tokens."""
    input_ids, targets, loss_mask, move_ids, game_lengths, term_codes = (
        engine.generate_clm_batch(4, 32, 42)
    )

    ids = input_ids.ravel()

    # No tokens should be >= OUTCOME_TOKEN_BASE (1969)
    assert (ids >= OUTCOME_TOKEN_BASE).sum() == 0, (
        "No outcome tokens expected in default mode"
    )

    # Move tokens should be in [0, 1967]
    non_pad = ids[ids != PAD_TOKEN]
    assert len(non_pad) > 0, "Expected at least some move tokens"
    assert non_pad.min() >= 0
    assert non_pad.max() <= NUM_ACTIONS - 1  # 1967

    # Positions beyond game length should be PAD (1968)
    import numpy as np

    for i in range(4):
        gl = int(game_lengths[i])
        padding_region = input_ids[i, gl:]
        if len(padding_region) > 0:
            assert np.all(padding_region == PAD_TOKEN), (
                f"Game {i}: expected PAD in tail, got {padding_region}"
            )
