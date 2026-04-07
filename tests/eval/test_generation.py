"""Tests for pawn/eval_suite/generation.py.

Covers autoregressive_generate, OUTCOME_TOKENS, _map_term_code_to_outcome_name,
_outcome_mask, _analyze_generated_games, outcome_signal_test,
prefix_continuation_test, poisoned_prefix_test, impossible_task_test,
improbable_task_test, POISONING_PAIRS.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pawn.config import PAD_TOKEN, PLY_LIMIT, WHITE_CHECKMATES
from pawn.eval_suite.generation import (
    _analyze_generated_games,
    _map_term_code_to_outcome_name,
    _outcome_mask,
    autoregressive_generate,
    impossible_task_test,
    improbable_task_test,
    OUTCOME_TOKENS,
    poisoned_prefix_test,
    POISONING_PAIRS,
    prefix_continuation_test,
)


# ---------------------------------------------------------------------------
# OUTCOME_TOKENS
# ---------------------------------------------------------------------------


class TestOutcomeTokens:
    @pytest.mark.unit
    def test_has_5_outcomes(self):
        assert len(OUTCOME_TOKENS) == 5

    @pytest.mark.unit
    def test_contains_expected_keys(self):
        expected = {"WHITE_CHECKMATES", "BLACK_CHECKMATES",
                    "STALEMATE", "DRAW_BY_RULE", "PLY_LIMIT"}
        assert set(OUTCOME_TOKENS.keys()) == expected



# ---------------------------------------------------------------------------
# _map_term_code_to_outcome_name
# ---------------------------------------------------------------------------


class TestMapTermCodeToOutcomeName:
    @pytest.mark.unit
    def test_checkmate_white(self):
        assert _map_term_code_to_outcome_name(0, 5) == "WHITE_CHECKMATES"

    @pytest.mark.unit
    def test_checkmate_black(self):
        assert _map_term_code_to_outcome_name(0, 6) == "BLACK_CHECKMATES"

    @pytest.mark.unit
    def test_stalemate(self):
        assert _map_term_code_to_outcome_name(1, 40) == "STALEMATE"

    @pytest.mark.unit
    def test_draw_codes(self):
        for tc in (2, 3, 4):
            assert _map_term_code_to_outcome_name(tc, 40) == "DRAW_BY_RULE"

    @pytest.mark.unit
    def test_ply_limit(self):
        assert _map_term_code_to_outcome_name(5, 255) == "PLY_LIMIT"

    @pytest.mark.unit
    def test_premature_pad(self):
        assert _map_term_code_to_outcome_name(-2, 10) == "PREMATURE_PAD"

    @pytest.mark.unit
    def test_forfeit(self):
        assert _map_term_code_to_outcome_name(-3, 10) == "FORFEIT"

    @pytest.mark.unit
    def test_unknown(self):
        assert _map_term_code_to_outcome_name(99, 10) == "UNKNOWN"


# ---------------------------------------------------------------------------
# _outcome_mask
# ---------------------------------------------------------------------------


class TestOutcomeMask:
    @pytest.mark.unit
    def test_white_checkmates(self):
        tc = np.array([0, 0, 1, 0], dtype=np.int8)
        gl = np.array([5, 6, 40, 7], dtype=np.int32)  # odd = white wins
        mask = _outcome_mask(tc, gl, "WHITE_CHECKMATES")
        assert mask.tolist() == [True, False, False, True]

    @pytest.mark.unit
    def test_black_checkmates(self):
        tc = np.array([0, 0, 1, 0], dtype=np.int8)
        gl = np.array([5, 6, 40, 8], dtype=np.int32)  # even = black wins
        mask = _outcome_mask(tc, gl, "BLACK_CHECKMATES")
        assert mask.tolist() == [False, True, False, True]

    @pytest.mark.unit
    def test_stalemate(self):
        tc = np.array([0, 1, 2, 1], dtype=np.int8)
        gl = np.array([5, 40, 50, 60], dtype=np.int32)
        mask = _outcome_mask(tc, gl, "STALEMATE")
        assert mask.tolist() == [False, True, False, True]

    @pytest.mark.unit
    def test_draw_by_rule(self):
        tc = np.array([1, 2, 3, 4, 5], dtype=np.int8)
        gl = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        mask = _outcome_mask(tc, gl, "DRAW_BY_RULE")
        assert mask.tolist() == [False, True, True, True, False]

    @pytest.mark.unit
    def test_ply_limit(self):
        tc = np.array([0, 5, 5, 1], dtype=np.int8)
        gl = np.array([5, 255, 255, 40], dtype=np.int32)
        mask = _outcome_mask(tc, gl, "PLY_LIMIT")
        assert mask.tolist() == [False, True, True, False]

    @pytest.mark.unit
    def test_unknown_outcome_returns_all_false(self):
        tc = np.array([0, 1, 2], dtype=np.int8)
        gl = np.array([5, 10, 15], dtype=np.int32)
        mask = _outcome_mask(tc, gl, "SOMETHING_ELSE")
        assert mask.tolist() == [False, False, False]


# ---------------------------------------------------------------------------
# POISONING_PAIRS
# ---------------------------------------------------------------------------


class TestPoisoningPairs:
    @pytest.mark.unit
    def test_is_list(self):
        assert isinstance(POISONING_PAIRS, list)
        assert len(POISONING_PAIRS) > 0

    @pytest.mark.unit
    def test_entries_are_tuples(self):
        for pair in POISONING_PAIRS:
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    @pytest.mark.unit
    def test_outcomes_are_valid(self):
        for actual, poisoned in POISONING_PAIRS:
            assert actual in OUTCOME_TOKENS
            assert poisoned in OUTCOME_TOKENS


# ---------------------------------------------------------------------------
# _analyze_generated_games
# ---------------------------------------------------------------------------


def _make_gen_dict(n, max_seq_len=16, term_codes=None, game_lengths=None,
                   forfeit_ply=None):
    """Build a synthetic gen dict for analysis tests."""
    if term_codes is None:
        term_codes = np.zeros(n, dtype=np.int8)
    if game_lengths is None:
        game_lengths = np.full(n, 4, dtype=np.int32)
    if forfeit_ply is None:
        forfeit_ply = np.full(n, -1, dtype=np.int32)
    sequences = np.full((n, max_seq_len), PAD_TOKEN, dtype=np.int64)
    for i in range(n):
        gl = game_lengths[i]
        # Place outcome at 0, moves 1..gl
        sequences[i, 0] = WHITE_CHECKMATES
        for p in range(1, gl + 1):
            sequences[i, p] = 1  # arbitrary non-PAD token
    return {
        "sequences": sequences,
        "term_codes": term_codes,
        "game_lengths": game_lengths,
        "forfeit_ply": forfeit_ply,
    }


class TestAnalyzeGeneratedGames:
    @pytest.mark.unit
    def test_returns_expected_keys(self):
        gen = _make_gen_dict(3)
        result = _analyze_generated_games(gen, "WHITE_CHECKMATES")
        for k in ("n_games", "outcome_match_rate", "outcome_distribution",
                  "mean_game_length", "forfeit_rate",
                  "post_terminal_padding_rate", "post_terminal_move_count",
                  "premature_padding_rate"):
            assert k in result

    @pytest.mark.unit
    def test_n_games_correct(self):
        gen = _make_gen_dict(5)
        result = _analyze_generated_games(gen, "WHITE_CHECKMATES")
        assert result["n_games"] == 5

    @pytest.mark.unit
    def test_all_matches(self):
        # All white-checkmate games (odd length)
        gen = _make_gen_dict(4, term_codes=np.zeros(4, dtype=np.int8),
                             game_lengths=np.array([5, 7, 9, 11], dtype=np.int32))
        result = _analyze_generated_games(gen, "WHITE_CHECKMATES")
        assert result["outcome_match_rate"] == 1.0

    @pytest.mark.unit
    def test_no_matches(self):
        # Condition on STALEMATE but all games are checkmates
        gen = _make_gen_dict(3, term_codes=np.zeros(3, dtype=np.int8),
                             game_lengths=np.array([5, 7, 9], dtype=np.int32))
        result = _analyze_generated_games(gen, "STALEMATE")
        assert result["outcome_match_rate"] == 0.0

    @pytest.mark.unit
    def test_mean_game_length(self):
        gen = _make_gen_dict(
            2, game_lengths=np.array([4, 8], dtype=np.int32),
        )
        result = _analyze_generated_games(gen, "WHITE_CHECKMATES")
        assert result["mean_game_length"] == pytest.approx(6.0)

    @pytest.mark.unit
    def test_forfeit_rate(self):
        forfeit = np.array([-1, 3, -1], dtype=np.int32)
        gen = _make_gen_dict(3, forfeit_ply=forfeit)
        result = _analyze_generated_games(gen, "WHITE_CHECKMATES")
        assert result["forfeit_rate"] == pytest.approx(1.0 / 3.0)

    @pytest.mark.unit
    def test_premature_padding_rate(self):
        tc = np.array([-2, 0, -2], dtype=np.int8)
        gen = _make_gen_dict(3, term_codes=tc,
                             game_lengths=np.array([4, 4, 4], dtype=np.int32))
        result = _analyze_generated_games(gen, "WHITE_CHECKMATES")
        assert result["premature_padding_rate"] == pytest.approx(2.0 / 3.0)

    @pytest.mark.unit
    def test_outcome_distribution_sums_to_one(self):
        gen = _make_gen_dict(3)
        result = _analyze_generated_games(gen, "WHITE_CHECKMATES")
        assert sum(result["outcome_distribution"].values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# autoregressive_generate (integration with toy model, tiny sequences)
# ---------------------------------------------------------------------------




class TestAutoregressiveGenerate:
    @pytest.mark.unit
    def test_returns_expected_keys_no_mask(self, toy_model, cpu_device):
        gen = autoregressive_generate(
            toy_model, WHITE_CHECKMATES, n_games=2, device=cpu_device,
            mask_illegal=False, max_seq_len=8, temperature=1.0, batch_size=2,
        )
        for k in ("sequences", "term_codes", "game_lengths", "forfeit_ply"):
            assert k in gen

    @pytest.mark.unit
    def test_output_shapes_no_mask(self, toy_model, cpu_device):
        gen = autoregressive_generate(
            toy_model, WHITE_CHECKMATES, n_games=3, device=cpu_device,
            mask_illegal=False, max_seq_len=8, temperature=1.0, batch_size=3,
        )
        assert gen["sequences"].shape == (3, 8)
        assert gen["term_codes"].shape == (3,)
        assert gen["game_lengths"].shape == (3,)
        assert gen["forfeit_ply"].shape == (3,)

    @pytest.mark.unit
    def test_sequences_start_with_outcome(self, toy_model, cpu_device):
        gen = autoregressive_generate(
            toy_model, WHITE_CHECKMATES, n_games=2, device=cpu_device,
            mask_illegal=False, max_seq_len=8, temperature=1.0, batch_size=2,
        )
        assert (gen["sequences"][:, 0] == WHITE_CHECKMATES).all()

    @pytest.mark.unit
    def test_mask_illegal_produces_legal_moves_only(self, toy_model, cpu_device):
        """With mask_illegal=True, no forfeits should happen."""
        torch.manual_seed(0)
        np.random.seed(0)
        gen = autoregressive_generate(
            toy_model, WHITE_CHECKMATES, n_games=2, device=cpu_device,
            mask_illegal=True, max_seq_len=8, temperature=1.0, batch_size=2,
        )
        # No forfeits with masking
        assert (gen["forfeit_ply"] == -1).all()

    @pytest.mark.unit
    def test_temperature_and_determinism(self, toy_model, cpu_device):
        """Two calls with same seed should produce identical output."""
        torch.manual_seed(7)
        np.random.seed(7)
        gen1 = autoregressive_generate(
            toy_model, WHITE_CHECKMATES, n_games=2, device=cpu_device,
            mask_illegal=False, max_seq_len=8, temperature=1.0, batch_size=2,
        )
        torch.manual_seed(7)
        np.random.seed(7)
        gen2 = autoregressive_generate(
            toy_model, WHITE_CHECKMATES, n_games=2, device=cpu_device,
            mask_illegal=False, max_seq_len=8, temperature=1.0, batch_size=2,
        )
        assert np.array_equal(gen1["sequences"], gen2["sequences"])

    @pytest.mark.unit
    def test_game_lengths_nonneg(self, toy_model, cpu_device):
        gen = autoregressive_generate(
            toy_model, WHITE_CHECKMATES, n_games=2, device=cpu_device,
            mask_illegal=False, max_seq_len=8, temperature=1.0, batch_size=2,
        )
        assert (gen["game_lengths"] >= 0).all()

    @pytest.mark.unit
    def test_prefix_moves_included(self, toy_model, cpu_device):
        """When prefix_moves is provided, they appear in sequences."""
        # Use the standard opening move 1. e4 via engine pairs to grab a real token id
        import chess_engine
        # Generate a random 2-ply game to obtain legal tokens
        move_ids, gl, _ = chess_engine.generate_random_games(1, 4, 42)
        prefix = move_ids[:, :2].astype(np.uint16)
        prefix = np.repeat(prefix, 2, axis=0)
        prefix_lengths = np.array([2, 2], dtype=np.int32)
        gen = autoregressive_generate(
            toy_model, WHITE_CHECKMATES, n_games=2, device=cpu_device,
            mask_illegal=False, max_seq_len=8, temperature=1.0, batch_size=2,
            prefix_moves=prefix, prefix_lengths=prefix_lengths,
        )
        # Prefix moves at positions 1, 2
        assert (gen["sequences"][:, 1] == prefix[:, 0]).all()
        assert (gen["sequences"][:, 2] == prefix[:, 1]).all()


# ---------------------------------------------------------------------------
# outcome_signal_test / prefix_continuation_test / poisoned_prefix_test /
# impossible_task_test / improbable_task_test
#
# These are heavy-weight integration calls that spawn many generate calls.
# We run them with minimal sizes to get smoke coverage without exploding time.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_corpus() -> dict:
    """Tiny in-memory corpus dict compatible with prefix tests."""
    import chess_engine
    move_ids, gl, tc = chess_engine.generate_random_games(128, 64, 42)
    return {
        "move_ids": move_ids,
        "game_lengths": gl,
        "termination_codes": tc,
    }


class TestPrefixContinuationTest:
    @pytest.mark.unit
    def test_skips_when_not_enough_games(self, toy_model, cpu_device, toy_corpus):
        # 128 games, need n_per_bucket per outcome; most outcomes will be skipped
        # (n_per_bucket=1 would trigger gen calls which would crash via BUG-701).
        # Use a large n_per_bucket so every outcome is skipped.
        results = prefix_continuation_test(
            toy_model, toy_corpus, cpu_device,
            n_per_bucket=1_000_000,  # guaranteed skip
            prefix_pcts=(0.5,),
            absolute_plies=(5,),
        )
        assert isinstance(results, dict)


class TestPoisonedPrefixTest:
    @pytest.mark.unit
    def test_skips_when_not_enough_games(self, toy_model, cpu_device, toy_corpus):
        results = poisoned_prefix_test(
            toy_model, toy_corpus, cpu_device,
            n_per_pair=1_000_000, prefix_pct=0.5,
        )
        assert isinstance(results, dict)

    @pytest.mark.unit
    def test_smoke_with_enough_games(self, toy_model, cpu_device, toy_corpus):
        results = poisoned_prefix_test(
            toy_model, toy_corpus, cpu_device,
            n_per_pair=1, prefix_pct=0.5,
        )
        assert isinstance(results, dict)


class TestImpossibleTaskTest:
    @pytest.mark.unit
    def test_smoke_handles_missing_scenarios(self, toy_model, cpu_device, toy_corpus):
        # With n_per_scenario huge, all scenarios are skipped
        results = impossible_task_test(
            toy_model, toy_corpus, cpu_device, n_per_scenario=1_000_000,
        )
        assert isinstance(results, dict)


class TestImprobableTaskTest:
    @pytest.mark.unit
    def test_smoke_handles_missing_scenarios(self, toy_model, cpu_device, toy_corpus):
        results = improbable_task_test(
            toy_model, toy_corpus, cpu_device, n_per_scenario=1_000_000,
        )
        assert isinstance(results, dict)
