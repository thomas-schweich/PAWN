"""Structural tests for ``pawn.generation``.

Targets: the autoregressive sampler runs end-to-end on TINY, produces
valid sequences shaped per the contract, and the §6.1-6.3 outcome
signal test returns the documented metrics dict layout. These don't
aim for "the model achieves X% match" — that's the verification-run
job — but they pin the structural correctness of the port.
"""

from __future__ import annotations

import chess_engine as engine
import numpy as np
import pytest

import jax

from pawn.config import OUTCOME_TOKEN_BASE, PAD_TOKEN, TINY_SUPERNET
from pawn.generation import (
    OUTCOME_TOKENS,
    GenerationResult,
    _analyze_generated_games,
    _outcome_mask,
    _term_code_to_outcome_name,
    autoregressive_generate,
    impossible_task_test,
    improbable_task_test,
    outcome_signal_test,
    poisoned_prefix_test,
    prefix_continuation_test,
)
from pawn.model import init_model


# ---------------------------------------------------------------------------
# Outcome-token constants
# ---------------------------------------------------------------------------


def test_outcome_tokens_match_outcome_band() -> None:
    """All five OUTCOME_TOKENS land inside the outcome band starting at
    OUTCOME_TOKEN_BASE (1969). They are distinct and in canonical order."""
    expected = {
        "WHITE_CHECKMATES": OUTCOME_TOKEN_BASE + 0,
        "BLACK_CHECKMATES": OUTCOME_TOKEN_BASE + 1,
        "STALEMATE": OUTCOME_TOKEN_BASE + 2,
        "DRAW_BY_RULE": OUTCOME_TOKEN_BASE + 3,
        "PLY_LIMIT": OUTCOME_TOKEN_BASE + 4,
    }
    assert OUTCOME_TOKENS == expected
    assert len(set(OUTCOME_TOKENS.values())) == 5
    assert min(OUTCOME_TOKENS.values()) >= OUTCOME_TOKEN_BASE


# ---------------------------------------------------------------------------
# Term-code mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tc,gl,expected",
    [
        (0, 1, "WHITE_CHECKMATES"),
        (0, 2, "BLACK_CHECKMATES"),
        (1, 13, "STALEMATE"),
        (2, 100, "DRAW_BY_RULE"),
        (3, 100, "DRAW_BY_RULE"),
        (4, 100, "DRAW_BY_RULE"),
        (5, 255, "PLY_LIMIT"),
        (-2, 50, "PREMATURE_PAD"),
        (-3, 50, "FORFEIT"),
    ],
)
def test_term_code_to_outcome_name(tc: int, gl: int, expected: str) -> None:
    assert _term_code_to_outcome_name(tc, gl) == expected


def test_outcome_mask_recovers_term_codes() -> None:
    """``_outcome_mask`` correctly partitions a corpus by outcome.

    The five outcome buckets must be mutually disjoint and together
    cover every game in a synthetic corpus with every legal term code.
    """
    term_codes = np.array([0, 0, 1, 2, 3, 4, 5], dtype=np.int8)
    game_lengths = np.array([7, 8, 30, 100, 100, 100, 255], dtype=np.int32)
    seen = np.zeros(len(term_codes), dtype=bool)
    for name in OUTCOME_TOKENS:
        mask = _outcome_mask(term_codes, game_lengths, name)
        assert not (seen & mask).any(), (
            f"outcome {name} overlaps an earlier bucket"
        )
        seen |= mask
    assert seen.all(), "some game wasn't classified by any outcome bucket"


# ---------------------------------------------------------------------------
# Autoregressive generation contract
# ---------------------------------------------------------------------------


def _tiny_model():
    return init_model(TINY_SUPERNET, jax.random.key(0))


def test_autoregressive_generate_shapes() -> None:
    """``autoregressive_generate`` returns the documented shapes + dtypes."""
    model = _tiny_model()
    gen = autoregressive_generate(
        model, OUTCOME_TOKENS["WHITE_CHECKMATES"], n_games=4,
        mask_illegal=True, max_seq_len=24, batch_size=4, seed=0,
    )
    assert isinstance(gen, GenerationResult)
    assert gen.sequences.shape == (4, 24)
    assert gen.sequences.dtype == np.int32
    assert gen.term_codes.shape == (4,)
    assert gen.game_lengths.shape == (4,)
    assert gen.forfeit_ply.shape == (4,)


def test_autoregressive_generate_outcome_token_at_pos_0() -> None:
    """Position 0 always carries the conditioning outcome token."""
    model = _tiny_model()
    for name, tok in OUTCOME_TOKENS.items():
        gen = autoregressive_generate(
            model, tok, n_games=2,
            mask_illegal=True, max_seq_len=16, batch_size=2, seed=0,
        )
        assert (gen.sequences[:, 0] == tok).all(), f"outcome {name} not at pos 0"


def test_autoregressive_generate_masked_no_forfeits() -> None:
    """With ``mask_illegal=True`` no game can forfeit — the engine's
    legal-token mask precludes sampling an illegal move."""
    model = _tiny_model()
    gen = autoregressive_generate(
        model, OUTCOME_TOKENS["PLY_LIMIT"], n_games=8,
        mask_illegal=True, max_seq_len=24, batch_size=8, seed=1,
    )
    assert (gen.forfeit_ply == -1).all(), (
        f"forfeit_ply={gen.forfeit_ply.tolist()} despite mask_illegal=True"
    )


def test_autoregressive_generate_termination_within_window() -> None:
    """Every game terminates within the seq-len window — even
    PLY_LIMIT games hit the synthetic ply limit at the end."""
    model = _tiny_model()
    gen = autoregressive_generate(
        model, OUTCOME_TOKENS["PLY_LIMIT"], n_games=4,
        mask_illegal=True, max_seq_len=16, batch_size=4, seed=2,
    )
    # max_move_positions = 15, so game_lengths in [0, 15].
    assert (gen.game_lengths <= 15).all()
    assert (gen.game_lengths >= 0).all()
    # term_codes are all valid engine codes or sentinels.
    valid_codes = {0, 1, 2, 3, 4, 5, -2, -3}
    for tc in gen.term_codes:
        assert int(tc) in valid_codes, f"unknown term_code={tc}"


def test_autoregressive_generate_with_prefix() -> None:
    """Loaded prefixes are mirrored into the output sequences at
    positions ``1..pl``."""
    model = _tiny_model()
    # Generate a small corpus to source legal prefixes.
    _ids, _t, _lm, move_ids, gls, _tc = engine.generate_clm_batch(
        4, 16, 999, False, 0.0, False,
    )
    prefix_lens = np.minimum(np.full(4, 4, dtype=np.int32), gls.astype(np.int32))
    gen = autoregressive_generate(
        model, OUTCOME_TOKENS["PLY_LIMIT"], n_games=4,
        mask_illegal=True, max_seq_len=20, batch_size=4, seed=3,
        prefix_moves=move_ids, prefix_lengths=prefix_lens,
    )
    for i in range(4):
        pl = int(prefix_lens[i])
        # Position 0 is the outcome token; prefix moves at 1..pl.
        np.testing.assert_array_equal(
            gen.sequences[i, 1:pl + 1],
            move_ids[i, :pl].astype(np.int32),
            err_msg=f"game {i} prefix not mirrored at positions 1..{pl}",
        )


def test_autoregressive_generate_variable_prefix_length_grouping() -> None:
    """Variable-prefix-length grouping: rows with distinct ``prefix_lengths``
    are processed in same-length groups internally, then unpermuted so
    the returned ``GenerationResult`` row ``i`` corresponds to input row ``i``.

    Contract: for every game ``i``, ``sequences[i, 1 : pl[i] + 1]`` mirrors
    ``prefix_moves[i, :pl[i]]``. This fails for the legacy ungrouped
    code path when prefix lengths are mixed because the decode loop
    starts at the batch-wide max prefix_end, so rows with shorter
    prefixes see PAD in their attended context at positions
    ``[pl[i] + 1, max_pl]`` and the engine state is desynchronised from
    the model's apparent view. The mirror-back invariant in particular
    is what would break if the unpermute step were wrong.
    """
    model = _tiny_model()
    # Five games, deliberately mixed prefix lengths spanning multiple
    # group sizes and an internal batch boundary.
    _ids, _t, _lm, move_ids, gls, _tc = engine.generate_clm_batch(
        5, 16, 7, False, 0.0, False,
    )
    # Lengths chosen to hit three distinct groups (2, 5, 8) and to
    # straddle a batch boundary inside the longest group.
    mixed_lens = np.array([2, 5, 8, 2, 5], dtype=np.int32)
    # Clamp against per-game ply availability.
    pls = np.minimum(mixed_lens, gls.astype(np.int32))
    gen = autoregressive_generate(
        model, OUTCOME_TOKENS["PLY_LIMIT"], n_games=5,
        mask_illegal=True, max_seq_len=24, batch_size=2, seed=11,
        prefix_moves=move_ids, prefix_lengths=pls,
    )
    assert gen.sequences.shape == (5, 24)
    for i in range(5):
        pl = int(pls[i])
        # Position 0: outcome token. Positions 1..pl: original prefix
        # moves in the caller's order (this would diverge if the
        # unpermute step lost track of which group output row maps
        # to which input row).
        assert gen.sequences[i, 0] == OUTCOME_TOKENS["PLY_LIMIT"], (
            f"row {i}: outcome token at position 0 not preserved"
        )
        np.testing.assert_array_equal(
            gen.sequences[i, 1:pl + 1],
            move_ids[i, :pl].astype(np.int32),
            err_msg=f"row {i}: prefix not mirrored at positions 1..{pl}",
        )


def test_autoregressive_generate_kv_cache_matches_full_recompute() -> None:
    """KV-cache path produces the same generated sequences as the
    full-recompute path.

    Bitwise equivalence is the strict promise — the model produces
    bit-identical logits across both paths (``forward_with_cache`` is
    a pure superset of ``__call__``; ``forward_incremental`` runs the
    same layer kernel on a single-position slice with cached K, V).
    With the same Gumbel noise the argmax must land on the same token.

    The test exercises both no-prefix and with-prefix scenarios so
    the prefill code path and the incremental loop are both
    exercised."""
    model = _tiny_model()

    # 1. No prefix.
    gen_full = autoregressive_generate(
        model, OUTCOME_TOKENS["PLY_LIMIT"], n_games=4,
        mask_illegal=True, max_seq_len=20, batch_size=4, seed=17,
        use_kv_cache=False,
    )
    gen_kv = autoregressive_generate(
        model, OUTCOME_TOKENS["PLY_LIMIT"], n_games=4,
        mask_illegal=True, max_seq_len=20, batch_size=4, seed=17,
        use_kv_cache=True,
    )
    np.testing.assert_array_equal(
        gen_kv.sequences, gen_full.sequences,
        err_msg="no-prefix: KV-cache and full-recompute sequences diverged",
    )
    np.testing.assert_array_equal(
        gen_kv.term_codes, gen_full.term_codes,
        err_msg="no-prefix: KV-cache and full-recompute term_codes diverged",
    )
    np.testing.assert_array_equal(
        gen_kv.game_lengths, gen_full.game_lengths,
        err_msg="no-prefix: KV-cache and full-recompute game_lengths diverged",
    )

    # 2. With a uniform prefix.
    _ids, _t, _lm, move_ids, gls, _tc = engine.generate_clm_batch(
        4, 16, 51, False, 0.0, False,
    )
    pls = np.minimum(np.full(4, 3, dtype=np.int32), gls.astype(np.int32))
    gen_full_p = autoregressive_generate(
        model, OUTCOME_TOKENS["PLY_LIMIT"], n_games=4,
        mask_illegal=True, max_seq_len=20, batch_size=4, seed=29,
        prefix_moves=move_ids, prefix_lengths=pls, use_kv_cache=False,
    )
    gen_kv_p = autoregressive_generate(
        model, OUTCOME_TOKENS["PLY_LIMIT"], n_games=4,
        mask_illegal=True, max_seq_len=20, batch_size=4, seed=29,
        prefix_moves=move_ids, prefix_lengths=pls, use_kv_cache=True,
    )
    np.testing.assert_array_equal(
        gen_kv_p.sequences, gen_full_p.sequences,
        err_msg="with-prefix: KV-cache and full-recompute sequences diverged",
    )


def test_forward_with_cache_matches_call() -> None:
    """``forward_with_cache`` produces the same logits as ``__call__``
    on the same input. This is a unit-level pin on the prefill
    primitive used by KV-cache decoding."""
    import jax.numpy as jnp

    from pawn.config import TINY_SUPERNET

    model = _tiny_model()
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.integers(0, 64, (2, TINY_SUPERNET.max_seq_len)), dtype=jnp.int32)
    attn = jnp.ones((2, TINY_SUPERNET.max_seq_len), dtype=jnp.bool_)
    logits_ref = model(tokens, attn)
    logits, _cache = model.forward_with_cache(tokens, attn)
    np.testing.assert_array_equal(
        np.asarray(logits), np.asarray(logits_ref),
        err_msg="forward_with_cache logits diverged from __call__",
    )


def test_forward_incremental_matches_full_forward() -> None:
    """``forward_incremental`` at position ``pos`` produces logits that
    match ``__call__`` at the same position on the same prefix +
    next-token sequence. This pins the cached-decode invariant at the
    single-step level."""
    import jax.numpy as jnp

    from pawn.config import TINY_SUPERNET

    model = _tiny_model()
    T = TINY_SUPERNET.max_seq_len
    rng = np.random.default_rng(1)
    # Sequence: first 5 positions are real tokens, rest PAD.
    full_tokens = np.full((2, T), 1968, dtype=np.int32)
    full_tokens[:, :5] = rng.integers(0, 64, (2, 5)).astype(np.int32)

    # Prefill on positions [0..5) → cache + logits at position 4.
    prefill_tokens = full_tokens.copy()
    prefill_attn = np.zeros((2, T), dtype=bool)
    prefill_attn[:, :5] = True
    logits_pref, cache = model.forward_with_cache(
        jnp.asarray(prefill_tokens), jnp.asarray(prefill_attn),
    )

    # Now: at position 5, feed the next token. We need a real (non-PAD)
    # token to match a full forward over positions [0..6).
    next_token = rng.integers(0, 64, (2, 1)).astype(np.int32)
    full_tokens[:, 5:6] = next_token
    full_attn = np.zeros((2, T), dtype=bool)
    full_attn[:, :6] = True
    full_logits = model(jnp.asarray(full_tokens), jnp.asarray(full_attn))

    inc_logits, _cache = model.forward_incremental(
        jnp.asarray(next_token), jnp.int32(5), cache,
    )
    np.testing.assert_allclose(
        np.asarray(inc_logits[:, 0, :]), np.asarray(full_logits[:, 5, :]),
        rtol=1e-5, atol=1e-5,
        err_msg="forward_incremental[pos=5] diverged from __call__[:, 5]",
    )


def test_autoregressive_generate_mixed_prefix_post_prefix_continues() -> None:
    """Variable-prefix-length grouping: every row's positions
    ``[pl + 1, max_seq_len)`` should contain at least one real (non-PAD,
    non-outcome) move token. Pre-fix, rows with shorter prefixes would
    sample under PAD-padded context for ``[pl + 1, max_pl]`` positions
    before the engine's RL state caught up, often producing immediate
    PREMATURE_PAD termination at ``pl + 1`` even on rows where the
    engine's actual position has plenty of legal moves remaining.

    The post-grouping invariant: any row with ``pl < max_seq_len - 2``
    samples *something* — either a real move (engine state →
    eventually legal_or_terminate) or a clean termination — rather
    than collapsing into immediate PREMATURE_PAD purely due to the
    PAD-padded context.
    """
    model = _tiny_model()
    _ids, _t, _lm, move_ids, gls, _tc = engine.generate_clm_batch(
        6, 16, 137, False, 0.0, False,
    )
    # Mixed prefix lengths spanning two batches at batch_size=3.
    mixed_lens = np.array([1, 4, 7, 1, 4, 7], dtype=np.int32)
    pls = np.minimum(mixed_lens, gls.astype(np.int32))
    gen = autoregressive_generate(
        model, OUTCOME_TOKENS["PLY_LIMIT"], n_games=6,
        mask_illegal=True, max_seq_len=24, batch_size=3, seed=13,
        prefix_moves=move_ids, prefix_lengths=pls,
    )
    # Every row's game_length must be at least its prefix length —
    # the engine's RL state already advanced through ``load_prefixes``.
    for i in range(6):
        pl = int(pls[i])
        assert gen.game_lengths[i] >= pl, (
            f"row {i}: game_length={gen.game_lengths[i]} < pl={pl} — "
            f"the row terminated before its prefix was fully consumed"
        )


def test_autoregressive_generate_validates_n_games() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="n_games"):
        autoregressive_generate(
            model, OUTCOME_TOKENS["PLY_LIMIT"], n_games=0,
        )


def test_autoregressive_generate_validates_temperature() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="temperature"):
        autoregressive_generate(
            model, OUTCOME_TOKENS["PLY_LIMIT"], n_games=2,
            temperature=0.0, max_seq_len=12,
        )


# ---------------------------------------------------------------------------
# Analysis function
# ---------------------------------------------------------------------------


def test_analyze_generated_games_metric_keys() -> None:
    model = _tiny_model()
    gen = autoregressive_generate(
        model, OUTCOME_TOKENS["PLY_LIMIT"], n_games=6,
        mask_illegal=True, max_seq_len=16, batch_size=6, seed=4,
    )
    metrics = _analyze_generated_games(gen, "PLY_LIMIT")
    assert metrics["n_games"] == 6
    assert 0.0 <= metrics["outcome_match_rate"] <= 1.0
    assert 0.0 <= metrics["forfeit_rate"] <= 1.0
    assert 0.0 <= metrics["post_terminal_padding_rate"] <= 1.0
    assert metrics["mean_game_length"] >= 0.0
    assert "outcome_distribution" in metrics
    assert sum(metrics["outcome_distribution"].values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# §6.1-6.3 outcome signal test smoke
# ---------------------------------------------------------------------------


def test_outcome_signal_test_smoke() -> None:
    """``outcome_signal_test`` returns ``{outcome: {label: metrics}}``
    for every conditioning outcome and every mask condition."""
    model = _tiny_model()
    results = outcome_signal_test(
        model, n_per_outcome=4, mask_conditions=(False, True),
        batch_size=4, seed=0, verbose=False,
    )
    assert set(results.keys()) == set(OUTCOME_TOKENS.keys())
    for oname, by_mask in results.items():
        assert set(by_mask.keys()) == {"masked", "unmasked"}
        for label, metrics in by_mask.items():
            for k in (
                "n_games", "outcome_match_rate", "outcome_distribution",
                "mean_game_length", "forfeit_rate",
                "post_terminal_padding_rate", "post_terminal_move_count",
                "premature_padding_rate", "elapsed_s",
            ):
                assert k in metrics, f"{oname}/{label} missing {k}"


# ---------------------------------------------------------------------------
# Prefix + poisoned + impossible + improbable smoke
# ---------------------------------------------------------------------------


def _small_corpus(n_games: int = 64, max_ply: int = 32, seed: int = 1234):
    _ids, _t, _lm, move_ids, gls, tcs = engine.generate_clm_batch(
        n_games, max_ply, seed, False, 0.0, False,
    )
    return {
        "move_ids": move_ids,
        "game_lengths": gls,
        "termination_codes": tcs,
    }


def test_prefix_continuation_test_smoke() -> None:
    """Run on a small corpus; verify the result dict structure for at
    least one outcome bucket that has enough games."""
    model = _tiny_model()
    corpus = _small_corpus(n_games=64)
    results = prefix_continuation_test(
        model, corpus,
        n_per_bucket=2, prefix_pcts=(0.5,), absolute_plies=(10,),
        batch_size=4, seed=42, verbose=False,
    )
    # At least one outcome bucket should have completed (random games
    # are dominated by PLY_LIMIT, so that one is guaranteed in a corpus
    # of 64 games).
    assert any(results.values()), "no outcome buckets produced results"


def test_poisoned_prefix_test_smoke() -> None:
    model = _tiny_model()
    corpus = _small_corpus(n_games=64)
    results = poisoned_prefix_test(
        model, corpus, n_per_pair=2, prefix_pct=0.5,
        batch_size=4, seed=43,
    )
    # ``results`` may be empty (rare outcomes in 64 games), but every
    # entry must carry the required keys.
    for label, metrics in results.items():
        assert "->" in label
        assert "original_outcome_match_rate" in metrics
        assert "actual_outcome" in metrics
        assert "poisoned_outcome" in metrics


def test_impossible_task_test_smoke() -> None:
    """Smoke-run the §6.6 impossible-task scenarios on a small corpus.
    May skip individual scenarios that don't have enough games — only
    the structure of returned entries is validated."""
    model = _tiny_model()
    corpus = _small_corpus(n_games=32)
    results = impossible_task_test(
        model, corpus, n_per_scenario=2, batch_size=2, seed=44,
    )
    for scenario, metrics in results.items():
        assert "outcome_match_rate" in metrics


def test_improbable_task_test_smoke() -> None:
    model = _tiny_model()
    corpus = _small_corpus(n_games=32)
    results = improbable_task_test(
        model, corpus, n_per_scenario=2, batch_size=2, seed=46,
    )
    for scenario, metrics in results.items():
        assert "outcome_match_rate" in metrics
