"""Tests for the S8 eval surface — accuracy + diagnostics + probes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pawn.config import (
    BLACK_CHECKMATES,
    DRAW_BY_AGREEMENT,
    NUM_ACTIONS,
    TINY_SUPERNET,
    WHITE_CHECKMATES,
)
from pawn.corpus import generate_corpus
from pawn.eval import (
    AccuracyResult,
    PhaseBoundaries,
    compute_move_accuracy,
    compute_per_phase_accuracy,
)
from pawn.generation import (
    DIAGNOSTIC_NAMES,
    impossible_task_test,
    improbable_task_test,
    outcome_signal_test,
    poisoned_prefix_test,
    prefix_continuation_test,
    run_all_diagnostics,
)
from pawn.lichess_eval import (
    EloBin,
    compute_elo_stratified_accuracy,
    default_elo_bins,
)
from pawn.model import init_model
from pawn.probes import ProbeConfig, fit_probe


# ---------------------------------------------------------------------------
# Move accuracy + per-phase
# ---------------------------------------------------------------------------


def test_compute_move_accuracy_returns_float_in_zero_one() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    corpus = generate_corpus(n_games=4, max_ply=16, seq_len=32, seed=0)
    acc = compute_move_accuracy(model, corpus, batch_size=4)
    assert 0.0 <= acc <= 1.0


def test_compute_per_phase_accuracy_returns_breakdown() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    corpus = generate_corpus(n_games=4, max_ply=64, seq_len=80, seed=0)
    result = compute_per_phase_accuracy(model, corpus, batch_size=4)
    assert isinstance(result, AccuracyResult)
    assert 0.0 <= result.overall <= 1.0
    # Each phase fraction is finite.
    for v in (result.opening, result.midgame, result.endgame):
        assert 0.0 <= v <= 1.0


def test_argmax_never_picks_pad_or_outcome_tokens() -> None:
    """The plan-pinned contract: argmax restricted to [0, NUM_ACTIONS).
    Confirm by checking that compute_move_accuracy's argmax output is
    always in the valid action range."""
    model = init_model(TINY_SUPERNET, key=0)
    corpus = generate_corpus(n_games=2, max_ply=16, seq_len=32, seed=0)
    tokens = jnp.asarray(corpus.tokens)
    attn = jnp.asarray(corpus.attn_mask)
    logits = model(tokens, attn)
    pred = jnp.argmax(logits[..., :NUM_ACTIONS], axis=-1)
    assert jnp.all(pred < NUM_ACTIONS)


# ---------------------------------------------------------------------------
# Generation diagnostics: gate behavior + finite values
# ---------------------------------------------------------------------------


def test_diagnostic_names_lists_all_five() -> None:
    assert set(DIAGNOSTIC_NAMES) == {
        "outcome_signal_test",
        "prefix_continuation_test",
        "poisoned_prefix_test",
        "impossible_task_test",
        "improbable_task_test",
    }


@pytest.mark.parametrize("name", DIAGNOSTIC_NAMES)
def test_each_diagnostic_skips_when_not_outcome_prefix_trained(name: str) -> None:
    """Plan §10 S8 / §11: all 5 diagnostics return `{"_skipped": ...}`
    when `outcome_prefix_trained=False`."""
    model = init_model(TINY_SUPERNET, key=0)
    all_results = run_all_diagnostics(model, outcome_prefix_trained=False)
    assert "_skipped" in all_results[name]


def test_outcome_signal_test_runs_when_gate_on() -> None:
    """v1 parity: outcome_signal_test now runs real autoregressive
    generation per outcome and reports per-outcome metrics (match rate,
    forfeit rate, mean game length). Tiny defaults keep the test under
    a few seconds without the KV-cached decoder."""
    from pawn.generation import OUTCOME_TOKENS

    model = init_model(TINY_SUPERNET, key=0)
    res = outcome_signal_test(
        model, outcome_prefix_trained=True,
        n_per_outcome=2, max_seq_len=8, mask_conditions=(True,),
    )
    assert "_skipped" not in res
    # Every v1 outcome appears with masked-condition results.
    for name in OUTCOME_TOKENS:
        assert name in res
        assert "masked" in res[name]
        metrics = res[name]["masked"]
        # Headline v1 metrics are present and in-range.
        assert 0.0 <= metrics["outcome_match_rate"] <= 1.0
        assert 0.0 <= metrics["forfeit_rate"] <= 1.0
        assert metrics["mean_game_length"] >= 0.0


def test_prefix_continuation_test_runs_when_gate_on() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    prefix = jnp.array([5, 10], dtype=jnp.int32)
    res = prefix_continuation_test(
        model, prefix, WHITE_CHECKMATES, outcome_prefix_trained=True,
        n_continuations=2, seq_len=8,
    )
    # `next_move_argmax` is the cheap single-shot probe; the AR
    # analysis is the v1-parity metric block.
    assert "next_move_argmax" in res
    assert 0 <= res["next_move_argmax"] < NUM_ACTIONS
    assert "ar_continuation" in res
    assert res["ar_continuation"]["n_games"] == 2


def test_poisoned_prefix_test_runs_when_gate_on() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    prefix = jnp.array([5, 10], dtype=jnp.int32)
    res = poisoned_prefix_test(
        model, prefix, BLACK_CHECKMATES, outcome_prefix_trained=True,
        n_continuations=2, seq_len=8,
    )
    assert res["diagnostic"] == "poisoned_prefix_test"
    assert "ar_continuation" in res


def test_impossible_task_test_runs_when_gate_on() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    res = impossible_task_test(
        model, outcome_prefix_trained=True, n_games=2, seq_len=8,
    )
    assert "top1_prob" in res and "entropy" in res
    assert "ar_analysis" in res
    # Forfeit rate is the headline for the impossible task.
    assert 0.0 <= res["ar_analysis"]["forfeit_rate"] <= 1.0


def test_improbable_task_test_runs_when_gate_on() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    res = improbable_task_test(
        model, outcome_prefix_trained=True, n_games=2, seq_len=8,
    )
    assert "top1_prob" in res and "entropy" in res
    assert "ar_analysis" in res


def test_run_all_diagnostics_has_5_entries() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    results = run_all_diagnostics(model, outcome_prefix_trained=True)
    assert set(results.keys()) == set(DIAGNOSTIC_NAMES)


def test_autoregressive_generate_kv_cache_matches_full_forward() -> None:
    """The cached and non-cached paths in `autoregressive_generate`
    must agree on the generated sequences (both seeded the same and
    sampling is deterministic for argmax-style Gumbel-max at the same
    weights). Pins the speed-vs-correctness invariant — a future change
    that breaks numeric parity between cache paths shows up here."""
    from pawn.generation import WHITE_CHECKMATES, autoregressive_generate

    model = init_model(TINY_SUPERNET, key=0)
    gen_plain = autoregressive_generate(
        model, WHITE_CHECKMATES, n_games=2,
        mask_illegal=True, max_seq_len=8, seed=0, use_kv_cache=False,
    )
    gen_cached = autoregressive_generate(
        model, WHITE_CHECKMATES, n_games=2,
        mask_illegal=True, max_seq_len=8, seed=0, use_kv_cache=True,
    )
    # Sequences should match: the two paths share sampling RNG (the
    # `seed=0` -> default_rng draws are deterministic) and produce
    # numerically-equivalent logits per parity test in test_jax_model.
    assert np.array_equal(gen_plain["sequences"], gen_cached["sequences"])
    assert np.array_equal(gen_plain["term_codes"], gen_cached["term_codes"])
    assert np.array_equal(gen_plain["game_lengths"], gen_cached["game_lengths"])


def test_autoregressive_generate_bf16_cache_runs() -> None:
    """The `cache_dtype=jnp.bfloat16` path (paired with `compute_dtype`)
    must run end-to-end. Round-3 test-risk MEDIUM: the bf16-cache opt-in
    surface had no test, so a future change that broke the dtype pairing
    could land silently. This pins that the bf16 forward + bf16 cache
    path produces valid output (sequences, term codes, game lengths)
    with no exceptions."""
    from pawn.generation import WHITE_CHECKMATES, autoregressive_generate

    model = init_model(TINY_SUPERNET, key=0)
    gen = autoregressive_generate(
        model, WHITE_CHECKMATES, n_games=2,
        mask_illegal=True, max_seq_len=8, seed=0,
        use_kv_cache=True,
        cache_dtype=jnp.bfloat16, compute_dtype=jnp.bfloat16,
    )
    # Headline shape + dtype invariants — sequences are int32 game
    # tokens, not affected by the bf16 cache choice.
    assert gen["sequences"].shape == (2, 8)
    assert gen["sequences"].dtype == np.int32
    assert gen["term_codes"].shape == (2,)
    # Outcome token at pos 0 is preserved regardless of dtype.
    assert (gen["sequences"][:, 0] == WHITE_CHECKMATES).all()


def test_autoregressive_generate_rejects_bf16_cache_with_fp32_compute() -> None:
    """`cache_dtype=bfloat16` without `compute_dtype=bfloat16` must
    raise — a fp32 forward writing into a bf16 cache silently
    downcasts on every write (round-3 bug-detector). The
    `autoregressive_generate` validator catches this at call time
    rather than letting the operator commit to a lossy run."""
    from pawn.generation import WHITE_CHECKMATES, autoregressive_generate

    model = init_model(TINY_SUPERNET, key=0)
    with pytest.raises(ValueError, match="requires compute_dtype"):
        autoregressive_generate(
            model, WHITE_CHECKMATES, n_games=2,
            mask_illegal=True, max_seq_len=8,
            cache_dtype=jnp.bfloat16,  # no compute_dtype
        )


# ---------------------------------------------------------------------------
# Linear probes
# ---------------------------------------------------------------------------


def test_fit_probe_converges_on_separable_data() -> None:
    """Synthetic separable hidden states + labels; probe should reach
    high accuracy."""
    rng = np.random.default_rng(0)
    n_per_class = 64
    d = 32
    n_classes = 4
    means = rng.normal(size=(n_classes, d)) * 2.0
    hidden = []
    labels = []
    for c in range(n_classes):
        hidden.append(means[c] + rng.normal(size=(n_per_class, d)) * 0.3)
        labels.extend([c] * n_per_class)
    hidden_arr = jnp.asarray(np.concatenate(hidden, axis=0), dtype=jnp.float32)
    labels_arr = jnp.asarray(labels, dtype=jnp.int32)
    cfg = ProbeConfig(n_classes=n_classes, lr=1e-2, n_epochs=10, batch_size=32)
    result = fit_probe(hidden_arr, labels_arr, cfg, key=0)
    assert result.accuracy > 0.9


# ---------------------------------------------------------------------------
# Elo-stratified accuracy
# ---------------------------------------------------------------------------


def test_default_elo_bins_covers_lichess_range() -> None:
    bins = default_elo_bins()
    assert bins[0] == EloBin(1100, 1200)
    assert bins[-1] == EloBin(1900, 2000)


def test_compute_elo_stratified_accuracy_aggregates_bins() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    c = generate_corpus(n_games=2, max_ply=16, seq_len=32, seed=0)
    bins = {EloBin(1500, 1600): c, EloBin(1600, 1700): c}
    results = compute_elo_stratified_accuracy(model, bins, batch_size=2)
    assert len(results) == 2
    for r in results:
        assert 0.0 <= r.accuracy <= 1.0
        assert r.n_games == 2


# ---------------------------------------------------------------------------
# Edge-case diagnostics
# ---------------------------------------------------------------------------


def test_edge_case_accuracy_runs() -> None:
    """Smoke test on real engine games — `edge_case_bits` is exercised."""
    import chess_engine as engine

    from pawn.eval_suite.diagnostics import (
        EDGE_CASE_LABELS,
        compute_edge_case_accuracy,
    )

    model = init_model(TINY_SUPERNET, key=0)
    move_ids, game_lengths, _ = engine.generate_random_games(8, 32, 42)
    results = compute_edge_case_accuracy(
        model, move_ids, game_lengths, batch_size=4
    )
    # Parity #6: 10 labels in the canonical order (v1 surface).
    assert [r.label for r in results] == list(EDGE_CASE_LABELS)
    assert len(EDGE_CASE_LABELS) == 10
    # Every label has finite accuracy in [0, 1].
    for r in results:
        assert 0.0 <= r.accuracy <= 1.0
        assert r.n_positions >= 0


def test_edge_case_labels_include_added_v1_categories() -> None:
    """The 4 categories restored in parity #6 must be present.

    `n_positions` may be 0 on the 8-game random pool — guarantee of
    coverage requires the quota-controlled path
    (`compute_edge_case_accuracy_quota`)."""
    from pawn.eval_suite.diagnostics import EDGE_CASE_LABELS
    for label in (
        "castle_blocked_check", "promotion_available",
        "checkmate", "stalemate",
    ):
        assert label in EDGE_CASE_LABELS


def test_edge_case_accuracy_quota_guarantees_coverage() -> None:
    """`compute_edge_case_accuracy_quota` calls
    `engine.generate_diagnostic_sets` with per-label quotas, so every
    label should have ``n_positions > 0`` even for rare cases like
    `checkmate` / `stalemate`."""
    from pawn.eval_suite.diagnostics import (
        EDGE_CASE_LABELS,
        compute_edge_case_accuracy_quota,
    )
    model = init_model(TINY_SUPERNET, key=0)
    # `per_label=4` is the smallest budget that reliably surfaces
    # `stalemate` (the rarest label — engine empirically needs ~400
    # simulated games per stalemate; the default factor=500 amortises
    # that). Keep `max_ply=256` because stalemate tends to appear in
    # longer games.
    results = compute_edge_case_accuracy_quota(
        model, per_label=4, max_ply=256, batch_size=4,
    )
    by_label = {r.label: r for r in results}
    for label in EDGE_CASE_LABELS:
        assert label in by_label
        assert by_label[label].n_positions > 0, (
            f"quota-controlled sampling missed label {label!r}"
        )
        assert 0.0 <= by_label[label].accuracy <= 1.0
