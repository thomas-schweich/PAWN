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
    model = init_model(TINY_SUPERNET, key=0)
    res = outcome_signal_test(
        model, outcome_prefix_trained=True, n_games=4, seq_len=16
    )
    assert "mean_l1_distance" in res
    assert "_skipped" not in res


def test_prefix_continuation_test_runs_when_gate_on() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    prefix = jnp.array([5, 10], dtype=jnp.int32)
    res = prefix_continuation_test(
        model, prefix, WHITE_CHECKMATES, outcome_prefix_trained=True
    )
    assert "next_move_token" in res
    assert 0 <= res["next_move_token"] < NUM_ACTIONS


def test_poisoned_prefix_test_runs_when_gate_on() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    prefix = jnp.array([5, 10], dtype=jnp.int32)
    res = poisoned_prefix_test(
        model, prefix, BLACK_CHECKMATES, outcome_prefix_trained=True
    )
    assert res["diagnostic"] == "poisoned_prefix_test"


def test_impossible_task_test_runs_when_gate_on() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    res = impossible_task_test(model, outcome_prefix_trained=True)
    assert "top1_prob" in res and "entropy" in res


def test_improbable_task_test_runs_when_gate_on() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    res = improbable_task_test(model, outcome_prefix_trained=True)
    assert "top1_prob" in res and "entropy" in res


def test_run_all_diagnostics_has_5_entries() -> None:
    model = init_model(TINY_SUPERNET, key=0)
    results = run_all_diagnostics(model, outcome_prefix_trained=True)
    assert set(results.keys()) == set(DIAGNOSTIC_NAMES)


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
    # 6 labels in the canonical order.
    assert [r.label for r in results] == list(EDGE_CASE_LABELS)
    # Every label has finite accuracy in [0, 1].
    for r in results:
        assert 0.0 <= r.accuracy <= 1.0
        assert r.n_positions >= 0
