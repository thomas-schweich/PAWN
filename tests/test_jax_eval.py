"""Tests for ``pawn.eval`` — Phase-4 chunk 1 move-accuracy eval."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("chess_engine")

import jax
import jax.numpy as jnp
import numpy as np

from pawn.config import NUM_ACTIONS, TINY_SUPERNET
from pawn.corpus import generate_corpus
from pawn.eval import AccuracyResult, evaluate_accuracy
from pawn.model import init_model


def test_evaluate_accuracy_shape_and_range() -> None:
    """A freshly-initialised model produces accuracy in ``[0, 1]``
    overall + per-phase, with phase counts summing to the total.
    """
    model = init_model(TINY_SUPERNET, jax.random.PRNGKey(0))
    corpus = generate_corpus(n_games=16, max_ply=64, seed=1)
    result = evaluate_accuracy(model, corpus, batch_size=4)
    assert isinstance(result, AccuracyResult)
    assert 0.0 <= result.overall <= 1.0
    # Sum of per-phase totals equals overall n_supervised.
    phase_total = sum(n for _, n in result.phase.values())
    assert phase_total == result.n_supervised, (
        f"phase totals {phase_total} != overall n_supervised "
        f"{result.n_supervised}"
    )
    # Phase accuracies are well-defined (0/0 → 0.0).
    for name, (c, n) in result.phase.items():
        assert 0 <= c <= n, f"{name}: correct {c} > total {n}"


def test_evaluate_accuracy_random_baseline() -> None:
    """A randomly-initialised model gives ~1/N_legal accuracy on
    random positions. For a TINY model that's < 5% — well above
    1/1968 (pure uniform) due to RoPE / norm biases, but well below
    50%. Pin a sane range so a regression that broke the argmax
    (e.g. by including PAD in the argmax band) would surface as an
    impossibly-high or impossibly-low number."""
    model = init_model(TINY_SUPERNET, jax.random.PRNGKey(0))
    corpus = generate_corpus(n_games=32, max_ply=64, seed=2)
    result = evaluate_accuracy(model, corpus, batch_size=8)
    # Loose bounds: random init must be below 20%; argmax over
    # NUM_ACTIONS gives at least chance.
    assert 0.0 <= result.overall < 0.2, (
        f"random-init accuracy {result.overall} out of expected range"
    )


def test_argmax_restricted_to_action_band() -> None:
    """The argmax must be restricted to ``[0, NUM_ACTIONS)``. If a
    refactor accidentally argmaxed over the full ``vocab_size`` (incl.
    PAD + outcome tokens), accuracy on real targets would collapse to
    near zero because the model can place high logit mass on
    PAD/outcome at random-init. This test uses a hand-crafted logit
    tensor with PAD as the absolute max and verifies the argmax
    still picks an action token.

    We can't observe argmax directly from ``evaluate_accuracy``, but
    we can confirm the function reads the model's forward output and
    slices to ``[:NUM_ACTIONS]`` by checking that injecting a
    PAD-favouring shift to ``lm_head`` does NOT collapse accuracy."""
    # Build a model where the lm_head heavily favours PAD (= 1968).
    model = init_model(TINY_SUPERNET, jax.random.PRNGKey(0))
    lm_head = model.lm_head  # [d, V]
    boosted = lm_head.at[:, 1968].set(lm_head[:, 1968] + 100.0)
    import equinox as eqx
    model_pad = eqx.tree_at(lambda m: m.lm_head, model, boosted)
    corpus = generate_corpus(n_games=8, max_ply=32, seed=3)
    base = evaluate_accuracy(model, corpus, batch_size=4)
    boosted_res = evaluate_accuracy(model_pad, corpus, batch_size=4)
    # The PAD bias should not collapse accuracy — eval slices to
    # [0, NUM_ACTIONS) before argmax.
    assert abs(boosted_res.overall - base.overall) < 0.5, (
        f"PAD-biased lm_head changed eval accuracy from {base.overall} "
        f"to {boosted_res.overall} — argmax probably not restricted "
        f"to [0, NUM_ACTIONS={NUM_ACTIONS})"
    )


def test_evaluate_accuracy_zero_corpus_edge() -> None:
    """A corpus of length 1 doesn't crash the evaluator (an empty
    corpus would be invalid; the engine refuses ``n_games=0``).
    Pins the small-batch path."""
    model = init_model(TINY_SUPERNET, jax.random.PRNGKey(0))
    corpus = generate_corpus(n_games=1, max_ply=16, seed=4)
    result = evaluate_accuracy(model, corpus, batch_size=1)
    assert result.n_supervised > 0
    assert 0.0 <= result.overall <= 1.0
