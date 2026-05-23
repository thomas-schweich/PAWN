"""Generation diagnostics — 5 outcome-conditional tests.

All five diagnostics condition on the outcome token at position 0, so
all five return a ``{"_skipped": ...}`` sentinel when
``outcome_prefix_trained=False`` (plan §10 S8, plan §11 "easy to drop
and shouldn't be").

The diagnostics:

- ``outcome_signal_test`` — does the model produce a different move
  distribution when given different outcomes at position 0?
- ``prefix_continuation_test`` — given an outcome + prefix of N moves,
  does the model continue plausibly?
- ``poisoned_prefix_test`` — given an outcome that contradicts the
  game's actual result, does the model still respect the outcome
  prefix?
- ``impossible_task_test`` — given an outcome that's geometrically
  impossible (mate before move 2), how does the model behave?
- ``improbable_task_test`` — given an outcome with implausibly low
  prior, does the model still emit consistent moves?

For the trainer-side smoke check the gate is the load-bearing
behavior; the actual diagnostic values are computed at S13 via
`scripts/eval_generation_jax.py`. This module ships the gate +
skeleton scoring loop.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from pawn.config import NUM_ACTIONS, OUTCOME_TOKEN_BASE, PAD_TOKEN
from pawn.model import PAWNModel

__all__ = [
    "DIAGNOSTIC_NAMES",
    "outcome_signal_test",
    "prefix_continuation_test",
    "poisoned_prefix_test",
    "impossible_task_test",
    "improbable_task_test",
    "run_all_diagnostics",
]


DIAGNOSTIC_NAMES = (
    "outcome_signal_test",
    "prefix_continuation_test",
    "poisoned_prefix_test",
    "impossible_task_test",
    "improbable_task_test",
)


_SKIP_REASON = (
    "model was not trained with prepend_outcome=True; this diagnostic "
    "conditions on the outcome token at position 0 and is uninterpretable "
    "without it"
)


def _skipped(name: str) -> dict[str, Any]:
    return {"_skipped": _SKIP_REASON, "diagnostic": name}


def _argmax_action(logits: Float[Array, "B T V"]) -> Int[Array, "B T"]:
    """Argmax restricted to ``[0, NUM_ACTIONS)`` so PAD / outcome tokens
    can't be sampled."""
    return jnp.argmax(logits[..., :NUM_ACTIONS], axis=-1)


def outcome_signal_test(
    model: PAWNModel,
    *,
    outcome_prefix_trained: bool,
    n_games: int = 64,
    seq_len: int = 16,
    key: int = 0,
) -> dict[str, Any]:
    """Does varying the outcome at position 0 change the move distribution?

    Constructs two batches of identical PAD-tail sequences, one with
    each of two different outcome tokens at slot 0. If the model is
    outcome-conditioned, the predicted next-move distributions should
    differ. The test reports the mean L1 distance between the two
    distributions at slot 1 (the first move position).
    """
    if not outcome_prefix_trained:
        return _skipped("outcome_signal_test")

    from pawn.config import (
        BLACK_CHECKMATES,
        WHITE_CHECKMATES,
    )

    tokens_w = jnp.full((n_games, seq_len), PAD_TOKEN, dtype=jnp.int32)
    tokens_w = tokens_w.at[:, 0].set(WHITE_CHECKMATES)
    tokens_b = tokens_w.at[:, 0].set(BLACK_CHECKMATES)
    attn = jnp.zeros((n_games, seq_len), dtype=jnp.bool_).at[:, 0].set(True)

    logits_w = model(tokens_w, attn)
    logits_b = model(tokens_b, attn)
    probs_w = jax.nn.softmax(logits_w[:, 1, :NUM_ACTIONS], axis=-1)
    probs_b = jax.nn.softmax(logits_b[:, 1, :NUM_ACTIONS], axis=-1)
    mean_l1 = float(jnp.abs(probs_w - probs_b).sum(axis=-1).mean())
    return {
        "diagnostic": "outcome_signal_test",
        "mean_l1_distance": mean_l1,
        "n_games": n_games,
    }


def prefix_continuation_test(
    model: PAWNModel,
    prefix: Int[Array, "P"],
    outcome_token: int,
    *,
    outcome_prefix_trained: bool,
    seq_len: int = 32,
) -> dict[str, Any]:
    """Given an outcome + the first P moves, does the model continue?

    Returns the argmax move at position P+1 (i.e. the next-move
    prediction conditioned on the outcome + prefix).
    """
    if not outcome_prefix_trained:
        return _skipped("prefix_continuation_test")
    p = int(prefix.shape[0])
    tokens = jnp.full((1, seq_len), PAD_TOKEN, dtype=jnp.int32)
    tokens = tokens.at[0, 0].set(outcome_token)
    tokens = tokens.at[0, 1 : 1 + p].set(prefix)
    attn = jnp.zeros((1, seq_len), dtype=jnp.bool_)
    attn = attn.at[0, : 1 + p].set(True)
    logits = model(tokens, attn)
    next_move = int(_argmax_action(logits)[0, p])
    return {
        "diagnostic": "prefix_continuation_test",
        "next_move_token": next_move,
        "prefix_length": p,
    }


def poisoned_prefix_test(
    model: PAWNModel,
    true_prefix: Int[Array, "P"],
    poisoned_outcome: int,
    *,
    outcome_prefix_trained: bool,
    seq_len: int = 32,
) -> dict[str, Any]:
    """Same shape as prefix_continuation_test but with an outcome that
    contradicts the actual game (e.g. white-checkmates prefix paired
    with black-checkmates outcome). Tests whether the model
    'capitulates' to the outcome or sticks to the prefix's reality."""
    if not outcome_prefix_trained:
        return _skipped("poisoned_prefix_test")
    return prefix_continuation_test(
        model,
        true_prefix,
        poisoned_outcome,
        outcome_prefix_trained=True,
        seq_len=seq_len,
    ) | {"diagnostic": "poisoned_prefix_test"}


def impossible_task_test(
    model: PAWNModel,
    *,
    outcome_prefix_trained: bool,
    seq_len: int = 16,
) -> dict[str, Any]:
    """Outcome at slot 0 = 'white checkmates', but no opening moves
    can deliver mate-in-1 from the initial position. How confidently
    does the model still emit moves?"""
    if not outcome_prefix_trained:
        return _skipped("impossible_task_test")
    from pawn.config import WHITE_CHECKMATES

    tokens = jnp.full((1, seq_len), PAD_TOKEN, dtype=jnp.int32)
    tokens = tokens.at[0, 0].set(WHITE_CHECKMATES)
    attn = jnp.zeros((1, seq_len), dtype=jnp.bool_).at[0, 0].set(True)
    logits = model(tokens, attn)
    probs = jax.nn.softmax(logits[0, 1, :NUM_ACTIONS], axis=-1)
    return {
        "diagnostic": "impossible_task_test",
        "top1_prob": float(probs.max()),
        "entropy": float(-(probs * jnp.log(probs + 1e-12)).sum()),
    }


def improbable_task_test(
    model: PAWNModel,
    *,
    outcome_prefix_trained: bool,
    seq_len: int = 16,
) -> dict[str, Any]:
    """Outcome at slot 0 = a low-prior outcome (DRAW_BY_AGREEMENT from
    initial position). Does the model emit a sensible distribution?"""
    if not outcome_prefix_trained:
        return _skipped("improbable_task_test")
    from pawn.config import DRAW_BY_AGREEMENT

    tokens = jnp.full((1, seq_len), PAD_TOKEN, dtype=jnp.int32)
    tokens = tokens.at[0, 0].set(DRAW_BY_AGREEMENT)
    attn = jnp.zeros((1, seq_len), dtype=jnp.bool_).at[0, 0].set(True)
    logits = model(tokens, attn)
    probs = jax.nn.softmax(logits[0, 1, :NUM_ACTIONS], axis=-1)
    return {
        "diagnostic": "improbable_task_test",
        "top1_prob": float(probs.max()),
        "entropy": float(-(probs * jnp.log(probs + 1e-12)).sum()),
    }


def run_all_diagnostics(
    model: PAWNModel, *, outcome_prefix_trained: bool
) -> dict[str, dict[str, Any]]:
    """Run all 5 diagnostics, return a dict keyed by name.

    Each entry is either a result dict or a `{"_skipped": ...}` sentinel
    when ``outcome_prefix_trained=False``."""
    from pawn.config import WHITE_CHECKMATES, BLACK_CHECKMATES

    prefix = jnp.array([5, 10], dtype=jnp.int32)
    return {
        "outcome_signal_test": outcome_signal_test(
            model, outcome_prefix_trained=outcome_prefix_trained
        ),
        "prefix_continuation_test": prefix_continuation_test(
            model, prefix, WHITE_CHECKMATES,
            outcome_prefix_trained=outcome_prefix_trained,
        ),
        "poisoned_prefix_test": poisoned_prefix_test(
            model, prefix, BLACK_CHECKMATES,
            outcome_prefix_trained=outcome_prefix_trained,
        ),
        "impossible_task_test": impossible_task_test(
            model, outcome_prefix_trained=outcome_prefix_trained
        ),
        "improbable_task_test": improbable_task_test(
            model, outcome_prefix_trained=outcome_prefix_trained
        ),
    }


# Late import (jax.nn) — `jax` is imported lazily inside this module so
# `_skipped` can return early without paying the JAX import cost.
import jax  # noqa: E402
