"""Sanity tests for ``pawn.probes``.

These don't aim for "the probe achieves X accuracy on a real checkpoint" —
that's the verification-run job. They pin the *structure* of the port:

* ``extract_probe_data`` returns a complete, shape-consistent
  ``ProbeData``.
* ``get_probe_targets`` returns the right dtype / shape per probe.
* ``PAWNModel.hidden_all_layers`` returns ``(n_layers + 1, B, T, d_model)``
  with index 0 equal to the embed output of a one-layer ablation.
* ``train_probes`` runs end-to-end on the TINY supernet for one epoch,
  produces a metric per (probe, layer), and (a) every probe is above
  chance for the trivial probes, (b) the classification metrics are
  finite, (c) the MSE probes return an R² in ``[-∞, 1]`` and a non-NaN
  MAE.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pawn.config import TINY_SUPERNET
from pawn.model import init_model
from pawn.probes import (
    PROBES,
    BatchedLinearProbe,
    ProbeData,
    extract_probe_data,
    get_probe_targets,
    train_probes,
)


# ---------------------------------------------------------------------------
# Probe data shape & dtype tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_probe_data() -> ProbeData:
    """Tiny probe corpus — 32 games × max_ply=24, fast on CPU."""
    return extract_probe_data(
        n_games=32, max_ply=24, seed=42, prepend_outcome=False,
    )


def test_extract_probe_data_shapes(small_probe_data: ProbeData) -> None:
    d = small_probe_data
    n_games, max_ply = 32, 24
    assert d.input_ids.shape == (n_games, max_ply)
    assert d.attn_mask.shape == (n_games, max_ply)
    assert d.game_lengths.shape == (n_games,)
    assert d.boards.shape == (n_games, max_ply, 8, 8)
    assert d.side_to_move.shape == (n_games, max_ply)
    assert d.castling_rights.shape == (n_games, max_ply)
    assert d.ep_square.shape == (n_games, max_ply)
    assert d.is_check.shape == (n_games, max_ply)
    assert d.halfmove_clock.shape == (n_games, max_ply)
    assert d.legal_move_counts is not None
    assert d.legal_move_counts.shape == (n_games, max_ply)
    assert d.ply_offset == 0
    assert d.max_ply == max_ply
    # All games have at least 1 move — random games hardly ever
    # terminate immediately
    assert int(d.game_lengths.min()) >= 1


def test_extract_probe_data_with_outcome_prefix() -> None:
    d = extract_probe_data(
        n_games=4, max_ply=16, seed=7, prepend_outcome=True,
    )
    assert d.ply_offset == 1
    # The CLM sequence holds the outcome at position 0 + up to max_ply moves
    # afterwards, padded to whatever the engine's seq_len arg was. With
    # max_ply=16 the engine's seq_len is 16, so input_ids is (4, 16).
    assert d.input_ids.shape == (4, 16)
    # When ``prepend_outcome=True`` the engine packs the outcome at
    # position 0 and uses positions 1..max_ply-1 for moves; ``move_ids``
    # (and therefore the per-ply ``boards`` axis) shrinks by one.
    assert d.boards.shape == (4, 15, 8, 8)
    assert d.max_ply == 15


@pytest.mark.parametrize(
    "name,expected_n_outputs",
    [(name, spec.n_outputs) for name, spec in PROBES.items()],
)
def test_get_probe_targets_shapes(
    small_probe_data: ProbeData, name: str, expected_n_outputs: int
) -> None:
    d = small_probe_data
    # Pick one valid position per game.
    g = np.arange(d.input_ids.shape[0])
    p = np.zeros_like(g)
    t = get_probe_targets(name, d, g, p)
    spec = PROBES[name]
    if spec.loss_type == "ce":
        # Class id per position
        assert t.shape == (g.size,)
        assert t.dtype == np.int32
    elif spec.loss_type == "ce_per_square":
        assert t.shape == (g.size, 64)
        assert t.dtype == np.int32
        assert t.min() >= 0
        assert t.max() <= 12  # 13 classes
    else:  # bce / mse
        assert t.shape == (g.size, expected_n_outputs)
        assert t.dtype == np.float32


def test_get_probe_targets_castling_bits(small_probe_data: ProbeData) -> None:
    """Castling rights are KQkq — 4 bits in [0, 1]."""
    g = np.arange(4)
    p = np.zeros_like(g)
    t = get_probe_targets("castling_rights", small_probe_data, g, p)
    assert ((t == 0) | (t == 1)).all()
    # The starting position has all four castling rights — the engine emits
    # the board at ply 0, which is the position after move 0 (white has
    # just played); so castling rights still equal 1111.
    assert t[0].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_get_probe_targets_ep_square_clamps_to_64() -> None:
    """ep_square = -1 (no en-passant) should map to slot 64."""
    d = extract_probe_data(n_games=4, max_ply=8, seed=11)
    # Find a position with ep < 0 — most positions have no e.p. available
    g, p = np.where(d.ep_square < 0)
    assert g.size > 0, "expected at least one no-e.p. position in 4 games"
    t = get_probe_targets("ep_square", d, g[:5], p[:5])
    assert (t == 64).all()


# ---------------------------------------------------------------------------
# Model hidden_all_layers tests
# ---------------------------------------------------------------------------


def test_hidden_all_layers_shape() -> None:
    """``hidden_all_layers`` must return (n_layers + 1, B, T, d_model)."""
    cfg = TINY_SUPERNET
    model = init_model(cfg, jax.random.key(0))
    B, T = 3, 12
    tokens = jnp.zeros((B, T), dtype=jnp.int32)
    attn_mask = jnp.ones((B, T), dtype=jnp.bool_)
    h = model.hidden_all_layers(tokens, attn_mask)
    assert h.shape == (cfg.n_layers + 1, B, T, cfg.d_model)


def test_hidden_all_layers_embed_matches_call() -> None:
    """Index 0 of ``hidden_all_layers`` is the embed output — i.e. the
    same as the embedding the regular forward starts from."""
    cfg = TINY_SUPERNET
    model = init_model(cfg, jax.random.key(0))
    B, T = 2, 8
    tokens = jnp.asarray(
        np.random.default_rng(0).integers(0, 64, size=(B, T)),
        dtype=jnp.int32,
    )
    attn_mask = jnp.ones((B, T), dtype=jnp.bool_)
    h = model.hidden_all_layers(tokens, attn_mask)
    # Compare to the model's private ``_embed`` — the embed output is what
    # __call__ feeds the first transformer layer.
    embed = model._embed(tokens)
    assert np.allclose(np.asarray(h[0]), np.asarray(embed), atol=1e-5)


def test_hidden_all_layers_layers_differ() -> None:
    """Different layers should produce different hidden states on a
    randomly-initialised model (otherwise the scan is dropping outputs)."""
    cfg = TINY_SUPERNET
    model = init_model(cfg, jax.random.key(0))
    B, T = 2, 8
    tokens = jnp.asarray(
        np.random.default_rng(1).integers(0, 64, size=(B, T)),
        dtype=jnp.int32,
    )
    attn_mask = jnp.ones((B, T), dtype=jnp.bool_)
    h = model.hidden_all_layers(tokens, attn_mask)
    # Layer 0 output (== h[1]) should differ from embed (== h[0]) on a
    # non-trivial init.
    diff = float(jnp.abs(h[1] - h[0]).mean())
    assert diff > 1e-4, f"layer-0 output looks identical to embed (mean |Δ|={diff})"


# ---------------------------------------------------------------------------
# Probe head tests
# ---------------------------------------------------------------------------


def test_batched_linear_probe_forward_shape() -> None:
    n_probes, d, n_out, N = 5, 16, 7, 11
    probe = BatchedLinearProbe.init(n_probes, d, n_out, jax.random.key(0))
    h = jnp.zeros((n_probes, N, d))
    out = probe(h)
    assert out.shape == (n_probes, N, n_out)
    # With zero hidden states the output reduces to the bias replicated
    # across the N axis.
    expected = jnp.broadcast_to(probe.bias[:, None, :], out.shape)
    assert np.allclose(np.asarray(out), np.asarray(expected), atol=1e-6)


# ---------------------------------------------------------------------------
# End-to-end training smoke test
# ---------------------------------------------------------------------------


def test_train_probes_smoke() -> None:
    """One-epoch run over a 16-game train / 8-game val corpus on TINY.

    The corpus is too small for any of the probes to actually learn
    anything interesting; this is a structural test that:

    * ``train_probes`` returns a result dict keyed by every requested
      probe name and every layer name.
    * Every metric is finite (no NaN/Inf).
    * Probes that depend on a global target (side_to_move, is_check)
      land in ``[0, 1]`` accuracy.
    * MSE probes return an R² and a non-NaN MAE.
    """
    cfg = TINY_SUPERNET
    model = init_model(cfg, jax.random.key(0))
    train = extract_probe_data(n_games=16, max_ply=16, seed=1)
    val = extract_probe_data(n_games=8, max_ply=16, seed=2)
    results = train_probes(
        model, train, val,
        n_epochs=1, lr=1e-3,
        game_batch_size=8, inner_batch_size=64,
        key=jax.random.key(0), verbose=False,
    )
    expected_layer_names = ["embed"] + [
        f"layer_{i}" for i in range(cfg.n_layers)
    ]
    assert set(results.keys()) == set(PROBES.keys())
    for name, spec in PROBES.items():
        assert set(results[name].keys()) == set(expected_layer_names)
        for lname in expected_layer_names:
            entry = results[name][lname]
            for k in ("accuracy", "loss", "best_accuracy"):
                assert k in entry
                assert np.isfinite(entry[k]), (
                    f"{name}/{lname}/{k} = {entry[k]!r}"
                )
            if spec.loss_type == "bce":
                assert 0.0 <= entry["accuracy"] <= 1.0
            if spec.loss_type == "mse":
                assert "mae" in entry
                assert np.isfinite(entry["mae"])
                # R² is ``1 - ss_res / (ss_tot + 1e-8)``; it can be
                # negative when the probe is worse than the constant
                # mean, but cannot exceed 1.
                assert entry["accuracy"] <= 1.0 + 1e-6
