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

import chess_engine as engine
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
    # The engine convention (engine/src/extract.rs — "The state at ply i
    # is the board BEFORE move_ids[i] is played") puts the starting
    # position at ply 0, so all four KQkq rights are set.
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


def test_train_probes_best_accuracy_invariant() -> None:
    """``best_accuracy`` tracks the per-epoch maximum across training, so
    by construction it must be ``>= final_accuracy`` at every (probe,
    layer) — Test-risk flagged the absence of this structural check."""
    cfg = TINY_SUPERNET
    model = init_model(cfg, jax.random.key(0))
    train = extract_probe_data(n_games=24, max_ply=16, seed=1)
    val = extract_probe_data(n_games=12, max_ply=16, seed=2)
    results = train_probes(
        model, train, val,
        n_epochs=3, lr=1e-3,
        game_batch_size=8, inner_batch_size=64,
        key=jax.random.key(0), verbose=False,
    )
    for name, layers in results.items():
        for lname, entry in layers.items():
            assert entry["best_accuracy"] >= entry["accuracy"] - 1e-6, (
                f"{name}/{lname}: best={entry['best_accuracy']} < "
                f"final={entry['accuracy']}"
            )


def test_train_probes_deterministic() -> None:
    """Same seed → same metrics within fp tolerance. Catches accidental
    silent non-determinism (donated buffers, mutated PRNG streams) that
    a structural test alone misses. (Test-risk)"""
    cfg = TINY_SUPERNET
    model = init_model(cfg, jax.random.key(0))
    train = extract_probe_data(n_games=16, max_ply=16, seed=3)
    val = extract_probe_data(n_games=8, max_ply=16, seed=4)
    def _run() -> dict:
        return train_probes(
            model, train, val,
            n_epochs=2, lr=1e-3, game_batch_size=8, inner_batch_size=64,
            key=jax.random.key(7), verbose=False,
        )
    r1 = _run()
    r2 = _run()
    for name in PROBES:
        for lname in r1[name]:
            assert r1[name][lname]["loss"] == pytest.approx(
                r2[name][lname]["loss"], abs=1e-5
            ), f"{name}/{lname} loss is non-deterministic"
            assert r1[name][lname]["accuracy"] == pytest.approx(
                r2[name][lname]["accuracy"], abs=1e-5
            ), f"{name}/{lname} accuracy is non-deterministic"


def test_train_probes_learns_side_to_move() -> None:
    """``side_to_move`` is a trivially-learnable signal (binary toggle).
    Even on a tiny corpus the probe should land well above 50% chance —
    catches a sign-flip or axis-transposition in the gather/loss that
    the shape tests would miss. (Test-risk)"""
    cfg = TINY_SUPERNET
    model = init_model(cfg, jax.random.key(0))
    train = extract_probe_data(n_games=64, max_ply=16, seed=5)
    val = extract_probe_data(n_games=32, max_ply=16, seed=6)
    results = train_probes(
        model, train, val,
        n_epochs=8, lr=3e-3,
        game_batch_size=16, inner_batch_size=128,
        probe_names=["side_to_move"],
        key=jax.random.key(0), verbose=False,
    )
    accs = [
        results["side_to_move"][lname]["best_accuracy"]
        for lname in results["side_to_move"]
    ]
    best = max(accs)
    # Side-to-move ≈ (ply % 2) on the unprefixed CLM. The embed output
    # already encodes the move-index parity, so even on a fresh-init
    # TINY model the linear probe lands far above chance. 0.80 is a
    # comfortable floor; observed ~1.0 on multiple seeds during dev.
    assert best > 0.80, f"side_to_move probe stuck near chance: per-layer accs={accs}"


def test_count_legal_moves_starting_position() -> None:
    """The starting chess position has exactly 20 legal white moves.
    ``_count_legal_moves`` should report ``20`` at every game's ply 0
    in a freshly-generated corpus, regardless of which random moves
    are subsequently played. (Test-risk: closes the gap on the
    bit-packed grid + promotion-adjustment pipeline.)"""
    from pawn.probes import _count_legal_moves

    _ids, _t, _lm, move_ids, game_lengths, _tc = (
        engine.generate_clm_batch(16, 24, 123, False, 0.0, False)
    )
    counts = _count_legal_moves(move_ids, game_lengths)
    # Every game starts with 20 legal moves for white. (game_lengths >= 1
    # for any non-trivial random game.)
    for i in range(counts.shape[0]):
        gl = int(game_lengths[i])
        if gl == 0:
            continue
        assert counts[i, 0] == 20, (
            f"game {i}: starting-position legal-move count "
            f"{counts[i, 0]} != 20"
        )


def test_count_legal_moves_within_known_range() -> None:
    """Sanity bound: legal-move counts at any valid ply land in
    ``[1, 218]`` (218 is the proven maximum for any legal chess
    position; ``Edwards 1988``). Catches gross miscount regressions
    even in positions we don't have an independent reference for."""
    from pawn.probes import _count_legal_moves

    _ids, _t, _lm, move_ids, game_lengths, _tc = (
        engine.generate_clm_batch(8, 32, 222, False, 0.0, False)
    )
    counts = _count_legal_moves(move_ids, game_lengths)
    for i in range(counts.shape[0]):
        gl = int(game_lengths[i])
        for p in range(gl):
            c = int(counts[i, p])
            assert 1 <= c <= 218, (
                f"game {i} ply {p}: legal-move count {c} outside [1, 218]"
            )
