"""Tests for `pawn.parity` — supernet-vs-canonical quality-parity harness.

Covers the Stage-B3 acceptance gates (`docs/phase_b_spec.md`): the harness
runs on two tiny checkpoints and emits a well-formed gap report with the
right metric signs/shapes. This is the *measurement-tool* contract — we pin
the mechanics, not a production-scale quality verdict.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from pawn.checkpoint import save_model
from pawn.config import NUM_ACTIONS, ModelConfig
from pawn.corpus import generate_corpus
from pawn.model import init_model, sliced
from pawn.parity import (
    GapReport,
    PhaseAccuracyGap,
    ProbeGap,
    ReferenceLoRASpec,
    ValLossGap,
    extract_hidden_states,
    phase_accuracy_gap,
    probe_decodability_gap,
    reference_lora_gap,
    reference_lora_val_loss,
    run_parity_harness,
    source_square_labels,
)
from pawn.trainer import slice_batch

# Tiny shapes keep these CPU-runnable. head_dim = d_model / n_heads. The
# "supernet" is wider than the "canonical" so the slicing path is exercised;
# the canonical width (16) nests under the supernet width (32).
_SUPERNET_CFG = ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64, head_dim=16)
_CANONICAL_CFG = ModelConfig(d_model=16, n_layers=2, n_heads=1, d_ff=64, head_dim=16)


def _corpus(n_games: int = 12, seq_len: int = 32):
    return generate_corpus(
        n_games=n_games, max_ply=seq_len, seq_len=seq_len, seed=7,
        conditioning=[],
    )


def _load_eval_parity() -> ModuleType:
    """Load ``scripts/eval_parity.py`` as a module (``scripts/`` isn't a
    package, so import-by-path via importlib — same pattern as the other
    script tests)."""
    script_path = Path("scripts") / "eval_parity.py"
    assert script_path.is_file()
    spec = importlib.util.spec_from_file_location(
        "scripts_eval_parity", script_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Hidden-state extraction + source-square labels
# ---------------------------------------------------------------------------


def test_extract_hidden_states_shapes() -> None:
    model = init_model(_CANONICAL_CFG, key=0)
    corpus = _corpus()
    hidden, tokens = extract_hidden_states(model, corpus, max_positions=64)
    assert hidden.ndim == 2
    assert hidden.shape[-1] == _CANONICAL_CFG.d_model
    assert tokens.ndim == 1
    assert hidden.shape[0] == tokens.shape[0]
    assert hidden.shape[0] <= 64
    # Every extracted target is a real move token (probe support).
    tok_np = np.asarray(tokens)
    assert tok_np.size > 0
    assert (tok_np >= 0).all() and (tok_np < NUM_ACTIONS).all()


def test_source_square_labels_in_range() -> None:
    model = init_model(_CANONICAL_CFG, key=0)
    corpus = _corpus()
    _hidden, tokens = extract_hidden_states(model, corpus, max_positions=64)
    labels = source_square_labels(model, tokens)
    lab = np.asarray(labels)
    assert lab.shape == np.asarray(tokens).shape
    assert (lab >= 0).all() and (lab < 64).all()
    # Labels match the decomp_table source-square column exactly.
    expected = np.asarray(model.decomp_table)[np.asarray(tokens), 0]
    np.testing.assert_array_equal(lab, expected)


# ---------------------------------------------------------------------------
# (1) Per-phase accuracy gap
# ---------------------------------------------------------------------------


def test_phase_accuracy_gap_zero_for_identical_models() -> None:
    """A model compared against itself has an exactly-zero accuracy gap."""
    model = init_model(_CANONICAL_CFG, key=1)
    corpus = _corpus()
    gap = phase_accuracy_gap(model, model, corpus, batch_size=4)
    assert isinstance(gap, PhaseAccuracyGap)
    assert gap.overall_delta == 0.0
    assert gap.opening_delta == 0.0
    assert gap.midgame_delta == 0.0
    assert gap.endgame_delta == 0.0
    # Both arms carry the same raw accuracy dict.
    assert gap.supernet == gap.canonical
    assert set(gap.supernet) == {
        "overall", "opening", "midgame", "endgame",
        "n_total", "n_opening", "n_midgame", "n_endgame",
    }


def test_phase_accuracy_gap_is_signed_difference() -> None:
    """The gap is exactly supernet_overall - canonical_overall."""
    from pawn.eval import compute_per_phase_accuracy

    sup = init_model(_CANONICAL_CFG, key=2)
    can = init_model(_CANONICAL_CFG, key=3)
    corpus = _corpus()
    gap = phase_accuracy_gap(sup, can, corpus, batch_size=4)
    sup_acc = compute_per_phase_accuracy(sup, corpus, batch_size=4)
    can_acc = compute_per_phase_accuracy(can, corpus, batch_size=4)
    assert gap.overall_delta == pytest.approx(
        sup_acc.overall - can_acc.overall, abs=1e-7
    )
    assert gap.supernet["overall"] == pytest.approx(sup_acc.overall, abs=1e-7)
    assert gap.canonical["overall"] == pytest.approx(can_acc.overall, abs=1e-7)


# ---------------------------------------------------------------------------
# (2) Probe decodability gap
# ---------------------------------------------------------------------------


def test_probe_gap_zero_for_identical_models() -> None:
    model = init_model(_CANONICAL_CFG, key=1)
    corpus = _corpus()
    gap = probe_decodability_gap(
        model, model, corpus, batch_size=4, max_positions=128, probe_epochs=3,
    )
    assert isinstance(gap, ProbeGap)
    assert gap.n_classes == 64
    assert gap.n_samples > 0
    # Same model, same seed, same data → identical probe accuracy.
    assert gap.supernet_accuracy == pytest.approx(gap.canonical_accuracy, abs=1e-7)
    assert gap.delta == pytest.approx(0.0, abs=1e-7)
    assert 0.0 <= gap.supernet_accuracy <= 1.0


def test_probe_gap_is_signed_difference() -> None:
    sup = init_model(_CANONICAL_CFG, key=2)
    can = init_model(_CANONICAL_CFG, key=3)
    corpus = _corpus()
    gap = probe_decodability_gap(
        sup, can, corpus, batch_size=4, max_positions=128, probe_epochs=3,
    )
    assert gap.delta == pytest.approx(
        gap.supernet_accuracy - gap.canonical_accuracy, abs=1e-7
    )
    # Discriminative: two *different* random backbones must yield *different*
    # probe accuracies (hence a non-zero gap). This is the load-bearing check
    # the harness exists for — it catches a degenerate harness that silently
    # measures nothing (e.g. constant/zero hidden states would make every
    # backbone's probe accuracy identical, collapsing the delta to 0).
    assert gap.supernet_accuracy != gap.canonical_accuracy
    assert gap.delta != 0.0


# ---------------------------------------------------------------------------
# (3) Reference-LoRA val-loss gap
# ---------------------------------------------------------------------------


def _ref_batches(corpus, batch_size: int = 4, n: int = 2):
    train = [
        slice_batch(corpus, (np.arange(batch_size) + b * batch_size) % corpus.n_games)
        for b in range(n)
    ]
    val = slice_batch(corpus, np.arange(batch_size))
    return train, val


def test_reference_lora_val_loss_is_finite() -> None:
    model = init_model(_CANONICAL_CFG, key=4)
    corpus = _corpus()
    train, val = _ref_batches(corpus)
    spec = ReferenceLoRASpec(rank=2, steps=4, seed=0)
    loss = reference_lora_val_loss(model, train, val, spec)
    assert np.isfinite(loss)
    assert loss > 0.0


def test_reference_lora_val_loss_deterministic_same_backbone() -> None:
    """Same backbone + spec + batches → identical val loss (seeded adapter)."""
    model = init_model(_CANONICAL_CFG, key=4)
    corpus = _corpus()
    train, val = _ref_batches(corpus)
    spec = ReferenceLoRASpec(rank=2, steps=4, seed=0)
    a = reference_lora_val_loss(model, train, val, spec)
    b = reference_lora_val_loss(model, train, val, spec)
    assert a == pytest.approx(b, rel=1e-6, abs=1e-6)


def test_reference_lora_gap_is_signed_difference() -> None:
    sup = init_model(_CANONICAL_CFG, key=5)
    can = init_model(_CANONICAL_CFG, key=6)
    corpus = _corpus()
    train, val = _ref_batches(corpus)
    spec = ReferenceLoRASpec(rank=2, steps=4, seed=0)
    gap = reference_lora_gap(sup, can, train, val, spec)
    assert isinstance(gap, ValLossGap)
    assert gap.steps == 4
    assert gap.delta == pytest.approx(
        gap.supernet_val_loss - gap.canonical_val_loss, rel=1e-6, abs=1e-6
    )
    # Discriminative: two *different* random backbones must reach *different*
    # held-out losses under the identical light finetune (hence a non-zero
    # gap). Guards against a no-op harness that returns the same loss for any
    # backbone (e.g. ignoring the backbone weights / returning a constant).
    assert gap.supernet_val_loss != gap.canonical_val_loss
    assert gap.delta != 0.0


def test_reference_lora_gap_zero_for_identical_backbone() -> None:
    model = init_model(_CANONICAL_CFG, key=5)
    corpus = _corpus()
    train, val = _ref_batches(corpus)
    spec = ReferenceLoRASpec(rank=2, steps=4, seed=0)
    gap = reference_lora_gap(model, model, train, val, spec)
    assert gap.delta == pytest.approx(0.0, rel=1e-6, abs=1e-6)


def test_reference_lora_reuses_train_batches_without_deletion() -> None:
    """``steps`` > ``len(train_batches)`` must not delete a reused batch.

    The inner ``step`` is ``eqx.filter_jit``-donated; the train loop cycles
    the same ``Batch`` objects when ``steps`` exceeds the number of batches.
    If ``batch`` were donated, a donation-honoring backend would alias and
    delete it after the first call, and the *next* iteration that re-passes
    the same array would raise ``RuntimeError: Array has been deleted``. With
    ``donate="all-except-first"`` (batch not donated) the reuse is safe. We
    assert the loop runs cleanly and the reused batch's arrays stay live.
    """
    model = init_model(_CANONICAL_CFG, key=4)
    corpus = _corpus()
    # 2 batches, 6 steps → each batch is reused 3×.
    train, val = _ref_batches(corpus, n=2)
    spec = ReferenceLoRASpec(rank=2, steps=6, seed=0)
    loss = reference_lora_val_loss(model, train, val, spec)
    assert np.isfinite(loss)
    # The reused train batches must not have been donated/deleted: their
    # device arrays are still readable after the loop completes.
    for batch in train:
        assert np.isfinite(np.asarray(batch.tokens)).all()


def test_reference_lora_requires_train_batches() -> None:
    model = init_model(_CANONICAL_CFG, key=5)
    corpus = _corpus()
    _train, val = _ref_batches(corpus)
    spec = ReferenceLoRASpec(rank=2, steps=4, seed=0)
    with pytest.raises(ValueError):
        reference_lora_val_loss(model, [], val, spec)


# ---------------------------------------------------------------------------
# Full harness — slices the supernet to the canonical width + emits a report
# ---------------------------------------------------------------------------


def test_run_parity_harness_emits_well_formed_report() -> None:
    supernet = init_model(_SUPERNET_CFG, key=0)
    canonical = init_model(_CANONICAL_CFG, key=1)
    eval_c = _corpus()
    train_c = _corpus(n_games=8)
    val_c = _corpus(n_games=8)
    report = run_parity_harness(
        supernet, canonical,
        eval_corpus=eval_c, train_corpus=train_c, val_corpus=val_c,
        batch_size=4,
        probe_max_positions=128,
        probe_epochs=3,
        ref_lora_spec=ReferenceLoRASpec(rank=2, steps=4, seed=0),
        ref_lora_train_batches=2,
    )
    assert isinstance(report, GapReport)
    # Comparison runs at the canonical's width.
    assert report.width == _CANONICAL_CFG.d_model

    # Report is JSON-serialisable and round-trips.
    d = report.to_dict()
    text = json.dumps(d)
    restored = json.loads(text)
    assert restored["width"] == _CANONICAL_CFG.d_model
    for block in ("phase_accuracy", "probe", "val_loss"):
        assert block in restored

    # Metric shapes: every delta is a finite float, accuracies in [0, 1].
    pa = report.phase_accuracy
    for delta in (
        pa.overall_delta, pa.opening_delta, pa.midgame_delta, pa.endgame_delta
    ):
        assert np.isfinite(delta)
    assert 0.0 <= report.probe.supernet_accuracy <= 1.0
    assert 0.0 <= report.probe.canonical_accuracy <= 1.0
    assert np.isfinite(report.val_loss.delta)
    assert report.val_loss.supernet_val_loss > 0.0
    assert report.val_loss.canonical_val_loss > 0.0


def test_harness_slices_supernet_against_explicit_slice() -> None:
    """The harness's supernet arm is exactly the nested slice of the supernet.

    Comparing the supernet against its own slice as the 'canonical' must
    produce an all-zero accuracy + probe gap, because the harness slices the
    supernet to the canonical width and the slice *is* the canonical.
    """
    supernet = init_model(_SUPERNET_CFG, key=0)
    canonical = sliced(supernet, _CANONICAL_CFG)
    eval_c = _corpus()
    train_c = _corpus(n_games=8)
    val_c = _corpus(n_games=8)
    report = run_parity_harness(
        supernet, canonical,
        eval_corpus=eval_c, train_corpus=train_c, val_corpus=val_c,
        batch_size=4,
        probe_max_positions=128,
        probe_epochs=3,
        ref_lora_spec=ReferenceLoRASpec(rank=2, steps=4, seed=0),
        ref_lora_train_batches=2,
    )
    # supernet-slice == canonical → zero gap on every metric.
    assert report.phase_accuracy.overall_delta == pytest.approx(0.0, abs=1e-7)
    assert report.probe.delta == pytest.approx(0.0, abs=1e-7)
    assert report.val_loss.delta == pytest.approx(0.0, rel=1e-6, abs=1e-6)


def test_harness_same_width_no_slice() -> None:
    """When supernet and canonical share a config, no slicing is needed and
    the harness still emits a valid report."""
    sup = init_model(_CANONICAL_CFG, key=0)
    can = init_model(_CANONICAL_CFG, key=1)
    eval_c = _corpus()
    train_c = _corpus(n_games=8)
    val_c = _corpus(n_games=8)
    report = run_parity_harness(
        sup, can,
        eval_corpus=eval_c, train_corpus=train_c, val_corpus=val_c,
        batch_size=4,
        probe_max_positions=128,
        probe_epochs=3,
        ref_lora_spec=ReferenceLoRASpec(rank=2, steps=4, seed=0),
        ref_lora_train_batches=2,
    )
    assert report.width == _CANONICAL_CFG.d_model


# ---------------------------------------------------------------------------
# Script entry point — loads two checkpoints + emits JSON
# ---------------------------------------------------------------------------


def test_eval_parity_script_emits_json(tmp_path: Path) -> None:
    eval_parity = _load_eval_parity()

    supernet = init_model(_SUPERNET_CFG, key=0)
    canonical = init_model(_CANONICAL_CFG, key=1)
    sup_dir = tmp_path / "supernet"
    can_dir = tmp_path / "canonical"
    save_model(supernet, sup_dir, run_config={"conditioning": []})
    save_model(canonical, can_dir, run_config={"conditioning": []})

    out = tmp_path / "parity.json"
    rc = eval_parity.main([
        "--supernet-checkpoint", str(sup_dir),
        "--canonical-checkpoint", str(can_dir),
        "--n-games", "8",
        "--max-ply", "32",
        "--seq-len", "32",
        "--batch-size", "4",
        "--ref-lora-train-games", "8",
        "--ref-lora-val-games", "8",
        "--ref-lora-steps", "4",
        "--ref-lora-rank", "2",
        "--probe-max-positions", "128",
        "--probe-epochs", "3",
        "--output", str(out),
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["width"] == _CANONICAL_CFG.d_model
    assert payload["supernet_checkpoint"] == str(sup_dir)
    assert "phase_accuracy" in payload
    assert "probe" in payload
    assert "val_loss" in payload


def test_eval_parity_script_rejects_conditioning_mismatch(tmp_path: Path) -> None:
    eval_parity = _load_eval_parity()

    supernet = init_model(_SUPERNET_CFG, key=0)
    canonical = init_model(_CANONICAL_CFG, key=1)
    sup_dir = tmp_path / "supernet"
    can_dir = tmp_path / "canonical"
    # Different conditioning layouts → harness must refuse.
    save_model(supernet, sup_dir, run_config={"conditioning": ["outcome"]})
    save_model(canonical, can_dir, run_config={"conditioning": []})

    with pytest.raises(SystemExit):
        eval_parity.main([
            "--supernet-checkpoint", str(sup_dir),
            "--canonical-checkpoint", str(can_dir),
            "--n-games", "8",
            "--max-ply", "32",
            "--seq-len", "32",
        ])
