"""Smoke tests for the v2 entry-point scripts.

Confirms each script imports cleanly + responds to `--help` / arg
parsing. The full functional verification (1000-step train, real
LoRA fine-tune, etc.) runs as the S13 / S16 acceptance criterion
checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = (
    "train_jax",
    "train_jax_adapter",
    "train_jax_distill",
    "eval_jax",
    "eval_parity",
    "eval_probes_jax",
    "eval_generation_jax",
    "eval_vs_stockfish",
    "sweep",
    "run_evals_backbone",
)


@pytest.mark.parametrize("script_name", SCRIPTS)
def test_script_module_imports(script_name: str) -> None:
    """Each script under `scripts/` imports cleanly without side effects."""
    import importlib.util

    script_path = Path("scripts") / f"{script_name}.py"
    assert script_path.is_file()
    spec = importlib.util.spec_from_file_location(
        f"scripts_{script_name}", script_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")


@pytest.mark.parametrize("script_name", SCRIPTS)
def test_script_help_works(script_name: str) -> None:
    """`--help` exits 0 — argparse is wired up correctly."""
    import subprocess

    result = subprocess.run(
        [sys.executable, f"scripts/{script_name}.py", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{script_name} --help failed:\n{result.stdout}\n{result.stderr}"
    )


def _subprocess_env() -> "dict[str, str]":
    """Subprocess env for CPU-friendly script tests.

    `_require_accelerator()` refuses to run on CPU unless
    `PAWN_ALLOW_CPU=1` is set (parity with v1). Round-3 codex P2: the
    subprocess tests below need this override or they fail before
    reaching the guard they're trying to pin on CPU-only CI.
    """
    import os
    env = os.environ.copy()
    env["PAWN_ALLOW_CPU"] = "1"
    return env


def test_train_jax_adapter_rejects_rosa_resume(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """RoSA + --resume with step > 0 must fail loudly before any
    optimizer-state damage. Round-2 test-risk: the prior fix was
    verified only by manual smoke run; this test pins the rejection.

    Constructs a minimal fake checkpoint directory with just enough
    `training_state.json` for the early `_resume_step_peek` read to
    fire, then runs the script as a subprocess and asserts the exit
    message comes from our SystemExit rather than a downstream
    cryptic shape mismatch.

    Uses an in-tree TINY_SUPERNET-shaped fake (the `--checkpoint`
    arg refers to a path on disk, but the script's early-exit code
    path doesn't load it before the RoSA guard fires)."""
    import json
    import subprocess

    fake_ckpt = tmp_path / "step_00000100"
    fake_ckpt.mkdir()
    (fake_ckpt / "training_state.json").write_text(json.dumps({"step": 100}))

    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax_adapter.py",
            "--strategy", "rosa", "--rosa-mode", "rosa",
            "--supernet", "tiny", "--variant", "small",
            "--checkpoint", "thomas-schweich/pawn-small",
            "--no-pgn", "--total-steps", "6",
            "--batch-size", "4", "--seq-len", "16", "--k", "2",
            "--lora-rank", "2", "--density", "0.1",
            "--local-checkpoints", "--lr", "1e-3",
            "--resume", str(fake_ckpt),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=_subprocess_env(),
    )
    combined = result.stdout + result.stderr
    # Specific guard message (not just any non-zero exit — that could
    # be an unrelated import error). Round-3 test-risk MEDIUM.
    assert "--resume is not supported for RoSA" in combined, (
        f"Expected the RoSA-resume guard message; got stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert result.returncode != 0, "RoSA --resume should fail"


def test_train_jax_adapter_rejects_conditioning_mismatch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Phase-A spec Chunk 4 / C1: an adapter run whose `--conditioning`
    disagrees with the backbone's persisted conditioning must fail loudly
    (the load-time C guard), not silently shift every move's absolute
    RoPE offset.

    Builds a tiny backbone checkpoint persisting `conditioning=["outcome"]`
    (C=2), then runs the adapter with the default empty conditioning
    (C=1) and asserts the `assert_conditioning_C` failure surfaces.
    """
    import subprocess

    from pawn.checkpoint import save_model
    from pawn.config import TINY_SUPERNET
    from pawn.model import init_model

    backbone = init_model(TINY_SUPERNET, key=0)
    ckpt_dir = tmp_path / "backbone"
    save_model(
        backbone, ckpt_dir, training_state={"step": 0},
        run_config={"conditioning": ["outcome"]},
    )

    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax_adapter.py",
            "--strategy", "lora",
            "--supernet", "tiny", "--variant", "small",
            "--checkpoint", str(ckpt_dir),
            "--no-pgn", "--total-steps", "2",
            "--batch-size", "4", "--seq-len", "16", "--k", "1",
            "--lora-rank", "2",
            "--local-checkpoints", "--lr", "1e-3",
            # NB: no --conditioning → cfg.conditioning defaults to [] (C=1),
            # which disagrees with the backbone's persisted C=2.
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=_subprocess_env(),
    )
    combined = result.stdout + result.stderr
    assert "conditioning mismatch" in combined, (
        f"Expected the load-time C-mismatch guard; got "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.returncode != 0, "conditioning mismatch should fail"


def test_train_jax_conditioning_threaded_into_corpus(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A `--conditioning outcome` pretrain must TRAIN under C=2, not just
    record C=2 in config.json.

    Regression guard for SC-1 / A1: the trainer used to build the corpus
    with the default `conditioning=()` (C=1, BOS-only) while persisting
    `conditioning=["outcome"]` (C=2) — so the model trained at one
    absolute-RoPE offset and every eval/read path (which derives C from
    the persisted block) placed moves at a different offset. This pins
    that the train-time C and the persisted C can never diverge again:

    1. Run a tiny real pretrain with `--conditioning outcome`.
    2. The written checkpoint's run block records `conditioning=["outcome"]`.
    3. Rebuilding the corpus the way eval does (from that run block)
       lands moves at `outcome_offset[0] == C == 2` — the same C the
       trainer actually packed, because both now derive from the one
       persisted `conditioning` field.
    """
    import subprocess

    from pawn.checkpoint import load_model
    from pawn.corpus import (
        conditioning_from_run_block,
        conditioning_to_C,
        generate_corpus,
    )

    logs_dir = tmp_path / "logs"
    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax.py",
            "--supernet", "tiny", "--total-steps", "4",
            "--batch-size", "4", "--seq-len", "32", "--k", "2",
            "--conditioning", "outcome",
            "--checkpoint-interval", "2",
            "--local-checkpoints", "--lr", "1e-3",
            "--logs-dir", str(logs_dir),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, (
        f"tiny pretrain failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )

    ckpts = sorted(logs_dir.glob("*/step_*"))
    assert ckpts, (
        f"no checkpoint written under {logs_dir}; "
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    _model, run_block = load_model(ckpts[-1])
    # The persisted run block must record the conditioning the run used.
    conditioning = conditioning_from_run_block(run_block)
    assert conditioning == ["outcome"], (
        f"checkpoint run block conditioning={conditioning!r}, expected "
        f"['outcome'] — the trainer dropped cfg.conditioning on the floor"
    )
    C = conditioning_to_C(conditioning)
    assert C == 2

    # Rebuild the corpus exactly as eval_jax does (from the persisted
    # block). If the trainer had built C=1 while persisting C=2, this
    # eval-side corpus would place moves one absolute-RoPE slot away
    # from where the model trained. The per-game prefix width is the
    # constant C, recorded in `outcome_offset`.
    corpus = generate_corpus(
        n_games=8, max_ply=32, seq_len=32, seed=0,
        conditioning=conditioning,
    )
    assert int(corpus.outcome_offset[0]) == C, (
        f"corpus prefix width {int(corpus.outcome_offset[0])} != C={C}"
    )


def test_train_jax_adapter_rejects_resume_without_training_state(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """--resume against a directory missing `training_state.json` must
    fail loudly. The prior code silently treated the absent file as
    `step=0`, bypassing the RoSA guard and producing a cryptic
    downstream error. Round-2 test-risk MEDIUM."""
    import subprocess

    fake_ckpt = tmp_path / "step_00000050_no_ts"
    fake_ckpt.mkdir()
    # Intentionally do NOT write training_state.json.

    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax_adapter.py",
            "--strategy", "lora",
            "--supernet", "tiny", "--variant", "small",
            "--checkpoint", "thomas-schweich/pawn-small",
            "--no-pgn", "--total-steps", "6",
            "--batch-size", "4", "--seq-len", "16", "--k", "2",
            "--lora-rank", "2",
            "--local-checkpoints", "--lr", "1e-3",
            "--resume", str(fake_ckpt),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=_subprocess_env(),
    )
    combined = result.stdout + result.stderr
    # Match the guard's distinctive phrase so a generic traceback
    # mentioning the filename can't satisfy the assertion
    # (round-4 bug-detector MINOR).
    assert "--resume requires" in combined and "training_state.json" in combined, (
        f"Expected the missing-sidecar guard; got stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert result.returncode != 0, (
        "Resume against a sidecar-less dir should fail"
    )


# ---------------------------------------------------------------------------
# B2 — inert-knob parity: mate_boost / accumulation_steps / adapter cadence
# ---------------------------------------------------------------------------


def _load_adapter_module():  # type: ignore[no-untyped-def]
    """Import `scripts/train_jax_adapter.py` as a module to reach
    `_resolve_cadence` (the pure cadence/sampling resolver)."""
    import importlib.util

    script_path = Path("scripts") / "train_jax_adapter.py"
    spec = importlib.util.spec_from_file_location(
        "scripts_train_jax_adapter_cadence", script_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _adapter_cfg(**overrides: object):  # type: ignore[no-untyped-def]
    from typing import Any

    from pawn.run_config import AdapterConfig

    base: dict[str, Any] = dict(
        local_checkpoints=True, total_steps=100, strategy="lora", lora_rank=4,
        batch_size=4,
    )
    base.update(overrides)
    return AdapterConfig(**base)


def test_resolve_cadence_data_seed_changes_sampling_seed() -> None:
    """B2: `data_seed` is consumed — it seeds the train sampler (and the
    val sampler at `data_seed + 1`). Different `data_seed` ⇒ different seed
    drives different game orders."""
    mod = _load_adapter_module()
    c_default = mod._resolve_cadence(_adapter_cfg(), n_train_games=1000)
    assert c_default.data_seed == 0  # None → 0
    c_seeded = mod._resolve_cadence(
        _adapter_cfg(data_seed=42), n_train_games=1000
    )
    assert c_seeded.data_seed == 42
    # The seed actually changes the sampled game order.
    import numpy as np

    a = np.random.default_rng(c_default.data_seed).integers(0, 1000, size=8)
    b = np.random.default_rng(c_seeded.data_seed).integers(0, 1000, size=8)
    assert not np.array_equal(a, b)


def test_resolve_cadence_epochs_steps_per_epoch_set_budget() -> None:
    """B2: `epochs` × `steps_per_epoch` resolve the step budget (v1
    semantics). With them unset the budget is `total_steps` and the
    step-based `eval_interval` drives val (epochs/val_every are no-ops)."""
    mod = _load_adapter_module()
    # Unset steps_per_epoch → total_steps is the budget; epochs ignored.
    c_none = mod._resolve_cadence(
        _adapter_cfg(total_steps=100, epochs=7), n_train_games=1000
    )
    assert c_none.effective_total_steps == 100
    assert c_none.eval_interval == 100  # eval_interval None → log_interval

    # Explicit int steps_per_epoch → epochs × steps_per_epoch.
    c_int = mod._resolve_cadence(
        _adapter_cfg(epochs=3, steps_per_epoch=10, val_every=2),
        n_train_games=1000,
    )
    assert c_int.epoch_steps == 10
    assert c_int.effective_total_steps == 30
    assert c_int.eval_interval == 20  # val_every × epoch_steps

    # steps_per_epoch="all" → n_train_games // batch_size.
    c_all = mod._resolve_cadence(
        _adapter_cfg(epochs=2, steps_per_epoch="all", batch_size=4),
        n_train_games=400,
    )
    assert c_all.epoch_steps == 100  # 400 // 4
    assert c_all.effective_total_steps == 200


def test_resolve_cadence_val_every_changes_eval_interval() -> None:
    """B2: `val_every` is consumed — it scales the eval cadence in epoch
    units (only meaningful when steps_per_epoch defines an epoch)."""
    mod = _load_adapter_module()
    c1 = mod._resolve_cadence(
        _adapter_cfg(steps_per_epoch=10, val_every=1), n_train_games=1000
    )
    c3 = mod._resolve_cadence(
        _adapter_cfg(steps_per_epoch=10, val_every=3), n_train_games=1000
    )
    assert c1.eval_interval == 10
    assert c3.eval_interval == 30
    assert c3.eval_interval != c1.eval_interval


def test_mate_boost_threaded_into_generate_corpus() -> None:
    """B2: `mate_boost` is consumed by `generate_corpus` (it maps onto the
    engine's mate-biasing arg). A positive boost changes the generated
    corpus for a fixed seed, proving the field is no longer inert."""
    import numpy as np

    from pawn.corpus import generate_corpus

    plain = generate_corpus(
        n_games=64, max_ply=60, seq_len=64, seed=7, mate_boost=0.0
    )
    boosted = generate_corpus(
        n_games=64, max_ply=60, seq_len=64, seed=7, mate_boost=5.0
    )
    assert not np.array_equal(plain.tokens, boosted.tokens), (
        "mate_boost did not reach the engine — corpus identical to the "
        "mate_boost=0 baseline"
    )


def test_discard_ply_limit_threaded_into_generate_corpus() -> None:
    """B2: the sibling `discard_ply_limit` engine knob is likewise threaded
    through `generate_corpus` (consumed, not inert)."""
    import numpy as np

    from pawn.corpus import generate_corpus

    keep = generate_corpus(
        n_games=128, max_ply=20, seq_len=24, seed=3, discard_ply_limit=False
    )
    drop = generate_corpus(
        n_games=128, max_ply=20, seq_len=24, seed=3, discard_ply_limit=True
    )
    assert not np.array_equal(keep.tokens, drop.tokens)


def test_train_jax_accumulation_steps_runs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """B2 smoke: `--accumulation-steps 2` runs end-to-end (no
    NotImplementedError) and writes a checkpoint — the prefetcher now emits
    `(K, N, B, T)` batches that the accumulation kernel consumes."""
    import subprocess

    logs_dir = tmp_path / "logs"
    result = subprocess.run(
        [
            sys.executable, "scripts/train_jax.py",
            "--supernet", "tiny", "--total-steps", "4",
            "--accumulation-steps", "2",
            "--batch-size", "4", "--seq-len", "32", "--k", "2",
            "--checkpoint-interval", "2",
            "--local-checkpoints", "--lr", "1e-3",
            "--logs-dir", str(logs_dir),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=_subprocess_env(),
    )
    assert "NotImplementedError" not in (result.stdout + result.stderr), (
        f"accumulation path still raised NotImplementedError:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert result.returncode == 0, (
        f"accumulation_steps=2 pretrain failed:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    ckpts = sorted(logs_dir.glob("*/step_*"))
    assert ckpts, (
        f"no checkpoint written under {logs_dir}; "
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
