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
    "eval_jax",
    "eval_probes_jax",
    "eval_generation_jax",
    "eval_vs_stockfish",
    "sweep",
    "convert_published_checkpoints",
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
