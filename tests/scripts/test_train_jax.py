"""Tests for ``scripts/train_jax.py`` — Phase-2 driver.

Focused on the upfront validation guards (the ``SystemExit`` paths the
script promises to fire before any filesystem write or corpus
generation). The full end-to-end happy path is exercised by the
Phase-2 verification run that ships with the PR body.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("optax")
pytest.importorskip("chess_engine")

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script() -> ModuleType:
    name = "train_jax_test_module"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(SCRIPTS / "train_jax.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def _run(args: list[str], tmp_path: Path) -> None:
    """Invoke ``main(args)`` with ``--logs-dir`` defaulting under tmp.
    Returns whatever ``main`` returns (typically 0 on success); raises
    SystemExit for validation failures."""
    script = _load_script()
    if "--logs-dir" not in args:
        args = args + ["--logs-dir", str(tmp_path)]
    script.main(args)


def test_rejects_k_zero(tmp_path: Path) -> None:
    """``--k 0`` fails fast with a clear SystemExit, not a
    ZeroDivisionError downstream."""
    with pytest.raises(SystemExit, match="--k"):
        _run(
            [
                "--supernet", "tiny",
                "--total-steps", "10",
                "--k", "0",
                "--batch-size", "2",
                "--seq-len", "16",
                "--warmup-steps", "1",
            ],
            tmp_path,
        )


def test_rejects_total_steps_not_multiple_of_k(tmp_path: Path) -> None:
    """``--total-steps`` not divisible by ``--k`` fails fast."""
    with pytest.raises(SystemExit, match="multiple of"):
        _run(
            [
                "--supernet", "tiny",
                "--total-steps", "7",
                "--k", "3",
                "--batch-size", "2",
                "--seq-len", "16",
                "--warmup-steps", "1",
            ],
            tmp_path,
        )


def test_rejects_seq_len_exceeding_max(tmp_path: Path) -> None:
    """``--seq-len`` > supernet.max_seq_len fails BEFORE corpus
    generation (Codex P2). The TINY supernet has max_seq_len=512;
    request 1024 and the script must exit immediately."""
    with pytest.raises(SystemExit, match="seq-len"):
        _run(
            [
                "--supernet", "tiny",
                "--total-steps", "10",
                "--k", "5",
                "--batch-size", "2",
                "--seq-len", "1024",
                "--warmup-steps", "1",
            ],
            tmp_path,
        )


def test_rejects_oversized_corpus(tmp_path: Path) -> None:
    """``--max-corpus-gb`` guard fires when the requested corpus
    exceeds the limit. Use a tiny limit to force the guard with a
    small request."""
    with pytest.raises(SystemExit, match="max-corpus-gb"):
        _run(
            [
                "--supernet", "tiny",
                "--total-steps", "1000",
                "--k", "10",
                "--batch-size", "256",
                "--seq-len", "512",
                "--warmup-steps", "10",
                "--max-corpus-gb", "0.0001",  # 100 KiB cap; request is ~1.2 GiB
            ],
            tmp_path,
        )


def test_validation_failures_do_not_create_run_dir(tmp_path: Path) -> None:
    """All upfront validation must abort BEFORE the run directory is
    created. Pre-existing review-test-risk HIGH finding: an orphaned
    ``jax_run_*`` dir for every misconfigured invocation is
    operationally noisy.

    Each case fully specifies the failing arg list (no merging with a
    default — argparse uses the last occurrence and would silently
    accept a fix-up override)."""
    cases = [
        # --k 0
        ["--supernet", "tiny", "--total-steps", "10", "--k", "0",
         "--batch-size", "2", "--seq-len", "16", "--warmup-steps", "1"],
        # total_steps not divisible
        ["--supernet", "tiny", "--total-steps", "7", "--k", "3",
         "--batch-size", "2", "--seq-len", "16", "--warmup-steps", "1"],
        # seq_len > max_seq_len
        ["--supernet", "tiny", "--total-steps", "10", "--k", "5",
         "--batch-size", "2", "--seq-len", "9999", "--warmup-steps", "1"],
        # max_corpus_gb too small
        ["--supernet", "tiny", "--total-steps", "1000", "--k", "10",
         "--batch-size", "256", "--seq-len", "512", "--warmup-steps", "10",
         "--max-corpus-gb", "1e-10"],
    ]
    for args in cases:
        with pytest.raises(SystemExit):
            _run(args, tmp_path)
    # No jax_run_* directory should have been created by any of the
    # failed invocations.
    runs = list(tmp_path.glob("jax_run_*"))
    assert not runs, f"validation failures leaked run dirs: {runs}"
