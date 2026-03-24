"""Regression test: training processes exit gracefully on SIGTERM.

Spawns a toy training run as a subprocess, sends SIGTERM, and verifies
that the process exits cleanly with valid checkpoints.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from pawn.checkpoint import load_backbone_weights, _verify_complete_sentinel


@pytest.fixture
def train_tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _wait_for_training_start(proc: subprocess.Popen, timeout: float = 30) -> None:
    """Wait until the training process prints 'Starting training'."""
    deadline = time.time() + timeout
    assert proc.stdout is not None
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        text = line.decode("utf-8", errors="replace")
        if "Starting training" in text:
            return
    raise TimeoutError("Training process did not start within timeout")


def test_sigterm_produces_valid_checkpoint(train_tmpdir):
    """Spawn a toy training run, send SIGTERM, verify checkpoint is valid."""
    ckpt_dir = train_tmpdir / "checkpoints"
    log_dir = train_tmpdir / "logs"

    proc = subprocess.Popen(
        [
            sys.executable, "scripts/train.py",
            "--toy",
            "--local-checkpoints",
            "--total-steps", "5000",
            "--device", "cpu",
            "--num-workers", "0",
            "--checkpoint-dir", str(ckpt_dir),
            "--log-dir", str(log_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    try:
        _wait_for_training_start(proc)

        # Let it train for a few checkpoints
        time.sleep(5)

        # Send SIGTERM
        proc.send_signal(signal.SIGTERM)

        # Wait for graceful exit
        returncode = proc.wait(timeout=30)
    except Exception:
        proc.kill()
        proc.wait()
        raise

    # Process should exit cleanly (0 from break, not 128+signal from sys.exit)
    assert returncode == 0, f"Training exited with code {returncode}"

    # Find checkpoints (trainer saves under {log_dir}/run_*/checkpoints/)
    checkpoints = sorted(log_dir.glob("run_*/checkpoints/step_*/"))
    assert len(checkpoints) > 0, "No checkpoints were saved"

    # Every checkpoint must have .complete sentinel and pass integrity check
    for ckpt in checkpoints:
        assert (ckpt / ".complete").exists(), f"Missing .complete in {ckpt.name}"
        _verify_complete_sentinel(ckpt)  # raises on failure

        # Verify we can actually load the weights
        weights, config = load_backbone_weights(ckpt)
        assert config is not None, f"No config in {ckpt.name}"
        assert len(weights) > 0, f"Empty weights in {ckpt.name}"


def test_sigterm_does_not_leave_tmp_dirs(train_tmpdir):
    """SIGTERM should not leave .tmp directories from interrupted saves."""
    ckpt_dir = train_tmpdir / "checkpoints"
    log_dir = train_tmpdir / "logs"

    proc = subprocess.Popen(
        [
            sys.executable, "scripts/train.py",
            "--toy",
            "--local-checkpoints",
            "--total-steps", "5000",
            "--device", "cpu",
            "--num-workers", "0",
            "--checkpoint-dir", str(ckpt_dir),
            "--log-dir", str(log_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    try:
        _wait_for_training_start(proc)
        time.sleep(5)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
        proc.wait()
        raise

    # No .tmp directories should exist anywhere under log_dir
    tmp_dirs = list(log_dir.glob("**/*.tmp"))
    assert len(tmp_dirs) == 0, f"Leftover .tmp directories: {tmp_dirs}"
