"""Tests for `pawn.lifecycle` — SIGTERM, HF push, --resume helpers."""

from __future__ import annotations

import os
import signal
import threading
import time
import unittest.mock as mock
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pytest

from pawn.checkpoint import save_model
from pawn.config import TINY_SUPERNET
from pawn.lifecycle import (
    HFPushTracker,
    _reset_shutdown_state_for_tests,
    drain_push_queue,
    install_sigterm_handler,
    load_resume_state,
    push_checkpoint_async,
)
from pawn.model import init_model


@pytest.fixture(autouse=True)
def _reset_shutdown() -> None:
    _reset_shutdown_state_for_tests()


# ---------------------------------------------------------------------------
# HFPushTracker
# ---------------------------------------------------------------------------


class _FakeHfApi:
    """Test double for `huggingface_hub.HfApi.upload_folder`."""

    calls: list[dict] = []

    def upload_folder(self, **kwargs):
        _FakeHfApi.calls.append(kwargs)


def test_push_checkpoint_async_enqueues_upload(tmp_path: Path) -> None:
    """The async push submits an upload via the tracker's executor;
    `drain_push_queue` then blocks until it completes."""
    _FakeHfApi.calls = []
    ckpt = tmp_path / "step_00000010"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"x")
    tracker = HFPushTracker(repo_id="ns/test", branch="run/test")
    try:
        push_checkpoint_async(ckpt, tracker, upload_cls=_FakeHfApi)
        failures = drain_push_queue(tracker, timeout=5.0)
        assert failures == 0
        assert len(_FakeHfApi.calls) == 1
        call = _FakeHfApi.calls[0]
        assert call["repo_id"] == "ns/test"
        assert call["revision"] == "run/test"
        assert call["path_in_repo"] == "step_00000010"
    finally:
        tracker.shutdown()


def test_push_checkpoint_async_failures_dont_raise(tmp_path: Path) -> None:
    """Upload failures are reported as a count, not raised — the
    training loop keeps going."""

    class _FailingApi:
        def upload_folder(self, **kwargs):
            raise RuntimeError("network down")

    ckpt = tmp_path / "step_00000010"
    ckpt.mkdir()
    tracker = HFPushTracker(repo_id="ns/test")
    try:
        push_checkpoint_async(ckpt, tracker, upload_cls=_FailingApi)
        failures = drain_push_queue(tracker, timeout=5.0)
        assert failures == 1
    finally:
        tracker.shutdown()


def test_push_checkpoint_async_requires_huggingface_hub_when_no_cls(
    tmp_path: Path,
) -> None:
    """If huggingface_hub isn't available and the caller doesn't pass
    `upload_cls`, the function raises a clear RuntimeError."""
    ckpt = tmp_path / "step_00000010"
    ckpt.mkdir()
    tracker = HFPushTracker(repo_id="ns/test")
    try:
        with mock.patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(RuntimeError, match="huggingface_hub"):
                push_checkpoint_async(ckpt, tracker)
    finally:
        tracker.shutdown()


# ---------------------------------------------------------------------------
# SIGTERM handler
# ---------------------------------------------------------------------------


def test_install_sigterm_handler_sets_should_shutdown_flag() -> None:
    """SIGTERM flips the should_shutdown poll to True."""
    should_shutdown = install_sigterm_handler()
    assert should_shutdown() is False
    # Trigger the handler synchronously.
    os.kill(os.getpid(), signal.SIGTERM)
    # The signal handler runs synchronously in the main thread, so by
    # the time `kill` returns, the flag should be set.
    time.sleep(0.05)  # Give signal delivery a moment.
    assert should_shutdown() is True


def test_install_sigterm_handler_fires_on_shutdown_callback() -> None:
    called = threading.Event()

    def cb() -> None:
        called.set()

    install_sigterm_handler(on_shutdown=cb)
    os.kill(os.getpid(), signal.SIGTERM)
    assert called.wait(timeout=1.0)


def test_install_sigterm_handler_is_idempotent() -> None:
    """Multiple SIGTERMs only fire the callback once."""
    count = {"n": 0}

    def cb() -> None:
        count["n"] += 1

    install_sigterm_handler(on_shutdown=cb)
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(0.05)
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(0.05)
    assert count["n"] == 1


# ---------------------------------------------------------------------------
# load_resume_state
# ---------------------------------------------------------------------------


def test_load_resume_state_splices_step_from_training_state_json(
    tmp_path: Path,
) -> None:
    """Plan §10 S12: state.step is spliced from the saved value so
    metrics stay monotonic across the resume."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00050000"
    save_model(model, out_dir, training_state={"step": 50000})

    opt = optax.adamw(1e-3)
    state = load_resume_state(out_dir, opt, key=jax.random.key(0))
    assert int(state.step) == 50000


def test_load_resume_state_falls_back_to_dir_name(tmp_path: Path) -> None:
    """When training_state.json is absent, parse `step_<N>` from the
    dir name."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00012345"
    save_model(model, out_dir)  # no training_state
    opt = optax.adamw(1e-3)
    state = load_resume_state(out_dir, opt, key=jax.random.key(0))
    assert int(state.step) == 12345


def test_load_resume_state_returns_train_state_compatible_with_train_step(
    tmp_path: Path,
) -> None:
    """The returned state plugs directly into the trainer's step
    function (model + opt_state + step + key fields all present)."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir, training_state={"step": 10})
    opt = optax.adamw(1e-3)
    state = load_resume_state(out_dir, opt, key=jax.random.key(0))
    assert state.model is not None
    assert state.opt_state is not None
    assert isinstance(state.step, jax.Array)
    assert state.key is not None
