"""Tests for `pawn.lifecycle` — SIGTERM, HF push, --resume helpers."""

from __future__ import annotations

import os
import signal
import threading
import time
import unittest.mock as mock
from pathlib import Path
from typing import Any

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
        timeouts, errors = drain_push_queue(tracker, timeout=5.0)
        assert (timeouts, errors) == (0, 0)
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
        timeouts, errors = drain_push_queue(tracker, timeout=5.0)
        # Per round-2: a raised exception is an `error`, not a
        # `timeout`. The worker exited normally; no thread is stuck.
        assert (timeouts, errors) == (0, 1)
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


def test_join_distinguishes_timeouts_from_errors(tmp_path: Path) -> None:
    """Round-2 review-bug-detector: a transient upload exception
    (worker thread exited normally) must not be conflated with a
    timeout (worker still running). Only the latter should drive the
    abandon-thread path at shutdown.
    """
    ckpt = tmp_path / "step_00000010"
    ckpt.mkdir()
    tracker = HFPushTracker(repo_id="ns/test")

    class _RaisingHfApi:
        def upload_folder(self, **_kwargs: Any) -> None:
            raise RuntimeError("simulated upload failure")

    try:
        push_checkpoint_async(ckpt, tracker, upload_cls=_RaisingHfApi)
        timeouts, errors = tracker.join(timeout=5.0)
        assert timeouts == 0, "exception ≠ timeout"
        assert errors == 1, "the raising upload should count as an error"
    finally:
        tracker.shutdown()


def test_shutdown_with_stuck_upload_returns_promptly(
    tmp_path: Path,
) -> None:
    """Round-2 codex P2 + review-bug-detector Critical: a stuck upload
    thread used to keep the process alive via
    `concurrent.futures.thread._python_exit`'s join-on-non-daemon. The
    fix: worker threads are daemonic via `_DaemonThreadPoolExecutor`,
    and `shutdown(drain_succeeded=False)` returns without waiting. This
    test injects a never-completing upload and asserts the shutdown
    call returns inside a tight wall-clock bound.
    """
    ckpt = tmp_path / "step_00000010"
    ckpt.mkdir()
    tracker = HFPushTracker(repo_id="ns/test")
    blocker = threading.Event()

    class _StuckHfApi:
        def upload_folder(self, **_kwargs: Any) -> None:
            # Never completes — simulates a network hang.
            blocker.wait()

    try:
        push_checkpoint_async(ckpt, tracker, upload_cls=_StuckHfApi)
        # Drain with a tiny budget so we definitely time out.
        timeouts, errors = tracker.join(timeout=0.2)
        assert timeouts == 1, "stuck upload should produce a timeout"
        assert errors == 0
        # The shutdown call itself must return promptly — the daemon
        # thread is left running but the executor doesn't block on it.
        t0 = time.monotonic()
        tracker.shutdown(drain_succeeded=False)
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, (
            f"shutdown(drain_succeeded=False) blocked for {elapsed:.2f}s; "
            f"the daemon-thread path is meant to return promptly"
        )
    finally:
        # Release the stuck thread so it doesn't linger across tests.
        blocker.set()


def test_executor_worker_threads_are_daemonic(tmp_path: Path) -> None:
    """Round-2 codex/bug: the executor's worker threads must be
    daemonic so `concurrent.futures.thread._python_exit` doesn't join
    them at interpreter shutdown — which would re-introduce the hang
    `drain_succeeded=False` is meant to prevent.
    """
    ckpt = tmp_path / "step_00000010"
    ckpt.mkdir()
    tracker = HFPushTracker(repo_id="ns/test")

    class _NoopHfApi:
        def upload_folder(self, **_kwargs: Any) -> None:
            pass

    try:
        # Submit one upload to force the executor to spin a worker.
        push_checkpoint_async(ckpt, tracker, upload_cls=_NoopHfApi)
        tracker.join(timeout=5.0)
        # After at least one submission, _threads is non-empty and
        # every entry must be daemonic.
        assert tracker._executor._threads, "expected at least one worker"
        for t in tracker._executor._threads:
            assert t.daemon, f"worker {t.name} is not daemonic"
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


def test_load_resume_state_restores_opt_state_when_present(
    tmp_path: Path,
) -> None:
    """PR-review #1: when the checkpoint includes `optimizer.safetensors`,
    Adam's moment estimates survive the resume — no cold restart."""
    import equinox as eqx

    from pawn.trainer import flatten_opt_state

    model = init_model(TINY_SUPERNET, key=0)
    opt = optax.adamw(1e-3)
    template = opt.init(eqx.filter(model, eqx.is_inexact_array))
    # Synthesise a "trained" opt_state by injecting non-zero leaves on
    # BOTH the float (mu / nu — Adam moments) and the int (count —
    # Adam step counter) leaves. Round-1 review caught that bumping
    # only floats meant a corrupted `count` would slip through —
    # bias-correction divides by `1 - b1^count`, so a count stuck at
    # zero NaNs every gradient on the first resumed step.
    def _bump(leaf):
        if not hasattr(leaf, "dtype"):
            return leaf
        if leaf.dtype.kind == "f":
            return leaf + 0.5
        if leaf.dtype.kind in ("i", "u"):
            # +42 picks a non-zero, non-default value the round-trip
            # has to preserve. (jnp.int32 + python int → jnp.int32.)
            return leaf + 42
        return leaf
    trained = jax.tree_util.tree_map(_bump, template)

    out_dir = tmp_path / "step_00000010"
    save_model(
        model, out_dir,
        optimizer_state=flatten_opt_state(trained),
        training_state={"step": 10},
    )

    restored = load_resume_state(out_dir, opt, key=jax.random.key(0))
    # The Adam moments (mu / nu) AND the step counter (count) must
    # equal the synthesised trained values, not the fresh-init zeros.
    # Assert both value AND dtype equality leaf-wise — the dtype-bind
    # in `unflatten_opt_state` is the defense against silent int
    # narrowing across the safetensors round-trip.
    trained_leaves = jax.tree_util.tree_leaves(trained)
    restored_leaves = jax.tree_util.tree_leaves(restored.opt_state)
    assert len(trained_leaves) == len(restored_leaves)
    for t, r in zip(trained_leaves, restored_leaves):
        t_arr = jnp.asarray(t)
        r_arr = jnp.asarray(r)
        assert t_arr.dtype == r_arr.dtype, (
            f"dtype mismatch on round-trip: {t_arr.dtype} vs {r_arr.dtype}"
        )
        assert (t_arr == r_arr).all()


def test_load_resume_state_warns_on_missing_opt_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """PR-review #1 follow-up: older checkpoints without
    `optimizer.safetensors` fall back to a fresh init, but the cold
    restart must be loud on stderr so the operator can spot it."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(model, out_dir, training_state={"step": 10})
    opt = optax.adamw(1e-3)
    load_resume_state(out_dir, opt, key=jax.random.key(0))
    err = capsys.readouterr().err
    assert "no optimizer.safetensors" in err
    assert "cold-start" in err
