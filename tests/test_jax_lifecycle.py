"""Tests for `pawn.lifecycle` — SIGTERM, HF push, --resume helpers."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
import unittest.mock as mock
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from pawn.checkpoint import save_model
from pawn.config import TINY_SUPERNET
from pawn.lifecycle import (
    HFBucketTracker,
    HFPushTracker,
    SCHEDULES_THAT_REACH_ZERO,
    _reset_shutdown_state_for_tests,
    build_training_state,
    deserialize_jax_key,
    deserialize_numpy_rng,
    drain_bucket_queue,
    drain_push_queue,
    find_best_step,
    install_sigterm_handler,
    load_resume_state,
    push_checkpoint_async,
    push_to_bucket_async,
    read_resume_best_checkpoint,
    read_resume_data_anchor,
    read_resume_rng_blocks,
    serialize_jax_key,
    serialize_numpy_rng,
    truncate_metrics_jsonl,
    write_schedule_health,
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
        # The checkpoint lands under `checkpoints/<step>` on the per-run
        # branch (v1 `push_checkpoint_to_hf` layout — the published metrics
        # view sits at the branch root, the checkpoints in a subtree).
        assert call["path_in_repo"] == "checkpoints/step_00000010"
    finally:
        tracker.shutdown()


class _RecordingHfApi:
    """Test double recording both `upload_folder` and `upload_file` plus
    `create_branch` — for the metrics-co-upload + branch-creation paths."""

    def __init__(self) -> None:
        # Class-level call logs so the helper can be passed as the
        # `upload_cls` (which `push_checkpoint_async` instantiates).
        pass

    folder_calls: list[dict] = []
    file_calls: list[dict] = []
    branch_calls: list[dict] = []

    def create_branch(self, **kwargs: Any) -> None:
        _RecordingHfApi.branch_calls.append(kwargs)

    def upload_folder(self, **kwargs: Any) -> None:
        _RecordingHfApi.folder_calls.append(kwargs)

    def upload_file(self, **kwargs: Any) -> None:
        # Record the *content* of the uploaded metrics file so the test
        # can assert truncation, not just that an upload happened.
        rec = dict(kwargs)
        path = kwargs.get("path_or_fileobj")
        if isinstance(path, str) and Path(path).exists():
            rec["_content"] = Path(path).read_text(encoding="utf-8")
        _RecordingHfApi.file_calls.append(rec)


def test_push_checkpoint_async_creates_branch_and_couploads_metrics(
    tmp_path: Path,
) -> None:
    """The HF push creates the per-run branch (exist_ok) and co-uploads a
    `metrics.jsonl` truncated to the checkpoint's step, to the branch root.

    Pins ckpt-hf-branch-regression (per-run branch + create_branch) and
    ckpt-hf-metrics-not-pushed (metrics co-upload) together.
    """
    _RecordingHfApi.folder_calls = []
    _RecordingHfApi.file_calls = []
    _RecordingHfApi.branch_calls = []
    ckpt = tmp_path / "step_00000100"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"x")
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"type": "train", "step": 50, "loss": 1.0}\n'
        '{"type": "val", "step": 100, "val/loss": 0.9}\n'
        '{"type": "train", "step": 150, "loss": 0.8}\n',  # beyond step 100
        encoding="utf-8",
    )
    tracker = HFPushTracker(repo_id="ns/test", branch="run/slug-1")
    try:
        push_checkpoint_async(
            ckpt, tracker,
            metrics_path=metrics, step=100,
            upload_cls=_RecordingHfApi,
        )
        timeouts, errors = drain_push_queue(tracker, timeout=5.0)
        assert (timeouts, errors) == (0, 0)
        # Branch created on the per-run isolation branch.
        assert len(_RecordingHfApi.branch_calls) == 1
        assert _RecordingHfApi.branch_calls[0]["branch"] == "run/slug-1"
        assert _RecordingHfApi.branch_calls[0]["exist_ok"] is True
        # Checkpoint uploaded under checkpoints/<step> on that branch.
        assert len(_RecordingHfApi.folder_calls) == 1
        fc = _RecordingHfApi.folder_calls[0]
        assert fc["revision"] == "run/slug-1"
        assert fc["path_in_repo"] == "checkpoints/step_00000100"
        # metrics.jsonl co-uploaded to the branch root, truncated at step 100
        # (the step-150 train record is dropped).
        assert len(_RecordingHfApi.file_calls) == 1
        mc = _RecordingHfApi.file_calls[0]
        assert mc["path_in_repo"] == "metrics.jsonl"
        assert mc["revision"] == "run/slug-1"
        content = mc["_content"]
        assert '"step": 50' in content
        assert '"step": 100' in content
        assert '"step": 150' not in content
    finally:
        tracker.shutdown()


def test_push_checkpoint_async_step_defaults_to_dir_name(
    tmp_path: Path,
) -> None:
    """When `step` is omitted, it's parsed from the `step_NNNN` dir name so
    the metrics truncation boundary is still correct."""
    _RecordingHfApi.folder_calls = []
    _RecordingHfApi.file_calls = []
    _RecordingHfApi.branch_calls = []
    ckpt = tmp_path / "adapter_step_00000042"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"x")
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"type": "val", "step": 42, "val_loss": 0.5}\n'
        '{"type": "val", "step": 99, "val_loss": 0.4}\n',
        encoding="utf-8",
    )
    tracker = HFPushTracker(repo_id="ns/test", branch="run/x")
    try:
        push_checkpoint_async(
            ckpt, tracker, metrics_path=metrics, upload_cls=_RecordingHfApi
        )
        drain_push_queue(tracker, timeout=5.0)
        assert _RecordingHfApi.folder_calls[0]["path_in_repo"] == (
            "checkpoints/adapter_step_00000042"
        )
        content = _RecordingHfApi.file_calls[0]["_content"]
        assert '"step": 42' in content
        assert '"step": 99' not in content  # truncated at the parsed step 42
    finally:
        tracker.shutdown()


def test_truncate_metrics_jsonl_inclusive_boundary(tmp_path: Path) -> None:
    """`truncate_metrics_jsonl` keeps train+val pairs at the boundary step
    and stops before the first record beyond it; malformed / config lines
    pass through (v1 parity)."""
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"type": "config", "x": 1}\n'
        '{"type": "train", "step": 100, "loss": 1.0}\n'
        '{"type": "val", "step": 100, "val/loss": 0.9}\n'
        'not json\n'
        '{"type": "train", "step": 200, "loss": 0.5}\n',
        encoding="utf-8",
    )
    out = truncate_metrics_jsonl(metrics, 100)
    assert '"type": "config"' in out
    assert '"step": 100' in out
    # Malformed lines pass through verbatim (they don't gate the boundary);
    # the `not json` line sits before the first record beyond step 100, so
    # it's retained, while the step-200 record stops the scan.
    assert "not json" in out
    assert '"step": 200' not in out


# ---------------------------------------------------------------------------
# HFBucketTracker
# ---------------------------------------------------------------------------


def test_push_to_bucket_async_syncs_with_run_slug_layout(
    tmp_path: Path,
) -> None:
    """The bucket push enqueues a sync carrying the bucket, run_slug, the
    checkpoint dir, the metrics path, and the parsed step. This is the
    blocker fix: a bucket-targeted run actually pushes something.
    """
    calls: list[dict] = []

    def _fake_sync(
        ckpt_dir: Path, bucket: str, *, run_slug: str,
        metrics_path: Path | None, step: int,
    ) -> None:
        calls.append({
            "ckpt_dir": ckpt_dir, "bucket": bucket, "run_slug": run_slug,
            "metrics_path": metrics_path, "step": step,
        })

    ckpt = tmp_path / "step_00000010"
    ckpt.mkdir()
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("{}\n", encoding="utf-8")
    tracker = HFBucketTracker(bucket="ns/bkt", run_slug="my-slug")
    try:
        push_to_bucket_async(
            ckpt, tracker, metrics_path=metrics, sync_fn=_fake_sync
        )
        timeouts, errors = drain_bucket_queue(tracker, timeout=5.0)
        assert (timeouts, errors) == (0, 0)
        assert len(calls) == 1
        assert calls[0]["bucket"] == "ns/bkt"
        assert calls[0]["run_slug"] == "my-slug"
        assert calls[0]["step"] == 10  # parsed from step_00000010
        assert calls[0]["metrics_path"] == metrics
    finally:
        tracker.shutdown()


def test_push_to_bucket_async_failures_dont_raise(tmp_path: Path) -> None:
    """A failing bucket sync is counted as an error, not raised — training
    keeps going (same posture as the HF-repo path)."""

    def _failing_sync(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("hf sync exploded")

    ckpt = tmp_path / "step_00000010"
    ckpt.mkdir()
    tracker = HFBucketTracker(bucket="ns/bkt", run_slug="s")
    try:
        push_to_bucket_async(ckpt, tracker, sync_fn=_failing_sync)
        timeouts, errors = drain_bucket_queue(tracker, timeout=5.0)
        assert (timeouts, errors) == (0, 1)
    finally:
        tracker.shutdown()


def test_push_checkpoint_to_bucket_builds_url_and_runs_hf_sync(
    tmp_path: Path,
) -> None:
    """`push_checkpoint_to_bucket` stages the checkpoint + truncated metrics
    under the run-slug subtree and shells out to `hf sync <staging>
    <bucket-url>/logs/<run_slug>` (v1 bucket layout)."""
    ckpt = tmp_path / "step_00000005"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"x")
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"type": "val", "step": 5, "val_loss": 1.0}\n'
        '{"type": "val", "step": 9, "val_loss": 0.5}\n',
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    class _Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def _fake_run(cmd: list[str], **_kw: Any) -> "_Result":
        captured["cmd"] = cmd
        # Inspect the staged tree the CLI would sync.
        staging = Path(cmd[2])
        ckpt_dst = staging / "checkpoints" / "step_00000005"
        captured["ckpt_staged"] = ckpt_dst.exists()
        captured["metrics_content"] = (
            (staging / "metrics.jsonl").read_text(encoding="utf-8")
        )
        return _Result()

    from pawn.lifecycle import push_checkpoint_to_bucket

    with mock.patch("subprocess.run", _fake_run):
        push_checkpoint_to_bucket(
            ckpt, "ns/bkt", run_slug="my-slug", metrics_path=metrics, step=5
        )
    cmd = captured["cmd"]
    assert cmd[0:2] == ["hf", "sync"]
    assert cmd[3] == "hf://buckets/ns/bkt/logs/my-slug"
    assert captured["ckpt_staged"] is True
    # Metrics truncated at step 5 — the step-9 record is dropped.
    assert '"step": 5' in captured["metrics_content"]
    assert '"step": 9' not in captured["metrics_content"]


def test_push_checkpoint_to_bucket_raises_on_nonzero_exit() -> None:
    """A non-zero `hf sync` exit surfaces as a RuntimeError so the tracker
    counts it as a failed push."""

    class _Result:
        returncode = 1
        stdout = ""
        stderr = "permission denied"

    def _fake_run(cmd: list[str], **_kw: Any) -> "_Result":
        return _Result()

    from pawn.lifecycle import push_checkpoint_to_bucket

    with mock.patch("subprocess.run", _fake_run):
        with pytest.raises(RuntimeError, match="hf sync exited 1"):
            push_checkpoint_to_bucket(
                Path("/nonexistent"), "ns/bkt", run_slug="s"
            )


def test_push_checkpoint_to_bucket_raises_on_auth_signal_despite_exit_0() -> (
    None
):
    """`hf sync` exits 0 even on per-blob 403/401/429 — the wrapper
    re-greps the output and raises so the failure isn't silent (v1 parity)."""

    class _Result:
        returncode = 0
        stdout = "uploading...\nblob foo: 403 Forbidden\n"
        stderr = ""

    def _fake_run(cmd: list[str], **_kw: Any) -> "_Result":
        return _Result()

    from pawn.lifecycle import push_checkpoint_to_bucket

    with mock.patch("subprocess.run", _fake_run):
        with pytest.raises(RuntimeError, match="auth / quota / rate-limit"):
            push_checkpoint_to_bucket(
                Path("/nonexistent"), "ns/bkt", run_slug="s"
            )


# ---------------------------------------------------------------------------
# find_best_step (best-checkpoint selection)
# ---------------------------------------------------------------------------


def test_find_best_step_pretrain_schema(tmp_path: Path) -> None:
    """Pretrain val records carry `val/loss`; `find_best_step` returns the
    step with the lowest one (ties to the earliest)."""
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"type": "train", "step": 100, "loss": 5.0}\n'
        '{"type": "val", "step": 100, "val/loss": 2.0}\n'
        '{"type": "val", "step": 200, "val/loss": 1.0}\n'
        '{"type": "val", "step": 300, "val/loss": 1.5}\n',
        encoding="utf-8",
    )
    assert find_best_step(metrics) == 200


def test_find_best_step_adapter_schema(tmp_path: Path) -> None:
    """Adapter val records carry `val_loss`; `find_best_step` reads it too."""
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"type": "val", "step": 10, "val_loss": 0.9}\n'
        '{"type": "val", "step": 20, "val_loss": 0.3}\n'
        '{"type": "val", "step": 30, "val_loss": 0.3}\n',  # tie — earlier wins
        encoding="utf-8",
    )
    assert find_best_step(metrics) == 20


def test_find_best_step_none_when_no_val_records(tmp_path: Path) -> None:
    """No `type=val` record (validation disabled) → None."""
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"type": "train", "step": 1, "loss": 1.0}\n', encoding="utf-8"
    )
    assert find_best_step(metrics) is None
    assert find_best_step(tmp_path / "missing.jsonl") is None


def test_find_best_step_skips_non_finite(tmp_path: Path) -> None:
    """A NaN-sanitised (null) loss is skipped, not treated as -inf."""
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"type": "val", "step": 10, "val/loss": null}\n'
        '{"type": "val", "step": 20, "val/loss": 0.7}\n',
        encoding="utf-8",
    )
    assert find_best_step(metrics) == 20


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


def test_shutdown_drain_failed_removes_workers_from_python_exit_join(
    tmp_path: Path,
) -> None:
    """Round-3 codex P1 + bug-detector Critical: the daemon flag alone
    doesn't prevent `concurrent.futures.thread._python_exit` from
    `Thread.join()`-ing a stuck worker at interpreter shutdown.
    `_python_exit` iterates ``_threads_queues`` unconditionally.
    `shutdown(drain_succeeded=False)` must pop our workers from that
    dict so the atexit hook doesn't see them.
    """
    import concurrent.futures.thread as _cf_thread

    ckpt = tmp_path / "step_00000010"
    ckpt.mkdir()
    tracker = HFPushTracker(repo_id="ns/test")
    blocker = threading.Event()

    class _StuckHfApi:
        def upload_folder(self, **_kwargs: Any) -> None:
            blocker.wait()

    try:
        push_checkpoint_async(ckpt, tracker, upload_cls=_StuckHfApi)
        # Drain times out → drain_succeeded=False path
        timeouts, _errors = tracker.join(timeout=0.2)
        assert timeouts == 1
        worker_threads = list(tracker._executor._threads)
        assert worker_threads, "expected at least one worker"
        # Before shutdown the worker IS in `_threads_queues` (cpython
        # registered it at thread creation).
        assert all(
            t in _cf_thread._threads_queues for t in worker_threads
        ), "worker thread is not registered in _threads_queues"
        tracker.shutdown(drain_succeeded=False)
        # After shutdown the worker is gone from `_threads_queues`,
        # so `_python_exit` won't try to join it.
        for t in worker_threads:
            assert t not in _cf_thread._threads_queues, (
                f"worker {t.name} still in _threads_queues; "
                f"_python_exit would block on it"
            )
    finally:
        blocker.set()


def test_join_treats_cancelled_future_as_neither_timeout_nor_error(
    tmp_path: Path,
) -> None:
    """Round-3 bug-detector Important: a future cancelled before its
    worker runs raises `CancelledError` on `result()`. It's not a
    timeout (no stuck thread) and not an upload failure — `join` must
    skip it cleanly so the trainer doesn't surface a spurious
    checkpoint-failure to the operator.
    """
    ckpt = tmp_path / "step_00000010"
    ckpt.mkdir()
    tracker = HFPushTracker(repo_id="ns/test")
    blocker = threading.Event()

    class _NeverCalled:
        def upload_folder(self, **_kwargs: Any) -> None:
            blocker.wait()

    try:
        push_checkpoint_async(ckpt, tracker, upload_cls=_NeverCalled)
        push_checkpoint_async(ckpt, tracker, upload_cls=_NeverCalled)
        push_checkpoint_async(ckpt, tracker, upload_cls=_NeverCalled)
        # First future is running on the single-worker executor;
        # the next two are queued. Cancel them — `cancel()` returns
        # True for queued futures, False for the running one.
        with tracker._lock:
            futures = list(tracker._futures)
        assert futures[1].cancel(), "queued future #1 must be cancellable"
        assert futures[2].cancel(), "queued future #2 must be cancellable"
        # Release the running future so it completes; `join` then
        # observes 0 timeouts + 0 errors (the cancelled ones are
        # skipped) + 1 normal completion.
        blocker.set()
        timeouts, errors = tracker.join(timeout=5.0)
        assert timeouts == 0
        assert errors == 0, (
            f"cancelled futures should not be counted as errors "
            f"(got {errors})"
        )
    finally:
        blocker.set()


def test_unflatten_opt_state_python_int_template_uses_int32(
    tmp_path: Path,
) -> None:
    """Round-3 test-risk Important + bug-detector follow-up: the
    `_dtype_for` fallback for python-int template leaves must
    normalise to `np.int32` (the JAX default int dtype). Without
    this, `np.asarray(42)`'s Linux-default int64 would survive the
    round-trip, JAX would emit a `UserWarning` truncating to int32,
    and a sufficiently large value would silently wrap.
    """
    import jax.numpy as jnp_local

    from pawn.trainer import unflatten_opt_state

    # Synthesise a tiny opt-state-shaped PyTree with a *python int*
    # template leaf (rather than a JAX int32 ArrayImpl). The Optax
    # state in production doesn't have python-int leaves, but this
    # is the defense-in-depth path the fallback was written for.
    template = {"count": 0, "moment": jnp_local.zeros((4,), dtype=jnp_local.float32)}

    # Simulate a saved checkpoint where the count was serialised as
    # int32 (the value `_dtype_for` should pick).
    flat = {
        "['count']": np.int32(7),
        "['moment']": np.ones((4,), dtype=np.float32),
    }

    from typing import cast
    # `unflatten_opt_state` declares `optax.OptState` (an `Any` alias);
    # narrow to the dict shape we built the template with.
    restored_d = cast(
        "dict[str, jax.Array]", unflatten_opt_state(template, flat)
    )
    assert restored_d["count"].dtype == jnp_local.int32, (
        f"expected int32 restored count, got {restored_d['count'].dtype}"
    )
    assert int(restored_d["count"]) == 7
    assert restored_d["moment"].dtype == jnp_local.float32


def test_unflatten_opt_state_numpy_scalar_template_normalises_to_int32(
    tmp_path: Path,
) -> None:
    """Round-4 bug-detector Important: a `numpy.int64` *scalar*
    template leaf (not a multi-dim array) has `hasattr("dtype")` =
    True, so the round-3 fix's `hasattr` check let int64 fall
    through. JAX then truncated to int32 with a `UserWarning`.
    `_dtype_for` now treats numpy scalars (`np.generic`) the same as
    python scalars and normalises to int32 / float32 / bool_.
    """
    import jax.numpy as jnp_local

    from pawn.trainer import unflatten_opt_state

    # numpy scalar (np.int64) as the template leaf — what
    # `np.asarray(python_int)` produces on Linux.
    template = {
        "count": np.int64(0),
        "moment": jnp_local.zeros((4,), dtype=jnp_local.float32),
    }
    flat = {
        "['count']": np.int32(7),
        "['moment']": np.ones((4,), dtype=np.float32),
    }

    from typing import cast
    restored_d = cast(
        "dict[str, jax.Array]", unflatten_opt_state(template, flat)
    )
    assert restored_d["count"].dtype == jnp_local.int32, (
        f"numpy.int64 scalar template should normalise to int32, "
        f"got {restored_d['count'].dtype}"
    )
    assert int(restored_d["count"]) == 7


def test_unflatten_opt_state_preserves_multidim_array_dtype(
    tmp_path: Path,
) -> None:
    """Round-4 bug-detector follow-up: while numpy scalars get
    normalised, multi-dimensional numpy arrays must preserve their
    declared dtype — that's the user's intent, not a default-width
    artifact.

    Uses `uint8` (a non-default dtype JAX represents natively without
    `JAX_ENABLE_X64`) so the test demonstrates the array-dtype-
    preservation path independent of JAX's int64 / float64 truncation.
    """
    import jax.numpy as jnp_local

    from pawn.trainer import unflatten_opt_state

    template = {
        "byte_counter": np.zeros((3,), dtype=np.uint8),
        "moment": jnp_local.zeros((4,), dtype=jnp_local.float32),
    }
    flat = {
        "['byte_counter']": np.array([1, 2, 3], dtype=np.uint8),
        "['moment']": np.ones((4,), dtype=np.float32),
    }

    from typing import cast
    restored_d = cast(
        "dict[str, jax.Array]", unflatten_opt_state(template, flat)
    )
    assert restored_d["byte_counter"].dtype == jnp_local.uint8, (
        f"multi-dim numpy array should keep its declared dtype, "
        f"got {restored_d['byte_counter'].dtype}"
    )


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


# Note: the end-to-end subprocess SIGTERM test (kill a real
# `scripts/train_jax.py` mid-run, assert exit 0 + a `.complete` checkpoint)
# lives in `tests/scripts/test_train_jax_smoke.py::
# test_train_jax_sigterm_saves_final_checkpoint` — it drives the actual
# training loop's graceful save/push/exit dance, which is the faithful
# realization of the `ckpt-sigterm-no-subprocess-test` item. The tests above
# pin the handler's in-process contract (flag flip + idempotent callback).


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


def test_load_resume_state_accepts_matching_conditioning(tmp_path: Path) -> None:
    """Phase-A spec Chunk 4: resuming with the same conditioning the
    checkpoint was trained under passes the load-time C guard."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(
        model, out_dir, training_state={"step": 10},
        run_config={"conditioning": ["outcome"]},
    )
    opt = optax.adamw(1e-3)
    state = load_resume_state(
        out_dir, opt, key=jax.random.key(0), conditioning=["outcome"],
    )
    assert int(state.step) == 10


def test_load_resume_state_rejects_mismatched_conditioning(tmp_path: Path) -> None:
    """Phase-A spec Chunk 4: resuming a C=2 checkpoint with a C=1 run
    (default empty conditioning) must fail loudly rather than silently
    shift every move's absolute RoPE offset."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(
        model, out_dir, training_state={"step": 10},
        run_config={"conditioning": ["outcome"]},
    )
    opt = optax.adamw(1e-3)
    with pytest.raises(ValueError, match="conditioning mismatch"):
        load_resume_state(
            out_dir, opt, key=jax.random.key(0), conditioning=[],
        )


def test_load_resume_state_skips_guard_when_conditioning_none(tmp_path: Path) -> None:
    """When the caller does not pass `conditioning` the guard is a no-op
    (preserves the bare-resume call path for tooling that has no run
    config to cross-check)."""
    model = init_model(TINY_SUPERNET, key=0)
    out_dir = tmp_path / "step_00000010"
    save_model(
        model, out_dir, training_state={"step": 10},
        run_config={"conditioning": ["outcome"]},
    )
    opt = optax.adamw(1e-3)
    # No `conditioning=` kwarg → no cross-check even though the
    # checkpoint is C=2.
    state = load_resume_state(out_dir, opt, key=jax.random.key(0))
    assert int(state.step) == 10


# ---------------------------------------------------------------------------
# H7 — schedule_health.json (observability contract)
# ---------------------------------------------------------------------------


def test_write_schedule_health_records_all_fields(tmp_path: Path) -> None:
    """The helper writes every field the contract pins and returns the
    same dict it serialised."""
    health = write_schedule_health(
        tmp_path,
        schedule="cosine",
        planned_total_steps=1000,
        actual_total_steps=1000,
        lr_peak=3e-4,
        actual_final_lr=0.0,
        reason_for_stop="completed",
    )
    on_disk = json.loads((tmp_path / "schedule_health.json").read_text())
    assert on_disk == health
    for key in (
        "format_version", "schedule", "should_reach_zero",
        "planned_total_steps", "actual_total_steps", "completion_ratio",
        "lr_peak", "actual_final_lr", "reason_for_stop",
    ):
        assert key in health
    assert health["should_reach_zero"] is True  # cosine reaches zero
    assert health["reason_for_stop"] == "completed"
    assert health["completion_ratio"] == 1.0


def test_write_schedule_health_constant_schedule_not_zero_reaching() -> None:
    """`constant` is not in the zero-reaching set, so a step mismatch on
    it never triggers the structural-bug banner."""
    assert "constant" not in SCHEDULES_THAT_REACH_ZERO
    assert "cosine" in SCHEDULES_THAT_REACH_ZERO


def test_write_schedule_health_banner_on_structural_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """`actual != planned` AND a zero-reaching schedule AND a
    `completed`/`step_limit` reason prints the red banner (structural-bug
    signal). SIGTERM with the same mismatch stays silent."""
    write_schedule_health(
        tmp_path, schedule="cosine",
        planned_total_steps=1000, actual_total_steps=500,
        lr_peak=3e-4, actual_final_lr=1e-4, reason_for_stop="completed",
    )
    err = capsys.readouterr().err
    assert "did not run to completion" in err

    write_schedule_health(
        tmp_path, schedule="cosine",
        planned_total_steps=1000, actual_total_steps=500,
        lr_peak=3e-4, actual_final_lr=1e-4, reason_for_stop="sigterm",
    )
    err = capsys.readouterr().err
    assert "did not run to completion" not in err


# ---------------------------------------------------------------------------
# H7 — RNG + scheduler persistence
# ---------------------------------------------------------------------------


def test_serialize_jax_key_roundtrip() -> None:
    """A JAX PRNG key survives the JSON-safe round-trip bit-exactly and
    draws the same downstream randomness."""
    key = jax.random.key(12345)
    block = serialize_jax_key(key)
    # JSON-safe: dumps without error.
    assert json.loads(json.dumps(block)) == block
    restored = deserialize_jax_key(block)
    assert bool(
        (jax.random.key_data(restored) == jax.random.key_data(key)).all()
    )
    # Equivalent randomness.
    a = jax.random.normal(key, (8,))
    b = jax.random.normal(restored, (8,))
    assert jnp.allclose(a, b)


def test_serialize_numpy_rng_roundtrip() -> None:
    """A numpy Generator's bit-generator state round-trips so a resumed
    data stream continues the *same* sequence."""
    rng = np.random.default_rng(7)
    rng.integers(0, 100, size=10)  # advance the state
    block = serialize_numpy_rng(rng)
    assert json.loads(json.dumps(block)) == block
    restored = deserialize_numpy_rng(block)
    # Both generators must now produce identical draws.
    assert (
        rng.integers(0, 1000, size=20) == restored.integers(0, 1000, size=20)
    ).all()


def test_build_training_state_assembles_blocks() -> None:
    rng = np.random.default_rng(3)
    state = build_training_state(
        step=42, schedule="wsd", lr_peak=1e-3,
        rng_key=jax.random.key(1), numpy_rngs={"train": rng},
    )
    assert state["step"] == 42
    assert state["scheduler"] == {"schedule": "wsd", "lr_peak": 1e-3}
    assert state["rng_key"]["encoding"] == "jax_key_data_uint32_b64"
    assert "train" in state["numpy_rngs"]
    # Whole payload must be JSON-serialisable (checkpoint writes it).
    json.dumps(state)


def test_read_resume_rng_blocks_reads_persisted_state(tmp_path: Path) -> None:
    """`read_resume_rng_blocks` recovers the JAX key + numpy generators
    that `build_training_state` persisted."""
    key = jax.random.key(99)
    train_rng = np.random.default_rng(11)
    train_rng.integers(0, 10, size=5)
    state = build_training_state(
        step=10, rng_key=key, numpy_rngs={"train": train_rng},
    )
    out_dir = tmp_path / "step_00000010"
    save_model(
        init_model(TINY_SUPERNET, key=0), out_dir, training_state=state,
    )
    jax_key, np_rngs = read_resume_rng_blocks(out_dir)
    assert jax_key is not None
    assert bool(
        (jax.random.key_data(jax_key) == jax.random.key_data(key)).all()
    )
    assert "train" in np_rngs
    assert (
        train_rng.integers(0, 1000, size=8)
        == np_rngs["train"].integers(0, 1000, size=8)
    ).all()


def test_read_resume_rng_blocks_empty_when_absent(tmp_path: Path) -> None:
    """Older checkpoints without RNG blocks yield (None, {})."""
    out_dir = tmp_path / "step_00000010"
    save_model(
        init_model(TINY_SUPERNET, key=0), out_dir, training_state={"step": 10},
    )
    jax_key, np_rngs = read_resume_rng_blocks(out_dir)
    assert jax_key is None
    assert np_rngs == {}


def test_read_resume_data_anchor_round_trips(tmp_path: Path) -> None:
    """`read_resume_data_anchor` recovers the pretrain prefetcher consume
    anchor (chunk index + intra-chunk batch offset) persisted under the
    `data_anchor` block (H7 / D2)."""
    state = build_training_state(
        step=10, extra={
            "data_anchor": {
                "base_seed": 0, "chunk_index": 3, "batch_offset": 2,
            }
        },
    )
    out_dir = tmp_path / "step_00000010"
    save_model(
        init_model(TINY_SUPERNET, key=0), out_dir, training_state=state,
    )
    chunk_index, batch_offset = read_resume_data_anchor(out_dir)
    assert chunk_index == 3
    assert batch_offset == 2


def test_read_resume_data_anchor_falls_back_to_zero(tmp_path: Path) -> None:
    """Checkpoints without a `data_anchor` block (older / look-ahead-rng
    predecessors) yield (0, 0) so the data stream restarts from the
    beginning rather than skipping batches."""
    out_dir = tmp_path / "step_00000010"
    save_model(
        init_model(TINY_SUPERNET, key=0), out_dir, training_state={"step": 10},
    )
    assert read_resume_data_anchor(out_dir) == (0, 0)


def test_load_resume_state_restores_persisted_jax_key(tmp_path: Path) -> None:
    """When `training_state.json` carries `rng_key`, `load_resume_state`
    restores it instead of using the caller-supplied key."""
    saved_key = jax.random.key(54321)
    state_dict = build_training_state(step=10, rng_key=saved_key)
    out_dir = tmp_path / "step_00000010"
    save_model(init_model(TINY_SUPERNET, key=0), out_dir,
               training_state=state_dict)
    opt = optax.adamw(1e-3)
    # Pass a *different* key; the persisted one must win.
    state = load_resume_state(out_dir, opt, key=jax.random.key(0))
    assert bool(
        (jax.random.key_data(state.key) == jax.random.key_data(saved_key)).all()
    )


# ---------------------------------------------------------------------------
# Best-checkpoint / early-stop anchor resume round-trip
# (v1 CLMTrainer.load_state parity — patience must survive --resume)
# ---------------------------------------------------------------------------


def test_read_resume_best_checkpoint_round_trips(tmp_path: Path) -> None:
    """The `best_checkpoint` block `_save_checkpoint` persists (best val
    loss, the step that achieved it, the best late-game legality, and the
    running patience counter) is recovered intact by
    `read_resume_best_checkpoint`.

    This is the v1 `CLMTrainer.load_state` parity guard: a `--patience N`
    run SIGTERM-paused near convergence and resumed must keep its
    no-improvement budget instead of resetting it to 0, which would defeat
    early stopping indefinitely across chunked / preemptible-pod runs.

    `best_late_legality` is the *second* compound early-stop leg
    (main:pawn/trainer.py:1319/1345-1346). It must survive the round trip:
    if it reset to 0.0, the first post-resume val (essentially always
    carrying a positive late legality) would spuriously beat the reset best,
    register as an improvement, and clobber the restored patience_counter
    back to 0 — defeating the loss-leg restoration via the legality leg.
    """
    # The block shape mirrors `scripts/train_jax.py:_save_checkpoint`.
    state = build_training_state(
        step=500,
        extra={
            "best_checkpoint": {
                "best_val_loss": 1.2345,
                "best_val_step": 300,
                "best_late_legality": 0.9876,
                "patience_counter": 2,
            }
        },
    )
    out_dir = tmp_path / "step_00000500"
    save_model(init_model(TINY_SUPERNET, key=0), out_dir, training_state=state)
    best_val_loss, best_val_step, best_late_legality, patience_counter = (
        read_resume_best_checkpoint(out_dir)
    )
    assert best_val_loss == pytest.approx(1.2345)
    assert best_val_step == 300
    assert best_late_legality == pytest.approx(0.9876)
    assert patience_counter == 2


def test_read_resume_best_checkpoint_null_best_maps_to_sentinels(
    tmp_path: Path,
) -> None:
    """A checkpoint saved before any val record (null persisted best) maps
    back to the trainer's inf / -1 / 0.0 init sentinels so a fresh resume is
    indistinguishable from a cold start — but a non-zero patience counter
    still survives."""
    state = build_training_state(
        step=10,
        extra={
            "best_checkpoint": {
                "best_val_loss": None,
                "best_val_step": None,
                "best_late_legality": None,
                "patience_counter": 0,
            }
        },
    )
    out_dir = tmp_path / "step_00000010"
    save_model(init_model(TINY_SUPERNET, key=0), out_dir, training_state=state)
    best_val_loss, best_val_step, best_late_legality, patience_counter = (
        read_resume_best_checkpoint(out_dir)
    )
    assert best_val_loss == float("inf")
    assert best_val_step == -1
    assert best_late_legality == 0.0
    assert patience_counter == 0


def test_read_resume_best_checkpoint_falls_back_when_absent(
    tmp_path: Path,
) -> None:
    """Older checkpoints written before the `best_checkpoint` anchor existed
    yield (inf, -1, 0.0, 0) so the patience clock starts from scratch — the
    safe, documented fallback rather than a crash."""
    out_dir = tmp_path / "step_00000010"
    save_model(
        init_model(TINY_SUPERNET, key=0), out_dir, training_state={"step": 10},
    )
    assert read_resume_best_checkpoint(out_dir) == (float("inf"), -1, 0.0, 0)
