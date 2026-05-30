"""Training-run lifecycle helpers: SIGTERM, HF push, --resume support.

Three operational concerns covered:

- :func:`install_sigterm_handler` — installs a SIGTERM handler that
  finishes the current chunk, drains the HF push queue, and exits 0.
  Acceptance criterion 17.

- :func:`push_checkpoint_async` — fire-and-forget upload of a
  ``step_<N>`` directory to a HuggingFace repo branch via
  :mod:`huggingface_hub`. Failures don't block training (the trainer
  keeps going); successes flush before SIGTERM. Acceptance criterion 18.

- :func:`load_resume_state` — load a :class:`TrainState` from a
  checkpoint directory, splicing ``state.step`` from the saved value
  so the metrics log stays monotonic across the resume. Acceptance
  criterion 16.

These are library functions the training scripts in S13 call; the
CLI plumbing (`--hf-repo` / `--resume` flags + the SIGTERM signal
install) is the script's job.
"""

from __future__ import annotations

import base64
import json
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import (
    CancelledError,
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pawn.checkpoint import load_model, load_model_config
from pawn.model import PAWNModel
from pawn.trainer import TrainState

__all__ = [
    "HFPushTracker",
    "push_checkpoint_async",
    "install_sigterm_handler",
    "drain_push_queue",
    "load_resume_state",
    "write_schedule_health",
    "SCHEDULES_THAT_REACH_ZERO",
    "SCHEDULE_HEALTH_FILE",
    "serialize_jax_key",
    "deserialize_jax_key",
    "serialize_numpy_rng",
    "deserialize_numpy_rng",
    "build_training_state",
    "read_resume_rng_blocks",
    "read_resume_data_anchor",
]


# ---------------------------------------------------------------------------
# HF push — async, fire-and-forget, joinable at SIGTERM
# ---------------------------------------------------------------------------


class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """`ThreadPoolExecutor` whose worker threads are daemonic.

    The stock executor uses non-daemon worker threads, which the
    interpreter joins at exit via
    ``concurrent.futures.thread._python_exit``. A stuck HF upload
    thread therefore keeps the process alive past ``main()`` returning
    — the 300s SIGTERM budget in ``drain_push_queue`` only bounds the
    *drain* call, not interpreter exit.

    Daemonic threads are killed at interpreter shutdown. That is the
    correct behavior here: by the time we abandon a future
    (``drain_succeeded=False``), the upload is past its budget and the
    operator wants the process to exit. Marking the worker daemonic
    moves the abandon point from "hang forever" to "kill on exit".

    `daemon` can only be set before `Thread.start()` — so this needs
    to override `_adjust_thread_count` entirely (cpython's
    implementation constructs and starts the thread atomically with
    ``daemon=False``). We mirror the cpython internals (verified
    against `concurrent.futures.thread._adjust_thread_count` in
    Python 3.12) — keep this in sync if a future Python release
    rearranges the worker-creation path.
    """

    def _adjust_thread_count(self) -> None:
        # CPython implementation detail — `_threads_queues` and
        # `_worker` live in `concurrent.futures.thread`. Import here so
        # the dependency is contained to this override.
        import concurrent.futures.thread as _cf_thread
        import weakref
        from typing import cast

        # If a thread is already idle and waiting for work, no need to
        # spin a new one. Mirror the cpython early-return.
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):  # noqa: ANN001
            # `q.put(None)` is how the worker is signalled to exit. The
            # cpython stub types `_work_queue.put` as `put(item: _WorkItem)`
            # but `None` is the sentinel — cast to satisfy pyright.
            cast(Any, q).put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (
                self._thread_name_prefix or self, num_threads
            )
            t = threading.Thread(
                name=thread_name,
                target=_cf_thread._worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
                daemon=True,
            )
            t.start()
            # The cpython stubs type `_threads` as `AbstractSet[Thread]`
            # and `_threads_queues` as `Mapping[Any, Any]`. At runtime
            # both are mutable (`set` and `WeakKeyDictionary`); cast
            # through `Any` so the mutation is type-clean.
            cast(Any, self._threads).add(t)
            cast(Any, _cf_thread._threads_queues)[t] = self._work_queue


@dataclass
class HFPushTracker:
    """Tracks the in-flight HuggingFace upload futures.

    The training loop calls :func:`push_checkpoint_async(ckpt_dir,
    hf_repo, tracker)` after each save; the SIGTERM handler calls
    :meth:`join` to wait for all pending uploads before exit.

    `_executor` is a 1-worker pool — serialise uploads so a slow
    network doesn't queue up gigabytes of pending payloads.
    """

    repo_id: str
    branch: str = "main"
    _executor: ThreadPoolExecutor = field(
        default_factory=lambda: _DaemonThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hf-push"
        )
    )
    _futures: list[Future[None]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def submit(self, fn: Callable[[], None]) -> None:
        with self._lock:
            self._futures.append(self._executor.submit(fn))

    def join(self, *, timeout: float | None = None) -> tuple[int, int]:
        """Wait for all pending uploads. Returns ``(timeouts, errors)``.

        ``timeouts`` counts futures we couldn't wait for before
        ``timeout`` elapsed — those represent threads still running.
        ``errors`` counts uploads that raised inside the worker (and
        therefore released the thread normally).

        The two are reported separately because they have different
        SIGTERM consequences:
        - ``timeouts > 0`` means a worker is still alive; the trainer
          must take the abandon-thread path at ``shutdown`` time
          (``drain_succeeded=False``) so the daemon worker is killed
          at interpreter exit instead of joining indefinitely.
        - ``errors > 0`` means uploads failed but threads exited; the
          trainer can ``shutdown(drain_succeeded=True)`` safely.

        Failures don't raise — training already finished, the goal is
        best-effort cleanup before exit.
        """
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        with self._lock:
            futures = list(self._futures)
            self._futures.clear()
        timeouts = 0
        errors = 0
        for fut in futures:
            try:
                remaining = (
                    deadline - time.monotonic() if deadline is not None else None
                )
                if remaining is not None and remaining < 0:
                    fut.cancel()
                    timeouts += 1
                    continue
                fut.result(timeout=remaining)
            except FutureTimeoutError:
                # `fut.result(timeout=...)` raises this if the worker
                # didn't finish in `remaining`. A still-running thread.
                # Import explicitly from `concurrent.futures` rather
                # than relying on the 3.11+ alias with builtin
                # `TimeoutError` (round-3 bug-detector Important).
                timeouts += 1
            except CancelledError:
                # Future was cancelled before the worker started —
                # not a stuck thread, not a failed upload. Treat as
                # neither timeout nor error (round-3 bug-detector
                # Important: a `cancel_futures=True` shutdown between
                # `submit()` and `result()` would have triggered
                # this, and conflating it with `errors` would surface
                # a spurious checkpoint-failure to the operator).
                continue
            except Exception:
                # The upload raised — thread exited, just a failed
                # checkpoint. No abandon-thread needed.
                errors += 1
        return timeouts, errors

    def shutdown(self, *, drain_succeeded: bool = True) -> None:
        """Drain the executor.

        Two-state contract:

        - ``drain_succeeded=True`` (caller verified every prior upload
          completed within budget): ``wait=True`` so running uploads
          finish cleanly. ``cancel_futures=True`` drops queued-but-not-
          started uploads so we don't block on a backlog.

        - ``drain_succeeded=False`` (the prior ``drain_push_queue``
          timed out — a future is stuck mid-upload and ``cancel()`` is
          a no-op on a running thread): ``wait=False`` so the trainer
          can exit promptly. We also **remove the worker thread from
          ``concurrent.futures.thread._threads_queues``** so cpython's
          ``_python_exit`` atexit hook doesn't ``Thread.join()`` it
          unconditionally (round-3 codex P1 + bug-detector Critical:
          the daemon flag alone doesn't help — ``_python_exit`` runs
          before daemon-thread-kill, and a ``join()`` on a stuck
          worker blocks on the GIL-internal ``_tstate_lock``).

        Without this gating, a stuck upload makes the trainer hang
        forever past the bounded SIGTERM budget.
        """
        if drain_succeeded:
            self._executor.shutdown(wait=True, cancel_futures=True)
            return

        # Abandon path: pop our workers from cpython's atexit join
        # list, *then* shut down the executor. After this returns,
        # the daemon worker is left running but the interpreter will
        # exit normally — `_python_exit` no longer sees the worker.
        import concurrent.futures.thread as _cf_thread
        from typing import cast as _cast

        threads_queues = _cast(Any, _cf_thread._threads_queues)
        for t in list(self._executor._threads):
            threads_queues.pop(t, None)
        self._executor.shutdown(wait=False, cancel_futures=True)


def push_checkpoint_async(
    ckpt_dir: Path,
    tracker: HFPushTracker,
    *,
    revision: str | None = None,
    upload_cls: Any | None = None,
) -> None:
    """Enqueue an async upload of ``ckpt_dir`` to ``tracker.repo_id``.

    ``upload_cls`` is a test seam — defaults to the real
    :class:`huggingface_hub.HfApi` upload path. Tests can pass a
    mock that records calls without hitting the network.
    """
    if upload_cls is None:
        try:
            from huggingface_hub import HfApi

            upload_cls = HfApi
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is required for HF push (install via "
                "`pip install huggingface_hub`)"
            ) from e

    target_branch = revision if revision is not None else tracker.branch

    def _do_upload() -> None:
        # The actual upload — runs in the executor thread.
        api = upload_cls()
        api.upload_folder(
            repo_id=tracker.repo_id,
            folder_path=str(ckpt_dir),
            path_in_repo=ckpt_dir.name,
            revision=target_branch,
            commit_message=f"Checkpoint {ckpt_dir.name}",
        )

    tracker.submit(_do_upload)


def drain_push_queue(
    tracker: HFPushTracker, *, timeout: float = 300.0
) -> tuple[int, int]:
    """Wait for all in-flight pushes (called from the SIGTERM handler).

    Returns ``(timeouts, errors)`` per :meth:`HFPushTracker.join`.
    """
    return tracker.join(timeout=timeout)


# ---------------------------------------------------------------------------
# SIGTERM handler
# ---------------------------------------------------------------------------


@dataclass
class _ShutdownState:
    """Internal flag the handler flips when SIGTERM fires.

    The training loop polls `should_shutdown()` between K-step chunks
    and exits gracefully on True.
    """

    requested: bool = False


_SHUTDOWN = _ShutdownState()


def install_sigterm_handler(
    on_shutdown: Callable[[], None] | None = None,
) -> Callable[[], bool]:
    """Install a SIGTERM handler. Returns ``should_shutdown()`` — a
    function the training loop polls between chunks.

    The handler:
    1. Sets ``_SHUTDOWN.requested = True``.
    2. Calls ``on_shutdown()`` if provided (sync — should be quick).
    3. The handler does NOT call sys.exit — the training loop is
       responsible for the graceful save/push/exit dance.

    The handler is idempotent — multiple SIGTERMs only fire
    ``on_shutdown`` once.

    Returns a closure ``should_shutdown()`` that returns the requested
    flag. The training loop polls this between chunks.
    """

    def handler(signum: int, frame: Any) -> None:
        del signum, frame
        if _SHUTDOWN.requested:
            return
        _SHUTDOWN.requested = True
        print(
            "[lifecycle] SIGTERM received — finishing current chunk and saving",
            file=sys.stderr,
            flush=True,
        )
        if on_shutdown is not None:
            try:
                on_shutdown()
            except Exception as e:
                print(
                    f"[lifecycle] on_shutdown callback failed: {e}",
                    file=sys.stderr,
                    flush=True,
                )

    signal.signal(signal.SIGTERM, handler)
    return lambda: _SHUTDOWN.requested


def _reset_shutdown_state_for_tests() -> None:
    """Test hook — reset the global between test cases."""
    _SHUTDOWN.requested = False


# ---------------------------------------------------------------------------
# --resume support
# ---------------------------------------------------------------------------


def load_resume_state(
    ckpt_dir: Path | str,
    optimizer: optax.GradientTransformation,
    key: jax.Array,
    conditioning: Sequence[str] | None = None,
) -> TrainState:
    """Build a :class:`TrainState` from a saved checkpoint directory.

    Loads the model via :func:`pawn.checkpoint.load_model`, reads the
    ``training_state.json`` sidecar for the step counter, and returns
    a TrainState ready to continue training. ``state.step`` is spliced
    from the saved value so the metrics log stays monotonic across
    the resume.

    When ``conditioning`` is supplied (the active run's
    ``cfg.conditioning``), the checkpoint's persisted conditioning is
    read from its run block and cross-checked via
    :func:`pawn.corpus.assert_conditioning_C` before returning. This is
    the load-time ``C``-mismatch guard (Phase-A spec Chunk 4): resuming
    a checkpoint trained at one prefix width ``C`` against a run that
    builds its corpus at a different ``C`` would silently shift every
    move to a different absolute offset (RoPE drift). The guard turns
    that into a loud failure at the resume boundary instead.

    If the checkpoint includes ``optimizer.safetensors`` (written by
    :func:`pawn.checkpoint.save_model` when the trainer passed
    ``optimizer_state``), the saved opt-state is restored via
    :func:`pawn.trainer.unflatten_opt_state` — Adam moments + clip
    counters survive the resume intact. If the slot is missing (older
    checkpoints / runs that opted out), fall back to a fresh
    ``optimizer.init(...)`` and emit a warning so the cold restart
    surfaces in logs rather than silently spiking the loss.
    """
    ckpt_dir = Path(ckpt_dir)
    model, run_block = load_model(ckpt_dir)
    # Load-time C-mismatch guard (Phase-A spec Chunk 4): if the active run
    # declares its conditioning, assert it matches the checkpoint's so
    # resumed move positions land at the same absolute RoPE offset.
    if conditioning is not None:
        from pawn.corpus import (
            assert_conditioning_C,
            conditioning_from_run_block,
            conditioning_to_C,
        )

        checkpoint_C = conditioning_to_C(conditioning_from_run_block(run_block))
        assert_conditioning_C(conditioning, checkpoint_C)
    # Splice step from training_state.json if present.
    ts_path = ckpt_dir / "training_state.json"
    ts: dict[str, Any] | None = None
    if ts_path.is_file():
        ts = dict(json.loads(ts_path.read_text(encoding="utf-8")))
        step = int(ts.get("step", 0))
    else:
        # Fall back to parsing the dir name (`step_00050000`).
        name = ckpt_dir.name
        if name.startswith("step_"):
            try:
                step = int(name[len("step_"):])
            except ValueError:
                step = 0
        else:
            step = 0
    import equinox as eqx

    from pawn.checkpoint import OPTIMIZER_FILE
    from pawn.trainer import unflatten_opt_state

    template = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    opt_path = ckpt_dir / OPTIMIZER_FILE
    if opt_path.is_file():
        from safetensors.numpy import load_file as st_load
        flat = st_load(str(opt_path))
        opt_state = unflatten_opt_state(template, flat)
    else:
        # Older checkpoints didn't persist opt_state. The cold restart
        # is loud rather than silent — a multi-hundred-step loss spike
        # mid-cosine schedule is the typical failure mode and a stderr
        # line is the cheapest signal for the operator.
        print(
            f"[pawn.lifecycle] WARNING: no {OPTIMIZER_FILE} in {ckpt_dir}; "
            "resuming with a fresh opt_state (Adam moments will cold-start). "
            "Long pretraining runs may see a loss spike at the resume point.",
            file=sys.stderr,
        )
        opt_state = template
    # RNG restore: prefer the persisted JAX key so the resumed run draws
    # the same downstream randomness it would have without the interruption.
    # Fall back to the caller-supplied `key` for older checkpoints that
    # predate RNG persistence.
    resume_key = key
    if ts is not None:
        saved_key = _read_rng_block(ts, "rng_key")
        if saved_key is not None:
            resume_key = deserialize_jax_key(saved_key)
    return TrainState(
        model=model,
        opt_state=opt_state,
        step=jnp.int32(step),
        key=resume_key,
    )


# ---------------------------------------------------------------------------
# Observability — schedule_health.json (H7 / plan §8.3)
# ---------------------------------------------------------------------------

SCHEDULE_HEALTH_FILE: str = "schedule_health.json"

# LR schedules that are *supposed* to decay all the way to (or near) zero by
# the end of the planned timeline. For these, an `actual != planned` step
# count on a `completed` stop is the structural-bug signal: the loop fell off
# the end without an early-exit, yet the step counts disagree. Constant /
# plateau schedules never reach zero, so the mismatch warning doesn't apply.
SCHEDULES_THAT_REACH_ZERO: tuple[str, ...] = (
    "cosine", "wsd", "infinite", "one_cycle",
)


def write_schedule_health(
    run_dir: Path | str,
    *,
    schedule: str,
    planned_total_steps: int,
    actual_total_steps: int,
    lr_peak: float,
    actual_final_lr: float,
    reason_for_stop: str,
) -> dict[str, Any]:
    """Write ``schedule_health.json`` and return its contents.

    Records whether training reached the planned end of the LR schedule
    (plan §8.3 H7). Written at *both* exit paths (normal completion and
    SIGTERM) so a post-hoc reader can answer "did the LR schedule run to
    completion?" without replaying the run.

    The "loud" path — ``actual_total_steps != planned_total_steps`` on a
    schedule that should reach zero, with ``reason_for_stop`` of
    ``completed`` / ``step_limit`` — also prints a red banner to stderr so
    the structural-bug signal surfaces immediately. With cache-first that
    combination is unreachable; a future regression that brings it back is
    caught here. SIGTERM / patience / pause are legitimate early exits and
    write the file without warning.
    """
    run_dir = Path(run_dir)
    should_reach_zero = schedule in SCHEDULES_THAT_REACH_ZERO
    completion_ratio = (
        actual_total_steps / planned_total_steps
        if planned_total_steps > 0
        else 0.0
    )
    health: dict[str, Any] = {
        "format_version": 1,
        "schedule": schedule,
        "should_reach_zero": should_reach_zero,
        "planned_total_steps": int(planned_total_steps),
        "actual_total_steps": int(actual_total_steps),
        "completion_ratio": completion_ratio,
        "lr_peak": float(lr_peak),
        "actual_final_lr": float(actual_final_lr),
        "reason_for_stop": reason_for_stop,
    }
    (run_dir / SCHEDULE_HEALTH_FILE).write_text(
        json.dumps(health, indent=2) + "\n", encoding="utf-8"
    )

    if (
        actual_total_steps != planned_total_steps
        and should_reach_zero
        and reason_for_stop in ("completed", "step_limit")
    ):
        print(
            "\033[31m"
            f"WARNING: schedule did not run to completion: "
            f"actual_total_steps={actual_total_steps} != "
            f"planned_total_steps={planned_total_steps}. "
            f"Final LR={actual_final_lr:.3e} "
            f"(peak={lr_peak:.3e}). reason={reason_for_stop}.\033[0m",
            file=sys.stderr,
            flush=True,
        )

    return health


# ---------------------------------------------------------------------------
# RNG + scheduler persistence (H7 / plan §8.3)
# ---------------------------------------------------------------------------


def serialize_jax_key(key: jax.Array) -> dict[str, Any]:
    """Serialise a JAX PRNG key into a JSON-safe block.

    Uses :func:`jax.random.key_data` to extract the underlying ``uint32``
    array (independent of the typed-key wrapper) and base64-encodes it so
    it round-trips losslessly through ``training_state.json``. The impl
    name + shape are stored alongside so a future key-impl change surfaces
    as an explicit mismatch rather than silently wrapping the wrong bytes.
    """
    data = np.asarray(jax.random.key_data(key)).astype(np.uint32)
    return {
        "encoding": "jax_key_data_uint32_b64",
        "shape": list(data.shape),
        "data": base64.b64encode(data.tobytes()).decode("ascii"),
    }


def deserialize_jax_key(block: Mapping[str, Any]) -> jax.Array:
    """Rebuild a JAX PRNG key from a :func:`serialize_jax_key` block."""
    if block.get("encoding") != "jax_key_data_uint32_b64":
        raise ValueError(
            f"unrecognised RNG key encoding {block.get('encoding')!r}; "
            "expected 'jax_key_data_uint32_b64'"
        )
    raw = np.frombuffer(
        base64.b64decode(block["data"]), dtype=np.uint32
    ).reshape(tuple(block["shape"]))
    return jax.random.wrap_key_data(jnp.asarray(raw.copy()))


def serialize_numpy_rng(rng: np.random.Generator) -> dict[str, Any]:
    """Serialise a :class:`numpy.random.Generator`'s bit-generator state.

    The adapter data stream samples batch indices from a numpy
    ``default_rng``; persisting its bit-generator state lets a resumed run
    continue the *same* index sequence rather than re-drawing from the seed
    (which would replay already-seen batches). ``json.dumps`` handles the
    nested dict the bit generator returns (ints + a uint array we coerce to
    a list).
    """
    state = rng.bit_generator.state

    def _coerce(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, dict):
            return {k: _coerce(v) for k, v in obj.items()}
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    return {"encoding": "numpy_bit_generator_state", "state": _coerce(state)}


def deserialize_numpy_rng(block: Mapping[str, Any]) -> np.random.Generator:
    """Rebuild a :class:`numpy.random.Generator` from a serialised block."""
    if block.get("encoding") != "numpy_bit_generator_state":
        raise ValueError(
            f"unrecognised numpy RNG encoding {block.get('encoding')!r}; "
            "expected 'numpy_bit_generator_state'"
        )

    def _restore(obj: Any) -> Any:
        if isinstance(obj, dict) and "__ndarray__" in obj:
            return np.asarray(obj["__ndarray__"], dtype=obj["dtype"])
        if isinstance(obj, dict):
            return {k: _restore(v) for k, v in obj.items()}
        return obj

    state = _restore(block["state"])
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return rng


def build_training_state(
    *,
    step: int,
    schedule: str | None = None,
    lr_peak: float | None = None,
    rng_key: jax.Array | None = None,
    numpy_rngs: Mapping[str, np.random.Generator] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ``training_state.json`` payload for a checkpoint.

    Carries the step counter (canonical resume anchor) plus an optional
    ``scheduler`` block (the schedule name + peak LR — the schedule is a
    pure function of cfg+total_steps, so persisting its identity is enough
    to reconstruct it deterministically) and the RNG state needed to make a
    resume bit-reproducible vs an uninterrupted run.
    """
    payload: dict[str, Any] = {"step": int(step)}
    if schedule is not None:
        payload["scheduler"] = {"schedule": schedule, "lr_peak": lr_peak}
    if rng_key is not None:
        payload["rng_key"] = serialize_jax_key(rng_key)
    if numpy_rngs:
        payload["numpy_rngs"] = {
            name: serialize_numpy_rng(rng) for name, rng in numpy_rngs.items()
        }
    if extra:
        payload.update(dict(extra))
    return payload


def _read_rng_block(ts: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    """Pull a serialised RNG block out of a ``training_state.json`` dict."""
    block = ts.get(name)
    if isinstance(block, Mapping):
        return block
    return None


def read_resume_rng_blocks(
    ckpt_dir: Path | str,
) -> tuple[jax.Array | None, dict[str, np.random.Generator]]:
    """Read persisted RNG state from a checkpoint's ``training_state.json``.

    Returns ``(jax_key_or_None, {name: numpy_Generator})``. Used by the
    adapter resume path, which carries numpy data-stream RNGs in addition
    to the model JAX key. Missing blocks (older checkpoints) yield ``None``
    / an empty dict so the caller can fall back to seed-derived RNG.
    """
    ckpt_dir = Path(ckpt_dir)
    ts_path = ckpt_dir / "training_state.json"
    if not ts_path.is_file():
        return None, {}
    ts = json.loads(ts_path.read_text(encoding="utf-8"))
    jax_key = None
    key_block = _read_rng_block(ts, "rng_key")
    if key_block is not None:
        jax_key = deserialize_jax_key(key_block)
    numpy_rngs: dict[str, np.random.Generator] = {}
    raw_np = ts.get("numpy_rngs")
    if isinstance(raw_np, Mapping):
        for name, block in raw_np.items():
            if isinstance(block, Mapping):
                numpy_rngs[name] = deserialize_numpy_rng(block)
    return jax_key, numpy_rngs


def read_resume_data_anchor(ckpt_dir: Path | str) -> tuple[int, int]:
    """Read the pretrain data-stream consume anchor from a checkpoint.

    Returns ``(chunk_index, batch_offset)`` — the outer-chunk index and the
    intra-chunk batch offset of the next batch the interrupted run was about
    to consume. The pretrain prefetcher (``scripts/train_jax.py``) derives
    each outer-chunk seed purely from ``(base_seed, chunk_index)``, so
    re-deriving from this anchor reproduces the exact tail of the
    uninterrupted run's batch sequence (H7 / D2 bit-reproducible resume).

    Missing block (older checkpoints written before the anchor existed, or
    the look-ahead-``numpy_rngs`` predecessor) yields ``(0, 0)`` so the
    caller starts the data stream from the beginning — the safe, documented
    fallback (a cold data-stream restart rather than a silent skip).
    """
    ckpt_dir = Path(ckpt_dir)
    ts_path = ckpt_dir / "training_state.json"
    if not ts_path.is_file():
        return 0, 0
    ts = json.loads(ts_path.read_text(encoding="utf-8"))
    block = ts.get("data_anchor")
    if not isinstance(block, Mapping):
        return 0, 0
    chunk_index = int(block.get("chunk_index", 0))
    batch_offset = int(block.get("batch_offset", 0))
    return chunk_index, batch_offset
