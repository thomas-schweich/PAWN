"""Tests for pawn/eval_suite/worker.py.

Covers run_in_worker (timeout, normal return, exception propagation)
and per-section wrapper functions (via mocks — no real subprocess spawn).

These tests mock multiprocessing so no actual fork happens.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pawn.eval_suite import worker as worker_mod
from pawn.eval_suite.worker import (
    _worker_entry,
    run_diagnostic_eval,
    run_impossible_task_test,
    run_improbable_task_test,
    run_in_worker,
    run_outcome_signal_test,
    run_poisoned_prefix_test,
    run_prefix_continuation_test,
    run_probes,
)


# ---------------------------------------------------------------------------
# _worker_entry
# ---------------------------------------------------------------------------


class TestWorkerEntry:
    @pytest.mark.unit
    def test_invokes_function_with_args_kwargs(self):
        def f(a, b, c=None):
            return (a, b, c)
        result = _worker_entry(f, (1, 2), {"c": 3})
        assert result == (1, 2, 3)

    @pytest.mark.unit
    def test_empty_args_kwargs(self):
        def f():
            return 42
        result = _worker_entry(f, (), {})
        assert result == 42

    @pytest.mark.unit
    def test_propagates_exceptions(self):
        def f():
            raise ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            _worker_entry(f, (), {})


# ---------------------------------------------------------------------------
# run_in_worker — heavily mocked, does not actually spawn processes
# ---------------------------------------------------------------------------


def _mk_pool_mock(return_value=None, raise_exc: Exception | None = None):
    """Build a Pool context-manager mock that returns value or raises."""
    pool = MagicMock()
    async_result = MagicMock()
    if raise_exc is not None:
        async_result.get.side_effect = raise_exc
    else:
        async_result.get.return_value = return_value
    pool.apply_async.return_value = async_result
    pool.__enter__.return_value = pool
    pool.__exit__.return_value = False
    return pool


class TestRunInWorker:
    @pytest.mark.unit
    def test_returns_value_from_fn(self):
        pool = _mk_pool_mock(return_value=123)
        with patch.object(worker_mod._ctx, "Pool", return_value=pool):
            result = run_in_worker(lambda x: x, 5)
        assert result == 123

    @pytest.mark.unit
    def test_passes_timeout_to_get(self):
        pool = _mk_pool_mock(return_value=7)
        with patch.object(worker_mod._ctx, "Pool", return_value=pool):
            run_in_worker(lambda: 0, timeout=30.5)
        # async_result.get should have been called with timeout=30.5
        async_result = pool.apply_async.return_value
        async_result.get.assert_called_once_with(timeout=30.5)

    @pytest.mark.unit
    def test_raises_timeout_and_terminates(self):
        pool = _mk_pool_mock(raise_exc=TimeoutError("timed out"))
        with patch.object(worker_mod._ctx, "Pool", return_value=pool):
            with pytest.raises(TimeoutError):
                run_in_worker(lambda: 0, timeout=0.1)
        pool.terminate.assert_called_once()
        pool.join.assert_called_once()

    @pytest.mark.unit
    def test_raises_keyboard_interrupt_and_terminates(self):
        pool = _mk_pool_mock(raise_exc=KeyboardInterrupt())
        with patch.object(worker_mod._ctx, "Pool", return_value=pool):
            with pytest.raises(KeyboardInterrupt):
                run_in_worker(lambda: 0)
        pool.terminate.assert_called_once()
        pool.join.assert_called_once()

    @pytest.mark.unit
    def test_raises_general_exception_and_terminates(self):
        pool = _mk_pool_mock(raise_exc=RuntimeError("worker failed"))
        with patch.object(worker_mod._ctx, "Pool", return_value=pool):
            with pytest.raises(RuntimeError, match="worker failed"):
                run_in_worker(lambda: 0)
        pool.terminate.assert_called_once()
        pool.join.assert_called_once()

    @pytest.mark.unit
    def test_apply_async_called_with_worker_entry(self):
        pool = _mk_pool_mock(return_value=42)
        with patch.object(worker_mod._ctx, "Pool", return_value=pool):
            def my_fn(x, y, z=None):
                return x + y
            run_in_worker(my_fn, 1, 2, z="kw", timeout=None)
        # The first call arg of apply_async is _worker_entry; second is (fn, args, kwargs)
        call = pool.apply_async.call_args
        assert call[0][0] is worker_mod._worker_entry
        assert call[0][1] == (my_fn, (1, 2), {"z": "kw"})

    @pytest.mark.unit
    def test_default_timeout_is_none(self):
        pool = _mk_pool_mock(return_value=0)
        with patch.object(worker_mod._ctx, "Pool", return_value=pool):
            run_in_worker(lambda: 0)
        async_result = pool.apply_async.return_value
        # No timeout -> timeout=None
        async_result.get.assert_called_once_with(timeout=None)


# ---------------------------------------------------------------------------
# Per-section wrapper functions — verify they delegate to run_in_worker
# ---------------------------------------------------------------------------


class TestRunProbes:
    @pytest.mark.unit
    def test_delegates_to_run_in_worker(self):
        with patch.object(worker_mod, "run_in_worker", return_value={"ok": 1}) as mock:
            result = run_probes("/path/to/ckpt", "cuda", n_train=100, n_val=20,
                                n_epochs=3, seed_train=1, seed_val=2)
        assert result == {"ok": 1}
        # run_in_worker should have been called with the probes worker
        assert mock.called
        called_fn = mock.call_args[0][0]
        assert called_fn is worker_mod._probes_worker
        # Verify ALL positional args passed to run_in_worker
        args = mock.call_args[0]
        assert args[1] == "/path/to/ckpt"  # checkpoint_path (stringified)
        assert args[2] == "cuda"            # device
        assert args[3] == 100               # n_train
        assert args[4] == 20                # n_val
        assert args[5] == 3                 # n_epochs
        assert args[6] == 1                 # seed_train
        assert args[7] == 2                 # seed_val

    @pytest.mark.unit
    def test_stringifies_path(self):
        from pathlib import Path
        with patch.object(worker_mod, "run_in_worker", return_value={}) as mock:
            run_probes(Path("/abs/path"), "cpu")
        # The first positional after _probes_worker should be the str path
        args = mock.call_args[0]
        assert args[1] == "/abs/path"

    @pytest.mark.unit
    def test_uses_default_values(self):
        with patch.object(worker_mod, "run_in_worker", return_value={}) as mock:
            run_probes("/ckpt", "cuda")
        args = mock.call_args[0]
        # Positional args: (fn, ckpt, device, n_train, n_val, n_epochs, seed_train, seed_val)
        assert args[1] == "/ckpt"
        assert args[2] == "cuda"
        assert args[3] == 5_000  # default n_train
        assert args[4] == 1_000  # default n_val
        assert args[5] == 20     # default n_epochs
        assert args[6] == 3000   # default seed_train
        assert args[7] == 4000   # default seed_val


class TestRunOutcomeSignalTest:
    @pytest.mark.unit
    def test_delegates_to_run_in_worker(self):
        with patch.object(worker_mod, "run_in_worker", return_value={"r": 1}) as mock:
            result = run_outcome_signal_test("/ckpt", "cpu", n_per_outcome=50)
        assert result == {"r": 1}
        called_fn = mock.call_args[0][0]
        assert called_fn is worker_mod._signal_test_worker
        # Verify all positional args
        args = mock.call_args[0]
        assert args[1] == "/ckpt"            # checkpoint_path
        assert args[2] == "cpu"              # device
        assert args[3] == 50                 # n_per_outcome
        assert args[4] == [False, True]      # default mask_conditions as list

    @pytest.mark.unit
    def test_mask_conditions_converted_to_list(self):
        with patch.object(worker_mod, "run_in_worker", return_value={}) as mock:
            run_outcome_signal_test("/ckpt", "cpu",
                                    mask_conditions=(True, False, True))
        args = mock.call_args[0]
        # mask_conditions is last positional, a list
        assert args[-1] == [True, False, True]


class TestRunPrefixContinuationTest:
    @pytest.mark.unit
    def test_delegates(self):
        with patch.object(worker_mod, "run_in_worker", return_value={}) as mock:
            run_prefix_continuation_test("/ckpt", "/corpus", "cpu",
                                         n_per_bucket=10)
        called_fn = mock.call_args[0][0]
        assert called_fn is worker_mod._prefix_continuation_worker
        # Verify all positional args
        args = mock.call_args[0]
        assert args[1] == "/ckpt"            # checkpoint_path
        assert args[2] == "/corpus"          # corpus_dir
        assert args[3] == "cpu"              # device
        assert args[4] == 10                 # n_per_bucket
        # Default prefix_pcts and absolute_plies converted to lists
        assert args[5] == [0.1, 0.5, 0.9]
        assert args[6] == [10, 50, 100, 200]

    @pytest.mark.unit
    def test_converts_tuples_to_lists(self):
        with patch.object(worker_mod, "run_in_worker", return_value={}) as mock:
            run_prefix_continuation_test(
                "/ckpt", "/corpus", "cpu",
                prefix_pcts=(0.1, 0.2),
                absolute_plies=(10, 20, 30),
            )
        args = mock.call_args[0]
        # Last two positional args: prefix_pcts list, absolute_plies list
        assert args[-2] == [0.1, 0.2]
        assert args[-1] == [10, 20, 30]


class TestRunPoisonedPrefixTest:
    @pytest.mark.unit
    def test_delegates(self):
        with patch.object(worker_mod, "run_in_worker", return_value={}) as mock:
            run_poisoned_prefix_test("/ckpt", "/corpus", "cpu", n_per_pair=3)
        called_fn = mock.call_args[0][0]
        assert called_fn is worker_mod._poisoned_prefix_worker
        # Verify all positional args
        args = mock.call_args[0]
        assert args[1] == "/ckpt"    # checkpoint_path
        assert args[2] == "/corpus"  # corpus_dir
        assert args[3] == "cpu"      # device
        assert args[4] == 3          # n_per_pair
        assert args[5] == 0.5        # default prefix_pct

    @pytest.mark.unit
    def test_passes_prefix_pct(self):
        with patch.object(worker_mod, "run_in_worker", return_value={}) as mock:
            run_poisoned_prefix_test("/ckpt", "/corpus", "cpu", prefix_pct=0.33)
        args = mock.call_args[0]
        assert args[-1] == 0.33


class TestRunImpossibleTaskTest:
    @pytest.mark.unit
    def test_delegates(self):
        with patch.object(worker_mod, "run_in_worker", return_value={}) as mock:
            run_impossible_task_test("/ckpt", "/corpus", "cpu", n_per_scenario=7)
        called_fn = mock.call_args[0][0]
        assert called_fn is worker_mod._impossible_task_worker
        # Verify all positional args
        args = mock.call_args[0]
        assert args[1] == "/ckpt"    # checkpoint_path
        assert args[2] == "/corpus"  # corpus_dir
        assert args[3] == "cpu"      # device
        assert args[4] == 7          # n_per_scenario


class TestRunImprobableTaskTest:
    @pytest.mark.unit
    def test_delegates(self):
        with patch.object(worker_mod, "run_in_worker", return_value={}) as mock:
            run_improbable_task_test("/ckpt", "/corpus", "cpu", n_per_scenario=9)
        called_fn = mock.call_args[0][0]
        assert called_fn is worker_mod._improbable_task_worker
        # Verify all positional args
        args = mock.call_args[0]
        assert args[1] == "/ckpt"    # checkpoint_path
        assert args[2] == "/corpus"  # corpus_dir
        assert args[3] == "cpu"      # device
        assert args[4] == 9          # n_per_scenario


class TestRunDiagnosticEval:
    @pytest.mark.unit
    def test_delegates(self):
        with patch.object(worker_mod, "run_in_worker", return_value={}) as mock:
            run_diagnostic_eval("/ckpt", "cpu",
                                n_per_category=5, max_per_category=10,
                                n_samples=50, batch_size=16)
        called_fn = mock.call_args[0][0]
        assert called_fn is worker_mod._diagnostic_worker
        args = mock.call_args[0]
        # (fn, ckpt, device, n_per_category, max_per_category, n_samples, batch_size)
        assert args[3] == 5
        assert args[4] == 10
        assert args[5] == 50
        assert args[6] == 16


# ---------------------------------------------------------------------------
# End-to-end (real subprocess) — single smoke test to confirm multiprocessing
# actually works with a trivial function. Kept tiny to avoid slowness.
# ---------------------------------------------------------------------------


def _trivial_add(x: int, y: int) -> int:
    return x + y


class TestRunInWorkerRealSubprocess:
    @pytest.mark.unit
    def test_real_spawn_returns_value(self):
        result = run_in_worker(_trivial_add, 2, 3, timeout=30.0)
        assert result == 5
