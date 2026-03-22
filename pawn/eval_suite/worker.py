"""Subprocess isolation for GPU-heavy evaluation sections.

Each heavy eval section runs in a fresh worker process with its own GPU
context. On KeyboardInterrupt, only the worker dies — the notebook kernel
and all previously-computed results survive.

Usage::

    import pawn.eval_suite as eval_suite

    probe_results = eval_suite.run_probes(
        CHECKPOINT_PATH, DEVICE, n_train=5000, n_val=1000,
    )
    signal_results = eval_suite.run_outcome_signal_test(
        CHECKPOINT_PATH, DEVICE, n_per_outcome=1000,
    )
"""

from __future__ import annotations

import gc
import multiprocessing as mp
from collections.abc import Callable
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pawn.model import PAWNCLM

# Use "spawn" so the child gets a clean process with no inherited GPU state.
_ctx = mp.get_context("spawn")


# ---------------------------------------------------------------------------
# Core isolation primitive
# ---------------------------------------------------------------------------


def _worker_entry(fn: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    return fn(*args, **kwargs)


def run_in_worker(fn: Callable[..., Any], *args: Any, timeout: float | None = None, **kwargs: Any) -> Any:
    """Run fn(*args, **kwargs) in an isolated worker process.

    On KeyboardInterrupt, the worker is terminated and the interrupt is
    re-raised in the notebook — the kernel and all prior results survive.
    """
    with _ctx.Pool(1) as pool:
        try:
            return pool.apply_async(_worker_entry, (fn, args, kwargs)).get(timeout=timeout)
        except (KeyboardInterrupt, Exception):
            pool.terminate()
            pool.join()
            raise


# ---------------------------------------------------------------------------
# Shared helpers (called inside worker processes)
# ---------------------------------------------------------------------------


def _load_model(checkpoint_path: str, device: str) -> PAWNCLM:
    """Load and freeze a PAWNCLM checkpoint. Runs inside worker processes."""
    import torch
    from pawn.config import CLMConfig
    from pawn.model import PAWNCLM

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = CLMConfig(**ckpt["model_config"]) if "model_config" in ckpt else CLMConfig()
    model = PAWNCLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    gc.collect()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _load_corpus(corpus_dir: str) -> dict:
    from pawn.eval_suite.corpus import load_corpus
    return load_corpus(Path(corpus_dir))


# ---------------------------------------------------------------------------
# Per-section worker functions
# ---------------------------------------------------------------------------


def _probes_worker(checkpoint_path: str, device: str, n_train: int, n_val: int,
                   n_epochs: int, seed_train: int, seed_val: int) -> dict:
    from pawn.eval_suite.probes import extract_probe_data, train_all_probes
    model = _load_model(checkpoint_path, device)
    train_data = extract_probe_data(n_train, max_ply=256, seed=seed_train)
    val_data = extract_probe_data(n_val, max_ply=256, seed=seed_val)
    return train_all_probes(model, train_data, val_data, device,
                            per_layer=True, n_epochs=n_epochs, verbose=True)


def run_probes(
    checkpoint_path: str | Path,
    device: str,
    n_train: int = 5_000,
    n_val: int = 1_000,
    n_epochs: int = 20,
    seed_train: int = 3000,
    seed_val: int = 4000,
) -> dict:
    """Train all linear probes per-layer in an isolated worker process."""
    return run_in_worker(
        _probes_worker,
        str(checkpoint_path), device, n_train, n_val, n_epochs,
        seed_train, seed_val,
    )


def _signal_test_worker(checkpoint_path: str, device: str, n_per_outcome: int,
                         mask_conditions: list[bool]) -> dict:
    from pawn.eval_suite.generation import outcome_signal_test
    model = _load_model(checkpoint_path, device)
    return outcome_signal_test(model, device, n_per_outcome=n_per_outcome,
                               mask_conditions=tuple(mask_conditions))


def run_outcome_signal_test(
    checkpoint_path: str | Path,
    device: str,
    n_per_outcome: int = 1000,
    mask_conditions: tuple[bool, ...] = (False, True),
) -> dict:
    """Run outcome token signal test (§6.1–6.3) in an isolated worker."""
    return run_in_worker(
        _signal_test_worker,
        str(checkpoint_path), device, n_per_outcome, list(mask_conditions),
    )


def _prefix_continuation_worker(checkpoint_path: str, corpus_dir: str, device: str,
                                 n_per_bucket: int, prefix_pcts: list[float],
                                 absolute_plies: list[int]) -> dict:
    from pawn.eval_suite.generation import prefix_continuation_test
    model = _load_model(checkpoint_path, device)
    corpus = _load_corpus(corpus_dir)
    return prefix_continuation_test(model, corpus, device,
                                    n_per_bucket=n_per_bucket,
                                    prefix_pcts=tuple(prefix_pcts),
                                    absolute_plies=tuple(absolute_plies))


def run_prefix_continuation_test(
    checkpoint_path: str | Path,
    corpus_dir: str | Path,
    device: str,
    n_per_bucket: int = 200,
    prefix_pcts: tuple[float, ...] = (0.1, 0.5, 0.9),
    absolute_plies: tuple[int, ...] = (10, 50, 100, 200),
) -> dict:
    """Run prefix continuation test (§6.4) in an isolated worker."""
    return run_in_worker(
        _prefix_continuation_worker,
        str(checkpoint_path), str(corpus_dir), device,
        n_per_bucket, list(prefix_pcts), list(absolute_plies),
    )


def _poisoned_prefix_worker(checkpoint_path: str, corpus_dir: str, device: str,
                             n_per_pair: int, prefix_pct: float) -> dict:
    from pawn.eval_suite.generation import poisoned_prefix_test
    model = _load_model(checkpoint_path, device)
    corpus = _load_corpus(corpus_dir)
    return poisoned_prefix_test(model, corpus, device,
                                n_per_pair=n_per_pair, prefix_pct=prefix_pct)


def run_poisoned_prefix_test(
    checkpoint_path: str | Path,
    corpus_dir: str | Path,
    device: str,
    n_per_pair: int = 500,
    prefix_pct: float = 0.5,
) -> dict:
    """Run poisoned prefix test (§6.5) in an isolated worker."""
    return run_in_worker(
        _poisoned_prefix_worker,
        str(checkpoint_path), str(corpus_dir), device, n_per_pair, prefix_pct,
    )


def _impossible_task_worker(checkpoint_path: str, corpus_dir: str, device: str,
                             n_per_scenario: int) -> dict:
    from pawn.eval_suite.generation import impossible_task_test
    model = _load_model(checkpoint_path, device)
    corpus = _load_corpus(corpus_dir)
    return impossible_task_test(model, corpus, device, n_per_scenario=n_per_scenario)


def run_impossible_task_test(
    checkpoint_path: str | Path,
    corpus_dir: str | Path,
    device: str,
    n_per_scenario: int = 200,
) -> dict:
    """Run impossible task test (§6.6) in an isolated worker."""
    return run_in_worker(
        _impossible_task_worker,
        str(checkpoint_path), str(corpus_dir), device, n_per_scenario,
    )


def _improbable_task_worker(checkpoint_path: str, corpus_dir: str, device: str,
                             n_per_scenario: int) -> dict:
    from pawn.eval_suite.generation import improbable_task_test
    model = _load_model(checkpoint_path, device)
    corpus = _load_corpus(corpus_dir)
    return improbable_task_test(model, corpus, device, n_per_scenario=n_per_scenario)


def run_improbable_task_test(
    checkpoint_path: str | Path,
    corpus_dir: str | Path,
    device: str,
    n_per_scenario: int = 200,
) -> dict:
    """Run improbable task test (§6.7) in an isolated worker."""
    return run_in_worker(
        _improbable_task_worker,
        str(checkpoint_path), str(corpus_dir), device, n_per_scenario,
    )


def _diagnostic_worker(checkpoint_path: str, corpus_dir: str, device: str,
                        min_per_category: int, max_per_category: int,
                        n_samples: int, batch_size: int) -> dict:
    from pawn.eval_suite.diagnostics import (
        extract_diagnostic_positions, evaluate_diagnostic_positions,
    )
    model = _load_model(checkpoint_path, device)
    corpus = _load_corpus(corpus_dir)
    positions = extract_diagnostic_positions(
        corpus, min_per_category=min_per_category,
        max_per_category=max_per_category,
    )
    for cat, pos_list in positions.items():
        print(f"  {cat}: {len(pos_list)} positions")
    return evaluate_diagnostic_positions(
        model, positions, corpus, device,
        n_samples=n_samples, batch_size=batch_size,
    )


def run_diagnostic_eval(
    checkpoint_path: str | Path,
    corpus_dir: str | Path,
    device: str,
    min_per_category: int = 2000,
    max_per_category: int = 5000,
    n_samples: int = 100,
    batch_size: int = 32,
) -> dict:
    """Extract diagnostic positions and evaluate model on them in an isolated worker."""
    return run_in_worker(
        _diagnostic_worker,
        str(checkpoint_path), str(corpus_dir), device,
        min_per_category, max_per_category, n_samples, batch_size,
    )
