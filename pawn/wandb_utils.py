"""Metrics-only W&B integration for PAWN training runs.

Every training entry point (pretrain, cotrain, adapter) funnels through
:func:`init_wandb` / :func:`log_metrics` / :func:`finish_wandb` so run
naming, reproducibility metadata, and lifecycle are consistent.

Design notes:

- **Metrics-only.** We forward every ``logger.log_train`` / ``logger.log_val``
  to ``wandb.log``. No artifact uploads, no checkpoint lineage.
- **Option A resume.** Each process invocation creates a fresh W&B run.
  Resumes join sibling runs via ``group=<slug>`` and ``tags=["git:<hash>"]``.
  Nothing about W&B state is persisted to checkpoints.
- **No silent swallow.** If the user asks for W&B and ``wandb.init`` fails,
  the training job fails loudly.

The ``PAWN_WANDB_MODE`` env var (``online`` / ``offline`` / ``disabled``)
is honored at init time; tests and CI default to ``disabled``.
"""

from __future__ import annotations

import os
import platform
import socket
import sys
from typing import Any, Literal, Mapping

import torch
import wandb
from wandb import Run

from pawn.logging import MetricsLogger, get_git_info

__all__ = ["init_wandb", "log_metrics", "finish_wandb"]

Mode = Literal["online", "offline", "disabled"]


def _gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"gpu_name": None, "gpu_count": 0}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "gpu_name": props.name,
        "gpu_count": torch.cuda.device_count(),
        "gpu_total_memory_gb": round(props.total_memory / (1024**3), 2),
        "cuda_version": torch.version.cuda,
        "hip_version": getattr(torch.version, "hip", None),
    }


def _reproducibility_config(logger: MetricsLogger) -> dict[str, Any]:
    git = get_git_info()
    cfg: dict[str, Any] = {
        "slug": logger.slug,
        "run_dir": str(logger.run_dir),
        "hostname": socket.gethostname(),
        "command_line": " ".join(sys.argv),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    cfg.update(git)
    cfg.update(_gpu_info())
    return cfg


def init_wandb(
    *,
    enabled: bool,
    project: str,
    logger: MetricsLogger,
    run_type: str,
    config: Mapping[str, Any],
    group: str | None = None,
    job_type: str | None = None,
    tags: list[str] | None = None,
) -> Run | None:
    """Initialize a W&B run for a training invocation.

    Returns ``None`` when ``enabled`` is False, so callers can pass the
    return value straight into :func:`log_metrics` and :func:`finish_wandb`.

    The W&B run name matches ``logger.run_dir.name`` so the W&B UI and the
    local run directory cross-reference trivially. ``wandb.config`` is
    populated with the caller's ``config`` plus reproducibility metadata
    (slug, git hash/tag, hostname, command line, python/torch versions,
    GPU info).
    """
    if not enabled:
        return None

    mode: Mode | None = None
    env_mode = os.environ.get("PAWN_WANDB_MODE")
    if env_mode == "online" or env_mode == "offline" or env_mode == "disabled":
        mode = env_mode
    elif env_mode is not None and env_mode != "":
        # Fail loud on a typo like "ONLINE" or "dryrun" — silently
        # ignoring it would be confusing during debugging.
        print(
            f"WARNING: PAWN_WANDB_MODE={env_mode!r} is not a recognized "
            "value (expected 'online', 'offline', or 'disabled'). "
            "Falling back to wandb's default mode.",
            file=sys.stderr,
        )

    repro = _reproducibility_config(logger)
    full_config: dict[str, Any] = {**dict(config), **repro, "run_type": run_type}

    all_tags = list(tags) if tags else []
    git_hash = repro.get("git_hash")
    if git_hash:
        all_tags.append(f"git:{git_hash[:8]}")
    all_tags.append(f"run_type:{run_type}")

    # ``WANDB_PROJECT`` takes precedence over the caller's ``project``
    # so users can redirect runs to a different project without touching
    # ``TrainingConfig.wandb_project``. This mirrors wandb's own env-var
    # fallback — we restore it here because we always pass ``project=``
    # explicitly, which would otherwise disable the fallback.
    effective_project = os.environ.get("WANDB_PROJECT") or project

    run = wandb.init(
        project=effective_project,
        name=logger.run_dir.name,
        group=group,
        job_type=job_type,
        tags=all_tags,
        config=full_config,
        dir=str(logger.run_dir),
        mode=mode,
        reinit=True,
    )
    # wandb.init's stub declares ``Run | None``, but in practice it either
    # returns a Run (online/offline) or a RunDisabled that quacks the same
    # (disabled mode) — never None. The narrowing check is defense in depth
    # against a future stub/behavior change.
    if run is None:
        raise RuntimeError(
            "wandb.init() unexpectedly returned None; aborting to avoid "
            "silently running without metrics logging."
        )
    return run


def log_metrics(
    run: Run | None,
    metrics: Mapping[str, Any],
    step: int | None = None,
) -> None:
    """Forward a metrics dict to W&B. No-op when ``run`` is None."""
    if run is None:
        return
    if step is None:
        run.log(dict(metrics))
    else:
        run.log(dict(metrics), step=step)


def finish_wandb(run: Run | None, exit_code: int = 0) -> None:
    """Close a W&B run. No-op when ``run`` is None."""
    if run is None:
        return
    run.finish(exit_code=exit_code)
