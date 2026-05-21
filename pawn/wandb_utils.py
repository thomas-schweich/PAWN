"""Metrics-only W&B integration for PAWN v2 training runs.

Both training entry points (``scripts/train_jax.py``,
``scripts/train_jax_adapter.py``) call :func:`init_wandb` once at run
start, :func:`log_metrics` per chunk, and :func:`finish_wandb` at
exit. The interface mirrors the v1 ``pawn.wandb_utils`` so any
downstream consumer that was previously reading W&B runs sees the
same metric shape.

Design notes:

- **Metrics-only.** Forwards each metrics dict to ``wandb.log``. No
  artifact uploads, no checkpoint lineage.
- **No silent swallow.** If the user opts in to W&B and ``wandb.init``
  fails, the training job fails loudly.
- **No torch / no MetricsLogger dependency.** v1 pulled in torch for
  GPU info and ``MetricsLogger`` for slug/run_dir/git plumbing — both
  removed in Phase 4. v2 reads GPU info from JAX and accepts the
  run-dir path + slug directly from the caller.

The ``PAWN_WANDB_MODE`` env var (``online`` / ``offline`` / ``disabled``)
is honored at init time; tests and CI default to ``disabled``.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, Mapping

import wandb
from wandb import Run

__all__ = ["init_wandb", "log_metrics", "finish_wandb", "get_git_info"]

Mode = Literal["online", "offline", "disabled"]


_git_info_cache: dict[str, str | None] | None = None


def get_git_info() -> dict[str, str | None]:
    """Return ``{"git_hash": ..., "git_tag": ...}`` for the working tree.

    Honors ``PAWN_GIT_HASH`` / ``PAWN_GIT_TAG`` env vars (set on the
    Docker images where the container is built outside a git
    checkout). Cached after the first call.
    """
    global _git_info_cache
    if _git_info_cache is not None:
        return _git_info_cache

    git_hash = os.environ.get("PAWN_GIT_HASH")
    git_tag = os.environ.get("PAWN_GIT_TAG")
    if not git_hash:
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip() or None
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            git_hash = None
    if not git_tag:
        try:
            git_tag = subprocess.check_output(
                ["git", "tag", "--points-at", "HEAD"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip() or None
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            git_tag = None
    _git_info_cache = {"git_hash": git_hash, "git_tag": git_tag}
    return _git_info_cache


def _gpu_info() -> dict[str, Any]:
    """Probe JAX for accelerator details. Returns ``{gpu_name, gpu_count, ...}``.

    Falls back to CPU-only metadata on a CPU-jaxlib install (the
    base v2 install). ``platform``-name is used as a stable backend
    identifier (``"cpu"`` / ``"cuda"`` / ``"rocm"``).
    """
    try:
        import jax  # local import — keep wandb_utils importable on
                    # the (rare) install without jax in scope.
    except ImportError:
        return {"gpu_name": None, "gpu_count": 0, "jax_backend": None}

    try:
        devices = jax.devices()
    except RuntimeError:
        return {"gpu_name": None, "gpu_count": 0, "jax_backend": None}

    if not devices:
        return {"gpu_name": None, "gpu_count": 0, "jax_backend": None}

    backend = devices[0].platform  # "cpu" / "gpu" (cuda or rocm)
    if backend == "cpu":
        return {"gpu_name": None, "gpu_count": 0, "jax_backend": "cpu"}
    # JAX's per-device `device_kind` is the human-readable accelerator
    # name (e.g. ``"NVIDIA H100 80GB HBM3"`` or ``"AMD Instinct MI300X"``).
    name = getattr(devices[0], "device_kind", None) or str(devices[0])
    return {
        "gpu_name": name,
        "gpu_count": len(devices),
        "jax_backend": backend,
    }


def _reproducibility_config(*, slug: str, run_dir: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "slug": slug,
        "run_dir": str(run_dir),
        "hostname": socket.gethostname(),
        "command_line": " ".join(sys.argv),
        "python_version": platform.python_version(),
    }
    cfg.update(get_git_info())
    cfg.update(_gpu_info())
    return cfg


def init_wandb(
    *,
    enabled: bool,
    project: str,
    slug: str,
    run_dir: Path,
    run_type: str,
    config: Mapping[str, Any],
    group: str | None = None,
    job_type: str | None = None,
    tags: list[str] | None = None,
) -> Run | None:
    """Initialise a W&B run for a training invocation.

    Returns ``None`` when ``enabled`` is False, so callers can pass the
    return value straight into :func:`log_metrics` / :func:`finish_wandb`
    without an extra branch.

    The W&B run name matches ``run_dir.name`` so the local logs
    directory and the W&B UI cross-reference trivially. ``wandb.config``
    is populated with the caller's ``config`` dict plus reproducibility
    metadata (slug, git hash/tag, hostname, command line, Python
    version, JAX backend + GPU info).
    """
    if not enabled:
        return None

    mode: Mode | None = None
    env_mode = os.environ.get("PAWN_WANDB_MODE")
    if env_mode in ("online", "offline", "disabled"):
        mode = env_mode  # type: ignore[assignment]
    elif env_mode is not None and env_mode != "":
        # Fail loud on a typo like "ONLINE" or "dryrun" — silently
        # ignoring it would be confusing during debugging.
        print(
            f"WARNING: PAWN_WANDB_MODE={env_mode!r} is not a recognized "
            "value (expected 'online', 'offline', or 'disabled'). "
            "Falling back to wandb's default mode.",
            file=sys.stderr,
        )

    repro = _reproducibility_config(slug=slug, run_dir=run_dir)
    full_config: dict[str, Any] = {**dict(config), **repro, "run_type": run_type}

    all_tags = list(tags) if tags else []
    git_hash = repro.get("git_hash")
    if git_hash:
        all_tags.append(f"git:{git_hash[:8]}")
    all_tags.append(f"run_type:{run_type}")

    # ``WANDB_PROJECT`` takes precedence over ``project`` so users can
    # redirect runs without touching the CLI flag. Mirrors wandb's own
    # env fallback (which we'd otherwise disable by always passing
    # ``project=`` explicitly).
    effective_project = os.environ.get("WANDB_PROJECT") or project

    run = wandb.init(
        project=effective_project,
        name=run_dir.name,
        group=group,
        job_type=job_type,
        tags=all_tags,
        config=full_config,
        dir=str(run_dir),
        mode=mode,
        reinit=True,
    )
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
