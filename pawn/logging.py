"""Unified JSONL metrics logging for PAWN training pipelines.

Every training driver — pretraining (``scripts/train_jax.py``) and
adapter training (``scripts/train_jax_adapter.py``) — goes through
``MetricsLogger`` to ensure consistent, information-rich records.
One JSON object per line, per-record flush so a SIGKILL never loses
in-flight rows.

Guaranteed baseline fields on every record:
    type, step, timestamp, elapsed

``type: "config"`` records additionally include:
    hostname, git_hash, git_tag, slug, run_dir

``type: "train"`` and ``type: "val"`` records additionally include:
    mem/system_rss_gb, mem/system_used_gb, mem/system_total_gb,
    mem/cpu_percent, mem/gpu_current_gb, mem/gpu_peak_gb,
    mem/gpu_reserved_gb, mem/gpu_total_gb (the four GPU keys only
    when ``device != "cpu"`` and either ``nvidia-smi`` or
    ``rocm-smi`` is on PATH), lr (if provided).

The ``type`` discriminator is what the dashboard's train/val chart
split keys on (``pawn/dashboard/metrics.py``). Without it every row
falls into the same bucket — the lesson the JAX migration's first
attempt rediscovered when it replaced this module with inline
``json.dumps(...)`` calls in the trainers (see
``docs/jax-migration.md`` §8.2).

NaN / Inf values are sanitised to ``None`` so records stay RFC-7159
valid; downstream tooling can ``json.loads`` every row without a
``json.JSONDecodeError``.

This module is intentionally torch-free — the v1 version branched
on ``torch.cuda.*`` for GPU memory stats; the JAX surface uses
``psutil`` for host stats and shells out to ``nvidia-smi`` /
``rocm-smi`` for GPU stats (the same approach
``pawn/lab/runner._discover_gpus`` already uses).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Any

import psutil

# ---------------------------------------------------------------------------
# Git info (cached on first call)
# ---------------------------------------------------------------------------

_git_info: dict[str, str | None] | None = None


def get_git_info() -> dict[str, str | None]:
    """Return ``{"git_hash": ..., "git_tag": ...}`` for the current working tree.

    Honors ``PAWN_GIT_HASH`` / ``PAWN_GIT_TAG`` env vars (set on
    container images where the runtime is built outside a git
    checkout). Result is cached.
    """
    global _git_info
    if _git_info is not None:
        return _git_info

    git_hash: str | None = os.environ.get("PAWN_GIT_HASH")
    git_tag: str | None = os.environ.get("PAWN_GIT_TAG")

    if not git_hash:
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            pass

    if not git_tag:
        try:
            git_tag = (
                subprocess.check_output(
                    ["git", "tag", "--points-at", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                or None
            )
        except Exception:
            pass

    _git_info = {"git_hash": git_hash, "git_tag": git_tag}
    return _git_info


# ---------------------------------------------------------------------------
# Slug generator
# ---------------------------------------------------------------------------

_ADJECTIVES = (
    "amber", "bold", "calm", "deft", "eager", "fair", "grim", "hale",
    "keen", "lush", "mild", "neat", "pale", "quick", "rare", "sly",
    "taut", "vast", "warm", "zesty", "brisk", "crisp", "dense", "fleet",
    "grand", "hardy", "jolly", "lucid", "noble", "prime", "stark", "vivid",
)
_ANIMALS = (
    "puma", "lynx", "hawk", "wolf", "bear", "deer", "fox", "owl",
    "pike", "wren", "crane", "otter", "raven", "cobra", "heron", "bison",
    "finch", "marten", "osprey", "falcon", "badger", "salmon", "condor",
    "coyote", "ferret", "jackal", "marmot", "parrot", "turtle", "walrus",
)


def random_slug() -> str:
    """Two-word slug, e.g. ``zesty-osprey``.

    Drawn from small built-in word lists (no extra dependency on
    petname / coolname). Collisions are rare across short runs and
    fatal nowhere — the run-dir timestamp + microseconds + pid make
    the directory path unique even on a collision.
    """
    import random

    return f"{random.choice(_ADJECTIVES)}-{random.choice(_ANIMALS)}"


# ---------------------------------------------------------------------------
# GPU memory stats — shell-out, framework-neutral
# ---------------------------------------------------------------------------


def _discover_gpu_tool() -> str | None:
    """Return ``"nvidia-smi"``, ``"rocm-smi"``, or ``None`` if no GPU CLI
    is on PATH. Result is cached for the duration of the process."""
    if hasattr(_discover_gpu_tool, "_cached"):
        return _discover_gpu_tool._cached  # type: ignore[attr-defined]
    tool: str | None = None
    if shutil.which("nvidia-smi"):
        tool = "nvidia-smi"
    elif shutil.which("rocm-smi"):
        tool = "rocm-smi"
    _discover_gpu_tool._cached = tool  # type: ignore[attr-defined]
    return tool


def _read_nvidia_gpu_memory_mb() -> tuple[float, float] | None:
    """Return ``(used_mb, total_mb)`` for GPU 0 via nvidia-smi, or None."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    line = output.strip().splitlines()[0] if output.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _read_rocm_gpu_memory_mb() -> tuple[float, float] | None:
    """Return ``(used_mb, total_mb)`` for GPU 0 via rocm-smi --json."""
    try:
        output = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not data:
        return None
    # rocm-smi keys cards as ``card0`` / ``card1`` / ...; we read the
    # lowest-indexed one to mirror nvidia-smi's ``-i 0``.
    keys = sorted(k for k in data.keys() if k.startswith("card"))
    if not keys:
        return None
    card = data[keys[0]]
    used_bytes_str = card.get("VRAM Total Used Memory (B)")
    total_bytes_str = card.get("VRAM Total Memory (B)")
    if used_bytes_str is None or total_bytes_str is None:
        return None
    try:
        used_mb = float(used_bytes_str) / (1024 * 1024)
        total_mb = float(total_bytes_str) / (1024 * 1024)
    except ValueError:
        return None
    return used_mb, total_mb


def _read_gpu_memory_gb() -> tuple[float, float] | None:
    """Return ``(used_gb, total_gb)`` for GPU 0, or None if no GPU CLI works."""
    tool = _discover_gpu_tool()
    if tool == "nvidia-smi":
        mb = _read_nvidia_gpu_memory_mb()
    elif tool == "rocm-smi":
        mb = _read_rocm_gpu_memory_mb()
    else:
        return None
    if mb is None:
        return None
    used_mb, total_mb = mb
    return used_mb / 1024.0, total_mb / 1024.0


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _sanitize(obj: object) -> object:
    """Replace NaN/Inf with None recursively so records stay RFC-7159 valid."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize(v) for v in obj]
    return obj


def _json_default(obj: object) -> object:
    """Serialiser for types not natively supported.

    Intentionally torch-free — the JAX surface either lands ``float``s
    + numpy scalars in the metrics dict (handled by the ``hasattr
    item`` branch) or, in the rare case a ``jax.Array`` slips
    through, falls back to the ``str(obj)`` last branch.
    """
    if hasattr(obj, "item"):  # numpy scalar OR jax scalar OR python np-like
        try:
            return getattr(obj, "item")()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return getattr(obj, "tolist")()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------


class MetricsLogger:
    """JSONL metrics logger with rich baseline fields on every record.

    Usage::

        with MetricsLogger("logs", run_prefix="jax_run") as logger:
            logger.log_config(run_type="pretrain", supernet="tiny",
                              total_steps=1000, batch_size=16)
            logger.log_train(step=100, lr=3e-4, loss=3.5)
            logger.log_val(step=100, val_loss=3.4)

    The run directory name is
    ``<run_prefix>_<YYYYMMDD>_<HHMMSS>_<microseconds>_<suffix>_<slug>``
    so two runs that share a wall-clock second still produce distinct
    directories.
    """

    def __init__(
        self,
        logs_dir: str | Path,
        run_prefix: str = "jax_run",
        device: str = "cpu",
        slug: str | None = None,
        suffix: str = "",
    ) -> None:
        """Create a logger for a new run.

        Args:
            logs_dir: Parent directory for all runs.
            run_prefix: Prefix for the run directory name.
            device: Device string for GPU memory stats; "cpu" skips
                the GPU shell-out entirely.
            slug: Human-readable slug (e.g. "zesty-osprey").
                Auto-generated if None.
            suffix: Extra suffix for the run directory name
                (e.g. variant or strategy name).
        """
        self.slug = slug or random_slug()
        now = datetime.now()
        ts = now.strftime("%Y%m%d_%H%M%S")
        us = f"{now.microsecond:06d}"
        parts = [run_prefix, ts, us]
        if suffix:
            parts.append(suffix)
        parts.append(self.slug)
        dir_name = "_".join(parts)

        self.run_dir = Path(logs_dir) / dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._file: IO[str] = open(self.metrics_path, "a")
        self._proc = psutil.Process()
        self._device = device
        self._start_time = time.time()

    @property
    def path(self) -> Path:
        return self.metrics_path

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------

    def log_config(self, **kwargs: Any) -> None:
        """Log a config record. Called once at start of training.

        All keyword arguments are merged into the record. Common
        fields: ``run_type``, ``supernet``, ``model``, ``training``,
        ``param_count``, etc. The fixed baseline fields
        (``slug``, ``hostname``, ``timestamp``, ``run_dir``, plus
        ``get_git_info()``) are added automatically.
        """
        record: dict[str, Any] = {"type": "config"}
        record.update(kwargs)
        record["slug"] = self.slug
        record["hostname"] = socket.gethostname()
        record["timestamp"] = datetime.now().isoformat()
        record["run_dir"] = str(self.run_dir)
        record.update(get_git_info())
        self._write(record)

    def write_config_json(self, **kwargs: Any) -> Path:
        """Write a ``config.json`` alongside ``metrics.jsonl``.

        Used by the trainers to bundle the resolved ``PretrainConfig``
        / ``AdapterConfig`` (the single source of truth from
        ``pawn.run_config``) inside the run dir so a checkpoint pulled
        out of the run dir is self-describing.
        """
        data: dict[str, Any] = {}
        data.update(kwargs)
        data["slug"] = self.slug
        data.update(get_git_info())
        path = self.run_dir / "config.json"
        with open(path, "w") as f:
            json.dump(_sanitize(data), f, indent=2, default=_json_default)
        return path

    # -----------------------------------------------------------------------
    # Train records
    # -----------------------------------------------------------------------

    def log_train(self, step: int, **metrics: Any) -> None:
        """Log a training metrics record.

        Standard fields (pass as kwargs):
            epoch, lr, grad_norm, loss, accuracy, step_time,
            games_per_sec, train_loss, etc.

        Adapter-specific fields are passed through as-is:
            film/gamma_norm_L0, lora/B_norm_q, adapter/up_norm, etc.
        """
        record: dict[str, Any] = {"type": "train", "step": step}
        record.update(metrics)
        self._add_baseline(record)
        self._write(record)

    # -----------------------------------------------------------------------
    # Validation records
    # -----------------------------------------------------------------------

    def log_val(self, step: int, **metrics: Any) -> None:
        """Log a validation metrics record.

        Standard fields (pass as kwargs):
            epoch, loss (or val_loss), accuracy, top5_accuracy,
            patience, best_val_loss, best_val_step, etc.
        """
        record: dict[str, Any] = {"type": "val", "step": step}
        record.update(metrics)
        self._add_baseline(record)
        self._write(record)

    # -----------------------------------------------------------------------
    # Generic log (for callers that genuinely need a freeform record;
    # prefer log_train / log_val in new code so the type discriminator
    # stays meaningful).
    # -----------------------------------------------------------------------

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        epoch: int | None = None,
        record_type: str = "train",
        include_resources: bool = True,
    ) -> None:
        record: dict[str, Any] = {"type": record_type}
        if step is not None:
            record["step"] = step
        if epoch is not None:
            record["epoch"] = epoch
        record.update(metrics)
        self._add_baseline(record, include_resources=include_resources)
        self._write(record)

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _add_baseline(
        self, record: dict[str, Any], include_resources: bool = True
    ) -> None:
        """Add timestamp, elapsed, and resource stats to a record."""
        record["timestamp"] = datetime.now().isoformat()
        record["elapsed"] = round(time.time() - self._start_time, 3)
        if include_resources:
            self._add_resource_stats(record)

    def _add_resource_stats(self, record: dict[str, Any]) -> None:
        """Add host CPU / memory stats and (if available) GPU memory."""
        mem_info = self._proc.memory_info()
        sys_mem = psutil.virtual_memory()
        # ``psutil.cpu_percent(interval=None, percpu=True)`` returns the
        # CPU usage since the previous call; the first call after
        # process start returns zeros. That is fine for our purposes —
        # the first config record gets a zero CPU reading, every
        # subsequent record gets a real delta.
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)

        record["mem/system_rss_gb"] = round(mem_info.rss / (1024**3), 3)
        record["mem/system_used_gb"] = round(sys_mem.used / (1024**3), 3)
        record["mem/system_total_gb"] = round(sys_mem.total / (1024**3), 3)
        record["mem/cpu_percent"] = round(sum(per_cpu) if per_cpu else 0.0, 1)

        if self._device != "cpu":
            gpu = _read_gpu_memory_gb()
            if gpu is not None:
                used_gb, total_gb = gpu
                # **Compatibility aliases, not torch allocator
                # semantics.** v1's ``mem/gpu_peak_gb`` /
                # ``mem/gpu_reserved_gb`` / ``mem/gpu_current_gb``
                # came from ``torch.cuda.max_memory_allocated`` /
                # ``memory_reserved`` / ``memory_allocated`` — three
                # distinct numbers the torch allocator tracks
                # internally. ``nvidia-smi`` and ``rocm-smi`` only
                # expose the OS-level used VRAM (a single number),
                # so under JAX all three v1 keys carry the same
                # value: the OS-level used bytes from the smi tool.
                # Dashboards keyed on these names keep rendering
                # without a schema bump, but readers should not
                # interpret divergence between the three keys as
                # meaningful — they are aliased on purpose.
                # ``mem/gpu_total_gb`` is the new (informational)
                # key that doesn't exist in v1.
                record["mem/gpu_current_gb"] = round(used_gb, 3)
                record["mem/gpu_peak_gb"] = round(used_gb, 3)
                record["mem/gpu_reserved_gb"] = round(used_gb, 3)
                record["mem/gpu_total_gb"] = round(total_gb, 3)

    def _write(self, record: dict[str, Any]) -> None:
        """Write one JSONL record, sanitising NaN/Inf to ``null``.

        Per-record flush is deliberate — every record we lose to a
        SIGKILL in the chunk-loop body is one we don't recover from
        for run-level analysis.
        """
        self._file.write(
            json.dumps(_sanitize(record), default=_json_default) + "\n"
        )
        self._file.flush()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


__all__ = [
    "MetricsLogger",
    "get_git_info",
    "random_slug",
]
