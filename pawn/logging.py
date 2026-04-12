"""Unified JSONL metrics logging for PAWN training pipelines.

Every training script — pretraining, multi-model, adapters — uses MetricsLogger
to ensure consistent, information-rich records. One JSON object per line.

Guaranteed baseline fields on every record:
    type, step, timestamp, elapsed

Config records additionally include:
    hostname, git_hash, git_tag, slug (if set)

Train/val records additionally include:
    mem/* (system + GPU), lr (if provided)
"""

import json
import math
import os
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil

try:
    import torch
except ImportError:
    torch = None


# ---------------------------------------------------------------------------
# Git info (cached on first call)
# ---------------------------------------------------------------------------

_git_info: dict[str, str | None] | None = None


def _get_git_info() -> dict[str, str | None]:
    global _git_info
    if _git_info is not None:
        return _git_info

    git_hash = os.environ.get("PAWN_GIT_HASH")
    git_tag = os.environ.get("PAWN_GIT_TAG")

    if not git_hash:
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            pass

    if not git_tag:
        try:
            git_tag = subprocess.check_output(
                ["git", "tag", "--points-at", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip() or None
        except Exception:
            pass

    _git_info = {"git_hash": git_hash, "git_tag": git_tag}
    return _git_info


# ---------------------------------------------------------------------------
# Slug generator
# ---------------------------------------------------------------------------

_ADJECTIVES = [
    "amber", "bold", "calm", "deft", "eager", "fair", "grim", "hale",
    "keen", "lush", "mild", "neat", "pale", "quick", "rare", "sly",
    "taut", "vast", "warm", "zesty", "brisk", "crisp", "dense", "fleet",
    "grand", "hardy", "jolly", "lucid", "noble", "prime", "stark", "vivid",
]
_ANIMALS = [
    "puma", "lynx", "hawk", "wolf", "bear", "deer", "fox", "owl",
    "pike", "wren", "crane", "otter", "raven", "cobra", "heron", "bison",
    "finch", "marten", "osprey", "falcon", "badger", "salmon", "condor",
    "coyote", "ferret", "jackal", "marmot", "parrot", "turtle", "walrus",
]


def random_slug() -> str:
    import random
    return f"{random.choice(_ADJECTIVES)}-{random.choice(_ANIMALS)}"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _sanitize(obj: object) -> object:
    """Replace NaN/Inf with None for valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _json_default(obj: object) -> object:
    """JSON serializer for types not natively supported."""
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if hasattr(obj, "item"):  # numpy scalar
        return getattr(obj, "item")()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------

class MetricsLogger:
    """JSONL metrics logger with rich baseline fields on every record.

    Usage::

        logger = MetricsLogger("logs", run_prefix="film", device="cuda")
        logger.log_config(run_type="film", model=cfg.__dict__, training={...})

        # Step-based training (pretraining)
        logger.log_train(step=100, lr=3e-4, loss=3.5, accuracy=0.06,
                         step_time=0.34, games_per_sec=750)
        logger.log_val(step=100, loss=3.6, accuracy=0.05, patience=2)

        # Epoch-based training (adapters)
        logger.log_train(step=500, epoch=3, lr=1e-4, train_loss=2.1,
                         val_loss=2.3, val_top1=0.12, epoch_time_s=45)

        logger.close()
    """

    def __init__(
        self,
        log_dir: str | Path,
        run_prefix: str = "run",
        device: str = "cpu",
        slug: str | None = None,
        suffix: str = "",
    ):
        """Create a logger for a new run.

        Args:
            log_dir: Parent directory for all runs.
            run_prefix: Prefix for the run directory name.
            device: Device string for GPU memory stats.
            slug: Human-readable slug (e.g. "zesty-osprey"). Auto-generated if None.
            suffix: Extra suffix for run directory name (e.g. variant name).
        """
        self.slug = slug or random_slug()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [run_prefix, ts]
        if suffix:
            parts.append(suffix)
        parts.append(self.slug)
        dir_name = "_".join(parts)

        self.run_dir = Path(log_dir) / dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._file = open(self.metrics_path, "a")
        self._proc = psutil.Process()
        self._device = device
        self._start_time = time.time()

    @property
    def path(self) -> Path:
        return self.metrics_path

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------

    def log_config(self, **kwargs: object) -> None:
        """Log a config record. Called once at start of training.

        All keyword arguments are included in the record. Common fields:
            run_type, model, training, param_count, variant, etc.
        """
        record: dict[str, object] = {"type": "config"}
        record.update(kwargs)
        record["slug"] = self.slug
        record["hostname"] = socket.gethostname()
        record["timestamp"] = datetime.now().isoformat()
        record.update(_get_git_info())
        self._write(record)

    # -----------------------------------------------------------------------
    # Config file (JSON, for checkpoint bundling)
    # -----------------------------------------------------------------------

    def write_config_json(self, **kwargs: object) -> Path:
        """Write config.json alongside metrics.jsonl. Returns the path."""
        data: dict[str, object] = {}
        data.update(kwargs)
        data["slug"] = self.slug
        data.update(_get_git_info())
        path = self.run_dir / "config.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)
        return path

    # -----------------------------------------------------------------------
    # Train records
    # -----------------------------------------------------------------------

    def log_train(
        self,
        step: int,
        **metrics: object,
    ) -> None:
        """Log a training metrics record.

        Standard fields (pass as kwargs):
            epoch, lr, grad_norm, loss, accuracy, step_time, games_per_sec,
            train_loss, train_top1, etc.

        Adapter-specific fields are passed through as-is:
            film/gamma_norm_L0, lora/B_norm_q, adapter/up_norm, etc.
        """
        record: dict[str, object] = {"type": "train", "step": step}
        for k, v in metrics.items():
            record[k] = v

        self._add_baseline(record)
        self._write(record)

    # -----------------------------------------------------------------------
    # Validation records
    # -----------------------------------------------------------------------

    def log_val(
        self,
        step: int,
        **metrics: object,
    ) -> None:
        """Log a validation metrics record.

        Standard fields (pass as kwargs):
            epoch, loss (or val/loss), accuracy, top5_accuracy,
            patience, best_val_loss, best_val_step, etc.
        """
        record: dict[str, object] = {"type": "val", "step": step}
        for k, v in metrics.items():
            record[k] = v

        self._add_baseline(record)
        self._write(record)

    # -----------------------------------------------------------------------
    # Generic log (backward compat)
    # -----------------------------------------------------------------------

    def log(
        self,
        metrics: dict,
        step: int | None = None,
        epoch: int | None = None,
        record_type: str = "train",
        include_resources: bool = True,
    ) -> None:
        """Log a generic metrics record. Prefer log_train/log_val for new code."""
        record: dict[str, object] = {"type": record_type}
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

    def _add_baseline(self, record: dict, include_resources: bool = True) -> None:
        """Add timestamp, elapsed, and resource stats to a record."""
        record["timestamp"] = datetime.now().isoformat()
        record["elapsed"] = round(time.time() - self._start_time, 3)

        if include_resources:
            self._add_resource_stats(record)

    def _add_resource_stats(self, record: dict) -> None:
        """Add memory and CPU stats to a record."""
        mem_info = self._proc.memory_info()
        sys_mem = psutil.virtual_memory()
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)

        record["mem/system_rss_gb"] = round(mem_info.rss / (1024**3), 3)
        record["mem/system_used_gb"] = round(sys_mem.used / (1024**3), 3)
        record["mem/system_total_gb"] = round(sys_mem.total / (1024**3), 3)
        record["mem/cpu_percent"] = round(sum(per_cpu) if per_cpu else 0.0, 1)

        if self._device != "cpu" and torch is not None and torch.cuda.is_available():
            record["mem/gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
            record["mem/gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024**3), 3)
            record["mem/gpu_current_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 3)
            torch.cuda.reset_peak_memory_stats()

    def _write(self, record: dict) -> None:
        """Write a record, sanitizing NaN/Inf to null."""
        self._file.write(json.dumps(_sanitize(record), default=_json_default) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
