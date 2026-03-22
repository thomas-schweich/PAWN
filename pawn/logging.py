"""Unified JSONL metrics logging for PAWN training pipelines.

Every training/fine-tuning script should use MetricsLogger to ensure consistent
baseline fields across all run types. The logger writes one JSON object per line,
with a guaranteed set of fields on every record.
"""

import json
import math
import socket
import time
from datetime import datetime
from pathlib import Path

import psutil

try:
    import torch
except ImportError:
    torch = None


class MetricsLogger:
    """JSONL metrics logger with baseline fields on every record.

    Usage:
        logger = MetricsLogger("logs", run_prefix="bc")
        logger.log_config({"lr": 3e-4, "batch_size": 64, ...})
        logger.log({"loss": 0.5, "accuracy": 0.9}, step=100)
        logger.close()
    """

    def __init__(
        self,
        log_dir: str | Path,
        run_prefix: str = "run",
        device: str = "cpu",
    ):
        self.run_dir = Path(log_dir) / f"{run_prefix}_{datetime.now():%Y%m%d_%H%M%S}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._file = open(self.metrics_path, "a")
        self._proc = psutil.Process()
        self._device = device
        self._start_time = time.time()

    @property
    def path(self) -> Path:
        return self.metrics_path

    def log_config(self, config: dict) -> None:
        """Log a config record (type=config). Called once at start of training."""
        record = {"type": "config", **config}
        record["timestamp"] = datetime.now().isoformat()
        record["hostname"] = socket.gethostname()
        self._write(record)

    def log(
        self,
        metrics: dict,
        step: int | None = None,
        epoch: int | None = None,
        record_type: str = "train",
        include_resources: bool = True,
    ) -> None:
        """Log a metrics record with baseline fields.

        Args:
            metrics: Task-specific metrics (loss, accuracy, etc.)
            step: Global step number (for step-based training)
            epoch: Epoch number (for epoch-based training)
            record_type: Record type (train, eval, batch, etc.)
            include_resources: Whether to include memory/CPU stats
        """
        record = {"type": record_type}

        if step is not None:
            record["step"] = step
        if epoch is not None:
            record["epoch"] = epoch

        # Task-specific metrics
        record.update(metrics)

        # Baseline fields
        record["timestamp"] = datetime.now().isoformat()
        record["elapsed"] = time.time() - self._start_time

        if "lr" not in record:
            pass  # caller should include LR if relevant

        if include_resources:
            self._add_resource_stats(record)

        self._write(record)

    def _add_resource_stats(self, record: dict) -> None:
        """Add memory and CPU stats to a record."""
        mem_info = self._proc.memory_info()
        sys_mem = psutil.virtual_memory()
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)

        record["mem/system_rss_gb"] = mem_info.rss / (1024**3)
        record["mem/system_used_gb"] = sys_mem.used / (1024**3)
        record["mem/system_total_gb"] = sys_mem.total / (1024**3)
        record["mem/cpu_percent"] = sum(per_cpu) if per_cpu else 0.0

        if self._device != "cpu" and torch is not None and torch.cuda.is_available():
            record["mem/gpu_peak_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            record["mem/gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            record["mem/gpu_current_gb"] = torch.cuda.memory_allocated() / (1024**3)
            torch.cuda.reset_peak_memory_stats()

    def _write(self, record: dict) -> None:
        """Write a record, sanitizing NaN/Inf to null."""
        self._file.write(json.dumps(_sanitize(record)) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _sanitize(obj):
    """Replace NaN/Inf with None for valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj
