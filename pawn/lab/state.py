"""Trial dataclass and helpers."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Trial:
    trial_id: int
    strategy: str
    params: dict[str, Any]
    cli_command: list[str]
    config: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    pid: int | None = None
    gpu_id: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    # Live metrics (updated by monitor)
    current_step: int = 0
    total_steps: int = 0
    steps_per_sec: float = 0.0
    last_train_loss: float | None = None
    last_train_acc: float | None = None
    best_val_loss: float | None = None
    best_accuracy: float | None = None
    actual_param_count: int | None = None
    # Files
    log_path: str = ""
    run_dir: str | None = None
    # Sweep
    optuna_number: int | None = None
    # Agent notes
    notes: str = ""

    def eta_seconds(self) -> float | None:
        if self.steps_per_sec > 0 and self.total_steps > self.current_step:
            return (self.total_steps - self.current_step) / self.steps_per_sec
        return None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        eta = self.eta_seconds()
        d["eta_seconds"] = eta
        d["eta_human"] = _format_duration(eta)
        elapsed = (time.time() - self.start_time) if self.start_time else None
        d["elapsed_human"] = _format_duration(elapsed)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trial:
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
