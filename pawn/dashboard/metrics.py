"""Metrics loader for the dashboard.

Reads `metrics.jsonl` files written by :class:`pawn.logging.MetricsLogger`
and splits train/val records by the `type` discriminator (plan §10 S9).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "MetricsBundle",
    "load_metrics",
    "discover_runs",
]


@dataclass(frozen=True)
class MetricsBundle:
    """Per-run metrics, split by the `type` discriminator."""

    run_dir: Path
    config: dict[str, Any] | None
    train_records: list[dict[str, Any]]
    val_records: list[dict[str, Any]]

    @property
    def slug(self) -> str | None:
        if self.config is not None:
            return self.config.get("slug")
        for r in (*self.train_records, *self.val_records):
            if "slug" in r:
                return r["slug"]
        return None


def load_metrics(run_dir: Path | str) -> MetricsBundle:
    """Read a run's `metrics.jsonl` and split records by `type`.

    Returns a MetricsBundle. A malformed line (JSON parse error) is
    silently skipped — partial-write resilience matches the v1
    contract.
    """
    run_dir = Path(run_dir)
    config: dict[str, Any] | None = None
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    jsonl = run_dir / "metrics.jsonl"
    if not jsonl.is_file():
        return MetricsBundle(run_dir=run_dir, config=None, train_records=[], val_records=[])
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rec_type = rec.get("type")
        if rec_type == "config":
            config = rec
        elif rec_type == "train":
            train.append(rec)
        elif rec_type == "val":
            val.append(rec)
    return MetricsBundle(
        run_dir=run_dir, config=config, train_records=train, val_records=val
    )


def discover_runs(log_dir: Path | str) -> list[Path]:
    """Find every subdirectory of ``log_dir`` that has a `metrics.jsonl`."""
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return []
    return sorted(
        p.parent for p in log_dir.rglob("metrics.jsonl")
    )
