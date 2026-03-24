"""Metrics loading and processing for PAWN dashboard."""

import json
import time
from pathlib import Path


def load_runs(log_dir: Path, max_age_hours: float = 1.0) -> list[str]:
    """Find run directories with recent metrics.jsonl, sorted newest first.

    Args:
        log_dir: Directory containing run subdirectories.
        max_age_hours: Only include runs whose metrics.jsonl was modified
            within this many hours.  Pass 0 to include all.
    """
    if not log_dir.is_dir():
        return []
    now = time.time()
    cutoff = now - max_age_hours * 3600 if max_age_hours > 0 else 0
    runs = []
    for r in log_dir.iterdir():
        mf = r / "metrics.jsonl"
        if r.is_dir() and mf.exists() and mf.stat().st_mtime >= cutoff:
            runs.append(r)
    runs.sort(key=lambda r: (r / "metrics.jsonl").stat().st_mtime, reverse=True)
    return [r.name for r in runs]


def get_run_meta(log_dir: Path, run_name: str) -> dict[str, str]:
    """Extract metadata (hostname, slug, variant) from the config record of a run."""
    path = log_dir / run_name / "metrics.jsonl"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            first_line = f.readline().strip()
            if first_line:
                rec = json.loads(first_line)
                return {
                    "hostname": rec.get("hostname", ""),
                    "slug": rec.get("slug", ""),
                    "variant": rec.get("variant", ""),
                }
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def get_run_hostname(log_dir: Path, run_name: str) -> str:
    """Extract hostname from the config record of a run."""
    return get_run_meta(log_dir, run_name).get("hostname", "")


def load_metrics(log_dir: Path, run_name: str) -> dict[str, list]:
    """Load and bucket metrics from a run's metrics.jsonl."""
    path = log_dir / run_name / "metrics.jsonl"
    buckets: dict[str, list] = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                buckets.setdefault(rec.get("type", "train"), []).append(rec)
    return buckets


def detect_run_type(config: dict) -> str:
    """Detect run type from config record."""
    rt = config.get("run_type")
    if rt in ("film", "lora", "hybrid", "sparse", "bottleneck", "tiny"):
        return rt
    if config.get("formulation") == "clm":
        return "pawn"
    if config.get("pgn_file"):
        return "bc"
    return "pawn"


def col(records: list[dict], key: str) -> list:
    """Extract a column from records, skipping missing/None values."""
    return [r[key] for r in records if key in r and r[key] is not None]
