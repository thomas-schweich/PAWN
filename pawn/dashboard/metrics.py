"""Metrics loading and processing for PAWN dashboard."""

import json
import time
from pathlib import Path


def _iter_run_dirs(log_dir: Path) -> list[Path]:
    """Yield every directory under ``log_dir`` that contains a ``metrics.jsonl``.

    Walks recursively so a nested layout like ``log_dir/trial_0001/run_foo``
    is discovered alongside a flat ``log_dir/run_foo``.
    """
    if not log_dir.is_dir():
        return []
    found: list[Path] = []
    stack: list[Path] = [log_dir]
    while stack:
        d = stack.pop()
        try:
            children = list(d.iterdir())
        except (OSError, PermissionError):
            continue
        mf = d / "metrics.jsonl"
        if mf.exists() and mf.is_file():
            found.append(d)
            # A run dir may still contain subdirs like checkpoints/ — don't
            # descend looking for nested metrics.jsonl inside a run.
            continue
        for c in children:
            if c.is_dir():
                stack.append(c)
    return found


def load_runs(log_dir: Path, max_age_hours: float = 1.0) -> list[str]:
    """Find run directories with recent metrics.jsonl, sorted newest first.

    Runs are returned as paths relative to ``log_dir`` (POSIX-style). This
    lets nested layouts like ``trial_0001/run_foo`` round-trip through the
    other helpers without name collisions between trials.

    Args:
        log_dir: Root directory to search (recursively).
        max_age_hours: Only include runs whose metrics.jsonl was modified
            within this many hours.  Pass 0 to include all.
    """
    dirs = _iter_run_dirs(log_dir)
    if not dirs:
        return []
    now = time.time()
    cutoff = now - max_age_hours * 3600 if max_age_hours > 0 else 0
    pairs = []
    for d in dirs:
        mtime = (d / "metrics.jsonl").stat().st_mtime
        if mtime >= cutoff:
            pairs.append((mtime, d))
    pairs.sort(key=lambda p: p[0], reverse=True)
    return [d.relative_to(log_dir).as_posix() for _, d in pairs]


def list_trials(log_dir: Path) -> list[str]:
    """Return the set of top-level directories under ``log_dir`` that contain
    any discovered run, sorted newest first by run mtime.

    Returns an empty list if ``log_dir`` only contains flat (trial-less) runs,
    i.e. every run is a direct child. Callers should treat that as "no trial
    grouping" and show the full run list.
    """
    dirs = _iter_run_dirs(log_dir)
    if not dirs:
        return []
    trial_to_mtime: dict[str, float] = {}
    for d in dirs:
        rel = d.relative_to(log_dir)
        parts = rel.parts
        if len(parts) <= 1:
            continue
        trial = parts[0]
        m = (d / "metrics.jsonl").stat().st_mtime
        if m > trial_to_mtime.get(trial, 0.0):
            trial_to_mtime[trial] = m
    if not trial_to_mtime:
        return []
    return sorted(trial_to_mtime, key=lambda t: trial_to_mtime[t], reverse=True)


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


def notes_path(log_dir: Path, run_name: str) -> Path:
    """Return the path to the notes file associated with a run.

    When the run lives under a trial dir (``trial_0001/run_foo``) the notes
    file is stored at the trial level so it is shared across every run in
    the trial. Flat runs keep notes next to their metrics.
    """
    rel = Path(run_name)
    parts = rel.parts
    if len(parts) >= 2:
        return log_dir / parts[0] / "notes.md"
    return log_dir / run_name / "notes.md"


def load_notes(log_dir: Path, run_name: str) -> str:
    """Read the notes file for ``run_name`` — empty string if missing."""
    path = notes_path(log_dir, run_name)
    try:
        return path.read_text()
    except (FileNotFoundError, OSError):
        return ""


def save_notes(log_dir: Path, run_name: str, text: str) -> Path:
    """Write notes to disk and return the path written."""
    path = notes_path(log_dir, run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def detect_run_type(config: dict) -> str:
    """Detect run type from config record."""
    rt = config.get("run_type")
    if rt in ("film", "lora", "hybrid", "sparse", "bottleneck", "tiny", "rosa"):
        return rt
    if config.get("formulation") == "clm":
        return "pawn"
    if config.get("pgn_file"):
        return "bc"
    return "pawn"


def col(records: list[dict], key: str) -> list:
    """Extract a column from records, skipping missing/None values."""
    return [r[key] for r in records if key in r and r[key] is not None]


# ---------------------------------------------------------------------------
# HuggingFace metrics sync
# ---------------------------------------------------------------------------

HF_REPOS = ["thomas-schweich/pawn-small", "thomas-schweich/pawn-base", "thomas-schweich/pawn-large"]


def sync_hf_metrics(log_dir: Path) -> list[str]:
    """Pull metrics.jsonl from all active HF run branches into log_dir.

    Returns list of synced run names.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        return []

    api = HfApi()
    synced = []
    for repo in HF_REPOS:
        try:
            branches = [
                b.name for b in api.list_repo_refs(repo, repo_type="model").branches
                if b.name.startswith("run/")
            ]
        except Exception:
            continue

        for branch in branches:
            # Branch name: run/run_YYYYMMDD_HHMMSS_variant
            run_name = branch.removeprefix("run/")
            run_dir = log_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            try:
                hf_hub_download(
                    repo_id=repo, filename="metrics.jsonl",
                    revision=branch, repo_type="model",
                    local_dir=str(run_dir), local_dir_use_symlinks=False,
                )
                synced.append(run_name)
            except Exception:
                continue

    return synced
