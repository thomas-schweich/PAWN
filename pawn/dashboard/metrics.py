"""Metrics loader + run-navigation helpers for the dashboard.

Reads `metrics.jsonl` files written by :class:`pawn.logging.MetricsLogger`
and splits records by the `type` discriminator (plan §10 S9).

Two loader surfaces coexist by design:

- :func:`load_metrics` (single-arg, returns a typed :class:`MetricsBundle`) —
  the structured loader used by tests and any caller that wants the
  config/train/val split as attributes.
- :func:`load_run_buckets` (two-arg, returns ``dict[str, list]`` keyed by
  the raw ``type`` discriminator) — the v1-shaped loader the Solara shell
  (:mod:`pawn.dashboard.sol`) consumes via ``data.get("train")`` /
  ``data.get("val")`` / ``data.get("config")`` / ``data.get("batch")``.

The remaining helpers (:func:`load_runs`, :func:`list_trials`,
:func:`get_run_meta`, :func:`detect_run_type`, :func:`load_notes`,
:func:`save_notes`, :func:`notes_path`, :func:`sync_hf_metrics`, :func:`col`)
restore the v1 dashboard's trial-grouping, per-run notes, and HF-sync
navigation surface.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "MetricsBundle",
    "load_metrics",
    "load_run_buckets",
    "discover_runs",
    "load_runs",
    "list_trials",
    "get_run_meta",
    "get_run_hostname",
    "detect_run_type",
    "notes_path",
    "load_notes",
    "save_notes",
    "sync_hf_metrics",
    "col",
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


def load_run_buckets(log_dir: Path | str, run_name: str) -> dict[str, list[dict[str, Any]]]:
    """Load and bucket a run's `metrics.jsonl` keyed by the raw `type`.

    Returns ``{type: [records...]}`` — the v1 dashboard shape the Solara
    shell consumes (``data.get("config")`` / ``"train"`` / ``"val"`` /
    ``"batch"``). Records with no ``type`` default to the ``"train"``
    bucket (v1 parity). Malformed lines are skipped so a tail-read of a
    live file stays safe.
    """
    path = Path(log_dir) / run_name / "metrics.jsonl"
    buckets: dict[str, list[dict[str, Any]]] = {}
    if not path.is_file():
        return buckets
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        buckets.setdefault(rec.get("type", "train"), []).append(rec)
    return buckets


def discover_runs(log_dir: Path | str) -> list[Path]:
    """Find every subdirectory of ``log_dir`` that has a `metrics.jsonl`.

    Reuses :func:`_iter_run_dirs`, which stops descending at the first
    ``metrics.jsonl`` it finds in a directory — so a run dir that carries a
    nested ``checkpoints/metrics.jsonl`` is returned exactly once (as the
    real run dir), not twice. An ``rglob`` walk would descend into the
    ``checkpoints/`` subdir and double-count the run.
    """
    log_dir = Path(log_dir)
    return sorted(_iter_run_dirs(log_dir))


# ---------------------------------------------------------------------------
# Run discovery / trial grouping (v1 dashboard navigation surface)
# ---------------------------------------------------------------------------


def _iter_run_dirs(log_dir: Path) -> list[Path]:
    """Yield every directory under ``log_dir`` that contains a ``metrics.jsonl``.

    Walks recursively so a nested layout like ``log_dir/trial_0001/run_foo``
    is discovered alongside a flat ``log_dir/run_foo``. A run dir is not
    descended into (a run may carry ``checkpoints/`` subdirs that should
    never be mistaken for nested runs).
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
        if mf.is_file():
            found.append(d)
            continue
        for c in children:
            if c.is_dir():
                stack.append(c)
    return found


def load_runs(log_dir: Path | str, max_age_hours: float = 1.0) -> list[str]:
    """Find run directories with a recent metrics.jsonl, newest first.

    Runs are returned as POSIX-style paths relative to ``log_dir`` so a
    nested layout like ``trial_0001/run_foo`` round-trips through the other
    helpers without name collisions between trials.

    Args:
        log_dir: Root directory to search (recursively).
        max_age_hours: Only include runs whose metrics.jsonl was modified
            within this many hours. Pass ``0`` (or a negative value) to
            include every run regardless of age.
    """
    log_dir = Path(log_dir)
    dirs = _iter_run_dirs(log_dir)
    if not dirs:
        return []
    cutoff = time.time() - max_age_hours * 3600 if max_age_hours > 0 else 0.0
    pairs: list[tuple[float, Path]] = []
    for d in dirs:
        mtime = (d / "metrics.jsonl").stat().st_mtime
        if mtime >= cutoff:
            pairs.append((mtime, d))
    pairs.sort(key=lambda p: p[0], reverse=True)
    return [d.relative_to(log_dir).as_posix() for _, d in pairs]


def list_trials(log_dir: Path | str) -> list[str]:
    """Return top-level directories under ``log_dir`` that contain any run.

    Sorted newest-first by the most recent run mtime inside each trial.
    Returns ``[]`` when every run is a direct child (flat, trial-less
    layout) — callers should treat that as "no trial grouping" and show the
    full run list.
    """
    log_dir = Path(log_dir)
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


def get_run_meta(log_dir: Path | str, run_name: str) -> dict[str, str]:
    """Extract ``hostname`` / ``slug`` / ``variant`` from a run's config record.

    Reads only the first line (the config record is written first), so this
    stays cheap enough to call once per run while rendering the selector.
    """
    path = Path(log_dir) / run_name / "metrics.jsonl"
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            first_line = f.readline().strip()
    except OSError:
        return {}
    if not first_line:
        return {}
    try:
        rec = json.loads(first_line)
    except json.JSONDecodeError:
        return {}
    return {
        "hostname": rec.get("hostname", ""),
        "slug": rec.get("slug", ""),
        "variant": rec.get("variant", ""),
    }


def get_run_hostname(log_dir: Path | str, run_name: str) -> str:
    """Extract just the hostname from a run's config record."""
    return get_run_meta(log_dir, run_name).get("hostname", "")


_STRATEGY_RUN_TYPES: frozenset[str] = frozenset(
    {"film", "lora", "hybrid", "sparse", "bottleneck", "rosa", "unfreeze"}
)


def detect_run_type(config: dict[str, Any]) -> str:
    """Detect the dashboard run type from a config record.

    The pretrain supernet trainer is the ``"pawn"`` run type; adapter
    strategies surface under their own keys so the chart grid picks the
    matching diagnostics. ``specialized_clm`` renders under ``"tiny"``.

    The adapter trainer always logs the literal ``run_type="adapter"`` and
    stores the per-strategy name in a nested ``config`` sub-dict (see
    :class:`pawn.run_config.AdapterConfig` /
    ``scripts/train_jax_adapter.py`` —
    ``logger.log_config(run_type="adapter", config=cfg.model_dump())``). So
    the real adapter config record is
    ``{"type": "config", "run_type": "adapter", "config": {"strategy":
    "lora", ...}}``. We therefore consult the ``strategy`` field — both at
    the top level (older flat records) and inside the nested ``config``
    sub-dict (the live trainer shape) — before falling back to the bare
    ``run_type`` discriminator.
    """
    nested = config.get("config")
    nested_dict: dict[str, Any] = nested if isinstance(nested, dict) else {}
    strategy = config.get("strategy") or nested_dict.get("strategy")
    if isinstance(strategy, str):
        if strategy == "specialized_clm":
            return "tiny"
        if strategy in _STRATEGY_RUN_TYPES:
            return strategy

    rt: object = config.get("run_type")
    if isinstance(rt, str):
        if rt in _STRATEGY_RUN_TYPES:
            return rt
        if rt == "specialized_clm":
            return "tiny"
        if rt in ("pretrain", "pawn"):
            return "pawn"
    if config.get("formulation") == "clm":
        return "pawn"
    if config.get("pgn_file") or config.get("pgn"):
        return "bc"
    return "pawn"


def col(records: list[dict[str, Any]], key: str) -> list[Any]:
    """Extract a column from records, skipping missing / None values."""
    return [r[key] for r in records if key in r and r[key] is not None]


# ---------------------------------------------------------------------------
# Per-run notes (trial-scoped annotations)
# ---------------------------------------------------------------------------


def notes_path(log_dir: Path | str, run_name: str) -> Path:
    """Return the notes file path for ``run_name``.

    When the run lives under a trial dir (``trial_0001/run_foo``) the notes
    file is stored at the trial level so it is shared across every run in
    the trial. Flat runs keep notes next to their metrics.
    """
    log_dir = Path(log_dir)
    parts = Path(run_name).parts
    if len(parts) >= 2:
        return log_dir / parts[0] / "notes.md"
    return log_dir / run_name / "notes.md"


def load_notes(log_dir: Path | str, run_name: str) -> str:
    """Read the notes file for ``run_name`` — empty string if missing."""
    path = notes_path(log_dir, run_name)
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""


def save_notes(log_dir: Path | str, run_name: str, text: str) -> Path:
    """Write notes to disk and return the path written."""
    path = notes_path(log_dir, run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# HuggingFace metrics sync
# ---------------------------------------------------------------------------

# v2 republishes to new repos (`pawn-{small,base,large}-v2`); the v1 PyTorch
# repos are frozen and never carry live `run/*` training branches, so the
# sync pulls only from the v2 repos.
HF_REPOS = [
    "thomas-schweich/pawn-small-v2",
    "thomas-schweich/pawn-base-v2",
    "thomas-schweich/pawn-large-v2",
]


def sync_hf_metrics(log_dir: Path | str) -> list[str]:
    """Pull ``metrics.jsonl`` from active HF ``run/*`` branches into ``log_dir``.

    Each in-flight training run pushes to a ``run/<run_id>`` branch
    (``pawn.checkpoint`` HF mode); this mirrors the latest ``metrics.jsonl``
    from every such branch so the dashboard can watch a remote pod's run
    without rsyncing checkpoint files off it. Returns the list of synced run
    names. Best-effort: a repo/branch that fails (private, deleted, no
    metrics yet) is skipped rather than aborting the whole sync.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        return []

    log_dir = Path(log_dir)
    api = HfApi()
    synced: list[str] = []
    for repo in HF_REPOS:
        try:
            refs = api.list_repo_refs(repo, repo_type="model")
            branches = [b.name for b in refs.branches if b.name.startswith("run/")]
        except Exception:  # noqa: BLE001 — network / auth / missing repo
            continue

        for branch in branches:
            run_name = branch.removeprefix("run/")
            run_dir = log_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                hf_hub_download(
                    repo_id=repo,
                    filename="metrics.jsonl",
                    revision=branch,
                    repo_type="model",
                    local_dir=str(run_dir),
                )
                synced.append(run_name)
            except Exception:  # noqa: BLE001 — branch may carry no metrics yet
                continue

    return synced
