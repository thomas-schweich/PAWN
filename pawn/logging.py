"""Unified JSONL metrics logger for the v2 JAX training stack.

The module is the **only** path metrics take to disk. Every record in
``metrics.jsonl`` carries a ``type ∈ {"config", "train", "val"}``
discriminator plus baseline metadata (``timestamp``, ``slug``,
``hostname``, ``git_hash``, ``git_tag``, ``elapsed``) plus, on
train/val records, host CPU + memory stats (via :mod:`psutil`) and GPU
memory stats (via a shell-out to ``nvidia-smi`` or ``rocm-smi`` cached
at process start).
NaN / Inf values are sanitised to ``null`` so the file stays
JSON-parseable. Per-record :func:`file.flush()` keeps the dashboard
able to tail the log without losing the tail of every run.

This module is deliberately **torch-free** — the v1 path used
``torch.cuda.*`` to query GPU memory; v2 shells out so the JAX-only
import graph stays lightweight.

Usage::

    logger = MetricsLogger("logs", run_prefix="lora", device="cuda")
    logger.log_config(run_type="adapter", model=cfg.model_dump(),
                      strategy="lora", lora_rank=4)
    logger.log_train(step=100, lr=3e-4, loss=3.5, accuracy=0.06)
    logger.log_val(step=100, loss=3.6, accuracy=0.05)
    logger.close()

Run directory naming (slug-based, with microsecond precision):

    ``<prefix>_<YYYYMMDD>_<HHMMSS>_<microseconds>_<suffix?>_<slug>``

The microsecond field disambiguates rapid-succession runs that share a
second (e.g. sweep child processes spawned in parallel).
"""

from __future__ import annotations

import functools
import json
import math
import os
import random
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

__all__ = [
    "MetricsLogger",
    "get_git_info",
    "random_slug",
]


# ---------------------------------------------------------------------------
# Git info (cached on first call)
# ---------------------------------------------------------------------------

_git_info: dict[str, str | None] | None = None


def get_git_info() -> dict[str, str | None]:
    """Return ``{"git_hash": ..., "git_tag": ...}`` for the current working tree.

    Honors ``PAWN_GIT_HASH`` / ``PAWN_GIT_TAG`` env vars (set on runpod
    images where the container is built outside a git checkout). Result
    is cached for the lifetime of the process — git state doesn't
    change mid-run.
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
        except (subprocess.SubprocessError, OSError):
            git_hash = None

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
        except (subprocess.SubprocessError, OSError):
            git_tag = None

    _git_info = {"git_hash": git_hash, "git_tag": git_tag}
    return _git_info


def _reset_git_info_cache() -> None:
    """Test hook — clear the cached git info so a test can re-exercise
    the discovery path with different env vars."""
    global _git_info
    _git_info = None


# ---------------------------------------------------------------------------
# Slug generator (verbatim from v1 — durable runs share the same word lists)
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
    """Return a fresh ``adjective-animal`` slug for run-dir disambiguation."""
    return f"{random.choice(_ADJECTIVES)}-{random.choice(_ANIMALS)}"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with ``None`` so the JSON output
    stays parseable.

    Walking lists/dicts catches metrics nested inside structured fields
    (e.g. ``"film/gamma_norm": {"L0": nan, "L1": 1.23}``).
    """
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize(v) for v in obj]
    return obj


def _json_default(obj: Any) -> Any:
    """Best-effort JSON serializer for types not natively supported.

    Handles JAX / numpy scalar arrays (``.item()``), Path (``str``),
    fallback to ``str(obj)``. Deliberately does **not** import torch —
    the v1 ``torch.Tensor`` branch is gone (the v2 stack writes JAX
    scalars only).

    Critically: re-sanitises the result of ``.item()`` for non-finite
    floats. ``_sanitize`` ran *before* serialisation against the raw
    PyTree (where the value was a JAX/numpy scalar, not a Python float),
    so a ``jnp.array(float('nan')).item()`` produces a Python NaN that
    would otherwise become literal ``NaN`` in the JSONL — breaking the
    JSON-parseable contract. The catch here is the only place that
    sees the scalar-converted Python value.
    """
    if hasattr(obj, "item"):  # numpy / JAX scalar with item()
        try:
            scalar = obj.item()
        except Exception:
            return str(obj)
        if isinstance(scalar, float) and (math.isnan(scalar) or math.isinf(scalar)):
            return None
        return scalar
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# GPU memory query (shell-out, cached at process start)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _gpu_backend() -> str | None:
    """Decide which GPU query CLI to use. Cached for the process lifetime.

    Returns ``"nvidia-smi"``, ``"rocm-smi"``, or ``None`` (no GPU CLI
    available — caller falls back to skipping GPU stats).
    """
    if shutil.which("nvidia-smi"):
        return "nvidia-smi"
    if shutil.which("rocm-smi"):
        return "rocm-smi"
    return None


def _reset_gpu_backend_cache() -> None:
    """Test hook — clear the cached backend so a test can swap PATH and
    re-exercise discovery."""
    _gpu_backend.cache_clear()


def _query_nvidia_smi() -> dict[str, float] | None:
    """Return ``{"gpu_used_gb": ..., "gpu_total_gb": ...}`` from nvidia-smi.

    Uses ``--query-gpu=memory.used,memory.total --format=csv,noheader,nounits``
    which returns one row per GPU; we report the first GPU only (a
    single-GPU training run is the v2 norm).
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 2:
        return None
    # Some nvidia-smi versions emit "4096.00" instead of "4096" even with
    # `--format=...,nounits`. Coerce via float() first so both forms work.
    try:
        used_mb = int(float(parts[0]))
        total_mb = int(float(parts[1]))
    except ValueError:
        return None
    return {
        "gpu_used_gb": round(used_mb / 1024, 3),
        "gpu_total_gb": round(total_mb / 1024, 3),
    }


def _query_rocm_smi() -> dict[str, float] | None:
    """Return ``{"gpu_used_gb": ..., "gpu_total_gb": ...}`` from rocm-smi.

    Uses ``rocm-smi --showmeminfo VRAM --json``. The JSON shape is
    ``{"card0": {"VRAM Total Memory (B)": "...", "VRAM Total Used Memory (B)": "..."}}``.
    """
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "VRAM", "--json"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    for card_data in data.values():
        if not isinstance(card_data, dict):
            continue
        try:
            total = int(card_data.get("VRAM Total Memory (B)", 0))
            used = int(card_data.get("VRAM Total Used Memory (B)", 0))
        except (TypeError, ValueError):
            continue
        if total <= 0:
            continue
        return {
            "gpu_used_gb": round(used / (1024 ** 3), 3),
            "gpu_total_gb": round(total / (1024 ** 3), 3),
        }
    return None


def _query_gpu_stats() -> dict[str, float] | None:
    """Dispatch to the cached backend; return None if no backend available
    or the query failed."""
    backend = _gpu_backend()
    if backend == "nvidia-smi":
        return _query_nvidia_smi()
    if backend == "rocm-smi":
        return _query_rocm_smi()
    return None


def _query_jax_memory_stats() -> dict[str, float] | None:
    """Return process-side JAX allocator memory in v1-parity field names.

    v1 used :func:`torch.cuda.{max_memory_allocated, memory_reserved,
    memory_allocated}` which report this process's torch allocator.
    JAX exposes the equivalent via ``device.memory_stats()``:

    - ``peak_bytes_in_use``  → ``gpu_peak_gb``   (cumulative since
                                                  process start)
    - ``pool_bytes``         → ``gpu_reserved_gb`` (allocator pool size)
    - ``bytes_in_use``       → ``gpu_current_gb`` (currently allocated)

    Returns ``None`` on the CPU backend, or if the JAX import / device
    query fails. Best-effort: the smi-side numbers
    (:func:`_query_gpu_stats`) still cover the system-wide view.

    Cumulative-peak semantics differ slightly from v1 — v1 reset the
    peak counter after every record so each row's ``gpu_peak_gb`` was a
    per-record max. JAX doesn't expose a ``reset_peak`` knob, so v2's
    field is monotonic-since-process-start. The field *name* matches v1
    so existing dashboards parse it unchanged.
    """
    try:
        import jax
    except ImportError:
        return None
    try:
        devices = jax.devices()
    except Exception:  # noqa: BLE001 — JAX raises a variety of types here
        return None
    if not devices:
        return None
    dev = devices[0]
    if not hasattr(dev, "memory_stats"):
        return None
    try:
        stats = dev.memory_stats()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(stats, dict):
        return None
    gib = 1024 ** 3
    return {
        "gpu_peak_gb": round(stats.get("peak_bytes_in_use", 0) / gib, 3),
        "gpu_reserved_gb": round(stats.get("pool_bytes", 0) / gib, 3),
        "gpu_current_gb": round(stats.get("bytes_in_use", 0) / gib, 3),
    }


# Fields the logger stamps itself — callers can't override them via kwargs.
# Reserved so a stale `type=` (or, worse, a `type` field inside a pydantic
# run-config dump-then-unpack) can't silently replace the dashboard's
# record-type discriminator.
_RESERVED_KWARGS: frozenset[str] = frozenset({
    "type", "timestamp", "slug", "hostname",
    "git_hash", "git_tag", "elapsed",
})


def _reject_reserved_kwargs(kwargs: dict[str, Any]) -> None:
    """Raise ValueError if a caller passes any reserved field as a kwarg.

    The reserved set is everything :meth:`_add_baseline` stamps + the
    `type` discriminator. We refuse silently-overwritable inputs at the
    public-API boundary instead of letting them through and counting on
    the writer to reassert (the writer used to set `type` first, then
    `update(kwargs)`, so a `type="oops"` kwarg would silently win).
    """
    bad = set(kwargs.keys()) & _RESERVED_KWARGS
    if bad:
        raise ValueError(
            f"MetricsLogger reserved field(s) cannot be passed as kwargs: "
            f"{sorted(bad)}. These are stamped automatically — drop them "
            f"from the call site (or rename your field, e.g. `type` → "
            f"`run_type`)."
        )


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------


class MetricsLogger:
    """JSONL metrics logger — the only path metrics take to disk.

    Every record carries the baseline contract described at module
    scope. Train / val records additionally carry CPU and GPU memory
    stats. Records are sanitised (NaN / Inf → null) and flushed
    per-write so the dashboard can tail the file safely.

    Usage::

        with MetricsLogger("logs", run_prefix="lora", device="cuda") as log:
            log.log_config(run_type="adapter", ...)
            log.log_train(step=100, loss=3.5)
            log.log_val(step=100, loss=3.6)

    Constructor args:
        log_dir: parent directory for all runs.
        run_prefix: leading component of the run-dir name.
        device: ``"cpu"`` skips GPU stats entirely; anything else
            triggers the shell-out backend.
        slug: optional explicit slug (auto-generated if None).
        suffix: optional extra component of the run-dir name (e.g.
            variant name or strategy).
    """

    def __init__(
        self,
        log_dir: str | Path,
        run_prefix: str = "run",
        device: str = "cpu",
        slug: str | None = None,
        suffix: str = "",
    ) -> None:
        self.slug = slug or random_slug()
        # Microsecond precision — see module docstring; matches plan §10 S4.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        parts: list[str] = [run_prefix, ts]
        if suffix:
            parts.append(suffix)
        parts.append(self.slug)
        dir_name = "_".join(parts)

        self.run_dir = Path(log_dir) / dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._file = open(self.metrics_path, "a", encoding="utf-8")
        self._proc = psutil.Process()
        # psutil documents that `cpu_percent(interval=None)` returns 0.0 on
        # its first per-process call (no baseline yet). Pre-warm so the
        # first log_train record gets a real reading.
        psutil.cpu_percent(interval=None, percpu=True)
        self._device = device
        self._start_time = time.time()
        self._closed = False

    @property
    def path(self) -> Path:
        return self.metrics_path

    # ---- public log_* methods ----------------------------------------------

    def log_config(self, **kwargs: Any) -> None:
        """Write a ``type=config`` record. Typically called once at the
        start of a run with the resolved run-config + metadata."""
        _reject_reserved_kwargs(kwargs)
        record: dict[str, Any] = {}
        record.update(kwargs)
        record["type"] = "config"  # set AFTER update so the discriminator wins
        self._add_baseline(record, include_resources=False)
        self._write(record)

    def log_train(self, step: int, **metrics: Any) -> None:
        """Write a ``type=train`` record.

        Mandatory: ``step`` (required positional). Standard kwargs:
        ``lr``, ``loss``, ``accuracy``, ``step_time``, etc. Free-form
        adapter / scheduler / probe fields pass through as-is.
        """
        _reject_reserved_kwargs(metrics)
        record: dict[str, Any] = {}
        record.update(metrics)
        record["type"] = "train"
        record["step"] = step
        self._add_baseline(record, include_resources=True)
        self._write(record)

    def log_val(self, step: int, **metrics: Any) -> None:
        """Write a ``type=val`` record.

        Same shape as :meth:`log_train` — the discriminator is what
        the dashboard splits on.
        """
        _reject_reserved_kwargs(metrics)
        record: dict[str, Any] = {}
        record.update(metrics)
        record["type"] = "val"
        record["step"] = step
        self._add_baseline(record, include_resources=True)
        self._write(record)

    def write_config_json(self, **kwargs: Any) -> Path:
        """Write a sibling ``config.json`` alongside ``metrics.jsonl``
        (the run config persisted for checkpoint bundling). Returns the
        path."""
        data: dict[str, Any] = {}
        data.update(kwargs)
        data["slug"] = self.slug
        data.update(get_git_info())
        path = self.run_dir / "config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_sanitize(data), f, indent=2, default=_json_default)
        return path

    # ---- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Flush and close the JSONL file. Idempotent."""
        if self._closed:
            return
        try:
            self._file.flush()
            self._file.close()
        finally:
            self._closed = True

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup if the caller forgot a `with` block.
        try:
            self.close()
        except Exception:
            pass

    # ---- internals ---------------------------------------------------------

    def _add_baseline(self, record: dict[str, Any], *, include_resources: bool) -> None:
        """Stamp the universal baseline fields (timestamp / slug /
        hostname / git_* / elapsed). On train/val records additionally
        attach host + GPU memory stats."""
        record["timestamp"] = datetime.now().isoformat()
        record["elapsed"] = round(time.time() - self._start_time, 3)
        record["slug"] = self.slug
        record["hostname"] = socket.gethostname()
        record.update(get_git_info())
        if include_resources:
            self._add_resource_stats(record)

    def _add_resource_stats(self, record: dict[str, Any]) -> None:
        """Attach host CPU / system memory + GPU memory stats. GPU is
        skipped entirely when ``device == "cpu"`` so a CPU-only test
        environment doesn't pay the (failing) shell-out cost."""
        mem_info = self._proc.memory_info()
        sys_mem = psutil.virtual_memory()
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        record["mem/system_rss_gb"] = round(mem_info.rss / (1024 ** 3), 3)
        record["mem/system_used_gb"] = round(sys_mem.used / (1024 ** 3), 3)
        record["mem/system_total_gb"] = round(sys_mem.total / (1024 ** 3), 3)
        record["mem/cpu_percent"] = round(sum(per_cpu) if per_cpu else 0.0, 1)
        if self._device == "cpu":
            return
        # v1-parity allocator stats (process-side, JAX backend). The
        # field names match v1's torch path so existing dashboards key
        # off them unchanged. See `_query_jax_memory_stats` for the
        # cumulative-peak semantic note.
        jax_stats = _query_jax_memory_stats()
        if jax_stats is not None:
            record["mem/gpu_peak_gb"] = jax_stats["gpu_peak_gb"]
            record["mem/gpu_reserved_gb"] = jax_stats["gpu_reserved_gb"]
            record["mem/gpu_current_gb"] = jax_stats["gpu_current_gb"]
        # System-wide view via *-smi (covers other processes on the
        # GPU). Retained alongside the v1-parity fields — the
        # dashboard tolerates additional fields, and the smi numbers
        # were the original v2 reporting surface.
        gpu = _query_gpu_stats()
        if gpu is None:
            return
        record["mem/gpu_used_gb"] = gpu["gpu_used_gb"]
        record["mem/gpu_total_gb"] = gpu["gpu_total_gb"]

    def _write(self, record: dict[str, Any]) -> None:
        """Sanitise + serialise + flush. The per-write flush keeps the
        dashboard's tail consistent and lets a trainer crash leave a
        valid (truncated) JSONL on disk."""
        sanitised = _sanitize(record)
        line = json.dumps(sanitised, default=_json_default) + "\n"
        self._file.write(line)
        self._file.flush()
