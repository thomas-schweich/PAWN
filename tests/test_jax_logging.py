"""Tests for the restored MetricsLogger (S4.C2 of the JAX migration).

Pins the v1-parity contract from ``docs/jax-migration.md`` §8.2: the
``type: "config" | "train" | "val"`` discriminator, baseline fields
on every record, NaN/Inf → ``null`` sanitisation, per-record flush,
slug-based run-dir naming.

The GPU-memory shell-out (``nvidia-smi`` / ``rocm-smi``) is exercised
with monkeypatched subprocess to keep the suite hermetic.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from pawn.logging import (
    MetricsLogger,
    _discover_gpu_tool,
    _read_gpu_memory_gb,
    _sanitize,
    get_git_info,
    random_slug,
)


def _read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Helpers (slug + git + sanitise)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_random_slug_format() -> None:
    slug = random_slug()
    parts = slug.split("-")
    assert len(parts) == 2, f"slug {slug!r} should be two-word"
    assert all(p.isalpha() and p.islower() for p in parts)


@pytest.mark.unit
def test_get_git_info_returns_dict_with_two_keys() -> None:
    info = get_git_info()
    assert set(info.keys()) == {"git_hash", "git_tag"}


@pytest.mark.unit
def test_get_git_info_honors_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAWN_GIT_HASH", "deadbeef")
    monkeypatch.setenv("PAWN_GIT_TAG", "v0.2.0")
    # Reset module-level cache by re-importing — get_git_info caches
    # on the first call.
    import importlib

    import pawn.logging as logging_mod

    importlib.reload(logging_mod)
    info = logging_mod.get_git_info()
    assert info == {"git_hash": "deadbeef", "git_tag": "v0.2.0"}


@pytest.mark.unit
def test_sanitize_nan_inf_to_none() -> None:
    out = _sanitize({"a": float("nan"), "b": float("inf"), "c": float("-inf")})
    assert out == {"a": None, "b": None, "c": None}
    # Recurses through dicts / lists / tuples
    nested = _sanitize(
        {"d": [1.0, float("nan"), {"e": float("inf")}], "f": (float("nan"),)}
    )
    assert nested == {"d": [1.0, None, {"e": None}], "f": [None]}
    # Finite floats pass through
    assert _sanitize(1.5) == 1.5
    assert _sanitize("hello") == "hello"


# ---------------------------------------------------------------------------
# Run-dir naming
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_dir_layout(tmp_path: Path) -> None:
    with MetricsLogger(tmp_path, run_prefix="jax_run", slug="zesty-osprey") as lg:
        assert lg.run_dir.exists()
        assert lg.run_dir.parent == tmp_path
        # Pattern: <prefix>_<YYYYMMDD>_<HHMMSS>_<microseconds>_<slug>
        parts = lg.run_dir.name.split("_")
        assert parts[0] == "jax"
        assert parts[1] == "run"
        # Date + time + microseconds = 3 numeric blocks
        assert parts[2].isdigit() and len(parts[2]) == 8  # YYYYMMDD
        assert parts[3].isdigit() and len(parts[3]) == 6  # HHMMSS
        assert parts[4].isdigit() and len(parts[4]) == 6  # microseconds
        assert "zesty-osprey" in lg.run_dir.name


@pytest.mark.unit
def test_run_dir_includes_suffix(tmp_path: Path) -> None:
    with MetricsLogger(tmp_path, run_prefix="jax_run", suffix="lora_rank4") as lg:
        assert "lora_rank4" in lg.run_dir.name


# ---------------------------------------------------------------------------
# Type discriminator + record schema
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_log_config_writes_type_config(tmp_path: Path) -> None:
    with MetricsLogger(tmp_path, run_prefix="jax_run") as lg:
        lg.log_config(run_type="pretrain", supernet="tiny", total_steps=100)
    rows = _read_rows(lg.metrics_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["type"] == "config"
    assert row["run_type"] == "pretrain"
    assert row["supernet"] == "tiny"
    # Baseline fields on a config record
    assert "slug" in row
    assert "hostname" in row
    assert "timestamp" in row
    assert "run_dir" in row
    assert "git_hash" in row
    assert "git_tag" in row


@pytest.mark.unit
def test_log_train_log_val_type_discriminator(tmp_path: Path) -> None:
    with MetricsLogger(tmp_path, run_prefix="jax_run") as lg:
        lg.log_train(step=10, lr=3e-4, loss=3.5)
        lg.log_val(step=10, val_loss=3.4)
    rows = _read_rows(lg.metrics_path)
    assert [r["type"] for r in rows] == ["train", "val"]
    assert rows[0]["step"] == 10
    assert rows[1]["step"] == 10
    # Train + val records include the host stats
    for r in rows:
        assert "mem/system_rss_gb" in r
        assert "mem/cpu_percent" in r
        assert "timestamp" in r
        assert "elapsed" in r


# ---------------------------------------------------------------------------
# NaN / Inf sanitisation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_nan_inf_in_metrics_become_null(tmp_path: Path) -> None:
    with MetricsLogger(tmp_path, run_prefix="jax_run") as lg:
        lg.log_train(
            step=1,
            loss=float("nan"),
            grad_norm=float("inf"),
            ok=3.5,
        )
    raw = lg.metrics_path.read_text()
    # Records must be RFC-7159 valid — the literal "NaN" / "Infinity"
    # are not allowed in JSON. json.loads on the raw record must work.
    row = json.loads(raw.strip())
    assert row["loss"] is None
    assert row["grad_norm"] is None
    assert row["ok"] == 3.5


# ---------------------------------------------------------------------------
# Per-record flush
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_per_record_flush(tmp_path: Path) -> None:
    """log_train flushes the file before returning — a SIGKILL after
    log_train must leave the row on disk."""
    lg = MetricsLogger(tmp_path, run_prefix="jax_run")
    lg.log_train(step=1, loss=3.0)
    # Read WITHOUT closing the logger — content must already be flushed.
    rows = _read_rows(lg.metrics_path)
    assert len(rows) == 1
    assert rows[0]["loss"] == 3.0
    lg.close()


# ---------------------------------------------------------------------------
# write_config_json
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_config_json(tmp_path: Path) -> None:
    with MetricsLogger(tmp_path, run_prefix="jax_run") as lg:
        path = lg.write_config_json(supernet="tiny", lr=3e-4)
    assert path == lg.run_dir / "config.json"
    data = json.loads(path.read_text())
    assert data["supernet"] == "tiny"
    assert data["lr"] == 3e-4
    assert "slug" in data
    assert "git_hash" in data


# ---------------------------------------------------------------------------
# GPU-memory shell-out (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_gpu_memory_from_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    import pawn.logging as mod

    # Reset the discovery cache
    if hasattr(mod._discover_gpu_tool, "_cached"):
        delattr(mod._discover_gpu_tool, "_cached")

    monkeypatch.setattr(
        mod.shutil,
        "which",
        lambda binary: "/usr/bin/nvidia-smi" if binary == "nvidia-smi" else None,
    )

    fake_output = "1024, 8192\n"  # MB

    def fake_check_output(*args, **kwargs):
        return fake_output

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    result = _read_gpu_memory_gb()
    assert result is not None
    used_gb, total_gb = result
    assert used_gb == pytest.approx(1024 / 1024, abs=1e-3)
    assert total_gb == pytest.approx(8192 / 1024, abs=1e-3)


@pytest.mark.unit
def test_gpu_memory_from_rocm_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    import pawn.logging as mod

    if hasattr(mod._discover_gpu_tool, "_cached"):
        delattr(mod._discover_gpu_tool, "_cached")

    monkeypatch.setattr(
        mod.shutil,
        "which",
        lambda binary: "/opt/rocm/bin/rocm-smi" if binary == "rocm-smi" else None,
    )

    fake_json = json.dumps(
        {
            "card0": {
                "VRAM Total Used Memory (B)": str(2 * 1024**3),
                "VRAM Total Memory (B)": str(8 * 1024**3),
            }
        }
    )

    def fake_check_output(*args, **kwargs):
        return fake_json

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    result = _read_gpu_memory_gb()
    assert result is not None
    used_gb, total_gb = result
    assert used_gb == pytest.approx(2.0, abs=1e-3)
    assert total_gb == pytest.approx(8.0, abs=1e-3)


@pytest.mark.unit
def test_gpu_memory_absent_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    import pawn.logging as mod

    if hasattr(mod._discover_gpu_tool, "_cached"):
        delattr(mod._discover_gpu_tool, "_cached")

    monkeypatch.setattr(mod.shutil, "which", lambda _: None)
    assert _read_gpu_memory_gb() is None


@pytest.mark.unit
def test_logger_does_not_emit_gpu_keys_with_device_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """device='cpu' (the unit-test default) must skip the GPU shell-out
    entirely — otherwise a slow `which` call adds latency to every
    train/val record on a GPU-less CI box."""
    import pawn.logging as mod

    # Make absolutely sure no GPU tool is invoked
    called = []

    monkeypatch.setattr(
        mod, "_read_gpu_memory_gb", lambda: called.append(1) or None
    )

    with MetricsLogger(tmp_path, run_prefix="jax_run", device="cpu") as lg:
        lg.log_train(step=1, loss=3.0)

    assert called == []


# ---------------------------------------------------------------------------
# torch-freedom — pawn.logging must NOT import torch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_logging_module_does_not_import_torch() -> None:
    """The v1 ``MetricsLogger`` had a ``try: import torch`` shim for
    its GPU memory branch. The JAX surface must stay torch-free under
    a CPU-jax install — that's the §8.2 contract."""
    import pawn.logging as mod
    import sys

    # If pawn.logging pulled torch transitively, ``torch`` would be in
    # sys.modules by now.
    src = Path(mod.__file__).read_text()
    assert "import torch" not in src, (
        "pawn/logging.py must not import torch (see plan §8.2)"
    )
