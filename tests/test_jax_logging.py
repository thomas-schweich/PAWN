"""Tests for :mod:`pawn.logging` — the JSONL MetricsLogger.

Coverage per plan §10 S4 verification list:

- Run-dir naming includes microseconds (`<prefix>_<YYYYMMDD>_<HHMMSS>_<microseconds>_<suffix?>_<slug>`).
- Slug + git info baseline fields stamped on every record.
- `type` discriminator on config / train / val records.
- NaN / Inf sanitisation to `null`.
- Per-record `flush()` — readable while the logger is still open.
- `device="cpu"` skips the GPU shell-out entirely.
- `nvidia-smi` backend dispatch (mocked).
- `rocm-smi` backend dispatch (mocked).
- Backend selection cached at process start.
- Module imports in a CPU-only env without torch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import unittest.mock as mock
from pathlib import Path

import pytest

from pawn.logging import (
    _RESERVED_KWARGS,
    _json_default,
    MetricsLogger,
    _gpu_backend,
    _query_gpu_stats,
    _query_nvidia_smi,
    _query_rocm_smi,
    _reset_git_info_cache,
    _reset_gpu_backend_cache,
    _sanitize,
    get_git_info,
    random_slug,
)


# ---------------------------------------------------------------------------
# Run-dir naming + baseline contract
# ---------------------------------------------------------------------------


def test_run_dir_name_includes_microseconds(tmp_path: Path) -> None:
    """The plan §10 S4 dir-name format is
    `<prefix>_<YYYYMMDD>_<HHMMSS>_<microseconds>_<suffix?>_<slug>`."""
    logger = MetricsLogger(tmp_path, run_prefix="run", slug="zesty-puma")
    name = logger.run_dir.name
    parts = name.split("_")
    # ["run", "20260522", "203045", "123456", "zesty-puma"]
    assert parts[0] == "run"
    assert parts[1].isdigit() and len(parts[1]) == 8  # YYYYMMDD
    assert parts[2].isdigit() and len(parts[2]) == 6  # HHMMSS
    assert parts[3].isdigit() and len(parts[3]) == 6  # microseconds
    assert parts[-1] == "zesty-puma"
    logger.close()


def test_run_dir_name_includes_suffix(tmp_path: Path) -> None:
    logger = MetricsLogger(
        tmp_path, run_prefix="lora", slug="bold-fox", suffix="base"
    )
    name = logger.run_dir.name
    parts = name.split("_")
    assert "base" in parts
    assert parts[-1] == "bold-fox"
    logger.close()


def test_run_dir_microseconds_disambiguate_rapid_succession(tmp_path: Path) -> None:
    """Two loggers spawned with the same prefix AND same slug land in
    different dirs — the microseconds field is the only disambiguator
    available in that case (parallel sweep children share both)."""
    log1 = MetricsLogger(tmp_path, run_prefix="r", slug="zesty-puma")
    log2 = MetricsLogger(tmp_path, run_prefix="r", slug="zesty-puma")
    assert log1.run_dir != log2.run_dir
    # Both dirs share the same prefix and slug; only the timestamp
    # (down to microseconds) distinguishes them.
    p1 = log1.run_dir.name.split("_")
    p2 = log2.run_dir.name.split("_")
    assert p1[0] == p2[0] == "r"
    assert p1[-1] == p2[-1] == "zesty-puma"
    # Microsecond field (index 3) is what differs (or in the unlikely
    # case the seconds field rolled over, that one — either is fine).
    assert p1[3] != p2[3] or p1[2] != p2[2]
    log1.close()
    log2.close()


def test_random_slug_format() -> None:
    slug = random_slug()
    assert "-" in slug
    adj, animal = slug.split("-")
    assert adj and animal


# ---------------------------------------------------------------------------
# Type discriminator + record contents
# ---------------------------------------------------------------------------


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_log_config_writes_type_config_record(tmp_path: Path) -> None:
    with MetricsLogger(tmp_path, slug="s") as logger:
        logger.log_config(run_type="pretrain", lr=3e-4, lora_rank=None)
    records = _read_records(logger.path)
    assert len(records) == 1
    assert records[0]["type"] == "config"
    assert records[0]["run_type"] == "pretrain"
    assert records[0]["lr"] == 3e-4
    assert "lora_rank" in records[0]


def test_log_train_writes_type_train_record(tmp_path: Path) -> None:
    with MetricsLogger(tmp_path, slug="s") as logger:
        logger.log_train(step=100, lr=3e-4, loss=2.5, accuracy=0.05)
    records = _read_records(logger.path)
    assert records[0]["type"] == "train"
    assert records[0]["step"] == 100
    assert records[0]["loss"] == 2.5


def test_log_val_writes_type_val_record(tmp_path: Path) -> None:
    with MetricsLogger(tmp_path, slug="s") as logger:
        logger.log_val(step=100, loss=2.7, accuracy=0.04)
    records = _read_records(logger.path)
    assert records[0]["type"] == "val"
    assert records[0]["loss"] == 2.7


def test_all_records_carry_baseline_fields(tmp_path: Path) -> None:
    """`type`, `timestamp`, `slug`, `hostname`, `git_hash`/`git_tag`,
    `elapsed` are stamped on every record per the v1 contract preserved
    by plan §7."""
    os.environ["PAWN_GIT_HASH"] = "abcdef1234"
    os.environ["PAWN_GIT_TAG"] = "v2-test"
    _reset_git_info_cache()
    try:
        with MetricsLogger(tmp_path, slug="zesty-puma") as logger:
            logger.log_config(run_type="pretrain")
            logger.log_train(step=1, loss=1.0)
            logger.log_val(step=1, loss=1.1)
        records = _read_records(logger.path)
        for r in records:
            assert "type" in r
            assert "timestamp" in r
            assert r["slug"] == "zesty-puma"
            assert "hostname" in r
            assert r["git_hash"] == "abcdef1234"
            assert r["git_tag"] == "v2-test"
            assert "elapsed" in r
    finally:
        del os.environ["PAWN_GIT_HASH"]
        del os.environ["PAWN_GIT_TAG"]
        _reset_git_info_cache()


def test_train_val_records_carry_resource_stats(tmp_path: Path) -> None:
    """train + val records both include CPU + system memory stats (the
    paths share `_add_baseline(include_resources=True)`)."""
    with MetricsLogger(tmp_path, slug="s", device="cpu") as logger:
        logger.log_train(step=1, loss=1.0)
        logger.log_val(step=1, loss=1.1)
    train_rec, val_rec = _read_records(logger.path)
    for r in (train_rec, val_rec):
        assert "mem/system_rss_gb" in r
        assert "mem/system_used_gb" in r
        assert "mem/system_total_gb" in r
        assert "mem/cpu_percent" in r


def test_config_records_do_not_carry_resource_stats(tmp_path: Path) -> None:
    """config records skip CPU/GPU stats — they're for hyperparameters,
    not in-flight resource pressure."""
    with MetricsLogger(tmp_path, slug="s") as logger:
        logger.log_config(run_type="pretrain")
    r = _read_records(logger.path)[0]
    assert "mem/system_rss_gb" not in r


# ---------------------------------------------------------------------------
# NaN / Inf sanitisation
# ---------------------------------------------------------------------------


def test_sanitize_replaces_nan_and_inf_with_none() -> None:
    """`_sanitize` is the pure function the writer uses."""
    assert _sanitize(float("nan")) is None
    assert _sanitize(float("inf")) is None
    assert _sanitize(float("-inf")) is None
    assert _sanitize(3.14) == 3.14
    assert _sanitize(0) == 0
    assert _sanitize("ok") == "ok"


def test_sanitize_walks_nested_structures() -> None:
    """Nested dict / list / tuple containers get cleaned recursively."""
    out = _sanitize({"a": float("nan"), "b": [1.0, float("inf"), {"c": float("nan")}]})
    assert out == {"a": None, "b": [1.0, None, {"c": None}]}


def test_log_train_sanitises_nan_in_record(tmp_path: Path) -> None:
    """NaN/Inf in a train record becomes `null` on disk so the JSON line
    stays parseable."""
    with MetricsLogger(tmp_path, slug="s", device="cpu") as logger:
        logger.log_train(step=1, loss=float("nan"), accuracy=float("inf"))
    line = logger.path.read_text().splitlines()[0]
    # Lines parse cleanly and the offending metrics are null.
    parsed = json.loads(line)
    assert parsed["loss"] is None
    assert parsed["accuracy"] is None


def test_log_train_sanitises_nested_nan(tmp_path: Path) -> None:
    """A nested structured metric (e.g. probe results) gets its NaNs
    cleaned too."""
    with MetricsLogger(tmp_path, slug="s", device="cpu") as logger:
        logger.log_train(
            step=1, probe={"L0": float("nan"), "L1": 0.4}
        )
    parsed = json.loads(logger.path.read_text().splitlines()[0])
    assert parsed["probe"] == {"L0": None, "L1": 0.4}


class _NaNScalar:
    """Mimics a JAX/numpy scalar wrapping a non-finite Python float."""

    def __init__(self, val: float) -> None:
        self._val = val

    def item(self) -> float:
        return self._val


def test_json_default_sanitises_nan_from_array_scalars(tmp_path: Path) -> None:
    """A JAX/numpy scalar wrapping NaN must serialize as `null`, not
    bare `NaN`. `_sanitize` runs against the raw PyTree (where the
    scalar isn't a Python float yet), so `_json_default` has to
    re-sanitise the result of `.item()`."""
    with MetricsLogger(tmp_path, slug="s", device="cpu") as logger:
        logger.log_train(step=1, loss=_NaNScalar(float("nan")),
                         accuracy=_NaNScalar(float("inf")))
    line = logger.path.read_text().splitlines()[0]
    # The line must be parseable as JSON (no bare NaN).
    parsed = json.loads(line)
    assert parsed["loss"] is None
    assert parsed["accuracy"] is None


def test_json_default_passes_finite_scalars_through(tmp_path: Path) -> None:
    """Finite JAX/numpy scalars unwrap to their Python value, not the
    stringified `str(obj)` fallback."""
    assert _json_default(_NaNScalar(1.5)) == 1.5


# ---------------------------------------------------------------------------
# Per-record flush — readable before close
# ---------------------------------------------------------------------------


def test_records_are_readable_before_close(tmp_path: Path) -> None:
    """Per-record flush means the dashboard can tail metrics.jsonl while
    the logger is still open."""
    logger = MetricsLogger(tmp_path, slug="s", device="cpu")
    logger.log_train(step=1, loss=1.0)
    logger.log_train(step=2, loss=0.9)
    # File is still open at this point — read directly without closing.
    records = _read_records(logger.path)
    assert len(records) == 2
    assert [r["step"] for r in records] == [1, 2]
    logger.close()


# ---------------------------------------------------------------------------
# GPU memory shell-out — device="cpu" skips
# ---------------------------------------------------------------------------


def test_device_cpu_skips_gpu_query(tmp_path: Path) -> None:
    """`device="cpu"` should never invoke the GPU CLI, even if nvidia-smi
    is present on PATH."""
    with mock.patch("pawn.logging._query_gpu_stats") as mock_query:
        with MetricsLogger(tmp_path, slug="s", device="cpu") as logger:
            logger.log_train(step=1, loss=1.0)
        mock_query.assert_not_called()


def test_device_cuda_invokes_gpu_query(tmp_path: Path) -> None:
    """`device != "cpu"` triggers the GPU shell-out backend."""
    with mock.patch(
        "pawn.logging._query_gpu_stats",
        return_value={"gpu_used_gb": 4.0, "gpu_total_gb": 24.0},
    ) as mock_query:
        with MetricsLogger(tmp_path, slug="s", device="cuda") as logger:
            logger.log_train(step=1, loss=1.0)
        assert mock_query.called
    parsed = json.loads(logger.path.read_text().splitlines()[0])
    assert parsed["mem/gpu_used_gb"] == 4.0
    assert parsed["mem/gpu_total_gb"] == 24.0


def test_device_cuda_with_no_gpu_backend_skips_gpu_fields(tmp_path: Path) -> None:
    """If the shell-out returns None (no CLI, query failed), no GPU
    fields are added — the run still logs cleanly."""
    with mock.patch("pawn.logging._query_gpu_stats", return_value=None):
        with MetricsLogger(tmp_path, slug="s", device="cuda") as logger:
            logger.log_train(step=1, loss=1.0)
    parsed = json.loads(logger.path.read_text().splitlines()[0])
    assert "mem/gpu_used_gb" not in parsed


# ---------------------------------------------------------------------------
# Backend dispatch — nvidia-smi vs rocm-smi mocks
# ---------------------------------------------------------------------------


def test_gpu_backend_picks_nvidia_smi_when_available() -> None:
    _reset_gpu_backend_cache()
    try:
        with mock.patch(
            "pawn.logging.shutil.which",
            side_effect=lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
        ):
            assert _gpu_backend() == "nvidia-smi"
    finally:
        _reset_gpu_backend_cache()


def test_gpu_backend_picks_rocm_smi_when_no_nvidia() -> None:
    _reset_gpu_backend_cache()
    try:
        with mock.patch(
            "pawn.logging.shutil.which",
            side_effect=lambda name: "/usr/bin/rocm-smi" if name == "rocm-smi" else None,
        ):
            assert _gpu_backend() == "rocm-smi"
    finally:
        _reset_gpu_backend_cache()


def test_gpu_backend_returns_none_when_neither_available() -> None:
    _reset_gpu_backend_cache()
    try:
        with mock.patch("pawn.logging.shutil.which", return_value=None):
            assert _gpu_backend() is None
    finally:
        _reset_gpu_backend_cache()


def test_gpu_backend_picks_nvidia_smi_when_both_available() -> None:
    """When both CLIs are on PATH, nvidia-smi wins (the priority order
    matters for ROCm boxes that happen to ship a stub nvidia-smi)."""
    _reset_gpu_backend_cache()
    try:
        with mock.patch(
            "pawn.logging.shutil.which",
            side_effect=lambda name: f"/usr/bin/{name}",
        ):
            assert _gpu_backend() == "nvidia-smi"
    finally:
        _reset_gpu_backend_cache()


def test_gpu_backend_is_cached_across_calls() -> None:
    """Second call to `_gpu_backend()` reuses the lru_cache result and
    doesn't re-invoke `shutil.which`."""
    _reset_gpu_backend_cache()
    try:
        with mock.patch(
            "pawn.logging.shutil.which",
            side_effect=lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
        ) as which:
            _gpu_backend()
            calls_after_first = which.call_count
            _gpu_backend()
            assert which.call_count == calls_after_first
    finally:
        _reset_gpu_backend_cache()


def test_query_nvidia_smi_parses_csv_output() -> None:
    """The csv backend returns "<used_mb>, <total_mb>"."""
    with mock.patch(
        "pawn.logging.subprocess.check_output",
        return_value="4096, 24576\n",
    ):
        out = _query_nvidia_smi()
    assert out is not None
    assert out["gpu_used_gb"] == round(4096 / 1024, 3)
    assert out["gpu_total_gb"] == round(24576 / 1024, 3)


def test_query_nvidia_smi_returns_none_on_failure() -> None:
    with mock.patch(
        "pawn.logging.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
    ):
        assert _query_nvidia_smi() is None


def test_query_nvidia_smi_returns_none_on_malformed_output() -> None:
    with mock.patch(
        "pawn.logging.subprocess.check_output",
        return_value="some garbage that's not csv\n",
    ):
        assert _query_nvidia_smi() is None


def test_query_nvidia_smi_tolerates_float_format() -> None:
    """Some nvidia-smi driver versions emit `"4096.00, 24576.00"` even
    with `--format=...,nounits`. The parser coerces via `float()` so
    both `4096` and `4096.00` work."""
    with mock.patch(
        "pawn.logging.subprocess.check_output",
        return_value="4096.00, 24576.00\n",
    ):
        out = _query_nvidia_smi()
    assert out is not None
    assert out["gpu_used_gb"] == round(4096 / 1024, 3)


def test_query_rocm_smi_parses_json_output() -> None:
    """The rocm-smi JSON shape is `{"card0": {"VRAM Total Memory (B)":
    "...", "VRAM Total Used Memory (B)": "..."}}`."""
    rocm_payload = json.dumps(
        {
            "card0": {
                "VRAM Total Memory (B)": str(24 * 1024 ** 3),
                "VRAM Total Used Memory (B)": str(8 * 1024 ** 3),
            }
        }
    )
    with mock.patch(
        "pawn.logging.subprocess.check_output", return_value=rocm_payload
    ):
        out = _query_rocm_smi()
    assert out is not None
    assert out["gpu_used_gb"] == 8.0
    assert out["gpu_total_gb"] == 24.0


def test_query_rocm_smi_returns_none_on_failure() -> None:
    with mock.patch(
        "pawn.logging.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "rocm-smi"),
    ):
        assert _query_rocm_smi() is None


def test_query_rocm_smi_returns_none_on_bad_json() -> None:
    with mock.patch(
        "pawn.logging.subprocess.check_output", return_value="not json"
    ):
        assert _query_rocm_smi() is None


def test_query_gpu_stats_dispatches_through_backend() -> None:
    """The dispatcher uses the cached backend choice."""
    _reset_gpu_backend_cache()
    try:
        with mock.patch(
            "pawn.logging.shutil.which",
            side_effect=lambda name: "/x" if name == "nvidia-smi" else None,
        ):
            with mock.patch(
                "pawn.logging._query_nvidia_smi",
                return_value={"gpu_used_gb": 1.0, "gpu_total_gb": 2.0},
            ) as q:
                out = _query_gpu_stats()
            assert q.called
        assert out == {"gpu_used_gb": 1.0, "gpu_total_gb": 2.0}
    finally:
        _reset_gpu_backend_cache()


# ---------------------------------------------------------------------------
# Lightweight import contract (no torch, no JAX)
# ---------------------------------------------------------------------------


def test_logging_module_imports_without_torch_or_jax() -> None:
    """`pawn.logging` is torch-free per plan §10 S4. Verify in a fresh
    subprocess that the import doesn't pull in torch / jax / equinox."""
    probe = textwrap.dedent(
        """
        import sys
        import pawn.logging  # noqa: F401
        heavy = [
            m for m in sys.modules
            if m.split('.')[0] in ('jax', 'jaxlib', 'torch', 'equinox', 'optax')
        ]
        if heavy:
            print('LEAKED:' + ','.join(sorted(heavy)))
            raise SystemExit(1)
        print('OK')
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, (
        f"pawn.logging dragged in heavy modules:\n"
        f"  stdout: {result.stdout.strip()}\n  stderr: {result.stderr.strip()}"
    )
    assert result.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# Git info caching + env-var override
# ---------------------------------------------------------------------------


def test_git_info_honors_env_vars() -> None:
    os.environ["PAWN_GIT_HASH"] = "deadbeef"
    os.environ["PAWN_GIT_TAG"] = "v9.9.9"
    _reset_git_info_cache()
    try:
        info = get_git_info()
        assert info["git_hash"] == "deadbeef"
        assert info["git_tag"] == "v9.9.9"
    finally:
        del os.environ["PAWN_GIT_HASH"]
        del os.environ["PAWN_GIT_TAG"]
        _reset_git_info_cache()


def test_git_info_returns_none_when_no_git_or_env() -> None:
    """If git fails and no env vars set, return Nones (not crash)."""
    _reset_git_info_cache()
    saved = (
        os.environ.pop("PAWN_GIT_HASH", None),
        os.environ.pop("PAWN_GIT_TAG", None),
    )
    try:
        with mock.patch(
            "pawn.logging.subprocess.check_output",
            side_effect=subprocess.CalledProcessError(1, "git"),
        ):
            info = get_git_info()
        # All three branches should fail; result is a dict with two Nones.
        assert info == {"git_hash": None, "git_tag": None}
    finally:
        if saved[0] is not None:
            os.environ["PAWN_GIT_HASH"] = saved[0]
        if saved[1] is not None:
            os.environ["PAWN_GIT_TAG"] = saved[1]
        _reset_git_info_cache()


# ---------------------------------------------------------------------------
# Close / context manager / idempotency
# ---------------------------------------------------------------------------


def test_close_is_idempotent(tmp_path: Path) -> None:
    logger = MetricsLogger(tmp_path, slug="s")
    logger.close()
    logger.close()  # no exception


def test_context_manager_closes_on_exit(tmp_path: Path) -> None:
    with MetricsLogger(tmp_path, slug="s") as logger:
        logger.log_config(run_type="pretrain")
    # File handle is closed after exit. The internal flag flips even
    # though `_closed` is a private attribute (we read it for the
    # contract check; callers shouldn't).
    assert logger._closed is True
    assert logger._file.closed


def test_log_after_close_raises(tmp_path: Path) -> None:
    """Logging after `close()` raises (rather than silently dropping).
    Important for SIGTERM handlers: shutdown code that races with a
    log call should see a loud error, not a silent metric loss."""
    logger = MetricsLogger(tmp_path, slug="s", device="cpu")
    logger.log_train(step=1, loss=1.0)
    logger.close()
    with pytest.raises(ValueError, match="closed file"):
        logger.log_train(step=2, loss=0.9)


# ---------------------------------------------------------------------------
# Reserved-kwarg defense (type discriminator can't be silently overridden)
# ---------------------------------------------------------------------------


def test_log_methods_reject_reserved_kwargs(tmp_path: Path) -> None:
    """Passing any reserved field as a kwarg raises — the dashboard
    splits records on `type`, and a stale `type="oops"` from a
    pydantic config dump would silently break that. Rejecting at the
    public API boundary keeps the discriminator load-bearing.
    """
    with MetricsLogger(tmp_path, slug="s", device="cpu") as logger:
        for field in _RESERVED_KWARGS:
            with pytest.raises(ValueError, match="reserved field"):
                logger.log_config(**{field: "x"})
            with pytest.raises(ValueError, match="reserved field"):
                logger.log_train(step=1, **{field: "x"})
            with pytest.raises(ValueError, match="reserved field"):
                logger.log_val(step=1, **{field: "x"})


def test_log_train_does_not_overwrite_step(tmp_path: Path) -> None:
    """Even though `step` isn't a reserved kwarg (it's a positional),
    a caller can't sneak a different step in via a config dict —
    the positional is the source of truth."""
    with MetricsLogger(tmp_path, slug="s", device="cpu") as logger:
        logger.log_train(step=42, loss=1.0)
    parsed = json.loads(logger.path.read_text().splitlines()[0])
    assert parsed["step"] == 42


# ---------------------------------------------------------------------------
# write_config_json
# ---------------------------------------------------------------------------


def test_write_config_json_creates_sibling_file(tmp_path: Path) -> None:
    os.environ["PAWN_GIT_HASH"] = "deadbeefabc"
    os.environ["PAWN_GIT_TAG"] = "v-test"
    _reset_git_info_cache()
    try:
        with MetricsLogger(tmp_path, slug="zesty-puma") as logger:
            path = logger.write_config_json(run_type="pretrain", lr=3e-4)
        assert path == logger.run_dir / "config.json"
        raw = json.loads(path.read_text())
        assert raw["run_type"] == "pretrain"
        assert raw["lr"] == 3e-4
        assert raw["slug"] == "zesty-puma"
        # git_hash / git_tag are stamped into config.json too — important
        # for the published-model-card pipeline that grabs these via
        # `read_checkpoint_metadata` downstream.
        assert raw["git_hash"] == "deadbeefabc"
        assert raw["git_tag"] == "v-test"
    finally:
        del os.environ["PAWN_GIT_HASH"]
        del os.environ["PAWN_GIT_TAG"]
        _reset_git_info_cache()
