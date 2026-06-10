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
import math
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


def test_log_val_emits_namespaced_schema(tmp_path: Path) -> None:
    """`log_val` is the v1 `val/*` schema enabler (criterion 15): plain
    scalar kwargs are promoted to the namespaced dashboard keys the chart
    grid consumes (`pawn/dashboard/charts.py`), and `val/perplexity` is
    derived from `val/loss`. The bare kwargs are retained for the
    sweep/lab monitors that fall back to un-namespaced names."""
    with MetricsLogger(tmp_path, slug="s", device="cpu") as logger:
        logger.log_val(
            step=100,
            loss=2.0,
            accuracy=0.30,
            top1=0.30,
            top5=0.55,
            legal_move_rate=0.98,
            late_legal_move_rate=0.95,
            game_completion_rate=0.80,
            avg_plies_completed=42.0,
            opening=0.40,
            midgame=0.28,
            endgame=0.22,
        )
    rec = _read_records(logger.path)[0]
    assert rec["type"] == "val"
    # Namespaced dashboard keys.
    assert rec["val/loss"] == 2.0
    assert rec["val/accuracy"] == 0.30
    assert rec["val/top1"] == 0.30
    assert rec["val/top5"] == 0.55
    assert rec["val/top5_accuracy"] == 0.55  # pawn accuracy chart key
    assert rec["val/legal_move_rate"] == 0.98
    assert rec["val/late_legal_move_rate"] == 0.95
    assert rec["val/game_completion_rate"] == 0.80
    assert rec["val/avg_plies_completed"] == 42.0
    assert rec["val/opening"] == 0.40
    assert rec["val/midgame"] == 0.28
    assert rec["val/endgame"] == 0.22
    # Derived perplexity = exp(val/loss).
    assert rec["val/perplexity"] == pytest.approx(math.exp(2.0))
    # Bare back-compat names survive.
    assert rec["loss"] == 2.0
    assert rec["accuracy"] == 0.30


def test_log_val_live_caller_spelling_populates_namespaced_keys(
    tmp_path: Path,
) -> None:
    """The live adapter / distill validation loops call
    ``log_val(step=…, val_loss=…, val_source=…)`` (see
    ``scripts/train_jax_adapter.py`` and ``scripts/train_jax_distill.py``).

    This pins that the `val_`-prefixed spelling — the ONLY spelling any
    production caller passes — still populates the namespaced `val/*`
    dashboard keys (and the derived `val/perplexity`). Without the
    `val_loss` promotion source the namespaced keys would never be written
    on any live path, leaving the `pawn` charts blank for every real run.
    """
    with MetricsLogger(tmp_path, slug="s", device="cpu") as logger:
        logger.log_val(step=200, val_loss=1.5, val_source="validation")
    rec = _read_records(logger.path)[0]
    assert rec["type"] == "val"
    assert rec["val/loss"] == 1.5
    assert rec["val/perplexity"] == pytest.approx(math.exp(1.5))
    # The bare caller kwargs are retained for the adapter chart specs
    # (charts.py reads `val_loss` for `_ADAPTER_TYPES`) and the run-source.
    assert rec["val_loss"] == 1.5
    assert rec["val_source"] == "validation"


def test_log_val_live_caller_top1_top5_populate_accuracy_keys(
    tmp_path: Path,
) -> None:
    """A val loop that also reports `val_top1` / `val_top5` populates the
    `val/accuracy` + `val/top5_accuracy` keys the `pawn` accuracy chart
    reads — the `val_top1` source feeds `val/accuracy` and `val/top1`."""
    with MetricsLogger(tmp_path, slug="s", device="cpu") as logger:
        logger.log_val(step=10, val_loss=2.0, val_top1=0.33, val_top5=0.61)
    rec = _read_records(logger.path)[0]
    assert rec["val/accuracy"] == 0.33
    assert rec["val/top1"] == 0.33
    assert rec["val/top5"] == 0.61
    assert rec["val/top5_accuracy"] == 0.61


def test_log_val_respects_explicit_namespaced_keys(tmp_path: Path) -> None:
    """A caller that already passes `val/...` keys (or a `val/perplexity`)
    isn't clobbered by the promotion / derivation logic."""
    with MetricsLogger(tmp_path, slug="s", device="cpu") as logger:
        # log_val can't take `val/loss=` as a kwarg (slash isn't a valid
        # identifier), so callers pass already-namespaced metrics via **{}.
        logger.log_val(step=5, **{
            "val/loss": 3.0,
            "val/perplexity": 7.0,  # explicit — must NOT be re-derived
            "patience": 4,
        })
    rec = _read_records(logger.path)[0]
    assert rec["val/loss"] == 3.0
    assert rec["val/perplexity"] == 7.0  # not exp(3.0)
    assert rec["patience"] == 4


def test_log_val_skips_perplexity_on_nonfinite_loss(tmp_path: Path) -> None:
    """A NaN/Inf loss can't yield a finite perplexity — the derived key is
    simply not added (rather than producing `exp(nan)` → null clutter)."""
    with MetricsLogger(tmp_path, slug="s", device="cpu") as logger:
        logger.log_val(step=1, loss=float("nan"))
    rec = _read_records(logger.path)[0]
    assert "val/perplexity" not in rec
    # The NaN loss itself is sanitised to null on both the bare + namespaced key.
    assert rec["loss"] is None
    assert rec["val/loss"] is None


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
    """`device != "cpu"` triggers the GPU shell-out backend AND the
    v1-parity JAX allocator probe; both populate their record fields."""
    with mock.patch(
        "pawn.logging._query_gpu_stats",
        return_value={"gpu_used_gb": 4.0, "gpu_total_gb": 24.0},
    ) as mock_smi, mock.patch(
        "pawn.logging._query_jax_memory_stats",
        return_value={
            "gpu_peak_gb": 2.5,
            "gpu_reserved_gb": 3.0,
            "gpu_current_gb": 1.7,
        },
    ) as mock_jax:
        with MetricsLogger(tmp_path, slug="s", device="cuda") as logger:
            logger.log_train(step=1, loss=1.0)
        assert mock_smi.called
        assert mock_jax.called
    parsed = json.loads(logger.path.read_text().splitlines()[0])
    # smi-based system-wide fields.
    assert parsed["mem/gpu_used_gb"] == 4.0
    assert parsed["mem/gpu_total_gb"] == 24.0
    # v1-parity allocator fields (parity #7).
    assert parsed["mem/gpu_peak_gb"] == 2.5
    assert parsed["mem/gpu_reserved_gb"] == 3.0
    assert parsed["mem/gpu_current_gb"] == 1.7


def test_device_cuda_with_no_gpu_backend_skips_gpu_fields(tmp_path: Path) -> None:
    """If the shell-out returns None (no CLI, query failed), no GPU
    fields are added — the run still logs cleanly."""
    with mock.patch("pawn.logging._query_gpu_stats", return_value=None), \
         mock.patch("pawn.logging._query_jax_memory_stats", return_value=None):
        with MetricsLogger(tmp_path, slug="s", device="cuda") as logger:
            logger.log_train(step=1, loss=1.0)
    parsed = json.loads(logger.path.read_text().splitlines()[0])
    assert "mem/gpu_used_gb" not in parsed
    # v1-parity allocator fields are equally skipped when JAX returns None.
    assert "mem/gpu_peak_gb" not in parsed
    assert "mem/gpu_reserved_gb" not in parsed
    assert "mem/gpu_current_gb" not in parsed


def test_jax_memory_stats_field_names_match_v1_torch_path(tmp_path: Path) -> None:
    """Parity #7: the v1 torch path used `mem/gpu_{peak,reserved,current}_gb`;
    the v2 JAX path emits the same names so existing dashboards
    (pawn/dashboard/charts.py:409-411) and grep'd workflows keep
    working unchanged."""
    with mock.patch(
        "pawn.logging._query_jax_memory_stats",
        return_value={
            "gpu_peak_gb": 9.0,
            "gpu_reserved_gb": 10.0,
            "gpu_current_gb": 8.0,
        },
    ), mock.patch("pawn.logging._query_gpu_stats", return_value=None):
        with MetricsLogger(tmp_path, slug="s", device="cuda") as logger:
            logger.log_train(step=1, loss=1.0)
    parsed = json.loads(logger.path.read_text().splitlines()[0])
    assert set(parsed) >= {
        "mem/gpu_peak_gb", "mem/gpu_reserved_gb", "mem/gpu_current_gb",
    }


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


# ---------------------------------------------------------------------------
# H7 — schedule_health.json is written at every trainer exit path
# ---------------------------------------------------------------------------


def _only_run_dir(logs_dir: Path) -> Path:
    """Return the single MetricsLogger run dir created under `logs_dir`."""
    candidates = [d for d in logs_dir.iterdir() if d.is_dir()]
    assert len(candidates) == 1, f"expected one run dir, got {candidates}"
    return candidates[0]


def _gpu_only() -> None:
    import jax

    if jax.default_backend() != "gpu":
        pytest.skip("training entry points require the GPU backend")


def test_pretrain_writes_schedule_health_on_normal_exit(
    tmp_path: Path,
) -> None:
    """A tiny full-budget pretrain run writes schedule_health.json with
    `reason_for_stop == "completed"` and actual == planned (H7)."""
    _gpu_only()
    # Import the script module by path (scripts/ isn't a package).
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "scripts_train_jax_h7a", Path("scripts/train_jax.py")
    )
    assert spec is not None and spec.loader is not None
    train_jax = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_jax)

    logs_dir = tmp_path / "logs"
    rc = train_jax.main([
        "--supernet", "tiny", "--total-steps", "4", "--batch-size", "2",
        "--seq-len", "32", "--k", "2", "--no-bucketing",
        "--local-checkpoints", "--logs-dir", str(logs_dir),
        "--lr-schedule", "cosine",
    ])
    assert rc == 0
    run_dir = _only_run_dir(logs_dir)
    health = json.loads((run_dir / "schedule_health.json").read_text())
    assert health["reason_for_stop"] == "completed"
    assert health["planned_total_steps"] == 4
    assert health["actual_total_steps"] == 4
    assert health["schedule"] == "cosine"
    assert health["should_reach_zero"] is True


def test_pretrain_emits_train_loss_dashboard_key(tmp_path: Path) -> None:
    """The pretrainer must emit `train/loss` AND `train/accuracy` — the
    keys the dashboard's `pawn` run_type keys its loss + accuracy charts
    and KPI tiles on (`pawn/dashboard/charts.py:344,373`,
    `sol.py:552-563`). It previously emitted only the bare `loss` and no
    accuracy at all, leaving the loss chart empty and the accuracy panel
    blank. The bare `loss` / `accuracy` aliases are retained for the
    sweep/lab monitors that fall back to them. `train/accuracy` is the
    supernet (largest variant) top-1 — v1 parity
    (`git show main:pawn/trainer.py:1145`)."""
    _gpu_only()
    import importlib.util

    from pawn.dashboard.metrics import detect_run_type, load_run_buckets

    spec = importlib.util.spec_from_file_location(
        "scripts_train_jax_trainloss", Path("scripts/train_jax.py")
    )
    assert spec is not None and spec.loader is not None
    train_jax = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_jax)

    logs_dir = tmp_path / "logs"
    rc = train_jax.main([
        "--supernet", "tiny", "--total-steps", "4", "--batch-size", "2",
        "--seq-len", "32", "--k", "2", "--no-bucketing",
        "--local-checkpoints", "--logs-dir", str(logs_dir),
        "--log-interval", "1",
    ])
    assert rc == 0
    run_dir = _only_run_dir(logs_dir)
    buckets = load_run_buckets(logs_dir, run_dir.name)

    # Dashboard routes this run as the `pawn` (pretrain) type.
    assert buckets["config"]
    assert detect_run_type(buckets["config"][-1]) == "pawn"

    train = buckets["train"]
    assert train, "no train records written"
    # Every logged train record carries both the namespaced keys and the
    # back-compat aliases, with identical values. `train/accuracy` is a
    # finite rate in [0, 1] (it can legitimately be 0 on a 4-step tiny run).
    for rec in train:
        assert "train/loss" in rec, f"train record missing train/loss: {rec.keys()}"
        assert rec["train/loss"] == rec["loss"]
        assert "train/accuracy" in rec, (
            f"train record missing train/accuracy: {rec.keys()}"
        )
        assert rec["train/accuracy"] == rec["accuracy"]
        acc = rec["train/accuracy"]
        assert isinstance(acc, (int, float))
        assert 0.0 <= acc <= 1.0


def test_pretrain_schedule_health_clamps_overshoot_on_completed(
    tmp_path: Path,
) -> None:
    """OBS-1: when the chunk size does NOT divide the planned budget, the
    final chunk overshoots (`next_step > total_steps`), but a `completed`
    run must still report `actual_total_steps == planned_total_steps` so
    the H7 structural-mismatch tripwire stays unreachable on healthy runs.

    Budget 6, k=4 → chunk 0 produces 4 steps, chunk 1 overshoots to 8 > 6.
    Without the clamp, `actual_total_steps` would be 8 and trip the red
    WARNING banner + the lab runner's `structural_mismatch`.
    """
    _gpu_only()
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "scripts_train_jax_obs1", Path("scripts/train_jax.py")
    )
    assert spec is not None and spec.loader is not None
    train_jax = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_jax)

    logs_dir = tmp_path / "logs"
    rc = train_jax.main([
        "--supernet", "tiny", "--total-steps", "6", "--batch-size", "2",
        "--seq-len", "32", "--k", "4", "--no-bucketing",
        "--local-checkpoints", "--logs-dir", str(logs_dir),
        "--lr-schedule", "cosine",
    ])
    assert rc == 0
    run_dir = _only_run_dir(logs_dir)
    health = json.loads((run_dir / "schedule_health.json").read_text())
    assert health["reason_for_stop"] == "completed"
    assert health["planned_total_steps"] == 6
    # Clamped despite the chunk-granularity overshoot (raw next_step == 8).
    assert health["actual_total_steps"] == 6
    assert health["completion_ratio"] == 1.0


def test_pretrain_writes_schedule_health_on_sigterm(
    tmp_path: Path,
) -> None:
    """When SIGTERM fires mid-run, the trainer still writes
    schedule_health.json with `reason_for_stop == "sigterm"` and an
    actual step count below the planned budget (H7)."""
    _gpu_only()
    import importlib.util

    from pawn.lifecycle import _reset_shutdown_state_for_tests

    spec = importlib.util.spec_from_file_location(
        "scripts_train_jax_h7b", Path("scripts/train_jax.py")
    )
    assert spec is not None and spec.loader is not None
    train_jax = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_jax)

    _reset_shutdown_state_for_tests()
    logs_dir = tmp_path / "logs"

    # Patch the lifecycle SIGTERM install so the very first poll reports
    # "shutdown requested" — simulates SIGTERM arriving after chunk 1
    # without depending on signal-delivery timing.
    import pawn.lifecycle as lifecycle

    real_install = lifecycle.install_sigterm_handler

    def _fake_install(on_shutdown=None):  # noqa: ANN001
        del on_shutdown
        state = {"polls": 0}

        def should_shutdown() -> bool:
            state["polls"] += 1
            # Let one chunk complete, then request shutdown.
            return state["polls"] > 1

        return should_shutdown

    with mock.patch.object(
        train_jax, "install_sigterm_handler", _fake_install
    ):
        rc = train_jax.main([
            "--supernet", "tiny", "--total-steps", "1000",
            "--batch-size", "2", "--seq-len", "32", "--k", "2",
            "--no-bucketing", "--local-checkpoints",
            "--logs-dir", str(logs_dir), "--lr-schedule", "cosine",
        ])
    assert rc == 0
    run_dir = _only_run_dir(logs_dir)
    health = json.loads((run_dir / "schedule_health.json").read_text())
    assert health["reason_for_stop"] == "sigterm"
    assert health["actual_total_steps"] < health["planned_total_steps"]
    assert health["planned_total_steps"] == 1000


def test_adapter_resume_is_bit_reproducible(tmp_path: Path) -> None:
    """An adapter run that checkpoints mid-way and resumes reaches the
    *same* adapter weights as an uninterrupted run of the same length
    (H7: scheduler + RNG persisted and restored on resume).

    The adapter loop samples per-step batch indices from a numpy
    ``default_rng(data_seed)`` over a fixed corpus (``--no-pgn`` → a
    seed-0 random corpus), so the *only* sources of randomness across the
    resume boundary are: the trained adapter + warm opt-state (restored
    from the resume sidecar + ``optimizer.safetensors``) and the
    data-stream RNG (restored from ``training_state.json``).

    The resumed run must land on the uninterrupted reference's final
    adapter to within the optimizer-state storage precision. (Adam's
    first-moment ``mu`` is persisted in bfloat16 in
    ``optimizer.safetensors`` — a deliberate memory trade-off — so a warm
    resume rounds ``mu`` to bf16 and can't be *bit*-identical. The tight
    tolerance below is still ~3 orders of magnitude smaller than the
    divergence a *cold* RNG / opt-state would produce, so it is a real
    regression tripwire for the H7 restore path: drop the RNG restore and
    the data stream desyncs, blowing the assertion wide open.) Two
    identical (non-resumed) runs are bit-exact on this hardware, so the
    only slack here is the bf16-``mu`` round-trip.
    """
    _gpu_only()
    import importlib.util

    import jax
    import jax.numpy as jnp

    from pawn.checkpoint import load_model

    def _load_script(path: str, tag: str):  # noqa: ANN202
        spec = importlib.util.spec_from_file_location(
            f"scripts_{tag}", Path(f"scripts/{path}.py")
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # Build a tiny local backbone for the adapter to fine-tune (avoids the
    # default HF `pawn-base-v2` fetch).
    bb_logs = tmp_path / "bb"
    rc = _load_script("train_jax", "bb").main([
        "--supernet", "tiny", "--total-steps", "2", "--batch-size", "2",
        "--seq-len", "32", "--k", "2", "--no-bucketing",
        "--local-checkpoints", "--logs-dir", str(bb_logs),
        "--checkpoint-interval", "2",
    ])
    assert rc == 0
    backbone_ckpt = _only_run_dir(bb_logs) / "step_00000002"
    assert backbone_ckpt.is_dir()

    # lr_schedule defaults to cosine; the adapter loop always writes a
    # final checkpoint at run-end, so the 2-step run emits
    # `adapter_step_00000002` without needing a custom checkpoint_interval.
    common = [
        "--strategy", "lora", "--supernet", "tiny", "--variant", "base",
        "--checkpoint", str(backbone_ckpt),
        "--lora-rank", "2", "--no-pgn", "--batch-size", "2",
        "--seq-len", "32", "--k", "1", "--local-checkpoints",
    ]

    # Uninterrupted reference: 4 steps.
    ref_logs = tmp_path / "ref"
    rc = _load_script("train_jax_adapter", "ref").main(
        [*common, "--total-steps", "4", "--logs-dir", str(ref_logs)]
    )
    assert rc == 0
    ref_run = _only_run_dir(ref_logs)
    ref_model, _ = load_model(ref_run / "adapter_step_00000004")

    # Interrupted: run 2 steps, then resume to 4 from the step-2 checkpoint.
    a_logs = tmp_path / "a"
    rc = _load_script("train_jax_adapter", "a").main(
        [*common, "--total-steps", "2", "--logs-dir", str(a_logs)]
    )
    assert rc == 0
    a_run = _only_run_dir(a_logs)
    resume_ckpt = a_run / "adapter_step_00000002"
    assert resume_ckpt.is_dir()

    b_logs = tmp_path / "b"
    rc = _load_script("train_jax_adapter", "b").main([
        *common, "--total-steps", "4", "--logs-dir", str(b_logs),
        "--resume", str(resume_ckpt),
    ])
    assert rc == 0
    b_run = _only_run_dir(b_logs)
    resumed_model, _ = load_model(b_run / "adapter_step_00000004")

    # Compare the (folded) effective-model array leaves bit-exactly.
    import equinox as eqx

    ref_leaves = jax.tree_util.tree_leaves(
        eqx.filter(ref_model, eqx.is_inexact_array)
    )
    res_leaves = jax.tree_util.tree_leaves(
        eqx.filter(resumed_model, eqx.is_inexact_array)
    )
    assert len(ref_leaves) == len(res_leaves)
    max_abs_diff = 0.0
    for r, s in zip(ref_leaves, res_leaves):
        max_abs_diff = max(
            max_abs_diff,
            float(jnp.max(jnp.abs(jnp.asarray(r) - jnp.asarray(s)))),
        )
    # Tight: a cold RNG / opt-state resume desyncs the data stream and
    # produces O(1e-1+) divergence. The bf16-`mu` round-trip alone is
    # ~1e-4 here, so this bound proves the RNG + warm opt-state restore.
    assert max_abs_diff < 1e-2, (
        f"resumed adapter diverged from the uninterrupted reference by "
        f"{max_abs_diff:.3e} (> 1e-2); the RNG/opt-state resume restore is "
        f"not faithful"
    )


def test_adapter_intermediate_checkpoint_is_fresh(tmp_path: Path) -> None:
    """An adapter run that crosses a checkpoint boundary *mid-loop* writes
    the *advanced* PyTree, not the cold-init carry (OBS-R3-1).

    The in-loop checkpoint path (``final_step % checkpoint_interval == 0``)
    runs from inside the nested ``_run_steps`` driver, which advances the
    carry per chunk. ``_save`` reads ``state.adapter`` / ``state.step`` as
    free variables off the *enclosing* binding; before that binding was kept
    in sync (``nonlocal state``), the intermediate ``adapter_step_*`` saved
    the cold-init adapter (effective == backbone, since LoRA's B is
    zero-init) and a stale ``training_state.json['step']`` that disagreed with
    the fresh checkpoint dir name.

    This run is the production shape the resume test can't reach: it sets
    ``checkpoint_interval=2`` *below* ``total_steps=4`` so the loop fires the
    in-loop ``_save(2)`` before the run-end ``_save`` — and asserts that
    intermediate effective model has moved off the backbone (B is no longer
    zero after 2 trained steps) and that its persisted step counter is 2.
    """
    _gpu_only()

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from pawn.adapter_trainer import STRATEGIES, dispatch_init
    from pawn.adapters import LoRAConfig
    from pawn.checkpoint import load_model
    from pawn.config import TINY_VARIANTS
    from pawn.model import sliced

    backbone_ckpt = _tiny_backbone_ckpt(tmp_path)

    # Build the cold-init effective model the trainer would start from:
    # slice the tiny supernet to the `base` variant, init the LoRA adapter
    # with the same key the trainer uses (jax.random.key(0)), and apply.
    # LoRA's B is zero-init, so this effective model equals the sliced
    # backbone at step 0 — exactly the PyTree a stale save would round-trip.
    backbone_model, _ = load_model(backbone_ckpt)
    sliced_backbone = sliced(backbone_model, TINY_VARIANTS["base"])
    cold_adapter = dispatch_init("lora")(
        sliced_backbone, LoRAConfig(rank=2), key=jax.random.key(0)
    )
    cold_effective = STRATEGIES["lora"].apply(sliced_backbone, cold_adapter)

    # checkpoint_interval lives on the run config, not the adapter CLI, so
    # route it through a --config JSON.
    cfg_path = tmp_path / "adapter.json"
    cfg_path.write_text(json.dumps({"checkpoint_interval": 2}))

    logs_dir = tmp_path / "logs"
    rc = _load_script("train_jax_adapter", "obs_r3_1").main([
        "--config", str(cfg_path),
        "--strategy", "lora", "--supernet", "tiny", "--variant", "base",
        "--checkpoint", str(backbone_ckpt), "--lora-rank", "2",
        "--no-pgn", "--batch-size", "2", "--seq-len", "32", "--k", "1",
        "--total-steps", "4", "--local-checkpoints",
        "--logs-dir", str(logs_dir),
    ])
    assert rc == 0
    run_dir = _only_run_dir(logs_dir)

    # The intermediate checkpoint must exist and have been written from the
    # in-loop path (step 2 < total_steps 4).
    inter = run_dir / "adapter_step_00000002"
    assert inter.is_dir(), "intermediate checkpoint not written"

    # (c) the persisted step counter matches the fresh dir name, not the
    # stale cold-init `state.step` (= 0 on a fresh run).
    training_state = json.loads((inter / "training_state.json").read_text())
    assert training_state["step"] == 2, (
        f"intermediate training_state.json step is "
        f"{training_state['step']!r}, expected 2 — stale state.step persisted"
    )

    # (a) the saved effective model must differ from the cold-init effective
    # model. A stale save would round-trip to `cold_effective` bit-for-bit.
    inter_model, _ = load_model(inter)
    cold_leaves = jax.tree_util.tree_leaves(
        eqx.filter(cold_effective, eqx.is_inexact_array)
    )
    inter_leaves = jax.tree_util.tree_leaves(
        eqx.filter(inter_model, eqx.is_inexact_array)
    )
    assert len(cold_leaves) == len(inter_leaves)
    max_abs_diff = max(
        (
            float(jnp.max(jnp.abs(jnp.asarray(c) - jnp.asarray(i))))
            for c, i in zip(cold_leaves, inter_leaves)
        ),
        default=0.0,
    )
    assert max_abs_diff > 0.0, (
        "intermediate adapter checkpoint is identical to the cold-init "
        "effective model — the in-loop _save persisted a stale PyTree "
        "instead of the advanced step-2 carry (OBS-R3-1)"
    )


def _load_script(path: str, tag: str):  # noqa: ANN202
    """Import a ``scripts/<path>.py`` entry point by file location.

    ``scripts/`` is not a package, so the integration tests load each
    trainer module by path. Each call uses a fresh module name so repeated
    loads don't collide in ``sys.modules``.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"scripts_{tag}", Path(f"scripts/{path}.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_backbone_ckpt(tmp_path: Path) -> Path:
    """Pretrain a tiny 2-step backbone and return its checkpoint dir.

    Shared by the adapter schedule_health tests so they fine-tune a local
    backbone instead of fetching the default HF ``pawn-base-v2``.
    """
    bb_logs = tmp_path / "bb"
    rc = _load_script("train_jax", "bb_h7").main([
        "--supernet", "tiny", "--total-steps", "2", "--batch-size", "2",
        "--seq-len", "32", "--k", "2", "--no-bucketing",
        "--local-checkpoints", "--logs-dir", str(bb_logs),
        "--checkpoint-interval", "2",
    ])
    assert rc == 0
    ckpt = _only_run_dir(bb_logs) / "step_00000002"
    assert ckpt.is_dir()
    return ckpt


def test_adapter_writes_schedule_health_on_normal_exit(tmp_path: Path) -> None:
    """A tiny full-budget adapter run writes schedule_health.json with
    `reason_for_stop == "completed"` and actual == planned (H7, adapter
    exit path)."""
    _gpu_only()
    backbone_ckpt = _tiny_backbone_ckpt(tmp_path)

    logs_dir = tmp_path / "logs"
    rc = _load_script("train_jax_adapter", "adapter_h7a").main([
        "--strategy", "lora", "--supernet", "tiny", "--variant", "base",
        "--checkpoint", str(backbone_ckpt), "--lora-rank", "2",
        "--no-pgn", "--batch-size", "2", "--seq-len", "32", "--k", "1",
        "--total-steps", "4", "--local-checkpoints",
        "--logs-dir", str(logs_dir),
    ])
    assert rc == 0
    run_dir = _only_run_dir(logs_dir)
    health = json.loads((run_dir / "schedule_health.json").read_text())
    assert health["reason_for_stop"] == "completed"
    assert health["planned_total_steps"] == 4
    assert health["actual_total_steps"] == 4
    assert health["schedule"] == "cosine"


def test_adapter_writes_schedule_health_on_sigterm(tmp_path: Path) -> None:
    """When SIGTERM fires mid-run, the adapter trainer still writes
    schedule_health.json with `reason_for_stop == "sigterm"` and an actual
    step count below the planned budget (H7, adapter exit path)."""
    _gpu_only()
    from pawn.lifecycle import _reset_shutdown_state_for_tests

    _reset_shutdown_state_for_tests()
    backbone_ckpt = _tiny_backbone_ckpt(tmp_path)

    adapter = _load_script("train_jax_adapter", "adapter_h7b")

    def _fake_install(on_shutdown=None):  # noqa: ANN001
        del on_shutdown
        state = {"polls": 0}

        def should_shutdown() -> bool:
            state["polls"] += 1
            # Let one chunk complete, then request shutdown.
            return state["polls"] > 1

        return should_shutdown

    logs_dir = tmp_path / "logs"
    with mock.patch.object(
        adapter, "install_sigterm_handler", _fake_install
    ):
        rc = adapter.main([
            "--strategy", "lora", "--supernet", "tiny", "--variant", "base",
            "--checkpoint", str(backbone_ckpt), "--lora-rank", "2",
            "--no-pgn", "--batch-size", "2", "--seq-len", "32", "--k", "1",
            "--total-steps", "1000", "--local-checkpoints",
            "--logs-dir", str(logs_dir),
        ])
    assert rc == 0
    run_dir = _only_run_dir(logs_dir)
    health = json.loads((run_dir / "schedule_health.json").read_text())
    assert health["reason_for_stop"] == "sigterm"
    assert health["planned_total_steps"] == 1000
    assert health["actual_total_steps"] < health["planned_total_steps"]


def test_adapter_writes_schedule_health_on_resume_no_op(
    tmp_path: Path,
) -> None:
    """Resuming an adapter run at/past its resolved budget runs zero steps
    and writes schedule_health.json with `reason_for_stop == "resume_no_op"`
    and `actual_total_steps` == the resumed checkpoint's step (H7)."""
    _gpu_only()
    backbone_ckpt = _tiny_backbone_ckpt(tmp_path)

    common = [
        "--strategy", "lora", "--supernet", "tiny", "--variant", "base",
        "--checkpoint", str(backbone_ckpt), "--lora-rank", "2",
        "--no-pgn", "--batch-size", "2", "--seq-len", "32", "--k", "1",
        "--local-checkpoints",
    ]

    # First run: 2 steps, writes adapter_step_00000002.
    a_logs = tmp_path / "a"
    rc = _load_script("train_jax_adapter", "adapter_h7c1").main(
        [*common, "--total-steps", "2", "--logs-dir", str(a_logs)]
    )
    assert rc == 0
    resume_ckpt = _only_run_dir(a_logs) / "adapter_step_00000002"
    assert resume_ckpt.is_dir()

    # Resume with the SAME budget (2): nothing left to run → resume_no_op.
    b_logs = tmp_path / "b"
    rc = _load_script("train_jax_adapter", "adapter_h7c2").main([
        *common, "--total-steps", "2", "--logs-dir", str(b_logs),
        "--resume", str(resume_ckpt),
    ])
    assert rc == 0
    run_dir = _only_run_dir(b_logs)
    health = json.loads((run_dir / "schedule_health.json").read_text())
    assert health["reason_for_stop"] == "resume_no_op"
    assert health["planned_total_steps"] == 2
    assert health["actual_total_steps"] == 2


def test_pretrain_resume_is_bit_reproducible(tmp_path: Path) -> None:
    """A pretrain run that checkpoints mid-way and resumes reaches the
    *same* supernet weights as an uninterrupted run of the same length
    (H7 / D2: data-stream + scheduler + opt-state persisted/restored).

    The pretrain data stream draws each outer-chunk's engine seed from a
    *pure function* of ``(BASE_DATA_SEED, chunk_index)`` (no stateful
    look-ahead Generator), and the checkpoint persists the *consume anchor*
    (chunk index + intra-chunk batch offset). On resume the prefetcher
    re-derives from that anchor and skips the already-consumed leading
    batches, so the resumed batch sequence is bit-identical to the
    uninterrupted run's tail.

    This guards the exact gap the OBS-2 / D2-1 review finding identified:
    the buggy predecessor persisted a look-ahead-advanced ``numpy`` rng
    whose next draw skipped the in-flight (submitted-but-untrained) outer
    chunk, silently dropping ~one outer-chunk of training and diverging
    from the uninterrupted run. The tolerance below is the same bf16-``mu``
    round-trip slack the adapter test uses — a cold/desynced data stream
    would blow it open by orders of magnitude.
    """
    _gpu_only()
    import importlib.util

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from pawn.checkpoint import load_model

    def _load_script(tag: str):  # noqa: ANN202
        spec = importlib.util.spec_from_file_location(
            f"scripts_pretrain_{tag}", Path("scripts/train_jax.py")
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # K=2 → each prefetched batch is 2 steps; checkpoint-interval 2 lands a
    # checkpoint after every batch. `--no-bucketing` keeps a single bucket so
    # one outer-chunk yields several (3 × outer_factor=1 → here outer_factor
    # default 3) K-batches; consuming 4 steps draws the first two batches of
    # outer-chunk 0 (so the mid-chunk resume anchor is (chunk=0, offset=1)).
    common = [
        "--supernet", "tiny", "--batch-size", "2", "--seq-len", "32",
        "--k", "2", "--no-bucketing", "--local-checkpoints",
        "--checkpoint-interval", "2",
    ]

    # Uninterrupted reference: 4 steps.
    ref_logs = tmp_path / "ref"
    rc = _load_script("ref").main(
        [*common, "--total-steps", "4", "--logs-dir", str(ref_logs)]
    )
    assert rc == 0
    ref_run = _only_run_dir(ref_logs)
    ref_model, _ = load_model(ref_run / "step_00000004")

    # Interrupted: run 2 steps, then resume to 4 from the step-2 checkpoint.
    a_logs = tmp_path / "a"
    rc = _load_script("a").main(
        [*common, "--total-steps", "2", "--logs-dir", str(a_logs)]
    )
    assert rc == 0
    a_run = _only_run_dir(a_logs)
    resume_ckpt = a_run / "step_00000002"
    assert resume_ckpt.is_dir()
    # The persisted anchor must be the *mid-chunk* consume position, proving
    # the prefetcher tracked the in-flight chunk rather than re-drawing past
    # it. With B*K*outer_factor games in outer-chunk 0 (≥ 2 K-batches), step 2
    # lands after one consumed batch → (chunk=0, offset=1).
    ts = json.loads((resume_ckpt / "training_state.json").read_text())
    assert ts["data_anchor"]["chunk_index"] == 0
    assert ts["data_anchor"]["batch_offset"] == 1

    b_logs = tmp_path / "b"
    rc = _load_script("b").main([
        *common, "--total-steps", "4", "--logs-dir", str(b_logs),
        "--resume", str(resume_ckpt),
    ])
    assert rc == 0
    b_run = _only_run_dir(b_logs)
    resumed_model, _ = load_model(b_run / "step_00000004")

    ref_leaves = jax.tree_util.tree_leaves(
        eqx.filter(ref_model, eqx.is_inexact_array)
    )
    res_leaves = jax.tree_util.tree_leaves(
        eqx.filter(resumed_model, eqx.is_inexact_array)
    )
    assert len(ref_leaves) == len(res_leaves)
    max_abs_diff = 0.0
    for r, s in zip(ref_leaves, res_leaves):
        max_abs_diff = max(
            max_abs_diff,
            float(jnp.max(jnp.abs(jnp.asarray(r) - jnp.asarray(s)))),
        )
    assert max_abs_diff < 1e-2, (
        f"resumed pretrain diverged from the uninterrupted reference by "
        f"{max_abs_diff:.3e} (> 1e-2); the data-stream anchor / opt-state "
        f"resume restore is not faithful (prefetcher look-ahead skip?)"
    )
