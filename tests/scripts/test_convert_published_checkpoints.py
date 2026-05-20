"""Tests for ``scripts/convert_published_checkpoints.py``.

The script is a Phase-1 verification one-shot, but its per-variant
try/except resilience contract (one variant's failure must not skip the
others) is load-bearing and easy to regress on. These tests cover the
error path directly without touching the network — the actual
end-to-end conversion is exercised by ``scripts/convert_published_checkpoints.py``
itself against the HF cache.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("torch")
pytest.importorskip("chess_engine")

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script() -> ModuleType:
    """Import ``convert_published_checkpoints.py`` as a top-level module
    (``scripts/`` is not a package; this matches the convention used by
    ``tests/scripts/test_script_smoke.py``)."""
    name = "convert_published_checkpoints_test_module"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, str(SCRIPTS / "convert_published_checkpoints.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_convert_one_returns_failure_result_on_snapshot_error(
    tmp_path: Path,
) -> None:
    """A network failure in ``snapshot_download`` returns a FAIL row, not
    a raise. ``passed`` is False, ``error`` is populated, downstream
    fields are NaN/zero placeholders."""
    script = _load_script()
    out_root = tmp_path
    with patch.object(
        script.huggingface_hub,
        "snapshot_download",
        side_effect=ConnectionError("simulated network timeout"),
    ):
        result = script._convert_one(
            "pawn-small", out_root, batch_size=1, seq_len=4, tol=1e-3
        )
    assert result["variant"] == "pawn-small"
    assert result["passed"] is False
    assert result["error"] is not None
    assert "ConnectionError" in result["error"]
    assert "simulated network timeout" in result["error"]
    # Numeric placeholders; the summary loop guards on ``error is not None``
    # before formatting them.
    assert result["d_model"] == 0
    assert result["max_diff"] != result["max_diff"]  # NaN


def test_convert_one_flags_stale_dst_in_error(tmp_path: Path) -> None:
    """If a previous run wrote ``out_root/<variant>``, and this run fails
    before overwriting it, the error string flags the stale-on-disk
    state so an operator doesn't load last-week's conversion."""
    script = _load_script()
    out_root = tmp_path
    stale_dst = out_root / "pawn-small"
    stale_dst.mkdir()
    (stale_dst / "model.safetensors").write_bytes(b"stale")

    with patch.object(
        script.huggingface_hub,
        "snapshot_download",
        side_effect=ConnectionError("simulated"),
    ):
        result = script._convert_one(
            "pawn-small", out_root, batch_size=1, seq_len=4, tol=1e-3
        )
    assert result["error"] is not None
    assert "stale dst" in result["error"]


def _make_pass_row(script: ModuleType, variant: str) -> object:
    return script._VariantResult(
        variant=variant,
        d_model=512,
        n_layers=8,
        n_heads=8,
        head_dim=64,
        max_diff=1e-5,
        mean_diff=1e-6,
        passed=True,
        elapsed_s=0.0,
        error=None,
    )


def _make_fail_row(script: ModuleType, variant: str, msg: str) -> object:
    return script._VariantResult(
        variant=variant,
        d_model=0,
        n_layers=0,
        n_heads=0,
        head_dim=0,
        max_diff=float("nan"),
        mean_diff=float("nan"),
        passed=False,
        elapsed_s=0.0,
        error=msg,
    )


def test_main_continues_through_failed_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``main`` must process every variant even when one raises. This is
    the load-bearing resilience contract: a network blip on pawn-small
    should not skip pawn-base + pawn-large."""
    script = _load_script()
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    call_log: list[str] = []

    def fake_convert(
        variant: str, out_root: Path, **kw: object
    ) -> object:
        call_log.append(variant)
        if variant == "pawn-small":
            return _make_fail_row(script, variant, "ConnectionError: simulated")
        return _make_pass_row(script, variant)

    with patch.object(script, "_convert_one", side_effect=fake_convert):
        exit_code = script.main(
            ["--variants", "pawn-small", "pawn-base", "pawn-large"]
        )

    # All three variants attempted (resilience contract).
    assert call_log == ["pawn-small", "pawn-base", "pawn-large"]
    # Mixed success -> exit 1.
    assert exit_code == 1
    # Summary line for the failed variant labels it ERROR, not parity FAIL.
    captured = capsys.readouterr()
    assert "ERROR: ConnectionError: simulated" in captured.out


def test_main_exits_zero_when_all_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All-pass run exits 0; pins the success-path exit code."""
    script = _load_script()
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    def all_pass(variant: str, out_root: Path, **kw: object) -> object:
        return _make_pass_row(script, variant)

    with patch.object(script, "_convert_one", side_effect=all_pass):
        assert script.main(["--variants", *script.VARIANTS]) == 0
