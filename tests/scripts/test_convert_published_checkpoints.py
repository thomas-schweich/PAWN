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
    ``tests/scripts/test_script_smoke.py``).

    If ``exec_module`` raises (syntax error, import-time exception),
    remove the half-initialised entry from ``sys.modules`` so a
    subsequent retry doesn't take the fast-path and return a broken
    module."""
    name = "convert_published_checkpoints_test_module"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, str(SCRIPTS / "convert_published_checkpoints.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def test_convert_one_returns_failure_result_on_snapshot_error(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """A network failure in ``snapshot_download`` returns a FAIL row, not
    a raise. ``passed`` is False, ``error`` is populated, downstream
    fields are NaN/zero placeholders. The FAILED line + traceback +
    elapsed timing must all land on stderr (CI split-tee invariant)."""
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
    # Stderr-routing invariant: all three failure diagnostics on stderr.
    # The FAILED line and the elapsed line are explicit ``file=sys.stderr``;
    # the traceback goes through ``traceback.print_exc()`` which defaults
    # to stderr. Pin all three independently — a future regression that
    # routes any one of them to stdout would break a CI split-tee.
    captured = capfd.readouterr()
    assert "FAILED: ConnectionError: simulated network timeout" in captured.err
    assert "elapsed:" in captured.err
    assert "Traceback" in captured.err
    assert "ConnectionError" in captured.err  # appears in the traceback body


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


def test_main_exits_one_on_parity_breach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Parity-breach path: conversion succeeded (no exception, error=None)
    but ``max_diff >= tol``. The summary should still format the row
    cleanly, and the overall exit must be 1. This is the most likely
    production failure mode (all I/O fine, numerical parity degrades);
    pinning it independently of the snapshot-error path."""
    script = _load_script()
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    def parity_fail(variant: str, out_root: Path, **kw: object) -> object:
        return script._VariantResult(
            variant=variant,
            d_model=512,
            n_layers=8,
            n_heads=8,
            head_dim=64,
            max_diff=5e-2,       # well above any reasonable tol
            mean_diff=1e-3,
            passed=False,        # parity breach
            elapsed_s=0.0,
            error=None,          # conversion itself succeeded
        )

    with patch.object(script, "_convert_one", side_effect=parity_fail):
        assert script.main(["--variants", "pawn-base"]) == 1
    # Summary must use the parity-FAIL branch (not the ERROR branch);
    # max_diff must format. A regression that misroutes a parity-failed
    # row into the ``error is not None`` branch would print
    # ``ERROR: None`` and skip the numeric formatting.
    captured = capsys.readouterr()
    assert "[FAIL]" in captured.out
    assert "max |Δ|" in captured.out
    assert "5.000e-02" in captured.out
    assert "ERROR:" not in captured.out


def test_convert_one_succeeds_does_not_label_stale(
    tmp_path: Path,
) -> None:
    """If a prior run wrote ``dst`` and this run's failure happens
    *after* ``convert_legacy_checkpoint`` overwrites it, the bytes on
    disk are this run's fresh output — not stale — and the error must
    NOT carry the ``[stale dst...]`` marker. Pins the
    ``converted_this_run`` guard."""
    script = _load_script()
    out_root = tmp_path
    stale_dst = out_root / "pawn-small"
    stale_dst.mkdir()

    # snapshot_download returns a real-looking path; convert_legacy_checkpoint
    # is a no-op (claims success); _build_torch_reference raises *after*
    # the no-op overwrite "happened". The test pins that the error in
    # that scenario does NOT carry the stale-dst marker.
    with patch.object(
        script.huggingface_hub, "snapshot_download", return_value=str(tmp_path)
    ), patch.object(
        script, "convert_legacy_checkpoint", return_value=None
    ), patch.object(
        script,
        "_build_torch_reference",
        side_effect=RuntimeError("post-conversion failure"),
    ):
        result = script._convert_one(
            "pawn-small", out_root, batch_size=1, seq_len=4, tol=1e-3
        )
    assert result["error"] is not None
    assert "RuntimeError" in result["error"]
    assert "stale dst" not in result["error"], (
        f"converted_this_run guard regressed: error={result['error']!r}"
    )
