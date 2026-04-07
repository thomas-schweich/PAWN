"""Unit tests for pawn.gpu: configure_gpu, is_rocm, apply_gpu_config.

Uses pytest-mock to stub torch.cuda to keep these tests device-independent.
"""

from __future__ import annotations

import os
import types

import pytest
import torch

import pawn.gpu as gpu_module
from pawn.gpu import (
    apply_gpu_config,
    configure_gpu,
    is_rocm,
)


# ---------------------------------------------------------------------------
# is_rocm
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsRocm:
    def test_returns_false_when_no_cuda(self, mocker):
        mocker.patch("torch.cuda.is_available", return_value=False)
        assert is_rocm() is False

    @pytest.mark.parametrize("name", [
        "AMD Radeon RX 7900 XTX",  # matches "radeon" + "rx "
        "AMD Radeon Pro W7900",    # matches "radeon"
        "Radeon Pro VII",          # matches "radeon"
        "AMD MI300X",              # matches "mi3"
        "AMD Instinct MI300X",     # matches "mi3"
    ])
    def test_returns_true_for_amd_names(self, mocker, name):
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.get_device_name", return_value=name)
        assert is_rocm() is True

    @pytest.mark.xfail(
        strict=True,
        reason="BUG-200: is_rocm heuristic misses 'AMD Instinct MI250X' (no 'mi ' substring, no 'mi3')",
    )
    def test_is_rocm_matches_mi250x(self, mocker):
        """AMD Instinct MI250X is a production ROCm card but is missed.

        `get_device_name()` returns 'AMD Instinct MI250X'. When lowercased,
        neither 'mi ' nor 'mi3' is present, so the heuristic returns False.
        """
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch(
            "torch.cuda.get_device_name",
            return_value="AMD Instinct MI250X",
        )
        assert is_rocm() is True

    @pytest.mark.parametrize("name", [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA H100 PCIe",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA L40S",
        "Tesla V100",
    ])
    def test_returns_false_for_nvidia_names(self, mocker, name):
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.get_device_name", return_value=name)
        assert is_rocm() is False


# ---------------------------------------------------------------------------
# configure_gpu — return shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigureGpuReturnShape:
    def test_returns_dict_with_expected_keys(self, mocker):
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.get_device_name", return_value="NVIDIA H100 PCIe")
        cfg = configure_gpu()
        assert isinstance(cfg, dict)
        assert set(cfg.keys()) == {"use_compile", "use_amp", "sdpa_backend"}

    def test_return_types(self, mocker):
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.get_device_name", return_value="NVIDIA H100 PCIe")
        cfg = configure_gpu()
        assert isinstance(cfg["use_compile"], bool)
        assert isinstance(cfg["use_amp"], bool)


# ---------------------------------------------------------------------------
# configure_gpu — NVIDIA path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigureGpuNvidia:
    @pytest.fixture(autouse=True)
    def _mock_nvidia(self, mocker):
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.get_device_name", return_value="NVIDIA H100 PCIe")

    def test_compile_and_amp_enabled(self):
        cfg = configure_gpu()
        assert cfg["use_compile"] is True
        assert cfg["use_amp"] is True

    def test_sdpa_backend_is_none_by_default(self):
        cfg = configure_gpu()
        assert cfg["sdpa_backend"] is None

    def test_no_compile_flag(self):
        cfg = configure_gpu(no_compile=True)
        assert cfg["use_compile"] is False

    def test_no_amp_flag(self):
        cfg = configure_gpu(no_amp=True)
        assert cfg["use_amp"] is False

    def test_sdpa_math_flag(self):
        from torch.nn.attention import SDPBackend
        cfg = configure_gpu(sdpa_math=True)
        assert cfg["sdpa_backend"] == SDPBackend.MATH


# ---------------------------------------------------------------------------
# configure_gpu — AMD / ROCm path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigureGpuAmd:
    @pytest.fixture(autouse=True)
    def _mock_amd(self, mocker):
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.get_device_name", return_value="AMD Radeon RX 7900 XTX")

    def test_sdpa_math_selected_on_amd_with_compile(self):
        from torch.nn.attention import SDPBackend
        cfg = configure_gpu()
        assert cfg["sdpa_backend"] == SDPBackend.MATH
        assert cfg["use_compile"] is True

    def test_sdpa_backend_none_when_no_compile_on_amd(self):
        cfg = configure_gpu(no_compile=True)
        assert cfg["sdpa_backend"] is None

    def test_amp_still_enabled_on_amd(self):
        cfg = configure_gpu()
        assert cfg["use_amp"] is True


# ---------------------------------------------------------------------------
# configure_gpu — CPU path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigureGpuCpu:
    def test_raises_without_cuda_and_without_override(self, mocker, monkeypatch):
        mocker.patch("torch.cuda.is_available", return_value=False)
        monkeypatch.delenv("PAWN_ALLOW_CPU", raising=False)
        with pytest.raises(RuntimeError) as exc_info:
            configure_gpu()
        assert "No GPU available" in str(exc_info.value)
        assert "PAWN_ALLOW_CPU" in str(exc_info.value)

    def test_allows_cpu_with_override(self, mocker, monkeypatch):
        mocker.patch("torch.cuda.is_available", return_value=False)
        monkeypatch.setenv("PAWN_ALLOW_CPU", "1")
        cfg = configure_gpu()
        # CPU mode: compile, amp disabled
        assert cfg["use_compile"] is False
        assert cfg["use_amp"] is False

    def test_cpu_mode_wrong_value_not_allowed(self, mocker, monkeypatch):
        mocker.patch("torch.cuda.is_available", return_value=False)
        # "0" is not "1" — should still raise
        monkeypatch.setenv("PAWN_ALLOW_CPU", "0")
        with pytest.raises(RuntimeError):
            configure_gpu()

    def test_cpu_mode_sdpa_backend_is_none(self, mocker, monkeypatch):
        mocker.patch("torch.cuda.is_available", return_value=False)
        monkeypatch.setenv("PAWN_ALLOW_CPU", "1")
        cfg = configure_gpu()
        assert cfg["sdpa_backend"] is None

    def test_cpu_mode_ignores_no_compile_noop(self, mocker, monkeypatch):
        """no_compile on CPU is already implied."""
        mocker.patch("torch.cuda.is_available", return_value=False)
        monkeypatch.setenv("PAWN_ALLOW_CPU", "1")
        cfg = configure_gpu(no_compile=True)
        assert cfg["use_compile"] is False


# ---------------------------------------------------------------------------
# apply_gpu_config
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyGpuConfig:
    def test_sets_sdpa_backend_attribute(self):
        from torch.nn.attention import SDPBackend

        fake_module = types.SimpleNamespace(SDPA_BACKEND=None)
        def fwd(x): return x
        config = {
            "use_compile": False,
            "use_amp": True,
            "sdpa_backend": SDPBackend.MATH,
        }
        result = apply_gpu_config(config, fake_module, fwd)
        assert fake_module.SDPA_BACKEND == SDPBackend.MATH
        assert result is fwd  # unchanged when no compile

    def test_does_not_touch_backend_when_none(self):
        fake_module = types.SimpleNamespace(SDPA_BACKEND="untouched")
        def fwd(x): return x
        config = {
            "use_compile": False,
            "use_amp": False,
            "sdpa_backend": None,
        }
        apply_gpu_config(config, fake_module, fwd)
        assert fake_module.SDPA_BACKEND == "untouched"

    def test_compiles_when_use_compile_true(self, mocker):
        """torch.compile is called when use_compile=True."""
        fake_module = types.SimpleNamespace(SDPA_BACKEND=None)
        def fwd(x): return x
        wrapped = lambda x: x
        compile_mock = mocker.patch("torch.compile", return_value=wrapped)

        config = {
            "use_compile": True,
            "use_amp": True,
            "sdpa_backend": None,
        }
        result = apply_gpu_config(config, fake_module, fwd)
        compile_mock.assert_called_once_with(fwd)
        assert result is wrapped

    def test_does_not_compile_when_false(self, mocker):
        fake_module = types.SimpleNamespace(SDPA_BACKEND=None)
        def fwd(x): return x
        compile_mock = mocker.patch("torch.compile")

        config = {
            "use_compile": False,
            "use_amp": True,
            "sdpa_backend": None,
        }
        apply_gpu_config(config, fake_module, fwd)
        compile_mock.assert_not_called()

    def test_sets_backend_before_compile(self, mocker):
        """Ordering matters: SDPA_BACKEND must be set before torch.compile.

        We verify by recording the SDPA_BACKEND value at the time torch.compile
        is invoked.
        """
        from torch.nn.attention import SDPBackend

        fake_module = types.SimpleNamespace(SDPA_BACKEND=None)
        def fwd(x): return x

        recorded = {}
        def _fake_compile(f):
            recorded["backend"] = fake_module.SDPA_BACKEND
            return f
        mocker.patch("torch.compile", side_effect=_fake_compile)

        config = {
            "use_compile": True,
            "use_amp": True,
            "sdpa_backend": SDPBackend.MATH,
        }
        apply_gpu_config(config, fake_module, fwd)
        assert recorded["backend"] == SDPBackend.MATH
