"""Tests for pawn.checkpoint: atomic save/load with SHA-256 integrity.

Covers the new safetensors directory format AND legacy .pt backward compat.
Extends the original tests/test_checkpoint.py with coverage of:
- atomic rollback on failure
- SHA-256 round-trip against known files
- optimizer flatten/unflatten on synthetic states
- RNG state round-trip (hypothesis)
- adapter + backbone loaders on bare safetensors
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st

from pawn.config import CLMConfig
from pawn.model import PAWNCLM

from pawn.checkpoint import (
    CheckpointIntegrityError,
    IncompleteCheckpointError,
    _atomic_directory_write,
    _flatten_optimizer_state,
    _json_to_rng,
    _rng_to_json,
    _sha256_file,
    _unflatten_optimizer_state,
    _verify_complete_sentinel,
    _write_complete_sentinel,
    is_legacy_checkpoint,
    load_adapter_checkpoint,
    load_backbone_weights,
    load_pretrain_checkpoint,
    save_adapter_checkpoint,
    save_pretrain_checkpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_toy_model(device: str = "cpu") -> tuple[PAWNCLM, CLMConfig]:
    cfg = CLMConfig.toy()
    model = PAWNCLM(cfg).to(device)
    return model, cfg


def _make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


def _run_step(model: PAWNCLM, cfg: CLMConfig, opt: torch.optim.Optimizer) -> None:
    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    mask = torch.ones(2, cfg.max_seq_len, dtype=torch.bool)
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    loss, _ = model.forward_train(x, mask, targets)
    loss.backward()
    opt.step()


class FakeScheduler:
    """Minimal scheduler stand-in for testing."""
    def __init__(self):
        self._step = 0

    def state_dict(self):
        return {"step": self._step}

    def load_state_dict(self, d):
        self._step = d["step"]


class FakeScaler:
    """Minimal scaler stand-in for testing."""
    def __init__(self):
        self._scale = 65536.0

    def state_dict(self):
        return {"scale": self._scale, "_growth_tracker": 0}

    def load_state_dict(self, d):
        self._scale = d["scale"]


# ---------------------------------------------------------------------------
# _sha256_file
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSha256File:
    def test_sha256_matches_known(self, tmp_path):
        """SHA-256 of 'hello\\n' is a well-known fixed string."""
        import hashlib
        p = tmp_path / "hello.txt"
        p.write_bytes(b"hello\n")
        expected = hashlib.sha256(b"hello\n").hexdigest()
        assert _sha256_file(p) == expected

    def test_sha256_length(self, tmp_path):
        p = tmp_path / "x"
        p.write_bytes(b"abc")
        assert len(_sha256_file(p)) == 64

    def test_sha256_different_contents_different_hashes(self, tmp_path):
        p1 = tmp_path / "a"
        p2 = tmp_path / "b"
        p1.write_bytes(b"hello")
        p2.write_bytes(b"world")
        assert _sha256_file(p1) != _sha256_file(p2)

    def test_sha256_empty_file(self, tmp_path):
        import hashlib
        p = tmp_path / "empty"
        p.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert _sha256_file(p) == expected

    def test_sha256_large_multichunk_file(self, tmp_path):
        """File >1MB is hashed in chunks; must match single-shot hash."""
        import hashlib
        data = b"x" * (2 * (1 << 20) + 7)  # 2MB+7
        p = tmp_path / "big"
        p.write_bytes(data)
        assert _sha256_file(p) == hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# _write_complete_sentinel + _verify_complete_sentinel
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompleteSentinel:
    def test_write_and_verify_roundtrip(self, tmp_path):
        d = tmp_path / "ckpt"
        d.mkdir()
        (d / "a.txt").write_text("alpha")
        (d / "b.bin").write_bytes(b"\x00\x01\x02")

        _write_complete_sentinel(d)

        sentinel_path = d / ".complete"
        assert sentinel_path.exists()

        data = json.loads(sentinel_path.read_text())
        assert data["format_version"] == 1
        assert "files" in data
        assert "a.txt" in data["files"]
        assert "b.bin" in data["files"]
        assert ".complete" not in data["files"]  # self-excluded

        # Verify passes after write
        _verify_complete_sentinel(d)

    def test_missing_sentinel_raises_incomplete(self, tmp_path):
        d = tmp_path / "ckpt"
        d.mkdir()
        (d / "a.txt").write_text("alpha")
        with pytest.raises(IncompleteCheckpointError) as exc_info:
            _verify_complete_sentinel(d)
        assert ".complete" in str(exc_info.value)

    def test_missing_file_raises_integrity(self, tmp_path):
        d = tmp_path / "ckpt"
        d.mkdir()
        (d / "a.txt").write_text("alpha")
        _write_complete_sentinel(d)
        # Remove the file the sentinel references
        (d / "a.txt").unlink()
        with pytest.raises(CheckpointIntegrityError) as exc_info:
            _verify_complete_sentinel(d)
        assert "a.txt" in str(exc_info.value)

    def test_modified_file_raises_integrity(self, tmp_path):
        d = tmp_path / "ckpt"
        d.mkdir()
        (d / "a.txt").write_text("alpha")
        _write_complete_sentinel(d)
        (d / "a.txt").write_text("BETA")
        with pytest.raises(CheckpointIntegrityError) as exc_info:
            _verify_complete_sentinel(d)
        assert "mismatch" in str(exc_info.value).lower()

    def test_sentinel_ignores_subdirectories(self, tmp_path):
        d = tmp_path / "ckpt"
        d.mkdir()
        (d / "a.txt").write_text("alpha")
        (d / "b.bin").write_bytes(b"\x00\x01")
        (d / "subdir").mkdir()
        (d / "subdir" / "inner.txt").write_text("inner")

        _write_complete_sentinel(d)
        data = json.loads((d / ".complete").read_text())

        # Top-level files ARE recorded in the sentinel
        assert "a.txt" in data["files"], "top-level file a.txt missing from sentinel"
        assert "b.bin" in data["files"], "top-level file b.bin missing from sentinel"

        # Subdirectory and its contents are NOT recorded
        assert "subdir" not in data["files"], "subdir should not appear in sentinel"
        assert "inner.txt" not in data["files"], "subdir file should not appear"
        assert "subdir/inner.txt" not in data["files"], "subdir path should not appear"

        # Sentinel is still verifiable (subdir doesn't interfere)
        _verify_complete_sentinel(d)


# ---------------------------------------------------------------------------
# _atomic_directory_write
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAtomicDirectoryWrite:
    def test_success_renames_tmp_to_target(self, tmp_path):
        target = tmp_path / "ckpt"
        with _atomic_directory_write(target) as tmp:
            assert tmp.name == "ckpt.tmp"
            (tmp / "a.txt").write_text("data")
        assert target.exists()
        assert not (tmp_path / "ckpt.tmp").exists()
        assert (target / "a.txt").read_text() == "data"
        assert (target / ".complete").exists()

    def test_failure_removes_tmp_and_no_target(self, tmp_path):
        target = tmp_path / "ckpt"
        with pytest.raises(RuntimeError):
            with _atomic_directory_write(target) as tmp:
                (tmp / "a.txt").write_text("partial")
                raise RuntimeError("boom")
        assert not target.exists()
        assert not (tmp_path / "ckpt.tmp").exists()

    def test_leftover_tmp_cleaned_up(self, tmp_path):
        """Stale .tmp from previous crashed save is cleaned up."""
        target = tmp_path / "ckpt"
        stale = tmp_path / "ckpt.tmp"
        stale.mkdir()
        (stale / "stale.txt").write_text("old")

        with _atomic_directory_write(target) as tmp:
            assert not (tmp / "stale.txt").exists(), "stale tmp should be wiped"
            (tmp / "new.txt").write_text("fresh")
        assert (target / "new.txt").exists()
        assert not (target / "stale.txt").exists()

    def test_overwrites_existing_target(self, tmp_path):
        target = tmp_path / "ckpt"
        target.mkdir()
        (target / "old.txt").write_text("prev")

        with _atomic_directory_write(target) as tmp:
            (tmp / "new.txt").write_text("fresh")

        assert (target / "new.txt").exists()
        assert not (target / "old.txt").exists()

    def test_complete_sentinel_written_on_success(self, tmp_path):
        target = tmp_path / "ckpt"
        with _atomic_directory_write(target) as tmp:
            (tmp / "a.txt").write_text("x")
        data = json.loads((target / ".complete").read_text())
        assert "a.txt" in data["files"]


# ---------------------------------------------------------------------------
# Pretrain checkpoint save/load (integration)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPretrainCheckpoint:
    def test_roundtrip(self, tmp_path):
        model1, cfg = _make_toy_model()
        opt1 = _make_optimizer(model1)
        sched1 = FakeScheduler()
        sched1._step = 42
        scaler1 = FakeScaler()
        _run_step(model1, cfg, opt1)

        ckpt_path = tmp_path / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model1, opt1, sched1, scaler1,
            global_step=1, model_config=cfg.__dict__,
            training_config={"lr": 1e-3},
        )

        assert (ckpt_path / "model.safetensors").exists()
        assert (ckpt_path / "optimizer.safetensors").exists()
        assert (ckpt_path / "training_state.json").exists()
        assert (ckpt_path / "config.json").exists()

        model2, _ = _make_toy_model()
        opt2 = _make_optimizer(model2)
        sched2 = FakeScheduler()
        scaler2 = FakeScaler()
        meta = load_pretrain_checkpoint(ckpt_path, model2, opt2, sched2, scaler2)

        assert meta["global_step"] == 1
        assert meta["model_config"]["d_model"] == cfg.d_model
        assert sched2._step == 42
        for k in model1.state_dict():
            torch.testing.assert_close(
                model1.state_dict()[k], model2.state_dict()[k],
            )

    def test_no_tmp_directory_after_save(self, tmp_path):
        model, cfg = _make_toy_model()
        opt = _make_optimizer(model)
        ckpt_path = tmp_path / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model, opt, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__, training_config={},
        )
        assert not (tmp_path / "step_00000001.tmp").exists()

    def test_complete_sentinel_has_expected_files(self, tmp_path):
        model, cfg = _make_toy_model()
        opt = _make_optimizer(model)
        ckpt_path = tmp_path / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model, opt, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__, training_config={},
        )
        sentinel = json.loads((ckpt_path / ".complete").read_text())
        assert "model.safetensors" in sentinel["files"]
        assert "training_state.json" in sentinel["files"]
        assert "config.json" in sentinel["files"]
        # Each hash should be a 64-char hex string
        for name, h in sentinel["files"].items():
            assert len(h) == 64, f"Bad hash for {name}: {len(h)}"

    def test_config_json_contents(self, tmp_path):
        model, cfg = _make_toy_model()
        opt = _make_optimizer(model)
        ckpt_path = tmp_path / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model, opt, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__,
            training_config={"lr": 1e-3, "batch_size": 32},
        )
        config = json.loads((ckpt_path / "config.json").read_text())
        assert config["format_version"] == 1
        assert config["checkpoint_type"] == "pretrain"
        assert config["model_config"]["d_model"] == cfg.d_model
        assert config["training_config"]["lr"] == 1e-3

    def test_load_without_optimizer(self, tmp_path):
        """load_pretrain_checkpoint should work with optimizer=None."""
        model1, cfg = _make_toy_model()
        opt1 = _make_optimizer(model1)
        _run_step(model1, cfg, opt1)
        ckpt_path = tmp_path / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model1, opt1, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__, training_config={},
        )
        model2, _ = _make_toy_model()
        meta = load_pretrain_checkpoint(ckpt_path, model2)
        assert meta["global_step"] == 1

    def test_load_incomplete_raises(self, tmp_path):
        """Loading a pretrain checkpoint without .complete raises."""
        model, cfg = _make_toy_model()
        opt = _make_optimizer(model)
        ckpt_path = tmp_path / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model, opt, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__, training_config={},
        )
        (ckpt_path / ".complete").unlink()
        model2, _ = _make_toy_model()
        with pytest.raises(IncompleteCheckpointError):
            load_pretrain_checkpoint(ckpt_path, model2)

    def test_load_corrupted_weights_raises(self, tmp_path):
        """Tampering with model.safetensors causes CheckpointIntegrityError."""
        model, cfg = _make_toy_model()
        opt = _make_optimizer(model)
        ckpt_path = tmp_path / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model, opt, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__, training_config={},
        )
        (ckpt_path / "model.safetensors").write_bytes(b"CORRUPT")
        model2, _ = _make_toy_model()
        with pytest.raises(CheckpointIntegrityError):
            load_pretrain_checkpoint(ckpt_path, model2)


# ---------------------------------------------------------------------------
# Integrity error surface (sentinel-level)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckpointErrors:
    def test_incomplete_checkpoint_raises(self, tmp_path):
        ckpt_path = tmp_path / "step_bad"
        ckpt_path.mkdir()
        (ckpt_path / "model.safetensors").touch()
        (ckpt_path / "config.json").write_text("{}")
        with pytest.raises(IncompleteCheckpointError):
            _verify_complete_sentinel(ckpt_path)

    def test_corrupted_file_raises(self, tmp_path):
        model, cfg = _make_toy_model()
        opt = _make_optimizer(model)
        ckpt_path = tmp_path / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model, opt, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__, training_config={},
        )
        (ckpt_path / "config.json").write_text("CORRUPTED DATA")
        with pytest.raises(CheckpointIntegrityError):
            _verify_complete_sentinel(ckpt_path)


# ---------------------------------------------------------------------------
# Optimizer state flatten/unflatten
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOptimizerFlattenUnflatten:
    def test_roundtrip_adamw(self):
        model, cfg = _make_toy_model()
        opt = _make_optimizer(model)
        _run_step(model, cfg, opt)
        orig = opt.state_dict()

        tensors, meta = _flatten_optimizer_state(orig)
        for k in tensors:
            assert k.startswith("state.")

        restored = _unflatten_optimizer_state(tensors, meta)
        assert len(restored["param_groups"]) == len(orig["param_groups"])

        for param_id in orig["state"]:
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    orig["state"][param_id][key],
                    restored["state"][param_id][key],
                )

    def test_flatten_creates_state_prefixed_keys(self):
        model, cfg = _make_toy_model()
        opt = _make_optimizer(model)
        _run_step(model, cfg, opt)
        tensors, _ = _flatten_optimizer_state(opt.state_dict())
        assert all(k.startswith("state.") for k in tensors)

    def test_load_via_pytorch_optimizer(self):
        """Load restored state back into a fresh AdamW and verify step continues."""
        model1, cfg = _make_toy_model()
        opt1 = _make_optimizer(model1)
        _run_step(model1, cfg, opt1)
        _run_step(model1, cfg, opt1)

        tensors, meta = _flatten_optimizer_state(opt1.state_dict())
        restored = _unflatten_optimizer_state(tensors, meta)

        # Build a fresh model + optimizer with same shape and load state
        model2, _ = _make_toy_model()
        model2.load_state_dict(model1.state_dict())
        opt2 = _make_optimizer(model2)
        opt2.load_state_dict(restored)

        # Step counts should match
        for pid in opt1.state_dict()["state"]:
            if "step" in opt1.state_dict()["state"][pid]:
                assert (
                    opt2.state_dict()["state"][pid]["step"]
                    == opt1.state_dict()["state"][pid]["step"]
                )

    def test_flatten_scalar_tensors_round_trip(self):
        """Scalar (0-d) step tensors in Adam survive flatten/unflatten."""
        model, cfg = _make_toy_model()
        opt = _make_optimizer(model)
        _run_step(model, cfg, opt)

        # Adam stores step as 0-d tensor
        state = opt.state_dict()
        tensors, meta = _flatten_optimizer_state(state)
        restored = _unflatten_optimizer_state(tensors, meta)
        for pid, pstate in state["state"].items():
            if "step" in pstate and isinstance(pstate["step"], torch.Tensor):
                rstate = restored["state"][pid]
                # Original is 0-d; restored must hold same value
                assert torch.equal(
                    pstate["step"].reshape(()),
                    rstate["step"].reshape(()),
                )


# ---------------------------------------------------------------------------
# RNG state round-trip (hypothesis)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRngRoundtrip:
    def test_torch_rng_roundtrip(self):
        torch_rng = torch.get_rng_state()
        data = _rng_to_json(torch_rng, None)
        assert "torch_rng_state" in data
        assert "cuda_rng_state" not in data
        restored_torch, restored_cuda = _json_to_rng(data)
        assert restored_cuda is None
        assert torch.equal(torch_rng, restored_torch)

    def test_cuda_rng_roundtrip(self):
        """Simulated CUDA RNG (uint8 tensor) survives serialization."""
        fake_cuda = torch.randint(0, 256, (816,), dtype=torch.uint8)
        data = _rng_to_json(None, fake_cuda)
        assert "torch_rng_state" not in data
        assert "cuda_rng_state" in data
        _, restored_cuda = _json_to_rng(data)
        assert torch.equal(fake_cuda, restored_cuda)

    def test_both_rng_roundtrip(self):
        t = torch.get_rng_state()
        c = torch.randint(0, 256, (32,), dtype=torch.uint8)
        data = _rng_to_json(t, c)
        rt, rc = _json_to_rng(data)
        assert torch.equal(t, rt)
        assert torch.equal(c, rc)

    def test_neither_rng_empty(self):
        data = _rng_to_json(None, None)
        assert data == {}
        rt, rc = _json_to_rng(data)
        assert rt is None
        assert rc is None

    @pytest.mark.unit
    @given(n=st.integers(min_value=1, max_value=4096))
    @settings(max_examples=20, deadline=None)
    def test_random_bytes_roundtrip(self, n):
        """Random-byte tensors round-trip through JSON serialization."""
        tensor = torch.randint(0, 256, (n,), dtype=torch.uint8)
        data = _rng_to_json(tensor, None)
        restored, _ = _json_to_rng(data)
        assert torch.equal(tensor, restored)

    def test_reproducibility_via_roundtrip(self, freeze_rng):
        """RNG restored from roundtrip produces same samples."""
        torch.manual_seed(42)
        state = torch.get_rng_state()
        data = _rng_to_json(state, None)

        # Generate some samples
        samples1 = torch.randn(10)

        # Restore and regenerate
        restored, _ = _json_to_rng(data)
        torch.set_rng_state(restored.cpu().byte())
        samples2 = torch.randn(10)

        torch.testing.assert_close(samples1, samples2)


# ---------------------------------------------------------------------------
# Adapter checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAdapterCheckpoint:
    def test_roundtrip(self, tmp_path):
        adapter_weights = {
            "down.weight": torch.randn(8, 64),
            "up.weight": torch.zeros(64, 8),
        }
        ckpt_path = tmp_path / "best"
        save_adapter_checkpoint(
            ckpt_path, adapter_weights,
            config={"bottleneck_dim": 8, "checkpoint_type": "bottleneck"},
            epoch=5, step=1000,
            val_metrics={"loss": 3.14, "top1_accuracy": 0.07},
            extra={"best_val_loss": 3.10, "patience_counter": 2},
        )
        assert (ckpt_path / "adapter.safetensors").exists()
        assert (ckpt_path / "config.json").exists()
        assert (ckpt_path / "training_state.json").exists()
        assert not (ckpt_path / "optimizer.safetensors").exists()

        loaded = load_adapter_checkpoint(ckpt_path)
        assert loaded["epoch"] == 5
        assert loaded["step"] == 1000
        assert loaded["val_metrics"]["loss"] == 3.14
        assert loaded["best_val_loss"] == 3.10

        for k in adapter_weights:
            torch.testing.assert_close(adapter_weights[k], loaded["adapter_state_dict"][k])

    def test_has_complete_sentinel(self, tmp_path):
        adapter_weights = {"down.weight": torch.randn(8, 64)}
        ckpt_path = tmp_path / "best"
        save_adapter_checkpoint(
            ckpt_path, adapter_weights,
            config={"checkpoint_type": "bottleneck"},
            epoch=1, step=100, val_metrics={"loss": 3.0},
        )
        assert (ckpt_path / ".complete").exists()
        _verify_complete_sentinel(ckpt_path)

    def test_with_optimizer(self, tmp_path):
        model, cfg = _make_toy_model()
        opt = _make_optimizer(model)
        _run_step(model, cfg, opt)
        adapter_weights = {"down.weight": torch.randn(8, 64)}

        ckpt_path = tmp_path / "best"
        save_adapter_checkpoint(
            ckpt_path, adapter_weights,
            config={"checkpoint_type": "bottleneck"},
            epoch=1, step=100, val_metrics={"loss": 3.0},
            optimizer=opt,
        )
        assert (ckpt_path / "optimizer.safetensors").exists()
        loaded = load_adapter_checkpoint(ckpt_path)
        assert "optimizer_state_dict" in loaded

    def test_load_corrupted_raises(self, tmp_path):
        adapter_weights = {"down.weight": torch.randn(8, 64)}
        ckpt_path = tmp_path / "best"
        save_adapter_checkpoint(
            ckpt_path, adapter_weights,
            config={"checkpoint_type": "bottleneck"},
            epoch=1, step=100, val_metrics={"loss": 3.0},
        )
        (ckpt_path / "config.json").write_text("{}")  # corrupt
        with pytest.raises(CheckpointIntegrityError):
            load_adapter_checkpoint(ckpt_path)


# ---------------------------------------------------------------------------
# load_backbone_weights
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLoadBackboneWeights:
    def test_new_format_directory(self, tmp_path):
        model1, cfg = _make_toy_model()
        ckpt_path = tmp_path / "step_00000001"
        opt = _make_optimizer(model1)
        save_pretrain_checkpoint(
            ckpt_path, model1, opt, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__, training_config={},
        )
        state_dict, model_config = load_backbone_weights(ckpt_path)
        assert model_config["d_model"] == cfg.d_model
        for k in model1.state_dict():
            torch.testing.assert_close(model1.state_dict()[k], state_dict[k])

    def test_legacy_pt_file(self, tmp_path):
        model1, cfg = _make_toy_model()
        pt_path = tmp_path / "model.pt"
        torch.save({
            "model_state_dict": model1.state_dict(),
            "model_config": cfg.__dict__,
        }, pt_path)

        state_dict, model_config = load_backbone_weights(pt_path)
        assert model_config["d_model"] == cfg.d_model
        for k in model1.state_dict():
            torch.testing.assert_close(model1.state_dict()[k], state_dict[k])

    def test_bare_safetensors_file(self, tmp_path):
        """load_backbone_weights works on a stripped model.safetensors."""
        from safetensors.torch import save_file
        model1, cfg = _make_toy_model()
        sf_path = tmp_path / "model.safetensors"
        sd = {k: v.cpu().contiguous() for k, v in model1.state_dict().items()}
        save_file(sd, sf_path)
        # Also drop a config.json alongside
        (tmp_path / "config.json").write_text(json.dumps(
            {"model_config": cfg.__dict__}
        ))
        state_dict, model_config = load_backbone_weights(sf_path)
        assert model_config["d_model"] == cfg.d_model
        for k in model1.state_dict():
            torch.testing.assert_close(model1.state_dict()[k], state_dict[k])

    def test_bare_safetensors_without_config(self, tmp_path):
        from safetensors.torch import save_file
        model1, _ = _make_toy_model()
        sf_path = tmp_path / "model.safetensors"
        save_file({k: v.cpu().contiguous() for k, v in model1.state_dict().items()}, sf_path)
        state_dict, model_config = load_backbone_weights(sf_path)
        assert model_config is None

    def test_directory_missing_safetensors(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            load_backbone_weights(d)

    def test_unrecognized_format(self, tmp_path):
        p = tmp_path / "weird.tar"
        p.touch()
        with pytest.raises(ValueError):
            load_backbone_weights(p)

    def test_verify_integrity_on_load(self, tmp_path):
        """load_backbone_weights verifies .complete when present."""
        model, cfg = _make_toy_model()
        opt = _make_optimizer(model)
        ckpt_path = tmp_path / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model, opt, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__, training_config={},
        )
        # Corrupt model.safetensors
        (ckpt_path / "model.safetensors").write_bytes(b"CORRUPTED")
        with pytest.raises(CheckpointIntegrityError):
            load_backbone_weights(ckpt_path)


# ---------------------------------------------------------------------------
# Legacy detection + load
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLegacyCheckpoint:
    def test_detects_pt(self, tmp_path):
        pt = tmp_path / "step.pt"
        pt.touch()
        assert is_legacy_checkpoint(pt)

    def test_detects_pth(self, tmp_path):
        pth = tmp_path / "step.pth"
        pth.touch()
        assert is_legacy_checkpoint(pth)

    def test_does_not_detect_directory(self, tmp_path):
        d = tmp_path / "step_00001"
        d.mkdir()
        assert not is_legacy_checkpoint(d)

    def test_does_not_detect_safetensors(self, tmp_path):
        p = tmp_path / "model.safetensors"
        p.touch()
        assert not is_legacy_checkpoint(p)

    def test_does_not_detect_missing_file(self, tmp_path):
        assert not is_legacy_checkpoint(tmp_path / "nonexistent.pt")

    def test_accepts_str_or_path(self, tmp_path):
        pt = tmp_path / "x.pt"
        pt.touch()
        assert is_legacy_checkpoint(str(pt))
        assert is_legacy_checkpoint(pt)

    def test_legacy_load_pretrain(self, tmp_path):
        """load_pretrain_checkpoint handles .pt format."""
        model1, cfg = _make_toy_model()
        opt1 = _make_optimizer(model1)
        _run_step(model1, cfg, opt1)
        pt_path = tmp_path / "ckpt.pt"
        torch.save({
            "model_state_dict": model1.state_dict(),
            "optimizer_state_dict": opt1.state_dict(),
            "global_step": 5,
            "model_config": cfg.__dict__,
            "training_config": {"lr": 1e-3},
        }, pt_path)

        model2, _ = _make_toy_model()
        meta = load_pretrain_checkpoint(pt_path, model2)
        assert meta["global_step"] == 5
        assert meta["model_config"]["d_model"] == cfg.d_model
        for k in model1.state_dict():
            torch.testing.assert_close(
                model1.state_dict()[k], model2.state_dict()[k],
            )

    def test_legacy_load_adapter(self, tmp_path):
        """load_adapter_checkpoint handles .pt format with lora_state_dict key."""
        pt_path = tmp_path / "adapter.pt"
        torch.save({
            "lora_state_dict": {"A.weight": torch.randn(4, 8)},
            "config": {"lora_rank": 4},
            "epoch": 3,
            "step": 300,
            "val_loss": 2.5,
            "val_top1": 0.1,
        }, pt_path)

        loaded = load_adapter_checkpoint(pt_path)
        assert loaded["epoch"] == 3
        assert loaded["step"] == 300
        assert "A.weight" in loaded["adapter_state_dict"]
        assert loaded["val_metrics"]["loss"] == 2.5
        assert loaded["val_metrics"]["top1_accuracy"] == 0.1


# ---------------------------------------------------------------------------
# HF push (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHfPushMocked:
    def test_upload_folder_called(self, tmp_path, mocker):
        from pawn.checkpoint import push_checkpoint_to_hf

        model, cfg = _make_toy_model()
        opt = _make_optimizer(model)
        ckpt_path = tmp_path / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model, opt, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__, training_config={},
        )

        fake_api = mocker.MagicMock()
        mocker.patch("huggingface_hub.HfApi", return_value=fake_api)

        push_checkpoint_to_hf(
            ckpt_path, repo_id="user/repo", branch="run/test", step=1,
        )
        fake_api.upload_folder.assert_called_once()
        call = fake_api.upload_folder.call_args
        assert call.kwargs["repo_id"] == "user/repo"
        assert call.kwargs["revision"] == "run/test"

    def test_uploads_truncated_metrics_when_path_given(self, tmp_path, mocker):
        from pawn.checkpoint import push_checkpoint_to_hf

        model, cfg = _make_toy_model()
        opt = _make_optimizer(model)
        ckpt_path = tmp_path / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model, opt, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__, training_config={},
        )

        metrics_path = tmp_path / "metrics.jsonl"
        metrics_path.write_text(
            '{"type":"train","step":1,"loss":1.0}\n'
            '{"type":"train","step":2,"loss":0.9}\n'
            '{"type":"train","step":3,"loss":0.8}\n'
        )

        fake_api = mocker.MagicMock()
        mocker.patch("huggingface_hub.HfApi", return_value=fake_api)

        push_checkpoint_to_hf(
            ckpt_path, repo_id="user/repo", branch="run/test",
            metrics_path=metrics_path, step=2,
        )
        fake_api.upload_folder.assert_called_once()
        fake_api.upload_file.assert_called_once()

    def test_no_metrics_upload_when_path_none(self, tmp_path, mocker):
        from pawn.checkpoint import push_checkpoint_to_hf

        model, cfg = _make_toy_model()
        opt = _make_optimizer(model)
        ckpt_path = tmp_path / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model, opt, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__, training_config={},
        )

        fake_api = mocker.MagicMock()
        mocker.patch("huggingface_hub.HfApi", return_value=fake_api)
        push_checkpoint_to_hf(
            ckpt_path, repo_id="user/repo", branch="run/test", step=1,
        )
        fake_api.upload_file.assert_not_called()
