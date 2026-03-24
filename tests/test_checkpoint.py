"""Tests for pawn.checkpoint safetensors save/load."""

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from pawn.config import CLMConfig
from pawn.model import PAWNCLM
import pytest

from pawn.checkpoint import (
    save_pretrain_checkpoint,
    load_pretrain_checkpoint,
    save_adapter_checkpoint,
    load_adapter_checkpoint,
    load_backbone_weights,
    is_legacy_checkpoint,
    IncompleteCheckpointError,
    CheckpointIntegrityError,
    _flatten_optimizer_state,
    _unflatten_optimizer_state,
    _rng_to_json,
    _json_to_rng,
    _verify_complete_sentinel,
)


def _make_toy_model(device="cpu"):
    cfg = CLMConfig.toy()
    model = PAWNCLM(cfg).to(device)
    return model, cfg


def _make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


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


def test_pretrain_checkpoint_roundtrip():
    """Save and load a pretrain checkpoint, verify model weights match."""
    model1, cfg = _make_toy_model()
    opt1 = _make_optimizer(model1)
    sched1 = FakeScheduler()
    sched1._step = 42
    scaler1 = FakeScaler()

    # Run a fake step to populate optimizer state
    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    mask = torch.ones(2, cfg.max_seq_len, dtype=torch.bool)
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    loss, _ = model1.forward_train(x, mask, targets)
    loss.backward()
    opt1.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model1, opt1, sched1, scaler1,
            global_step=1,
            model_config=cfg.__dict__,
            training_config={"lr": 1e-3},
        )

        # Verify files exist
        assert (ckpt_path / "model.safetensors").exists()
        assert (ckpt_path / "optimizer.safetensors").exists()
        assert (ckpt_path / "training_state.json").exists()
        assert (ckpt_path / "config.json").exists()

        # Load into fresh model
        model2, _ = _make_toy_model()
        opt2 = _make_optimizer(model2)
        sched2 = FakeScheduler()
        scaler2 = FakeScaler()

        meta = load_pretrain_checkpoint(
            ckpt_path, model2, opt2, sched2, scaler2
        )

        assert meta["global_step"] == 1
        assert meta["model_config"]["d_model"] == cfg.d_model
        assert sched2._step == 42

        # Verify model weights match
        for k in model1.state_dict():
            torch.testing.assert_close(
                model1.state_dict()[k],
                model2.state_dict()[k],
            )


def test_optimizer_flatten_unflatten():
    """Verify optimizer state survives flatten/unflatten."""
    model, cfg = _make_toy_model()
    opt = _make_optimizer(model)

    # Populate optimizer state
    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    mask = torch.ones(2, cfg.max_seq_len, dtype=torch.bool)
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    loss, _ = model.forward_train(x, mask, targets)
    loss.backward()
    opt.step()

    orig_state = opt.state_dict()
    tensors, meta = _flatten_optimizer_state(orig_state)

    # All tensors should be named "state.{id}.{key}"
    for k in tensors:
        assert k.startswith("state."), f"Unexpected key: {k}"

    restored = _unflatten_optimizer_state(tensors, meta)

    # Verify param_groups match
    assert len(restored["param_groups"]) == len(orig_state["param_groups"])

    # Verify state tensors match
    for param_id in orig_state["state"]:
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                orig_state["state"][param_id][key],
                restored["state"][param_id][key],
            )


def test_rng_roundtrip():
    """Verify RNG state survives JSON serialization."""
    torch_rng = torch.get_rng_state()
    cuda_rng = None

    data = _rng_to_json(torch_rng, cuda_rng)
    assert "torch_rng_state" in data
    assert "cuda_rng_state" not in data

    restored_torch, restored_cuda = _json_to_rng(data)
    assert restored_cuda is None
    assert torch.equal(torch_rng, restored_torch)


def test_adapter_checkpoint_roundtrip():
    """Save and load an adapter checkpoint."""
    adapter_weights = {
        "down.weight": torch.randn(8, 64),
        "up.weight": torch.zeros(64, 8),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "best"
        save_adapter_checkpoint(
            ckpt_path,
            adapter_weights,
            config={"bottleneck_dim": 8, "checkpoint_type": "bottleneck"},
            epoch=5,
            step=1000,
            val_metrics={"loss": 3.14, "top1_accuracy": 0.07},
            extra={"best_val_loss": 3.10, "patience_counter": 2},
        )

        assert (ckpt_path / "adapter.safetensors").exists()
        assert (ckpt_path / "config.json").exists()
        assert (ckpt_path / "training_state.json").exists()
        assert not (ckpt_path / "optimizer.safetensors").exists()  # no optimizer passed

        loaded = load_adapter_checkpoint(ckpt_path)
        assert loaded["epoch"] == 5
        assert loaded["step"] == 1000
        assert loaded["val_metrics"]["loss"] == 3.14
        assert loaded["best_val_loss"] == 3.10

        for k in adapter_weights:
            torch.testing.assert_close(adapter_weights[k], loaded["adapter_state_dict"][k])


def test_load_backbone_weights_new_format():
    """load_backbone_weights works with new directory format."""
    model1, cfg = _make_toy_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "step_00000001"
        opt = _make_optimizer(model1)
        save_pretrain_checkpoint(
            ckpt_path, model1, opt, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__, training_config={},
        )

        state_dict, model_config = load_backbone_weights(ckpt_path)
        assert model_config["d_model"] == cfg.d_model
        for k in model1.state_dict():
            torch.testing.assert_close(model1.state_dict()[k], state_dict[k])


def test_load_backbone_weights_legacy():
    """load_backbone_weights works with legacy .pt files."""
    model1, cfg = _make_toy_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        pt_path = Path(tmpdir) / "model.pt"
        torch.save({
            "model_state_dict": model1.state_dict(),
            "model_config": cfg.__dict__,
        }, pt_path)

        state_dict, model_config = load_backbone_weights(pt_path)
        assert model_config["d_model"] == cfg.d_model
        for k in model1.state_dict():
            torch.testing.assert_close(model1.state_dict()[k], state_dict[k])


def test_is_legacy_checkpoint():
    """Detect legacy vs new format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pt = Path(tmpdir) / "step.pt"
        pt.touch()
        assert is_legacy_checkpoint(pt)

        d = Path(tmpdir) / "step_00001"
        d.mkdir()
        assert not is_legacy_checkpoint(d)


def test_config_json_contents():
    """Verify config.json has expected structure."""
    model, cfg = _make_toy_model()
    opt = _make_optimizer(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model, opt, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__,
            training_config={"lr": 1e-3, "batch_size": 32},
        )

        with open(ckpt_path / "config.json") as f:
            config = json.load(f)

        assert config["format_version"] == 1
        assert config["checkpoint_type"] == "pretrain"
        assert config["model_config"]["d_model"] == cfg.d_model
        assert config["training_config"]["lr"] == 1e-3


def test_complete_sentinel_written():
    """Verify .complete sentinel is created with SHA-256 hashes."""
    model, cfg = _make_toy_model()
    opt = _make_optimizer(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model, opt, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__, training_config={},
        )

        sentinel_path = ckpt_path / ".complete"
        assert sentinel_path.exists()

        with open(sentinel_path) as f:
            sentinel = json.load(f)

        assert "files" in sentinel
        assert "model.safetensors" in sentinel["files"]
        assert "config.json" in sentinel["files"]
        assert "training_state.json" in sentinel["files"]
        # Each hash should be a 64-char hex string
        for name, h in sentinel["files"].items():
            assert len(h) == 64, f"Bad hash length for {name}: {len(h)}"


def test_no_tmp_directory_after_save():
    """Verify .tmp directory is cleaned up after successful save."""
    model, cfg = _make_toy_model()
    opt = _make_optimizer(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model, opt, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__, training_config={},
        )

        tmp_path = Path(tmpdir) / "step_00000001.tmp"
        assert not tmp_path.exists(), ".tmp directory should be removed after rename"


def test_incomplete_checkpoint_raises():
    """Loading a checkpoint without .complete raises IncompleteCheckpointError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "step_bad"
        ckpt_path.mkdir()
        # Write some files but no .complete
        (ckpt_path / "model.safetensors").touch()
        (ckpt_path / "config.json").write_text("{}")

        with pytest.raises(IncompleteCheckpointError):
            _verify_complete_sentinel(ckpt_path)


def test_corrupted_file_raises():
    """Loading a checkpoint with a tampered file raises CheckpointIntegrityError."""
    model, cfg = _make_toy_model()
    opt = _make_optimizer(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model, opt, FakeScheduler(), FakeScaler(),
            global_step=1, model_config=cfg.__dict__, training_config={},
        )

        # Corrupt a file after save
        config_path = ckpt_path / "config.json"
        config_path.write_text("CORRUPTED DATA")

        with pytest.raises(CheckpointIntegrityError):
            _verify_complete_sentinel(ckpt_path)


def test_adapter_checkpoint_has_sentinel():
    """Adapter checkpoints also get .complete sentinel."""
    adapter_weights = {"down.weight": torch.randn(8, 64)}

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "best"
        save_adapter_checkpoint(
            ckpt_path, adapter_weights,
            config={"checkpoint_type": "bottleneck"},
            epoch=1, step=100, val_metrics={"loss": 3.0},
        )

        assert (ckpt_path / ".complete").exists()
        # Should not raise
        _verify_complete_sentinel(ckpt_path)
