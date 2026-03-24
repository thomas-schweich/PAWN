"""Checkpoint save/load using safetensors + JSON.

Replaces monolithic torch.save() .pt files with directory-based checkpoints:
  - model.safetensors / adapter.safetensors — tensor data
  - optimizer.safetensors — flattened optimizer state tensors
  - training_state.json — scalars, scheduler, scaler, RNG, optimizer metadata
  - config.json — model and training configuration

Backward compatible: all load functions transparently handle legacy .pt files.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

CHECKPOINT_FORMAT_VERSION = 1
LEGACY_EXTENSIONS = {".pt", ".pth"}


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    """JSON serializer for types not natively supported."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# RNG state serialization
# ---------------------------------------------------------------------------

def _rng_to_json(
    torch_rng: torch.Tensor | None, cuda_rng: torch.Tensor | None
) -> dict:
    data = {}
    if torch_rng is not None:
        data["torch_rng_state"] = base64.b64encode(
            torch_rng.numpy().tobytes()
        ).decode("ascii")
    if cuda_rng is not None:
        data["cuda_rng_state"] = base64.b64encode(
            cuda_rng.numpy().tobytes()
        ).decode("ascii")
    return data


def _json_to_rng(data: dict) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    torch_rng = None
    if "torch_rng_state" in data:
        raw = base64.b64decode(data["torch_rng_state"])
        torch_rng = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    cuda_rng = None
    if "cuda_rng_state" in data:
        raw = base64.b64decode(data["cuda_rng_state"])
        cuda_rng = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return torch_rng, cuda_rng


# ---------------------------------------------------------------------------
# Optimizer state flattening for safetensors
# ---------------------------------------------------------------------------

def _flatten_optimizer_state(
    opt_state_dict: dict,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Flatten optimizer state into safetensors-compatible tensors + JSON metadata.

    AdamW state: state[param_id] = {"step": tensor, "exp_avg": tensor, "exp_avg_sq": tensor}
    Flattened: "state.{param_id}.exp_avg" etc. in the tensors dict.
    param_groups (pure Python) go into metadata.
    """
    tensors: dict[str, torch.Tensor] = {}
    scalars: dict[str, float | int] = {}

    for param_id, param_state in opt_state_dict["state"].items():
        for key, val in param_state.items():
            flat_key = f"state.{param_id}.{key}"
            if isinstance(val, torch.Tensor):
                t = val.cpu().contiguous()
                # safetensors requires at least 1 dim
                if t.ndim == 0:
                    t = t.unsqueeze(0)
                tensors[flat_key] = t
            else:
                scalars[flat_key] = val

    meta = {
        "param_groups": opt_state_dict["param_groups"],
        "scalars": scalars if scalars else None,
    }
    return tensors, meta


def _unflatten_optimizer_state(
    tensors: dict[str, torch.Tensor],
    meta: dict,
    device: str = "cpu",
) -> dict:
    """Reconstruct optimizer state_dict from flattened tensors + metadata."""
    state: dict[int, dict[str, torch.Tensor | float | int]] = {}
    scalars = meta.get("scalars") or {}

    for flat_key, val in tensors.items():
        # "state.{param_id}.{key}"
        parts = flat_key.split(".", 2)
        if len(parts) != 3 or parts[0] != "state":
            continue
        param_id = int(parts[1])
        key = parts[2]
        if param_id not in state:
            state[param_id] = {}
        t = val.to(device)
        # Undo the unsqueeze for scalar tensors (step counters)
        if t.shape == (1,) and key == "step":
            t = t.squeeze(0)
        state[param_id][key] = t

    # Restore any non-tensor scalars
    for flat_key, val in scalars.items():
        parts = flat_key.split(".", 2)
        if len(parts) != 3:
            continue
        param_id = int(parts[1])
        key = parts[2]
        if param_id not in state:
            state[param_id] = {}
        state[param_id][key] = val

    return {
        "state": state,
        "param_groups": meta["param_groups"],
    }


# ---------------------------------------------------------------------------
# Legacy checkpoint detection
# ---------------------------------------------------------------------------

def is_legacy_checkpoint(path: str | Path) -> bool:
    """True if path is a single .pt/.pth file (legacy format)."""
    p = Path(path)
    return p.is_file() and p.suffix in LEGACY_EXTENSIONS


def _load_legacy_pt(path: str | Path, device: str = "cpu") -> dict:
    return torch.load(str(path), map_location=device, weights_only=False)


# ---------------------------------------------------------------------------
# Pretrain checkpoint save/load
# ---------------------------------------------------------------------------

def save_pretrain_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    global_step: int,
    model_config: dict,
    training_config: dict,
) -> None:
    """Save a pretraining checkpoint as a directory of safetensors + JSON files."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # 1. Model weights
    state_dict = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, path / "model.safetensors")

    # 2. Optimizer tensors
    opt_tensors, opt_meta = _flatten_optimizer_state(optimizer.state_dict())
    if opt_tensors:
        save_file(opt_tensors, path / "optimizer.safetensors")

    # 3. Training state (JSON)
    training_state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "global_step": global_step,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "optimizer_meta": opt_meta,
        **_rng_to_json(
            torch.get_rng_state(),
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        ),
    }
    with open(path / "training_state.json", "w") as f:
        json.dump(training_state, f, indent=2, default=_json_default)

    # 4. Config
    config = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_type": "pretrain",
        "model_config": model_config,
        "training_config": training_config,
    }
    with open(path / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=_json_default)


def load_pretrain_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler=None,
    device: str = "cpu",
) -> dict:
    """Load a pretraining checkpoint. Returns metadata dict with global_step, configs, etc.

    Handles both legacy .pt files and new directory format.
    """
    path = Path(path)

    if is_legacy_checkpoint(path):
        ckpt = _load_legacy_pt(path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if ckpt.get("torch_rng_state") is not None:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu().byte())
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu().byte())
        return {
            "global_step": ckpt.get("global_step", 0),
            "model_config": ckpt.get("model_config"),
            "training_config": ckpt.get("training_config"),
        }

    # New directory format
    weights = load_file(path / "model.safetensors", device=device)
    model.load_state_dict(weights)

    with open(path / "training_state.json") as f:
        ts = json.load(f)

    if optimizer and (path / "optimizer.safetensors").exists():
        opt_tensors = load_file(path / "optimizer.safetensors", device=device)
        opt_state = _unflatten_optimizer_state(opt_tensors, ts["optimizer_meta"], device)
        optimizer.load_state_dict(opt_state)

    if scheduler and "scheduler_state_dict" in ts:
        scheduler.load_state_dict(ts["scheduler_state_dict"])

    if scaler and "scaler_state_dict" in ts:
        scaler.load_state_dict(ts["scaler_state_dict"])

    torch_rng, cuda_rng = _json_to_rng(ts)
    if torch_rng is not None:
        torch.set_rng_state(torch_rng.cpu().byte())
    if cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng.cpu().byte())

    with open(path / "config.json") as f:
        config = json.load(f)

    return {
        "global_step": ts.get("global_step", 0),
        "model_config": config.get("model_config"),
        "training_config": config.get("training_config"),
    }


# ---------------------------------------------------------------------------
# Adapter checkpoint save/load
# ---------------------------------------------------------------------------

def save_adapter_checkpoint(
    path: str | Path,
    adapter_state_dict: dict[str, torch.Tensor],
    config: dict,
    epoch: int,
    step: int,
    val_metrics: dict,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler=None,
    extra: dict | None = None,
) -> None:
    """Save an adapter checkpoint as a directory of safetensors + JSON files."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # 1. Adapter weights
    tensors = {k: v.cpu().contiguous() for k, v in adapter_state_dict.items()}
    save_file(tensors, path / "adapter.safetensors")

    # 2. Optimizer tensors (if provided)
    opt_meta = None
    if optimizer is not None:
        opt_tensors, opt_meta = _flatten_optimizer_state(optimizer.state_dict())
        if opt_tensors:
            save_file(opt_tensors, path / "optimizer.safetensors")

    # 3. Training state
    training_state: dict = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "step": step,
        "val_metrics": val_metrics,
    }
    if opt_meta is not None:
        training_state["optimizer_meta"] = opt_meta
    if scheduler is not None:
        training_state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        training_state["scaler_state_dict"] = scaler.state_dict()
    if extra:
        training_state.update(extra)

    with open(path / "training_state.json", "w") as f:
        json.dump(training_state, f, indent=2, default=_json_default)

    # 4. Config
    with open(path / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=_json_default)


def load_adapter_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> dict:
    """Load an adapter checkpoint. Returns dict with adapter weights, config, metrics, etc.

    Handles both legacy .pt files and new directory format.
    """
    path = Path(path)

    if is_legacy_checkpoint(path):
        ckpt = _load_legacy_pt(path, device)
        # Find the adapter state dict key (varies by adapter type)
        adapter_key = None
        for key in ("lora_state_dict", "bottleneck_state_dict", "film_state_dict",
                     "sparse_state_dict", "adapter_state_dict", "model_state_dict"):
            if key in ckpt:
                adapter_key = key
                break
        return {
            "adapter_state_dict": ckpt.get(adapter_key, {}),
            "config": ckpt.get("config", {}),
            "epoch": ckpt.get("epoch", 0),
            "step": ckpt.get("step", 0),
            "val_metrics": {
                "loss": ckpt.get("val_loss"),
                "top1_accuracy": ckpt.get("val_top1"),
            },
            "optimizer_state_dict": ckpt.get("optimizer_state_dict"),
            "scheduler_state_dict": ckpt.get("scheduler_state_dict"),
            "scaler_state_dict": ckpt.get("scaler_state_dict"),
            "best_val_loss": ckpt.get("best_val_loss"),
            "patience_counter": ckpt.get("patience_counter"),
        }

    # New directory format
    adapter_weights = load_file(path / "adapter.safetensors", device=device)

    with open(path / "config.json") as f:
        config = json.load(f)

    ts = {}
    ts_path = path / "training_state.json"
    if ts_path.exists():
        with open(ts_path) as f:
            ts = json.load(f)

    result: dict = {
        "adapter_state_dict": adapter_weights,
        "config": config,
        "epoch": ts.get("epoch", 0),
        "step": ts.get("step", 0),
        "val_metrics": ts.get("val_metrics", {}),
        "best_val_loss": ts.get("best_val_loss"),
        "patience_counter": ts.get("patience_counter"),
    }

    if (path / "optimizer.safetensors").exists() and "optimizer_meta" in ts:
        opt_tensors = load_file(path / "optimizer.safetensors", device=device)
        result["optimizer_state_dict"] = _unflatten_optimizer_state(
            opt_tensors, ts["optimizer_meta"], device
        )

    if "scheduler_state_dict" in ts:
        result["scheduler_state_dict"] = ts["scheduler_state_dict"]

    if "scaler_state_dict" in ts:
        result["scaler_state_dict"] = ts["scaler_state_dict"]

    return result


# ---------------------------------------------------------------------------
# Backbone-only loading (inference)
# ---------------------------------------------------------------------------

def load_backbone_weights(
    path: str | Path,
    device: str = "cpu",
) -> tuple[dict[str, torch.Tensor], dict | None]:
    """Load model weights and config for inference. No optimizer/scheduler.

    Works with:
    - Legacy .pt files (extracts model_state_dict + model_config)
    - New checkpoint directories (reads model.safetensors + config.json)
    - Bare model.safetensors files

    Returns (state_dict, model_config_dict_or_None).
    """
    path = Path(path)

    if is_legacy_checkpoint(path):
        ckpt = _load_legacy_pt(path, device)
        return ckpt["model_state_dict"], ckpt.get("model_config")

    # Directory with model.safetensors
    if path.is_dir():
        sf_path = path / "model.safetensors"
        if not sf_path.exists():
            raise FileNotFoundError(f"No model.safetensors in {path}")
        weights = load_file(sf_path, device=device)
        config = None
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f).get("model_config")
        return weights, config

    # Bare safetensors file
    if path.suffix == ".safetensors":
        weights = load_file(path, device=device)
        config = None
        config_path = path.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f).get("model_config")
        return weights, config

    raise ValueError(f"Unrecognized checkpoint format: {path}")
