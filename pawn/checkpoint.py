"""Checkpoint save/load using safetensors + JSON.

Directory-based checkpoints:
  - model.safetensors / adapter.safetensors — tensor data
  - optimizer.safetensors — flattened optimizer state tensors
  - training_state.json — scalars, scheduler, scaler, RNG, optimizer metadata
  - config.json — model and training configuration
  - .complete — SHA-256 hashes of all files (integrity sentinel)

Writes are atomic: files are written to a .tmp directory, then renamed.
Loads always verify the .complete sentinel and SHA-256 hashes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

CHECKPOINT_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class IncompleteCheckpointError(Exception):
    """Raised when a checkpoint directory is missing its .complete sentinel."""
    pass


class CheckpointIntegrityError(Exception):
    """Raised when a checkpoint file's SHA-256 hash doesn't match .complete."""
    pass


# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1MB chunks
            h.update(chunk)
    return h.hexdigest()


def _write_complete_sentinel(directory: Path) -> None:
    """Write .complete sentinel with SHA-256 hashes of all checkpoint files."""
    hashes = {}
    for f in sorted(directory.iterdir()):
        if f.name == ".complete" or f.is_dir():
            continue
        hashes[f.name] = _sha256_file(f)

    sentinel = {"format_version": CHECKPOINT_FORMAT_VERSION, "files": hashes}
    with open(directory / ".complete", "w") as f:
        json.dump(sentinel, f, indent=2)


def _verify_complete_sentinel(directory: Path) -> None:
    """Verify .complete sentinel exists and all hashes match.

    Raises IncompleteCheckpointError if sentinel is missing.
    Raises CheckpointIntegrityError if any hash mismatches.
    """
    sentinel_path = directory / ".complete"
    if not sentinel_path.exists():
        raise IncompleteCheckpointError(
            f"Checkpoint {directory} is missing .complete sentinel — "
            f"likely a partial write from a crashed or interrupted save."
        )

    with open(sentinel_path) as f:
        sentinel = json.load(f)

    for filename, expected_hash in sentinel["files"].items():
        filepath = directory / filename
        if not filepath.exists():
            raise CheckpointIntegrityError(
                f"File {filename} listed in .complete but missing from {directory}"
            )
        actual_hash = _sha256_file(filepath)
        if actual_hash != expected_hash:
            raise CheckpointIntegrityError(
                f"SHA-256 mismatch for {filename} in {directory}: "
                f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
            )


# ---------------------------------------------------------------------------
# Atomic directory write
# ---------------------------------------------------------------------------

def _atomic_directory_write(target: Path):
    """Context manager for atomic directory writes.

    Usage:
        with _atomic_directory_write(Path("step_00001")) as tmp:
            save_file(tensors, tmp / "model.safetensors")
            ...
        # Directory is now at step_00001/ with .complete sentinel
    """
    class _AtomicDir:
        def __init__(self, target: Path):
            self.target = target
            self.tmp = target.parent / f"{target.name}.tmp"

        def __enter__(self) -> Path:
            # Clean up any leftover .tmp from a previous crash
            if self.tmp.exists():
                shutil.rmtree(self.tmp)
            self.tmp.mkdir(parents=True, exist_ok=True)
            return self.tmp

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                # Error during write — clean up temp dir
                if self.tmp.exists():
                    shutil.rmtree(self.tmp)
                return False

            # Write .complete sentinel with hashes
            _write_complete_sentinel(self.tmp)

            # Atomic rename (same filesystem)
            if self.target.exists():
                shutil.rmtree(self.target)
            os.rename(self.tmp, self.target)
            return False

    return _AtomicDir(target)


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
    """Flatten optimizer state into safetensors-compatible tensors + JSON metadata."""
    tensors: dict[str, torch.Tensor] = {}
    scalars: dict[str, float | int] = {}

    for param_id, param_state in opt_state_dict["state"].items():
        for key, val in param_state.items():
            flat_key = f"state.{param_id}.{key}"
            if isinstance(val, torch.Tensor):
                t = val.cpu().contiguous()
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
        parts = flat_key.split(".", 2)
        if len(parts) != 3 or parts[0] != "state":
            continue
        try:
            param_id = int(parts[1])
        except ValueError:
            continue
        key = parts[2]
        if param_id not in state:
            state[param_id] = {}
        t = val.to(device)
        if t.shape == (1,) and key == "step":
            t = t.squeeze(0)
        state[param_id][key] = t

    for flat_key, val in scalars.items():
        parts = flat_key.split(".", 2)
        if len(parts) != 3:
            continue
        try:
            param_id = int(parts[1])
        except ValueError:
            continue
        key = parts[2]
        if param_id not in state:
            state[param_id] = {}
        state[param_id][key] = val

    return {
        "state": state,
        "param_groups": meta["param_groups"],
    }


# ---------------------------------------------------------------------------
# Peek at checkpoint metadata without loading weights
# ---------------------------------------------------------------------------

def read_checkpoint_metadata(path: str | Path) -> dict:
    """Return ``{"model_config": ..., "training_config": ...}`` from a
    checkpoint without touching model weights or verifying the hash
    sentinel. Used by resume/eval paths that need to peek at the saved
    sequence format before building data pipelines.

    Raises ``FileNotFoundError`` if the path or its ``config.json`` is
    missing.
    """
    p = Path(path)
    cfg_path = p / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.json at {cfg_path}")
    with open(cfg_path) as f:
        config = json.load(f)
    return {
        "model_config": config.get("model_config"),
        "training_config": config.get("training_config"),
    }


def get_prepend_outcome(training_config: dict | None) -> bool:
    """Return the saved ``prepend_outcome`` flag for a checkpoint.

    Reads ``training_config["prepend_outcome"]`` and returns it as a
    bool. Raises ``ValueError`` when the field is absent — every current
    checkpoint writes the full ``TrainingConfig.__dict__``, so a missing
    field means either a partially-written checkpoint or hand-edited
    config.json; callers must fail closed rather than guess the
    sequence format.
    """
    training = training_config or {}
    if "prepend_outcome" not in training:
        raise ValueError(
            "training_config has no 'prepend_outcome' field — cannot "
            "determine sequence format. Pass prepend_outcome explicitly "
            "in the run config."
        )
    return bool(training["prepend_outcome"])


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
    extra: dict | None = None,
) -> None:
    """Save a pretraining checkpoint atomically.

    Writes to {path}.tmp/, then renames to {path}/ with .complete sentinel.
    """
    path = Path(path)

    with _atomic_directory_write(path) as tmp:
        # 1. Model weights
        state_dict = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}
        save_file(state_dict, tmp / "model.safetensors")

        # 2. Optimizer tensors
        opt_tensors, opt_meta = _flatten_optimizer_state(optimizer.state_dict())
        if opt_tensors:
            save_file(opt_tensors, tmp / "optimizer.safetensors")

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
        if extra:
            collisions = extra.keys() & training_state.keys()
            if collisions:
                raise ValueError(
                    f"extra keys collide with training_state: {collisions}"
                )
            training_state.update(extra)
        with open(tmp / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2, default=_json_default)

        # 4. Config
        config = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "checkpoint_type": "pretrain",
            "model_config": model_config,
            "training_config": training_config,
        }
        with open(tmp / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=_json_default)


def load_pretrain_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler=None,
    device: str = "cpu",
) -> dict:
    """Load a pretraining checkpoint with integrity verification.

    Verifies the .complete sentinel and SHA-256 hashes before loading.
    """
    path = Path(path)

    _verify_complete_sentinel(path)

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
        "best_val_loss": ts.get("best_val_loss"),
        "best_late_legality": ts.get("best_late_legality"),
        "patience_counter": ts.get("patience_counter"),
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
    """Save an adapter checkpoint atomically."""
    path = Path(path)

    with _atomic_directory_write(path) as tmp:
        # 1. Adapter weights
        tensors = {k: v.cpu().contiguous() for k, v in adapter_state_dict.items()}
        save_file(tensors, tmp / "adapter.safetensors")

        # 2. Optimizer tensors (if provided)
        opt_meta = None
        if optimizer is not None:
            opt_tensors, opt_meta = _flatten_optimizer_state(optimizer.state_dict())
            if opt_tensors:
                save_file(opt_tensors, tmp / "optimizer.safetensors")

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
            collisions = extra.keys() & training_state.keys()
            if collisions:
                raise ValueError(
                    f"extra keys collide with training_state: {collisions}"
                )
            training_state.update(extra)

        with open(tmp / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2, default=_json_default)

        # 4. Config
        with open(tmp / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=_json_default)


def load_adapter_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> dict:
    """Load an adapter checkpoint with integrity verification."""
    path = Path(path)

    _verify_complete_sentinel(path)

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
    """Load model weights and config for inference with integrity verification.

    Works with:
    - HuggingFace repo IDs (e.g. "thomas-schweich/pawn-small") — downloads
      model.safetensors + config.json via huggingface_hub
    - Checkpoint directories (reads model.safetensors + config.json, verifies .complete)
    - Bare model.safetensors files (no .complete check — used for HF downloads)

    Returns (state_dict, model_config_dict_or_None).
    """
    path_str = str(path)

    # HuggingFace repo ID: contains "/" and doesn't exist as a local path
    if "/" in path_str and not Path(path_str).exists():
        return _load_from_hf_repo(path_str, device)

    path = Path(path)

    # Directory with model.safetensors
    if path.is_dir():
        sf_path = path / "model.safetensors"
        if not sf_path.exists():
            raise FileNotFoundError(f"No model.safetensors in {path}")
        # Verify integrity if .complete exists (new format checkpoints)
        if (path / ".complete").exists():
            _verify_complete_sentinel(path)
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


def _load_from_hf_repo(
    repo_id: str,
    device: str = "cpu",
) -> tuple[dict[str, torch.Tensor], dict | None]:
    """Download and load model weights from a HuggingFace model repo."""
    from huggingface_hub import hf_hub_download

    print(f"Downloading weights from HuggingFace: {repo_id}")
    sf_path = hf_hub_download(repo_id, "model.safetensors")
    weights = load_file(sf_path, device=device)

    config = None
    try:
        config_path = hf_hub_download(repo_id, "config.json")
        with open(config_path) as f:
            config = json.load(f).get("model_config")
    except Exception:
        pass

    return weights, config


# ---------------------------------------------------------------------------
# HuggingFace push
# ---------------------------------------------------------------------------

def push_checkpoint_to_hf(
    checkpoint_path: str | Path,
    repo_id: str,
    branch: str,
    metrics_path: str | Path | None = None,
    step: int = 0,
) -> None:
    """Push a complete checkpoint directory to a HuggingFace repo branch.

    Uploads checkpoint files to checkpoints/step_NNNN/ on the branch.
    Optionally uploads metrics.jsonl (truncated to current step) to the root.

    Requires HF_TOKEN environment variable or prior `huggingface_hub.login()`.
    """
    from huggingface_hub import HfApi

    checkpoint_path = Path(checkpoint_path)
    api = HfApi()

    # Ensure branch exists
    try:
        api.create_branch(repo_id, repo_type="model", branch=branch, exist_ok=True)
    except Exception:
        pass  # Branch may already exist

    # Upload checkpoint directory
    api.upload_folder(
        folder_path=str(checkpoint_path),
        path_in_repo=f"checkpoints/{checkpoint_path.name}",
        repo_id=repo_id,
        repo_type="model",
        revision=branch,
        commit_message=f"Checkpoint step {step}",
    )

    # Upload truncated metrics.jsonl to repo root
    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        if metrics_path.exists():
            import tempfile
            # Truncate metrics to current step
            truncated_lines = []
            with open(metrics_path) as f:
                for line in f:
                    truncated_lines.append(line)
                    try:
                        record = json.loads(line)
                        if record.get("type") in ("train", "val") and record.get("step", 0) >= step:
                            break
                    except json.JSONDecodeError:
                        continue

            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
                tmp.writelines(truncated_lines)
                tmp_path = tmp.name

            try:
                api.upload_file(
                    path_or_fileobj=tmp_path,
                    path_in_repo="metrics.jsonl",
                    repo_id=repo_id,
                    repo_type="model",
                    revision=branch,
                    commit_message=f"Metrics through step {step}",
                )
            finally:
                os.unlink(tmp_path)
