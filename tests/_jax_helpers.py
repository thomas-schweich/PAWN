"""Shared test helpers for the ``pawn.jax`` test suite.

Imported by ``tests/test_jax_legacy.py``, ``tests/test_jax_checkpoint.py``,
and ``tests/test_jax_torch_loader.py``. The leading underscore keeps pytest
from collecting it as a test module; importers run their own
``pytest.importorskip("jax" / "torch")`` first so this module is only loaded
when its (torch / safetensors / chess_engine) dependencies resolve.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file as torch_save_file

from pawn._torch_legacy_fixture import CLMConfig
from pawn._torch_legacy_fixture import PAWNCLM


def write_legacy_checkpoint(
    dest: Path, cfg: CLMConfig, *, seed: int = 0
) -> PAWNCLM:
    """Materialise a legacy PyTorch PAWN checkpoint directory at ``dest``.

    Writes ``model.safetensors`` (state_dict) and ``config.json``
    (``{format_version, checkpoint_type, model_config}``) in the exact layout
    a real legacy training checkpoint uses, modulo the ``.complete`` sentinel
    (full pretraining checkpoints have one; HF-format bare directories don't).
    Returns the constructed PyTorch model so callers can compute reference
    forward outputs.
    """
    torch.manual_seed(seed)
    model = PAWNCLM(cfg).eval()
    dest.mkdir(parents=True, exist_ok=True)
    state = {
        k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
    }
    torch_save_file(state, dest / "model.safetensors")
    (dest / "config.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "checkpoint_type": "pretrain",
                "model_config": asdict(cfg),
            }
        ),
        encoding="utf-8",
    )
    return model


def corrupt_safetensors(ckpt_dir: Path) -> None:
    """Flip the last 4 bytes of ``model.safetensors`` to guarantee a hash
    mismatch.

    XOR-ing with ``0xFF`` is the cheapest mutation that cannot leave the file
    bit-identical (in contrast to overwriting with zeros, which has a ~2^-32
    pass-by-chance probability per byte if those bytes were already zero).
    """
    path = ckpt_dir / "model.safetensors"
    with open(path, "r+b") as f:
        f.seek(-4, 2)
        original = f.read(4)
        f.seek(-4, 2)
        f.write(bytes(b ^ 0xFF for b in original))


def stamp_format_version(ckpt_dir: Path, version: int) -> None:
    """Overwrite the checkpoint's ``format_version`` *and re-sign the
    ``.complete`` sentinel*.

    Re-signing is what lets a test exercise the version gate rather than the
    integrity gate — without it, tampering with ``config.json`` invalidates
    the sentinel hash and ``load_model`` / ``load_pawn`` fail on integrity
    long before reaching the version check.
    """
    from pawn.checkpoint import _write_sentinel

    cfg_path = ckpt_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["format_version"] = version
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    _write_sentinel(ckpt_dir)
