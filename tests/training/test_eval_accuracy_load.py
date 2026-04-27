"""Regression tests for ``scripts/eval_accuracy.py:load_model``.

The session postmortem flagged that retro-bottleneck adapters loaded
through the eval path silently fell into the bottleneck-only loader
because ``_detect_adapter_type`` checked for ``bottleneck_dim`` before
``rosa_mode``. The eval reported ~9% top-1 (random) on what should have
been a 51 %+ run. Each test below builds a real adapter, saves a
checkpoint, then reloads via ``load_model`` and verifies the wrapper
class is the one the saved config asks for.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from pawn.adapters.bottleneck import BottleneckCLM
from pawn.adapters.lora import LoRACLM
from pawn.adapters.rosa import RetroBottleneckCLM
from pawn.adapters.sparse import SparseCLM
from pawn.checkpoint import save_adapter_checkpoint, save_pretrain_checkpoint
from pawn.config import CLMConfig
from pawn.model import PAWNCLM
from pawn.specialized_clm import SpecializedCLM


def _import_eval_accuracy() -> Any:
    """``scripts/eval_accuracy.py`` lives outside the package, so load
    it dynamically. Each test calls this once to get ``load_model``."""
    path = (
        Path(__file__).resolve().parents[2] / "scripts" / "eval_accuracy.py"
    )
    spec = importlib.util.spec_from_file_location("eval_accuracy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load eval_accuracy from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save_backbone(tmp_path: Path) -> Path:
    """Persist a toy PAWN backbone to disk and return its directory."""
    cfg = CLMConfig.toy()
    model = PAWNCLM(cfg)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    class _Stub:
        def state_dict(self):  # pragma: no cover — sched/scaler stubs
            return {}

        def load_state_dict(self, _):
            pass

    out = tmp_path / "backbone"
    save_pretrain_checkpoint(
        out,
        model,
        opt,
        _Stub(),
        _Stub(),
        global_step=0,
        model_config=cfg.__dict__,
        training_config={},
    )
    return out


def _save_adapter(
    ckpt_dir: Path,
    state_dict: dict[str, torch.Tensor],
    config: dict,
) -> None:
    save_adapter_checkpoint(
        ckpt_dir,
        state_dict,
        config=config,
        epoch=0,
        step=0,
        val_metrics={},
    )


@pytest.mark.integration
class TestLoadModelDispatch:
    """The crucial guarantee: each saved ``strategy`` / ``rosa_mode``
    combination resolves to the right wrapper class on load."""

    def test_retro_bottleneck_loads_as_retro_not_plain_bottleneck(
        self, tmp_path
    ):
        """The exact regression the session postmortem identified.

        A retro-bottleneck checkpoint carries both ``bottleneck_dim``
        and ``density``. The pre-fix detector matched ``bottleneck_dim``
        first and produced a ``BottleneckCLM`` that ignored the saved
        sparse deltas, eval reporting ~9 %.
        """
        load_model = _import_eval_accuracy().load_model

        backbone_dir = _save_backbone(tmp_path)
        backbone = PAWNCLM(CLMConfig.toy())

        # Decorate the backbone with SparseLinear in place, then wrap
        # with RetroBottleneckCLM (mirrors ``rosa_build_phase3``).
        SparseCLM(backbone, density=0.05, attn_targets=("wq", "wk", "wv", "wo"))
        retro = RetroBottleneckCLM(backbone, bottleneck_dim=4)

        ckpt_dir = tmp_path / "step_00000010"
        _save_adapter(
            ckpt_dir,
            retro.adapter_state_dict(),
            config={
                "strategy": "rosa",
                "rosa_mode": "retro-bottleneck",
                "checkpoint": str(backbone_dir),
                "density": 0.05,
                "sparse_targets": "qkvo",
                "sparse_ffn": False,
                "bottleneck_dim": 4,
                "adapt_attn": True,
                "adapt_ffn": True,
                "adapter_layers": None,
                "lora_targets": "qkvo",
                "lora_ffn": False,
            },
        )

        model, adapter_type = load_model(str(backbone_dir), str(ckpt_dir), "cpu")
        assert adapter_type == "retro_bottleneck"
        assert isinstance(model, RetroBottleneckCLM)

    def test_plain_bottleneck_loads_as_bottleneck(self, tmp_path):
        load_model = _import_eval_accuracy().load_model

        backbone_dir = _save_backbone(tmp_path)
        backbone = PAWNCLM(CLMConfig.toy())
        bn = BottleneckCLM(backbone, bottleneck_dim=4)

        ckpt_dir = tmp_path / "step_00000010"
        _save_adapter(
            ckpt_dir,
            bn.adapter_state_dict(),
            config={
                "strategy": "bottleneck",
                "checkpoint": str(backbone_dir),
                "bottleneck_dim": 4,
                "adapt_attn": True,
                "adapt_ffn": True,
                "adapter_layers": None,
            },
        )

        model, adapter_type = load_model(str(backbone_dir), str(ckpt_dir), "cpu")
        assert adapter_type == "bottleneck"
        assert isinstance(model, BottleneckCLM)
        # And critically NOT a RetroBottleneckCLM — separate class.
        assert not isinstance(model, RetroBottleneckCLM)

    def test_lora_loads_as_lora(self, tmp_path):
        load_model = _import_eval_accuracy().load_model

        backbone_dir = _save_backbone(tmp_path)
        backbone = PAWNCLM(CLMConfig.toy())
        lora = LoRACLM(backbone, rank=2)

        ckpt_dir = tmp_path / "step_00000010"
        _save_adapter(
            ckpt_dir,
            lora.lora_state_dict(),
            config={
                "strategy": "lora",
                "checkpoint": str(backbone_dir),
                "lora_rank": 2,
                "lora_targets": "qkvo",
                "lora_ffn": False,
            },
        )

        model, adapter_type = load_model(str(backbone_dir), str(ckpt_dir), "cpu")
        assert adapter_type == "lora"
        assert isinstance(model, LoRACLM)

    def test_specialized_clm_loads_without_backbone(self, tmp_path):
        """``--checkpoint`` is now optional for spec_clm; the adapter
        weights *are* the entire model. The saved state_dict drops the
        tied ``lm_head.weight`` (matches the production
        ``state_dict_fn`` in ``_build_specialized_clm``)."""
        load_model = _import_eval_accuracy().load_model

        spec = SpecializedCLM(
            vocab_size=CLMConfig().vocab_size,
            d_model=32,
            n_layers=2,
            n_heads=4,
            d_ff=64,
            max_seq_len=32,
        )
        sd = dict(spec.state_dict())
        sd.pop("lm_head.weight", None)
        ckpt_dir = tmp_path / "step_00000010"
        _save_adapter(
            ckpt_dir,
            sd,
            config={
                "strategy": "specialized_clm",
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
                "d_ff": 64,
                "max_seq_len": 32,
            },
        )

        model, adapter_type = load_model(None, str(ckpt_dir), "cpu")
        assert adapter_type == "specialized_clm"
        assert isinstance(model, SpecializedCLM)
        # Tying is restored: lm_head.weight and embed.weight share storage.
        assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()
        # And forward() runs end-to-end.
        ids = torch.zeros(2, 8, dtype=torch.long)
        logits = model(ids)
        assert logits.shape == (2, 8, CLMConfig().vocab_size)
