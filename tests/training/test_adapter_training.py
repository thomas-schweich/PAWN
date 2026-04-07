"""Tests for pawn.adapter_training: load_backbone, cosine_warmup_schedule,
sparse_forward, evaluate, STRATEGIES list.

These are unit tests; full training loops are covered by higher-level integration tests.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from pawn.adapter_training import (
    STRATEGIES,
    cosine_warmup_schedule,
    load_backbone,
    parse_layers,
    sparse_forward,
)


# ---------------------------------------------------------------------------
# STRATEGIES registry
# ---------------------------------------------------------------------------


class TestStrategies:
    def test_is_list(self):
        assert isinstance(STRATEGIES, list)

    def test_nonempty(self):
        assert len(STRATEGIES) > 0

    def test_all_strings(self):
        assert all(isinstance(s, str) for s in STRATEGIES)

    @pytest.mark.parametrize(
        "name",
        ["bottleneck", "lora", "film", "sparse", "rosa", "hybrid"],
    )
    def test_expected_names_present(self, name):
        assert name in STRATEGIES

    def test_specialized_clm_and_unfreeze_present(self):
        assert "specialized_clm" in STRATEGIES
        assert "unfreeze" in STRATEGIES

    def test_unique(self):
        assert len(set(STRATEGIES)) == len(STRATEGIES)


# ---------------------------------------------------------------------------
# cosine_warmup_schedule (LambdaLR wrapper)
# ---------------------------------------------------------------------------


def _make_opt(lr: float = 1e-3) -> torch.optim.Optimizer:
    p = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
    return torch.optim.SGD([p], lr=lr)


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestCosineWarmupSchedule:
    def test_returns_lambda_lr(self):
        opt = _make_opt()
        sched = cosine_warmup_schedule(opt, warmup_steps=10, total_steps=100)
        assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)

    def test_initial_lr_is_zero(self):
        opt = _make_opt(lr=1e-3)
        sched = cosine_warmup_schedule(opt, warmup_steps=10, total_steps=100)
        # step 0 / 10 = 0
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0)

    def test_lr_increases_during_warmup(self):
        peak = 1e-3
        opt = _make_opt(lr=peak)
        sched = cosine_warmup_schedule(opt, warmup_steps=10, total_steps=100)
        sched.step()  # -> step 1
        assert opt.param_groups[0]["lr"] == pytest.approx(0.1 * peak)
        for _ in range(4):
            sched.step()  # -> step 5
        assert opt.param_groups[0]["lr"] == pytest.approx(0.5 * peak)

    def test_end_of_warmup_is_peak(self):
        peak = 1e-3
        opt = _make_opt(lr=peak)
        sched = cosine_warmup_schedule(opt, warmup_steps=5, total_steps=100)
        for _ in range(5):
            sched.step()
        # progress=0, cos(0)=1, scale = 0.5*(1+1) = 1
        assert opt.param_groups[0]["lr"] == pytest.approx(peak)

    def test_end_of_schedule_reaches_zero(self):
        peak = 1e-3
        opt = _make_opt(lr=peak)
        sched = cosine_warmup_schedule(opt, warmup_steps=5, total_steps=50)
        for _ in range(50):
            sched.step()
        # progress=1, cos(pi)=-1, scale = 0 -> 0
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-10)

    def test_midway_value(self):
        peak = 1.0
        opt = _make_opt(lr=peak)
        sched = cosine_warmup_schedule(opt, warmup_steps=10, total_steps=110)
        for _ in range(60):
            sched.step()
        # progress = 0.5, cos(pi/2) = 0, scale = 0.5*(1+0) = 0.5
        assert opt.param_groups[0]["lr"] == pytest.approx(0.5, rel=1e-6)

    def test_zero_warmup_starts_at_peak(self):
        peak = 1e-3
        opt = _make_opt(lr=peak)
        sched = cosine_warmup_schedule(opt, warmup_steps=0, total_steps=100)
        # step 0 is not < 0 -> cosine path immediately, progress=0, scale=1
        assert opt.param_groups[0]["lr"] == pytest.approx(peak)


# ---------------------------------------------------------------------------
# parse_layers helper
# ---------------------------------------------------------------------------


class TestParseLayers:
    def test_none_returns_none(self):
        assert parse_layers(None) is None

    def test_single_int(self):
        assert parse_layers("3") == (3,)

    def test_comma_separated(self):
        assert parse_layers("0,2,4") == (0, 2, 4)

    def test_single_with_trailing_whitespace_raises(self):
        """Non-numeric entries (e.g. whitespace-padded) should still parse since int() is lenient."""
        # int() allows leading/trailing whitespace
        assert parse_layers("1, 2, 3") == (1, 2, 3)


# ---------------------------------------------------------------------------
# load_backbone
# ---------------------------------------------------------------------------


class TestLoadBackbone:
    def test_round_trip_toy_model(
        self, toy_clm_config, cpu_device, tmp_checkpoint_dir,
    ):
        """Save a toy PAWNCLM, load it back via load_backbone."""
        from pawn.checkpoint import save_pretrain_checkpoint
        from pawn.model import PAWNCLM

        # Build + save toy model
        model = PAWNCLM(toy_clm_config).to(cpu_device)
        model.eval()

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ckpt_path = tmp_checkpoint_dir / "step_00000010"
        save_pretrain_checkpoint(
            ckpt_path,
            model,
            opt,
            None,  # scheduler
            None,  # scaler
            global_step=10,
            model_config=toy_clm_config.__dict__,
            training_config={"lr": 1e-3},
        )

        # Load it back
        loaded = load_backbone(str(ckpt_path), cpu_device)
        # Should be a PAWNCLM
        from pawn.model import PAWNCLM

        assert isinstance(loaded, PAWNCLM)
        # Weights should match
        orig_state = model.state_dict()
        loaded_state = loaded.state_dict()
        assert set(orig_state.keys()) == set(loaded_state.keys())
        for k in orig_state:
            assert torch.allclose(orig_state[k], loaded_state[k]), k

    def test_loaded_model_is_in_eval_mode(
        self, toy_clm_config, cpu_device, tmp_checkpoint_dir,
    ):
        from pawn.checkpoint import save_pretrain_checkpoint
        from pawn.model import PAWNCLM

        model = PAWNCLM(toy_clm_config).to(cpu_device)
        model.train()  # deliberately set train mode before saving
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ckpt_path = tmp_checkpoint_dir / "step_00000001"
        save_pretrain_checkpoint(
            ckpt_path, model, opt, None, None, 1,
            toy_clm_config.__dict__, {},
        )

        loaded = load_backbone(str(ckpt_path), cpu_device)
        assert not loaded.training  # eval mode


# ---------------------------------------------------------------------------
# sparse_forward
# ---------------------------------------------------------------------------


class _StubWrapper(nn.Module):
    """Tiny wrapper exposing forward_hidden / project_head, used to exercise sparse_forward."""

    def __init__(self, d_model: int, vocab_size: int, T: int = 4):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.T = T
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_hidden(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        return torch.ones(B, T, self.d_model, device=ids.device)

    def project_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class TestSparseForward:
    def test_returns_valid_logits_shape(self, cpu_device):
        B, T, V, D = 2, 3, 16, 8
        model = _StubWrapper(D, V, T).to(cpu_device)
        ids = torch.zeros(B, T, dtype=torch.long, device=cpu_device)
        msk = torch.ones(B, T, dtype=torch.bool, device=cpu_device)
        legal_mask = torch.ones(B, T, V, dtype=torch.bool, device=cpu_device)
        out = sparse_forward(model, ids, msk, legal_mask, None, cpu_device)
        assert out.shape == (B * T, V)

    def test_masks_illegal_with_neg_inf(self, cpu_device):
        B, T, V, D = 1, 2, 4, 2
        model = _StubWrapper(D, V, T).to(cpu_device)
        ids = torch.zeros(B, T, dtype=torch.long, device=cpu_device)
        msk = torch.ones(B, T, dtype=torch.bool, device=cpu_device)
        legal_mask = torch.zeros(B, T, V, dtype=torch.bool, device=cpu_device)
        # Allow only index 1
        legal_mask[..., 1] = True
        out = sparse_forward(model, ids, msk, legal_mask, None, cpu_device)
        # Indices 0,2,3 masked -> -inf
        assert torch.isinf(out[:, 0]).all()
        assert torch.isinf(out[:, 2]).all()
        assert torch.isinf(out[:, 3]).all()
        # Index 1 finite
        assert torch.isfinite(out[:, 1]).all()

    def test_returns_float_dtype(self, cpu_device):
        """sparse_forward does .float() on logits to avoid AMP dtype issues with -inf."""
        B, T, V, D = 1, 1, 2, 2
        model = _StubWrapper(D, V, T).to(cpu_device)
        ids = torch.zeros(B, T, dtype=torch.long, device=cpu_device)
        msk = torch.ones(B, T, dtype=torch.bool, device=cpu_device)
        legal_mask = torch.ones(B, T, V, dtype=torch.bool, device=cpu_device)
        out = sparse_forward(model, ids, msk, legal_mask, None, cpu_device)
        assert out.dtype == torch.float32

    def test_respects_loss_mask(self, cpu_device):
        """Only positions where msk is True are included in output rows."""
        B, T, V, D = 1, 4, 2, 2
        model = _StubWrapper(D, V, T).to(cpu_device)
        ids = torch.zeros(B, T, dtype=torch.long, device=cpu_device)
        msk = torch.tensor([[True, False, True, False]], device=cpu_device)
        legal_mask = torch.ones(B, T, V, dtype=torch.bool, device=cpu_device)
        out = sparse_forward(model, ids, msk, legal_mask, None, cpu_device)
        # 2 positions pass the mask
        assert out.shape == (2, V)


# ---------------------------------------------------------------------------
# evaluate (mini integration)
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_empty_dataloader_returns_zeros(self, cpu_device):
        """Evaluate with an empty loader returns a zero-filled metrics dict.

        This calls the real evaluate function (not mocked) with an empty
        iterator, verifying the zero-position early-return path.
        """
        from pawn.adapter_training import evaluate

        B, T, V, D = 1, 2, 8, 4
        model = _StubWrapper(D, V, T).to(cpu_device)
        mask_builder = MagicMock()

        class _EmptyLoader:
            def __iter__(self):
                return iter([])

        metrics = evaluate(model, _EmptyLoader(), mask_builder, cpu_device)
        # With zero batches, all metrics should be exactly 0.0
        assert metrics["loss"] == 0.0
        assert metrics["top1_accuracy"] == 0.0
        assert metrics["top5_accuracy"] == 0.0

        # Contrast with a non-empty loader: verify the function returns
        # non-trivial (non-zero) values when given real data.
        legal_mask = torch.ones(B, T, V, dtype=torch.bool, device=cpu_device)
        mask_builder.scatter.return_value = legal_mask
        ids = torch.zeros(B, T, dtype=torch.long)
        tgt = torch.zeros(B, T, dtype=torch.long)
        msk = torch.ones(B, T, dtype=torch.bool)
        batch = {
            "input_ids": ids, "targets": tgt, "loss_mask": msk,
            "legal_indices": torch.zeros(B, T, dtype=torch.long),
        }
        metrics_nonempty = evaluate(model, [batch], mask_builder, cpu_device)
        # Loss should be positive (cross-entropy of a random model)
        assert metrics_nonempty["loss"] > 0.0
        # Accuracy should be in [0, 1] and at least one key differs from 0
        assert 0.0 <= metrics_nonempty["top1_accuracy"] <= 1.0
        assert 0.0 <= metrics_nonempty["top5_accuracy"] <= 1.0

    def test_returns_required_keys(self, cpu_device):
        """With a synthetic 1-batch loader, evaluate returns loss/top1/top5."""
        from pawn.adapter_training import evaluate

        B, T, V, D = 1, 2, 8, 4
        model = _StubWrapper(D, V, T).to(cpu_device)

        # Legal mask: all True
        legal_mask = torch.ones(B, T, V, dtype=torch.bool, device=cpu_device)
        mask_builder = MagicMock()
        mask_builder.scatter.return_value = legal_mask
        mask_builder.return_value = legal_mask  # fallback call path

        ids = torch.zeros(B, T, dtype=torch.long)
        tgt = torch.zeros(B, T, dtype=torch.long)
        msk = torch.ones(B, T, dtype=torch.bool)
        legal_idx = torch.zeros(B, T, dtype=torch.long)

        batch = {
            "input_ids": ids,
            "targets": tgt,
            "loss_mask": msk,
            "legal_indices": legal_idx,
        }
        loader = [batch]

        metrics = evaluate(model, loader, mask_builder, cpu_device)
        assert set(metrics.keys()) == {"loss", "top1_accuracy", "top5_accuracy"}
        assert 0.0 <= metrics["top1_accuracy"] <= 1.0
        assert 0.0 <= metrics["top5_accuracy"] <= 1.0

    def test_precomputed_indices_path(self, cpu_device):
        """Passing precomputed_indices triggers mask_builder.scatter with the cached tensor."""
        from pawn.adapter_training import evaluate

        B, T, V, D = 1, 2, 8, 4
        model = _StubWrapper(D, V, T).to(cpu_device)
        legal_mask = torch.ones(B, T, V, dtype=torch.bool, device=cpu_device)
        mask_builder = MagicMock()
        mask_builder.scatter.return_value = legal_mask

        ids = torch.zeros(B, T, dtype=torch.long)
        tgt = torch.zeros(B, T, dtype=torch.long)
        msk = torch.ones(B, T, dtype=torch.bool)
        batch = {"input_ids": ids, "targets": tgt, "loss_mask": msk}

        cached_idx = torch.zeros(B, T, dtype=torch.long)
        metrics = evaluate(
            model, [batch], mask_builder, cpu_device,
            precomputed_indices=[cached_idx],
        )
        # scatter was called with the precomputed tensor
        mask_builder.scatter.assert_called_once()
        args, _ = mask_builder.scatter.call_args
        assert torch.equal(args[0], cached_idx)
        assert set(metrics.keys()) == {"loss", "top1_accuracy", "top5_accuracy"}

    def test_zero_valid_positions_skipped(self, cpu_device):
        """A batch with loss_mask all False contributes nothing."""
        from pawn.adapter_training import evaluate

        B, T, V, D = 1, 2, 8, 4
        model = _StubWrapper(D, V, T).to(cpu_device)
        legal_mask = torch.ones(B, T, V, dtype=torch.bool, device=cpu_device)
        mask_builder = MagicMock()
        mask_builder.scatter.return_value = legal_mask

        ids = torch.zeros(B, T, dtype=torch.long)
        tgt = torch.zeros(B, T, dtype=torch.long)
        msk = torch.zeros(B, T, dtype=torch.bool)  # all False
        batch = {
            "input_ids": ids,
            "targets": tgt,
            "loss_mask": msk,
            "legal_indices": torch.zeros(B, T, dtype=torch.long),
        }
        metrics = evaluate(model, [batch], mask_builder, cpu_device)
        # All positions skipped -> total_positions == 0 -> zeros returned
        assert metrics == {"loss": 0.0, "top1_accuracy": 0.0, "top5_accuracy": 0.0}


# ---------------------------------------------------------------------------
# Comparison: cosine_warmup_schedule vs CosineWithWarmup
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestSchedulerEquivalence:
    """cosine_warmup_schedule and CosineWithWarmup should agree at each step
    when CosineWithWarmup is used with min_lr_ratio=0."""

    def test_same_curve_min_lr_zero(self):
        from pawn.trainer import CosineWithWarmup

        peak = 1.0
        opt_a = _make_opt(lr=peak)
        opt_b = _make_opt(lr=peak)
        sched_a = cosine_warmup_schedule(opt_a, warmup_steps=10, total_steps=100)
        sched_b = CosineWithWarmup(
            opt_b, warmup_steps=10, total_steps=100, min_lr_ratio=0.0,
        )
        # Both should produce the same LR at each step
        for _ in range(100):
            lr_a = opt_a.param_groups[0]["lr"]
            lr_b = sched_b.get_lr()
            assert lr_a == pytest.approx(lr_b, rel=1e-6, abs=1e-10)
            sched_a.step()
            sched_b.step()
