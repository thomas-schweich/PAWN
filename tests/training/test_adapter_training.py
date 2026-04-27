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
    build_scheduler,
    compute_adapter_loss,
    constant_warmup_schedule,
    cosine_warmup_schedule,
    load_backbone,
    one_cycle_schedule,
    parse_layers,
    sparse_forward,
    wsd_schedule,
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
# wsd_schedule (warmup-stable-decay)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestWSDSchedule:
    def test_returns_lambda_lr(self):
        opt = _make_opt()
        sched = wsd_schedule(opt, warmup_steps=10, decay_steps=20, total_steps=100)
        assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)

    def test_initial_lr_is_zero(self):
        opt = _make_opt(lr=1e-3)
        wsd_schedule(opt, warmup_steps=10, decay_steps=20, total_steps=100)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0)

    def test_warmup_ramps_linearly(self):
        peak = 1e-3
        opt = _make_opt(lr=peak)
        sched = wsd_schedule(opt, warmup_steps=10, decay_steps=20, total_steps=100)
        sched.step()  # step 1
        assert opt.param_groups[0]["lr"] == pytest.approx(0.1 * peak)
        for _ in range(4):
            sched.step()  # step 5
        assert opt.param_groups[0]["lr"] == pytest.approx(0.5 * peak)

    def test_stable_phase_holds_peak(self):
        peak = 1e-3
        opt = _make_opt(lr=peak)
        sched = wsd_schedule(opt, warmup_steps=10, decay_steps=20, total_steps=100)
        # Walk through warmup → land in stable phase at multiple points.
        for _ in range(10):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(peak)
        for _ in range(40):  # step 50 — still in stable (10..80)
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(peak)
        for _ in range(29):  # step 79 — still in stable
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(peak)

    def test_decay_phase_falls_linearly(self):
        peak = 1.0
        opt = _make_opt(lr=peak)
        sched = wsd_schedule(opt, warmup_steps=10, decay_steps=20, total_steps=100)
        # Advance to the start of decay (step 80).
        for _ in range(80):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(peak)
        # Midway through decay (step 90): LR = 0.5 * peak.
        for _ in range(10):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.5)
        # End of schedule (step 100): LR = 0.
        for _ in range(10):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-10)

    def test_min_lr_ratio_floor(self):
        opt = _make_opt(lr=1.0)
        sched = wsd_schedule(
            opt, warmup_steps=5, decay_steps=10, total_steps=50,
            min_lr_ratio=0.1,
        )
        for _ in range(50):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.1)

    def test_decay_overflow_clamps_stable(self):
        # warmup + decay > total: stable phase clipped to zero.
        peak = 1.0
        opt = _make_opt(lr=peak)
        sched = wsd_schedule(opt, warmup_steps=10, decay_steps=100, total_steps=50)
        for _ in range(10):
            sched.step()  # end of warmup → immediately into decay
        # Decay window = 50 - 10 = 40 (clipped by stable_end = max(warmup, total-decay))
        # stable_end = max(10, 50-100) = 10. So LR at step 10 is already at start of decay: 1.0.
        assert opt.param_groups[0]["lr"] == pytest.approx(peak)
        for _ in range(20):  # step 30 → halfway through a 40-step decay
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# build_scheduler dispatch
# ---------------------------------------------------------------------------


class TestBuildScheduler:
    def test_cosine_dispatch(self):
        opt = _make_opt()
        sched = build_scheduler(opt, 10, 100, schedule="cosine")
        assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)

    def test_wsd_dispatch(self):
        opt = _make_opt()
        sched = build_scheduler(
            opt, 10, 100, schedule="wsd", decay_steps=20,
        )
        assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)

    def test_wsd_requires_decay_steps(self):
        opt = _make_opt()
        with pytest.raises(ValueError, match="decay_steps"):
            build_scheduler(opt, 10, 100, schedule="wsd")

    def test_wsd_cosine_decay_shape(self):
        opt = _make_opt()
        sched = build_scheduler(
            opt, 10, 100, schedule="wsd",
            decay_steps=20, wsd_decay_shape="cosine",
        )
        assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)

    def test_constant_dispatch(self):
        opt = _make_opt()
        sched = build_scheduler(opt, 10, 100, schedule="constant")
        assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)

    def test_one_cycle_dispatch(self):
        opt = _make_opt()
        sched = build_scheduler(opt, 30, 100, schedule="one_cycle")
        assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)

    def test_unknown_schedule_raises(self):
        opt = _make_opt()
        with pytest.raises(ValueError, match="Unknown lr_schedule"):
            build_scheduler(opt, 10, 100, schedule="step")


# ---------------------------------------------------------------------------
# constant_warmup_schedule
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestConstantWarmupSchedule:
    def test_initial_lr_is_zero(self):
        opt = _make_opt(lr=1.0)
        constant_warmup_schedule(opt, warmup_steps=10)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0)

    def test_warmup_linear(self):
        peak = 1.0
        opt = _make_opt(lr=peak)
        sched = constant_warmup_schedule(opt, warmup_steps=10)
        sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.1)
        for _ in range(4):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.5)

    def test_holds_peak_after_warmup(self):
        peak = 1.0
        opt = _make_opt(lr=peak)
        sched = constant_warmup_schedule(opt, warmup_steps=10)
        for _ in range(10):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(peak)
        for _ in range(5000):
            sched.step()
        # still at peak, no decay
        assert opt.param_groups[0]["lr"] == pytest.approx(peak)

    def test_zero_warmup_starts_at_peak(self):
        peak = 1.0
        opt = _make_opt(lr=peak)
        constant_warmup_schedule(opt, warmup_steps=0)
        assert opt.param_groups[0]["lr"] == pytest.approx(peak)


# ---------------------------------------------------------------------------
# wsd_schedule with cosine decay
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestWSDCosineDecay:
    def test_cosine_decay_is_smooth(self):
        """Cosine-shaped decay is strictly monotonic and hits 0 at the end."""
        peak = 1.0
        opt = _make_opt(lr=peak)
        sched = wsd_schedule(
            opt, warmup_steps=10, decay_steps=20, total_steps=100,
            decay_shape="cosine",
        )
        for _ in range(80):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(peak)
        # midway through decay: cosine at π/2 → 0.5 * (1 + 0) = 0.5
        for _ in range(10):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.5)
        for _ in range(10):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-10)

    def test_unknown_decay_shape_raises(self):
        opt = _make_opt()
        with pytest.raises(ValueError, match="decay_shape"):
            wsd_schedule(opt, 10, 20, 100, decay_shape="exponential")


# ---------------------------------------------------------------------------
# one_cycle_schedule (Smith)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestOneCycleSchedule:
    def test_initial_lr_is_low(self):
        peak = 1.0
        opt = _make_opt(lr=peak)
        one_cycle_schedule(opt, peak_step=30, total_steps=100)
        # Step 0 → initial_frac = 1/25 = 0.04
        assert opt.param_groups[0]["lr"] == pytest.approx(0.04)

    def test_reaches_peak_at_peak_step(self):
        peak = 1.0
        opt = _make_opt(lr=peak)
        sched = one_cycle_schedule(opt, peak_step=30, total_steps=100)
        for _ in range(30):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(peak)

    def test_decays_to_final_floor(self):
        peak = 1.0
        opt = _make_opt(lr=peak)
        sched = one_cycle_schedule(
            opt, peak_step=30, total_steps=100,
            final_div=1e4,
        )
        for _ in range(100):
            sched.step()
        # End of schedule → final_frac = 1/10000
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-4, abs=1e-6)

    def test_peak_step_must_be_positive(self):
        opt = _make_opt()
        with pytest.raises(ValueError, match="peak_step > 0"):
            one_cycle_schedule(opt, peak_step=0, total_steps=100)

    def test_peak_step_must_be_less_than_total(self):
        opt = _make_opt()
        with pytest.raises(ValueError, match="peak_step < total_steps"):
            one_cycle_schedule(opt, peak_step=100, total_steps=100)


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
        out, legal = sparse_forward(
            model, ids, msk, legal_mask, None, cpu_device
        )
        assert out.shape == (B * T, V)
        assert legal.shape == (B * T, V)

    def test_masks_illegal_with_neg_inf(self, cpu_device):
        B, T, V, D = 1, 2, 4, 2
        model = _StubWrapper(D, V, T).to(cpu_device)
        ids = torch.zeros(B, T, dtype=torch.long, device=cpu_device)
        msk = torch.ones(B, T, dtype=torch.bool, device=cpu_device)
        legal_mask = torch.zeros(B, T, V, dtype=torch.bool, device=cpu_device)
        # Allow only index 1
        legal_mask[..., 1] = True
        out, _ = sparse_forward(
            model, ids, msk, legal_mask, None, cpu_device
        )
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
        out, _ = sparse_forward(
            model, ids, msk, legal_mask, None, cpu_device
        )
        assert out.dtype == torch.float32

    def test_respects_loss_mask(self, cpu_device):
        """Only positions where msk is True are included in output rows."""
        B, T, V, D = 1, 4, 2, 2
        model = _StubWrapper(D, V, T).to(cpu_device)
        ids = torch.zeros(B, T, dtype=torch.long, device=cpu_device)
        msk = torch.tensor([[True, False, True, False]], device=cpu_device)
        legal_mask = torch.ones(B, T, V, dtype=torch.bool, device=cpu_device)
        out, legal = sparse_forward(
            model, ids, msk, legal_mask, None, cpu_device
        )
        # 2 positions pass the mask
        assert out.shape == (2, V)
        assert legal.shape == (2, V)

    def test_disable_legal_mask_keeps_illegal_logits_finite(self, cpu_device):
        """With apply_legal_mask=False, illegal logits are NOT filled with -inf."""
        B, T, V, D = 1, 2, 4, 2
        model = _StubWrapper(D, V, T).to(cpu_device)
        ids = torch.zeros(B, T, dtype=torch.long, device=cpu_device)
        msk = torch.ones(B, T, dtype=torch.bool, device=cpu_device)
        legal_mask = torch.zeros(B, T, V, dtype=torch.bool, device=cpu_device)
        legal_mask[..., 1] = True
        out, legal = sparse_forward(
            model, ids, msk, legal_mask, None, cpu_device,
            apply_legal_mask=False,
        )
        # No -inf anywhere: the model is allowed to assign probability
        # to illegal moves and is expected to learn the distinction.
        assert torch.isfinite(out).all()
        # Legality tensor still reports which positions were legal.
        assert (legal[:, 1]).all()
        assert (~legal[:, 0]).all() and (~legal[:, 2]).all() and (~legal[:, 3]).all()


class TestComputeAdapterLoss:
    def _setup(self, cpu_device):
        # Two positions, 4 vocab, target = legal index 1.
        # Logits put half the mass on an illegal index to exercise the
        # penalty term.
        V = 4
        logits = torch.tensor(
            [[0.0, 0.0, 10.0, 0.0], [0.0, 10.0, 0.0, 0.0]],
            device=cpu_device,
        )
        targets = torch.tensor([1, 1], device=cpu_device)
        legal = torch.zeros(2, V, dtype=torch.bool, device=cpu_device)
        legal[:, 1] = True  # only index 1 is legal
        return logits, targets, legal

    def test_penalty_zero_equals_plain_cross_entropy(self, cpu_device):
        logits, targets, legal = self._setup(cpu_device)
        loss = compute_adapter_loss(
            logits, targets, legal, illegal_penalty=0.0
        )
        expected = nn.functional.cross_entropy(logits, targets)
        assert torch.allclose(loss, expected)

    def test_penalty_increases_loss_when_illegal_mass_present(self, cpu_device):
        logits, targets, legal = self._setup(cpu_device)
        base = compute_adapter_loss(
            logits, targets, legal, illegal_penalty=0.0
        )
        penalized = compute_adapter_loss(
            logits, targets, legal, illegal_penalty=1.0
        )
        assert penalized > base
        # The penalty term equals lambda * mean(sum_probs_over_illegal).
        probs = torch.softmax(logits, dim=-1)
        illegal_mass = (probs * (~legal).float()).sum(dim=-1).mean()
        assert torch.allclose(penalized - base, illegal_mass)

    def test_penalty_scales_linearly_with_lambda(self, cpu_device):
        logits, targets, legal = self._setup(cpu_device)
        base = compute_adapter_loss(logits, targets, legal, illegal_penalty=0.0)
        half = compute_adapter_loss(logits, targets, legal, illegal_penalty=0.5)
        one = compute_adapter_loss(logits, targets, legal, illegal_penalty=1.0)
        # Linear in lambda: (one - base) ≈ 2 * (half - base).
        assert torch.allclose(one - base, 2 * (half - base))

    def test_penalty_no_op_when_logits_are_masked_to_neg_inf(self, cpu_device):
        """If the mask was already applied, illegal prob mass is 0, so the penalty
        term must contribute nothing regardless of lambda."""
        logits, targets, legal = self._setup(cpu_device)
        logits = logits.clone()
        logits.masked_fill_(~legal, float("-inf"))
        base = compute_adapter_loss(logits, targets, legal, illegal_penalty=0.0)
        penalized = compute_adapter_loss(
            logits, targets, legal, illegal_penalty=10.0
        )
        assert torch.allclose(base, penalized)


class TestAdapterConfigLegalityValidator:
    def test_defaults(self):
        from pawn.run_config import AdapterConfig

        cfg = AdapterConfig(strategy="lora", local_checkpoints=True)
        assert cfg.disable_legal_mask is False
        assert cfg.illegal_penalty == 0.0

    def test_penalty_without_disable_is_rejected(self):
        from pydantic import ValidationError

        from pawn.run_config import AdapterConfig

        with pytest.raises(ValidationError, match="illegal_penalty"):
            AdapterConfig(
                strategy="lora",
                local_checkpoints=True,
                illegal_penalty=0.5,
            )

    def test_negative_penalty_is_rejected(self):
        from pydantic import ValidationError

        from pawn.run_config import AdapterConfig

        with pytest.raises(ValidationError, match="illegal_penalty"):
            AdapterConfig(
                strategy="lora",
                local_checkpoints=True,
                disable_legal_mask=True,
                illegal_penalty=-0.1,
            )

    def test_disable_without_penalty_ok(self):
        from pawn.run_config import AdapterConfig

        cfg = AdapterConfig(
            strategy="lora",
            local_checkpoints=True,
            disable_legal_mask=True,
        )
        assert cfg.disable_legal_mask is True
        assert cfg.illegal_penalty == 0.0

    def test_disable_with_positive_penalty_ok(self):
        from pawn.run_config import AdapterConfig

        cfg = AdapterConfig(
            strategy="lora",
            local_checkpoints=True,
            disable_legal_mask=True,
            illegal_penalty=2.0,
        )
        assert cfg.disable_legal_mask is True
        assert cfg.illegal_penalty == 2.0


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
        assert set(metrics.keys()) == {
            "loss",
            "top1_accuracy",
            "top5_accuracy",
            "illegal_pred_rate",
            "illegal_prob_mass",
        }
        assert 0.0 <= metrics["top1_accuracy"] <= 1.0
        assert 0.0 <= metrics["top5_accuracy"] <= 1.0
        # All moves are legal in this fixture → illegal metrics are 0.
        assert metrics["illegal_pred_rate"] == 0.0
        assert metrics["illegal_prob_mass"] == 0.0

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
        assert {"loss", "top1_accuracy", "top5_accuracy"} <= set(metrics.keys())

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
        assert metrics == {
            "loss": 0.0,
            "top1_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "illegal_pred_rate": 0.0,
            "illegal_prob_mass": 0.0,
        }

    def test_apply_legal_mask_false_tracks_illegal_metrics(self, cpu_device):
        """When masking is off and the model argmaxes onto illegal tokens,
        the metrics must reflect that — not silently stay zero.

        This is the one path the default evaluate fixture doesn't cover:
        previously illegal_pred_rate and illegal_prob_mass were 0 because
        every logit went through the -inf hard mask and thus every argmax
        was legal by construction.
        """
        from pawn.adapter_training import evaluate

        # Hand-built wrapper whose project_head returns logits we fully
        # control, so we can force the argmax onto an illegal index.
        # V must be >= 5 since evaluate() computes top-5 accuracy.
        B, T, V = 2, 1, 8

        class _FixedLogits(nn.Module):
            def __init__(self, logits: torch.Tensor):
                super().__init__()
                # (B, T, V): we just forward this every call.
                self._logits = logits

            def forward_hidden(self, ids: torch.Tensor) -> torch.Tensor:
                # dummy, never read
                return torch.zeros(
                    ids.shape[0], ids.shape[1], 1, device=ids.device
                )

            def project_head(self, x: torch.Tensor) -> torch.Tensor:
                # x is (N, 1) — ignore, just fan out the precomputed
                # logits in the same flat shape evaluate() expects.
                return self._logits.reshape(-1, V)

        # Row 0: peak at illegal idx 2. Row 1: peak at legal idx 1.
        logits = torch.zeros(B, T, V, device=cpu_device)
        logits[0, 0, 2] = 5.0
        logits[1, 0, 1] = 10.0
        model = _FixedLogits(logits)

        # Legal mask allows only index 1.
        legal_mask = torch.zeros(
            B, T, V, dtype=torch.bool, device=cpu_device
        )
        legal_mask[..., 1] = True

        mask_builder = MagicMock()
        mask_builder.scatter.return_value = legal_mask

        # Target is the one legal index; the model's argmax on row 0
        # lands on illegal index 2, on row 1 it lands on legal index 1.
        batch = {
            "input_ids": torch.zeros(B, T, dtype=torch.long),
            "targets": torch.tensor([[1], [1]], device=cpu_device),
            "loss_mask": torch.ones(B, T, dtype=torch.bool),
            "legal_indices": torch.zeros(B, T, dtype=torch.long),
        }

        metrics = evaluate(
            model, [batch], mask_builder, cpu_device,
            apply_legal_mask=False,
        )
        # One of two positions argmaxed onto an illegal index.
        assert metrics["illegal_pred_rate"] == pytest.approx(0.5)
        # Illegal prob mass is positive (row 0 puts >99% on idx 2; row 1
        # puts ~all on idx 1, which is legal). Expected: (softmax row0
        # mass on idx 0/2/3) + (row 1 mass on idx 0/2/3), averaged over
        # 2 positions.
        probs = torch.softmax(logits.reshape(-1, V), dim=-1)
        expected_mass = (
            probs.masked_fill(legal_mask.reshape(-1, V), 0.0)
            .sum(dim=-1)
            .mean()
            .item()
        )
        assert metrics["illegal_prob_mass"] == pytest.approx(
            expected_mass, rel=1e-5
        )
        assert metrics["illegal_prob_mass"] > 0.0

    def test_apply_legal_mask_true_skips_illegal_softmax(self, cpu_device):
        """With hard masking on, a row with zero legal moves used to NaN
        the softmax-based illegal_prob_mass metric. The guard short-
        circuits and returns analytically-zero metrics instead."""
        from pawn.adapter_training import evaluate

        # V must be >= 5 for the top-5 step in evaluate().
        B, T, V, D = 1, 1, 8, 2
        model = _StubWrapper(D, V, T).to(cpu_device)

        # Pathological: the single position has *no* legal moves. The
        # invariant "every loss-mask position has at least one legal
        # move" is normally enforced upstream, but the metric block
        # must never NaN if that invariant ever cracks.
        legal_mask = torch.zeros(
            B, T, V, dtype=torch.bool, device=cpu_device
        )
        mask_builder = MagicMock()
        mask_builder.scatter.return_value = legal_mask

        batch = {
            "input_ids": torch.zeros(B, T, dtype=torch.long),
            "targets": torch.zeros(B, T, dtype=torch.long),
            "loss_mask": torch.ones(B, T, dtype=torch.bool),
            "legal_indices": torch.zeros(B, T, dtype=torch.long),
        }

        metrics = evaluate(
            model, [batch], mask_builder, cpu_device,
            apply_legal_mask=True,
        )
        # Must be exactly zero, not NaN.
        assert metrics["illegal_pred_rate"] == 0.0
        assert metrics["illegal_prob_mass"] == 0.0
        assert not math.isnan(metrics["illegal_pred_rate"])
        assert not math.isnan(metrics["illegal_prob_mass"])


class TestBuildConfigJsonLegality:
    """build_config_json must surface disable_legal_mask + illegal_penalty."""

    def test_round_trip_default_values(self):
        import argparse

        from pawn.adapter_training import build_config_json

        args = argparse.Namespace(
            strategy="lora",
            checkpoint="thomas-schweich/pawn-base",
            pgn="thomas-schweich/pawn-lichess-full",
            elo_min=None, elo_max=None, max_games=None, val_games=0,
            total_steps=None, eval_interval=None, epochs=1,
            batch_size=1, lr=0.0, warmup_frac=0.0, weight_decay=0.0,
            max_grad_norm=0.0, amp_dtype="none", patience=None,
            adapter_layers=None,
            bottleneck_dim=None, no_adapt_attn=False, no_adapt_ffn=False,
            lora_rank=4, lora_targets="qv", lora_ffn=False,
            density=None, sparse_targets=None, sparse_ffn=False,
            use_output_film=False,
            rosa_mode=None, rosa_warmup_steps=0, mask_samples=0, grad_alpha=1,
            d_model=None, n_layers=None, n_heads=None,
            unfreeze_layers=None,
            disable_legal_mask=False, illegal_penalty=0.0,
        )
        cfg = build_config_json(args, param_count=0)
        assert cfg["disable_legal_mask"] is False
        assert cfg["illegal_penalty"] == 0.0

    def test_round_trip_nondefault_values(self):
        import argparse

        from pawn.adapter_training import build_config_json

        args = argparse.Namespace(
            strategy="lora",
            checkpoint="thomas-schweich/pawn-base",
            pgn="thomas-schweich/pawn-lichess-full",
            elo_min=None, elo_max=None, max_games=None, val_games=0,
            total_steps=None, eval_interval=None, epochs=1,
            batch_size=1, lr=0.0, warmup_frac=0.0, weight_decay=0.0,
            max_grad_norm=0.0, amp_dtype="none", patience=None,
            adapter_layers=None,
            bottleneck_dim=None, no_adapt_attn=False, no_adapt_ffn=False,
            lora_rank=4, lora_targets="qv", lora_ffn=False,
            density=None, sparse_targets=None, sparse_ffn=False,
            use_output_film=False,
            rosa_mode=None, rosa_warmup_steps=0, mask_samples=0, grad_alpha=1,
            d_model=None, n_layers=None, n_heads=None,
            unfreeze_layers=None,
            disable_legal_mask=True, illegal_penalty=0.75,
        )
        cfg = build_config_json(args, param_count=0)
        assert cfg["disable_legal_mask"] is True
        assert cfg["illegal_penalty"] == 0.75


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


# ---------------------------------------------------------------------------
# write_schedule_health
# ---------------------------------------------------------------------------


class TestWriteScheduleHealth:
    @pytest.mark.unit
    def test_completed_run_writes_clean_health(self, tmp_path, capsys):
        from pawn.adapter_training import write_schedule_health

        h = write_schedule_health(
            tmp_path,
            schedule="cosine",
            planned_total_steps=1000,
            actual_total_steps=1000,
            lr_peak=3e-4,
            actual_final_lr=0.0,
            reason_for_stop="completed",
        )
        assert h["completion_ratio"] == 1.0
        assert h["should_reach_zero"] is True
        assert (tmp_path / "schedule_health.json").exists()
        # No banner.
        assert "WARNING: schedule did not run to completion" not in (
            capsys.readouterr().out
        )

    @pytest.mark.unit
    def test_step_mismatch_with_completed_reason_warns(self, tmp_path, capsys):
        """The combination ``actual != planned`` AND
        ``reason_for_stop == "completed"`` is the structural-bug signal —
        with cache-first it should never happen, and we want a loud
        red banner if it does."""
        from pawn.adapter_training import write_schedule_health

        write_schedule_health(
            tmp_path,
            schedule="cosine",
            planned_total_steps=1000,
            actual_total_steps=950,
            lr_peak=3e-4,
            actual_final_lr=2e-5,
            reason_for_stop="completed",
        )
        out = capsys.readouterr().out
        assert "WARNING: schedule did not run to completion" in out

    @pytest.mark.unit
    def test_sigterm_does_not_warn(self, tmp_path, capsys):
        """SIGTERM is a legitimate early exit; no banner."""
        from pawn.adapter_training import write_schedule_health

        write_schedule_health(
            tmp_path,
            schedule="cosine",
            planned_total_steps=1000,
            actual_total_steps=500,
            lr_peak=3e-4,
            actual_final_lr=1.5e-4,
            reason_for_stop="sigterm",
        )
        out = capsys.readouterr().out
        assert "WARNING: schedule did not run to completion" not in out

    @pytest.mark.unit
    def test_constant_schedule_does_not_warn(self, tmp_path, capsys):
        """``constant`` does not decay to 0; mismatch is normal."""
        from pawn.adapter_training import write_schedule_health

        h = write_schedule_health(
            tmp_path,
            schedule="constant",
            planned_total_steps=1000,
            actual_total_steps=900,
            lr_peak=3e-4,
            actual_final_lr=3e-4,
            reason_for_stop="completed",
        )
        assert h["should_reach_zero"] is False
        assert "WARNING" not in capsys.readouterr().out



# ---------------------------------------------------------------------------
# resume_state
# ---------------------------------------------------------------------------


class TestResumeState:
    @pytest.mark.unit
    def test_fresh_run(self):
        from pawn.adapter_training import resume_state

        assert resume_state(0, 1000) == (0, 0)

    @pytest.mark.unit
    def test_exact_epoch_boundary(self):
        """End-of-epoch save: ``global_step`` lands on a clean
        boundary, so the next loop iteration starts the next epoch
        with no skip."""
        from pawn.adapter_training import resume_state

        assert resume_state(1000, 1000) == (1, 0)
        assert resume_state(2000, 1000) == (2, 0)

    @pytest.mark.unit
    def test_mid_epoch(self):
        """Mid-epoch save (e.g. SIGTERM): re-enter the in-progress
        epoch and skip the consumed prefix."""
        from pawn.adapter_training import resume_state

        assert resume_state(1500, 1000) == (1, 500)
        assert resume_state(2500, 1000) == (2, 500)

    @pytest.mark.unit
    def test_past_end(self):
        """Resume on a checkpoint already past ``epochs * steps_per_epoch``
        is the no-op case — start_epoch is computed honestly; the
        trainer's epoch loop just iterates an empty range."""
        from pawn.adapter_training import resume_state

        # epochs=3 × spe=1000 = 3000, save at 3000 → start=3.
        assert resume_state(3000, 1000) == (3, 0)
        # Can technically go further (extension scenario).
        assert resume_state(4500, 1000) == (4, 500)

    @pytest.mark.unit
    def test_validates_inputs(self):
        from pawn.adapter_training import resume_state

        with pytest.raises(ValueError, match="steps_per_epoch"):
            resume_state(100, 0)
        with pytest.raises(ValueError, match="global_step"):
            resume_state(-1, 100)
