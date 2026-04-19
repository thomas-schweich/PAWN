"""Tests for CosineWithWarmup LR scheduler in pawn.trainer.

The scheduler implements linear warmup from 0 -> peak_lr over warmup_steps,
then cosine decay from peak_lr -> min_lr_ratio * peak_lr from
warmup_steps -> total_steps.
"""

from __future__ import annotations

import math

import pytest
import torch

from pawn.trainer import ConstantWithWarmup, CosineWithWarmup, OneCycle, WSDSchedule


def _make_optimizer(lr: float = 1e-3, n_groups: int = 1) -> torch.optim.Optimizer:
    """Construct a tiny optimizer with n_groups parameter groups."""
    params = [
        {"params": [torch.nn.Parameter(torch.zeros(1, requires_grad=True))], "lr": lr}
        for _ in range(n_groups)
    ]
    return torch.optim.SGD(params, lr=lr)


class TestWarmupPhase:
    """Linear warmup from 0 to peak_lr over warmup_steps."""

    def test_step_zero_lr_is_zero(self):
        opt = _make_optimizer(lr=1e-3)
        sched = CosineWithWarmup(opt, warmup_steps=10, total_steps=100)
        # __init__ calls _apply_lr(0) -> warmup, step 0 / 10 = 0
        assert sched.get_lr() == pytest.approx(0.0)

    def test_lr_increases_linearly_during_warmup(self):
        peak = 1e-3
        opt = _make_optimizer(lr=peak)
        sched = CosineWithWarmup(opt, warmup_steps=10, total_steps=100)

        # After step 1: 1/10 * peak
        sched.step()
        assert sched.get_lr() == pytest.approx(0.1 * peak)
        # After step 5: 5/10 * peak
        for _ in range(4):
            sched.step()
        assert sched.get_lr() == pytest.approx(0.5 * peak)

    def test_lr_at_end_of_warmup_reaches_peak(self):
        peak = 2e-3
        opt = _make_optimizer(lr=peak)
        sched = CosineWithWarmup(opt, warmup_steps=5, total_steps=100)
        # step 5 times so _step == 5 == warmup_steps -> boundary: cosine at progress=0
        for _ in range(5):
            sched.step()
        # progress = 0, cos(0) = 1, scale = min_lr_ratio + 0.5*(1-min_lr_ratio)*(1+1)
        # = min_lr_ratio + (1 - min_lr_ratio) = 1.0
        assert sched.get_lr() == pytest.approx(peak)

    def test_zero_warmup_starts_at_peak(self):
        """warmup_steps=0 means the cosine path is taken immediately."""
        peak = 1e-3
        opt = _make_optimizer(lr=peak)
        sched = CosineWithWarmup(opt, warmup_steps=0, total_steps=100)
        # step 0: warmup condition `step < warmup_steps` == `0 < 0` is False.
        # Take cosine path: progress = 0, scale = 1.0 -> peak
        assert sched.get_lr() == pytest.approx(peak)


class TestCosineDecayPhase:
    """Cosine decay from peak_lr down to min_lr_ratio * peak_lr."""

    def test_midway_cosine_is_half_plus_min(self):
        """At progress=0.5, cos(pi/2) = 0, scale = min_lr_ratio + 0.5*(1-min_lr_ratio)."""
        peak = 1e-3
        opt = _make_optimizer(lr=peak)
        sched = CosineWithWarmup(
            opt, warmup_steps=10, total_steps=110, min_lr_ratio=0.1,
        )
        # step 10+50 = 60 steps -> halfway through 100-step cosine
        for _ in range(60):
            sched.step()
        expected = peak * (0.1 + 0.5 * 0.9)
        assert sched.get_lr() == pytest.approx(expected, rel=1e-5)

    def test_end_of_schedule_reaches_min_lr(self):
        """At progress=1.0, cos(pi) = -1, scale = min_lr_ratio."""
        peak = 1e-3
        min_lr_ratio = 0.1
        opt = _make_optimizer(lr=peak)
        sched = CosineWithWarmup(
            opt, warmup_steps=5, total_steps=50, min_lr_ratio=min_lr_ratio,
        )
        for _ in range(50):
            sched.step()
        assert sched.get_lr() == pytest.approx(peak * min_lr_ratio, rel=1e-6)

    def test_min_lr_ratio_zero_decays_to_zero(self):
        peak = 1e-3
        opt = _make_optimizer(lr=peak)
        sched = CosineWithWarmup(
            opt, warmup_steps=5, total_steps=50, min_lr_ratio=0.0,
        )
        for _ in range(50):
            sched.step()
        assert sched.get_lr() == pytest.approx(0.0, abs=1e-10)

    def test_beyond_total_clamps_at_min(self):
        """Stepping past total_steps clamps progress to 1.0."""
        peak = 1e-3
        opt = _make_optimizer(lr=peak)
        sched = CosineWithWarmup(
            opt, warmup_steps=5, total_steps=50, min_lr_ratio=0.2,
        )
        for _ in range(200):  # over-step by 4x
            sched.step()
        # Should still be clamped at min_lr_ratio * peak
        assert sched.get_lr() == pytest.approx(peak * 0.2, rel=1e-6)

    def test_cosine_is_monotone_decreasing(self):
        peak = 1e-3
        opt = _make_optimizer(lr=peak)
        sched = CosineWithWarmup(
            opt, warmup_steps=5, total_steps=105, min_lr_ratio=0.1,
        )
        # warmup
        for _ in range(5):
            sched.step()
        prev = sched.get_lr()
        for _ in range(99):
            sched.step()
            cur = sched.get_lr()
            assert cur <= prev + 1e-12
            prev = cur


class TestWarmupEqualsTotal:
    """Edge case: warmup_steps == total_steps (never reaches cosine)."""

    def test_lr_only_warms_up(self):
        peak = 1e-3
        opt = _make_optimizer(lr=peak)
        sched = CosineWithWarmup(opt, warmup_steps=20, total_steps=20)
        # Any step < 20 is warmup
        for i in range(1, 20):
            sched.step()
            assert sched.get_lr() == pytest.approx(peak * i / 20, rel=1e-6)
        # At step == warmup_steps == 20, falls into cosine with denominator max(1, 0)
        sched.step()  # _step = 20
        # progress = 0 / max(1, 0) = 0 -> scale = 1.0
        assert sched.get_lr() == pytest.approx(peak)


class TestStateDictRoundTrip:
    def test_state_dict_contains_step(self):
        opt = _make_optimizer(lr=1e-3)
        sched = CosineWithWarmup(opt, warmup_steps=10, total_steps=100)
        for _ in range(7):
            sched.step()
        state = sched.state_dict()
        assert state["step"] == 7

    def test_load_state_dict_restores_step(self):
        opt = _make_optimizer(lr=1e-3)
        sched = CosineWithWarmup(opt, warmup_steps=10, total_steps=100)
        for _ in range(15):
            sched.step()
        lr_after_15 = sched.get_lr()
        state = sched.state_dict()

        # Fresh scheduler, load from saved state
        opt2 = _make_optimizer(lr=1e-3)
        sched2 = CosineWithWarmup(opt2, warmup_steps=10, total_steps=100)
        sched2.load_state_dict(state)
        assert sched2.get_lr() == pytest.approx(lr_after_15, rel=1e-6)
        assert sched2._step == 15

    def test_load_applies_lr_immediately(self):
        opt = _make_optimizer(lr=1e-3)
        sched = CosineWithWarmup(opt, warmup_steps=10, total_steps=100)
        # Before load, we're at step 0 -> lr=0
        sched.load_state_dict({"step": 50})
        # Should have applied lr at step=50 to optimizer param groups
        assert opt.param_groups[0]["lr"] == pytest.approx(sched.get_lr())
        assert sched._step == 50


class TestMultipleParamGroups:
    def test_all_groups_get_scaled_lr(self):
        opt = _make_optimizer(lr=1e-3, n_groups=3)
        # Override one group's lr for variety
        opt.param_groups[1]["lr"] = 2e-3
        opt.param_groups[2]["lr"] = 5e-4
        sched = CosineWithWarmup(opt, warmup_steps=10, total_steps=100)
        # Step 5 -> scale = 0.5
        for _ in range(5):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.5 * 1e-3)
        assert opt.param_groups[1]["lr"] == pytest.approx(0.5 * 2e-3)
        assert opt.param_groups[2]["lr"] == pytest.approx(0.5 * 5e-4)

    def test_get_lr_returns_first_group(self):
        opt = _make_optimizer(lr=1e-3, n_groups=2)
        opt.param_groups[1]["lr"] = 9e-4
        sched = CosineWithWarmup(opt, warmup_steps=10, total_steps=100)
        # At step 0, lr = 0 for everything
        assert sched.get_lr() == opt.param_groups[0]["lr"]


class TestCosineFormulaExact:
    """Verify exact cosine formula: scale = min + 0.5*(1-min)*(1+cos(pi*t/T))."""

    def test_quarter_point(self):
        """progress=0.25, cos(pi*0.25) = sqrt(2)/2 ≈ 0.7071."""
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = CosineWithWarmup(
            opt, warmup_steps=0, total_steps=100, min_lr_ratio=0.0,
        )
        for _ in range(25):
            sched.step()
        expected = 0.5 * (1.0 + math.cos(math.pi * 0.25))
        assert sched.get_lr() == pytest.approx(expected, rel=1e-6)

    def test_three_quarter_point(self):
        """progress=0.75, cos(pi*0.75) = -sqrt(2)/2."""
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = CosineWithWarmup(
            opt, warmup_steps=0, total_steps=100, min_lr_ratio=0.0,
        )
        for _ in range(75):
            sched.step()
        expected = 0.5 * (1.0 + math.cos(math.pi * 0.75))
        assert sched.get_lr() == pytest.approx(expected, rel=1e-6)


class TestBaseLrsCaptured:
    """Base LRs are snapshotted at construction and drive later scaling."""

    def test_changing_optimizer_lr_after_construction_does_not_affect(self):
        opt = _make_optimizer(lr=1e-3)
        sched = CosineWithWarmup(opt, warmup_steps=10, total_steps=100)
        # Mutate param group after construction
        opt.param_groups[0]["lr"] = 99.0
        sched.step()
        # scheduler uses base_lrs captured at construction (1e-3)
        assert sched.get_lr() == pytest.approx(1e-3 * 1/10)


class TestWSDSchedule:
    """Warmup-Stable-Decay schedule for pretraining."""

    def test_step_zero_is_zero(self):
        opt = _make_optimizer(lr=1e-3)
        sched = WSDSchedule(opt, warmup_steps=10, decay_steps=20, total_steps=100)
        assert sched.get_lr() == pytest.approx(0.0)

    def test_warmup_linear(self):
        peak = 1e-3
        opt = _make_optimizer(lr=peak)
        sched = WSDSchedule(opt, warmup_steps=10, decay_steps=20, total_steps=100)
        sched.step()
        assert sched.get_lr() == pytest.approx(0.1 * peak)
        for _ in range(4):
            sched.step()
        assert sched.get_lr() == pytest.approx(0.5 * peak)

    def test_stable_phase_holds_peak(self):
        peak = 1e-3
        opt = _make_optimizer(lr=peak)
        sched = WSDSchedule(opt, warmup_steps=10, decay_steps=20, total_steps=100)
        for _ in range(10):
            sched.step()
        assert sched.get_lr() == pytest.approx(peak)
        for _ in range(40):
            sched.step()
        assert sched.get_lr() == pytest.approx(peak)

    def test_decay_phase_linear(self):
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = WSDSchedule(opt, warmup_steps=10, decay_steps=20, total_steps=100)
        for _ in range(80):
            sched.step()
        assert sched.get_lr() == pytest.approx(peak)
        for _ in range(10):
            sched.step()
        assert sched.get_lr() == pytest.approx(0.5)
        for _ in range(10):
            sched.step()
        assert sched.get_lr() == pytest.approx(0.0, abs=1e-10)

    def test_state_dict_roundtrip(self):
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = WSDSchedule(opt, warmup_steps=5, decay_steps=10, total_steps=50)
        for _ in range(20):
            sched.step()
        lr_before = sched.get_lr()
        state = sched.state_dict()
        sched2 = WSDSchedule(_make_optimizer(lr=peak), 5, 10, 50)
        sched2.load_state_dict(state)
        assert sched2.get_lr() == pytest.approx(lr_before)

    def test_cosine_decay_shape(self):
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = WSDSchedule(
            opt, warmup_steps=10, decay_steps=20, total_steps=100,
            decay_shape="cosine",
        )
        for _ in range(90):
            sched.step()
        # Midway through cosine decay: 0.5 * (1 + 0) = 0.5.
        assert sched.get_lr() == pytest.approx(0.5)
        for _ in range(10):
            sched.step()
        assert sched.get_lr() == pytest.approx(0.0, abs=1e-10)

    def test_unknown_decay_shape_raises(self):
        opt = _make_optimizer()
        with pytest.raises(ValueError, match="decay_shape"):
            WSDSchedule(opt, 5, 10, 50, decay_shape="exponential")


class TestConstantWithWarmup:
    """Warmup → hold peak. No decay."""

    def test_initial_lr_zero(self):
        opt = _make_optimizer(lr=1.0)
        ConstantWithWarmup(opt, warmup_steps=10)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0)

    def test_warmup_linear(self):
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = ConstantWithWarmup(opt, warmup_steps=10)
        sched.step()
        assert sched.get_lr() == pytest.approx(0.1)
        for _ in range(4):
            sched.step()
        assert sched.get_lr() == pytest.approx(0.5)

    def test_holds_peak_forever(self):
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = ConstantWithWarmup(opt, warmup_steps=10)
        for _ in range(10):
            sched.step()
        for _ in range(10_000):
            sched.step()
        assert sched.get_lr() == pytest.approx(peak)

    def test_state_dict_roundtrip(self):
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = ConstantWithWarmup(opt, warmup_steps=10)
        for _ in range(25):
            sched.step()
        before = sched.get_lr()
        state = sched.state_dict()
        sched2 = ConstantWithWarmup(_make_optimizer(lr=peak), 10)
        sched2.load_state_dict(state)
        assert sched2.get_lr() == pytest.approx(before)


class TestOneCycle:
    """Smith one-cycle: ramp to peak → cosine decay."""

    def test_initial_lr_is_low(self):
        opt = _make_optimizer(lr=1.0)
        OneCycle(opt, peak_step=30, total_steps=100)
        # Initial = 1/25 = 0.04
        assert opt.param_groups[0]["lr"] == pytest.approx(0.04)

    def test_reaches_peak_at_peak_step(self):
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = OneCycle(opt, peak_step=30, total_steps=100)
        for _ in range(30):
            sched.step()
        assert sched.get_lr() == pytest.approx(peak)

    def test_decays_to_final_floor(self):
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = OneCycle(
            opt, peak_step=30, total_steps=100, final_div=1e4,
        )
        for _ in range(100):
            sched.step()
        assert sched.get_lr() == pytest.approx(1e-4, abs=1e-6)

    def test_peak_step_must_be_positive(self):
        opt = _make_optimizer()
        with pytest.raises(ValueError, match="peak_step > 0"):
            OneCycle(opt, peak_step=0, total_steps=100)

    def test_peak_step_must_be_less_than_total(self):
        opt = _make_optimizer()
        with pytest.raises(ValueError, match="peak_step < total_steps"):
            OneCycle(opt, peak_step=100, total_steps=100)

    def test_state_dict_roundtrip(self):
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = OneCycle(opt, peak_step=30, total_steps=100)
        for _ in range(50):
            sched.step()
        before = sched.get_lr()
        state = sched.state_dict()
        sched2 = OneCycle(_make_optimizer(lr=peak), 30, 100)
        sched2.load_state_dict(state)
        assert sched2.get_lr() == pytest.approx(before)
