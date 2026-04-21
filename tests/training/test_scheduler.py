"""Tests for CosineWithWarmup LR scheduler in pawn.trainer.

The scheduler implements linear warmup from 0 -> peak_lr over warmup_steps,
then cosine decay from peak_lr -> min_lr_ratio * peak_lr from
warmup_steps -> total_steps.
"""

from __future__ import annotations

import math

import pytest
import torch

from pawn.trainer import (
    ConstantWithWarmup,
    CosineWithWarmup,
    InfiniteSchedule,
    OneCycle,
    WSDSchedule,
)


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


class TestInfiniteSchedule:
    """Warmup → cosine cooldown to stable → flat → final decay."""

    def test_step_zero_is_zero(self):
        opt = _make_optimizer(lr=1.0)
        InfiniteSchedule(
            opt, warmup_steps=10, cooldown_steps=20,
            decay_steps=30, total_steps=200, stable_lr_ratio=0.1,
        )
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0)

    def test_warmup_linear(self):
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = InfiniteSchedule(
            opt, warmup_steps=10, cooldown_steps=20,
            decay_steps=30, total_steps=200, stable_lr_ratio=0.1,
        )
        sched.step()
        assert sched.get_lr() == pytest.approx(0.1)
        for _ in range(9):
            sched.step()
        assert sched.get_lr() == pytest.approx(peak)

    def test_cooldown_is_cosine_to_stable(self):
        peak = 1.0
        stable = 0.1
        opt = _make_optimizer(lr=peak)
        sched = InfiniteSchedule(
            opt, warmup_steps=10, cooldown_steps=20,
            decay_steps=30, total_steps=200, stable_lr_ratio=stable,
        )
        # After warmup (step 10): at peak.
        for _ in range(10):
            sched.step()
        assert sched.get_lr() == pytest.approx(peak)
        # Midway through cooldown (step 10 + 10 = 20): cos(pi/2) = 0,
        # scale = stable + (1-stable) * 0.5 = 0.1 + 0.45 = 0.55.
        for _ in range(10):
            sched.step()
        assert sched.get_lr() == pytest.approx(0.55, rel=1e-6)
        # End of cooldown (step 30): at stable.
        for _ in range(10):
            sched.step()
        assert sched.get_lr() == pytest.approx(stable, rel=1e-6)

    def test_stable_phase_holds_flat(self):
        peak = 1.0
        stable = 0.1
        opt = _make_optimizer(lr=peak)
        sched = InfiniteSchedule(
            opt, warmup_steps=10, cooldown_steps=20,
            decay_steps=30, total_steps=200, stable_lr_ratio=stable,
        )
        # Warmup + cooldown = 30, final_decay starts at 200 - 30 = 170.
        # Stable span: [30, 170).
        for _ in range(30):
            sched.step()
        # Sample several steps across the stable plateau.
        for s in (35, 80, 100, 150, 169):
            while sched._step < s:
                sched.step()
            assert sched.get_lr() == pytest.approx(stable, rel=1e-6)

    def test_final_decay_cosine_to_zero(self):
        peak = 1.0
        stable = 0.1
        opt = _make_optimizer(lr=peak)
        sched = InfiniteSchedule(
            opt, warmup_steps=10, cooldown_steps=20,
            decay_steps=30, total_steps=200, stable_lr_ratio=stable,
            final_decay_shape="cosine",
        )
        # Step to final_decay_start = 170.
        for _ in range(170):
            sched.step()
        assert sched.get_lr() == pytest.approx(stable, rel=1e-6)
        # Midway through final decay (step 185): cos(pi/2) = 0,
        # scale = 0 + (stable - 0) * 0.5 = 0.05.
        for _ in range(15):
            sched.step()
        assert sched.get_lr() == pytest.approx(0.05, rel=1e-6)
        # End (step 200): 0.
        for _ in range(15):
            sched.step()
        assert sched.get_lr() == pytest.approx(0.0, abs=1e-10)

    def test_final_decay_linear_shape(self):
        peak = 1.0
        stable = 0.2
        opt = _make_optimizer(lr=peak)
        sched = InfiniteSchedule(
            opt, warmup_steps=5, cooldown_steps=5,
            decay_steps=10, total_steps=50, stable_lr_ratio=stable,
            final_decay_shape="linear",
        )
        # final_decay_start = 40.
        for _ in range(40):
            sched.step()
        assert sched.get_lr() == pytest.approx(stable, rel=1e-6)
        # Halfway through linear decay (step 45): stable * 0.5.
        for _ in range(5):
            sched.step()
        assert sched.get_lr() == pytest.approx(stable * 0.5, rel=1e-6)
        # End (step 50): 0.
        for _ in range(5):
            sched.step()
        assert sched.get_lr() == pytest.approx(0.0, abs=1e-10)

    def test_resume_extension_keeps_stable_lr(self):
        """Key invariant: stable-phase LR is independent of total_steps.

        If you checkpoint mid-stable and resume with a larger
        total_steps, the LR at the resumption step must be unchanged.
        """
        peak = 1.0
        stable = 0.15
        opt_a = _make_optimizer(lr=peak)
        sched_a = InfiniteSchedule(
            opt_a, warmup_steps=10, cooldown_steps=20,
            decay_steps=30, total_steps=200, stable_lr_ratio=stable,
        )
        for _ in range(100):  # mid-stable (stable spans [30, 170))
            sched_a.step()
        lr_a = sched_a.get_lr()
        assert lr_a == pytest.approx(stable)

        # Same settings, but total_steps doubled — resume at step 100.
        opt_b = _make_optimizer(lr=peak)
        sched_b = InfiniteSchedule(
            opt_b, warmup_steps=10, cooldown_steps=20,
            decay_steps=30, total_steps=400, stable_lr_ratio=stable,
        )
        sched_b.load_state_dict({"step": 100})
        assert sched_b.get_lr() == pytest.approx(lr_a, rel=1e-10)

    def test_state_dict_roundtrip(self):
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = InfiniteSchedule(
            opt, warmup_steps=10, cooldown_steps=20,
            decay_steps=30, total_steps=200, stable_lr_ratio=0.1,
        )
        for _ in range(75):
            sched.step()
        before = sched.get_lr()
        state = sched.state_dict()
        sched2 = InfiniteSchedule(
            _make_optimizer(lr=peak), warmup_steps=10, cooldown_steps=20,
            decay_steps=30, total_steps=200, stable_lr_ratio=0.1,
        )
        sched2.load_state_dict(state)
        assert sched2.get_lr() == pytest.approx(before)
        assert sched2._step == 75

    def test_beyond_total_clamps_at_min(self):
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = InfiniteSchedule(
            opt, warmup_steps=5, cooldown_steps=5,
            decay_steps=10, total_steps=50, stable_lr_ratio=0.2,
            min_lr_ratio=0.05,
        )
        for _ in range(500):  # over-step by 10x
            sched.step()
        assert sched.get_lr() == pytest.approx(peak * 0.05, rel=1e-6)

    def test_cooldown_is_monotone_decreasing(self):
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = InfiniteSchedule(
            opt, warmup_steps=10, cooldown_steps=40,
            decay_steps=20, total_steps=200, stable_lr_ratio=0.1,
        )
        # Advance through warmup.
        for _ in range(10):
            sched.step()
        prev = sched.get_lr()
        for _ in range(40):
            sched.step()
            cur = sched.get_lr()
            assert cur <= prev + 1e-12
            prev = cur

    def test_unknown_final_decay_shape_raises(self):
        opt = _make_optimizer()
        with pytest.raises(ValueError, match="final_decay_shape"):
            InfiniteSchedule(
                opt, warmup_steps=5, cooldown_steps=5,
                decay_steps=10, total_steps=50, stable_lr_ratio=0.1,
                final_decay_shape="exponential",
            )

    def test_stable_lr_ratio_out_of_range_raises(self):
        opt = _make_optimizer()
        with pytest.raises(ValueError, match="stable_lr_ratio"):
            InfiniteSchedule(
                opt, warmup_steps=5, cooldown_steps=5,
                decay_steps=10, total_steps=50, stable_lr_ratio=1.5,
            )

    def test_min_lr_ratio_above_stable_raises(self):
        opt = _make_optimizer()
        with pytest.raises(ValueError, match="min_lr_ratio"):
            InfiniteSchedule(
                opt, warmup_steps=5, cooldown_steps=5,
                decay_steps=10, total_steps=50, stable_lr_ratio=0.1,
                min_lr_ratio=0.2,
            )

    def test_negative_cooldown_steps_raises(self):
        opt = _make_optimizer()
        with pytest.raises(ValueError, match="non-negative"):
            InfiniteSchedule(
                opt, warmup_steps=5, cooldown_steps=-1,
                decay_steps=10, total_steps=50, stable_lr_ratio=0.1,
            )

    def test_warmup_steps_zero_starts_at_peak(self):
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = InfiniteSchedule(
            opt, warmup_steps=0, cooldown_steps=10,
            decay_steps=10, total_steps=50, stable_lr_ratio=0.1,
        )
        # With no warmup, step 0 should already be at peak (start of cooldown).
        assert sched.get_lr() == pytest.approx(peak)

    def test_cooldown_steps_zero_jumps_to_stable(self):
        """cooldown_steps=0 skips the peak→stable interpolation entirely."""
        peak = 1.0
        stable = 0.2
        opt = _make_optimizer(lr=peak)
        sched = InfiniteSchedule(
            opt, warmup_steps=5, cooldown_steps=0,
            decay_steps=10, total_steps=50, stable_lr_ratio=stable,
        )
        # At step == warmup_steps, cooldown_end == warmup_steps, so we drop
        # straight to the stable plateau.
        for _ in range(5):
            sched.step()
        assert sched.get_lr() == pytest.approx(stable, rel=1e-6)
        for _ in range(10):
            sched.step()
        assert sched.get_lr() == pytest.approx(stable, rel=1e-6)

    def test_decay_steps_zero_holds_stable_to_end(self):
        """decay_steps=0 means no final decay — stable holds to total_steps."""
        peak = 1.0
        stable = 0.3
        opt = _make_optimizer(lr=peak)
        sched = InfiniteSchedule(
            opt, warmup_steps=5, cooldown_steps=5,
            decay_steps=0, total_steps=50, stable_lr_ratio=stable,
        )
        # Walk up through the schedule and sample points near the tail.
        for _ in range(49):
            sched.step()
        assert sched.get_lr() == pytest.approx(stable, rel=1e-6)

    def test_overlapping_phases_clips_stable(self):
        """When warmup + cooldown + decay > total_steps, the stable
        phase is squeezed to zero and final_decay_start falls back to
        cooldown_end. LR must stay in the well-defined range."""
        peak = 1.0
        opt = _make_optimizer(lr=peak)
        # warmup 10 + cooldown 20 = 30; decay 30; total 40 → overlap.
        sched = InfiniteSchedule(
            opt, warmup_steps=10, cooldown_steps=20,
            decay_steps=30, total_steps=40, stable_lr_ratio=0.2,
        )
        for _ in range(40):
            sched.step()
        # Final step should be at min (default 0).
        assert sched.get_lr() == pytest.approx(0.0, abs=1e-10)


class TestInfiniteScheduleAdapterVariant:
    """Mirror tests for the LambdaLR variant used in adapter training."""

    def test_shape_matches_class_variant(self):
        from pawn.adapter_training import infinite_schedule

        peak = 1.0
        opt_fn = _make_optimizer(lr=peak)
        fn_sched = infinite_schedule(
            opt_fn, warmup_steps=10, cooldown_steps=20,
            decay_steps=30, total_steps=200, stable_lr_ratio=0.1,
        )
        opt_cls = _make_optimizer(lr=peak)
        cls_sched = InfiniteSchedule(
            opt_cls, warmup_steps=10, cooldown_steps=20,
            decay_steps=30, total_steps=200, stable_lr_ratio=0.1,
        )
        # Compare LR trajectories across key phase boundaries.
        for target in (0, 5, 10, 20, 30, 100, 170, 185, 200):
            while fn_sched.last_epoch < target:
                fn_sched.step()
            while cls_sched._step < target:
                cls_sched.step()
            assert opt_fn.param_groups[0]["lr"] == pytest.approx(
                cls_sched.get_lr(), rel=1e-6, abs=1e-10
            )

    def test_build_scheduler_dispatch(self):
        from pawn.adapter_training import build_scheduler

        opt = _make_optimizer(lr=1.0)
        sched = build_scheduler(
            opt, warmup_steps=5, total_steps=50,
            schedule="infinite", decay_steps=10, cooldown_steps=10,
            stable_lr_ratio=0.25,
        )
        # Drive it to the stable plateau and check LR.
        for _ in range(20):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.25, rel=1e-6)

    def test_min_lr_ratio_respected_in_adapter_variant(self):
        """Adapter variant honors min_lr_ratio at the end of final decay."""
        from pawn.adapter_training import infinite_schedule

        peak = 1.0
        opt = _make_optimizer(lr=peak)
        sched = infinite_schedule(
            opt, warmup_steps=5, cooldown_steps=5,
            decay_steps=10, total_steps=50, stable_lr_ratio=0.2,
            min_lr_ratio=0.05,
        )
        for _ in range(500):  # step well past total_steps
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(peak * 0.05, rel=1e-6)

    def test_build_scheduler_infinite_requires_cooldown(self):
        from pawn.adapter_training import build_scheduler

        opt = _make_optimizer(lr=1.0)
        with pytest.raises(ValueError, match="cooldown_steps"):
            build_scheduler(
                opt, warmup_steps=5, total_steps=50,
                schedule="infinite", decay_steps=10,
            )
