"""Tests for ``pawn.distill``: KD loss math, config validation, and the
trainer's training-step plumbing on a toy model.

Heavy paths (full training run, HF teacher download) are intentionally
not exercised here — the smoke test below mirrors the structure of
``test_trainer.TestTrainStepSmoke`` to keep the suite CPU-cheap.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from pawn.config import CLMConfig
from pawn.distill import (
    DistillTrainer,
    _strip_compile_prefix,
    build_student_config,
    kd_loss,
)
from pawn.run_config import DistillConfig


# ---------------------------------------------------------------------------
# kd_loss
# ---------------------------------------------------------------------------


class TestKDLoss:
    """Temperature-scaled KL distillation loss."""

    def test_zero_when_logits_match(self) -> None:
        """KL is zero when teacher and student distributions are equal."""
        logits = torch.randn(8, 32)
        loss = kd_loss(logits.clone(), logits, temperature=2.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_when_logits_differ(self) -> None:
        student = torch.randn(8, 32)
        teacher = torch.randn(8, 32)
        loss = kd_loss(student, teacher, temperature=2.0)
        assert loss.item() > 0.0

    def test_equivalent_to_manual_forward_kl(self) -> None:
        """Forward direction equals ``T**2 * KL(softmax(t/T) || softmax(s/T))``."""
        student = torch.randn(4, 16)
        teacher = torch.randn(4, 16)
        T = 3.0
        # Manual reference
        s_log_p = F.log_softmax(student / T, dim=-1)
        t_p = F.softmax(teacher / T, dim=-1)
        ref = (t_p * (t_p.log() - s_log_p)).sum(dim=-1).mean() * (T * T)

        out = kd_loss(student, teacher, temperature=T, direction="forward")
        assert out.item() == pytest.approx(ref.item(), rel=1e-4)

    def test_reverse_direction_swaps_args(self) -> None:
        student = torch.randn(4, 16)
        teacher = torch.randn(4, 16)
        forward = kd_loss(student, teacher, temperature=2.0, direction="forward")
        reverse = kd_loss(student, teacher, temperature=2.0, direction="reverse")
        # Reverse over the same logits is generally != forward.
        assert forward.item() != pytest.approx(reverse.item(), abs=1e-3)

    def test_jsd_is_symmetric(self) -> None:
        """JSD(student, teacher) == JSD(teacher, student)."""
        a = torch.randn(4, 16)
        b = torch.randn(4, 16)
        ab = kd_loss(a, b, temperature=2.0, direction="jsd")
        ba = kd_loss(b, a, temperature=2.0, direction="jsd")
        assert ab.item() == pytest.approx(ba.item(), rel=1e-5)

    def test_jsd_bounded_by_log2_times_t_squared(self) -> None:
        """JSD between any two distributions is bounded by log(2)."""
        a = torch.randn(4, 32) * 5.0
        b = torch.randn(4, 32) * 5.0
        T = 1.0
        loss = kd_loss(a, b, temperature=T, direction="jsd")
        assert loss.item() <= math.log(2.0) * T * T + 1e-4

    def test_top_k_truncates_support(self) -> None:
        """top_k pulls a different KL value than full vocab when K < V."""
        torch.manual_seed(0)
        student = torch.randn(8, 64)
        teacher = torch.randn(8, 64)
        full = kd_loss(student, teacher, temperature=2.0)
        top4 = kd_loss(student, teacher, temperature=2.0, top_k=4)
        # Different support -> different value.
        assert full.item() != pytest.approx(top4.item(), abs=1e-4)

    def test_top_k_equal_to_full_when_k_eq_vocab(self) -> None:
        student = torch.randn(8, 16)
        teacher = torch.randn(8, 16)
        full = kd_loss(student, teacher, temperature=2.0)
        # top_k=V should match full (same support, same renormalization).
        full_via_topk = kd_loss(student, teacher, temperature=2.0, top_k=16)
        assert full.item() == pytest.approx(full_via_topk.item(), rel=1e-4)

    def test_invalid_temperature_raises(self) -> None:
        student = torch.zeros(2, 4)
        teacher = torch.zeros(2, 4)
        with pytest.raises(ValueError, match="temperature"):
            kd_loss(student, teacher, temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            kd_loss(student, teacher, temperature=-1.0)

    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            kd_loss(torch.zeros(2, 4), torch.zeros(2, 4), 1.0, direction="oops")

    def test_invalid_top_k_raises(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            kd_loss(torch.zeros(2, 4), torch.zeros(2, 4), 1.0, top_k=0)


# ---------------------------------------------------------------------------
# build_student_config
# ---------------------------------------------------------------------------


class TestBuildStudentConfig:
    def test_named_variant(self) -> None:
        cfg = build_student_config(
            "small", d_model=None, n_layers=None, n_heads=None,
            d_ff=None, max_seq_len=128,
        )
        assert cfg.d_model == 256  # CLMConfig.small() default
        assert cfg.n_layers == 8
        assert cfg.n_heads == 4
        assert cfg.max_seq_len == 128

    def test_overrides_apply_on_top_of_preset(self) -> None:
        cfg = build_student_config(
            "small", d_model=128, n_layers=None, n_heads=None,
            d_ff=None, max_seq_len=64,
        )
        # d_model overridden, others taken from small preset.
        assert cfg.d_model == 128
        assert cfg.n_layers == 8

    def test_custom_requires_all_arch_fields(self) -> None:
        with pytest.raises(ValueError, match="custom"):
            build_student_config(
                "custom", d_model=64, n_layers=None, n_heads=4,
                d_ff=128, max_seq_len=64,
            )

    def test_custom_with_full_arch(self) -> None:
        cfg = build_student_config(
            "custom", d_model=64, n_layers=2, n_heads=4,
            d_ff=128, max_seq_len=64,
        )
        assert cfg.d_model == 64
        assert cfg.n_layers == 2


# ---------------------------------------------------------------------------
# _strip_compile_prefix
# ---------------------------------------------------------------------------


class TestStripCompilePrefix:
    def test_drops_orig_mod(self) -> None:
        sd = {"_orig_mod.embed.weight": torch.zeros(1), "lm_head.weight": torch.zeros(1)}
        out = _strip_compile_prefix(sd)
        assert "embed.weight" in out
        assert "lm_head.weight" in out
        assert "_orig_mod.embed.weight" not in out

    def test_no_prefix_passthrough(self) -> None:
        sd = {"a": torch.zeros(1), "b": torch.zeros(2)}
        out = _strip_compile_prefix(sd)
        assert out.keys() == sd.keys()


# ---------------------------------------------------------------------------
# DistillConfig validation
# ---------------------------------------------------------------------------


class TestDistillConfig:
    """Pydantic validators on the run config."""

    def _base(self, **overrides: object) -> DistillConfig:
        kwargs: dict[str, object] = {
            "run_type": "distill",
            "teacher_checkpoint": "thomas-schweich/pawn-base",
            "variant": "small",
            "local_checkpoints": True,
        }
        kwargs.update(overrides)
        return DistillConfig.model_validate(kwargs)

    def test_defaults(self) -> None:
        cfg = self._base()
        assert cfg.temperature == 4.0
        assert cfg.alpha == 0.5
        assert cfg.kd_direction == "forward"
        assert cfg.top_k_teacher is None
        assert cfg.hidden_loss_weight == 0.0

    def test_alpha_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="alpha=1"):
            self._base(alpha=1.0)

    def test_alpha_zero_allowed(self) -> None:
        cfg = self._base(alpha=0.0)
        assert cfg.alpha == 0.0

    def test_alpha_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            self._base(alpha=-0.1)
        with pytest.raises(ValueError, match="alpha"):
            self._base(alpha=1.5)

    def test_temperature_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            self._base(temperature=0.0)

    def test_top_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="top_k_teacher"):
            self._base(top_k_teacher=0)

    def test_hidden_loss_weight_must_be_nonnegative(self) -> None:
        with pytest.raises(ValueError, match="hidden_loss_weight"):
            self._base(hidden_loss_weight=-1.0)

    def test_custom_variant_requires_all_arch(self) -> None:
        with pytest.raises(ValueError, match="custom"):
            self._base(variant="custom", d_model=64)

    def test_unknown_field_rejected(self) -> None:
        # extra="forbid" on BaseRunConfig — typos must surface immediately.
        with pytest.raises(ValueError):
            self._base(unknown_knob=42)

    def test_must_specify_checkpoint_mode(self) -> None:
        with pytest.raises(ValueError, match="hf-repo|local-checkpoints"):
            DistillConfig.model_validate({
                "run_type": "distill",
                "teacher_checkpoint": "thomas-schweich/pawn-base",
            })


# ---------------------------------------------------------------------------
# DistillTrainer end-to-end (toy student + toy teacher)
# ---------------------------------------------------------------------------


def _toy_teacher_checkpoint(tmp_path) -> str:
    """Materialize a toy teacher checkpoint on disk and return its path.

    Uses ``CLMConfig.toy()`` so building the model and the resulting
    save are CPU-cheap. Uses
    :func:`pawn.checkpoint.save_pretrain_checkpoint` so the on-disk
    layout exactly matches what production checkpoints look like.
    """
    import torch.optim as optim
    from pawn.checkpoint import save_pretrain_checkpoint
    from pawn.config import TrainingConfig
    from pawn.model import PAWNCLM
    from pawn.trainer import CosineWithWarmup

    cfg = CLMConfig.toy()
    cfg.max_seq_len = 64
    train_cfg = TrainingConfig.toy()
    train_cfg.max_ply = cfg.max_seq_len

    torch.manual_seed(0)
    model = PAWNCLM(cfg)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    sched = CosineWithWarmup(opt, warmup_steps=10, total_steps=100)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    out = tmp_path / "teacher_ckpt"
    save_pretrain_checkpoint(
        path=str(out),
        model=model,
        optimizer=opt,
        scheduler=sched,
        scaler=scaler,
        global_step=0,
        model_config=cfg.__dict__,
        training_config=train_cfg.__dict__,
    )
    return str(out)


@pytest.mark.integration
class TestDistillTrainerSmoke:
    """End-to-end: construct DistillTrainer with toy student + teacher,
    run a single train_step, and verify the loss flows backward and
    distillation-specific metrics are surfaced."""

    def test_train_step_returns_distill_metrics(self, tmp_path, cpu_device):
        from pawn.config import TrainingConfig

        teacher_path = _toy_teacher_checkpoint(tmp_path)

        student_cfg = CLMConfig.toy()
        student_cfg.max_seq_len = 64

        train_cfg = TrainingConfig.toy()
        train_cfg.device = cpu_device
        train_cfg.max_ply = student_cfg.max_seq_len
        train_cfg.batch_size = 4
        train_cfg.val_games = 4
        train_cfg.use_amp = False
        train_cfg.log_dir = str(tmp_path / "logs")
        train_cfg.checkpoint_dir = str(tmp_path / "ckpts")

        trainer = DistillTrainer(
            train_cfg,
            student_cfg,
            teacher_checkpoint=teacher_path,
            temperature=2.0,
            alpha=0.5,
        )

        # Generate a small batch through the engine and feed it directly.
        import chess_engine
        input_ids, targets, loss_mask, _, _, _ = chess_engine.generate_clm_batch(
            batch_size=4, seq_len=64, seed=42,
        )
        batch = {
            "input_ids": torch.from_numpy(input_ids).long(),
            "targets": torch.from_numpy(targets).long(),
            "loss_mask": torch.from_numpy(loss_mask),
        }

        metrics = trainer.train_step(batch)

        # Distill-specific keys must be present and finite.
        for k in ("loss", "accuracy", "ce_loss", "kd_loss", "teacher_agreement"):
            assert k in metrics, f"missing metric {k}"
            assert torch.isfinite(metrics[k]).item(), f"non-finite {k}: {metrics[k]}"

        # Backward should have populated gradients on the student.
        grads = [
            p.grad for p in trainer._model.parameters() if p.grad is not None
        ]
        assert grads, "no student gradients after train_step"
        assert any((g != 0).any() for g in grads)

        # Teacher must remain frozen — its parameters carry no grads.
        assert all(
            p.grad is None or (p.grad == 0).all()
            for p in trainer.teacher.parameters()
        )

    def test_teacher_vocab_mismatch_rejected(self, tmp_path, cpu_device):
        """A teacher with a different vocabulary must be rejected up-front."""
        from pawn.config import TrainingConfig

        teacher_path = _toy_teacher_checkpoint(tmp_path)

        # Shrink the student vocab to force a mismatch.
        student_cfg = CLMConfig.toy()
        student_cfg.max_seq_len = 64
        student_cfg.vocab_size = 999

        train_cfg = TrainingConfig.toy()
        train_cfg.device = cpu_device
        train_cfg.max_ply = student_cfg.max_seq_len
        train_cfg.batch_size = 4
        train_cfg.val_games = 4
        train_cfg.use_amp = False
        train_cfg.log_dir = str(tmp_path / "logs")
        train_cfg.checkpoint_dir = str(tmp_path / "ckpts")

        with pytest.raises(ValueError, match="vocab_size"):
            DistillTrainer(
                train_cfg,
                student_cfg,
                teacher_checkpoint=teacher_path,
                temperature=2.0,
                alpha=0.5,
            )
