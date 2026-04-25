"""Knowledge distillation for PAWN.

Trains a student :class:`~pawn.model.PAWNCLM` against a frozen teacher.
The student sees the same on-the-fly random-game batches the
pretraining trainer consumes; the teacher provides per-position soft
targets that the student matches via a temperature-scaled KL loss
(`Hinton et al., 2015 <https://arxiv.org/abs/1503.02531>`_).

The trainer subclasses :class:`~pawn.trainer.CLMTrainer` so it
inherits dataset construction, the LR schedules, atomic checkpointing,
the dashboard logger, the HF background pusher, eval, signal
handling, and the W&B integration unchanged. The two overrides are:

* ``__init__`` — load the teacher from a checkpoint, freeze it, and
  hold onto distillation hyperparameters.
* ``train_step`` — replace the standard CE forward with a combined
  ``alpha * CE + (1 - alpha) * T**2 * KL`` loss, plus optional
  hidden-state matching.

The teacher's ``forward_eval`` is invoked under ``torch.no_grad`` and
its full vocabulary projection is computed only at loss-mask positions
to match the memory profile of the student's training step.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from pawn.checkpoint import load_backbone_weights
from pawn.config import CLMConfig, TrainingConfig
from pawn.model import PAWNCLM
from pawn.trainer import CLMTrainer


def _strip_compile_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Drop the ``_orig_mod.`` prefix that ``torch.compile`` adds."""
    return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}


def load_teacher(
    checkpoint: str, device: str,
) -> tuple[PAWNCLM, CLMConfig]:
    """Load a frozen teacher model and its config.

    Returns the model in eval mode with all parameters frozen plus
    its :class:`CLMConfig`. ``checkpoint`` may be an HF repo id or a
    local checkpoint directory — :func:`load_backbone_weights` handles
    both. Raises ``ValueError`` if the checkpoint has no embedded
    model config (we need the architecture to instantiate the model).
    """
    state_dict, model_config = load_backbone_weights(checkpoint, device=device)
    if not model_config:
        raise ValueError(
            f"Teacher checkpoint {checkpoint!r} has no embedded "
            "model_config — cannot reconstruct teacher architecture. "
            "Re-export the teacher with a config.json next to its "
            "model.safetensors."
        )

    teacher_cfg = CLMConfig(**model_config)
    teacher = PAWNCLM(teacher_cfg).to(device)
    teacher.load_state_dict(_strip_compile_prefix(state_dict), strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, teacher_cfg


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    direction: str = "forward",
    top_k: int | None = None,
) -> torch.Tensor:
    """Temperature-scaled KL distillation loss.

    Args:
        student_logits: ``(N, V)`` student logits at supervised positions.
        teacher_logits: ``(N, V)`` teacher logits at the same positions
            (must be detached / under ``no_grad`` upstream — this
            function does not detach).
        temperature: softmax temperature ``T``.
        direction: ``"forward"`` for ``KL(teacher || student)`` (Hinton
            form), ``"reverse"`` for ``KL(student || teacher)``,
            ``"jsd"`` for the symmetric Jensen-Shannon divergence.
        top_k: if set, restrict the KL to the union of each row's
            top-k teacher tokens (renormalizing teacher mass over that
            subset). Cuts memory and dampens noise from low-probability
            classes — common practice when the teacher's vocab is much
            larger than the relevant support.

    Returns:
        Scalar tensor scaled by ``T**2`` (matches the gradient
        magnitude convention from the Hinton paper).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if direction not in ("forward", "reverse", "jsd"):
        raise ValueError(
            f"direction must be 'forward', 'reverse', or 'jsd', got "
            f"{direction!r}"
        )

    if top_k is not None:
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0 or None, got {top_k}")
        # Gather both logit slices over the teacher's top-k columns.
        # We restrict the student to the same columns so the two
        # distributions are defined over the same support — otherwise
        # the KL is meaningless.
        k = min(top_k, teacher_logits.shape[-1])
        topk_idx = teacher_logits.topk(k, dim=-1).indices  # (N, k)
        teacher_logits = teacher_logits.gather(-1, topk_idx)
        student_logits = student_logits.gather(-1, topk_idx)

    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

    if direction == "forward":
        # KL(teacher || student) = E_teacher[log p_t - log p_s]
        loss = F.kl_div(
            s_log_probs, t_log_probs, reduction="batchmean", log_target=True,
        )
    elif direction == "reverse":
        loss = F.kl_div(
            t_log_probs, s_log_probs, reduction="batchmean", log_target=True,
        )
    else:  # jsd
        # m = 0.5 * (p_t + p_s); JSD = 0.5 * KL(p_t||m) + 0.5 * KL(p_s||m)
        log_m = torch.logaddexp(t_log_probs, s_log_probs) - math.log(2.0)
        loss_t = F.kl_div(log_m, t_log_probs, reduction="batchmean", log_target=True)
        loss_s = F.kl_div(log_m, s_log_probs, reduction="batchmean", log_target=True)
        loss = 0.5 * (loss_t + loss_s)

    return loss * (temperature * temperature)


class _HiddenProjector(nn.Module):
    """Linear projection used when teacher and student have different
    ``d_model`` so we can MSE-match hidden states.

    A no-op when both dimensions are equal.
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        if student_dim == teacher_dim:
            self.proj: nn.Module = nn.Identity()
        else:
            self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DistillTrainer(CLMTrainer):
    """PAWN knowledge-distillation trainer.

    Subclasses :class:`CLMTrainer` so checkpointing, eval, the LR
    schedules, the data pipeline, and W&B integration are inherited
    unchanged. Only ``__init__`` (load teacher) and ``train_step``
    (KD loss) differ.
    """

    def __init__(
        self,
        train_cfg: TrainingConfig,
        student_cfg: CLMConfig,
        teacher_checkpoint: str,
        *,
        temperature: float = 4.0,
        alpha: float = 0.5,
        kd_direction: str = "forward",
        top_k_teacher: int | None = None,
        hidden_loss_weight: float = 0.0,
        hf_repo: str | None = None,
        patience: int | None = None,
        legality_late_ply: int | None = None,
        run_config: dict[str, object] | None = None,
        wandb_tags: list[str] | None = None,
    ):
        super().__init__(
            train_cfg,
            student_cfg,
            hf_repo=hf_repo,
            patience=patience,
            legality_late_ply=legality_late_ply,
            run_config=run_config,
            wandb_tags=wandb_tags,
        )

        print(f"Loading teacher from {teacher_checkpoint}...")
        teacher, teacher_cfg = load_teacher(teacher_checkpoint, self.device)
        if teacher_cfg.vocab_size != student_cfg.vocab_size:
            raise ValueError(
                f"Teacher vocab_size={teacher_cfg.vocab_size} does not "
                f"match student vocab_size={student_cfg.vocab_size}. "
                "Distillation requires a shared vocabulary."
            )
        if teacher_cfg.max_seq_len < student_cfg.max_seq_len:
            raise ValueError(
                f"Teacher max_seq_len={teacher_cfg.max_seq_len} is shorter "
                f"than student max_seq_len={student_cfg.max_seq_len}; the "
                "teacher cannot be queried on positions it never saw."
            )
        self.teacher = teacher
        self.teacher_cfg = teacher_cfg

        self.temperature = temperature
        self.alpha = alpha
        self.kd_direction = kd_direction
        self.top_k_teacher = top_k_teacher
        self.hidden_loss_weight = hidden_loss_weight

        # Hidden-state matching projector. Built only when needed so the
        # parameter count and optimizer state stay clean for the common
        # logits-only path.
        self.hidden_projector: _HiddenProjector | None = None
        if hidden_loss_weight > 0.0:
            self.hidden_projector = _HiddenProjector(
                student_cfg.d_model, teacher_cfg.d_model,
            ).to(self.device)
            # Add the projector params to the existing optimizer so they
            # train with the same LR/wd/scheduler as the student.
            extra = [p for p in self.hidden_projector.parameters() if p.requires_grad]
            if extra:
                self.optimizer.add_param_group({"params": extra})
                # Keep the scheduler's per-group base_lrs in sync — it
                # caches them at construction time.
                if hasattr(self.scheduler, "base_lrs"):
                    self.scheduler.base_lrs.append(train_cfg.lr)

        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self._model.parameters())
        print(
            f"Distillation: teacher {teacher_params:,} params -> "
            f"student {student_params:,} params"
        )
        print(
            f"  T={self.temperature}, alpha={self.alpha}, "
            f"direction={self.kd_direction}, "
            f"top_k_teacher={self.top_k_teacher}, "
            f"hidden_loss_weight={self.hidden_loss_weight}"
        )

    def _teacher_forward(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the teacher and return ``(valid_logits, valid_hidden)``.

        ``valid_*`` are gathered at ``loss_mask`` positions; ``valid_hidden``
        is returned regardless of whether hidden-matching is enabled
        (cheap to keep around) so the train step can stay branchless.
        """
        with torch.no_grad():
            hidden = self.teacher.forward_eval(input_ids, loss_mask)
            valid_hidden = hidden[loss_mask]
            valid_logits = self.teacher.lm_head(valid_hidden)
        return valid_logits, valid_hidden

    def train_step(
        self, batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        self.model.train()

        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        targets = batch["targets"].to(self.device, non_blocking=True)
        loss_mask = batch["loss_mask"].to(self.device, non_blocking=True)

        student = self._eager_model()

        # Teacher under autocast — same precision as the student so
        # logit magnitudes (and therefore KL) are comparable.
        with torch.amp.autocast(self.device, enabled=self.cfg.use_amp):
            teacher_valid_logits, teacher_valid_hidden = self._teacher_forward(
                input_ids, loss_mask,
            )

            # Student forward (mirrors PAWNCLM.forward_train but keeps
            # the hidden state around in case we need it for matching).
            hidden = student.forward_eval(input_ids, loss_mask)
            student_valid_hidden = hidden[loss_mask]
            student_valid_logits = student.lm_head(student_valid_hidden)
            valid_targets = targets[loss_mask]

            ce_loss = F.cross_entropy(student_valid_logits, valid_targets)
            kd = kd_loss(
                student_valid_logits,
                teacher_valid_logits,
                temperature=self.temperature,
                direction=self.kd_direction,
                top_k=self.top_k_teacher,
            )
            loss = self.alpha * ce_loss + (1.0 - self.alpha) * kd

            if self.hidden_projector is not None and self.hidden_loss_weight > 0:
                projected = self.hidden_projector(student_valid_hidden)
                hidden_mse = F.mse_loss(projected, teacher_valid_hidden)
                loss = loss + self.hidden_loss_weight * hidden_mse
            else:
                hidden_mse = torch.zeros((), device=self.device)

            with torch.no_grad():
                preds = student_valid_logits.argmax(dim=-1)
                accuracy = (preds == valid_targets).float().mean()
                teacher_preds = teacher_valid_logits.argmax(dim=-1)
                teacher_agreement = (preds == teacher_preds).float().mean()

        scaled_loss = loss / self.cfg.accumulation_steps
        self.scaler.scale(scaled_loss).backward()

        return {
            "loss": loss.detach(),
            "accuracy": accuracy,
            "ce_loss": ce_loss.detach(),
            "kd_loss": kd.detach(),
            "hidden_mse": hidden_mse.detach(),
            "teacher_agreement": teacher_agreement,
        }

def build_student_config(
    variant: str,
    *,
    d_model: int | None,
    n_layers: int | None,
    n_heads: int | None,
    d_ff: int | None,
    max_seq_len: int,
) -> CLMConfig:
    """Resolve the student :class:`CLMConfig` from RunConfig fields.

    Mirrors ``run_pretrain``'s variant + override logic but doesn't
    depend on it (kept here so the trainer can be used standalone).
    """
    factories = {
        "small": CLMConfig.small,
        "base": CLMConfig.base,
        "large": CLMConfig.large,
        "toy": CLMConfig.toy,
    }
    if variant == "custom":
        if None in (d_model, n_layers, n_heads, d_ff):
            raise ValueError(
                "variant='custom' requires d_model, n_layers, n_heads, "
                "and d_ff to be set."
            )
        cfg = CLMConfig(
            d_model=cast(int, d_model),
            n_layers=cast(int, n_layers),
            n_heads=cast(int, n_heads),
            d_ff=cast(int, d_ff),
        )
    else:
        cfg = factories[variant]()
        if d_model is not None:
            cfg.d_model = d_model
        if n_layers is not None:
            cfg.n_layers = n_layers
        if n_heads is not None:
            cfg.n_heads = n_heads
        if d_ff is not None:
            cfg.d_ff = d_ff
    cfg.max_seq_len = max_seq_len
    return cfg
