"""Supernet quality-parity harness (Phase-B Stage B3 / plan §7, Alt #1).

The supernet's nested-slice trick (one shared weight tensor, three width
slices) is the v1 mechanism the distillation ladder replaces. To justify
that swap we need to *measure* whether a supernet slice at a given width is
actually worse than an independently-trained ("canonical") model at the
same width — if the gap is negligible the slice trick was free quality, if
it's large the distilled-canonical ladder earns its keep. This module is
that measurement tool.

Given a **supernet** checkpoint and a **canonical** (distilled or
independently-trained) checkpoint at the **same width**, it computes three
gap metrics and emits a JSON-serialisable :class:`GapReport`:

1. **Per-phase move-accuracy delta** — :func:`pawn.eval.compute_per_phase_accuracy`
   on both models, reporting ``supernet - canonical`` for overall / opening
   / midgame / endgame. A *positive* delta means the supernet slice is more
   accurate than the canonical at that phase.
2. **Linear-probe decodability delta** — a single :mod:`pawn.probes` linear
   probe fit on each model's final hidden state, decoding a genuine board
   feature (the source square of the move each supervised position
   predicts, read off the engine ``decomp_table``). The delta is
   ``supernet_probe_acc - canonical_probe_acc`` — how much more linearly
   decodable the supernet's representation is.
3. **Reference-LoRA val-loss delta** — a tiny reference LoRA adapter is
   trained for a handful of steps on each frozen backbone (identical adapter
   config + seed), then the held-out cross-entropy is measured. The delta is
   ``supernet_val_loss - canonical_val_loss``; a *negative* delta means the
   supernet slice reaches a lower loss under the same light finetune.

The supernet stays behind its existing flag as the **contrast arm**: the
caller passes the supernet checkpoint and the harness slices it down to the
canonical's width via :func:`pawn.model.sliced` (the same nested-slice the
publish path uses), so the comparison is genuinely "the supernet's slice at
width W" vs "a canonical model at width W".

This is a *measurement* tool. The unit tests pin the harness mechanics
(it runs on two tiny checkpoints and emits a well-formed gap report with the
right metric signs/shapes); the full cross-model comparison runs at
production scale are later analysis, **not** a Phase-B gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, Int

from pawn.adapters.lora import LoRAConfig, apply_lora, init_lora_adapter
from pawn.config import NUM_ACTIONS
from pawn.corpus import Corpus
from pawn.eval import (
    AccuracyResult,
    PhaseBoundaries,
    compute_per_phase_accuracy,
)
from pawn.model import PAWNModel, sliced
from pawn.probes import ProbeConfig, ProbeResult, fit_probe
from pawn.trainer import (
    Batch,
    cross_entropy_loss,
    make_lr_schedule,
    make_optimizer,
    slice_batch,
)
from pawn.run_config import BaseRunConfig

__all__ = [
    "ReferenceLoRASpec",
    "PhaseAccuracyGap",
    "ProbeGap",
    "ValLossGap",
    "GapReport",
    "extract_hidden_states",
    "source_square_labels",
    "phase_accuracy_gap",
    "probe_decodability_gap",
    "reference_lora_val_loss",
    "reference_lora_gap",
    "run_parity_harness",
]


# ---------------------------------------------------------------------------
# Reference-LoRA training spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceLoRASpec:
    """Hyperparameters for the reference-LoRA val-loss probe.

    The same spec is applied to **both** backbones so the only variable in
    the resulting val-loss delta is the backbone weights. ``rank`` /
    ``targets`` / ``ffn`` size the adapter; ``steps`` is how many
    single-step updates to run on the train batches; ``lr`` /
    ``weight_decay`` / ``max_grad_norm`` parametrise the optimizer; ``seed``
    seeds the adapter init so both backbones get the *same* starting
    adapter.
    """

    rank: int = 4
    targets: Any = "qkvo"
    ffn: bool = False
    steps: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 0


# ---------------------------------------------------------------------------
# Gap report dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseAccuracyGap:
    """Per-phase move-accuracy gap (``supernet - canonical``).

    ``supernet`` / ``canonical`` carry the raw :class:`pawn.eval.AccuracyResult`
    accuracies (as plain dicts for JSON); the ``*_delta`` fields are the
    signed differences. A positive delta means the supernet slice is more
    accurate at that phase.
    """

    overall_delta: float
    opening_delta: float
    midgame_delta: float
    endgame_delta: float
    supernet: dict[str, float | int]
    canonical: dict[str, float | int]


@dataclass(frozen=True)
class ProbeGap:
    """Linear-probe decodability gap (``supernet - canonical``).

    ``*_accuracy`` are the fitted linear-probe accuracies on each model's
    final hidden state; ``delta`` is their signed difference. A positive
    delta means the supernet's representation is more linearly decodable for
    the probed board feature.
    """

    delta: float
    supernet_accuracy: float
    canonical_accuracy: float
    n_classes: int
    n_samples: int


@dataclass(frozen=True)
class ValLossGap:
    """Reference-LoRA held-out val-loss gap (``supernet - canonical``).

    ``*_val_loss`` are the cross-entropies after the identical light LoRA
    finetune on each backbone; ``delta`` is their signed difference. A
    *negative* delta means the supernet slice reached a lower loss under the
    same finetune.
    """

    delta: float
    supernet_val_loss: float
    canonical_val_loss: float
    steps: int


@dataclass(frozen=True)
class GapReport:
    """The full supernet-vs-canonical parity report.

    All three deltas are signed ``supernet - canonical``. ``width`` records
    the common width the comparison was run at (the canonical's ``d_model``);
    :meth:`to_dict` flattens the whole thing for JSON emission.
    """

    width: int
    phase_accuracy: PhaseAccuracyGap
    probe: ProbeGap
    val_loss: ValLossGap

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable nested dict of every gap metric."""
        return {
            "width": self.width,
            "phase_accuracy": asdict(self.phase_accuracy),
            "probe": asdict(self.probe),
            "val_loss": asdict(self.val_loss),
        }


# ---------------------------------------------------------------------------
# Width reconciliation
# ---------------------------------------------------------------------------


def _slice_supernet_to_canonical(
    supernet: PAWNModel, canonical: PAWNModel
) -> PAWNModel:
    """Return the supernet's nested slice at the canonical's width.

    If the two models already share a config the supernet is returned
    untouched; otherwise :func:`pawn.model.sliced` takes the inner
    ``[:d_V, :d_V]`` block (raising :class:`pawn.config.NestingError` when
    the canonical's config doesn't nest under the supernet's). This is the
    "supernet-slice vs canonical at that width" contract from the Stage-B3
    spec — the supernet is the *contrast arm*, sliced down to meet the
    canonical, never the other way around.
    """
    if supernet.cfg == canonical.cfg:
        return supernet
    return sliced(supernet, canonical.cfg)


# ---------------------------------------------------------------------------
# (1) Per-phase move-accuracy gap
# ---------------------------------------------------------------------------


def phase_accuracy_gap(
    supernet: PAWNModel,
    canonical: PAWNModel,
    corpus: Corpus,
    *,
    batch_size: int = 32,
    phases: PhaseBoundaries = PhaseBoundaries(),
) -> PhaseAccuracyGap:
    """Per-phase move-accuracy gap between the two models on ``corpus``.

    Both models are evaluated with :func:`pawn.eval.compute_per_phase_accuracy`;
    the returned gap is ``supernet - canonical`` per phase.
    """
    sup_acc = compute_per_phase_accuracy(
        supernet, corpus, batch_size=batch_size, phases=phases
    )
    can_acc = compute_per_phase_accuracy(
        canonical, corpus, batch_size=batch_size, phases=phases
    )
    return PhaseAccuracyGap(
        overall_delta=sup_acc.overall - can_acc.overall,
        opening_delta=sup_acc.opening - can_acc.opening,
        midgame_delta=sup_acc.midgame - can_acc.midgame,
        endgame_delta=sup_acc.endgame - can_acc.endgame,
        supernet=_accuracy_to_dict(sup_acc),
        canonical=_accuracy_to_dict(can_acc),
    )


def _accuracy_to_dict(acc: AccuracyResult) -> dict[str, float | int]:
    return {
        "overall": acc.overall,
        "opening": acc.opening,
        "midgame": acc.midgame,
        "endgame": acc.endgame,
        "n_total": acc.n_total,
        "n_opening": acc.n_opening,
        "n_midgame": acc.n_midgame,
        "n_endgame": acc.n_endgame,
    }


# ---------------------------------------------------------------------------
# (2) Linear-probe decodability gap
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _final_hidden(
    model: PAWNModel,
    tokens: Int[Array, "B T"],
    attn_mask: Int[Array, "B T"],
) -> Float[Array, "B T d"]:
    """Final-norm hidden state (the representation that feeds the head).

    Runs the same forward as :meth:`PAWNModel.__call__` but stops at the
    post-``final_norm`` activation rather than projecting to logits, so the
    probe sees the model's last-layer representation. fp32 throughout
    (``compute_dtype=None``) for a deterministic, bit-stable extraction.
    """
    from pawn.model import _build_rope, _rmsnorm

    T = tokens.shape[-1]
    x = model._embed(tokens)
    rope_cos, rope_sin = _build_rope(model.cfg.head_dim, T, model.cfg.rope_base)
    causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    pad = attn_mask.astype(jnp.bool_)[:, None, None, :]
    mask = causal[None, None, :, :] & pad
    x = model._run_layers(x, rope_cos, rope_sin, mask, attn_mask, None)
    return _rmsnorm(x, model.final_norm_w)


def extract_hidden_states(
    model: PAWNModel,
    corpus: Corpus,
    *,
    batch_size: int = 32,
    max_positions: int | None = None,
) -> tuple[Float[Array, "N d"], Int[Array, "N"]]:
    """Gather final hidden states + the move token each supervised position
    predicts, over the supervised positions of ``corpus``.

    Returns ``(hidden, target_tokens)`` where ``hidden`` is ``(N, d)`` and
    ``target_tokens`` is ``(N,)`` move-token ids in ``[0, NUM_ACTIONS)`` —
    only positions whose ``loss_mask`` is True and whose target is a real
    move token contribute. ``max_positions`` caps ``N`` (keeps the probe fit
    cheap); ``None`` keeps every supervised position.
    """
    n = corpus.n_games
    hidden_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    collected = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        tokens = jnp.asarray(corpus.tokens[start:end])
        targets = jnp.asarray(corpus.targets[start:end])
        attn = jnp.asarray(corpus.attn_mask[start:end])
        loss = jnp.asarray(corpus.loss_mask[start:end]).astype(jnp.bool_)
        hidden = _final_hidden(model, tokens, attn)  # (B, T, d)
        # Only supervised positions predicting a real move token.
        sup = np.asarray(loss)
        tgt = np.asarray(targets)
        keep = sup & (tgt >= 0) & (tgt < NUM_ACTIONS)
        if not keep.any():
            continue
        h_flat = np.asarray(hidden)[keep]  # (n_keep, d)
        t_flat = tgt[keep].astype(np.int32)  # (n_keep,)
        hidden_chunks.append(h_flat)
        label_chunks.append(t_flat)
        collected += h_flat.shape[0]
        if max_positions is not None and collected >= max_positions:
            break
    if not hidden_chunks:
        d = model.cfg.d_model
        return (
            jnp.zeros((0, d), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.int32),
        )
    hidden_all = np.concatenate(hidden_chunks, axis=0)
    labels_all = np.concatenate(label_chunks, axis=0)
    if max_positions is not None and hidden_all.shape[0] > max_positions:
        hidden_all = hidden_all[:max_positions]
        labels_all = labels_all[:max_positions]
    return jnp.asarray(hidden_all), jnp.asarray(labels_all)


def source_square_labels(
    model: PAWNModel, target_tokens: Int[Array, "N"]
) -> Int[Array, "N"]:
    """Map move tokens to their source square (0..63) via ``decomp_table``.

    The probe target is a genuine board feature: the source square of the
    move each supervised position predicts. ``decomp_table[token]`` is
    ``(src, dst, promo)``; the source square is column 0, always in
    ``[0, 64)``, so the probe has a fixed 64-class label space independent
    of the model width.
    """
    return model.decomp_table[target_tokens, 0]


def probe_decodability_gap(
    supernet: PAWNModel,
    canonical: PAWNModel,
    corpus: Corpus,
    *,
    batch_size: int = 32,
    max_positions: int | None = 2048,
    probe_epochs: int = 20,
    probe_lr: float = 1e-2,
    probe_seed: int = 0,
) -> ProbeGap:
    """Linear-probe decodability gap between the two models.

    Extracts each model's final hidden states over the same supervised
    positions, fits an independent :func:`pawn.probes.fit_probe` linear
    classifier to decode the next-move source square (64 classes), and
    returns ``supernet_probe_acc - canonical_probe_acc``. The probe is fit
    with the **same** seed / epochs / lr for both models so the only
    variable is the representation.
    """
    sup_hidden, sup_tokens = extract_hidden_states(
        supernet, corpus, batch_size=batch_size, max_positions=max_positions
    )
    can_hidden, can_tokens = extract_hidden_states(
        canonical, corpus, batch_size=batch_size, max_positions=max_positions
    )
    # Source-square label depends only on the move token (decomp_table is
    # width-invariant), so both label sets come out identical when the
    # corpus + supervised positions match — but read each model's own table
    # so the function stays correct for unrelated checkpoints.
    sup_labels = source_square_labels(supernet, sup_tokens)
    can_labels = source_square_labels(canonical, can_tokens)

    n_classes = 64  # source squares 0..63
    sup_result = _fit_probe_safe(
        sup_hidden, sup_labels, n_classes, probe_epochs, probe_lr, probe_seed
    )
    can_result = _fit_probe_safe(
        can_hidden, can_labels, n_classes, probe_epochs, probe_lr, probe_seed
    )
    n_samples = int(min(sup_hidden.shape[0], can_hidden.shape[0]))
    return ProbeGap(
        delta=sup_result.accuracy - can_result.accuracy,
        supernet_accuracy=sup_result.accuracy,
        canonical_accuracy=can_result.accuracy,
        n_classes=n_classes,
        n_samples=n_samples,
    )


def _fit_probe_safe(
    hidden: Float[Array, "N d"],
    labels: Int[Array, "N"],
    n_classes: int,
    epochs: int,
    lr: float,
    seed: int,
) -> ProbeResult:
    """Fit a probe, returning a 0-accuracy result on an empty sample set.

    An empty supervised-position set (degenerate corpus) would make
    :func:`pawn.probes.fit_probe` divide by zero in its accuracy reduction;
    guard it so the harness still emits a well-formed report.
    """
    if hidden.shape[0] == 0:
        d = hidden.shape[-1]
        return ProbeResult(
            accuracy=0.0,
            weight=jnp.zeros((d, n_classes), dtype=jnp.float32),
            bias=jnp.zeros((n_classes,), dtype=jnp.float32),
        )
    cfg = ProbeConfig(n_classes=n_classes, lr=lr, n_epochs=epochs)
    return fit_probe(hidden, labels, cfg, key=seed)


# ---------------------------------------------------------------------------
# (3) Reference-LoRA val-loss gap
# ---------------------------------------------------------------------------


def _ref_lora_config(spec: ReferenceLoRASpec) -> BaseRunConfig:
    """Build the :class:`BaseRunConfig` that feeds the reference-LoRA optimizer.

    :func:`pawn.trainer.make_optimizer` / :func:`pawn.trainer.make_lr_schedule`
    read ``lr`` / ``weight_decay`` / ``max_grad_norm`` / ``lr_schedule`` /
    ``optimizer`` off a :class:`BaseRunConfig`. A flat ``constant`` schedule
    with no warmup is right for a 20-step probe; ``local_checkpoints=True``
    satisfies the storage-mode validation. This config is never serialised —
    it exists only to parametrise the optimizer builders.
    """
    return BaseRunConfig(
        lr=spec.lr,
        weight_decay=spec.weight_decay,
        max_grad_norm=spec.max_grad_norm,
        lr_schedule="constant",
        warmup_frac=0.0,
        local_checkpoints=True,
    )


def reference_lora_val_loss(
    backbone: PAWNModel,
    train_batches: list[Batch],
    val_batch: Batch,
    spec: ReferenceLoRASpec,
) -> float:
    """Train a reference LoRA on ``backbone`` then return its held-out CE.

    A fresh LoRA adapter (seeded by ``spec.seed`` so it is *identical*
    across backbones) is trained for ``spec.steps`` single-step updates over
    ``train_batches`` (cycled if fewer than ``steps``), then the
    cross-entropy on ``val_batch`` is measured with the LoRA-composed model.
    Only the adapter trains; the backbone is frozen (gradients flow through
    the adapter leaves only, mirroring
    :func:`pawn.adapter_trainer.make_adapter_train_step`).
    """
    lora_cfg = LoRAConfig(rank=spec.rank, targets=spec.targets, ffn=spec.ffn)
    adapter = init_lora_adapter(backbone, lora_cfg, key=spec.seed)

    run_cfg = _ref_lora_config(spec)
    schedule = make_lr_schedule(run_cfg, max(spec.steps, 1))
    optimizer = make_optimizer(run_cfg, schedule)
    opt_state = optimizer.init(eqx.filter(adapter, eqx.is_inexact_array))

    # ``batch`` is reused across iterations (train_batches is cycled when it
    # is shorter than ``steps``), so it must NOT be donated — on a
    # donation-honoring backend (GPU/ROCm) a donated input is aliased and
    # deleted after the call, and re-passing the same array on a later
    # iteration raises ``RuntimeError: Array has been deleted``. Put ``batch``
    # first and donate "all-except-first" so only the per-iteration-fresh
    # ``adapter_`` / ``opt_state_`` buffers are donated.
    @eqx.filter_jit(donate="all-except-first")
    def step(
        batch: Batch, adapter_: Any, opt_state_: optax.OptState
    ) -> tuple[Any, optax.OptState, Float[Array, ""]]:
        def loss_fn(a: Any) -> Float[Array, ""]:
            effective = apply_lora(backbone, a)
            return cross_entropy_loss(effective, batch)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(adapter_)
        params: Any = adapter_
        updates, new_opt = optimizer.update(grads, opt_state_, params)
        new_adapter = eqx.apply_updates(adapter_, updates)
        return new_adapter, new_opt, loss

    if not train_batches:
        raise ValueError("reference_lora_val_loss requires >=1 train batch")
    for i in range(spec.steps):
        batch = train_batches[i % len(train_batches)]
        adapter, opt_state, _loss = step(batch, adapter, opt_state)

    effective = apply_lora(backbone, adapter)
    val_loss = cross_entropy_loss(effective, val_batch)
    return float(val_loss)


def reference_lora_gap(
    supernet: PAWNModel,
    canonical: PAWNModel,
    train_batches: list[Batch],
    val_batch: Batch,
    spec: ReferenceLoRASpec = ReferenceLoRASpec(),
) -> ValLossGap:
    """Reference-LoRA val-loss gap between the two backbones.

    The *same* reference-LoRA finetune (identical adapter config + seed +
    optimizer + train/val batches) is run against each backbone; the gap is
    ``supernet_val_loss - canonical_val_loss``. A negative delta means the
    supernet slice reached a lower held-out loss under the light finetune.
    """
    sup_loss = reference_lora_val_loss(supernet, train_batches, val_batch, spec)
    can_loss = reference_lora_val_loss(canonical, train_batches, val_batch, spec)
    return ValLossGap(
        delta=sup_loss - can_loss,
        supernet_val_loss=sup_loss,
        canonical_val_loss=can_loss,
        steps=spec.steps,
    )


# ---------------------------------------------------------------------------
# Top-level harness
# ---------------------------------------------------------------------------


def _build_reference_batches(
    train_corpus: Corpus,
    val_corpus: Corpus,
    *,
    batch_size: int,
    n_train_batches: int,
) -> tuple[list[Batch], Batch]:
    """Slice ``n_train_batches`` train batches + one val batch from corpora.

    Train batches are drawn round-robin from the front of ``train_corpus``
    (wrapping if it is smaller than ``n_train_batches * batch_size``); the
    val batch is the first ``batch_size`` games of ``val_corpus``.
    """
    n_train = train_corpus.n_games
    train_batches: list[Batch] = []
    for b in range(n_train_batches):
        start = (b * batch_size) % max(n_train, 1)
        idx = (np.arange(batch_size) + start) % max(n_train, 1)
        train_batches.append(slice_batch(train_corpus, idx))
    n_val = val_corpus.n_games
    val_idx = np.arange(min(batch_size, n_val))
    val_batch = slice_batch(val_corpus, val_idx)
    return train_batches, val_batch


def run_parity_harness(
    supernet: PAWNModel,
    canonical: PAWNModel,
    eval_corpus: Corpus,
    train_corpus: Corpus,
    val_corpus: Corpus,
    *,
    batch_size: int = 32,
    phases: PhaseBoundaries = PhaseBoundaries(),
    probe_max_positions: int | None = 2048,
    probe_epochs: int = 20,
    ref_lora_spec: ReferenceLoRASpec = ReferenceLoRASpec(),
    ref_lora_train_batches: int = 4,
) -> GapReport:
    """Run the full supernet-vs-canonical parity harness → :class:`GapReport`.

    ``supernet`` is sliced down to the ``canonical``'s width (the contrast
    arm); the three gap metrics are then computed at that common width:

    - per-phase move accuracy on ``eval_corpus``,
    - linear-probe decodability on ``eval_corpus`` hidden states,
    - reference-LoRA held-out val loss (trained on ``train_corpus`` batches,
      measured on ``val_corpus``).

    All deltas are signed ``supernet - canonical``. The returned report's
    :meth:`GapReport.to_dict` is JSON-serialisable for emission.
    """
    sup = _slice_supernet_to_canonical(supernet, canonical)
    width = canonical.cfg.d_model

    phase_gap = phase_accuracy_gap(
        sup, canonical, eval_corpus, batch_size=batch_size, phases=phases
    )
    probe_gap = probe_decodability_gap(
        sup, canonical, eval_corpus,
        batch_size=batch_size,
        max_positions=probe_max_positions,
        probe_epochs=probe_epochs,
        probe_seed=ref_lora_spec.seed,
    )
    train_batches, val_batch = _build_reference_batches(
        train_corpus, val_corpus,
        batch_size=batch_size, n_train_batches=ref_lora_train_batches,
    )
    val_gap = reference_lora_gap(
        sup, canonical, train_batches, val_batch, ref_lora_spec
    )
    return GapReport(
        width=width,
        phase_accuracy=phase_gap,
        probe=probe_gap,
        val_loss=val_gap,
    )
