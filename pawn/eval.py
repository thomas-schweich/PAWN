"""Move-accuracy evaluation — overall + per-phase breakdown.

Argmax is restricted to ``[0, NUM_ACTIONS)`` so PAD and outcome tokens
can't be sampled (the v1 contract per plan §10 S8). Per-phase
breakdown bins moves by game phase (opening / midgame / endgame) on a
ply threshold.

:func:`compute_val_metrics` is the held-out validation pass the
supernet pretrain loop runs on a freshly-generated val corpus — the v2
parity of v1's ``CLMTrainer.evaluate`` (``git show
main:pawn/trainer.py``). It returns the ``val/*`` schema the dashboard
``pawn`` run type charts (``val/loss``, ``val/top1``, ``val/top5``,
``val/perplexity``, ``val/legal_move_rate``, ``val/late_legal_move_rate``)
plus the per-phase breakdown, all computed with one device→host sync.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from pawn.config import NUM_ACTIONS
from pawn.corpus import Corpus
from pawn.model import EffectiveCallable

__all__ = [
    "PhaseBoundaries",
    "AccuracyResult",
    "ValMetrics",
    "CompoundLegalityResult",
    "PerPlyResult",
    "compute_compound_legality",
    "compute_move_accuracy",
    "compute_per_phase_accuracy",
    "compute_per_ply_accuracy",
    "compute_val_metrics",
]


@dataclass(frozen=True)
class PhaseBoundaries:
    """Ply thresholds for opening / midgame / endgame split.

    Defaults: opening ≤ 20 ply (10 full moves); midgame 21..60; endgame > 60.
    Boundaries are inclusive on the lower bound, exclusive on the upper.
    """

    opening_end: int = 20
    midgame_end: int = 60


@dataclass(frozen=True)
class AccuracyResult:
    """Aggregate accuracy + per-phase breakdown."""

    overall: float
    opening: float
    midgame: float
    endgame: float
    n_total: int
    n_opening: int
    n_midgame: int
    n_endgame: int


@eqx.filter_jit
def _argmax_over_actions(
    logits: Float[Array, "B T V"],
) -> Int[Array, "B T"]:
    """Argmax restricted to ``[0, NUM_ACTIONS)``. PAD (token 1968) and
    outcome tokens (1969+) can't be sampled."""
    move_logits = logits[..., :NUM_ACTIONS]
    return jnp.argmax(move_logits, axis=-1)


@eqx.filter_jit
def _batch_correct(
    model: EffectiveCallable,
    tokens: Int[Array, "B T"],
    targets: Int[Array, "B T"],
    attn_mask: Bool[Array, "B T"],
    loss_mask: Bool[Array, "B T"],
) -> tuple[Int[Array, ""], Int[Array, ""]]:
    """JIT'd inner: returns (n_correct, n_supervised) as device scalars
    so the Python loop accumulates on-device and syncs once at the end."""
    logits = model(tokens, attn_mask)
    pred = _argmax_over_actions(logits)
    correct = (pred == targets) & loss_mask
    return correct.sum(), loss_mask.sum()


@eqx.filter_jit
def _batch_phase_counts(
    model: EffectiveCallable,
    tokens: Int[Array, "B T"],
    targets: Int[Array, "B T"],
    attn_mask: Bool[Array, "B T"],
    loss_mask: Bool[Array, "B T"],
    phase_masks: Bool[Array, "P T"],
) -> tuple[Int[Array, ""], Int[Array, ""], Int[Array, "P"], Int[Array, "P"]]:
    """JIT'd inner for the per-phase breakdown.

    Returns ``(n_correct, n_supervised, phase_correct, phase_supervised)``
    where the two ``P``-vectors carry the per-phase counts (one entry per
    row of ``phase_masks``). All reductions happen inside the jit so the
    Python loop syncs a single small tuple per chunk instead of eight
    separate ``int()`` host round-trips.
    """
    logits = model(tokens, attn_mask)
    pred = _argmax_over_actions(logits)
    correct = (pred == targets) & loss_mask  # (B, T)
    # Broadcast the (P, T) per-position phase masks against (B, T).
    in_phase = loss_mask[None, :, :] & phase_masks[:, None, :]  # (P, B, T)
    phase_correct = (correct[None, :, :] & phase_masks[:, None, :]).sum(axis=(1, 2))
    phase_supervised = in_phase.sum(axis=(1, 2))
    return correct.sum(), loss_mask.sum(), phase_correct, phase_supervised


def compute_move_accuracy(
    model: EffectiveCallable,
    corpus: Corpus,
    *,
    batch_size: int = 32,
) -> float:
    """Overall move-prediction accuracy on a Corpus.

    Iterates in chunks of ``batch_size``; returns the fraction of
    supervised positions where ``argmax(logits[:NUM_ACTIONS]) ==
    target``. Returns 0.0 if the corpus has no supervised positions.
    """
    n = corpus.n_games
    total_correct: Array = jnp.zeros((), dtype=jnp.int32)
    total_supervised: Array = jnp.zeros((), dtype=jnp.int32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        tokens = jnp.asarray(corpus.tokens[start:end])
        targets = jnp.asarray(corpus.targets[start:end])
        attn = jnp.asarray(corpus.attn_mask[start:end])
        loss = jnp.asarray(corpus.loss_mask[start:end])
        correct, supervised = _batch_correct(model, tokens, targets, attn, loss)
        total_correct = total_correct + correct
        total_supervised = total_supervised + supervised
    # Single host sync after the device-side accumulation.
    total_c = int(total_correct)
    total_s = int(total_supervised)
    if total_s == 0:
        return 0.0
    return total_c / total_s


def compute_per_phase_accuracy(
    model: EffectiveCallable,
    corpus: Corpus,
    *,
    batch_size: int = 32,
    phases: PhaseBoundaries = PhaseBoundaries(),
) -> AccuracyResult:
    """Move accuracy broken down by game phase.

    Phase membership is keyed on the **ply** a position predicts, not its
    raw sequence index. The conditioning prefix (Phase-A Chunk 4) shifts
    every move ``C`` slots to the right, where ``C`` is the prefix width
    persisted as ``corpus.outcome_offset`` (constant across the corpus).
    A position at sequence index ``t`` therefore predicts ply ``t - C``;
    binning on the raw ``t`` would smear the boundaries by ``C`` and
    mislabel the first ``C`` plies as opening padding (plan §8.1).

    A position at ply ``p = t - C`` belongs to:
    - opening if ``p < phases.opening_end``,
    - midgame if ``phases.opening_end <= p < phases.midgame_end``,
    - endgame otherwise.

    Within each phase, only supervised positions contribute (the prefix
    slots are never supervised, so the negative-ply prefix region drops
    out of every phase regardless of how it bins).
    """
    seq_len = corpus.seq_len
    # ``outcome_offset`` is the constant prefix width C for every game in
    # the corpus (the slot where the first move lives). Read it off the
    # corpus rather than hardcoding a default so the binning tracks the
    # checkpoint's own conditioning layout.
    C = int(corpus.outcome_offset[0]) if corpus.n_games > 0 else 1
    positions = np.arange(seq_len)
    ply = positions - C
    opening_pos = ply < phases.opening_end
    midgame_pos = (ply >= phases.opening_end) & (ply < phases.midgame_end)
    endgame_pos = ply >= phases.midgame_end

    n = corpus.n_games

    # Stack the three per-position phase masks into one (P=3, T) tensor —
    # row 0 opening, row 1 midgame, row 2 endgame. The jitted body reduces
    # over (B, T) per phase, so the host only sees small device scalars.
    phase_masks = jnp.asarray(
        np.stack([opening_pos, midgame_pos, endgame_pos], axis=0), dtype=jnp.bool_,
    )

    total_correct: Array = jnp.zeros((), dtype=jnp.int32)
    total_sup: Array = jnp.zeros((), dtype=jnp.int32)
    phase_correct: Array = jnp.zeros((3,), dtype=jnp.int32)
    phase_sup: Array = jnp.zeros((3,), dtype=jnp.int32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        tokens = jnp.asarray(corpus.tokens[start:end])
        targets = jnp.asarray(corpus.targets[start:end])
        attn = jnp.asarray(corpus.attn_mask[start:end])
        loss = jnp.asarray(corpus.loss_mask[start:end])
        c, s, pc, ps = _batch_phase_counts(
            model, tokens, targets, attn, loss, phase_masks,
        )
        total_correct = total_correct + c
        total_sup = total_sup + s
        phase_correct = phase_correct + pc
        phase_sup = phase_sup + ps

    # Single host sync after the device-side accumulation: one transfer of
    # the two scalars plus the two length-3 vectors, instead of eight
    # per-chunk ``int()`` round-trips.
    total_c = int(total_correct)
    total_s = int(total_sup)
    pc_host = np.asarray(phase_correct)
    ps_host = np.asarray(phase_sup)
    o_correct, m_correct, e_correct = (int(x) for x in pc_host)
    o_sup, m_sup, e_sup = (int(x) for x in ps_host)

    def safe_div(num: int, den: int) -> float:
        return num / den if den > 0 else 0.0

    return AccuracyResult(
        overall=safe_div(total_c, total_s),
        opening=safe_div(o_correct, o_sup),
        midgame=safe_div(m_correct, m_sup),
        endgame=safe_div(e_correct, e_sup),
        n_total=total_s,
        n_opening=o_sup,
        n_midgame=m_sup,
        n_endgame=e_sup,
    )


# ---------------------------------------------------------------------------
# Held-out validation pass (pretrain val loop) — v1 CLMTrainer.evaluate parity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValMetrics:
    """Held-out validation metrics for one pass over a val corpus.

    Mirrors the scalar subset v1's ``CLMTrainer.evaluate`` returned
    (``val/loss`` / ``val/accuracy`` (top-1) / ``val/top5_accuracy`` /
    ``val/perplexity`` / ``val/legal_move_rate`` /
    ``val/late_legal_move_rate``) plus the per-phase breakdown. The
    ``best_*`` patience signal in the pretrain loop keys on ``val_loss``
    (lower is better) and ``late_legal_move_rate`` (higher is better).
    """

    val_loss: float
    top1: float
    top5: float
    perplexity: float
    legal_move_rate: float
    late_legal_move_rate: float
    phases: AccuracyResult

    def as_log_kwargs(self) -> dict[str, float]:
        """Flatten to the bare-name kwargs :meth:`MetricsLogger.log_val`
        promotes to the ``val/*`` dashboard keys.

        The logger maps ``loss`` → ``val/loss`` (+ derived
        ``val/perplexity``), ``top1`` → ``val/top1``, ``top5`` →
        ``val/top5`` (+ ``val/top5_accuracy``), ``accuracy`` →
        ``val/accuracy``, ``legal_move_rate`` /
        ``late_legal_move_rate`` / per-phase ``opening`` / ``midgame`` /
        ``endgame`` to their namespaced keys. Passing ``accuracy=top1``
        keeps the v1 ``val/accuracy`` chart fed (v1 reported top-1 as
        ``val/accuracy``).
        """
        return {
            "loss": self.val_loss,
            "accuracy": self.top1,
            "top1": self.top1,
            "top5": self.top5,
            "perplexity": self.perplexity,
            "legal_move_rate": self.legal_move_rate,
            "late_legal_move_rate": self.late_legal_move_rate,
            "opening": self.phases.opening,
            "midgame": self.phases.midgame,
            "endgame": self.phases.endgame,
        }


@eqx.filter_jit
def _batch_val_counts(
    model: EffectiveCallable,
    tokens: Int[Array, "B T"],
    targets: Int[Array, "B T"],
    attn_mask: Bool[Array, "B T"],
    eval_mask: Bool[Array, "B T"],
    loss_mask: Bool[Array, "B T"],
    legal_mask: Bool[Array, "B T A"],
    late_mask: Bool[Array, "B T"],
) -> tuple[
    Float[Array, ""], Int[Array, ""], Int[Array, ""],
    Int[Array, ""], Int[Array, ""], Int[Array, ""], Int[Array, ""],
]:
    """JIT'd inner for the validation pass.

    Returns ``(loss_sum, n_eval, n_loss_sup, n_top1_correct,
    n_top5_correct, n_legal, n_late_legal)`` as device scalars so the
    Python loop accumulates on-device and syncs one small tuple per
    chunk. ``n_eval`` is the MAIA-gated count (denominator for
    loss / top-1 / top-5); ``n_loss_sup`` is the plain supervised count
    (denominator for ``legal_move_rate``, matching the ``loss_mask``-gated
    ``n_legal`` numerator).

    ``eval_mask`` is the ``(B, T)`` set of supervised positions that
    count toward the *MAIA-skipped* loss / top-1 / top-5 stats — i.e.
    ``loss_mask & (ply >= min_eval_ply)``. The MAIA opening-skip (v1
    ``min_eval_ply``, default 10) drops the book-ish opening plies from
    the headline accuracy/loss numbers; the per-phase breakdown is
    computed separately (always from ply 0) so it still reports the full
    picture.

    ``loss_mask`` is the **plain** ``(B, T)`` supervised mask (no MAIA
    gate). Legality (``n_legal`` / ``n_late_legal``) is gated by it — NOT
    by ``eval_mask`` — so the legal-move-rate numerators match the
    denominators ``compute_val_metrics`` derives from the same plain
    supervised mask (``late_sup = (late_pos & loss).sum()``). Gating
    legality by ``eval_mask`` would silently deflate the late rate by the
    fraction of supervised late positions with ply ``< min_eval_ply``.

    ``loss_sum`` is the **sum** of per-position cross-entropy over
    ``eval_mask`` (not the mean) so the host can divide by the global
    eval count after the loop — a per-chunk mean would mis-weight ragged
    final chunks. ``legal_mask`` is the ``(B, T, A)`` per-position legal
    move-token set (True where a token is legal at that position);
    ``n_legal`` counts eval positions whose argmax prediction is legal,
    ``n_late_legal`` restricts that to the late-game positions flagged by
    ``late_mask`` (which carries its own supervised + ply gate).
    """
    logits = model(tokens, attn_mask)
    move_logits = logits[..., :NUM_ACTIONS].astype(jnp.float32)
    # Sum CE over the eval positions (mean is taken on the host).
    per_pos = jax.nn.log_softmax(move_logits, axis=-1)
    tgt = jnp.clip(targets, 0, NUM_ACTIONS - 1)
    gathered = jnp.take_along_axis(per_pos, tgt[..., None], axis=-1)[..., 0]
    loss_sum = jnp.where(eval_mask, -gathered, 0.0).sum()

    pred = jnp.argmax(move_logits, axis=-1)  # (B, T)
    top1_correct = (pred == targets) & eval_mask
    # Top-5 via the target's rank, NOT ``jax.lax.top_k``: the fused top-k
    # kernel requests more shared memory than RDNA3 exposes per CU (the
    # same 64 KB ceiling that OOMs the fused attention path), so it raises
    # ``hipError 98`` on gfx1100. The target is in the top-5 iff strictly
    # fewer than 5 move-token logits exceed the target's own logit — a pure
    # reduction over the vocab axis with no shared-memory kernel.
    tgt_logit = jnp.take_along_axis(move_logits, tgt[..., None], axis=-1)
    n_greater = (move_logits > tgt_logit).sum(axis=-1)  # (B, T)
    in_top5 = (n_greater < 5) & eval_mask

    # Legality: the argmax prediction is legal at that position.
    pred_legal = jnp.take_along_axis(
        legal_mask, pred[..., None], axis=-1
    )[..., 0]
    legal = pred_legal & loss_mask
    late_legal = legal & late_mask

    return (
        loss_sum,
        eval_mask.sum(),
        loss_mask.sum(),
        top1_correct.sum(),
        in_top5.sum(),
        legal.sum(),
        late_legal.sum(),
    )


def _legal_token_grid(corpus: Corpus) -> np.ndarray:
    """Build the ``(N, T, NUM_ACTIONS)`` per-position legal move-token mask.

    The Rust engine replays each game and returns a dense ``(N, max_ply,
    V)`` bool mask where index ``p`` is the legal set of the board state
    *before* ply ``p``. The pretrain corpus lays moves out at slots
    ``[C .. C + game_length)`` with the first move supervised by the last
    prefix slot ``C - 1`` (``pawn.corpus._pack_clm``), so supervised slot
    ``t`` predicts ply ``p = t - (C - 1)``. We therefore shift the
    ply-aligned engine mask right by ``C - 1`` into the sequence frame and
    truncate to the move-token columns ``[0, NUM_ACTIONS)``.

    Positions outside ``[C-1 .. C-1 + game_length)`` are never supervised
    (``loss_mask`` is False there), so their legal-mask rows are
    don't-cares and left all-False.
    """
    n = corpus.n_games
    seq_len = corpus.seq_len
    grid = np.zeros((n, seq_len, NUM_ACTIONS), dtype=np.bool_)
    if n == 0:
        return grid
    import chess_engine

    # The constant prefix width C (== outcome_offset) and the raw move IDs
    # (slots [C .. C + game_length)). Engine wants (N, max_ply) int move
    # tokens + per-game lengths.
    C = int(corpus.outcome_offset[0])
    # The engine's PyO3 binding wants int16 move IDs + int16 game lengths
    # (the engine vocab is the 1980-wide emission space, not the model's
    # 2000-wide table — only the move-token columns [0, NUM_ACTIONS) are
    # read back). The engine emits the legal mask of the board state
    # *before* each move (the set of legal moves to predict at that ply):
    # for a length-L game it emits masks for plies ``0..L-1`` and reads
    # ``move_ids[0..L-1]`` to replay. Pass the whole ``[C .. seq_len)`` move
    # region (PAD past each game's length, which the engine ignores via
    # ``game_lengths``).
    move_ids = corpus.tokens[:, C:seq_len].astype(np.int16)
    n_move_slots = move_ids.shape[1]
    if n_move_slots == 0:
        return grid
    # ``corpus.game_lengths`` is the *untruncated* game length; the corpus
    # keeps only the first ``n_move_slots`` moves, so clamp the length the
    # engine replays to the slots actually present (a game longer than the
    # window contributes legal masks only for its retained, supervised
    # plies). The engine emits the legal mask of the board state *before*
    # each move, so a length-L clamp emits masks for plies ``0..L-1`` and
    # reads ``move_ids[0..L-1]`` — clamp to ``n_move_slots`` (NOT
    # ``n_move_slots - 1``): the corpus supervises plies ``0..capped-1``
    # with ``capped = min(game_length, n_move_slots)``, so the max
    # supervised ply for a full-window game is ``n_move_slots - 1`` and we
    # must emit its mask. ``gl == n_move_slots`` is a legal engine input
    # (the PyO3 bound check is ``gl > max_ply``) and the replay reads
    # ``move_ids[..t]`` only for ``t in 0..gl-1`` (all in range).
    game_lengths = np.minimum(
        corpus.game_lengths, n_move_slots
    ).astype(np.int16)
    # Dense (N, n_move_slots, 1980) legal mask, ply-aligned (index p =
    # legal set of the board before ply p).
    dense = np.asarray(
        chess_engine.compute_legal_token_masks(move_ids, game_lengths, 1980),
        dtype=np.bool_,
    )
    # Shift ply p into sequence slot t = p + (C - 1) and keep only the
    # move-token columns.
    shift = C - 1
    p_count = min(dense.shape[1], seq_len - shift)
    grid[:, shift : shift + p_count, :] = dense[:, :p_count, :NUM_ACTIONS]
    return grid


def compute_val_metrics(
    model: EffectiveCallable,
    corpus: Corpus,
    *,
    batch_size: int = 32,
    late_ply: int = 0,
    min_eval_ply: int = 0,
    compute_legal: bool = True,
) -> ValMetrics:
    """Held-out validation pass — v1 ``CLMTrainer.evaluate`` parity.

    Iterates the corpus in ``batch_size`` chunks, accumulating the CE loss
    sum + top-1 / top-5 / legality counts on-device, then divides once on
    the host. Returns a :class:`ValMetrics` carrying the ``val/*`` scalar
    schema plus the per-phase breakdown.

    Note on ``val_loss`` vs ``train/loss``: the val CE normalises over the
    move-token support (``[0, NUM_ACTIONS)`` = 1968 columns) while the
    training CE normalises over the full logit width minus the reserved
    columns (1980 effective for the uniform V=2000 vocab; all 1980 for the
    factored v1 vocab, where the reserved mask is a no-op). ``val_loss``
    therefore reads slightly lower than ``train/loss`` at equal quality.
    The discrepancy follows v1's eval contract and is identical across
    architectures, so cross-run comparisons stay apples-to-apples.

    ``late_ply`` is the legality late-game threshold (v1's
    ``legality_late_ply`` — positions predicting ply ``>= late_ply`` count
    toward ``late_legal_move_rate``).

    ``min_eval_ply`` is the MAIA opening-skip (v1 ``min_eval_ply``,
    default 0 here for the pretrain val loop; ``scripts/eval_jax.py``
    defaults it to 10 for the MAIA-style accuracy report). Only supervised
    positions predicting ply ``>= min_eval_ply`` contribute to the overall
    loss / top-1 / top-5 / legality numbers — the book-ish opening plies
    are too easy to be informative. The per-phase breakdown is computed
    separately and **always** bins from ply 0, so it still reports the
    opening accuracy for the full picture (v1 ``eval_accuracy.py``:445).

    ``compute_legal=False`` skips the engine replay (legal rates report
    0.0) for callers that only need the loss / accuracy scalars.
    """
    n = corpus.n_games
    seq_len = corpus.seq_len
    C = int(corpus.outcome_offset[0]) if n > 0 else 1

    # Per-position late-game mask: supervised slot t predicts ply
    # p = t - (C - 1); flag p >= late_ply for the late-legality metric.
    positions = np.arange(seq_len, dtype=np.int32)
    ply = positions - (C - 1)
    late_pos = jnp.asarray(ply >= late_ply, dtype=jnp.bool_)  # (T,)
    # MAIA opening-skip: positions predicting ply >= min_eval_ply count
    # toward the overall headline metrics. ply >= 0 always for supervised
    # slots, so min_eval_ply=0 reduces to "every supervised position".
    keep_pos = jnp.asarray(ply >= min_eval_ply, dtype=jnp.bool_)  # (T,)

    legal_grid = (
        _legal_token_grid(corpus)
        if compute_legal
        else np.zeros((n, seq_len, NUM_ACTIONS), dtype=np.bool_)
    )

    loss_sum: Array = jnp.zeros((), dtype=jnp.float32)
    total_sup: Array = jnp.zeros((), dtype=jnp.int32)
    loss_sup: Array = jnp.zeros((), dtype=jnp.int32)
    top1_sum: Array = jnp.zeros((), dtype=jnp.int32)
    top5_sum: Array = jnp.zeros((), dtype=jnp.int32)
    legal_sum: Array = jnp.zeros((), dtype=jnp.int32)
    late_legal_sum: Array = jnp.zeros((), dtype=jnp.int32)
    late_sup: Array = jnp.zeros((), dtype=jnp.int32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        tokens = jnp.asarray(corpus.tokens[start:end])
        targets = jnp.asarray(corpus.targets[start:end])
        attn = jnp.asarray(corpus.attn_mask[start:end])
        loss = jnp.asarray(corpus.loss_mask[start:end])
        legal = jnp.asarray(legal_grid[start:end])
        late_b = jnp.broadcast_to(late_pos, loss.shape) & loss
        # Overall metrics gate on the MAIA opening-skip; late-legality
        # gates on its own ply threshold. Both are intersected with the
        # supervised mask so unpredicted (prefix / PAD) slots never count.
        eval_b = jnp.broadcast_to(keep_pos, loss.shape) & loss
        ls, s, lsup, t1, t5, lg, llg = _batch_val_counts(
            model, tokens, targets, attn, eval_b, loss, legal, late_b,
        )
        loss_sum = loss_sum + ls
        total_sup = total_sup + s
        loss_sup = loss_sup + lsup
        top1_sum = top1_sum + t1
        top5_sum = top5_sum + t5
        legal_sum = legal_sum + lg
        late_legal_sum = late_legal_sum + llg
        # Count of supervised late-game positions (denominator for the
        # late-legality rate). ``late_b`` already carries the supervised
        # gate (``& loss`` above).
        late_sup = late_sup + late_b.sum()

    n_sup = int(total_sup)
    n_loss_sup = int(loss_sup)
    n_late = int(late_sup)
    loss_total = float(loss_sum)
    if n_sup == 0:
        return ValMetrics(
            val_loss=0.0, top1=0.0, top5=0.0, perplexity=1.0,
            legal_move_rate=0.0, late_legal_move_rate=0.0,
            phases=compute_per_phase_accuracy(
                model, corpus, batch_size=batch_size
            ),
        )
    val_loss = loss_total / n_sup
    phases = compute_per_phase_accuracy(model, corpus, batch_size=batch_size)
    return ValMetrics(
        val_loss=val_loss,
        top1=int(top1_sum) / n_sup,
        top5=int(top5_sum) / n_sup,
        perplexity=math.exp(min(val_loss, 20.0)),
        # ``legal_sum`` is gated by the plain supervised mask (no MAIA
        # skip), so its denominator is the plain supervised count — not
        # the MAIA-gated ``n_sup`` used for loss / top-1 / top-5.
        legal_move_rate=(
            int(legal_sum) / n_loss_sup if n_loss_sup > 0 else 0.0
        ),
        late_legal_move_rate=(
            int(late_legal_sum) / n_late if n_late > 0 else 0.0
        ),
        phases=phases,
    )


@dataclass(frozen=True)
class CompoundLegalityResult:
    """Teacher-forced compound (game-completion) legality.

    The v1 "game completion rate" (``docs/LEGACY.md`` / ``ARCHITECTURE.md``):
    a game *completes* iff EVERY supervised ply's argmax move prediction is
    legal — i.e. the model would play the whole game without a single illegal
    move, given the ground-truth history at each ply (non-autoregressive /
    teacher-forced). Compounds per-move legality the way real generation
    would, so it depends far more strongly on model capacity than per-move
    accuracy does.

    ``n_games`` is the number of *evaluated* games (those with >=1 supervised
    ply) — the denominator of ``game_completion_rate``. It equals the corpus
    size at ``min_eval_ply=0``; with a larger skip, games shorter than the
    skip are excluded.
    """

    game_completion_rate: float
    per_move_legal_rate: float
    n_games: int


@eqx.filter_jit
def _batch_game_legal(
    model: EffectiveCallable,
    tokens: Int[Array, "B T"],
    attn_mask: Bool[Array, "B T"],
    sup_mask: Bool[Array, "B T"],
    legal_grid: Bool[Array, "B T A"],
) -> tuple[Int[Array, ""], Int[Array, ""], Int[Array, ""], Int[Array, ""]]:
    """Per-batch game-completion + per-move legality counts.

    Returns ``(complete_games, evaluated_games, legal_moves, supervised_moves)``.
    A game is *evaluated* iff it has at least one supervised position; it
    *completes* iff it is evaluated AND none of its supervised positions has an
    illegal argmax prediction. Games with zero supervised positions (e.g.
    shorter than ``min_eval_ply``) are excluded from BOTH the completed and the
    evaluated counts, so they cannot inflate the rate by vacuous truth
    (``~illegal_sup.any()`` is ``True`` for an all-False ``sup_mask`` row).
    """
    logits = model(tokens, attn_mask)
    pred = _argmax_over_actions(logits)  # (B, T)
    # Is the argmax action legal at each position? ``legal_grid`` is
    # (B, T, NUM_ACTIONS) bool; gather the predicted action's column.
    pred_legal = jnp.take_along_axis(
        legal_grid, pred[..., None], axis=-1
    )[..., 0]  # (B, T)
    illegal_sup = sup_mask & (~pred_legal)
    has_sup = sup_mask.any(axis=-1)  # (B,) — game has >=1 evaluated ply
    game_all_legal = ~illegal_sup.any(axis=-1)  # (B,)
    complete = (game_all_legal & has_sup).sum()
    evaluated = has_sup.sum()
    legal_moves = (pred_legal & sup_mask).sum()
    return complete, evaluated, legal_moves, sup_mask.sum()


def compute_compound_legality(
    model: EffectiveCallable,
    corpus: Corpus,
    *,
    batch_size: int = 32,
    min_eval_ply: int = 0,
) -> CompoundLegalityResult:
    """Teacher-forced game-completion rate (v1 "game completion rate").

    For each game the model predicts the move at every supervised ply given
    the ground-truth history; the game *completes* iff EVERY such argmax
    prediction is legal. ``game_completion_rate`` is the fraction of
    *evaluated* games (those with >=1 supervised ply) that complete. Also
    returns ``per_move_legal_rate`` (the per-position legal rate over the same
    supervised positions).

    ``min_eval_ply`` skips opening plies ``< min_eval_ply``; ``0`` counts every
    supervised ply. Game-completion is conventionally a no-opening-skip metric
    (v1 measured it over every supervised ply — ``eval_jax.py`` always calls
    this with ``min_eval_ply=0``), so prefer ``0``. Two notes for
    ``min_eval_ply > 0``: (1) games with no surviving supervised ply are
    excluded from both the numerator and denominator (no vacuous completions);
    (2) ``per_move_legal_rate`` then equals :func:`compute_val_metrics`'s
    ``legal_move_rate`` ONLY at ``min_eval_ply=0`` — ``legal_move_rate`` is
    computed over the plain (un-skipped) supervised mask.
    """
    n = corpus.n_games
    if n == 0:
        return CompoundLegalityResult(0.0, 0.0, 0)
    seq_len = corpus.seq_len
    C = int(corpus.outcome_offset[0])
    ply = np.arange(seq_len, dtype=np.int32) - (C - 1)
    keep_pos = jnp.asarray(ply >= min_eval_ply, dtype=jnp.bool_)  # (T,)
    legal_grid = _legal_token_grid(corpus)

    games_complete = 0
    games_evaluated = 0
    legal_moves = 0
    total_moves = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        tokens = jnp.asarray(corpus.tokens[start:end])
        attn = jnp.asarray(corpus.attn_mask[start:end])
        loss = jnp.asarray(corpus.loss_mask[start:end])
        legal = jnp.asarray(legal_grid[start:end])
        sup = loss & jnp.broadcast_to(keep_pos, loss.shape)
        complete, evaluated, legal_sup, sup_count = _batch_game_legal(
            model, tokens, attn, sup, legal
        )
        games_complete += int(complete)
        games_evaluated += int(evaluated)
        legal_moves += int(legal_sup)
        total_moves += int(sup_count)
    return CompoundLegalityResult(
        game_completion_rate=(
            games_complete / games_evaluated if games_evaluated else 0.0
        ),
        per_move_legal_rate=legal_moves / total_moves if total_moves else 0.0,
        n_games=games_evaluated,
    )


@eqx.filter_jit
def _batch_per_position_top1(
    model: EffectiveCallable,
    tokens: Int[Array, "B T"],
    targets: Int[Array, "B T"],
    attn_mask: Bool[Array, "B T"],
    loss_mask: Bool[Array, "B T"],
) -> tuple[Int[Array, "T"], Int[Array, "T"]]:
    """JIT'd inner for the per-ply breakdown.

    Returns ``(per_pos_correct, per_pos_supervised)`` — two length-``T``
    vectors summed over the batch axis, so the host accumulates the
    per-sequence-position counts and re-bins them by ply once at the end.
    """
    logits = model(tokens, attn_mask)
    pred = _argmax_over_actions(logits)
    correct = (pred == targets) & loss_mask  # (B, T)
    return correct.sum(axis=0), loss_mask.sum(axis=0)


@dataclass(frozen=True)
class PerPlyResult:
    """Per-ply top-1 accuracy breakdown (v1 ``--per-ply``).

    ``accuracy[p]`` / ``n[p]`` give the top-1 accuracy and supervised
    position count at ply ``p``. Plies are 0-indexed (ply 0 is the first
    move), keyed off the conditioning offset ``C`` so the report tracks
    the checkpoint's own layout rather than the raw sequence index.
    """

    accuracy: dict[int, float]
    n: dict[int, int]


def compute_per_ply_accuracy(
    model: EffectiveCallable,
    corpus: Corpus,
    *,
    batch_size: int = 32,
) -> PerPlyResult:
    """Top-1 accuracy broken down by ply (v1 ``eval_accuracy.py`` ``--per-ply``).

    Supervised slot ``t`` predicts ply ``p = t - (C - 1)`` (the first move
    is supervised by the last conditioning slot ``C - 1``); this bins the
    per-position top-1 counts by ``p`` and reports the accuracy for every
    ply with at least one supervised position. Plies with no supervised
    positions are omitted (matching v1, which only emits seen plies).
    """
    n = corpus.n_games
    seq_len = corpus.seq_len
    C = int(corpus.outcome_offset[0]) if n > 0 else 1
    per_pos_correct: Array = jnp.zeros((seq_len,), dtype=jnp.int32)
    per_pos_sup: Array = jnp.zeros((seq_len,), dtype=jnp.int32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        tokens = jnp.asarray(corpus.tokens[start:end])
        targets = jnp.asarray(corpus.targets[start:end])
        attn = jnp.asarray(corpus.attn_mask[start:end])
        loss = jnp.asarray(corpus.loss_mask[start:end])
        c, s = _batch_per_position_top1(model, tokens, targets, attn, loss)
        per_pos_correct = per_pos_correct + c
        per_pos_sup = per_pos_sup + s
    correct_host = np.asarray(per_pos_correct)
    sup_host = np.asarray(per_pos_sup)
    accuracy: dict[int, float] = {}
    counts: dict[int, int] = {}
    for t in range(seq_len):
        s = int(sup_host[t])
        if s == 0:
            continue
        ply = t - (C - 1)
        accuracy[ply] = int(correct_host[t]) / s
        counts[ply] = s
    return PerPlyResult(accuracy=accuracy, n=counts)
