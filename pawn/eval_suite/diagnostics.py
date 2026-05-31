"""Edge-case diagnostics — guaranteed coverage via `engine.edge_case_bits()`.

The Rust engine returns per-position bits that flag edge cases
(in_check / double_check / pin_restricts / ep_available / castle_legal_*
/ castle_blocked_check / promotion_available / checkmate / stalemate).
The diagnostic computes move accuracy on positions matching each bit so
the eval has guaranteed coverage of the rare cases.

Two corpus sources:

- :func:`compute_edge_case_accuracy` accepts a pre-existing
  ``(move_ids, game_lengths)`` corpus from
  ``engine.generate_random_games`` or the Lichess parquet path. Useful
  when the caller has already chosen the game pool.
- :func:`compute_edge_case_accuracy_quota` calls
  ``engine.generate_diagnostic_sets`` (v1 parity), which uses quota
  sampling to *guarantee* coverage for every label rather than hoping
  random games happen to surface them. This is the path
  ``scripts/eval_generation_jax.py`` uses when the operator passes
  ``--edge-cases``.

Both paths share the same per-bit accuracy computation and return a
list of :class:`EdgeCaseResult`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import chess_engine as engine
from pawn.config import NUM_ACTIONS, PAD_TOKEN, VOCAB_SIZE
from pawn.corpus import (
    _map_termination_to_outcome,
    build_prefix,
    conditioning_to_C,
)
from pawn.model import EffectiveCallable, PAWNModel

__all__ = [
    "EdgeCaseResult",
    "SampledDiagnosticResult",
    "EDGE_CASE_LABELS",
    "TERMINAL_LABELS",
    "compute_edge_case_accuracy",
    "compute_edge_case_accuracy_quota",
    "compute_edge_case_diagnostics",
    "compute_edge_case_diagnostics_quota",
    "default_diagnostic_quotas",
]


# Canonical v2 edge-case labels — full v1 parity per
# docs/JAX_PARITY_SHORTFALLS.md §11. Each maps to a bit-mask name in
# the Rust engine's ``edge_case_bits()`` dict. The user-facing label
# keeps the v1 snake_case names per plan §10 S8.
EDGE_CASE_LABELS = (
    "in_check",
    "double_check",
    "pin_restricts",
    "ep_available",
    "castle_legal_kingside",
    "castle_legal_queenside",
    "castle_blocked_check",
    "promotion_available",
    "checkmate",
    "stalemate",
)

# Mapping label → engine bit-mask name. The four labels added in
# parity #6 (`castle_blocked_check`, `promotion_available`, `checkmate`,
# `stalemate`) map to the matching bits already exported by the engine.
_LABEL_TO_BIT_NAME = {
    "in_check": "IN_CHECK",
    "double_check": "IN_DOUBLE_CHECK",
    "pin_restricts": "PIN_RESTRICTS_MOVEMENT",
    "ep_available": "EP_CAPTURE_AVAILABLE",
    "castle_legal_kingside": "CASTLE_LEGAL_KINGSIDE",
    "castle_legal_queenside": "CASTLE_LEGAL_QUEENSIDE",
    "castle_blocked_check": "CASTLE_BLOCKED_CHECK",
    "promotion_available": "PROMOTION_AVAILABLE",
    "checkmate": "CHECKMATE",
    "stalemate": "STALEMATE",
}

# Terminal labels live exclusively at the post-final ply (no move follows),
# so their meaningful sampled metric is the PAD ("game-is-over") probability
# rather than the legal-move rate. v1 (`eval_suite/diagnostics.py:331`)
# printed `pad_prob` for these and `legal_rate` for the rest; the model-card
# `format_diagnostic` consumer keys off the same split.
TERMINAL_LABELS = ("checkmate", "stalemate")

# `generate_diagnostic_sets` expects 64-element int32 quota arrays
# indexed by bit position. The engine's `edge_case_bits()` returns the
# bit *value* (1 << position); we convert.
def _quota_index_for_label(label: str, bit_table: dict[str, int]) -> int:
    bit_name = _LABEL_TO_BIT_NAME[label]
    val = int(bit_table[bit_name])
    # bit position = log2(value) since each edge-case constant is a
    # single set bit.
    return int(val.bit_length()) - 1


def default_diagnostic_quotas(
    per_label: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(quotas_white, quotas_black)`` arrays for
    :func:`compute_edge_case_accuracy_quota` from a per-label count.

    Same ``per_label`` applied to both colours so we get balanced
    coverage. Returns two int32 arrays of length 64 (the engine's
    contract; positions other than our 10 labels are zero).

    **Coverage vs v1.** v1's :func:`generate_diagnostic_corpus` defaulted
    to ``n_per_category=10_000`` (split ~5000/colour). The v2 default
    ``per_label=10`` is a deliberate ~1000× reduction so the suite stays
    fast on a small backbone — the operator raises ``--edge-per-label``
    (or ``per_label=``) when production-scale coverage is wanted. The
    quota mechanism still *guarantees* every label is non-empty; only the
    per-label sample count is smaller, so the rare-label metrics carry
    more variance at the default than v1 did.
    """
    bit_table = engine.edge_case_bits()
    quotas = np.zeros(64, dtype=np.int32)
    for label in EDGE_CASE_LABELS:
        quotas[_quota_index_for_label(label, bit_table)] = per_label
    return quotas, quotas.copy()


@dataclass(frozen=True)
class EdgeCaseResult:
    label: str
    accuracy: float
    n_positions: int


@dataclass(frozen=True)
class SampledDiagnosticResult:
    """Per-category distributional diagnostic — the v1 sampled-metrics
    surface (``eval_suite/diagnostics.py:343-351``) the model-card and viz
    consumers expect.

    v1 drew ``n_samples`` moves from the softmax at each diagnostic
    position and reported the empirical legal-move fraction. v2 computes
    the *analytic* legal-move probability mass (the softmax mass on the
    engine's legal-token set at that position) instead — the exact
    expectation v1's multinomial estimated, with zero sampling noise. The
    ``mean_legal_rate`` / ``std_legal_rate`` / ``mean_pad_prob`` /
    ``mean_entropy`` / ``std_entropy`` / ``terminal`` / ``n_positions``
    keys match v1 one-for-one so :func:`scripts.generate_model_cards.
    format_diagnostic` and :func:`pawn.eval_suite.viz.plot_diagnostic_results`
    consume the same schema.
    """

    label: str
    n_positions: int
    terminal: bool
    mean_legal_rate: float
    std_legal_rate: float
    mean_pad_prob: float
    mean_entropy: float
    std_entropy: float

    def to_dict(self) -> dict[str, float | int | bool]:
        """The v1 model-card / viz consumer dict for this category."""
        return {
            "n_positions": self.n_positions,
            "terminal": self.terminal,
            "mean_legal_rate": self.mean_legal_rate,
            "std_legal_rate": self.std_legal_rate,
            "mean_pad_prob": self.mean_pad_prob,
            "mean_entropy": self.mean_entropy,
            "std_entropy": self.std_entropy,
        }


def _resolve_outcome_tokens(
    term_codes: np.ndarray | None,
    game_lengths: np.ndarray,
) -> np.ndarray:
    """Per-game outcome token IDs for the ``"outcome"`` conditioning slot.

    Maps the engine's termination codes through the same
    :func:`pawn.corpus._map_termination_to_outcome` the trainer uses, so
    the diagnostic prefix carries the exact outcome token the model was
    conditioned on. When ``term_codes`` is unavailable the games carry no
    resolvable outcome; we fall back to a ``< 0`` sentinel that
    :func:`pawn.corpus.build_prefix` resolves to ``NULL_TOKEN`` (only
    consumed when ``"outcome"`` is actually in the conditioning list).
    """
    n = int(np.asarray(game_lengths).shape[0])
    if term_codes is None:
        return np.full(n, -1, dtype=np.int32)
    return _map_termination_to_outcome(
        np.asarray(term_codes), np.asarray(game_lengths)
    )


def _build_diagnostic_tokens(
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    outcome_tokens: np.ndarray,
    *,
    conditioning: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Assemble the ``[BOS][cond…][move…][PAD…]`` diagnostic tokens.

    Used by :func:`_compute_per_bit_all` so the accuracy and sampled
    surfaces score the model on the identical in-distribution
    absolute-RoPE layout (``move[j]`` at slot ``C + j``, one trailing
    predict-PAD slot). Returns ``(tokens, attn, C)``.
    """
    n, max_ply = move_ids.shape
    C = conditioning_to_C(conditioning)
    move_ids = move_ids.astype(np.int32)
    game_lengths = np.asarray(game_lengths, dtype=np.int32)

    # One trailing slot past the last move holds the predict-PAD (terminal)
    # target, so the terminal ply j = game_length is representable for every
    # game whose terminal bits fit (game_length < max_ply, the engine guard).
    seq_len = C + max_ply
    tokens = np.full((n, seq_len), PAD_TOKEN, dtype=np.int32)
    tokens[:, :C] = build_prefix(conditioning, outcome_tokens, n)
    move_positions = np.arange(max_ply, dtype=np.int32)[None]
    valid_move = move_positions < game_lengths[:, None]
    tokens[:, C:] = np.where(valid_move, move_ids, PAD_TOKEN)
    attn = tokens != PAD_TOKEN
    return tokens, attn, C


def _align_bits_to_slots(
    bits: np.ndarray, n: int, seq_len: int, C: int
) -> np.ndarray:
    """Map per-ply ``bits[:, j]`` onto sequence slots ``s = C - 1 + j``.

    The prediction at slot ``s`` concerns ply ``j = s + 1 - C``, so a
    per-ply edge-case bit attaches to slot ``s = C - 1 + j`` (H5). Returns
    an ``(n, seq_len)`` array zero-padded outside the valid window.
    """
    max_ply = bits.shape[1]
    aligned = np.zeros((n, seq_len), dtype=bits.dtype)
    aligned[:, C - 1 : C - 1 + max_ply] = bits
    return aligned


def _accuracy_results_from_slots(
    correct_slot: np.ndarray,
    aligned_bits: np.ndarray,
    bit_table: dict[str, int],
) -> list[EdgeCaseResult]:
    """Reduce per-slot correctness to one :class:`EdgeCaseResult` per label."""
    results: list[EdgeCaseResult] = []
    for label in EDGE_CASE_LABELS:
        mask_value = bit_table[_LABEL_TO_BIT_NAME[label]]
        mask = (aligned_bits & mask_value).astype(bool)
        n_pos = int(mask.sum())
        if n_pos == 0:
            results.append(EdgeCaseResult(label=label, accuracy=0.0, n_positions=0))
            continue
        n_correct = int((correct_slot & mask).sum())
        results.append(
            EdgeCaseResult(label=label, accuracy=n_correct / n_pos, n_positions=n_pos)
        )
    return results


def _sampled_results_from_slots(
    legal_rate: np.ndarray,
    pad_prob: np.ndarray,
    entropy: np.ndarray,
    aligned_bits: np.ndarray,
    bit_table: dict[str, int],
) -> list[SampledDiagnosticResult]:
    """Reduce per-slot distributional metrics to one
    :class:`SampledDiagnosticResult` per label."""
    results: list[SampledDiagnosticResult] = []
    for label in EDGE_CASE_LABELS:
        terminal = label in TERMINAL_LABELS
        mask_value = bit_table[_LABEL_TO_BIT_NAME[label]]
        mask = (aligned_bits & mask_value).astype(bool)
        n_pos = int(mask.sum())
        if n_pos == 0:
            results.append(
                SampledDiagnosticResult(
                    label=label, n_positions=0, terminal=terminal,
                    mean_legal_rate=0.0, std_legal_rate=0.0,
                    mean_pad_prob=0.0, mean_entropy=0.0, std_entropy=0.0,
                )
            )
            continue
        lr = legal_rate[mask]
        pp = pad_prob[mask]
        ent = entropy[mask]
        results.append(
            SampledDiagnosticResult(
                label=label,
                n_positions=n_pos,
                terminal=terminal,
                mean_legal_rate=float(lr.mean()),
                std_legal_rate=float(lr.std()),
                mean_pad_prob=float(pp.mean()),
                mean_entropy=float(ent.mean()),
                std_entropy=float(ent.std()),
            )
        )
    return results


def _compute_per_bit_all(
    model: "PAWNModel | EffectiveCallable",
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    bits: np.ndarray,
    outcome_tokens: np.ndarray,
    *,
    conditioning: Sequence[str],
    batch_size: int,
    want_accuracy: bool = True,
    want_sampled: bool = True,
) -> tuple[list[EdgeCaseResult] | None, list[SampledDiagnosticResult] | None]:
    """Single-forward-pass per-bit core for both the argmax-accuracy and
    the v1 sampled distributional surfaces.

    ``model(t, a)`` runs **once** per batch; every metric is derived from
    the one ``logits`` array. ``want_accuracy`` / ``want_sampled`` gate
    which metric is materialised so the accuracy-only callers
    (:func:`compute_edge_case_accuracy` and its quota variant) don't pay
    the sampled-metrics cost (softmax + the ``(n, seq_len, VOCAB_SIZE)``
    legal-mask reduction), while the combined
    :func:`compute_edge_case_diagnostics` paths get both surfaces from a
    single sweep over the corpus.

    ``bits`` is the ``(n, max_ply)`` uint64 from
    :func:`chess_engine.compute_edge_stats_per_ply`: ``bits[g, j]`` is
    the edge-case bitfield of the position the model plays ``move[j]``
    *from*, for ``j < game_length[g]``; ``bits[g, game_length[g]]`` is
    the *terminal* position reached after the last move (checkmate /
    stalemate live exclusively there). See ``engine/src/edgestats.rs``
    (``compute_edge_stats_per_ply`` writes ``length + 1`` plies).

    Layout (Phase-A): tokens are assembled with the run's conditioning
    prefix ``[BOS][cond…]`` of width ``C = 1 + len(conditioning)`` so the
    model sees the same absolute-RoPE layout it trained under; ``move[j]``
    lands at sequence slot ``C + j`` and is predicted at slot ``C + j - 1``
    (target ``tokens[:, C + j]``).

    Per-bit alignment (H5): the prediction made at slot ``s`` concerns
    ply ``j = s + 1 - C`` (the move it predicts is ``move[j]``), so each
    seq-slot metric is matched against ``bits[:, j]`` — i.e. ``bits``
    shifted right by ``C`` along the sequence axis. The terminal ply
    ``j = game_length`` has a PAD target (no move follows the final move);
    accuracy scores whether the model predicts the game is over there
    (``argmax`` over the *full* vocab equals PAD) rather than scoring a
    move-argmax against PAD, which previously forced the checkmate /
    stalemate labels to ``accuracy=0`` (the attn mask zeroed that slot).
    """
    tokens, attn, C = _build_diagnostic_tokens(
        move_ids, game_lengths, outcome_tokens, conditioning=conditioning
    )
    n, seq_len = tokens.shape
    bit_table = engine.edge_case_bits()

    # Accuracy scaffold: target at slot s is tokens[:, s + 1]; the PAD
    # trailing slot is benign (the per-bit masks decide which slots count).
    targets = np.full_like(tokens, PAD_TOKEN)
    targets[:, :-1] = tokens[:, 1:]

    # Sampled scaffold: engine legal-token masks per ply, aligned ply j →
    # slot s = C - 1 + j to match the prediction layout. Built lazily so
    # accuracy-only callers don't allocate the (n, seq_len, VOCAB_SIZE) mask.
    legal_slots: np.ndarray | None = None
    if want_sampled:
        move_ids_i16 = np.ascontiguousarray(move_ids, dtype=np.int16)
        game_lengths_i16 = np.asarray(game_lengths, dtype=np.int16)
        legal_per_ply = np.asarray(
            engine.compute_legal_token_masks(
                move_ids_i16, game_lengths_i16, VOCAB_SIZE
            ),
            dtype=bool,
        )
        max_ply = legal_per_ply.shape[1]
        legal_slots = np.zeros((n, seq_len, VOCAB_SIZE), dtype=bool)
        legal_slots[:, C - 1 : C - 1 + max_ply, :] = legal_per_ply

    action_pred_chunks: list[np.ndarray] = []
    full_pred_chunks: list[np.ndarray] = []
    legal_rate_chunks: list[np.ndarray] = []
    pad_prob_chunks: list[np.ndarray] = []
    entropy_chunks: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        t = jnp.asarray(tokens[start:end])
        a = jnp.asarray(attn[start:end])
        logits = model(t, a)  # one forward pass — both surfaces derive from this
        if want_accuracy:
            action_pred_chunks.append(
                np.asarray(jnp.argmax(logits[..., :NUM_ACTIONS], axis=-1))
            )
            full_pred_chunks.append(np.asarray(jnp.argmax(logits, axis=-1)))
        if want_sampled:
            assert legal_slots is not None
            probs = jax.nn.softmax(logits, axis=-1)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            entropy = -jnp.sum(probs * log_probs, axis=-1)
            pad_prob = probs[..., PAD_TOKEN]
            legal_mask = jnp.asarray(legal_slots[start:end])
            legal_rate = jnp.sum(jnp.where(legal_mask, probs, 0.0), axis=-1)
            legal_rate_chunks.append(np.asarray(legal_rate))
            pad_prob_chunks.append(np.asarray(pad_prob))
            entropy_chunks.append(np.asarray(entropy))

    # Align bits onto sequence slots: slot s scores ply j = s + 1 - C.
    aligned_bits = _align_bits_to_slots(bits, n, seq_len, C)

    accuracy_results: list[EdgeCaseResult] | None = None
    if want_accuracy:
        action_pred = np.concatenate(action_pred_chunks, axis=0)
        full_pred = np.concatenate(full_pred_chunks, axis=0)
        # Per-slot correctness. For a move target (the common case) the model
        # must argmax the right action; for the terminal predict-PAD slot it
        # must argmax PAD over the full vocab (game-is-over). The terminal slot
        # is where targets == PAD inside the supervised window; everywhere else
        # uses the action argmax.
        is_pad_target = targets == PAD_TOKEN
        move_correct = (action_pred == targets) & ~is_pad_target
        terminal_correct = (full_pred == PAD_TOKEN) & is_pad_target
        correct_slot = move_correct | terminal_correct
        accuracy_results = _accuracy_results_from_slots(
            correct_slot, aligned_bits, bit_table
        )

    sampled_results: list[SampledDiagnosticResult] | None = None
    if want_sampled:
        legal_rate = np.concatenate(legal_rate_chunks, axis=0)
        pad_prob = np.concatenate(pad_prob_chunks, axis=0)
        entropy = np.concatenate(entropy_chunks, axis=0)
        sampled_results = _sampled_results_from_slots(
            legal_rate, pad_prob, entropy, aligned_bits, bit_table
        )

    return accuracy_results, sampled_results


def _compute_per_bit_accuracy(
    model: "PAWNModel | EffectiveCallable",
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    bits: np.ndarray,
    outcome_tokens: np.ndarray,
    *,
    conditioning: Sequence[str],
    batch_size: int,
) -> list[EdgeCaseResult]:
    """Accuracy-only delegator over :func:`_compute_per_bit_all`.

    Lets :func:`compute_edge_case_accuracy` and
    :func:`compute_edge_case_accuracy_quota` get the per-bit argmax
    accuracy without paying the sampled-metrics forward-pass derivations
    (softmax + legal-mask reduction). Runs the model once per batch.
    """
    accuracy, _ = _compute_per_bit_all(
        model, move_ids, game_lengths, bits, outcome_tokens,
        conditioning=conditioning, batch_size=batch_size,
        want_accuracy=True, want_sampled=False,
    )
    assert accuracy is not None
    return accuracy


def compute_edge_case_accuracy(
    model: "PAWNModel | EffectiveCallable",
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    *,
    term_codes: np.ndarray | None = None,
    conditioning: Sequence[str] = (),
    batch_size: int = 16,
) -> list[EdgeCaseResult]:
    """Compute per-edge-case move accuracy on a pre-existing corpus.

    ``move_ids`` is ``(N, max_ply) int16`` from
    ``engine.generate_random_games`` (or the Lichess parquet path);
    ``game_lengths`` is ``(N,)``. The function calls
    ``engine.compute_edge_stats_per_ply`` to flag positions, runs the
    model's forward pass, and computes per-bit argmax accuracy.

    ``conditioning`` is the checkpoint's prefix-kind list (Phase-A); the
    tokens are assembled with that ``[BOS][cond…]`` prefix so the model
    runs on its in-distribution absolute-RoPE layout. ``term_codes`` (the
    engine's per-game termination codes) resolves the ``"outcome"``
    conditioning slot when present; with no ``"outcome"`` kind in
    ``conditioning`` it is unused and may be omitted.

    Returns a list of :class:`EdgeCaseResult`, one per label in
    :data:`EDGE_CASE_LABELS`. Rare edge cases (checkmate, stalemate)
    may have ``n_positions == 0`` on a small random corpus — use
    :func:`compute_edge_case_accuracy_quota` for guaranteed coverage.
    """
    move_ids = np.ascontiguousarray(move_ids, dtype=np.int16)
    game_lengths = np.asarray(game_lengths, dtype=np.int16)
    bits, _, _ = engine.compute_edge_stats_per_ply(move_ids, game_lengths)
    outcome_tokens = _resolve_outcome_tokens(term_codes, game_lengths)
    return _compute_per_bit_accuracy(
        model, move_ids, game_lengths, bits, outcome_tokens,
        conditioning=conditioning, batch_size=batch_size,
    )


def compute_edge_case_accuracy_quota(
    model: "PAWNModel | EffectiveCallable",
    *,
    per_label: int = 10,
    max_ply: int = 256,
    seed: int = 42,
    max_simulated_factor: float = 500.0,
    conditioning: Sequence[str] = (),
    batch_size: int = 16,
) -> list[EdgeCaseResult]:
    """Compute per-edge-case move accuracy with quota-controlled coverage.

    Calls :func:`chess_engine.generate_diagnostic_sets` with a flat
    ``per_label`` quota per (colour, label), so every label is
    *guaranteed* coverage (subject to ``max_simulated_factor * total_games``
    being enough simulated games). The v1 parity path per
    docs/JAX_PARITY_SHORTFALLS.md §11; without quota control the rare
    labels (checkmate, stalemate) silently report ``accuracy=0,
    n_positions=0`` on a small random pool.

    The default ``max_simulated_factor=500`` is tuned so the rarest v1
    label — ``stalemate`` — reliably fills its quota at
    ``per_label=10`` (empirically the engine needs ~400 simulated
    games per stalemate game; smaller factors silently truncate the
    rare-label coverage).
    """
    (
        move_ids_np, game_lengths_np, bits, outcome_tokens, _fill
    ) = _generate_diagnostic_corpus(
        per_label=per_label, max_ply=max_ply, seed=seed,
        max_simulated_factor=max_simulated_factor, report=False,
    )
    return _compute_per_bit_accuracy(
        model, move_ids_np, game_lengths_np, bits, outcome_tokens,
        conditioning=conditioning, batch_size=batch_size,
    )


def _generate_diagnostic_corpus(
    *,
    per_label: int,
    max_ply: int,
    seed: int,
    max_simulated_factor: float,
    report: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, tuple[int, int]]]:
    """Run the engine's quota-controlled diagnostic generator once.

    Returns ``(move_ids, game_lengths, bits, outcome_tokens, fill)`` where
    ``fill`` maps each label to ``(filled, requested)`` quota counts. When
    ``report`` is True, prints the per-label ``OK`` / ``SHORT`` fill-rate
    table — v1 parity (`eval_suite/diagnostics.py:93-99`).
    """
    quotas_w, quotas_b = default_diagnostic_quotas(per_label=per_label)
    # `total_games` is the number of distinct accepted games; the engine
    # pads the corpus to satisfy the per-label quotas.
    total_games = int(quotas_w.sum() + quotas_b.sum())
    output = engine.generate_diagnostic_sets(
        quotas_w, quotas_b, total_games, max_ply, seed, max_simulated_factor,
    )
    move_ids, game_lengths, term_codes, per_ply_stats = output[:4]
    filled_white = np.asarray(output[8], dtype=np.int64)
    filled_black = np.asarray(output[9], dtype=np.int64)
    move_ids_np = np.asarray(move_ids, dtype=np.int16)
    game_lengths_np = np.asarray(game_lengths, dtype=np.int16)
    bits = np.asarray(per_ply_stats, dtype=np.uint64)
    outcome_tokens = _resolve_outcome_tokens(
        np.asarray(term_codes), game_lengths_np
    )

    bit_table = engine.edge_case_bits()
    fill: dict[str, tuple[int, int]] = {}
    for label in EDGE_CASE_LABELS:
        idx = _quota_index_for_label(label, bit_table)
        filled = int(filled_white[idx]) + int(filled_black[idx])
        requested = int(quotas_w[idx]) + int(quotas_b[idx])
        fill[label] = (filled, requested)

    if report:
        print(
            f"  Diagnostic corpus: {len(game_lengths_np)} games, "
            f"per_label={per_label}, max_simulated_factor={max_simulated_factor}"
        )
        for label in EDGE_CASE_LABELS:
            filled, requested = fill[label]
            pct = filled / requested * 100 if requested > 0 else 0.0
            status = "OK" if filled >= requested else "SHORT"
            print(f"    {label}: {filled}/{requested} ({pct:.0f}%) [{status}]")

    return move_ids_np, game_lengths_np, bits, outcome_tokens, fill


def compute_edge_case_diagnostics(
    model: "PAWNModel | EffectiveCallable",
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    *,
    term_codes: np.ndarray | None = None,
    conditioning: Sequence[str] = (),
    batch_size: int = 16,
) -> tuple[list[EdgeCaseResult], list[SampledDiagnosticResult]]:
    """Both the argmax-accuracy *and* the v1 sampled distributional
    diagnostics on a pre-existing corpus.

    Returns ``(accuracy_results, sampled_results)`` — the superset
    accuracy metric (v2) alongside the v1
    ``mean_legal_rate`` / ``mean_pad_prob`` / ``mean_entropy`` surface the
    model-card and viz consumers expect. Both lists are one entry per
    :data:`EDGE_CASE_LABELS` label, in canonical order.
    """
    move_ids = np.ascontiguousarray(move_ids, dtype=np.int16)
    game_lengths = np.asarray(game_lengths, dtype=np.int16)
    bits, _, _ = engine.compute_edge_stats_per_ply(move_ids, game_lengths)
    bits = np.asarray(bits, dtype=np.uint64)
    outcome_tokens = _resolve_outcome_tokens(term_codes, game_lengths)
    accuracy, sampled = _compute_per_bit_all(
        model, move_ids, game_lengths, bits, outcome_tokens,
        conditioning=conditioning, batch_size=batch_size,
        want_accuracy=True, want_sampled=True,
    )
    assert accuracy is not None and sampled is not None
    return accuracy, sampled


def compute_edge_case_diagnostics_quota(
    model: "PAWNModel | EffectiveCallable",
    *,
    per_label: int = 10,
    max_ply: int = 256,
    seed: int = 42,
    max_simulated_factor: float = 500.0,
    conditioning: Sequence[str] = (),
    batch_size: int = 16,
    report: bool = False,
) -> tuple[list[EdgeCaseResult], list[SampledDiagnosticResult]]:
    """Quota-controlled coverage variant of
    :func:`compute_edge_case_diagnostics`.

    Generates a corpus with *guaranteed* per-label coverage via
    :func:`chess_engine.generate_diagnostic_sets`, then returns both the
    argmax-accuracy and the v1 sampled distributional metrics. When
    ``report`` is True the per-label quota fill-rate table is printed
    (v1 parity).
    """
    (
        move_ids_np, game_lengths_np, bits, outcome_tokens, _fill
    ) = _generate_diagnostic_corpus(
        per_label=per_label, max_ply=max_ply, seed=seed,
        max_simulated_factor=max_simulated_factor, report=report,
    )
    accuracy, sampled = _compute_per_bit_all(
        model, move_ids_np, game_lengths_np, bits, outcome_tokens,
        conditioning=conditioning, batch_size=batch_size,
        want_accuracy=True, want_sampled=True,
    )
    assert accuracy is not None and sampled is not None
    return accuracy, sampled
