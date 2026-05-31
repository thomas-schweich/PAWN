"""Linear probes — fit a linear classifier on frozen hidden states.

Probes the backbone's residual-stream hidden representations at each
layer for a position feature (side-to-move, square occupancy, …). The
probe is a single ``Linear`` over the layer's hidden state, fit with
Optax AdamW + cross-entropy and scored on a **held-out** split so the
reported accuracy reflects generalisation, not memorisation.

The realistic pipeline (H6):

1. :func:`extract_probe_dataset` forwards the **frozen** model on engine
   games (:meth:`pawn.model.PAWNModel.hidden_states`), takes one layer's
   per-position residual stream, and labels each supervised position from
   the engine's ground-truth board state
   (:func:`chess_engine.extract_board_states`).
2. :func:`fit_probe` trains the linear probe with an internal train/val
   split and reports the held-out accuracy.
3. :func:`run_layer_probes` repeats (1)+(2) for every layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, Int

import chess_engine as engine
from pawn.config import PAD_TOKEN
from pawn.corpus import build_prefix, conditioning_to_C
from pawn.model import PAWNModel

__all__ = [
    "ProbeConfig",
    "ProbeResult",
    "BoardLabeler",
    "ProbeFeature",
    "PROBE_FEATURES",
    "side_to_move_labeler",
    "occupancy_labeler",
    "piece_type_labeler",
    "piece_type_all_squares_labeler",
    "is_check_labeler",
    "castling_rights_labeler",
    "ep_square_labeler",
    "material_count_labeler",
    "legal_move_count_labeler",
    "halfmove_clock_labeler",
    "game_phase_labeler",
    "count_legal_moves_per_ply",
    "fit_probe",
    "extract_probe_dataset",
    "run_layer_probes",
]


@dataclass(frozen=True)
class ProbeConfig:
    """Linear probe hyperparameters.

    ``loss_type`` selects the probe head's objective and scoring:

    * ``"ce"`` — multi-class cross-entropy. ``labels`` are ``(N,)`` int class
      ids, ``n_classes`` is the number of classes, and ``accuracy`` is the
      fraction correct.
    * ``"ce_per_square"`` — independent per-square cross-entropy. ``labels``
      are ``(N, 64)`` int class ids and ``n_classes`` is ``13 * 64`` (the
      logit head is reshaped to ``(N, 64, 13)``); ``accuracy`` is the
      fraction of squares classified correctly.
    * ``"mse"`` — regression. ``labels`` are ``(N, n_classes)`` float targets,
      ``accuracy`` reports the held-out R² (with MAE available separately),
      and the loss is mean-squared error.
    """

    n_classes: int
    lr: float = 1e-2
    n_epochs: int = 20
    batch_size: int = 256
    val_frac: float = 0.2
    loss_type: str = "ce"


@dataclass(frozen=True)
class ProbeResult:
    """Held-out score + the trained linear weights.

    ``accuracy`` is the **validation** (held-out) headline number — fraction
    correct for classification probes, or the held-out R² for ``mse``
    regression probes. ``train_accuracy`` is the in-sample counterpart, kept
    for diagnosing under/over-fitting. ``best_accuracy`` is the best held-out
    ``accuracy`` observed across epochs (it can exceed the final-epoch number
    when a probe overshoots and settles). ``loss`` is the final held-out loss
    (cross-entropy or MSE). ``mae`` is the held-out mean-absolute-error and is
    ``None`` for classification probes. ``n_train`` / ``n_val`` record the
    split sizes.
    """

    accuracy: float
    train_accuracy: float
    n_train: int
    n_val: int
    weight: Float[Array, "d n_classes"]
    bias: Float[Array, "n_classes"]
    loss: float = 0.0
    best_accuracy: float = 0.0
    mae: float | None = None


def _probe_loss(
    logits: Float[Array, "B k"], targets: jax.Array, loss_type: str
) -> jax.Array:
    """Mean probe loss for one batch. ``logits`` is ``(B, n_classes)``.

    * ``ce`` — ``targets`` are ``(B,)`` int class ids.
    * ``ce_per_square`` — ``targets`` are ``(B, 64)`` int class ids; the head
      is reshaped to ``(B, 64, 13)`` and the per-square cross-entropies are
      averaged.
    * ``mse`` — ``targets`` are ``(B, n_classes)`` float regression targets.

    An empty batch (``logits.shape[0] == 0``) returns ``0.0`` rather than the
    ``nan`` that ``jnp.mean`` of an empty array would yield — mirroring the
    empty-input guards in :func:`_probe_score` / :func:`_probe_mae` so a probe
    over a degenerate (all-too-short games) pool reports a defined loss.
    """
    if logits.shape[0] == 0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    if loss_type == "ce":
        ce = -jax.nn.log_softmax(logits, axis=-1)
        return ce[jnp.arange(targets.shape[0]), targets].mean()
    if loss_type == "ce_per_square":
        per_sq = logits.reshape(logits.shape[0], 64, 13)
        ce = -jax.nn.log_softmax(per_sq, axis=-1)
        idx = jnp.arange(targets.shape[0])[:, None]
        sq = jnp.arange(64)[None, :]
        return ce[idx, sq, targets].mean()
    if loss_type == "mse":
        return jnp.mean((logits - targets) ** 2)
    raise ValueError(f"unknown probe loss_type {loss_type!r}")


def _probe_score(
    logits: Float[Array, "B k"], targets: jax.Array, loss_type: str
) -> float:
    """Headline held-out score for one split.

    Returns fraction-correct for ``ce`` / ``ce_per_square`` and the R²
    coefficient for ``mse`` (1 - SS_res / SS_tot, computed globally over the
    split so it is not the statistically noisy per-batch average).
    """
    if logits.shape[0] == 0:
        return 0.0
    if loss_type == "ce":
        preds = jnp.argmax(logits, axis=-1)
        return float((preds == targets).mean())
    if loss_type == "ce_per_square":
        per_sq = logits.reshape(logits.shape[0], 64, 13)
        preds = jnp.argmax(per_sq, axis=-1)
        return float((preds == targets).mean())
    if loss_type == "mse":
        ss_res = jnp.sum((logits - targets) ** 2)
        ss_tot = jnp.sum((targets - targets.mean(axis=0, keepdims=True)) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-8))
    raise ValueError(f"unknown probe loss_type {loss_type!r}")


def _probe_mae(logits: Float[Array, "B k"], targets: jax.Array) -> float:
    """Held-out mean absolute error (regression probes)."""
    if logits.shape[0] == 0:
        return 0.0
    return float(jnp.abs(logits - targets).mean())


def fit_probe(
    hidden_states: Float[Array, "N d"],
    labels: jax.Array,
    cfg: ProbeConfig,
    key: jax.Array | int = 0,
    *,
    val_hidden_states: Float[Array, "M d"] | None = None,
    val_labels: jax.Array | None = None,
) -> ProbeResult:
    """Fit a single-layer linear probe on frozen hidden states.

    Hidden states are assumed already extracted from the frozen backbone
    (the caller owns the no-grad forward pass — see
    :func:`extract_probe_dataset`). The probe trains via Optax AdamW on a
    train split and is scored on a held-out val split. ``cfg.loss_type``
    selects the objective:

    * ``"ce"`` / ``"ce_per_square"`` — cross-entropy; the headline
      ``accuracy`` is the held-out fraction correct. ``labels`` /
      ``val_labels`` are integer class-index arrays of shape ``(N,)`` /
      ``(M,)`` (``ce_per_square`` flattens its per-square axis into ``N``).
    * ``"mse"`` — regression; the headline ``accuracy`` is the held-out R²
      and ``mae`` carries the held-out mean-absolute-error. ``labels`` /
      ``val_labels`` are then float target arrays of shape ``(N, k)`` /
      ``(M, k)``.

    ``labels`` is typed as a bare :class:`jax.Array` (not a jaxtyping
    ``Int``/``Float`` alias) because its dtype and rank depend on
    ``cfg.loss_type`` per the contract above; the actual dtype is coerced
    below. ``val_hidden_states`` and ``val_labels`` must be supplied
    together — passing one without the other is an error.

    When ``val_hidden_states`` / ``val_labels`` are given, the entire
    ``hidden_states`` pool is the train set and scoring happens on that
    **explicit, independent** val pool (the ``--n-val-games`` path — train
    and val come from different games). Otherwise the probe carves a
    held-out split of ``cfg.val_frac`` from ``hidden_states`` itself, seeded
    by ``key`` so a re-run is reproducible; with ``cfg.val_frac == 0`` (or
    too few samples to carve a split) it falls back to in-sample scoring and
    ``n_val == 0`` flags that — callers needing held-out numbers should
    assert ``n_val > 0``.

    ``best_accuracy`` tracks the best held-out score across epochs (parity
    with the v1 probe suite, which reported the per-epoch best rather than
    only the final epoch). The reported ``accuracy`` / ``loss`` / ``mae`` are
    the **final-epoch** held-out numbers; ``train_accuracy`` is the in-sample
    score for over-fit diagnosis.
    """
    if isinstance(key, int):
        key = jax.random.key(key)
    loss_type = cfg.loss_type
    is_reg = loss_type == "mse"
    label_dtype = jnp.float32 if is_reg else None
    hidden_states = jnp.asarray(hidden_states, dtype=jnp.float32)
    labels = jnp.asarray(labels, dtype=label_dtype)
    n = hidden_states.shape[0]

    split_key, init_key = jax.random.split(key)
    if (val_hidden_states is None) != (val_labels is None):
        raise ValueError(
            "val_hidden_states and val_labels must be supplied together"
        )
    explicit_val = val_hidden_states is not None and val_labels is not None
    if explicit_val:
        # Independent val pool: train on everything, score on the held-out
        # pool. No within-pool split is carved.
        x_train = hidden_states
        y_train = labels
        x_val = jnp.asarray(val_hidden_states, dtype=jnp.float32)
        y_val = jnp.asarray(val_labels, dtype=label_dtype)
        n_val = int(x_val.shape[0])
    else:
        # Deterministic train/val split of the single pool.
        perm = np.asarray(jax.random.permutation(split_key, n))
        n_val = int(round(cfg.val_frac * n))
        n_val = min(max(n_val, 0), n - 1) if n > 1 else 0
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        x_train = hidden_states[train_idx]
        y_train = labels[train_idx]
        x_val = hidden_states[val_idx]
        y_val = labels[val_idx]

    d = hidden_states.shape[-1]
    weight = jax.random.normal(init_key, (d, cfg.n_classes), dtype=jnp.float32) * 0.02
    bias = jnp.zeros((cfg.n_classes,), dtype=jnp.float32)
    opt = optax.adamw(learning_rate=cfg.lr)
    state = opt.init((weight, bias))

    def loss_fn(params, x_b, y_b):
        w, b = params
        return _probe_loss(x_b @ w + b, y_b, loss_type)

    @jax.jit
    def step(params, opt_state, x_b, y_b):
        loss, grads = jax.value_and_grad(loss_fn)(params, x_b, y_b)
        upd, new_opt = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, upd)
        return new_params, new_opt, loss

    params = (weight, bias)
    n_tr = x_train.shape[0]
    rng = np.random.default_rng(0)
    best_acc = -float("inf")
    for _ in range(cfg.n_epochs):
        idx = rng.permutation(n_tr)
        for s in range(0, n_tr, cfg.batch_size):
            chunk = idx[s : s + cfg.batch_size]
            params, state, _ = step(params, state, x_train[chunk], y_train[chunk])
        # Per-epoch held-out score for best-accuracy tracking (v1 parity).
        w_e, b_e = params
        if n_val > 0:
            epoch_acc = _probe_score(x_val @ w_e + b_e, y_val, loss_type)
        else:
            epoch_acc = _probe_score(x_train @ w_e + b_e, y_train, loss_type)
        best_acc = max(best_acc, epoch_acc)

    w, b = params

    train_acc = _probe_score(x_train @ w + b, y_train, loss_type)
    if n_val > 0:
        val_logits = x_val @ w + b
        val_acc = _probe_score(val_logits, y_val, loss_type)
        val_loss = float(_probe_loss(val_logits, y_val, loss_type))
        val_mae = _probe_mae(val_logits, y_val) if is_reg else None
    else:
        # No held-out data carved — fall back to in-sample so the metric is
        # still defined, and signal it via n_val == 0.
        train_logits = x_train @ w + b
        val_acc = train_acc
        val_loss = float(_probe_loss(train_logits, y_train, loss_type))
        val_mae = _probe_mae(train_logits, y_train) if is_reg else None
    if best_acc == -float("inf"):
        best_acc = val_acc
    return ProbeResult(
        accuracy=val_acc,
        train_accuracy=train_acc,
        n_train=int(n_tr),
        n_val=int(n_val),
        weight=w,
        bias=b,
        loss=val_loss,
        best_accuracy=best_acc,
        mae=val_mae,
    )


# ---------------------------------------------------------------------------
# Board-feature labelers
# ---------------------------------------------------------------------------

# A labeler maps the engine's extracted board states (the tuple returned by
# ``chess_engine.extract_board_states``, extended with one trailing array)
# plus the per-position (game, ply) index arrays to (labels, n_outputs).
# ``states`` is, by index:
#   0  boards            (N, max_ply, 8, 8) int8 — piece codes (0 empty,
#                        1..6 white pawn..king, 7..12 black pawn..king)
#   1  side_to_move      (N, max_ply) bool
#   2  castling_rights   (N, max_ply) uint8  — KQkq bitmask
#   3  ep_square         (N, max_ply) int8   — en-passant square, <0 = none
#   4  is_check          (N, max_ply) bool
#   5  halfmove_clock    (N, max_ply) uint8
#   6  legal_move_counts (N, max_ply) uint16 — appended by the probe pipeline
#                        via ``count_legal_moves_per_ply`` (engine masks)
# ``g_idx`` / ``p_idx`` select the supervised positions.
#
# For classification probes the returned ``labels`` are ``(M,)`` int class
# ids and ``n_outputs`` is the number of classes; for regression (``mse``)
# probes ``labels`` are ``(M, n_outputs)`` float targets. The matching loss
# type lives in :data:`PROBE_FEATURES`.
BoardLabeler = Callable[
    [tuple[np.ndarray, ...], np.ndarray, np.ndarray],
    tuple[np.ndarray, int],
]

# Board piece-code encoding (engine contract). 0 = empty, 1..6 = white
# pawn/knight/bishop/rook/queen/king, 7..12 = the black counterparts.
_WHITE_PIECES = [1, 2, 3, 4, 5]   # WP, WN, WB, WR, WQ (king excluded: never captured)
_BLACK_PIECES = [7, 8, 9, 10, 11]  # BP, BN, BB, BR, BQ


def side_to_move_labeler(
    states: tuple[np.ndarray, ...], g_idx: np.ndarray, p_idx: np.ndarray
) -> tuple[np.ndarray, int]:
    """Binary label: is white to move at this position? (2 classes).

    Perfectly separable from ply parity, so it's the canonical
    above-chance sanity feature for the probe pipeline.
    """
    stm = np.asarray(states[1], dtype=np.int64)  # side_to_move bool
    return stm[g_idx, p_idx].astype(np.int64), 2


def occupancy_labeler(
    square: int,
) -> BoardLabeler:
    """Binary label: is ``square`` (0..63, rank-major) occupied? (2 classes)."""

    def _label(
        states: tuple[np.ndarray, ...], g_idx: np.ndarray, p_idx: np.ndarray
    ) -> tuple[np.ndarray, int]:
        boards = np.asarray(states[0])  # (N, max_ply, 8, 8) int8
        rank, file = divmod(square, 8)
        occ = (boards[g_idx, p_idx, rank, file] != 0).astype(np.int64)
        return occ, 2

    return _label


def piece_type_labeler(
    square: int,
) -> BoardLabeler:
    """7-class label of the piece type at ``square``: 0 = empty, 1..6 =
    pawn/knight/bishop/rook/queen/king.

    The engine encodes pieces as ``0`` (empty), ``1..6`` (white
    pawn..king) and ``7..12`` (black pawn..king), so the colour-agnostic
    type is ``((code - 1) % 6) + 1`` for non-empty squares."""

    def _label(
        states: tuple[np.ndarray, ...], g_idx: np.ndarray, p_idx: np.ndarray
    ) -> tuple[np.ndarray, int]:
        boards = np.asarray(states[0])
        rank, file = divmod(square, 8)
        code = boards[g_idx, p_idx, rank, file].astype(np.int64)
        ptype = np.where(code == 0, 0, ((code - 1) % 6) + 1)
        return ptype, 7

    return _label


def piece_type_all_squares_labeler(
    states: tuple[np.ndarray, ...], g_idx: np.ndarray, p_idx: np.ndarray
) -> tuple[np.ndarray, int]:
    """Per-square piece occupant over all 64 squares — 13 classes each.

    The v1 parity probe (``ce_per_square``): each supervised position is
    labeled with the raw 13-class piece code (0 = empty, 1..6 white
    pawn..king, 7..12 black pawn..king) at every one of the 64 squares.
    Returns ``(M, 64)`` int labels; the probe head is ``13 * 64`` logits
    reshaped to ``(M, 64, 13)``. Unlike :func:`piece_type_labeler` this keeps
    the piece *colour* (13 classes, not the colour-agnostic 7) and probes
    the whole board at once.
    """
    boards = np.asarray(states[0])  # (N, max_ply, 8, 8) int8
    codes = boards[g_idx, p_idx].reshape(-1, 64).astype(np.int64)
    return codes, 13 * 64


def is_check_labeler(
    states: tuple[np.ndarray, ...], g_idx: np.ndarray, p_idx: np.ndarray
) -> tuple[np.ndarray, int]:
    """Binary label: is the side to move in check at this position? (2 classes)."""
    chk = np.asarray(states[4], dtype=np.int64)  # is_check bool
    return chk[g_idx, p_idx].astype(np.int64), 2


def castling_rights_labeler(
    states: tuple[np.ndarray, ...], g_idx: np.ndarray, p_idx: np.ndarray
) -> tuple[np.ndarray, int]:
    """16-class label: the KQkq castling-rights bitmask (0..15).

    The engine packs the four castling flags into a single ``uint8`` (bit 0
    = white kingside … bit 3 = black queenside). We probe the joint 4-bit
    state as a single 16-way classification (v1 used four independent BCE
    heads; under the v2 cross-entropy probe head the equivalent joint target
    is the bitmask itself, which is strictly more informative and stays a
    single-head probe)."""
    raw = np.asarray(states[2], dtype=np.int64)  # castling_rights uint8 bitmask
    return (raw[g_idx, p_idx] & 0xF).astype(np.int64), 16


def ep_square_labeler(
    states: tuple[np.ndarray, ...], g_idx: np.ndarray, p_idx: np.ndarray
) -> tuple[np.ndarray, int]:
    """65-class label: the en-passant target square (0..63), or 64 for none."""
    ep = np.asarray(states[3], dtype=np.int64)  # ep_square int8, <0 = none
    vals = ep[g_idx, p_idx]
    return np.where(vals < 0, 64, vals).astype(np.int64), 65


def material_count_labeler(
    states: tuple[np.ndarray, ...], g_idx: np.ndarray, p_idx: np.ndarray
) -> tuple[np.ndarray, int]:
    """Regression target: per-type per-colour piece counts (10 outputs).

    Order: ``[WP, WN, WB, WR, WQ, BP, BN, BB, BR, BQ]`` (kings excluded — a
    king is never captured). Returns ``(M, 10)`` float counts for an ``mse``
    probe."""
    boards = np.asarray(states[0])  # (N, max_ply, 8, 8) int8
    flat = boards[g_idx, p_idx].reshape(-1, 64)
    piece_ids = np.asarray(_WHITE_PIECES + _BLACK_PIECES, dtype=np.int64)
    # (M, 64, 1) == (1, 1, 10) -> (M, 64, 10) -> sum over squares -> (M, 10)
    counts = (flat[:, :, None] == piece_ids[None, None, :]).sum(axis=1)
    return counts.astype(np.float32), 10


def legal_move_count_labeler(
    states: tuple[np.ndarray, ...], g_idx: np.ndarray, p_idx: np.ndarray
) -> tuple[np.ndarray, int]:
    """Regression target: number of legal moves available (1 output).

    Reads the legal-move-count array the probe pipeline appends at
    ``states[6]`` (see :func:`count_legal_moves_per_ply`). Returns
    ``(M, 1)`` float counts for an ``mse`` probe."""
    lmc = np.asarray(states[6], dtype=np.float32)
    return lmc[g_idx, p_idx].reshape(-1, 1), 1


def halfmove_clock_labeler(
    states: tuple[np.ndarray, ...], g_idx: np.ndarray, p_idx: np.ndarray
) -> tuple[np.ndarray, int]:
    """Regression target: plies since the last capture or pawn move (1 output)."""
    hmc = np.asarray(states[5], dtype=np.float32)  # halfmove_clock uint8
    return hmc[g_idx, p_idx].reshape(-1, 1), 1


def game_phase_labeler(
    states: tuple[np.ndarray, ...], g_idx: np.ndarray, p_idx: np.ndarray
) -> tuple[np.ndarray, int]:
    """3-class label: opening (0) / middlegame (1) / endgame (2).

    Mirrors the v1 heuristic: opening when ``ply <= 20`` and at least 28
    pieces remain; endgame when ``<= 6`` non-pawn, non-king pieces remain;
    everything else is middlegame."""
    boards = np.asarray(states[0])
    flat = boards[g_idx, p_idx].reshape(-1, 64).astype(np.int64)
    non_empty = (flat != 0).sum(axis=-1)
    is_pawn_or_king = (flat == 1) | (flat == 6) | (flat == 7) | (flat == 12)
    non_pawn_king = ((flat != 0) & ~is_pawn_or_king).sum(axis=-1)

    ply = p_idx.astype(np.int64)
    opening = (ply <= 20) & (non_empty >= 28)
    endgame = non_pawn_king <= 6
    phase = np.ones(len(p_idx), dtype=np.int64)  # 1 = middlegame
    phase[opening] = 0  # 0 = opening
    phase[endgame & ~opening] = 2  # 2 = endgame
    return phase, 3


# ---------------------------------------------------------------------------
# Legal-move counts (engine masks → per-ply scalar)
# ---------------------------------------------------------------------------

# 256-entry byte popcount LUT, used to count set bits in the packed uint64
# legal-destination bitboards from ``compute_legal_move_masks``.
_POPCOUNT_LUT = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.uint32
)


def _popcount_u64(arr: np.ndarray) -> np.ndarray:
    """Vectorised popcount over a ``uint64`` array (byte-LUT, 8 lookups)."""
    result = np.zeros(arr.shape, dtype=np.uint32)
    for shift in range(0, 64, 8):
        byte = ((arr >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.uint8)
        result += _POPCOUNT_LUT[byte]
    return result


def count_legal_moves_per_ply(
    move_ids: np.ndarray, game_lengths: np.ndarray
) -> np.ndarray:
    """Legal-move count per (game, ply) via the engine's bit-packed masks.

    Replays ``move_ids`` through ``chess_engine.compute_legal_move_masks``
    (a ``(N, max_ply, 64)`` uint64 destination-bitboard grid plus a
    ``(N, max_ply, 44, 4)`` promotion mask) and reduces to a per-position
    legal-move count, expanding each promotion source→dest square into its
    distinct promotion pieces. Returns ``(N, max_ply)`` uint16 — the same
    counting contract as the v1 ``_count_legal_moves`` helper.
    """
    move_ids = np.ascontiguousarray(move_ids, dtype=np.int16)
    game_lengths = np.asarray(game_lengths, dtype=np.int16)
    grid, promo_mask = engine.compute_legal_move_masks(move_ids, game_lengths)
    grid = np.asarray(grid)
    promo_mask = np.asarray(promo_mask)

    grid_counts = np.zeros(grid.shape[:2], dtype=np.uint32)
    for sq in range(64):
        grid_counts += _popcount_u64(grid[:, :, sq])

    promo_pairs = engine.export_move_vocabulary()["promo_pairs"]
    adj = np.zeros(grid.shape[:2], dtype=np.int32)
    for i, (src, dst) in enumerate(promo_pairs):
        bit = ((grid[:, :, src] >> np.uint64(dst)) & np.uint64(1)).astype(np.int32)
        n_pt = promo_mask[:, :, i, :].sum(axis=-1).astype(np.int32)
        has = (n_pt > 0).astype(np.int32)
        # A promotion source→dest occupies one bit in the grid but stands for
        # ``n_pt`` distinct promotion moves; add the extra (n_pt - 1).
        adj += (n_pt - 1) * bit * has
    return (grid_counts.astype(np.int32) + adj).astype(np.uint16)


# ---------------------------------------------------------------------------
# Probe-feature registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeFeature:
    """A named probe target: how to label it, its head width, and its loss.

    ``make_labeler`` builds the :data:`BoardLabeler` (the ``square`` arg is
    only consulted by the single-square ``occupancy`` / ``piece_type``
    features and ignored by the rest). ``n_outputs`` is the linear-head width
    (number of classes for cross-entropy, number of regression targets for
    ``mse``). ``needs_legal_counts`` flags the features whose labeler reads
    the appended ``states[6]`` legal-move-count array.
    """

    make_labeler: Callable[[int], BoardLabeler]
    n_outputs: int
    loss_type: str
    needs_legal_counts: bool = False


# Registry of every probe feature, name → spec. Mirrors the v1
# ``eval_suite.probes.PROBES`` table (carried-over + regression probes), in
# v2's labeler idiom.
PROBE_FEATURES: dict[str, ProbeFeature] = {
    "side_to_move": ProbeFeature(lambda sq: side_to_move_labeler, 2, "ce"),
    "occupancy": ProbeFeature(lambda sq: occupancy_labeler(sq), 2, "ce"),
    "piece_type": ProbeFeature(lambda sq: piece_type_labeler(sq), 7, "ce"),
    "piece_type_all": ProbeFeature(
        lambda sq: piece_type_all_squares_labeler, 13 * 64, "ce_per_square"
    ),
    "is_check": ProbeFeature(lambda sq: is_check_labeler, 2, "ce"),
    "castling_rights": ProbeFeature(lambda sq: castling_rights_labeler, 16, "ce"),
    "ep_square": ProbeFeature(lambda sq: ep_square_labeler, 65, "ce"),
    "game_phase": ProbeFeature(lambda sq: game_phase_labeler, 3, "ce"),
    "material_count": ProbeFeature(lambda sq: material_count_labeler, 10, "mse"),
    "legal_move_count": ProbeFeature(
        lambda sq: legal_move_count_labeler, 1, "mse", needs_legal_counts=True
    ),
    "halfmove_clock": ProbeFeature(lambda sq: halfmove_clock_labeler, 1, "mse"),
}


# ---------------------------------------------------------------------------
# Hidden-state extraction (the frozen-forward + engine-label pipeline)
# ---------------------------------------------------------------------------


def _extract_all_layers_probe_dataset(
    model: PAWNModel,
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    *,
    labeler: BoardLabeler,
    outcome_tokens: np.ndarray | None = None,
    conditioning: tuple[str, ...] = (),
    batch_size: int = 16,
    needs_legal_counts: bool = False,
) -> tuple[Float[Array, "L N d"], jax.Array]:
    """Forward the FROZEN model ONCE → (per-layer features, shared labels).

    This is the single-forward core shared by :func:`extract_probe_dataset`
    (which slices one layer) and :func:`run_layer_probes` (which fits a
    probe at every layer). :meth:`pawn.model.PAWNModel.hidden_states`
    already returns the full ``(n_layers + 1, B, seq_len, d)`` residual
    stack from one ``lax.scan``, so we materialise every layer's
    supervised feature rows in a single pass instead of recomputing the
    whole backbone forward once per layer (which was ``O(L²)`` forward
    work). The supervised-position bookkeeping (engine board states, the
    ``(g_idx, p_idx, slot_idx)`` index construction, and the labels) is
    independent of ``layer``, so it is computed exactly once here.

    Returns ``(per_layer_feats, labels)`` where ``per_layer_feats`` has
    shape ``(n_layers + 1, M, d)`` — index ``0`` is the post-embedding
    stream, ``i`` is block ``i``'s output — and ``labels`` is the shared
    ``(M,)`` class-label vector. See :func:`extract_probe_dataset` for the
    per-position alignment contract.
    """
    move_ids = np.ascontiguousarray(move_ids, dtype=np.int16)
    game_lengths = np.asarray(game_lengths, dtype=np.int16)
    n, max_ply = move_ids.shape
    C = conditioning_to_C(conditioning)
    if outcome_tokens is None:
        outcome_tokens = np.full(n, -1, dtype=np.int32)

    seq_len = C + max_ply
    tokens = np.full((n, seq_len), PAD_TOKEN, dtype=np.int32)
    tokens[:, :C] = build_prefix(conditioning, np.asarray(outcome_tokens), n)
    move_positions = np.arange(max_ply, dtype=np.int32)[None]
    valid_move = move_positions < game_lengths[:, None].astype(np.int32)
    tokens[:, C:] = np.where(valid_move, move_ids.astype(np.int32), PAD_TOKEN)
    attn = (tokens != PAD_TOKEN).astype(np.int32)

    # Per-batch forward → collect the FULL per-layer hidden-state stack in
    # one scan per batch. hidden[l] is layer l's (n, seq_len, d) stream.
    layer_chunks: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        t = jnp.asarray(tokens[start:end])
        a = jnp.asarray(attn[start:end])
        hs = model.hidden_states(t, a)  # (L+1, b, seq_len, d)
        layer_chunks.append(np.asarray(hs))
    hidden = np.concatenate(layer_chunks, axis=1)  # (L+1, n, seq_len, d)

    # Ground-truth board states: state at index t is the board BEFORE
    # move[t] (engine contract). We label position t's successor board
    # (index t+1) so the probe target is the position the model just
    # transitioned into after consuming move[t].
    states: tuple[np.ndarray, ...] = engine.extract_board_states(move_ids, game_lengths)
    if needs_legal_counts:
        # Append the per-ply legal-move counts as states[6] for labelers
        # (e.g. legal_move_count) that probe them. Computed only on demand —
        # it replays the games through the engine mask kernels.
        states = (*states, count_legal_moves_per_ply(move_ids, game_lengths))

    gl = game_lengths.astype(np.int64)
    g_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []
    slot_list: list[np.ndarray] = []
    for g in range(n):
        # t in [0, gl-1): need t+1 < gl so the successor board exists.
        usable = int(gl[g]) - 1
        if usable <= 0:
            continue
        t = np.arange(usable, dtype=np.int64)
        g_list.append(np.full(usable, g, dtype=np.int64))
        p_list.append(t + 1)  # label the board BEFORE move[t+1]
        slot_list.append(C + t)  # residual stream after consuming move[t]
    d = hidden.shape[-1]
    if g_list:
        g_idx = np.concatenate(g_list)
        p_idx = np.concatenate(p_list)
        slot_idx = np.concatenate(slot_list)
    else:
        # No supervisable positions (every game has length <= 1). Flow empty
        # index arrays through the normal path rather than short-circuiting:
        # this lets the labeler itself determine the empty labels' shape and
        # dtype — e.g. ``(0, 10)`` float32 for the material_count mse probe or
        # ``(0, 64)`` int for the ce_per_square probe — so downstream
        # ``fit_probe`` receives a correctly-shaped empty target instead of a
        # mismatched ``(0,)`` int tensor that broadcasts to ``nan`` loss.
        g_idx = np.zeros(0, dtype=np.int64)
        p_idx = np.zeros(0, dtype=np.int64)
        slot_idx = np.zeros(0, dtype=np.int64)

    labels, _n_outputs = labeler(states, g_idx, p_idx)
    feats = hidden[:, g_idx, slot_idx]  # (L+1, M, d)
    # Preserve the labeler's dtype family: integer class ids for
    # classification probes, float targets for regression (mse) probes.
    labels_arr = np.asarray(labels)
    if np.issubdtype(labels_arr.dtype, np.floating):
        jax_labels = jnp.asarray(labels_arr, dtype=jnp.float32)
    else:
        jax_labels = jnp.asarray(labels_arr, dtype=jnp.int32)
    return (jnp.asarray(feats, dtype=jnp.float32), jax_labels)


def extract_probe_dataset(
    model: PAWNModel,
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    *,
    layer: int,
    labeler: BoardLabeler,
    outcome_tokens: np.ndarray | None = None,
    conditioning: tuple[str, ...] = (),
    batch_size: int = 16,
    needs_legal_counts: bool = False,
) -> tuple[Float[Array, "N d"], jax.Array]:
    """Forward the FROZEN model on engine games → (hidden_states, labels).

    ``move_ids`` is ``(N, max_ply)`` int16 and ``game_lengths`` is
    ``(N,)`` (number of real moves) — the standard
    :func:`chess_engine.generate_random_games` output. The model is run
    through :meth:`pawn.model.PAWNModel.hidden_states` on the
    ``[BOS][cond…][moves…]`` layout (width ``C = 1 + len(conditioning)``),
    so probe states match the model's in-distribution absolute-RoPE
    layout. We keep the hidden state at sequence slot ``C + t`` — the
    residual stream *over* ``move[t]``, i.e. the representation after the
    model has consumed the token at ply ``t``. That position is then
    labeled from the engine's board state **before** ``move[t+1]`` (the
    position the model is now reasoning about), giving a strictly causal,
    in-context probe target.

    Only positions ``t`` with ``t + 1 < game_length`` contribute (we need
    a labelable successor position whose board state the model could have
    inferred from the moves seen so far). ``layer`` indexes the
    ``(n_layers + 1, …)`` stack from :meth:`hidden_states` — ``0`` is the
    post-embedding stream, ``i`` is block ``i``'s output.

    ``labeler`` produces per-position class labels (or regression targets)
    from the engine board states; see :data:`BoardLabeler` and the bundled
    labelers. Pass ``needs_legal_counts=True`` for the ``legal_move_count``
    labeler, which reads the appended ``states[6]`` array. To fit a probe at
    *every* layer, prefer :func:`run_layer_probes`, which forwards the
    backbone only once rather than once per layer.
    """
    per_layer, labels = _extract_all_layers_probe_dataset(
        model, move_ids, game_lengths,
        labeler=labeler, outcome_tokens=outcome_tokens,
        conditioning=conditioning, batch_size=batch_size,
        needs_legal_counts=needs_legal_counts,
    )
    return per_layer[layer], labels


def run_layer_probes(
    model: PAWNModel,
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    *,
    n_classes: int,
    labeler: BoardLabeler,
    loss_type: str = "ce",
    outcome_tokens: np.ndarray | None = None,
    conditioning: tuple[str, ...] = (),
    n_epochs: int = 20,
    val_frac: float = 0.2,
    lr: float = 1e-2,
    batch_size: int = 16,
    needs_legal_counts: bool = False,
    val_move_ids: np.ndarray | None = None,
    val_game_lengths: np.ndarray | None = None,
    val_outcome_tokens: np.ndarray | None = None,
    key: jax.Array | int = 0,
) -> dict[int, ProbeResult]:
    """Fit a held-out linear probe at every layer of the frozen model.

    Returns ``{layer_index -> ProbeResult}`` over the ``n_layers + 1``
    residual-stream layers (0 = post-embedding). The frozen backbone is
    forwarded **once** (via :func:`_extract_all_layers_probe_dataset`,
    which materialises every layer's supervised features from the single
    ``hidden_states`` scan); each layer's probe then trains via
    :func:`fit_probe` over the already-extracted features.

    By default each layer's probe carves an internal ``val_frac`` train/val
    split from the same game pool. When ``val_move_ids`` /
    ``val_game_lengths`` are supplied (the ``--n-val-games`` path), a
    **separate** game pool is forwarded for validation and each probe trains
    on the full train pool and scores on that independent val pool — so
    train and val come from different games.

    ``loss_type`` selects the probe objective (``"ce"`` /
    ``"ce_per_square"`` for classification, ``"mse"`` for regression — see
    :class:`ProbeConfig`); it must match the ``labeler``'s output. Pass
    ``needs_legal_counts=True`` for the ``legal_move_count`` labeler.
    """
    n_layers_plus = int(model.cfg.n_layers) + 1
    cfg = ProbeConfig(
        n_classes=n_classes, lr=lr, n_epochs=n_epochs,
        batch_size=256, val_frac=val_frac, loss_type=loss_type,
    )
    if isinstance(key, int):
        key = jax.random.key(key)
    # Mirror fit_probe's guard: an explicit val pool must supply both the
    # move ids and the game lengths, or neither. A half-supplied pool would
    # otherwise silently fall through to within-pool train/val splitting and
    # report a misleading n_val.
    if (val_move_ids is None) != (val_game_lengths is None):
        raise ValueError(
            "val_move_ids and val_game_lengths must be supplied together"
        )
    per_layer, labels = _extract_all_layers_probe_dataset(
        model, move_ids, game_lengths,
        labeler=labeler, outcome_tokens=outcome_tokens,
        conditioning=conditioning, batch_size=batch_size,
        needs_legal_counts=needs_legal_counts,
    )
    val_per_layer: jax.Array | None = None
    val_labels: jax.Array | None = None
    if val_move_ids is not None and val_game_lengths is not None:
        val_per_layer, val_labels = _extract_all_layers_probe_dataset(
            model, val_move_ids, val_game_lengths,
            labeler=labeler, outcome_tokens=val_outcome_tokens,
            conditioning=conditioning, batch_size=batch_size,
            needs_legal_counts=needs_legal_counts,
        )
    results: dict[int, ProbeResult] = {}
    for layer in range(n_layers_plus):
        layer_key = jax.random.fold_in(key, layer)
        if val_per_layer is not None and val_labels is not None:
            results[layer] = fit_probe(
                per_layer[layer], labels, cfg, key=layer_key,
                val_hidden_states=val_per_layer[layer], val_labels=val_labels,
            )
        else:
            results[layer] = fit_probe(per_layer[layer], labels, cfg, key=layer_key)
    return results
