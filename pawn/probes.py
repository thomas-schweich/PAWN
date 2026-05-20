"""Linear probes on PAWN hidden states — JAX/Equinox port of
``pawn.eval_suite.probes`` from the pre-migration codebase.

Trains one linear regressor per probe per layer (embed + each of the
``n_layers`` transformer outputs) on board-state features extracted by the
Rust engine. Used to gauge how much of each board-state attribute is linearly
recoverable from the hidden state at each depth.

Probes ported from the legacy implementation (legacy was 1,249 lines; this
port is ~700 because JAX's vmap + eqx + optax remove most of the bookkeeping):

| name              | n_outputs | loss / metric    | what it predicts                              |
|-------------------|-----------|------------------|-----------------------------------------------|
| piece_type        | 13 × 64   | ce_per_square    | per-square piece (13 classes, incl. empty)    |
| side_to_move      | 1         | bce              | whose turn it is                              |
| is_check          | 1         | bce              | side-to-move in check?                        |
| castling_rights   | 4         | bce              | KQkq castling rights                          |
| ep_square         | 65        | ce               | en-passant target square (64 + "none")        |
| material_count    | 10        | mse (R² / MAE)   | counts per piece type per colour              |
| legal_move_count  | 1         | mse (R² / MAE)   | legal moves available                         |
| halfmove_clock    | 1         | mse (R² / MAE)   | ply since last pawn/capture                   |
| game_phase        | 3         | ce               | opening / middlegame / endgame                |

Training loop is *streaming*: for each game-batch the model is run forward
once via ``PAWNModel.hidden_all_layers`` (one pass; produces all per-layer
hidden states), valid positions are gathered, and the probes take one SGD
step per inner mini-batch of valid positions. Final + best-during-training
metrics per (probe, layer) are returned by ``train_probes``. Hidden states
are *never* materialised across the full corpus, matching the legacy
``train_all_probes`` streaming behaviour.

The "is_square_attacked" probe present in the legacy module is omitted —
the engine still doesn't export an attack-map PyO3 binding (legacy comment
preserved).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import chess_engine as engine
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from pawn.config import PAD_TOKEN
from pawn.corpus import _map_term_to_outcome_offset
from pawn.model import PAWNModel

LossType = Literal["ce", "ce_per_square", "bce", "mse"]


@dataclass(frozen=True)
class ProbeSpec:
    """Static description of one probe: output width + loss head."""

    name: str
    n_outputs: int
    loss_type: LossType
    description: str


PROBES: dict[str, ProbeSpec] = {
    "piece_type": ProbeSpec(
        "piece_type", 13 * 64, "ce_per_square",
        "Per-square piece type (13 classes × 64 squares)",
    ),
    "side_to_move": ProbeSpec(
        "side_to_move", 1, "bce", "Whose turn it is",
    ),
    "is_check": ProbeSpec(
        "is_check", 1, "bce", "Whether side to move is in check",
    ),
    "castling_rights": ProbeSpec(
        "castling_rights", 4, "bce", "KQkq castling rights",
    ),
    "ep_square": ProbeSpec(
        "ep_square", 65, "ce", "En passant square (64 + none)",
    ),
    "material_count": ProbeSpec(
        "material_count", 10, "mse",
        "Piece counts per type per color (P/N/B/R/Q × W/B)",
    ),
    "legal_move_count": ProbeSpec(
        "legal_move_count", 1, "mse", "Number of legal moves available",
    ),
    "halfmove_clock": ProbeSpec(
        "halfmove_clock", 1, "mse", "Ply since last capture or pawn move",
    ),
    "game_phase": ProbeSpec(
        "game_phase", 3, "ce", "Opening / middlegame / endgame",
    ),
}

# Board encoding (mirrors ``engine.extract_board_states``):
# 0=empty, 1-6=WP/WN/WB/WR/WQ/WK, 7-12=BP/BN/BB/BR/BQ/BK
_WHITE_PIECES = (1, 2, 3, 4, 5)   # WP=1, WN=2, WB=3, WR=4, WQ=5  (king excluded)
_BLACK_PIECES = (7, 8, 9, 10, 11)  # BP=7, BN=8, BB=9, BR=10, BQ=11

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_u64(arr: np.ndarray) -> np.ndarray:
    """Vectorised SWAR popcount on a uint64 array (1 LUT lookup × 8 bytes)."""
    result = np.zeros(arr.shape, dtype=np.uint32)
    for shift in range(0, 64, 8):
        byte = ((arr >> shift) & 0xFF).astype(np.uint8)
        result += _POPCOUNT_LUT[byte].astype(np.uint32)
    return result


def _count_legal_moves(
    move_ids: np.ndarray, game_lengths: np.ndarray
) -> np.ndarray:
    """Per-ply legal-move count via the bit-packed grid + promotion mask.

    Mirrors the legacy ``_count_legal_moves`` from ``pawn.eval_suite.corpus``;
    every legal move is one bit in ``grid``, and ``promo_mask`` adjusts the
    count for promotion-destination squares (one bit covers 4 promotion
    piece types).
    """
    grid, promo_mask = engine.compute_legal_move_masks(move_ids, game_lengths)
    grid_counts = np.zeros(grid.shape[:2], dtype=np.uint32)
    for sq in range(64):
        grid_counts += _popcount_u64(grid[:, :, sq])

    promo_pairs = engine.export_move_vocabulary()["promo_pairs"]
    adj = np.zeros(grid.shape[:2], dtype=np.int32)
    for i, (src, dst) in enumerate(promo_pairs):
        bit = ((grid[:, :, src] >> dst) & 1).astype(np.int32)
        n_pt = promo_mask[:, :, i, :].sum(axis=-1).astype(np.int32)
        has = (n_pt > 0).astype(np.int32)
        adj += (n_pt - 1) * bit * has
    return (grid_counts.astype(np.int32) + adj).astype(np.uint16)


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


@dataclass
class ProbeData:
    """Bundled corpus + board-state side data for probe training.

    All arrays are NumPy (host-resident); per-batch slices are pushed to
    the device inside the training loop. Compact dtypes from the engine
    are preserved here and promoted at use-time in ``get_probe_targets``.

    Shape conventions: N games, max_ply per-game ply axis, T = max_ply
    (CLM sequence length matches max_ply when ``prepend_outcome=False``;
    when it's True there is one extra position at index 0).
    """

    input_ids: np.ndarray            # (N, T) int32 — CLM input tokens
    attn_mask: np.ndarray            # (N, T) bool — True at non-pad positions
    game_lengths: np.ndarray         # (N,) int32 — moves per game
    boards: np.ndarray               # (N, max_ply, 8, 8) int8
    side_to_move: np.ndarray         # (N, max_ply) bool
    castling_rights: np.ndarray      # (N, max_ply) uint8
    ep_square: np.ndarray            # (N, max_ply) int8
    is_check: np.ndarray             # (N, max_ply) bool
    halfmove_clock: np.ndarray       # (N, max_ply) uint8
    legal_move_counts: np.ndarray | None = None  # (N, max_ply) uint16
    ply_offset: int = 0              # 0 (pure-moves) or 1 (outcome-prepended)
    max_ply: int = 0


def extract_probe_data(
    n_games: int,
    max_ply: int,
    seed: int,
    prepend_outcome: bool = False,
    include_legal_counts: bool = True,
    legal_count_sub_batch: int = 2000,
) -> ProbeData:
    """Generate ``n_games`` random games via the Rust engine and pull out
    every per-ply board attribute the probe targets reference.

    Args:
        max_ply: total CLM sequence length passed to ``generate_clm_batch``.
            When ``prepend_outcome=True``, position 0 of the CLM input is
            the outcome token; the per-ply boards array still has
            ``max_ply`` slots indexed from 0 (the engine emits move plies
            without the outcome prefix).
        prepend_outcome: match the training-time outcome-prefix format.
        include_legal_counts: turn off only if the legal-move-count probe
            is excluded — counting moves is the bulk of the engine time.
        legal_count_sub_batch: how many games to feed
            ``engine.compute_legal_move_masks`` per call (the grid array
            grows linearly with this; default 2000 matches the legacy
            sub-batching to bound peak memory).
    """
    input_ids_i16, _targets, _loss_mask, move_ids, game_lengths, _tc = (
        engine.generate_clm_batch(
            n_games, max_ply, seed,
            False,        # discard_ply_limit — keep ply-limit games (legacy
                          # probes don't filter them)
            0.0,          # mate_boost — no boost (probe data should match
                          # what the model sees at eval time)
            prepend_outcome,
        )
    )

    boards, side_np, castling_np, ep_np, check_np, halfmove_np = (
        engine.extract_board_states(move_ids, game_lengths)
    )

    # CLM input attention mask: True where input_ids is not PAD. The engine
    # emits a stable left-aligned sequence; deriving the mask from PAD is
    # robust against either ``prepend_outcome`` mode.
    input_ids = input_ids_i16.astype(np.int32)
    attn_mask = input_ids != PAD_TOKEN

    legal_counts: np.ndarray | None = None
    if include_legal_counts:
        parts = []
        for s in range(0, n_games, legal_count_sub_batch):
            e = min(s + legal_count_sub_batch, n_games)
            parts.append(_count_legal_moves(move_ids[s:e], game_lengths[s:e]))
        legal_counts = np.concatenate(parts, axis=0)

    return ProbeData(
        input_ids=input_ids,
        attn_mask=attn_mask,
        game_lengths=game_lengths.astype(np.int32),
        boards=np.ascontiguousarray(boards),
        side_to_move=np.ascontiguousarray(side_np),
        castling_rights=np.ascontiguousarray(castling_np),
        ep_square=np.ascontiguousarray(ep_np),
        is_check=np.ascontiguousarray(check_np),
        halfmove_clock=np.ascontiguousarray(halfmove_np),
        legal_move_counts=legal_counts,
        ply_offset=1 if prepend_outcome else 0,
        max_ply=int(move_ids.shape[1]),
    )


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------


def get_probe_targets(
    probe_name: str,
    data: ProbeData,
    game_indices: np.ndarray,
    ply_indices: np.ndarray,
) -> np.ndarray:
    """Per-position probe targets at ``(game_indices[i], ply_indices[i])``.

    Output dtype + shape:

    * ``ce`` heads: int32 ``(N,)`` class id.
    * ``ce_per_square`` (piece_type): int32 ``(N, 64)`` per-square class id.
    * ``bce``: float32 ``(N, n_outputs)``.
    * ``mse``: float32 ``(N, n_outputs)``.

    Mirrors the legacy ``get_probe_targets`` semantics in numpy (no torch).
    """
    g = game_indices
    p = ply_indices

    if probe_name == "piece_type":
        # (N, 8, 8) int8 → (N, 64) int32
        boards = data.boards[g, p]
        return boards.reshape(-1, 64).astype(np.int32)

    if probe_name == "side_to_move":
        return data.side_to_move[g, p].astype(np.float32).reshape(-1, 1)

    if probe_name == "is_check":
        return data.is_check[g, p].astype(np.float32).reshape(-1, 1)

    if probe_name == "castling_rights":
        raw = data.castling_rights[g, p].astype(np.int64)
        bits = np.arange(4, dtype=np.int64)
        # ((N, 1) >> (4,)) & 1 → (N, 4); cast to float32 for BCE
        return ((raw[:, None] >> bits) & 1).astype(np.float32)

    if probe_name == "ep_square":
        ep = data.ep_square[g, p].astype(np.int32)
        # Legacy convention: ep < 0 means "no ep square"; map to slot 64.
        return np.where(ep < 0, 64, ep).astype(np.int32)

    if probe_name == "material_count":
        boards = data.boards[g, p].reshape(-1, 64)
        # Vectorised: ``(N, 64, 1) == (1, 1, 10)`` → ``(N, 64, 10)`` → sum to ``(N, 10)``
        piece_ids = np.array(
            _WHITE_PIECES + _BLACK_PIECES, dtype=boards.dtype,
        )
        eq = boards[:, :, None] == piece_ids[None, None, :]
        return eq.sum(axis=1).astype(np.float32)

    if probe_name == "legal_move_count":
        if data.legal_move_counts is None:
            raise ValueError(
                "ProbeData was built with include_legal_counts=False; "
                "legal_move_count probe is not available."
            )
        return data.legal_move_counts[g, p].astype(np.float32).reshape(-1, 1)

    if probe_name == "halfmove_clock":
        return data.halfmove_clock[g, p].astype(np.float32).reshape(-1, 1)

    if probe_name == "game_phase":
        boards = data.boards[g, p].reshape(-1, 64)
        non_empty = (boards != 0).sum(axis=-1)
        is_pawn_or_king = (
            (boards == 1) | (boards == 6) | (boards == 7) | (boards == 12)
        )
        non_pawn_king = ((boards != 0) & ~is_pawn_or_king).sum(axis=-1)
        opening = (p <= 20) & (non_empty >= 28)
        endgame = non_pawn_king <= 6
        phase = np.ones(p.shape, dtype=np.int32)  # middlegame
        phase[opening] = 0
        phase[endgame & ~opening] = 2
        return phase

    raise ValueError(f"Unknown probe: {probe_name}")


# ---------------------------------------------------------------------------
# Linear probe module
# ---------------------------------------------------------------------------


class BatchedLinearProbe(eqx.Module):
    """``n_probes`` independent linear maps packed into one tensor.

    For per-layer probing, ``n_probes = n_layers + 1`` and each "probe"
    consumes a different layer's hidden state, but they all share the
    same probe target. The leading axis is therefore the *layer*; an
    ``einsum("lbd,ldo->lbo", ...)`` runs all of them in parallel.
    """

    weight: jax.Array  # (n_probes, d_model, n_outputs)
    bias: jax.Array    # (n_probes, n_outputs)

    @classmethod
    def init(
        cls, n_probes: int, d_model: int, n_outputs: int, key: jax.Array
    ) -> "BatchedLinearProbe":
        # Same initialisation as the legacy torch BatchedLinearProbe:
        # zero-mean normal scaled by ``1/sqrt(d_model)`` for the weight
        # and zeros for the bias.
        scale = 1.0 / float(d_model) ** 0.5
        w = jax.random.normal(key, (n_probes, d_model, n_outputs)) * scale
        b = jnp.zeros((n_probes, n_outputs))
        return cls(weight=w, bias=b)

    def __call__(self, h: jax.Array) -> jax.Array:
        """``h: (n_probes, N, d_model)`` → ``(n_probes, N, n_outputs)``."""
        return jnp.einsum("lbd,ldo->lbo", h, self.weight) + self.bias[:, None, :]


# ---------------------------------------------------------------------------
# Loss + metric helpers
# ---------------------------------------------------------------------------


def _per_layer_loss(
    logits: jax.Array, targets: jax.Array, spec: ProbeSpec
) -> jax.Array:
    """Per-layer mean loss across ``N`` positions.

    Shapes:
        * ``ce``: logits ``(L, N, K)``, targets int ``(N,)`` → ``(L,)``
        * ``ce_per_square``: logits ``(L, N, 64*13)`` reshaped on the host
          to ``(L, N, 64, 13)`` (square-major: the last axis carries the
          13 class logits for one of 64 squares), targets int ``(N, 64)``
          → ``(L,)``
        * ``bce``: logits ``(L, N, K)``, targets float ``(N, K)`` → ``(L,)``
        * ``mse``: logits ``(L, N, K)``, targets float ``(N, K)`` → ``(L,)``
    """
    L, N = logits.shape[0], logits.shape[1]
    if spec.loss_type == "ce":
        per = optax.softmax_cross_entropy_with_integer_labels(
            logits, jnp.broadcast_to(targets[None], (L, N))
        )
        return per.mean(axis=1)
    if spec.loss_type == "ce_per_square":
        # (L, N, 64*13) → (L, N, 64, 13); targets (N, 64) → (L, N, 64)
        lg = logits.reshape(L, N, 64, 13)
        tgt = jnp.broadcast_to(targets[None], (L, N, 64))
        per = optax.softmax_cross_entropy_with_integer_labels(lg, tgt)
        return per.mean(axis=(1, 2))
    if spec.loss_type == "bce":
        tgt = jnp.broadcast_to(targets[None], logits.shape)
        per = optax.sigmoid_binary_cross_entropy(logits, tgt)
        return per.mean(axis=(1, 2))
    if spec.loss_type == "mse":
        tgt = jnp.broadcast_to(targets[None], logits.shape)
        return jnp.mean((logits - tgt) ** 2, axis=(1, 2))
    raise ValueError(f"Unknown loss type: {spec.loss_type}")


def _per_layer_accuracy(
    logits: jax.Array, targets: jax.Array, spec: ProbeSpec
) -> jax.Array:
    """Per-layer accuracy. For ``mse`` the caller computes R² via
    ``ss_res``/``ss_tot`` accumulation; this function returns NaN for that
    case so a stray call surfaces immediately."""
    L, N = logits.shape[0], logits.shape[1]
    if spec.loss_type == "ce":
        preds = logits.argmax(axis=-1)            # (L, N)
        tgt = jnp.broadcast_to(targets[None], (L, N))
        return (preds == tgt).mean(axis=1).astype(jnp.float32)
    if spec.loss_type == "ce_per_square":
        preds = logits.reshape(L, N, 64, 13).argmax(axis=-1)
        tgt = jnp.broadcast_to(targets[None], (L, N, 64))
        return (preds == tgt).mean(axis=(1, 2)).astype(jnp.float32)
    if spec.loss_type == "bce":
        preds = (logits > 0).astype(jnp.float32)
        tgt = jnp.broadcast_to(targets[None], logits.shape)
        return (preds == tgt).mean(axis=(1, 2)).astype(jnp.float32)
    # mse → caller uses the R² accumulator
    return jnp.full((L,), jnp.nan)


# ---------------------------------------------------------------------------
# Hidden-state gathering
# ---------------------------------------------------------------------------


def _forward_hidden(
    model: PAWNModel, input_ids: jax.Array, attn_mask: jax.Array,
) -> jax.Array:
    """JIT-shape-stable forward returning ``(n_layers + 1, B, T, d_model)``.

    Codex flagged the previous integer-indexed-gather variant for
    re-tracing on every distinct ``N_valid`` (per-batch valid-position
    count). The new layout keeps the JITted forward shape-stable —
    only ``(B, T)`` parametrise the trace, both fixed across a training
    run (``T = max_ply``; ``B`` is ``game_batch_size`` except for the
    final partial batch, so at most two compiles). The caller does the
    valid-position gather *outside* ``jit`` against the returned
    on-device tensor; the gather is one op per batch and stays on the
    device.
    """
    return model.hidden_all_layers(input_ids, attn_mask)


_forward_hidden_jit = eqx.filter_jit(_forward_hidden, donate="none")


def _gather_valid_positions(
    hidden: jax.Array,
    g_idx: jax.Array,
    seq_idx: jax.Array,
) -> jax.Array:
    """Outside-``jit`` fancy gather of ``(L, N_valid, d_model)``.

    ``hidden: (L, B, T, d_model)``; ``g_idx`` is game indices within the
    batch, ``seq_idx`` is CLM-sequence positions (already ``ply_offset +
    p_local`` from the caller). Numpy-style advanced indexing on the two
    middle axes — JAX dispatches it as a single gather op, no
    re-tracing.
    """
    return hidden[:, g_idx, seq_idx, :]


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


_ProbeStep = Callable[
    [
        "BatchedLinearProbe",
        optax.OptState,
        optax.GradientTransformation,
        jax.Array,
        jax.Array,
    ],
    tuple["BatchedLinearProbe", optax.OptState, jax.Array],
]


def _make_step(spec: ProbeSpec) -> _ProbeStep:
    """Return a JIT-compiled SGD step closure for one probe."""

    def loss_fn(probe: BatchedLinearProbe, h: jax.Array, t: jax.Array) -> jax.Array:
        return _per_layer_loss(probe(h), t, spec).sum()

    @eqx.filter_jit
    def step(
        probe: BatchedLinearProbe,
        opt_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        h: jax.Array,
        t: jax.Array,
    ) -> tuple[BatchedLinearProbe, optax.OptState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(probe, h, t)
        # AdamW's ``update`` accepts a params pytree for the weight-decay
        # leg; ``eqx.filter`` strips the static dataclass leaves so optax
        # sees only the array leaves it can decay. The cast through
        # ``params=...`` keeps optax's type stubs happy.
        params = eqx.filter(probe, eqx.is_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        probe = eqx.apply_updates(probe, updates)
        return probe, opt_state, loss

    return step


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------


@dataclass
class _ProbeAccum:
    """Per-probe accumulator for streaming validation."""

    spec: ProbeSpec
    loss_sum: np.ndarray   # (L,)
    n: int = 0
    # Classification:
    correct_sum: np.ndarray | None = None  # (L,)
    # MSE:
    ss_res: np.ndarray | None = None       # (L,)
    ss_tot: np.ndarray | None = None       # (L,)
    ae_sum: np.ndarray | None = None       # (L,)
    target_mean: np.ndarray | None = None  # (1, n_outputs)


def _compute_val_target_means(
    data: ProbeData, probe_names: list[str]
) -> dict[str, np.ndarray]:
    """For MSE probes, compute the per-output target mean over the *whole*
    validation set up front so the streaming R² accumulator can reference
    the true global mean rather than a per-batch one."""
    means: dict[str, np.ndarray] = {}
    mse_names = [p for p in probe_names if PROBES[p].loss_type == "mse"]
    if not mse_names:
        return means
    max_ply = data.max_ply
    gl = data.game_lengths
    ply_grid = np.arange(max_ply, dtype=np.int32)[None, :]
    valid_mask = ply_grid < gl[:, None]
    b_idx, p_idx = valid_mask.nonzero()
    for name in mse_names:
        t = get_probe_targets(name, data, b_idx, p_idx)
        means[name] = t.mean(axis=0, keepdims=True).astype(np.float32)
    return means


def _evaluate_streaming(
    model: PAWNModel,
    probes_dict: dict[str, BatchedLinearProbe],
    data: ProbeData,
    probe_names: list[str],
    target_means: dict[str, np.ndarray],
    game_batch_size: int,
    n_layers_out: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Stream the val corpus through the model + every probe once.

    Returns ``{probe_name: {"loss": (L,), "accuracy": (L,) [, "mae": (L,)]}}``.
    """
    accums: dict[str, _ProbeAccum] = {}
    for name in probe_names:
        spec = PROBES[name]
        accums[name] = _ProbeAccum(
            spec=spec,
            loss_sum=np.zeros(n_layers_out, dtype=np.float64),
            correct_sum=(
                None if spec.loss_type == "mse"
                else np.zeros(n_layers_out, dtype=np.float64)
            ),
            ss_res=(
                np.zeros(n_layers_out, dtype=np.float64)
                if spec.loss_type == "mse" else None
            ),
            ss_tot=(
                np.zeros(n_layers_out, dtype=np.float64)
                if spec.loss_type == "mse" else None
            ),
            ae_sum=(
                np.zeros(n_layers_out, dtype=np.float64)
                if spec.loss_type == "mse" else None
            ),
            target_mean=target_means.get(name),
        )

    n_games = data.input_ids.shape[0]
    max_ply = data.max_ply

    for gs in range(0, n_games, game_batch_size):
        ge = min(gs + game_batch_size, n_games)
        batch_idx = np.arange(gs, ge)
        ids = jnp.asarray(data.input_ids[gs:ge])
        attn = jnp.asarray(data.attn_mask[gs:ge])
        gl = data.game_lengths[gs:ge]
        ply_grid = np.arange(max_ply, dtype=np.int32)[None, :]
        valid_mask_np = ply_grid < gl[:, None]
        n_valid = int(valid_mask_np.sum())
        if n_valid == 0:
            continue
        b_local, p_local = valid_mask_np.nonzero()
        seq_local = p_local.astype(np.int32) + data.ply_offset
        # Jit-shape-stable forward: only ``(B, T)`` parametrise the
        # trace. The gather of valid positions runs outside ``jit`` so
        # ``N_valid`` (which varies per batch) doesn't trigger
        # retracing.
        hidden_all = _forward_hidden_jit(model, ids, attn)
        h = _gather_valid_positions(
            hidden_all,
            jnp.asarray(b_local.astype(np.int32)),
            jnp.asarray(seq_local),
        )
        g_global = batch_idx[b_local]

        for name in probe_names:
            spec = PROBES[name]
            t_np = get_probe_targets(name, data, g_global, p_local)
            t = jnp.asarray(t_np)
            logits = probes_dict[name](h)
            loss_per = _per_layer_loss(logits, t, spec)
            accums[name].loss_sum += np.asarray(loss_per) * n_valid
            accums[name].n += n_valid

            if spec.loss_type == "mse":
                tgt = jnp.broadcast_to(t[None], logits.shape)
                diff = logits - tgt
                accums[name].ss_res += np.asarray((diff ** 2).sum(axis=(1, 2)))
                mean = jnp.asarray(accums[name].target_mean)  # (1, n_out)
                centered = tgt - mean[None]
                accums[name].ss_tot += np.asarray(
                    (centered ** 2).sum(axis=(1, 2))
                )
                accums[name].ae_sum += np.asarray(diff.__abs__().sum(axis=(1, 2)))
            else:
                accs = _per_layer_accuracy(logits, t, spec)
                accums[name].correct_sum += np.asarray(accs) * n_valid

    out: dict[str, dict[str, np.ndarray]] = {}
    for name in probe_names:
        spec = PROBES[name]
        a = accums[name]
        denom = max(a.n, 1)
        entry: dict[str, np.ndarray] = {
            "loss": (a.loss_sum / denom).astype(np.float64),
        }
        if spec.loss_type == "mse":
            assert a.ss_res is not None and a.ss_tot is not None
            assert a.ae_sum is not None
            entry["accuracy"] = (
                1.0 - a.ss_res / (a.ss_tot + 1e-8)
            ).astype(np.float64)
            # ae_sum was accumulated as |diff|.sum(axis=(positions, outputs)),
            # so the per-element mean divides by n_positions × n_outputs.
            # The legacy torch streaming evaluator divided by only n_positions,
            # which produced an n_outputs-scaled "MAE" for material_count
            # (n_outputs = 10) — confusing when consumers compared MAE
            # across probes. Divide by n × n_outputs here so all probes
            # report MAE on the same per-element scale. (Bug + Type)
            entry["mae"] = (a.ae_sum / (denom * spec.n_outputs)).astype(np.float64)
        else:
            assert a.correct_sum is not None
            entry["accuracy"] = (a.correct_sum / denom).astype(np.float64)
        out[name] = entry
    return out


def train_probes(
    model: PAWNModel,
    train_data: ProbeData,
    val_data: ProbeData,
    *,
    n_epochs: int = 20,
    lr: float = 1e-3,
    game_batch_size: int = 64,
    inner_batch_size: int = 256,
    probe_names: list[str] | None = None,
    key: jax.Array | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, dict[str, float]]]:
    """Train every probe across embed + n_layers transformer outputs.

    The training loop is *streaming*: each iteration forwards one game
    batch through the model (single pass, all-layer outputs), gathers
    valid positions, and runs ``inner_batch_size``-sized SGD updates over
    every probe. Validation is also streaming and accumulates loss /
    accuracy / R² + MAE per layer across the full val corpus.

    Args:
        n_epochs: passes over the train games.
        lr: peak AdamW lr (no warmup — probes converge fast).
        game_batch_size: games per forward pass.
        inner_batch_size: per-position SGD mini-batch size. The legacy
            trainer used 256 to match the dense path's per-epoch update
            count; preserved here.
        probe_names: subset of ``PROBES`` keys (default: all 9).
        key: PRNG key for probe init. Default ``jax.random.key(0)``.

    Returns ``results[probe_name][layer_name] = {accuracy, loss, ...}``.
    ``layer_name`` is ``"embed"`` for the embedding output and
    ``"layer_<i>"`` for ``i = 0..n_layers - 1``.
    """
    if probe_names is None:
        probe_names = list(PROBES.keys())
    if key is None:
        key = jax.random.key(0)

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    n_layers_out = n_layers + 1
    layer_names = ["embed"] + [f"layer_{i}" for i in range(n_layers)]

    # Split the top-level key once: one stream for probe inits, one for
    # the train-loop permutations. Each downstream consumer then splits
    # its own stream further so the "use each key exactly once" rule
    # holds end-to-end. (Bug-detector found that the legacy port reused
    # the input key for both, which works because JAX random ops are
    # pairwise-independent on the same key, but couples the streams if a
    # later refactor reuses one of the ops.)
    init_key, perm_key = jax.random.split(key)
    keys = jax.random.split(init_key, len(probe_names))
    probes_dict: dict[str, BatchedLinearProbe] = {}
    opt_states: dict[str, optax.OptState] = {}
    optimizers: dict[str, optax.GradientTransformation] = {}
    steps: dict[str, _ProbeStep] = {}
    for name, k in zip(probe_names, keys):
        spec = PROBES[name]
        probe = BatchedLinearProbe.init(n_layers_out, d_model, spec.n_outputs, k)
        probes_dict[name] = probe
        opt = optax.adamw(learning_rate=lr)
        optimizers[name] = opt
        opt_states[name] = opt.init(eqx.filter(probe, eqx.is_array))
        steps[name] = _make_step(spec)

    val_target_means = _compute_val_target_means(val_data, probe_names)

    n_train_games = train_data.input_ids.shape[0]
    n_train_positions = int(train_data.game_lengths.sum())
    n_val_games = val_data.input_ids.shape[0]

    if verbose:
        print(
            f"[probes] streaming train: {len(probe_names)} probes × "
            f"{n_layers_out} layers × {n_epochs} epochs "
            f"({n_train_games} train games / {n_train_positions:,} positions / "
            f"{n_val_games} val games)"
        )

    best_acc: dict[str, np.ndarray] = {
        name: np.full(n_layers_out, -np.inf, dtype=np.float64)
        for name in probe_names
    }
    max_ply = train_data.max_ply
    perm_key = key

    for epoch in range(n_epochs):
        perm_key, sub = jax.random.split(perm_key)
        perm = np.asarray(jax.random.permutation(sub, n_train_games))
        for gs in range(0, n_train_games, game_batch_size):
            ge = min(gs + game_batch_size, n_train_games)
            batch_idx = perm[gs:ge]
            ids = jnp.asarray(train_data.input_ids[batch_idx])
            attn = jnp.asarray(train_data.attn_mask[batch_idx])
            gl = train_data.game_lengths[batch_idx]
            ply_grid = np.arange(max_ply, dtype=np.int32)[None, :]
            valid_mask_np = ply_grid < gl[:, None]
            n_valid = int(valid_mask_np.sum())
            if n_valid == 0:
                continue
            b_local, p_local = valid_mask_np.nonzero()
            seq_local = p_local.astype(np.int32) + train_data.ply_offset
            hidden_all = _forward_hidden_jit(model, ids, attn)
            h = _gather_valid_positions(
                hidden_all,
                jnp.asarray(b_local.astype(np.int32)),
                jnp.asarray(seq_local),
            )
            g_global = batch_idx[b_local]

            # Precompute targets per probe for this batch.
            batch_targets: dict[str, jax.Array] = {}
            for name in probe_names:
                batch_targets[name] = jnp.asarray(
                    get_probe_targets(name, train_data, g_global, p_local)
                )

            # One position-shuffled sweep over the valid positions, matching
            # the legacy 256-position SGD step count per batch.
            perm_key, sub = jax.random.split(perm_key)
            pos_perm = np.asarray(jax.random.permutation(sub, n_valid))
            for s in range(0, n_valid, inner_batch_size):
                e = min(s + inner_batch_size, n_valid)
                idx_jnp = jnp.asarray(pos_perm[s:e])
                h_inner = h[:, idx_jnp, :]
                for name in probe_names:
                    t_inner = batch_targets[name][idx_jnp]
                    new_probe, new_state, _loss = steps[name](
                        probes_dict[name],
                        opt_states[name],
                        optimizers[name],
                        h_inner,
                        t_inner,
                    )
                    probes_dict[name] = new_probe
                    opt_states[name] = new_state

        # Per-epoch validation to track best per-layer accuracy.
        val_metrics = _evaluate_streaming(
            model, probes_dict, val_data, probe_names,
            val_target_means, game_batch_size, n_layers_out,
        )
        for name in probe_names:
            best_acc[name] = np.maximum(
                best_acc[name], val_metrics[name]["accuracy"],
            ).astype(np.float64)

        if verbose:
            line = f"[probes] epoch {epoch + 1}/{n_epochs}"
            for name in probe_names:
                accs = best_acc[name]
                line += f"  {name}={float(accs.max()):.3f}"
            print(line)

    # Final val pass for the returned final metrics.
    final = _evaluate_streaming(
        model, probes_dict, val_data, probe_names,
        val_target_means, game_batch_size, n_layers_out,
    )

    results: dict[str, dict[str, dict[str, float]]] = {}
    for name in probe_names:
        spec = PROBES[name]
        results[name] = {}
        for li, lname in enumerate(layer_names):
            entry: dict[str, float] = {
                "accuracy": float(final[name]["accuracy"][li]),
                "loss": float(final[name]["loss"][li]),
                "best_accuracy": float(best_acc[name][li]),
                "n_train": int(n_train_positions),
                "n_val": int(val_data.game_lengths.sum()),
            }
            if spec.loss_type == "mse":
                entry["mae"] = float(final[name]["mae"][li])
            results[name][lname] = entry
    return results
