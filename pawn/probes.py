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
    "side_to_move_labeler",
    "occupancy_labeler",
    "piece_type_labeler",
    "fit_probe",
    "extract_probe_dataset",
    "run_layer_probes",
]


@dataclass(frozen=True)
class ProbeConfig:
    """Linear probe hyperparameters."""

    n_classes: int
    lr: float = 1e-2
    n_epochs: int = 20
    batch_size: int = 256
    val_frac: float = 0.2


@dataclass(frozen=True)
class ProbeResult:
    """Held-out accuracy + the trained linear weights.

    ``accuracy`` is the **validation** (held-out) accuracy — the headline
    generalisation number. ``train_accuracy`` is kept for diagnosing
    under/over-fitting. ``n_train`` / ``n_val`` record the split sizes.
    """

    accuracy: float
    train_accuracy: float
    n_train: int
    n_val: int
    weight: Float[Array, "d n_classes"]
    bias: Float[Array, "n_classes"]


def fit_probe(
    hidden_states: Float[Array, "N d"],
    labels: Int[Array, "N"],
    cfg: ProbeConfig,
    key: jax.Array | int = 0,
) -> ProbeResult:
    """Fit a single-layer linear classifier on hidden states.

    Hidden states are assumed already extracted from the frozen backbone
    (the caller owns the no-grad forward pass — see
    :func:`extract_probe_dataset`). The probe trains via Optax AdamW +
    cross-entropy on a train split and is scored on a held-out val split
    (``cfg.val_frac`` of the samples). The reported ``accuracy`` is the
    held-out number; ``train_accuracy`` is the in-sample number for
    over-fit diagnosis.

    The split is a deterministic permutation seeded by ``key`` so a probe
    re-run is reproducible. With ``cfg.val_frac == 0`` (or too few samples
    to carve a val split) the probe falls back to in-sample scoring and
    ``n_val == 0`` flags that — callers that require held-out numbers
    should assert ``n_val > 0``.
    """
    if isinstance(key, int):
        key = jax.random.key(key)
    hidden_states = jnp.asarray(hidden_states, dtype=jnp.float32)
    labels = jnp.asarray(labels)
    n = hidden_states.shape[0]

    # Deterministic train/val split.
    split_key, init_key = jax.random.split(key)
    perm = np.asarray(jax.random.permutation(split_key, n))
    n_val = int(round(cfg.val_frac * n))
    n_val = min(max(n_val, 0), n - 1) if n > 1 else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    x_train = hidden_states[train_idx]
    y_train = labels[train_idx]

    d = hidden_states.shape[-1]
    weight = jax.random.normal(init_key, (d, cfg.n_classes), dtype=jnp.float32) * 0.02
    bias = jnp.zeros((cfg.n_classes,), dtype=jnp.float32)
    opt = optax.adamw(learning_rate=cfg.lr)
    state = opt.init((weight, bias))

    def loss_fn(params, x_b, y_b):
        w, b = params
        logits = x_b @ w + b
        ce = -jax.nn.log_softmax(logits, axis=-1)
        return ce[jnp.arange(y_b.shape[0]), y_b].mean()

    @jax.jit
    def step(params, opt_state, x_b, y_b):
        loss, grads = jax.value_and_grad(loss_fn)(params, x_b, y_b)
        upd, new_opt = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, upd)
        return new_params, new_opt, loss

    params = (weight, bias)
    n_tr = x_train.shape[0]
    rng = np.random.default_rng(0)
    for _ in range(cfg.n_epochs):
        idx = rng.permutation(n_tr)
        for s in range(0, n_tr, cfg.batch_size):
            chunk = idx[s : s + cfg.batch_size]
            params, state, _ = step(params, state, x_train[chunk], y_train[chunk])

    w, b = params

    def _accuracy(x: jax.Array, y: jax.Array) -> float:
        if x.shape[0] == 0:
            return 0.0
        preds = jnp.argmax(x @ w + b, axis=-1)
        return float((preds == y).mean())

    train_acc = _accuracy(x_train, y_train)
    if n_val > 0:
        val_acc = _accuracy(hidden_states[val_idx], labels[val_idx])
    else:
        # No held-out data carved — fall back to in-sample so the metric is
        # still defined, and signal it via n_val == 0.
        val_acc = train_acc
    return ProbeResult(
        accuracy=val_acc,
        train_accuracy=train_acc,
        n_train=int(n_tr),
        n_val=int(n_val),
        weight=w,
        bias=b,
    )


# ---------------------------------------------------------------------------
# Board-feature labelers
# ---------------------------------------------------------------------------

# A labeler maps the engine's extracted board states (the tuple returned by
# ``chess_engine.extract_board_states``) plus the per-position (game, ply)
# index arrays to (labels, n_classes). ``boards`` is (N, max_ply, 8, 8) int8
# (piece codes; 0 = empty, 1..6 = white pawn..king, 7..12 = black pawn..king),
# and the auxiliary arrays (side_to_move, castling_rights, ep_square, is_check,
# halfmove_clock) are (N, max_ply). ``g_idx`` / ``p_idx`` select the
# supervised positions.
BoardLabeler = Callable[
    [tuple[np.ndarray, ...], np.ndarray, np.ndarray],
    tuple[np.ndarray, int],
]


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
) -> tuple[Float[Array, "L N d"], Int[Array, "N"]]:
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
    n_layers_plus = int(model.cfg.n_layers) + 1
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
    states = engine.extract_board_states(move_ids, game_lengths)

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
    if not g_list:
        return (
            jnp.zeros((n_layers_plus, 0, d), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.int32),
        )
    g_idx = np.concatenate(g_list)
    p_idx = np.concatenate(p_list)
    slot_idx = np.concatenate(slot_list)

    labels, _n_classes = labeler(states, g_idx, p_idx)
    feats = hidden[:, g_idx, slot_idx]  # (L+1, M, d)
    return (
        jnp.asarray(feats, dtype=jnp.float32),
        jnp.asarray(labels, dtype=jnp.int32),
    )


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
) -> tuple[Float[Array, "N d"], Int[Array, "N"]]:
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

    ``labeler`` produces per-position class labels from the engine board
    states; see :data:`BoardLabeler` and the bundled labelers. To fit a
    probe at *every* layer, prefer :func:`run_layer_probes`, which forwards
    the backbone only once rather than once per layer.
    """
    per_layer, labels = _extract_all_layers_probe_dataset(
        model, move_ids, game_lengths,
        labeler=labeler, outcome_tokens=outcome_tokens,
        conditioning=conditioning, batch_size=batch_size,
    )
    return per_layer[layer], labels


def run_layer_probes(
    model: PAWNModel,
    move_ids: np.ndarray,
    game_lengths: np.ndarray,
    *,
    n_classes: int,
    labeler: BoardLabeler,
    outcome_tokens: np.ndarray | None = None,
    conditioning: tuple[str, ...] = (),
    n_epochs: int = 20,
    val_frac: float = 0.2,
    lr: float = 1e-2,
    batch_size: int = 16,
    key: jax.Array | int = 0,
) -> dict[int, ProbeResult]:
    """Fit a held-out linear probe at every layer of the frozen model.

    Returns ``{layer_index -> ProbeResult}`` over the ``n_layers + 1``
    residual-stream layers (0 = post-embedding). The frozen backbone is
    forwarded **once** (via :func:`_extract_all_layers_probe_dataset`,
    which materialises every layer's supervised features from the single
    ``hidden_states`` scan); each layer's probe then trains via
    :func:`fit_probe` over the already-extracted features with an internal
    train/val split.
    """
    n_layers_plus = int(model.cfg.n_layers) + 1
    cfg = ProbeConfig(
        n_classes=n_classes, lr=lr, n_epochs=n_epochs,
        batch_size=256, val_frac=val_frac,
    )
    if isinstance(key, int):
        key = jax.random.key(key)
    per_layer, labels = _extract_all_layers_probe_dataset(
        model, move_ids, game_lengths,
        labeler=labeler, outcome_tokens=outcome_tokens,
        conditioning=conditioning, batch_size=batch_size,
    )
    results: dict[int, ProbeResult] = {}
    for layer in range(n_layers_plus):
        layer_key = jax.random.fold_in(key, layer)
        results[layer] = fit_probe(per_layer[layer], labels, cfg, key=layer_key)
    return results
