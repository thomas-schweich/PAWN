"""Extended linear probes for PAWN representation quality evaluation.

Includes the 5 original probes from grid-PAWN plus 5 new probes:
material_count, legal_move_count, halfmove_clock, game_phase, is_square_attacked.
"""

import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import chess_engine as engine

from pawn.config import CLMConfig
from pawn.model import PAWNCLM


# ---------------------------------------------------------------------------
# Probe definitions: name -> (n_outputs, loss_type, description)
# ---------------------------------------------------------------------------

PROBES = {
    # --- Carried over from grid-PAWN ---
    "piece_type": (13 * 64, "ce_per_square", "Per-square piece type (13 classes × 64 squares)"),
    "side_to_move": (1, "bce", "Whose turn it is"),
    "is_check": (1, "bce", "Whether side to move is in check"),
    "castling_rights": (4, "bce", "KQkq castling rights"),
    "ep_square": (65, "ce", "En passant square (64 + none)"),
    # --- New probes ---
    "material_count": (10, "mse", "Piece counts per type per color (P/N/B/R/Q × W/B)"),
    "legal_move_count": (1, "mse", "Number of legal moves available"),
    "halfmove_clock": (1, "mse", "Ply since last capture or pawn move"),
    "game_phase": (3, "ce", "Opening / middlegame / endgame"),
}

# Note: is_square_attacked is omitted — requires engine attack map support
# not yet exposed via PyO3 bindings.


class LinearProbe(nn.Module):
    def __init__(self, d_model: int, n_outputs: int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Piece type constants
# ---------------------------------------------------------------------------

# Board encoding: 0=empty, 1-6=WP/WN/WB/WR/WQ/WK, 7-12=BP/BN/BB/BR/BQ/BK
_PIECE_ORDER = ["P", "N", "B", "R", "Q"]  # per color
# Material count: [WP, WN, WB, WR, WQ, BP, BN, BB, BR, BQ]
_WHITE_PIECES = [1, 2, 3, 4, 5]   # WP=1, WN=2, WB=3, WR=4, WQ=5
_BLACK_PIECES = [7, 8, 9, 10, 11]  # BP=7, BN=8, BB=9, BR=10, BQ=11


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_probe_data(
    n_games: int,
    max_ply: int,
    seed: int,
    include_legal_counts: bool = True,
) -> dict:
    """Generate games and extract all probe data.

    Board states at ply t correspond to CLM position t+1.

    Returns dict with all arrays needed for all probes.
    """
    input_ids, targets, loss_mask, move_ids_np, game_lengths_np, _tc = \
        engine.generate_clm_batch(n_games, max_ply, seed)

    boards_np, side_np, castling_np, ep_np, check_np, halfmove_np = (
        engine.extract_board_states(move_ids_np, game_lengths_np)
    )

    result = {
        "input_ids": torch.from_numpy(input_ids).long(),
        "loss_mask": torch.from_numpy(loss_mask),
        "boards": torch.from_numpy(boards_np.copy()).long(),
        "side_to_move": torch.from_numpy(side_np.copy()).float(),
        "castling_rights": torch.from_numpy(castling_np.copy()),
        "ep_square": torch.from_numpy(ep_np.copy()).long(),
        "is_check": torch.from_numpy(check_np.copy()).float(),
        "halfmove_clock": torch.from_numpy(halfmove_np.copy()).float(),
        "game_lengths": game_lengths_np,
    }
    del boards_np, side_np, castling_np, ep_np, check_np, halfmove_np

    if include_legal_counts:
        # Process in sub-batches to avoid peak memory from the huge grid array
        from .corpus import _count_legal_moves
        sub_batch = 2000
        parts = []
        for s in range(0, n_games, sub_batch):
            e = min(s + sub_batch, n_games)
            parts.append(_count_legal_moves(move_ids_np[s:e], game_lengths_np[s:e]))
        legal_counts = np.concatenate(parts, axis=0)
        del parts
        result["legal_move_counts"] = torch.from_numpy(legal_counts).float()
        del legal_counts

    return result


def get_probe_targets(
    probe_name: str, data: dict, ply_indices: torch.Tensor,
    game_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract targets for a given probe at specific ply positions.

    Args:
        probe_name: which probe
        data: dict from extract_probe_data
        ply_indices: (N,) ply index per sample
        game_indices: (N,) game index per sample. If None, assumes
            sequential games (game i → ply_indices[i]).
    """
    B_idx = game_indices if game_indices is not None else torch.arange(len(ply_indices))

    if probe_name == "piece_type":
        boards = data["boards"][B_idx, ply_indices]  # (N, 8, 8)
        return boards.reshape(-1, 64)

    elif probe_name == "side_to_move":
        return data["side_to_move"][B_idx, ply_indices].unsqueeze(-1)

    elif probe_name == "is_check":
        return data["is_check"][B_idx, ply_indices].unsqueeze(-1)

    elif probe_name == "castling_rights":
        raw = data["castling_rights"][B_idx, ply_indices].long()
        bits = torch.arange(4, device=raw.device)
        return ((raw.unsqueeze(-1) >> bits) & 1).float()

    elif probe_name == "ep_square":
        ep = data["ep_square"][B_idx, ply_indices].long()
        return torch.where(ep < 0, 64, ep)

    elif probe_name == "material_count":
        boards = data["boards"][B_idx, ply_indices]  # (N, 8, 8)
        flat = boards.reshape(-1, 64)
        counts = []
        for piece_id in _WHITE_PIECES + _BLACK_PIECES:
            counts.append((flat == piece_id).sum(dim=-1).float())
        return torch.stack(counts, dim=-1)  # (N, 10)

    elif probe_name == "legal_move_count":
        return data["legal_move_counts"][B_idx, ply_indices].unsqueeze(-1)

    elif probe_name == "halfmove_clock":
        return data["halfmove_clock"][B_idx, ply_indices].unsqueeze(-1)

    elif probe_name == "game_phase":
        boards = data["boards"][B_idx, ply_indices]  # (N, 8, 8)
        flat = boards.reshape(-1, 64)
        # Count total pieces and non-pawn non-king pieces
        non_empty = (flat != 0).sum(dim=-1)
        is_pawn_or_king = (
            (flat == 1) | (flat == 6) | (flat == 7) | (flat == 12)
        )
        non_pawn_king = ((flat != 0) & ~is_pawn_or_king).sum(dim=-1)

        ply = ply_indices.float()
        # Opening: ply <= 20 and total pieces >= 28
        opening = (ply <= 20) & (non_empty >= 28)
        # Endgame: non-pawn non-king pieces <= 6
        endgame = non_pawn_king <= 6
        # Middlegame: everything else
        phase = torch.ones(len(ply_indices), dtype=torch.long)  # 1 = middlegame
        phase[opening] = 0  # 0 = opening
        phase[endgame & ~opening] = 2  # 2 = endgame
        return phase

    else:
        raise ValueError(f"Unknown probe: {probe_name}")


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------


def _forward_to_layer(
    model: PAWNCLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Run model forward only up to the needed layer.

    layer_idx=0: return embeddings (no transformer layers).
    layer_idx=i (1..n_layers): return output of transformer layer i-1.
    Avoids computing and storing unnecessary later layers.
    """
    x = model.embed(input_ids)
    if layer_idx == 0:
        return x

    T = input_ids.shape[1]
    causal = model.causal_mask[:T, :T]
    padding = attention_mask.unsqueeze(1).unsqueeze(2)
    mask = causal.unsqueeze(0) & padding
    rope_cos = model.rope_cos[:, :, :T, :]
    rope_sin = model.rope_sin[:, :, :T, :]

    for i in range(layer_idx):
        x = model.layers[i](x, rope_cos, rope_sin, mask)

    return x


@torch.no_grad()
def _extract_hidden_states(
    model: PAWNCLM,
    data: dict,
    device: str,
    layer_idx: int,
    max_batch: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract hidden states at one layer for all valid positions.

    Uses early-stop forward: only computes transformer layers up to
    layer_idx, avoiding wasted compute and GPU memory for later layers.

    Returns (h_valid, valid_mask) on CPU:
        h_valid: (N_valid, d_model) — hidden states at move positions
        valid_mask: (N, max_ply) bool — which positions are valid
    """
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"]
    N, T = input_ids.shape
    max_ply = data["boards"].shape[1]
    d_model = model.cfg.d_model

    # Pre-compute valid mask and total count to pre-allocate output
    game_lengths = data["game_lengths"]
    ply_grid = torch.arange(max_ply).unsqueeze(0)
    valid_mask = ply_grid < torch.from_numpy(game_lengths).long().unsqueeze(1)  # (N, max_ply)
    n_valid = int(valid_mask.sum())

    h_out = torch.empty(n_valid, d_model)
    offset = 0

    for start in range(0, N, max_batch):
        end = min(start + max_batch, N)
        B = end - start
        batch_ids = input_ids[start:end].to(device)
        batch_mask = loss_mask[start:end].to(device)

        h = _forward_to_layer(model, batch_ids, batch_mask, layer_idx)
        h = h[:, 1:max_ply + 1, :]  # (B, max_ply, d_model)

        batch_valid = valid_mask[start:end]  # (B, max_ply)
        h_flat = h.cpu().reshape(B * max_ply, d_model)
        m_flat = batch_valid.reshape(B * max_ply)
        n = int(m_flat.sum())
        h_out[offset:offset + n] = h_flat[m_flat]
        offset += n

        del h, batch_ids, batch_mask

    return h_out, valid_mask


def _extract_targets(
    probe_name: str, data: dict, valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract probe targets for all valid positions (CPU-only, fast).

    Uses direct (game, ply) indexing to avoid materializing a full
    (N, max_ply, ...) intermediate tensor.
    """
    g_idx, p_idx = valid_mask.nonzero(as_tuple=True)
    return get_probe_targets(probe_name, data, p_idx, g_idx)


# ---------------------------------------------------------------------------
# Loss and accuracy computation
# ---------------------------------------------------------------------------


def _compute_loss(logits: torch.Tensor, targets: torch.Tensor, loss_type: str, n_outputs: int) -> torch.Tensor:
    if loss_type == "ce":
        return F.cross_entropy(logits, targets)
    elif loss_type == "ce_per_square":
        return F.cross_entropy(logits.reshape(-1, 13), targets.reshape(-1))
    elif loss_type == "bce":
        return F.binary_cross_entropy_with_logits(logits, targets)
    elif loss_type == "mse":
        return F.mse_loss(logits, targets)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def _compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, loss_type: str, n_outputs: int) -> float:
    if loss_type == "ce":
        preds = logits.argmax(dim=-1)
        return (preds == targets).float().mean().item()
    elif loss_type == "ce_per_square":
        preds = logits.reshape(-1, 13).argmax(dim=-1)
        return (preds == targets.reshape(-1)).float().mean().item()
    elif loss_type == "bce":
        preds = (logits > 0).float()
        return (preds == targets).float().mean().item()
    elif loss_type == "mse":
        # For regression: R² score
        ss_res = ((logits - targets) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        return r2.item()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def _compute_mae(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean absolute error for regression probes."""
    return (logits - targets).abs().mean().item()


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------


def _train_probe_on_hidden(
    probe_name: str,
    train_h: torch.Tensor,
    train_t: torch.Tensor,
    val_h: torch.Tensor,
    val_t: torch.Tensor,
    d_model: int,
    device: str,
    n_epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> dict:
    """Train a linear probe given pre-extracted hidden states and targets.

    All tensors are on CPU; mini-batches are moved to GPU for training.
    """
    n_outputs, loss_type, _ = PROBES[probe_name]
    probe = LinearProbe(d_model, n_outputs).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)

    best_val_acc = -float("inf")
    n_train = len(train_h)

    for _epoch in range(n_epochs):
        perm = torch.randperm(n_train)

        probe.train()
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            h_b = train_h[idx].to(device)
            t_b = train_t[idx].to(device)
            logits = probe(h_b)
            loss = _compute_loss(logits, t_b, loss_type, n_outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_acc = _eval_in_batches(probe, val_h, val_t, loss_type, n_outputs, device, batch_size * 4)
        best_val_acc = max(best_val_acc, val_acc)

    probe.eval()
    with torch.no_grad():
        final_acc = _eval_in_batches(probe, val_h, val_t, loss_type, n_outputs, device, batch_size * 4)
        final_loss = _eval_loss_in_batches(probe, val_h, val_t, loss_type, n_outputs, device, batch_size * 4)
        result = {
            "accuracy": final_acc,
            "loss": final_loss,
            "best_accuracy": best_val_acc,
            "n_train": len(train_h),
            "n_val": len(val_h),
        }
        if loss_type == "mse":
            result["mae"] = _eval_mae_in_batches(probe, val_h, val_t, device, batch_size * 4)

    del probe, optimizer
    return result


def train_single_probe(
    model: PAWNCLM,
    probe_name: str,
    train_data: dict,
    val_data: dict,
    device: str,
    n_epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
    layer_idx: int = -1,
) -> dict:
    """Train a single probe (extracts hidden states internally).

    Convenience wrapper for one-off use. For per-layer sweeps, prefer
    train_all_probes which extracts hidden states once per layer.
    """
    model.eval()
    train_h, train_valid = _extract_hidden_states(model, train_data, device, layer_idx)
    val_h, val_valid = _extract_hidden_states(model, val_data, device, layer_idx)

    train_t = _extract_targets(probe_name, train_data, train_valid)
    val_t = _extract_targets(probe_name, val_data, val_valid)

    return _train_probe_on_hidden(
        probe_name, train_h, train_t, val_h, val_t,
        model.cfg.d_model, device, n_epochs, lr, batch_size,
    )


def _eval_in_batches(probe: LinearProbe, h: torch.Tensor, t: torch.Tensor, loss_type: str, n_outputs: int, device: str, batch_size: int) -> float:
    """Accuracy in mini-batches."""
    total_correct = 0.0
    total = 0
    for i in range(0, len(h), batch_size):
        h_b = h[i:i + batch_size].to(device)
        t_b = t[i:i + batch_size].to(device)
        n = len(h_b)
        acc = _compute_accuracy(probe(h_b), t_b, loss_type, n_outputs)
        total_correct += acc * n
        total += n
    return total_correct / total if total > 0 else 0.0


def _eval_loss_in_batches(probe: LinearProbe, h: torch.Tensor, t: torch.Tensor, loss_type: str, n_outputs: int, device: str, batch_size: int) -> float:
    """Loss in mini-batches (returns scalar)."""
    total_loss = 0.0
    total = 0
    for i in range(0, len(h), batch_size):
        h_b = h[i:i + batch_size].to(device)
        t_b = t[i:i + batch_size].to(device)
        n = len(h_b)
        loss = _compute_loss(probe(h_b), t_b, loss_type, n_outputs).item()
        total_loss += loss * n
        total += n
    return total_loss / total if total > 0 else 0.0


def _eval_mae_in_batches(probe: LinearProbe, h: torch.Tensor, t: torch.Tensor, device: str, batch_size: int) -> float:
    """MAE in mini-batches."""
    total_ae = 0.0
    total = 0
    for i in range(0, len(h), batch_size):
        h_b = h[i:i + batch_size].to(device)
        t_b = t[i:i + batch_size].to(device)
        total_ae += (probe(h_b) - t_b).abs().sum().item()
        total += len(h_b)
    return total_ae / total if total > 0 else 0.0


def train_all_probes(
    model: PAWNCLM,
    train_data: dict,
    val_data: dict,
    device: str,
    per_layer: bool = True,
    n_epochs: int = 20,
    lr: float = 1e-3,
    probe_names: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Train all probes per-layer.

    Loops LAYER-first, PROBE-second: hidden states are extracted once per
    layer and reused across all probes, eliminating redundant forward passes.

    Returns nested dict: results[probe_name][layer_name] = metrics
    """
    if probe_names is None:
        probe_names = [p for p in PROBES if p in PROBES]

    n_layers = model.cfg.n_layers
    layer_indices = list(range(n_layers + 1)) if per_layer else [n_layers]
    layer_names = (["embed"] + [f"layer_{i}" for i in range(n_layers)]) if per_layer else [f"layer_{n_layers - 1}"]

    model.eval()
    results = {pname: {} for pname in probe_names if pname in PROBES}

    for li, lname in zip(layer_indices, layer_names):
        if verbose:
            print(f"\n--- Layer: {lname} ---")

        # Free stale allocator caches before the next round of extractions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Extract hidden states ONCE for this layer
        train_h, train_valid = _extract_hidden_states(model, train_data, device, li)
        val_h, val_valid = _extract_hidden_states(model, val_data, device, li)

        # Train every probe on these hidden states
        for pname in probe_names:
            if pname not in PROBES:
                continue
            train_t = _extract_targets(pname, train_data, train_valid)
            val_t = _extract_targets(pname, val_data, val_valid)

            metrics = _train_probe_on_hidden(
                pname, train_h, train_t, val_h, val_t,
                model.cfg.d_model, device, n_epochs, lr,
            )
            results[pname][lname] = metrics

            if verbose:
                loss_type = PROBES[pname][1]
                if loss_type == "mse":
                    print(f"  {pname:>20s}: R²={metrics['best_accuracy']:.4f}  MAE={metrics.get('mae', 0):.3f}")
                else:
                    print(f"  {pname:>20s}: {metrics['best_accuracy']:.4f}")

            del train_t, val_t

        del train_h, val_h, train_valid, val_valid

    return results
