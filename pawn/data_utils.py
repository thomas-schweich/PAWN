"""Shared data utilities for PAWN training pipelines."""

import torch


_unpack_bits_cache: dict[torch.device, torch.Tensor] = {}


def unpack_grid(packed_grid: torch.Tensor) -> torch.Tensor:
    """Unpack bit-packed legal move grid to dense float targets.

    packed_grid: (..., 64) int64 — each value is a 64-bit destination mask
    Returns: (..., 64, 64) float32 — binary targets
    """
    device = packed_grid.device
    bits = _unpack_bits_cache.get(device)
    if bits is None:
        bits = torch.arange(64, device=device, dtype=torch.long)
        _unpack_bits_cache[device] = bits
    return ((packed_grid.unsqueeze(-1) >> bits) & 1).float()
