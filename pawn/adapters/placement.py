"""Per-layer adapter placement — the shared ``--adapter-layers`` consumer.

Several strategies (LoRA, sparse, hybrid, RoSA, bottleneck) build a
per-layer correction stacked on a leading ``n_layers`` axis and apply it
to *every* transformer layer. v1 let the user restrict that to an explicit
subset of layers (``git show main:pawn/adapters/lora.py:88-97`` —
``self.adapted_layers = set(layers if layers is not None else range(...))``).
This module is the single owner of:

- :func:`parse_adapter_layers` — turn the ``"5,6,7"`` CLI string into the
  validated tuple of layer indices (mirrors
  :func:`pawn.adapters.unfreeze.parse_unfreeze_layers`, but bounds-checks
  against ``n_layers`` here because the count is known).
- :func:`layer_placement_mask` — build the ``(n_layers,)`` boolean mask
  (True at adapted layers).
- :func:`apply_layer_mask` — gate a stacked ``(n_layers, ...)`` correction
  so only the adapted layers contribute (non-adapted layers fall back to
  zero correction, and — because the masked slices are multiplied by a
  constant 0 — receive zero gradient, so they never learn).

Storing the mask as a boolean leaf (not an inexact array) keeps it out of
the trainable filter, exactly like the unfreeze ``layer_mask``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

__all__ = [
    "parse_adapter_layers",
    "layer_placement_mask",
    "apply_layer_mask",
]


def parse_adapter_layers(spec: str | None, n_layers: int) -> tuple[int, ...] | None:
    """Parse the ``--adapter-layers`` comma-separated string.

    Returns ``None`` when ``spec`` is ``None`` (the "all layers" default).
    Otherwise returns the sorted, de-duplicated tuple of layer indices,
    raising :class:`ValueError` on a malformed entry or an index outside
    ``[0, n_layers)``. Whitespace around commas is tolerated.
    """
    if spec is None:
        return None
    s = spec.strip()
    if not s:
        raise ValueError("adapter_layers must not be empty when set")
    picks: list[int] = []
    for part in s.split(","):
        p = part.strip()
        if not p or not p.isdigit():
            raise ValueError(
                f"adapter_layers must be comma-separated non-negative ints "
                f"(e.g. '5,6,7'), got {spec!r}"
            )
        idx = int(p)
        if not 0 <= idx < n_layers:
            raise ValueError(
                f"adapter_layers index {idx} outside [0, {n_layers})"
            )
        picks.append(idx)
    return tuple(sorted(set(picks)))


def layer_placement_mask(
    layers: tuple[int, ...] | None, n_layers: int
) -> Bool[Array, "n_layers"]:
    """``(n_layers,)`` boolean mask — True at the layers the adapter touches.

    ``None`` (the default) means "every layer", so the mask is all-True
    and :func:`apply_layer_mask` is a no-op. An explicit subset marks only
    those indices True.

    Every index is bound-checked against the *real* ``n_layers`` here, at
    the adapter-init call site that finally knows the backbone depth.
    ``parse_adapter_layers`` accepts the syntax against a permissive
    sentinel (the backbone isn't loaded yet at config-build time), so this
    is the only place an out-of-range pick is caught. Without it a JAX
    out-of-bounds scatter (default ``mode='drop'``) would *silently* drop
    the write — ``--adapter-layers 5`` on a 4-layer model would yield an
    all-False mask and an adapter that corrects nothing.
    """
    if layers is None:
        return jnp.ones((n_layers,), dtype=jnp.bool_)
    out_of_range = [idx for idx in layers if not 0 <= idx < n_layers]
    if out_of_range:
        raise ValueError(
            f"adapter_layers {out_of_range} outside [0, {n_layers}) for a "
            f"{n_layers}-layer backbone"
        )
    mask = jnp.zeros((n_layers,), dtype=jnp.bool_)
    if layers:
        mask = mask.at[jnp.array(layers)].set(True)
    return mask


def apply_layer_mask(
    correction: Float[Array, "n_layers ..."],
    layer_mask: Bool[Array, "n_layers"],
) -> Float[Array, "n_layers ..."]:
    """Zero a stacked correction at non-adapted layers.

    ``correction`` has a leading ``n_layers`` axis; ``layer_mask`` is
    ``(n_layers,)``. The mask is broadcast over the trailing axes and the
    correction is kept only where the mask is True. The masked slices
    multiply by a constant 0, so the autograd graph zeros their gradient —
    the adapter params at non-adapted layers never learn.
    """
    shape = (layer_mask.shape[0],) + (1,) * (correction.ndim - 1)
    m = layer_mask.reshape(shape)
    return jnp.where(m, correction, 0.0)
