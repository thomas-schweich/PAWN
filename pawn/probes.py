"""Linear probes — fit a linear classifier on frozen hidden states.

Probes the supernet's hidden representations at each layer for
position-feature prediction (square occupancy, piece type, etc.).
The trained probe is a single Linear over the layer's hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, Int

from pawn.model import PAWNModel

__all__ = [
    "ProbeConfig",
    "ProbeResult",
    "fit_probe",
]


@dataclass(frozen=True)
class ProbeConfig:
    """Linear probe hyperparameters."""

    n_classes: int
    lr: float = 1e-2
    n_epochs: int = 20
    batch_size: int = 256


@dataclass(frozen=True)
class ProbeResult:
    """Final accuracy + the trained linear weights."""

    accuracy: float
    weight: Float[Array, "d n_classes"]
    bias: Float[Array, "n_classes"]


def fit_probe(
    hidden_states: Float[Array, "N d"],
    labels: Int[Array, "N"],
    cfg: ProbeConfig,
    key: jax.Array | int = 0,
) -> ProbeResult:
    """Fit a single-layer linear classifier on hidden states.

    Hidden states are assumed to be already extracted from the frozen
    backbone (caller is responsible for the no-grad forward pass that
    produces them). The probe trains via Optax AdamW + cross-entropy.
    """
    if isinstance(key, int):
        key = jax.random.key(key)
    d = hidden_states.shape[-1]
    weight = jax.random.normal(key, (d, cfg.n_classes), dtype=jnp.float32) * 0.02
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
    n = hidden_states.shape[0]
    rng = np.random.default_rng(0)
    for _ in range(cfg.n_epochs):
        idx = rng.permutation(n)
        for s in range(0, n, cfg.batch_size):
            chunk = idx[s : s + cfg.batch_size]
            x_b = hidden_states[chunk]
            y_b = labels[chunk]
            params, state, _ = step(params, state, x_b, y_b)

    w, b = params
    # Final accuracy.
    logits = hidden_states @ w + b
    preds = jnp.argmax(logits, axis=-1)
    acc = float((preds == labels).mean())
    return ProbeResult(accuracy=acc, weight=w, bias=b)
