"""JAX runtime setup that needs to fire before any traced computation.

Today this is just the persistent compilation cache: every
:func:`eqx.filter_jit` traces, lowers, and compiles a fresh XLA
executable on first call. At LARGE supernet shape (``d=640, L=10,
d_ff=2560``) with ``unroll=n_layers=10``, the full K-step ``scan_step``
program is large enough that compilation runs in the 10-60 s range.
Without a persistent cache we pay that cost on **every** fresh process
— pod restart, lab trial spawn, Optuna trial, resume from checkpoint,
each `--variant` switch.

Enabling the cache is a 3-line change that the JAX runtime keys by
`(jaxlib version, GPU platform, HLO hash)`, so the cache hits across
runs whose model shape and dtype match. It's correctness-neutral and
the only failure mode is a stale-cache miss after a `jaxlib` or driver
upgrade (in which case JAX silently falls back to recompilation).

The cache directory is:
- ``$JAX_COMPILATION_CACHE_DIR`` if set, or
- ``$XDG_CACHE_HOME/jax-pawn`` if XDG is set, or
- ``~/.cache/jax-pawn`` otherwise.

Round-3 multi-agent review (Sonnet OOB / Sonnet conv / Opus conv all
flagged this independently as a free win).
"""

from __future__ import annotations

import os
from pathlib import Path


def setup_jax_caching(cache_dir: str | Path | None = None) -> Path | None:
    """Enable JAX's persistent compilation cache.

    Pass an explicit ``cache_dir`` to override the default; otherwise
    the env-var / XDG / ``~/.cache`` cascade picks the location. Returns
    the resolved cache path so callers can log it, or ``None`` if
    caching couldn't be set up.

    Must be called before any ``jax.jit`` / ``eqx.filter_jit`` compile.
    Idempotent — safe to call repeatedly.
    """
    import jax

    if cache_dir is None:
        env_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
        if env_dir:
            cache_dir = env_dir
        else:
            xdg = os.environ.get("XDG_CACHE_HOME")
            cache_dir = Path(xdg) / "jax-pawn" if xdg else Path.home() / ".cache" / "jax-pawn"

    cache_path = Path(cache_dir).expanduser().resolve()
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Read-only home, no XDG, container quirks — skip cache rather than crash.
        return None

    jax.config.update("jax_compilation_cache_dir", str(cache_path))
    # ``min_entry_size_bytes=-1`` accepts every compiled program (the
    # default is 1 MB which can skip small but slow-compiling ones).
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    # ``min_compile_time_secs=0`` accepts every program regardless of
    # compile time. We do our own filtering by setting the cache dir;
    # no need to second-guess at the JAX level.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    return cache_path
