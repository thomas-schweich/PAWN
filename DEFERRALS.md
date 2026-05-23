# Deferrals

> Per `docs/jax_migration_plan.md` §9.4, a deferral is legitimate only
> when actually doing the work is *nonsensical, impossible, or actively
> detrimental*. Things on the §3 acceptance-criteria list cannot be
> deferred under any reason.

No deferrals.

The framework swap implements the full §3 acceptance-criteria contract.
Every v1 surface that existed on `main` has a v2 counterpart on this
branch — including `scripts/benchmark.py`, which was rewritten against
the JAX/Equinox/Optax stack (jit vs eager backbone steps, fresh-corpus
vs pre-staged data-pipeline bench, multi-process JAX concurrency
sweep, JAX adapter bench).
