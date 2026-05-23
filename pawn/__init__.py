"""PAWN: Playstyle-Agnostic World-model Network for Chess.

A causal transformer trained on random chess games, designed as a
testbed for finetuning and augmentation methods at small scales.

This package's top-level ``__init__`` is **deliberately empty** beyond
this docstring. Sub-modules (``pawn.config``, ``pawn.model``,
``pawn.checkpoint``, ``pawn.legacy``, ``pawn._sentinel``, …) are
imported explicitly by the code that needs them, so ``import pawn`` is
side-effect-free and cheap. Lightweight consumers — the dashboard, the
sweep driver, the lab MCP server, the legacy-checkpoint converter —
can import ``pawn._sentinel`` or ``pawn.config`` without paying for
the full JAX / Equinox / Optax import graph.
"""
