"""PAWN: Playstyle-Agnostic World-model Network for Chess.

A causal transformer trained on random chess games, designed as a
testbed for finetuning and augmentation methods at small scales.

The package surface is JAX-only as of Phase 4 of the JAX migration
(``docs/jax-migration.md``). Train + eval entry points live in
``pawn.model``, ``pawn.trainer``, ``pawn.adapter_trainer``,
``pawn.eval``, ``pawn.adapters.*`` and the
``scripts/train_jax*.py`` / ``scripts/eval_jax.py`` drivers.

External PyTorch users load published JAX checkpoints via the thin
loader at ``pawn.torch_loader.load_pawn``; that is the only path
that touches torch outside the legacy-converter parity fixture at
``pawn._torch_legacy_fixture``.
"""

__all__: list[str] = []
