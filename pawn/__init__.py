"""PAWN: Playstyle-Agnostic World-model Network for Chess.

A causal transformer trained on random chess games, designed as an ideal
testbed for finetuning and augmentation methods at small scales.
"""

from pawn.config import CLMConfig, TrainingConfig

# ``pawn.model`` pulls in PyTorch. During the JAX-migration window, importing
# ``pawn.jax.*`` should not force a torch install (e.g. JAX-only consumers).
# Re-export ``PAWNCLM`` when torch is available; otherwise leave it absent
# and let callers reach for ``from pawn.model import PAWNCLM`` explicitly.
try:
    from pawn.model import PAWNCLM
    __all__ = ["CLMConfig", "TrainingConfig", "PAWNCLM"]
except ImportError:
    __all__ = ["CLMConfig", "TrainingConfig"]
