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
except ModuleNotFoundError as exc:
    # Only swallow a *missing torch*; any other ModuleNotFoundError points at
    # a real bug in pawn.model (typo, missing optional dep) that should not
    # silently disappear into a confusing downstream AttributeError.
    if exc.name != "torch":
        raise
    __all__ = ["CLMConfig", "TrainingConfig"]
