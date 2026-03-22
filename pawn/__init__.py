"""PAWN: Playstyle-Agnostic World-model Network for Chess.

A causal transformer trained on random chess games, designed as an ideal
testbed for finetuning and augmentation methods at small scales.
"""

from pawn.config import CLMConfig, TrainingConfig
from pawn.model import PAWNCLM

__all__ = ["CLMConfig", "TrainingConfig", "PAWNCLM"]
