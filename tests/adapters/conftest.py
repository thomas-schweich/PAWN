"""Partition F (Adapters) local fixtures.

Owned by Partition F. Fixtures for frozen-backbone snapshots, adapter
config factories, and zero-init verification helpers.
"""

from __future__ import annotations

import pytest
import torch

from pawn.config import CLMConfig
from pawn.model import PAWNCLM


@pytest.fixture
def toy_backbone(toy_clm_config):
    """Fresh PAWNCLM(toy) on CPU — not session scoped since adapters mutate it in-place."""
    torch.manual_seed(0)
    model = PAWNCLM(toy_clm_config)
    model.eval()
    return model


@pytest.fixture
def toy_backbone_fresh(toy_clm_config):
    """Second independent backbone for state-dict roundtrip tests."""
    torch.manual_seed(0)
    model = PAWNCLM(toy_clm_config)
    model.eval()
    return model


@pytest.fixture
def toy_input_ids(toy_clm_config):
    """Deterministic input_ids tensor (B=2, T=16) suitable for toy backbones."""
    torch.manual_seed(123)
    return torch.randint(1, toy_clm_config.vocab_size, (2, 16))


@pytest.fixture
def toy_attention_mask():
    """Bool mask matching toy_input_ids (all real tokens)."""
    return torch.ones(2, 16, dtype=torch.bool)


def backbone_output(backbone: PAWNCLM, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the bare backbone logits (pre-adapter) for comparison."""
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    with torch.no_grad():
        logits, _ = backbone(input_ids, attention_mask)
    return logits
