"""Tests for scripts/train_all.py's argparse → CotrainConfig bridging.

Narrow scope: exercising ``_args_to_cotrain_config`` directly with
Namespace objects. Locks in:
  - ``--lr`` flag exists and propagates into ``CotrainConfig.lr``.
  - Omitting ``--lr`` leaves the field unset in ``model_fields_set``
    so ambiguous-resume detection still trusts user-supplied explicit
    values over the default.
"""

from __future__ import annotations

import argparse

import pytest

# scripts/ is importable because the project's uv env adds it; but the
# safer thing is to import via path manipulation. Since pytest already
# runs from the repo root, the scripts dir is reachable as a sibling.
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "train_all", Path(__file__).resolve().parents[2] / "scripts" / "train_all.py",
)
assert _spec is not None and _spec.loader is not None
train_all = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_all)


def _defaults(**overrides) -> argparse.Namespace:
    """Minimum Namespace that _args_to_cotrain_config expects."""
    ns = argparse.Namespace(
        device=None,
        total_steps=100_000,
        batch_size=256,
        lr=None,
        num_workers=4,
        log_dir="logs",
        log_interval=10,
        eval_interval=500,
        checkpoint_interval=5000,
        discard_ply_limit=False,
        no_outcome_token=False,
        prepend_outcome=False,
        patience=10,
        wandb=False,
        legacy_vocab=False,
        hf_repo=None,
        local_checkpoints=True,
        shm_checkpoints=False,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


@pytest.mark.unit
class TestTrainAllLrFlag:
    def test_default_lr_not_in_model_fields_set(self):
        """When --lr is omitted, CotrainConfig.lr defaults to 3e-4 and
        the field is NOT marked as user-set. That matters because the
        ambiguous-resume detection in cotrain trusts explicit user
        values over the default."""
        cfg = train_all._args_to_cotrain_config(_defaults(lr=None))
        assert cfg.lr == pytest.approx(3e-4)  # BaseRunConfig default
        assert "lr" not in cfg.model_fields_set

    def test_explicit_lr_propagates(self):
        cfg = train_all._args_to_cotrain_config(_defaults(lr=1.5e-3))
        assert cfg.lr == pytest.approx(1.5e-3)
        assert "lr" in cfg.model_fields_set

    def test_explicit_lr_at_large_batch_is_not_scaled(self):
        """Regression: passing --lr 3e-4 --batch-size 1024 must produce
        a CotrainConfig with lr=3e-4 exactly. The old scaling behavior
        would have produced an effective 1.2e-3 per variant."""
        cfg = train_all._args_to_cotrain_config(
            _defaults(lr=3e-4, batch_size=1024),
        )
        assert cfg.lr == pytest.approx(3e-4)
        assert cfg.batch_size == 1024
