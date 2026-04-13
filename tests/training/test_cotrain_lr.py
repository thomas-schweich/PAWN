"""Regression tests for the cotrain LR plumbing.

Cotrain used to silently scale the user-supplied ``config.lr`` by
``batch_size / 256``, so passing ``lr=3e-4, batch_size=512`` produced
an effective LR of 6e-4 per variant. That behaviour was surprising and
asymmetric with ``scripts/train.py``'s pretrain path, which passed
``config.lr`` through verbatim. These tests lock in the new contract:
``_build_variant_configs`` uses ``shared.lr`` exactly, for every
variant and every batch size.
"""

from __future__ import annotations

import pytest

from pawn.cotrain import _build_variant_configs
from pawn.run_config import CotrainConfig, CotrainVariant


def _cfg(**overrides) -> CotrainConfig:
    return CotrainConfig(
        local_checkpoints=True,
        variants=[CotrainVariant(name=n, variant="toy") for n in ("a", "b", "c")],
        **overrides,
    )


@pytest.mark.unit
class TestCotrainLr:
    def test_default_lr_passes_through(self):
        cfg = _cfg(lr=3e-4, batch_size=256)
        for v in cfg.variants:
            _, train_cfg = _build_variant_configs(v, cfg, device="cpu")
            assert train_cfg.lr == pytest.approx(3e-4)

    def test_large_batch_does_not_scale_lr(self):
        """Historically batch_size=512 silently doubled the LR via
        ``scaled_lr = base_lr * (batch_size / 256)``. That's gone — the
        user's lr is what training sees."""
        cfg = _cfg(lr=3e-4, batch_size=512)
        for v in cfg.variants:
            _, train_cfg = _build_variant_configs(v, cfg, device="cpu")
            assert train_cfg.lr == pytest.approx(3e-4)

    def test_small_batch_does_not_shrink_lr(self):
        """Symmetric regression: batch_size=128 used to halve the LR."""
        cfg = _cfg(lr=3e-4, batch_size=128)
        for v in cfg.variants:
            _, train_cfg = _build_variant_configs(v, cfg, device="cpu")
            assert train_cfg.lr == pytest.approx(3e-4)

    def test_explicit_lr_preserved_at_any_batch(self):
        """Regardless of batch size, the user's explicit LR is used
        verbatim across every variant. If the user wants linear scaling
        they can compute it themselves and pass the result."""
        cfg = _cfg(lr=1e-3, batch_size=64)
        for v in cfg.variants:
            _, train_cfg = _build_variant_configs(v, cfg, device="cpu")
            assert train_cfg.lr == pytest.approx(1e-3)

    def test_all_variants_share_the_same_lr(self):
        """Cotrain is about architecture comparison — per-variant LR is
        intentionally constant."""
        cfg = _cfg(lr=5e-4, batch_size=256)
        lrs = [
            _build_variant_configs(v, cfg, device="cpu")[1].lr
            for v in cfg.variants
        ]
        assert lrs == [5e-4, 5e-4, 5e-4]
