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

import json
from pathlib import Path

import pytest

from pawn.cotrain import _build_variant_configs, _warn_cotrain_resume_lr_mismatch
from pawn.run_config import CotrainConfig, CotrainVariant


def _cfg(**overrides) -> CotrainConfig:
    return CotrainConfig(
        local_checkpoints=True,
        variants=[CotrainVariant(name=n, variant="toy") for n in ("a", "b", "c")],
        **overrides,
    )


def _write_ckpt(path: Path, saved_lr: float | None) -> None:
    """Write a minimal directory-format checkpoint config.json with a
    training_config block. Used for the resume-lr warning tests."""
    path.mkdir(parents=True, exist_ok=True)
    training: dict = {}
    if saved_lr is not None:
        training["lr"] = saved_lr
    with open(path / "config.json", "w") as f:
        json.dump({
            "format_version": 1,
            "checkpoint_type": "pretrain",
            "model_config": {"vocab_size": 1980, "max_seq_len": 512},
            "training_config": training,
        }, f)


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


@pytest.mark.unit
class TestCotrainResumeLrMismatchWarning:
    """Codex PR #65 review: removing LR scaling can silently change
    the effective LR on resumed checkpoints that were saved under the
    old scaling behaviour. The helper must warn loudly about the
    mismatch without overriding anything."""

    def test_matching_lr_no_warning(self, tmp_path, capsys):
        ckpt = tmp_path / "a"
        _write_ckpt(ckpt, saved_lr=3e-4)
        cfg = CotrainConfig(
            local_checkpoints=True,
            lr=3e-4,
            variants=[CotrainVariant(name="a", variant="toy", resume=str(ckpt))],
        )
        _warn_cotrain_resume_lr_mismatch(cfg)
        out = capsys.readouterr().out
        assert "WARNING" not in out

    def test_mismatch_warns(self, tmp_path, capsys):
        """Saved LR from the old scaling (6e-4 for batch_size=512) vs.
        the new run config's 3e-4 should produce a loud warning."""
        ckpt = tmp_path / "a"
        _write_ckpt(ckpt, saved_lr=6e-4)  # legacy-scaled
        cfg = CotrainConfig(
            local_checkpoints=True,
            lr=3e-4,
            batch_size=512,
            variants=[CotrainVariant(name="a", variant="toy", resume=str(ckpt))],
        )
        _warn_cotrain_resume_lr_mismatch(cfg)
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "6.00e-04" in out  # saved_lr
        assert "3.00e-04" in out  # config.lr

    def test_helper_does_not_mutate_config_lr(self, tmp_path):
        """The warning is pure diagnostic — no override."""
        ckpt = tmp_path / "a"
        _write_ckpt(ckpt, saved_lr=6e-4)
        cfg = CotrainConfig(
            local_checkpoints=True,
            lr=3e-4,
            variants=[CotrainVariant(name="a", variant="toy", resume=str(ckpt))],
        )
        _warn_cotrain_resume_lr_mismatch(cfg)
        assert cfg.lr == pytest.approx(3e-4)

    def test_missing_lr_in_saved_config_is_silent(self, tmp_path, capsys):
        """Old checkpoints without training.lr in config.json (edge
        case, shouldn't happen but defensive) produce no warning."""
        ckpt = tmp_path / "a"
        _write_ckpt(ckpt, saved_lr=None)
        cfg = CotrainConfig(
            local_checkpoints=True,
            lr=3e-4,
            variants=[CotrainVariant(name="a", variant="toy", resume=str(ckpt))],
        )
        _warn_cotrain_resume_lr_mismatch(cfg)
        out = capsys.readouterr().out
        assert "WARNING" not in out

    def test_non_resuming_variants_skipped(self, tmp_path, capsys):
        cfg = _cfg(lr=3e-4, batch_size=512)
        # None of the three variants are resuming, so nothing to check.
        _warn_cotrain_resume_lr_mismatch(cfg)
        out = capsys.readouterr().out
        assert "WARNING" not in out

    def test_broken_resume_path_is_silent(self, tmp_path, capsys):
        """Missing checkpoint dir → read_checkpoint_metadata raises;
        the LR-mismatch helper swallows it because a separate peek
        (_resolve_cotrain_resume_prepend_outcome) already warns loudly
        about broken resume paths."""
        cfg = CotrainConfig(
            local_checkpoints=True,
            lr=3e-4,
            variants=[CotrainVariant(
                name="a", variant="toy", resume=str(tmp_path / "nope"),
            )],
        )
        _warn_cotrain_resume_lr_mismatch(cfg)
        out = capsys.readouterr().out
        assert "WARNING" not in out

    def test_metadata_cache_avoids_second_read(self, tmp_path, mocker):
        """Codex round-2 regression: legacy .pt resumes trigger a full
        pickle deserialization inside read_checkpoint_metadata. The
        warn helper must reuse the cache populated by the prepend-peek
        helper instead of re-reading."""
        ckpt = tmp_path / "a"
        _write_ckpt(ckpt, saved_lr=6e-4)
        cfg = CotrainConfig(
            local_checkpoints=True,
            lr=3e-4,
            variants=[CotrainVariant(name="a", variant="toy", resume=str(ckpt))],
        )
        # Pre-populate the cache (simulating the first peek helper's
        # behaviour) and then patch read_checkpoint_metadata to blow up
        # if it's called again.
        cache = {
            "a": {
                "training_config": {"lr": 6e-4},
                "model_config": {"vocab_size": 1980, "max_seq_len": 512},
            },
        }
        spy = mocker.patch(
            "pawn.checkpoint.read_checkpoint_metadata",
            side_effect=AssertionError("should not reload from disk"),
        )
        _warn_cotrain_resume_lr_mismatch(cfg, metadata_cache=cache)
        spy.assert_not_called()

    def test_cache_miss_falls_back_to_filesystem(self, tmp_path, capsys):
        """If a variant is absent from the cache (e.g. the first peek
        failed to read it), the warn helper still does its own read.
        Keeps the helper robust for standalone/test use."""
        ckpt = tmp_path / "a"
        _write_ckpt(ckpt, saved_lr=6e-4)
        cfg = CotrainConfig(
            local_checkpoints=True,
            lr=3e-4,
            variants=[CotrainVariant(name="a", variant="toy", resume=str(ckpt))],
        )
        _warn_cotrain_resume_lr_mismatch(cfg, metadata_cache={})  # empty cache
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "6.00e-04" in out
