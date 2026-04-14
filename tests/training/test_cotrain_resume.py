"""Regression tests for the cotrain resume / sequence-format detection path.

Focuses on ``_resolve_cotrain_resume_prepend_outcome`` — the helper that
peeks at each resuming variant's saved checkpoint and rewrites
``config.prepend_outcome`` BEFORE ``ModelSlot`` is constructed, so each
slot's ``write_config_json`` reflects the actual sequence format.

These tests do NOT run a full cotrain loop; they verify the invariants
that the (very small) helper exposes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pawn.cotrain import (
    _resolve_cotrain_resume_prepend_outcome,
    run_post_training_evals,
)
from pawn.run_config import CotrainConfig, CotrainVariant


def _write_checkpoint(
    path: Path,
    *,
    prepend_outcome: bool | None,
    vocab_size: int = 1980,
    max_seq_len: int = 512,
) -> None:
    """Write a directory-format checkpoint config.json that
    ``read_checkpoint_metadata`` can parse. Training-config layout matches
    what ``save_pretrain_checkpoint`` writes.

    When ``prepend_outcome`` is None, the field is omitted — exercises the
    fail-closed branch of ``get_prepend_outcome``.
    """
    path.mkdir(parents=True, exist_ok=True)
    training: dict[str, Any] = {}
    if prepend_outcome is not None:
        training["prepend_outcome"] = prepend_outcome
    config = {
        "format_version": 1,
        "checkpoint_type": "pretrain",
        "model_config": {
            "vocab_size": vocab_size,
            "max_seq_len": max_seq_len,
            "d_model": 64,
            "n_layers": 2,
        },
        "training_config": training,
    }
    with open(path / "config.json", "w") as f:
        json.dump(config, f)


def _make_config(
    variants: list[CotrainVariant],
    prepend_outcome: bool,
) -> CotrainConfig:
    return CotrainConfig(
        local_checkpoints=True,
        prepend_outcome=prepend_outcome,
        variants=variants,
    )


@pytest.mark.unit
class TestResolveCotrainResumePrependOutcome:
    def test_no_resume_is_noop(self):
        cfg = _make_config(
            [CotrainVariant(name="a", variant="toy")],
            prepend_outcome=False,
        )
        _resolve_cotrain_resume_prepend_outcome(cfg)
        assert cfg.prepend_outcome is False

    def test_resume_matches_no_override(self, tmp_path):
        ckpt = tmp_path / "step_1"
        _write_checkpoint(ckpt, prepend_outcome=True)
        cfg = _make_config(
            [CotrainVariant(name="a", variant="toy", resume=str(ckpt))],
            prepend_outcome=True,
        )
        _resolve_cotrain_resume_prepend_outcome(cfg)
        assert cfg.prepend_outcome is True

    def test_resume_overrides_config(self, tmp_path, capsys):
        """User passed prepend_outcome=False but checkpoint was outcome-
        prefixed — helper must flip the config so downstream slot
        construction and config.json writes use True."""
        ckpt = tmp_path / "step_1"
        _write_checkpoint(ckpt, prepend_outcome=True)
        cfg = _make_config(
            [CotrainVariant(name="a", variant="toy", resume=str(ckpt))],
            prepend_outcome=False,
        )
        _resolve_cotrain_resume_prepend_outcome(cfg)
        assert cfg.prepend_outcome is True
        out = capsys.readouterr().out
        assert "overriding prepend_outcome" in out

    def test_mixed_format_errors(self, tmp_path):
        """Two variants, one outcome-prefixed and one pure-moves: cotrain
        can't share one pipeline for both, so helper must exit loudly."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        _write_checkpoint(a, prepend_outcome=True)
        _write_checkpoint(b, prepend_outcome=False)
        cfg = _make_config(
            [
                CotrainVariant(name="a", variant="toy", resume=str(a)),
                CotrainVariant(name="b", variant="toy", resume=str(b)),
            ],
            prepend_outcome=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            _resolve_cotrain_resume_prepend_outcome(cfg)
        assert exc_info.value.code == 1

    def test_missing_checkpoint_warns_and_skips(self, tmp_path, capsys):
        """A broken resume path shouldn't crash the launch — log a warning
        and fall back to the user-supplied flag."""
        cfg = _make_config(
            [CotrainVariant(
                name="a", variant="toy", resume=str(tmp_path / "nope"),
            )],
            prepend_outcome=False,
        )
        _resolve_cotrain_resume_prepend_outcome(cfg)
        # User-supplied value preserved.
        assert cfg.prepend_outcome is False
        out = capsys.readouterr().out
        assert "WARNING" in out

    def test_override_happens_before_variants_are_built(self, tmp_path):
        """This is the Codex P1 regression: the override must land on
        config.prepend_outcome itself, not just on already-built slots.
        Verifying via direct mutation is the next best thing short of
        running the full pipeline."""
        ckpt = tmp_path / "step_1"
        _write_checkpoint(ckpt, prepend_outcome=True)
        cfg = _make_config(
            [CotrainVariant(name="a", variant="toy", resume=str(ckpt))],
            prepend_outcome=False,
        )
        before = cfg.prepend_outcome
        _resolve_cotrain_resume_prepend_outcome(cfg)
        after = cfg.prepend_outcome
        # The CotrainConfig itself (not just individual slots) was
        # mutated, so any subsequent slot or dataset construction picks
        # up the corrected flag.
        assert before is False
        assert after is True

    def test_ambiguous_resume_without_explicit_flag_errors(self, tmp_path):
        """Checkpoint without saved prepend_outcome field → helper can't
        determine the sequence format. Without an explicit
        prepend_outcome in the run config, fail closed instead of
        silently defaulting."""
        ckpt = tmp_path / "step_1"
        _write_checkpoint(
            ckpt, prepend_outcome=None,
        )
        # Note: passing prepend_outcome at all marks the field as "set"
        # on the Pydantic model, so we use model_construct to simulate a
        # user who accepted the default.
        cfg = CotrainConfig.model_construct(
            local_checkpoints=True,
            variants=[
                CotrainVariant(name="a", variant="toy", resume=str(ckpt)),
            ],
        )
        cfg.prepend_outcome = False  # default value
        # Clear model_fields_set so the helper sees "not explicit".
        cfg.__pydantic_fields_set__.discard("prepend_outcome")
        with pytest.raises(SystemExit) as exc_info:
            _resolve_cotrain_resume_prepend_outcome(cfg)
        assert exc_info.value.code == 1

    def test_run_post_training_evals_signature(self):
        """Public API lock: run_post_training_evals takes the four
        kwargs the cotrain dispatch relies on. This catches accidental
        signature drift — the deleted scripts/train_all.py used to
        forward --run-evals / --lichess-pgn / --publish-results to this
        function, and the canonical entry point is now the cotrain
        config in scripts/train.py."""
        import inspect
        sig = inspect.signature(run_post_training_evals)
        params = set(sig.parameters)
        assert "slots" in params
        assert "device" in params
        assert "lichess_pgn" in params
        assert "publish_results" in params

    def test_metadata_cache_is_populated(self, tmp_path):
        """The helper should populate the passed-in metadata cache so
        downstream helpers can reuse it instead of triggering a second
        checkpoint load."""
        ckpt = tmp_path / "step_1"
        _write_checkpoint(ckpt, prepend_outcome=True)
        cfg = _make_config(
            [CotrainVariant(name="a", variant="toy", resume=str(ckpt))],
            prepend_outcome=True,
        )
        cache: dict = {}
        _resolve_cotrain_resume_prepend_outcome(cfg, metadata_cache=cache)
        assert "a" in cache
        assert cache["a"]["training_config"]["prepend_outcome"] is True

    def test_ambiguous_resume_with_explicit_flag_trusts_user(
        self, tmp_path, capsys,
    ):
        """If the user explicitly passed prepend_outcome, the helper
        trusts their value for ambiguous variants (still warns)."""
        ckpt = tmp_path / "step_1"
        _write_checkpoint(
            ckpt, prepend_outcome=None, vocab_size=1980, max_seq_len=512,
        )
        cfg = _make_config(
            [CotrainVariant(name="a", variant="toy", resume=str(ckpt))],
            prepend_outcome=True,  # explicit
        )
        # _make_config passes prepend_outcome to CotrainConfig, so it
        # ends up in model_fields_set automatically.
        assert "prepend_outcome" in cfg.model_fields_set
        _resolve_cotrain_resume_prepend_outcome(cfg)
        assert cfg.prepend_outcome is True
        out = capsys.readouterr().out
        assert "ambiguous" in out
        assert "WARNING" in out
