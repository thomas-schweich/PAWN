"""Unit tests for pawn.config: CLMConfig and TrainingConfig.

FROZEN MODULE — do not edit pawn/config.py to make a test pass.
If an assertion fails because the source is wrong, mark it xfail with a BUG-N.
"""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given, strategies as st

from pawn.config import (
    CLMConfig,
    TrainingConfig,
    PAD_TOKEN,
    OUTCOME_TOKEN_BASE,
    N_PRETRAINING_OUTCOMES,
    N_TOTAL_OUTCOMES,
    WHITE_CHECKMATES,
    BLACK_CHECKMATES,
    STALEMATE,
    DRAW_BY_RULE,
    PLY_LIMIT,
    WHITE_RESIGNS,
    BLACK_RESIGNS,
    DRAW_BY_AGREEMENT,
    WHITE_WINS_ON_TIME,
    BLACK_WINS_ON_TIME,
    DRAW_BY_TIME,
)


# ---------------------------------------------------------------------------
# Constants (must match engine/src/vocab.rs)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConstants:
    def test_pad_token(self):
        assert PAD_TOKEN == 1968

    def test_outcome_token_base(self):
        assert OUTCOME_TOKEN_BASE == 1969

    def test_pretraining_outcomes_sequence(self):
        assert WHITE_CHECKMATES == 1969
        assert BLACK_CHECKMATES == 1970
        assert STALEMATE == 1971
        assert DRAW_BY_RULE == 1972
        assert PLY_LIMIT == 1973

    def test_lichess_outcomes_sequence(self):
        assert WHITE_RESIGNS == 1974
        assert BLACK_RESIGNS == 1975
        assert DRAW_BY_AGREEMENT == 1976
        assert WHITE_WINS_ON_TIME == 1977
        assert BLACK_WINS_ON_TIME == 1978
        assert DRAW_BY_TIME == 1979

    def test_n_pretraining_outcomes(self):
        assert N_PRETRAINING_OUTCOMES == 5

    def test_n_total_outcomes(self):
        assert N_TOTAL_OUTCOMES == 11

    def test_outcome_count_consistency(self):
        """Pretraining + lichess = total."""
        pretrain = [WHITE_CHECKMATES, BLACK_CHECKMATES, STALEMATE,
                    DRAW_BY_RULE, PLY_LIMIT]
        lichess = [WHITE_RESIGNS, BLACK_RESIGNS, DRAW_BY_AGREEMENT,
                   WHITE_WINS_ON_TIME, BLACK_WINS_ON_TIME, DRAW_BY_TIME]
        assert len(pretrain) == N_PRETRAINING_OUTCOMES
        assert len(pretrain) + len(lichess) == N_TOTAL_OUTCOMES

    def test_outcome_tokens_contiguous(self):
        """Outcome tokens are 1969..1979 with no gaps."""
        all_outcomes = [WHITE_CHECKMATES, BLACK_CHECKMATES, STALEMATE,
                        DRAW_BY_RULE, PLY_LIMIT, WHITE_RESIGNS, BLACK_RESIGNS,
                        DRAW_BY_AGREEMENT, WHITE_WINS_ON_TIME,
                        BLACK_WINS_ON_TIME, DRAW_BY_TIME]
        expected = list(range(OUTCOME_TOKEN_BASE, OUTCOME_TOKEN_BASE + N_TOTAL_OUTCOMES))
        assert all_outcomes == expected


# ---------------------------------------------------------------------------
# CLMConfig variants
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCLMConfigVariants:
    def test_default_fields(self):
        cfg = CLMConfig()
        assert cfg.vocab_size == 1980
        assert cfg.max_seq_len == 512
        assert cfg.n_outcomes == N_TOTAL_OUTCOMES
        assert cfg.d_model == 512
        assert cfg.n_layers == 8
        assert cfg.n_heads == 8
        assert cfg.d_ff == 2048
        assert cfg.dropout == 0.0
        assert cfg.rope_base == 10000.0

    def test_vocab_size_is_sum_of_components(self):
        """1968 actions + 1 pad + 11 outcomes = 1980."""
        assert CLMConfig().vocab_size == 1968 + 1 + 11

    def test_small_variant(self):
        cfg = CLMConfig.small()
        assert cfg.d_model == 256
        assert cfg.n_layers == 8
        assert cfg.n_heads == 4
        assert cfg.d_ff == 1024

    def test_base_variant(self):
        cfg = CLMConfig.base()
        assert cfg.d_model == 512
        assert cfg.n_layers == 8
        assert cfg.n_heads == 8
        assert cfg.d_ff == 2048

    def test_large_variant(self):
        cfg = CLMConfig.large()
        assert cfg.d_model == 640
        assert cfg.n_layers == 10
        assert cfg.n_heads == 8
        assert cfg.d_ff == 2560

    def test_toy_variant(self):
        cfg = CLMConfig.toy()
        assert cfg.d_model == 64
        assert cfg.n_layers == 2
        assert cfg.n_heads == 4
        assert cfg.d_ff == 256

    def test_variants_are_distinct(self):
        small = CLMConfig.small()
        base = CLMConfig.base()
        large = CLMConfig.large()
        toy = CLMConfig.toy()
        configs = [small, base, large, toy]
        # d_model should be distinct for each
        dims = {c.d_model for c in configs}
        assert len(dims) == 4

    def test_d_model_divisible_by_n_heads_variants(self):
        """Every standard variant must have d_model % n_heads == 0."""
        for cfg in [CLMConfig(), CLMConfig.small(), CLMConfig.base(),
                    CLMConfig.large(), CLMConfig.toy()]:
            assert cfg.d_model % cfg.n_heads == 0, (
                f"d_model={cfg.d_model} not divisible by n_heads={cfg.n_heads}"
            )

    def test_d_ff_is_4x_d_model_for_standard_variants(self):
        for cfg in [CLMConfig(), CLMConfig.small(), CLMConfig.base(),
                    CLMConfig.large(), CLMConfig.toy()]:
            assert cfg.d_ff == cfg.d_model * 4, (
                f"d_ff={cfg.d_ff} != 4 * d_model={cfg.d_model}"
            )

    def test_default_matches_base(self):
        """CLMConfig() and CLMConfig.base() produce identical configs."""
        assert CLMConfig() == CLMConfig.base()


# ---------------------------------------------------------------------------
# Dataclass semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCLMConfigDataclass:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(CLMConfig)

    def test_asdict_roundtrip(self):
        """CLMConfig → dict → CLMConfig preserves all fields."""
        cfg = CLMConfig.base()
        d = dataclasses.asdict(cfg)
        cfg2 = CLMConfig(**d)
        assert cfg == cfg2

    def test_equality(self):
        assert CLMConfig.toy() == CLMConfig.toy()
        assert CLMConfig.small() != CLMConfig.base()

    def test_dict_field_has_expected_keys(self):
        cfg = CLMConfig.base()
        d = dataclasses.asdict(cfg)
        expected_keys = {
            "vocab_size", "max_seq_len", "n_outcomes", "d_model",
            "n_layers", "n_heads", "d_ff", "dropout", "rope_base",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Property-based roundtrip
# ---------------------------------------------------------------------------


_valid_head_counts = [1, 2, 4, 8, 16]


@pytest.mark.unit
@given(
    d_model_factor=st.integers(min_value=1, max_value=16),
    n_heads=st.sampled_from(_valid_head_counts),
    n_layers=st.integers(min_value=1, max_value=16),
    dropout=st.floats(min_value=0.0, max_value=0.9, allow_nan=False, allow_infinity=False),
)
def test_clm_config_roundtrip_property(d_model_factor, n_heads, n_layers, dropout):
    """Random CLMConfig → asdict → CLMConfig is identity."""
    d_model = d_model_factor * n_heads  # ensure divisible
    cfg = CLMConfig(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_model * 4, dropout=dropout,
    )
    d = dataclasses.asdict(cfg)
    cfg2 = CLMConfig(**d)
    assert cfg == cfg2


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrainingConfigDefaults:
    def test_default_fields(self):
        cfg = TrainingConfig()
        assert cfg.lr == 3e-4
        assert cfg.weight_decay == 0.01
        assert cfg.max_grad_norm == 1.0
        assert cfg.warmup_steps == 1000
        assert cfg.total_steps == 100_000
        assert cfg.batch_size == 256
        assert cfg.max_ply == 512
        assert cfg.discard_ply_limit is False
        assert cfg.num_workers == 4
        assert cfg.use_amp is True
        assert cfg.accumulation_steps == 1
        assert cfg.log_interval == 10
        assert cfg.eval_interval == 500
        assert cfg.checkpoint_interval == 5000
        assert cfg.pause_after_steps is None
        assert cfg.no_outcome_token is False
        assert cfg.prepend_outcome is False
        assert cfg.mate_boost == 0.0
        assert cfg.base_seed == 42
        assert cfg.val_seed == (2**63) - 1
        assert cfg.val_games == 512
        assert cfg.checkpoint_dir == "checkpoints"
        assert cfg.log_dir == "logs"
        assert cfg.use_wandb is False
        assert cfg.wandb_project == "pawn"
        assert cfg.device == "cuda"

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(TrainingConfig)

    def test_asdict_roundtrip(self):
        cfg = TrainingConfig()
        d = dataclasses.asdict(cfg)
        cfg2 = TrainingConfig(**d)
        assert cfg == cfg2


@pytest.mark.unit
class TestTrainingConfigToy:
    def test_toy_fields(self):
        cfg = TrainingConfig.toy()
        assert cfg.lr == 1e-3
        assert cfg.batch_size == 32
        assert cfg.total_steps == 5000
        assert cfg.warmup_steps == 100
        assert cfg.eval_interval == 100
        assert cfg.checkpoint_interval == 1000
        assert cfg.num_workers == 2
        assert cfg.use_amp is False
        assert cfg.val_games == 64

    def test_toy_distinct_from_default(self):
        assert TrainingConfig.toy() != TrainingConfig()
