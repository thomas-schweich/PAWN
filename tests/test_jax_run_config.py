"""Tests for :mod:`pawn.run_config` — the pydantic keystone.

This module is the public surface every downstream JAX training driver,
sweep tool, and the lab MCP server reads through. Tests pin:

- The shape of each config (field names + defaults match v1 + plan §7).
- `extra="forbid"` rejects unknown / stale fields.
- `model_json_schema()` produces valid JSON Schema (consumed by lab).
- A `--config <json>` round-trip is lossless: dump → load → model_dump()
  matches the original.
- Every cross-field validator has at least one happy-path and one
  failure-path test.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from pawn.config import MAX_SEQ_LEN
from pawn.run_config import (
    AdapterConfig,
    DistillConfig,
    PretrainConfig,
    RunConfig,
    SpecializedCLMConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pretrain_kwargs(**overrides: Any) -> dict[str, Any]:
    """Minimal valid PretrainConfig kwargs (must satisfy
    BaseRunConfig._check_checkpoint_mode AND
    PretrainConfig._check_pretrain — which requires total_steps)."""
    base: dict[str, Any] = {"local_checkpoints": True, "total_steps": 100}
    base.update(overrides)
    return base


def _adapter_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "local_checkpoints": True,
        "total_steps": 100,
        "strategy": "lora",
        "lora_rank": 4,
    }
    base.update(overrides)
    return base


def _specialized_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "local_checkpoints": True,
        "total_steps": 100,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 2,
        "d_ff": 128,
    }
    base.update(overrides)
    return base


def _distill_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "local_checkpoints": True,
        "total_steps": 100,
        "distill_from": "thomas-schweich/pawn-base-v2",
        "student_supernet": "tiny",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Shape: durable v1 field names + plan §10 S3 defaults
# ---------------------------------------------------------------------------


def test_pretrain_minimal_valid() -> None:
    cfg = PretrainConfig(**_pretrain_kwargs())
    assert cfg.run_type == "pretrain"
    assert cfg.variant == "base"
    assert cfg.supernet == "production"
    assert cfg.seq_len == 512
    assert cfg.k == 50


def test_adapter_minimal_valid() -> None:
    cfg = AdapterConfig(**_adapter_kwargs())
    assert cfg.run_type == "adapter"
    assert cfg.strategy == "lora"
    assert cfg.lora_rank == 4
    assert cfg.variant == "base"


def test_specialized_minimal_valid() -> None:
    cfg = SpecializedCLMConfig(**_specialized_kwargs())
    assert cfg.run_type == "specialized_clm"
    assert cfg.d_model == 64


def test_distill_minimal_valid() -> None:
    cfg = DistillConfig(**_distill_kwargs())
    assert cfg.run_type == "distill"
    assert cfg.distill_from == "thomas-schweich/pawn-base-v2"
    assert cfg.student_supernet == "tiny"
    assert cfg.objective == "mix"
    assert cfg.temperature == 2.0
    assert cfg.alpha == 0.5


def test_distill_explicit_student_dims_valid() -> None:
    cfg = DistillConfig(**_distill_kwargs(
        student_supernet=None, d_model=64, n_layers=2, n_heads=2, d_ff=128,
    ))
    assert cfg.d_model == 64 and cfg.student_supernet is None


def test_distill_rejects_preset_and_explicit_dims() -> None:
    with pytest.raises(ValueError, match="not both"):
        DistillConfig(**_distill_kwargs(d_model=64))


def test_distill_rejects_no_student_arch() -> None:
    with pytest.raises(ValueError, match="student_supernet or all four"):
        DistillConfig(**_distill_kwargs(student_supernet=None))


def test_distill_rejects_partial_explicit_dims() -> None:
    with pytest.raises(ValueError, match="all four"):
        DistillConfig(**_distill_kwargs(
            student_supernet=None, d_model=64, n_layers=2,
        ))


def test_distill_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        DistillConfig(**_distill_kwargs(objective="kl", temperature=0.0))


def test_distill_rejects_alpha_out_of_unit_interval() -> None:
    with pytest.raises(ValueError, match="alpha must be in"):
        DistillConfig(**_distill_kwargs(objective="mix", alpha=1.5))


def test_distill_round_trips_through_json() -> None:
    cfg = DistillConfig(**_distill_kwargs(objective="kl", temperature=3.0))
    dumped = json.dumps(cfg.model_dump())
    reloaded = DistillConfig(**json.loads(dumped))
    assert reloaded.model_dump() == cfg.model_dump()


def test_distill_rejects_unknown_field() -> None:
    with pytest.raises(ValueError):
        DistillConfig(**_distill_kwargs(distil_temp=2.0))


def test_use_output_film_default_is_true() -> None:
    """Plan §10 S3 explicitly: default True for v2 (v1 default was False
    but the field name is preserved per §7)."""
    cfg = AdapterConfig(**_adapter_kwargs())
    assert cfg.use_output_film is True


def test_v1_field_names_preserved() -> None:
    """Plan §7 ("backward-compatibility contract") + §10 S3 enumerate the
    v1 field names that must keep working. Pin them explicitly so a
    silent rename surfaces here."""
    cfg = AdapterConfig(**_adapter_kwargs(strategy="rosa", rosa_mode="rosa"))
    fields = AdapterConfig.model_fields
    # lora_rank (not "rank")
    assert "lora_rank" in fields
    # density (not "sparse_density")
    assert "density" in fields
    # no_adapt_attn / no_adapt_ffn (not "bottleneck_no_attn"/"_ffn")
    assert "no_adapt_attn" in fields
    assert "no_adapt_ffn" in fields
    # unfreeze_layers (str — comma-sep), not a top-N count
    assert fields["unfreeze_layers"].annotation == (str | None)
    # rosa_mode is the trio Literal
    assert cfg.rosa_mode == "rosa"
    # rosa_warmup_steps / mask_samples / grad_alpha — v1 RoSA hyperparams
    assert "rosa_warmup_steps" in fields
    assert "mask_samples" in fields
    assert "grad_alpha" in fields


def test_specialized_clm_uses_bare_arch_fields() -> None:
    """Plan §10 S3: bare d_model / n_layers / n_heads / d_ff (no
    specialized_ prefix) inside SpecializedCLMConfig."""
    fields = SpecializedCLMConfig.model_fields
    assert "d_model" in fields
    assert "specialized_d_model" not in fields
    assert "n_layers" in fields
    assert "n_heads" in fields
    assert "d_ff" in fields


def test_torch_fields_dropped() -> None:
    """`device` / `num_workers` / `no_compile` / `sdpa_math` are JAX-
    irrelevant and stay dropped on the v2 BaseRunConfig.

    `amp_dtype` is *retained* per the parity audit
    (docs/JAX_PARITY_SHORTFALLS.md §1): v2 honours bf16 mixed-precision
    forward compute per plan §5. Covered by
    `test_amp_dtype_field_present_and_defaults_to_bfloat16`.
    """
    fields = set(PretrainConfig.model_fields)
    for absent in ("device", "num_workers", "no_compile", "sdpa_math"):
        assert absent not in fields, f"{absent} should have been dropped"


def test_amp_dtype_field_present_and_defaults_to_bfloat16() -> None:
    """v1 parity per docs/JAX_PARITY_SHORTFALLS.md §1: `amp_dtype` is a
    BaseRunConfig field, defaults to bf16 (matches v1 + plan §5), and
    rejects anything outside {bfloat16, float16, float32}."""
    cfg = PretrainConfig(**_pretrain_kwargs())
    assert cfg.amp_dtype == "bfloat16"
    cfg = PretrainConfig(**_pretrain_kwargs(amp_dtype="float32"))
    assert cfg.amp_dtype == "float32"
    with pytest.raises(ValueError, match="amp_dtype"):
        PretrainConfig(
            **_pretrain_kwargs(amp_dtype="int8")  # type: ignore[arg-type]
        )


def test_use_sdpa_field_defaults_off() -> None:
    """Parity #43 follow-up: ``use_sdpa`` is a bool field that defaults
    to False (keep the bit-stable baseline) and accepts True to opt
    into the :func:`jax.nn.dot_product_attention` path."""
    cfg = PretrainConfig(**_pretrain_kwargs())
    assert cfg.use_sdpa is False
    cfg = PretrainConfig(**_pretrain_kwargs(use_sdpa=True))
    assert cfg.use_sdpa is True


def test_supernet_field_options() -> None:
    """`supernet` is a Literal['tiny','production']."""
    cfg = PretrainConfig(**_pretrain_kwargs(supernet="tiny"))
    assert cfg.supernet == "tiny"
    with pytest.raises(ValueError):
        PretrainConfig(**_pretrain_kwargs(supernet="invalid"))  # type: ignore[arg-type]


def test_rosa_mode_all_three_modes_accepted() -> None:
    """Plan §6 'specifically and non-negotiably' includes retro-sparse +
    retro-bottleneck."""
    for mode in ("rosa", "retro-sparse", "retro-bottleneck"):
        cfg = AdapterConfig(
            **_adapter_kwargs(strategy="rosa", rosa_mode=mode)
        )
        assert cfg.rosa_mode == mode


# ---------------------------------------------------------------------------
# extra='forbid' — keystone defense against stale fields
# ---------------------------------------------------------------------------


def test_base_run_config_rejects_unknown_field() -> None:
    """`extra='forbid'` rejects stale CLI/JSON fields — the keystone defense
    against silently dropping renamed flags."""
    with pytest.raises(ValueError, match="extra"):
        PretrainConfig(**_pretrain_kwargs(legacy_vocab=True))  # type: ignore[arg-type]


def test_adapter_rejects_typo_in_lora_field() -> None:
    """Even close-to-correct typos like `lora_ranks` are rejected."""
    with pytest.raises(ValueError, match="extra"):
        AdapterConfig(**_adapter_kwargs(lora_ranks=4))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# JSON Schema generation (consumed by lab_schema)
# ---------------------------------------------------------------------------


def test_pretrain_model_json_schema_is_valid_dict() -> None:
    """`model_json_schema()` returns a JSON Schema dict the lab MCP
    server can hand to clients."""
    schema = PretrainConfig.model_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "total_steps" in schema["properties"]
    # Should be JSON-serializable
    assert json.loads(json.dumps(schema))


def test_adapter_model_json_schema_includes_strategy_enum() -> None:
    schema = AdapterConfig.model_json_schema()
    assert "properties" in schema
    strategy_field = schema["properties"]["strategy"]
    # Pydantic encodes Literal[...] as enum. All 10 keys in
    # `pawn.adapter_trainer.STRATEGIES` must be admissible — the three
    # RoSA modes (`rosa`, `rosa-retro-sparse`, `rosa-retro-bottleneck`)
    # are distinct CLI strategies per the plan adapter table.
    assert "enum" in strategy_field
    assert set(strategy_field["enum"]) == {
        "bottleneck", "lora", "film", "sparse",
        "rosa", "rosa-retro-sparse", "rosa-retro-bottleneck",
        "hybrid", "specialized_clm", "unfreeze",
    }


def test_specialized_model_json_schema_includes_required_arch() -> None:
    schema = SpecializedCLMConfig.model_json_schema()
    # d_model / n_layers / n_heads / d_ff are required (no defaults).
    assert "required" in schema
    for name in ("d_model", "n_layers", "n_heads", "d_ff"):
        assert name in schema["required"], f"{name} should be required"


# ---------------------------------------------------------------------------
# Round-trip: dump → load → model_dump() identity
# ---------------------------------------------------------------------------


def test_pretrain_round_trips_through_json() -> None:
    """A PretrainConfig serialised to JSON and reloaded is bit-identical
    (this is the contract `--config <json>` relies on)."""
    cfg = PretrainConfig(**_pretrain_kwargs(total_steps=1234, supernet="tiny"))
    dumped = cfg.model_dump()
    serialised = json.dumps(dumped)
    reloaded = PretrainConfig(**json.loads(serialised))
    assert reloaded.model_dump() == dumped


def test_adapter_round_trips_through_json() -> None:
    cfg = AdapterConfig(
        **_adapter_kwargs(
            strategy="lora",
            lora_rank=8,
            lora_targets="qkvo",
            elo_min=1800,
            elo_max=2000,
        )
    )
    dumped = cfg.model_dump()
    reloaded = AdapterConfig(**json.loads(json.dumps(dumped)))
    assert reloaded.model_dump() == dumped


def test_specialized_round_trips_through_json() -> None:
    cfg = SpecializedCLMConfig(
        **_specialized_kwargs(d_model=128, n_heads=4, d_ff=512)
    )
    dumped = cfg.model_dump()
    reloaded = SpecializedCLMConfig(**json.loads(json.dumps(dumped)))
    assert reloaded.model_dump() == dumped


# ---------------------------------------------------------------------------
# Cross-field validators: happy + failure paths
# ---------------------------------------------------------------------------


# _check_checkpoint_mode -----------------------------------------------------


def test_checkpoint_mode_requires_one_destination() -> None:
    with pytest.raises(ValueError, match="hf_repo.*hf_bucket.*local_checkpoints"):
        PretrainConfig(total_steps=100)  # no checkpoint destination
    # Happy path
    PretrainConfig(local_checkpoints=True, total_steps=100)
    PretrainConfig(hf_repo="thomas-schweich/scratch", total_steps=100)
    PretrainConfig(hf_bucket="ns/bucket", total_steps=100)


def test_checkpoint_mode_rejects_hf_repo_plus_local() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        PretrainConfig(
            hf_repo="thomas-schweich/scratch",
            local_checkpoints=True,
            total_steps=100,
        )


# _check_lr_schedule_fractions ----------------------------------------------


def test_lr_fractions_happy_wsd() -> None:
    PretrainConfig(
        **_pretrain_kwargs(
            lr_schedule="wsd", warmup_frac=0.05, decay_frac=0.1
        )
    )


def test_lr_fractions_wsd_oversum_rejected() -> None:
    with pytest.raises(ValueError, match="wsd.*warmup_frac.*decay_frac"):
        PretrainConfig(
            **_pretrain_kwargs(
                lr_schedule="wsd", warmup_frac=0.6, decay_frac=0.5
            )
        )


def test_lr_fractions_happy_infinite() -> None:
    PretrainConfig(
        **_pretrain_kwargs(
            lr_schedule="infinite",
            warmup_frac=0.05,
            cooldown_frac=0.2,
            decay_frac=0.1,
            stable_lr_ratio=0.1,
        )
    )


def test_lr_fractions_infinite_oversum_rejected() -> None:
    with pytest.raises(ValueError, match="infinite.*warmup_frac"):
        PretrainConfig(
            **_pretrain_kwargs(
                lr_schedule="infinite",
                warmup_frac=0.4,
                cooldown_frac=0.4,
                decay_frac=0.4,
            )
        )


def test_lr_fractions_infinite_rejects_zero_stable_lr_ratio() -> None:
    with pytest.raises(ValueError, match="stable_lr_ratio"):
        PretrainConfig(
            **_pretrain_kwargs(lr_schedule="infinite", stable_lr_ratio=0.0)
        )


def test_lr_fractions_warmup_frac_must_be_in_unit_interval() -> None:
    with pytest.raises(ValueError, match="warmup_frac"):
        PretrainConfig(**_pretrain_kwargs(warmup_frac=1.5))
    with pytest.raises(ValueError, match="warmup_frac"):
        PretrainConfig(**_pretrain_kwargs(warmup_frac=-0.1))


def test_lr_fractions_decay_frac_must_be_in_unit_interval() -> None:
    with pytest.raises(ValueError, match="decay_frac"):
        PretrainConfig(**_pretrain_kwargs(decay_frac=-0.5))
    with pytest.raises(ValueError, match="decay_frac"):
        PretrainConfig(**_pretrain_kwargs(decay_frac=1.5))


def test_lr_fractions_cooldown_frac_must_be_in_unit_interval() -> None:
    with pytest.raises(ValueError, match="cooldown_frac"):
        PretrainConfig(**_pretrain_kwargs(cooldown_frac=-0.5))
    with pytest.raises(ValueError, match="cooldown_frac"):
        PretrainConfig(**_pretrain_kwargs(cooldown_frac=1.5))


# _check_positive_training_dims --------------------------------------------


def test_seq_len_must_be_positive() -> None:
    with pytest.raises(ValueError, match="seq_len"):
        PretrainConfig(**_pretrain_kwargs(seq_len=0))


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="^k must be positive|k must be positive"):
        PretrainConfig(**_pretrain_kwargs(k=0))


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        PretrainConfig(**_pretrain_kwargs(batch_size=0))


def test_max_corpus_gb_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_corpus_gb"):
        PretrainConfig(**_pretrain_kwargs(max_corpus_gb=0))


# Conditioning + seq_len budget (Chunk 4) -----------------------------------


def test_conditioning_default_empty_gives_C_one() -> None:
    """No conditioning → BOS-only prefix, C == 1."""
    cfg = PretrainConfig(**_pretrain_kwargs())
    assert cfg.conditioning == []
    assert cfg.C == 1


def test_conditioning_outcome_gives_C_two() -> None:
    """conditioning=["outcome"] → C == 1 + 1 == 2."""
    cfg = PretrainConfig(**_pretrain_kwargs(conditioning=["outcome"]))
    assert cfg.conditioning == ["outcome"]
    assert cfg.C == 2


def test_conditioning_unknown_kind_rejected() -> None:
    """A kind not in CONDITIONING_KINDS is rejected with the registry
    message."""
    with pytest.raises(ValueError, match="unknown conditioning kind"):
        PretrainConfig(**_pretrain_kwargs(conditioning=["bogus"]))


def test_conditioning_duplicate_kind_rejected() -> None:
    """A repeated kind would silently double-condition; reject it."""
    with pytest.raises(ValueError, match="duplicate conditioning kind"):
        PretrainConfig(**_pretrain_kwargs(conditioning=["outcome", "outcome"]))


def test_prepend_outcome_true_migrates_to_conditioning_with_warning() -> None:
    """The legacy `prepend_outcome=True` key migrates to
    `conditioning=["outcome"]` and emits a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="prepend_outcome"):
        cfg = PretrainConfig(**_pretrain_kwargs(prepend_outcome=True))
    assert cfg.conditioning == ["outcome"]
    assert cfg.C == 2


def test_prepend_outcome_false_migrates_to_empty_conditioning() -> None:
    """`prepend_outcome=False` migrates to an empty conditioning list."""
    with pytest.warns(DeprecationWarning, match="prepend_outcome"):
        cfg = PretrainConfig(**_pretrain_kwargs(prepend_outcome=False))
    assert cfg.conditioning == []
    assert cfg.C == 1


def test_prepend_outcome_and_conditioning_both_set_rejected() -> None:
    """Passing both the legacy key and the new list is ambiguous → error."""
    with pytest.raises(ValueError, match="not both"):
        PretrainConfig(
            **_pretrain_kwargs(prepend_outcome=True, conditioning=["outcome"])
        )


def test_seq_len_exceeding_max_seq_len_rejected() -> None:
    """`seq_len > MAX_SEQ_LEN` is rejected by the net-new budget validator."""
    with pytest.raises(ValueError, match="MAX_SEQ_LEN"):
        PretrainConfig(**_pretrain_kwargs(seq_len=MAX_SEQ_LEN + 1))


def test_seq_len_must_exceed_conditioning_prefix_width() -> None:
    """`seq_len <= C` leaves no move slot — rejected. With
    conditioning=["outcome"], C=2, so seq_len=2 is too small."""
    with pytest.raises(ValueError, match="prefix width"):
        PretrainConfig(**_pretrain_kwargs(seq_len=2, conditioning=["outcome"]))


def test_seq_len_budget_happy_case_with_conditioning() -> None:
    """A seq_len that fits the prefix plus at least one move slot is
    accepted."""
    cfg = PretrainConfig(**_pretrain_kwargs(seq_len=64, conditioning=["outcome"]))
    assert cfg.seq_len == 64
    assert cfg.C == 2


def test_total_steps_must_be_positive_when_set() -> None:
    """`total_steps` is required by PretrainConfig (PR #115 review #5
    moved the runtime check into the model_validator). Zero or
    negative is a typo. None now also raises rather than slipping
    through to a `print + return 2` in the script."""
    with pytest.raises(ValueError, match="total_steps"):
        PretrainConfig(local_checkpoints=True, total_steps=None)
    with pytest.raises(ValueError, match="total_steps"):
        PretrainConfig(**_pretrain_kwargs(total_steps=0))


def test_val_games_must_be_positive() -> None:
    with pytest.raises(ValueError, match="val_games"):
        PretrainConfig(**_pretrain_kwargs(val_games=0))


def test_log_interval_must_be_positive() -> None:
    with pytest.raises(ValueError, match="log_interval"):
        PretrainConfig(**_pretrain_kwargs(log_interval=0))


def test_weight_decay_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="weight_decay"):
        PretrainConfig(**_pretrain_kwargs(weight_decay=-0.1))


def test_lr_must_be_positive() -> None:
    with pytest.raises(ValueError, match="lr"):
        PretrainConfig(**_pretrain_kwargs(lr=0))


def test_adapter_cadence_must_be_positive() -> None:
    """epochs / val_every / checkpoint_interval are downstream loop
    bounds; zero or negative breaks the trainer's modulo logic."""
    for field in ("epochs", "val_every", "checkpoint_interval"):
        with pytest.raises(ValueError, match=field):
            AdapterConfig(**_adapter_kwargs(**{field: 0}))


# Pretrain-specific validator -----------------------------------------------


def test_pretrain_accumulation_steps_positive() -> None:
    with pytest.raises(ValueError, match="accumulation_steps"):
        PretrainConfig(**_pretrain_kwargs(accumulation_steps=0))


def test_pretrain_checkpoint_interval_positive() -> None:
    with pytest.raises(ValueError, match="checkpoint_interval"):
        PretrainConfig(**_pretrain_kwargs(checkpoint_interval=0))


# Adapter strategy-input validator ------------------------------------------


def test_adapter_lora_requires_rank() -> None:
    with pytest.raises(ValueError, match="lora_rank"):
        AdapterConfig(local_checkpoints=True, total_steps=100, strategy="lora")


def test_adapter_lora_rank_must_be_positive() -> None:
    with pytest.raises(ValueError, match="lora_rank"):
        AdapterConfig(local_checkpoints=True, total_steps=100, strategy="lora", lora_rank=0)


def test_adapter_sparse_requires_density() -> None:
    with pytest.raises(ValueError, match="density"):
        AdapterConfig(local_checkpoints=True, total_steps=100, strategy="sparse")


def test_adapter_sparse_density_in_unit_interval() -> None:
    with pytest.raises(ValueError, match="density"):
        AdapterConfig(
            local_checkpoints=True, total_steps=100, strategy="sparse", density=1.5
        )


def test_adapter_bottleneck_requires_dim() -> None:
    with pytest.raises(ValueError, match="bottleneck_dim"):
        AdapterConfig(local_checkpoints=True, total_steps=100, strategy="bottleneck")


def test_adapter_bottleneck_dim_positive() -> None:
    with pytest.raises(ValueError, match="bottleneck_dim"):
        AdapterConfig(
            local_checkpoints=True, total_steps=100, strategy="bottleneck", bottleneck_dim=0
        )


def test_adapter_rosa_requires_mode() -> None:
    with pytest.raises(ValueError, match="rosa_mode"):
        AdapterConfig(local_checkpoints=True, total_steps=100, strategy="rosa")


def test_adapter_unfreeze_requires_layers() -> None:
    with pytest.raises(ValueError, match="unfreeze_layers"):
        AdapterConfig(local_checkpoints=True, total_steps=100, strategy="unfreeze")


def test_adapter_specialized_clm_requires_arch() -> None:
    """Per plan §10 S3, all four arch fields are required for the
    in-adapter specialized_clm path."""
    with pytest.raises(ValueError, match="d_model"):
        AdapterConfig(local_checkpoints=True, total_steps=100, strategy="specialized_clm")


def test_adapter_specialized_clm_requires_d_ff() -> None:
    """d_ff is part of the four-arg arch contract; missing it alone is a
    distinct failure path from missing d_model."""
    with pytest.raises(ValueError, match="d_ff"):
        AdapterConfig(
            local_checkpoints=True,
            total_steps=100,
            strategy="specialized_clm",
            d_model=64,
            n_layers=2,
            n_heads=2,
        )


def test_adapter_hybrid_requires_lora_rank() -> None:
    """`hybrid = LoRA + FiLM` per the plan adapter table; missing
    lora_rank should fail just like for `--strategy lora`."""
    with pytest.raises(ValueError, match="hybrid.*lora_rank"):
        AdapterConfig(local_checkpoints=True, total_steps=100, strategy="hybrid")


def test_adapter_specialized_clm_requires_d_model_divisible_by_n_heads() -> None:
    """Mirror of SpecializedCLMConfig's divisibility check — the
    adapter-dispatch specialized_clm path also crashes at first forward
    if head_dim is non-integer."""
    with pytest.raises(ValueError, match="divisible by n_heads"):
        AdapterConfig(
            local_checkpoints=True,
            total_steps=100,
            strategy="specialized_clm",
            d_model=33,
            n_layers=2,
            n_heads=4,
            d_ff=128,
        )


# unfreeze_layers form validator --------------------------------------------


def test_unfreeze_layers_happy_form() -> None:
    cfg = AdapterConfig(
        local_checkpoints=True, total_steps=100, strategy="unfreeze", unfreeze_layers="5,6,7"
    )
    assert cfg.unfreeze_layers == "5,6,7"


def test_unfreeze_layers_normalises_whitespace() -> None:
    """`"5, 6, 7"` is accepted but normalised to `"5,6,7"` so downstream
    `s.split(",")` doesn't trip on leading-space tokens like `" 6"`."""
    cfg = AdapterConfig(
        local_checkpoints=True,
        total_steps=100,
        strategy="unfreeze",
        unfreeze_layers="5, 6, 7",
    )
    assert cfg.unfreeze_layers == "5,6,7"


def test_unfreeze_layers_single_layer_form() -> None:
    cfg = AdapterConfig(
        local_checkpoints=True, total_steps=100, strategy="unfreeze", unfreeze_layers="0"
    )
    assert cfg.unfreeze_layers == "0"


def test_unfreeze_layers_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="unfreeze_layers"):
        AdapterConfig(
            local_checkpoints=True, total_steps=100, strategy="unfreeze", unfreeze_layers=""
        )


def test_unfreeze_layers_rejects_non_int_part() -> None:
    with pytest.raises(ValueError, match="unfreeze_layers"):
        AdapterConfig(
            local_checkpoints=True,
            total_steps=100,
            strategy="unfreeze",
            unfreeze_layers="5,abc,7",
        )


def test_unfreeze_layers_rejects_top_n_count_form() -> None:
    """Plan §6 explicit: the v1 contract is comma-separated explicit
    picks, NOT a top-N integer count."""
    cfg = AdapterConfig(
        local_checkpoints=True, total_steps=100, strategy="unfreeze", unfreeze_layers="3"
    )
    # The string "3" must remain "3" (not be re-interpreted as "top 3 layers").
    assert cfg.unfreeze_layers == "3"


# Legality flag validator ---------------------------------------------------


def test_adapter_illegal_penalty_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="illegal_penalty"):
        AdapterConfig(
            **_adapter_kwargs(illegal_penalty=-1.0)
        )


def test_adapter_illegal_penalty_requires_disable_legal_mask() -> None:
    with pytest.raises(ValueError, match="illegal_penalty.*legal masking"):
        AdapterConfig(
            **_adapter_kwargs(illegal_penalty=0.5, disable_legal_mask=False)
        )


def test_adapter_illegal_penalty_with_disable_mask_allowed() -> None:
    cfg = AdapterConfig(
        **_adapter_kwargs(illegal_penalty=0.5, disable_legal_mask=True)
    )
    assert cfg.illegal_penalty == 0.5


# Data-sizing validator -----------------------------------------------------


def test_steps_per_epoch_all_sentinel_accepted() -> None:
    cfg = AdapterConfig(**_adapter_kwargs(steps_per_epoch="all"))
    assert cfg.steps_per_epoch == "all"


def test_steps_per_epoch_int_accepted() -> None:
    cfg = AdapterConfig(**_adapter_kwargs(steps_per_epoch=1000))
    assert cfg.steps_per_epoch == 1000


def test_steps_per_epoch_zero_rejected() -> None:
    with pytest.raises(ValueError, match="steps_per_epoch"):
        AdapterConfig(**_adapter_kwargs(steps_per_epoch=0))


def test_steps_per_epoch_other_string_rejected() -> None:
    with pytest.raises(ValueError, match="steps_per_epoch.*'all'"):
        AdapterConfig(**_adapter_kwargs(steps_per_epoch="some"))


def test_steps_per_epoch_with_max_games_rejected() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        AdapterConfig(**_adapter_kwargs(steps_per_epoch=1000, max_games=50000))


# SpecializedCLMConfig validator --------------------------------------------


def test_specialized_d_model_must_divide_n_heads() -> None:
    with pytest.raises(ValueError, match="divisible by n_heads"):
        SpecializedCLMConfig(
            local_checkpoints=True,
            total_steps=100,
            d_model=33,
            n_layers=2,
            n_heads=4,
            d_ff=128,
        )


@pytest.mark.parametrize(
    "field,bad",
    [
        ("d_model", 0),
        ("n_layers", 0),
        ("n_heads", 0),
        ("d_ff", 0),
    ],
)
def test_specialized_rejects_zero_dim(field: str, bad: int) -> None:
    """Each of the four arch dims has its own rejection path."""
    kwargs = _specialized_kwargs()
    kwargs[field] = bad
    with pytest.raises(ValueError, match=field):
        SpecializedCLMConfig(**kwargs)


def test_specialized_steps_per_epoch_validators() -> None:
    """SpecializedCLMConfig has the same `steps_per_epoch` / `max_games`
    validators as AdapterConfig (both drive Lichess cache sizing)."""
    with pytest.raises(ValueError, match="steps_per_epoch"):
        SpecializedCLMConfig(**_specialized_kwargs(steps_per_epoch=0))
    with pytest.raises(ValueError, match="steps_per_epoch.*'all'"):
        SpecializedCLMConfig(**_specialized_kwargs(steps_per_epoch="bogus"))
    with pytest.raises(ValueError, match="mutually exclusive"):
        SpecializedCLMConfig(
            **_specialized_kwargs(steps_per_epoch=10, max_games=1000)
        )


def test_specialized_cadence_validators() -> None:
    """epochs / val_every / checkpoint_interval must be positive."""
    for field in ("epochs", "val_every", "checkpoint_interval"):
        with pytest.raises(ValueError, match=field):
            SpecializedCLMConfig(**_specialized_kwargs(**{field: 0}))


# ---------------------------------------------------------------------------
# RunConfig discriminated union
# ---------------------------------------------------------------------------


def test_run_config_dispatches_by_run_type() -> None:
    """The Annotated[Union, Field(discriminator="run_type")] picks the
    right subclass at parse time."""
    from pydantic import TypeAdapter

    adapter = TypeAdapter(RunConfig)
    pretrain = adapter.validate_python(
        {"run_type": "pretrain", "local_checkpoints": True, "total_steps": 100}
    )
    assert isinstance(pretrain, PretrainConfig)
    adapter_cfg = adapter.validate_python(
        {
            "run_type": "adapter",
            "local_checkpoints": True,
            "total_steps": 100,
            "strategy": "lora",
            "lora_rank": 4,
        }
    )
    assert isinstance(adapter_cfg, AdapterConfig)
    spec = adapter.validate_python(
        {
            "run_type": "specialized_clm",
            "local_checkpoints": True,
            "total_steps": 100,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 2,
            "d_ff": 128,
        }
    )
    assert isinstance(spec, SpecializedCLMConfig)


def test_run_config_rejects_unknown_run_type() -> None:
    from pydantic import TypeAdapter, ValidationError

    adapter = TypeAdapter(RunConfig)
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {"run_type": "cotrain", "local_checkpoints": True}
        )


def test_run_config_round_trips_through_union() -> None:
    """A discriminator round-trip via the union: dump → JSON → parse →
    same subclass, identical fields. This is the path a CLI driver
    using `--config <json>` plus a generic RunConfig adapter takes."""
    from pydantic import TypeAdapter

    adapter = TypeAdapter(RunConfig)
    cfg = AdapterConfig(
        **_adapter_kwargs(lora_rank=8, elo_min=1800, elo_max=2000)
    )
    dumped = cfg.model_dump()
    reloaded = adapter.validate_python(json.loads(json.dumps(dumped)))
    assert isinstance(reloaded, AdapterConfig)
    assert reloaded.model_dump() == dumped


# ---------------------------------------------------------------------------
# Cotrain is GONE BY DESIGN (plan §6) — config class shouldn't exist
# ---------------------------------------------------------------------------


def test_no_cotrain_config_class() -> None:
    """Plan §6: `pawn/cotrain.py` is GONE BY DESIGN; the supernet's joint
    loss replaces it. `CotrainConfig` should not be exported."""
    import pawn.run_config as rc
    assert not hasattr(rc, "CotrainConfig")


