"""Tests for the restored pydantic run_config (S4.C1 of the JAX migration).

Pins the v1-parity contract from ``docs/jax-migration.md`` §8.1 +
§8.3 + §8.4: every CLI/JSON field every training driver consumes
goes through one of ``PretrainConfig`` / ``AdapterConfig`` /
``SpecializedCLMConfig``; unknown fields fail loud
(``extra="forbid"``); v1 field NAMES are canonical
(``lora_rank``, ``density``, ``use_output_film``, ``no_adapt_attn``,
``no_adapt_ffn``); v2 substantive changes are kept
(``lora_targets: list[str]``, ``rosa_warmup_frac``, ``n_unfreeze``).
"""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from pawn.run_config import (
    AdapterConfig,
    BaseRunConfig,
    PretrainConfig,
    RunConfig,
    SpecializedCLMConfig,
)


# ---------------------------------------------------------------------------
# BaseRunConfig invariants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extra_forbid_rejects_unknown_field() -> None:
    """Unknown fields fail loud — the v1 lesson the first JAX-migration
    attempt forgot (§8.1)."""
    with pytest.raises(ValidationError, match="Extra inputs"):
        PretrainConfig(
            supernet="tiny",
            total_steps=100,
            k=50,
            local_checkpoints=True,
            legacy_vocab=True,  # type: ignore[call-arg]  # purposely unknown
        )


@pytest.mark.unit
def test_checkpoint_mode_xor() -> None:
    """One of hf_repo / hf_bucket / local_checkpoints is required."""
    with pytest.raises(ValidationError, match="hf-repo.*local-checkpoints"):
        PretrainConfig(supernet="tiny", total_steps=100, k=50)
    # hf_repo + local_checkpoints together → ValueError
    with pytest.raises(ValidationError, match="mutually exclusive"):
        PretrainConfig(
            supernet="tiny",
            total_steps=100,
            k=50,
            hf_repo="foo/bar",
            local_checkpoints=True,
        )


@pytest.mark.unit
def test_k_must_divide_total_steps() -> None:
    """The lax.scan chunk loop requires no padded final chunk."""
    with pytest.raises(ValidationError, match="multiple of k"):
        PretrainConfig(
            supernet="tiny",
            total_steps=999,
            k=50,
            local_checkpoints=True,
        )


@pytest.mark.unit
def test_k_must_be_positive() -> None:
    """k=0 or negative isn't a valid scan chunk size."""
    for bad_k in (0, -1):
        with pytest.raises(ValidationError, match="must be > 0"):
            PretrainConfig(
                supernet="tiny",
                total_steps=100,
                k=bad_k,
                local_checkpoints=True,
            )


@pytest.mark.unit
def test_lr_schedule_fraction_ranges() -> None:
    """warmup_frac / decay_frac / cooldown_frac / stable_lr_ratio
    must be in [0, 1]."""
    for field in (
        "warmup_frac",
        "decay_frac",
        "cooldown_frac",
        "stable_lr_ratio",
    ):
        with pytest.raises(ValidationError, match=r"\[0, 1\]"):
            PretrainConfig(
                supernet="tiny",
                total_steps=100,
                k=50,
                local_checkpoints=True,
                **{field: 1.5},  # type: ignore[arg-type]
            )
        with pytest.raises(ValidationError, match=r"\[0, 1\]"):
            PretrainConfig(
                supernet="tiny",
                total_steps=100,
                k=50,
                local_checkpoints=True,
                **{field: -0.1},  # type: ignore[arg-type]
            )


@pytest.mark.unit
def test_wsd_schedule_warmup_plus_decay_in_unit_interval() -> None:
    """WSD needs warmup + decay <= 1 so the stable plateau is
    non-negative."""
    with pytest.raises(ValidationError, match="WSD"):
        PretrainConfig(
            supernet="tiny",
            total_steps=100,
            k=50,
            local_checkpoints=True,
            lr_schedule="wsd",
            warmup_frac=0.6,
            decay_frac=0.6,
        )
    # Boundary: 0.5 + 0.5 == 1.0 → ok
    PretrainConfig(
        supernet="tiny",
        total_steps=100,
        k=50,
        local_checkpoints=True,
        lr_schedule="wsd",
        warmup_frac=0.5,
        decay_frac=0.5,
    )


@pytest.mark.unit
def test_infinite_schedule_warmup_plus_cooldown_plus_decay_in_unit_interval() -> None:
    with pytest.raises(ValidationError, match="infinite"):
        PretrainConfig(
            supernet="tiny",
            total_steps=100,
            k=50,
            local_checkpoints=True,
            lr_schedule="infinite",
            warmup_frac=0.4,
            cooldown_frac=0.4,
            decay_frac=0.4,
        )


# ---------------------------------------------------------------------------
# SpecializedCLMConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_specialized_clm_field_names_have_no_prefix() -> None:
    """§8.3: no ``specialized_`` prefix on d_model / n_layers /
    n_heads / d_ff — they live inside the per-strategy config."""
    sc = SpecializedCLMConfig(d_model=64, n_layers=2, n_heads=2, d_ff=128)
    assert sc.d_model == 64
    assert sc.n_layers == 2
    # Discarded v2 names must not appear
    assert not hasattr(sc, "specialized_d_model")
    assert not hasattr(sc, "specialized_n_heads")


@pytest.mark.unit
def test_specialized_clm_divisibility() -> None:
    with pytest.raises(ValidationError, match="divisible by"):
        SpecializedCLMConfig(d_model=65, n_layers=2, n_heads=2, d_ff=128)
    with pytest.raises(ValidationError, match="positive"):
        SpecializedCLMConfig(d_model=0, n_layers=2, n_heads=2, d_ff=128)
    # n_heads=0 must NOT raise ZeroDivisionError — the validator
    # checks positivity before divisibility.
    with pytest.raises(ValidationError, match="positive"):
        SpecializedCLMConfig(d_model=64, n_layers=2, n_heads=0, d_ff=128)
    # n_layers / d_ff positivity
    with pytest.raises(ValidationError, match="positive"):
        SpecializedCLMConfig(d_model=64, n_layers=0, n_heads=2, d_ff=128)
    with pytest.raises(ValidationError, match="positive"):
        SpecializedCLMConfig(d_model=64, n_layers=2, n_heads=2, d_ff=0)


# ---------------------------------------------------------------------------
# PretrainConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pretrain_happy_path() -> None:
    cfg = PretrainConfig(
        supernet="tiny",
        total_steps=1000,
        k=50,
        batch_size=16,
        local_checkpoints=True,
    )
    assert cfg.run_type == "pretrain"
    assert cfg.supernet == "tiny"
    assert cfg.total_steps == 1000


@pytest.mark.unit
def test_pretrain_variant_loss_weights_validation() -> None:
    # Valid keys
    PretrainConfig(
        supernet="tiny",
        total_steps=100,
        k=50,
        local_checkpoints=True,
        variant_loss_weights={"small": 1.0, "base": 1.0, "large": 2.0},
    )
    # Invalid variant name
    with pytest.raises(ValidationError, match="must be one of"):
        PretrainConfig(
            supernet="tiny",
            total_steps=100,
            k=50,
            local_checkpoints=True,
            variant_loss_weights={"giant": 1.0},
        )
    # Negative weight
    with pytest.raises(ValidationError, match=">= 0"):
        PretrainConfig(
            supernet="tiny",
            total_steps=100,
            k=50,
            local_checkpoints=True,
            variant_loss_weights={"base": -1.0},
        )


# ---------------------------------------------------------------------------
# AdapterConfig — per-strategy guards
# ---------------------------------------------------------------------------


def _adapter_base_kwargs() -> dict:
    return {
        "total_steps": 500,
        "k": 50,
        "supernet": "tiny",
        "variant": "base",
        "local_checkpoints": True,
    }


@pytest.mark.unit
def test_adapter_lora_requires_rank() -> None:
    with pytest.raises(ValidationError, match="--lora-rank"):
        AdapterConfig(strategy="lora", **_adapter_base_kwargs())
    AdapterConfig(strategy="lora", lora_rank=4, **_adapter_base_kwargs())


@pytest.mark.unit
def test_adapter_bottleneck_requires_dim() -> None:
    with pytest.raises(ValidationError, match="--bottleneck-dim"):
        AdapterConfig(strategy="bottleneck", **_adapter_base_kwargs())
    AdapterConfig(
        strategy="bottleneck", bottleneck_dim=64, **_adapter_base_kwargs()
    )


@pytest.mark.unit
def test_adapter_hybrid_requires_rank_and_dim() -> None:
    with pytest.raises(ValidationError, match="--lora-rank"):
        AdapterConfig(
            strategy="hybrid", bottleneck_dim=64, **_adapter_base_kwargs()
        )
    with pytest.raises(ValidationError, match="--bottleneck-dim"):
        AdapterConfig(
            strategy="hybrid", lora_rank=4, **_adapter_base_kwargs()
        )
    AdapterConfig(
        strategy="hybrid",
        lora_rank=4,
        bottleneck_dim=64,
        **_adapter_base_kwargs(),
    )


@pytest.mark.unit
def test_adapter_sparse_requires_density_in_range() -> None:
    with pytest.raises(ValidationError, match="--density"):
        AdapterConfig(strategy="sparse", **_adapter_base_kwargs())
    with pytest.raises(ValidationError, match=r"\(0, 1\]"):
        AdapterConfig(strategy="sparse", density=1.5, **_adapter_base_kwargs())
    AdapterConfig(strategy="sparse", density=0.1, **_adapter_base_kwargs())


@pytest.mark.unit
def test_adapter_rosa_requires_rank_and_frac_ranges() -> None:
    with pytest.raises(ValidationError, match="rosa_top_k_frac.*0, 1"):
        AdapterConfig(
            strategy="rosa",
            lora_rank=4,
            rosa_top_k_frac=2.0,
            **_adapter_base_kwargs(),
        )
    with pytest.raises(ValidationError, match="rosa_warmup_frac.*0, 1"):
        AdapterConfig(
            strategy="rosa",
            lora_rank=4,
            rosa_warmup_frac=1.5,
            **_adapter_base_kwargs(),
        )
    AdapterConfig(strategy="rosa", lora_rank=4, **_adapter_base_kwargs())


@pytest.mark.unit
def test_adapter_unfreeze_requires_n_unfreeze() -> None:
    with pytest.raises(ValidationError, match="--n-unfreeze"):
        AdapterConfig(strategy="unfreeze", **_adapter_base_kwargs())
    AdapterConfig(strategy="unfreeze", n_unfreeze=2, **_adapter_base_kwargs())


@pytest.mark.unit
def test_adapter_specialized_clm_requires_nested_and_no_backbone() -> None:
    """specialized_clm trains from scratch; checkpoint / supernet /
    variant must NOT be set."""
    # Missing nested config
    with pytest.raises(ValidationError, match="nested.*specialized_clm"):
        AdapterConfig(
            strategy="specialized_clm",
            total_steps=500,
            k=50,
            local_checkpoints=True,
        )
    # Backbone set
    with pytest.raises(ValidationError, match="trains from scratch"):
        AdapterConfig(
            strategy="specialized_clm",
            specialized_clm=SpecializedCLMConfig(
                d_model=64, n_layers=2, n_heads=2, d_ff=128
            ),
            total_steps=500,
            k=50,
            local_checkpoints=True,
            supernet="tiny",
            variant="base",
        )
    # Happy
    AdapterConfig(
        strategy="specialized_clm",
        specialized_clm=SpecializedCLMConfig(
            d_model=64, n_layers=2, n_heads=2, d_ff=128
        ),
        total_steps=500,
        k=50,
        local_checkpoints=True,
    )


@pytest.mark.unit
def test_adapter_bottleneck_no_op_guard() -> None:
    """--no-adapt-attn AND --no-adapt-ffn makes bottleneck/hybrid
    a no-op."""
    with pytest.raises(ValidationError, match="no-op"):
        AdapterConfig(
            strategy="bottleneck",
            bottleneck_dim=64,
            no_adapt_attn=True,
            no_adapt_ffn=True,
            **_adapter_base_kwargs(),
        )


@pytest.mark.unit
def test_adapter_lora_targets_subset_of_qkvo() -> None:
    # Valid
    AdapterConfig(
        strategy="lora",
        lora_rank=4,
        lora_targets=["q", "v"],
        **_adapter_base_kwargs(),
    )
    # Invalid entry
    with pytest.raises(ValidationError, match=r"\{q,k,v,o\}"):
        AdapterConfig(
            strategy="lora",
            lora_rank=4,
            lora_targets=["q", "z"],
            **_adapter_base_kwargs(),
        )
    # Empty
    with pytest.raises(ValidationError, match="at least one"):
        AdapterConfig(
            strategy="lora",
            lora_rank=4,
            lora_targets=[],
            **_adapter_base_kwargs(),
        )
    # Duplicates
    with pytest.raises(ValidationError, match="duplicates"):
        AdapterConfig(
            strategy="lora",
            lora_rank=4,
            lora_targets=["q", "q"],
            **_adapter_base_kwargs(),
        )


@pytest.mark.unit
def test_adapter_backbone_xor_checkpoint() -> None:
    """Pass EITHER --checkpoint OR --supernet + --variant; not both."""
    # Missing both
    with pytest.raises(ValidationError, match="needs a backbone"):
        AdapterConfig(
            strategy="lora",
            lora_rank=4,
            total_steps=500,
            k=50,
            local_checkpoints=True,
        )
    # Both
    with pytest.raises(ValidationError, match="not both"):
        AdapterConfig(
            strategy="lora",
            lora_rank=4,
            checkpoint="some/path",
            supernet="tiny",
            variant="base",
            total_steps=500,
            k=50,
            local_checkpoints=True,
        )


@pytest.mark.unit
def test_adapter_illegal_penalty_requires_disabled_mask() -> None:
    with pytest.raises(ValidationError, match="legal.*masking"):
        AdapterConfig(
            strategy="lora",
            lora_rank=4,
            illegal_penalty=0.5,
            **_adapter_base_kwargs(),
        )
    AdapterConfig(
        strategy="lora",
        lora_rank=4,
        illegal_penalty=0.5,
        disable_legal_mask=True,
        **_adapter_base_kwargs(),
    )


@pytest.mark.unit
def test_adapter_val_every_must_be_geq_k() -> None:
    with pytest.raises(ValidationError, match="val_every"):
        AdapterConfig(
            strategy="lora",
            lora_rank=4,
            k=100,
            val_every=50,
            total_steps=500,
            supernet="tiny",
            variant="base",
            local_checkpoints=True,
        )


# ---------------------------------------------------------------------------
# v1 field-name canonicality (§8.3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_v1_canonical_field_names() -> None:
    """Discarded v2 cosmetic renames must NOT be the accepted name."""
    cfg = AdapterConfig(
        strategy="lora",
        lora_rank=4,
        density=None,
        use_output_film=True,
        no_adapt_attn=False,
        no_adapt_ffn=False,
        **_adapter_base_kwargs(),
    )
    # v1 names accepted
    assert cfg.lora_rank == 4
    assert cfg.use_output_film is True

    # v2 cosmetic-rename names rejected
    for bad_field in (
        "rank",
        "sparse_density",
        "film_output",
        "bottleneck_no_attn",
        "bottleneck_no_ffn",
    ):
        with pytest.raises(ValidationError, match="Extra inputs"):
            AdapterConfig(
                strategy="lora",
                lora_rank=4,
                **_adapter_base_kwargs(),
                **{bad_field: 1},  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# §8.4 substantive changes — kept
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lora_targets_is_list_not_literal() -> None:
    """v2 widened ``lora_targets`` from Literal to list[str] per §8.4."""
    cfg = AdapterConfig(
        strategy="lora",
        lora_rank=4,
        lora_targets=["q", "k", "v", "o"],
        **_adapter_base_kwargs(),
    )
    assert cfg.lora_targets == ["q", "k", "v", "o"]


@pytest.mark.unit
def test_v1_removed_fields_rejected() -> None:
    """Fields removed in §8.4 must NOT be accepted."""
    for removed in (
        "rosa_mode",
        "rosa_warmup_steps",
        "mask_samples",
        "grad_alpha",
        "bucket_size",
        "lora_ffn",
        "sparse_ffn",
        "unfreeze_layers",
        "amp_dtype",
        "no_compile",
        "sdpa_math",
        "device",
        "num_workers",
    ):
        with pytest.raises(ValidationError, match="Extra inputs"):
            AdapterConfig(
                strategy="lora",
                lora_rank=4,
                **_adapter_base_kwargs(),
                **{removed: 1},  # type: ignore[arg-type]
            )


@pytest.mark.unit
def test_cotrain_config_does_not_exist() -> None:
    """CotrainConfig / CotrainVariant are GONE BY DESIGN (§2)."""
    import pawn.run_config as mod

    assert not hasattr(mod, "CotrainConfig")
    assert not hasattr(mod, "CotrainVariant")


# ---------------------------------------------------------------------------
# RunConfig discriminated union + JSON round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_config_discriminates_on_run_type() -> None:
    adapter_for_unit = TypeAdapter(RunConfig)
    pretrain = adapter_for_unit.validate_python(
        {
            "run_type": "pretrain",
            "supernet": "tiny",
            "total_steps": 1000,
            "k": 50,
            "local_checkpoints": True,
        }
    )
    assert isinstance(pretrain, PretrainConfig)

    adapter_cfg = adapter_for_unit.validate_python(
        {
            "run_type": "adapter",
            "strategy": "lora",
            "lora_rank": 4,
            **_adapter_base_kwargs(),
        }
    )
    assert isinstance(adapter_cfg, AdapterConfig)


@pytest.mark.unit
def test_json_schema_round_trip() -> None:
    """Lab uses model_json_schema() — it must produce valid JSON Schema."""
    for cfg_cls in (PretrainConfig, AdapterConfig):
        schema = cfg_cls.model_json_schema()
        assert "properties" in schema
        # The schema must be JSON-serialisable (lab passes it over stdio)
        json.dumps(schema)


@pytest.mark.unit
def test_config_round_trip_through_model_dump() -> None:
    """trainer CLI feeds JSON → model → model_dump → argparse path."""
    original = AdapterConfig(
        strategy="lora",
        lora_rank=4,
        lora_targets=["q", "v"],
        **_adapter_base_kwargs(),
    )
    dumped = original.model_dump()
    restored = AdapterConfig(**dumped)
    assert restored == original


@pytest.mark.unit
def test_base_run_config_is_abstract_in_practice() -> None:
    """BaseRunConfig is exported but bare instantiation would have
    no run_type discriminator. Sanity check: it still validates the
    shared invariants."""
    with pytest.raises(ValidationError, match="hf-repo"):
        BaseRunConfig(total_steps=100, k=50)
