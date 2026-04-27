"""Unit tests for pawn.run_config Pydantic RunConfig discriminated union.

FROZEN MODULE — do not edit pawn/run_config.py to make a test pass.
"""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from pawn.run_config import (
    AdapterConfig,
    BaseRunConfig,
    CotrainConfig,
    CotrainVariant,
    PretrainConfig,
    RunConfig,
)


# ---------------------------------------------------------------------------
# Checkpoint mode mutual exclusion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckpointModeValidation:
    def test_hf_repo_only(self):
        cfg = PretrainConfig(hf_repo="user/repo")
        assert cfg.hf_repo == "user/repo"
        assert cfg.local_checkpoints is False

    def test_local_checkpoints_only(self):
        cfg = PretrainConfig(local_checkpoints=True)
        assert cfg.local_checkpoints is True
        assert cfg.hf_repo is None

    def test_both_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            PretrainConfig(hf_repo="user/repo", local_checkpoints=True)
        assert "mutually exclusive" in str(exc_info.value)

    def test_neither_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            PretrainConfig()
        assert "required" in str(exc_info.value)

    def test_both_raises_adapter(self):
        with pytest.raises(ValidationError):
            AdapterConfig(
                strategy="lora",
                hf_repo="user/repo",
                local_checkpoints=True,
            )

    def test_neither_raises_adapter(self):
        with pytest.raises(ValidationError):
            AdapterConfig(strategy="lora")


# ---------------------------------------------------------------------------
# PretrainConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPretrainConfig:
    def test_run_type_literal(self):
        cfg = PretrainConfig(local_checkpoints=True)
        assert cfg.run_type == "pretrain"

    def test_default_variant(self):
        cfg = PretrainConfig(local_checkpoints=True)
        assert cfg.variant == "base"

    def test_variants_accepted(self):
        for v in ("toy", "small", "base", "large"):
            cfg = PretrainConfig.model_validate(
                {"local_checkpoints": True, "variant": v}
            )
            assert cfg.variant == v

    def test_invalid_variant_rejected(self):
        with pytest.raises(ValidationError):
            PretrainConfig.model_validate(
                {"local_checkpoints": True, "variant": "enormous"}
            )

    def test_architecture_overrides_default_none(self):
        cfg = PretrainConfig(local_checkpoints=True)
        assert cfg.d_model is None
        assert cfg.n_layers is None
        assert cfg.n_heads is None
        assert cfg.d_ff is None

    def test_architecture_overrides(self):
        cfg = PretrainConfig(
            local_checkpoints=True,
            d_model=384, n_layers=6, n_heads=6, d_ff=1536,
        )
        assert cfg.d_model == 384
        assert cfg.n_layers == 6
        assert cfg.n_heads == 6
        assert cfg.d_ff == 1536

    def test_default_accumulation_steps(self):
        cfg = PretrainConfig(local_checkpoints=True)
        assert cfg.accumulation_steps == 1

    def test_default_checkpoint_interval(self):
        cfg = PretrainConfig(local_checkpoints=True)
        assert cfg.checkpoint_interval == 5000


# ---------------------------------------------------------------------------
# AdapterConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterConfig:
    def test_strategy_required(self):
        with pytest.raises(ValidationError):
            AdapterConfig.model_validate({"local_checkpoints": True})

    @pytest.mark.parametrize("strat", [
        "bottleneck", "lora", "film", "sparse",
        "rosa", "hybrid", "specialized_clm", "unfreeze",
    ])
    def test_valid_strategies(self, strat):
        cfg = AdapterConfig(local_checkpoints=True, strategy=strat)
        assert cfg.strategy == strat

    def test_invalid_strategy_rejected(self):
        with pytest.raises(ValidationError):
            AdapterConfig.model_validate(
                {"local_checkpoints": True, "strategy": "nonexistent"}
            )

    def test_default_checkpoint_and_pgn(self):
        cfg = AdapterConfig(local_checkpoints=True, strategy="lora")
        assert cfg.checkpoint == "thomas-schweich/pawn-base"
        assert cfg.pgn == "thomas-schweich/pawn-lichess-full"

    def test_default_checkpoint_interval(self):
        cfg = AdapterConfig(local_checkpoints=True, strategy="lora")
        assert cfg.checkpoint_interval == 5000

    def test_lora_targets_valid(self):
        for tgt in ("qkvo", "qv", "qkv"):
            cfg = AdapterConfig.model_validate({
                "local_checkpoints": True, "strategy": "lora",
                "lora_targets": tgt,
            })
            assert cfg.lora_targets == tgt

    def test_lora_targets_invalid(self):
        with pytest.raises(ValidationError):
            AdapterConfig.model_validate({
                "local_checkpoints": True, "strategy": "lora",
                "lora_targets": "zzz",
            })

    def test_sparse_targets_valid(self):
        for tgt in ("qkvo", "qv", "qkv"):
            cfg = AdapterConfig.model_validate({
                "local_checkpoints": True, "strategy": "sparse",
                "sparse_targets": tgt,
            })
            assert cfg.sparse_targets == tgt

    def test_rosa_mode_valid(self):
        for m in ("rosa", "retro-sparse", "retro-bottleneck"):
            cfg = AdapterConfig.model_validate({
                "local_checkpoints": True, "strategy": "rosa",
                "rosa_mode": m,
            })
            assert cfg.rosa_mode == m

    def test_amp_dtype_valid(self):
        for d in ("float16", "bfloat16", "none"):
            cfg = AdapterConfig.model_validate({
                "local_checkpoints": True, "strategy": "lora",
                "amp_dtype": d,
            })
            assert cfg.amp_dtype == d

    def test_amp_dtype_invalid(self):
        with pytest.raises(ValidationError):
            AdapterConfig.model_validate({
                "local_checkpoints": True, "strategy": "lora",
                "amp_dtype": "fp32",
            })

    def test_grad_alpha_valid(self):
        for a in (1, 2):
            cfg = AdapterConfig.model_validate({
                "local_checkpoints": True, "strategy": "rosa",
                "grad_alpha": a,
            })
            assert cfg.grad_alpha == a

    def test_grad_alpha_invalid(self):
        with pytest.raises(ValidationError):
            AdapterConfig.model_validate({
                "local_checkpoints": True, "strategy": "rosa",
                "grad_alpha": 3,
            })

    def test_defaults(self):
        cfg = AdapterConfig(local_checkpoints=True, strategy="bottleneck")
        assert cfg.bottleneck_dim is None
        assert cfg.lora_rank is None
        assert cfg.no_adapt_attn is False
        assert cfg.no_adapt_ffn is False
        assert cfg.lora_ffn is False
        assert cfg.sparse_ffn is False
        assert cfg.use_output_film is False
        assert cfg.rosa_warmup_steps == 128
        assert cfg.mask_samples == 32
        assert cfg.grad_alpha == 2
        assert cfg.epochs == 50
        assert cfg.val_every == 1

    def test_steps_per_epoch_default_none(self):
        cfg = AdapterConfig(local_checkpoints=True, strategy="lora")
        assert cfg.steps_per_epoch is None
        assert cfg.data_seed is None

    def test_steps_per_epoch_int(self):
        cfg = AdapterConfig(
            local_checkpoints=True, strategy="lora", steps_per_epoch=12345,
        )
        assert cfg.steps_per_epoch == 12345

    def test_steps_per_epoch_all_sentinel(self):
        cfg = AdapterConfig(
            local_checkpoints=True, strategy="lora", steps_per_epoch="all",
        )
        assert cfg.steps_per_epoch == "all"

    def test_steps_per_epoch_invalid_string_rejected(self):
        # Pydantic's union discriminator catches non-"all" strings at the
        # field level (literal_error), before our validator runs.
        with pytest.raises(ValidationError, match="Input should be 'all'"):
            AdapterConfig.model_validate({
                "local_checkpoints": True, "strategy": "lora",
                "steps_per_epoch": "everything",
            })

    def test_steps_per_epoch_zero_rejected(self):
        with pytest.raises(ValidationError, match="must be > 0"):
            AdapterConfig.model_validate({
                "local_checkpoints": True, "strategy": "lora",
                "steps_per_epoch": 0,
            })

    def test_steps_per_epoch_negative_rejected(self):
        with pytest.raises(ValidationError, match="must be > 0"):
            AdapterConfig.model_validate({
                "local_checkpoints": True, "strategy": "lora",
                "steps_per_epoch": -1,
            })

    def test_max_games_emits_deprecation_warning(self):
        import warnings as _w
        with _w.catch_warnings(record=True) as recs:
            _w.simplefilter("always")
            AdapterConfig(
                local_checkpoints=True, strategy="lora", max_games=1_000_000,
            )
        depr = [r for r in recs if issubclass(r.category, DeprecationWarning)]
        assert any("steps_per_epoch" in str(r.message) for r in depr)

    def test_default_max_games_does_not_warn(self):
        import warnings as _w
        with _w.catch_warnings(record=True) as recs:
            _w.simplefilter("always")
            AdapterConfig(local_checkpoints=True, strategy="lora")
        depr = [r for r in recs if issubclass(r.category, DeprecationWarning)]
        assert not depr

    def test_both_steps_per_epoch_and_max_games_rejected(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            AdapterConfig.model_validate({
                "local_checkpoints": True, "strategy": "lora",
                "steps_per_epoch": 1000, "max_games": 5_000,
            })

    @pytest.mark.parametrize(
        "value, expected, expected_type",
        [
            (5, 5, int),                  # plain int
            ("5", 5, int),                # string-typed int → coerced to int
            ("all", "all", str),          # the literal sentinel
            (None, None, type(None)),
        ],
    )
    def test_steps_per_epoch_union_parsing(
        self, value, expected, expected_type,
    ):
        """Lock in pydantic's smart-union behavior for the
        ``int | Literal["all"] | None`` field. ``"5"`` must coerce to
        int (not match the literal), and ``"all"`` must stay a string."""
        cfg = AdapterConfig.model_validate({
            "local_checkpoints": True, "strategy": "lora",
            "steps_per_epoch": value,
        })
        assert cfg.steps_per_epoch == expected
        assert type(cfg.steps_per_epoch) is expected_type


# ---------------------------------------------------------------------------
# CotrainConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCotrainConfig:
    def test_valid_three_variants(self):
        cfg = CotrainConfig(
            local_checkpoints=True,
            variants=[
                CotrainVariant(name="small", variant="small"),
                CotrainVariant(name="base", variant="base"),
                CotrainVariant(name="large", variant="large"),
            ],
        )
        assert cfg.run_type == "cotrain"
        assert len(cfg.variants) == 3

    def test_single_variant(self):
        cfg = CotrainConfig(
            local_checkpoints=True,
            variants=[CotrainVariant(name="only", variant="toy")],
        )
        assert len(cfg.variants) == 1

    def test_custom_architecture_overrides(self):
        cfg = CotrainConfig(
            local_checkpoints=True,
            variants=[
                CotrainVariant(
                    name="custom", variant="base",
                    d_model=384, n_layers=6, n_heads=6, d_ff=1536,
                ),
            ],
        )
        v = cfg.variants[0]
        assert v.d_model == 384
        assert v.n_layers == 6

    def test_empty_variants_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            CotrainConfig(local_checkpoints=True, variants=[])
        assert "at least one" in str(exc_info.value).lower()

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            CotrainConfig(
                local_checkpoints=True,
                variants=[
                    CotrainVariant(name="dup", variant="small"),
                    CotrainVariant(name="dup", variant="base"),
                ],
            )
        assert "unique" in str(exc_info.value).lower()

    def test_shm_without_hf_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            CotrainConfig(
                local_checkpoints=True,
                shm_checkpoints=True,
                variants=[CotrainVariant(name="x", variant="toy")],
            )
        assert "hf-repo" in str(exc_info.value).lower() or "hf_repo" in str(exc_info.value).lower()

    def test_shm_with_hf_accepted(self):
        cfg = CotrainConfig(
            hf_repo="user/repo",
            shm_checkpoints=True,
            variants=[CotrainVariant(name="x", variant="toy")],
        )
        assert cfg.shm_checkpoints is True

    def test_top_level_resume_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            CotrainConfig(
                local_checkpoints=True,
                resume="/some/path",
                variants=[CotrainVariant(name="x", variant="toy")],
            )
        assert "per" in str(exc_info.value).lower() or "variant" in str(exc_info.value).lower()

    def test_default_val_games(self):
        cfg = CotrainConfig(
            local_checkpoints=True,
            variants=[CotrainVariant(name="x", variant="toy")],
        )
        assert cfg.val_games == 512

    def test_default_checkpoint_interval(self):
        cfg = CotrainConfig(
            local_checkpoints=True,
            variants=[CotrainVariant(name="x", variant="toy")],
        )
        assert cfg.checkpoint_interval == 5000

    def test_variant_resume_path(self):
        cfg = CotrainConfig(
            local_checkpoints=True,
            variants=[
                CotrainVariant(name="a", variant="toy", resume="/tmp/ckpt_a"),
                CotrainVariant(name="b", variant="toy", resume="/tmp/ckpt_b"),
            ],
        )
        assert cfg.variants[0].resume == "/tmp/ckpt_a"
        assert cfg.variants[1].resume == "/tmp/ckpt_b"

    def test_serialization_roundtrip(self):
        cfg = CotrainConfig(
            local_checkpoints=True,
            total_steps=1000,
            batch_size=64,
            variants=[
                CotrainVariant(name="small", variant="small"),
                CotrainVariant(name="base", variant="base", d_model=384),
            ],
        )
        data = cfg.model_dump()
        cfg2 = CotrainConfig(**data)
        assert cfg == cfg2

    def test_json_schema_generates(self):
        schema = CotrainConfig.model_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "variants" in schema["properties"]
        assert "run_type" in schema["properties"]

    def test_run_evals_defaults_off(self):
        cfg = CotrainConfig(
            local_checkpoints=True,
            variants=[CotrainVariant(name="x", variant="toy")],
        )
        assert cfg.run_evals is False
        assert cfg.lichess_pgn is None
        assert cfg.publish_results is False

    def test_run_evals_enabled(self):
        cfg = CotrainConfig(
            local_checkpoints=True,
            run_evals=True,
            variants=[CotrainVariant(name="x", variant="toy")],
        )
        assert cfg.run_evals is True

    def test_lichess_pgn_without_run_evals_rejected(self):
        """lichess_pgn only does anything inside run_post_training_evals,
        so setting it without run_evals is a user error — fail loudly."""
        with pytest.raises(ValidationError) as exc_info:
            CotrainConfig(
                local_checkpoints=True,
                lichess_pgn="/tmp/games.pgn",
                variants=[CotrainVariant(name="x", variant="toy")],
            )
        msg = str(exc_info.value).lower()
        assert "lichess_pgn" in msg and "run_evals" in msg

    def test_lichess_pgn_with_run_evals_accepted(self):
        cfg = CotrainConfig(
            local_checkpoints=True,
            run_evals=True,
            lichess_pgn="/tmp/games.pgn",
            variants=[CotrainVariant(name="x", variant="toy")],
        )
        assert cfg.lichess_pgn == "/tmp/games.pgn"

    def test_publish_results_without_hf_rejected(self):
        """publish_results uploads to an HF branch — it's meaningless
        without hf_repo."""
        with pytest.raises(ValidationError) as exc_info:
            CotrainConfig(
                local_checkpoints=True,
                run_evals=True,
                publish_results=True,
                variants=[CotrainVariant(name="x", variant="toy")],
            )
        assert "publish_results" in str(exc_info.value)

    def test_publish_results_without_run_evals_rejected(self):
        """publish_results is a no-op unless run_evals=True, because
        nothing writes eval_results.json without the post-training
        eval pass. Fail loudly on the silent-no-op config."""
        with pytest.raises(ValidationError) as exc_info:
            CotrainConfig(
                hf_repo="user/repo",
                publish_results=True,
                # run_evals intentionally omitted (default False)
                variants=[CotrainVariant(name="x", variant="toy")],
            )
        msg = str(exc_info.value)
        assert "publish_results" in msg
        assert "run_evals" in msg

    def test_publish_results_with_hf_and_run_evals_accepted(self):
        cfg = CotrainConfig(
            hf_repo="user/repo",
            run_evals=True,
            publish_results=True,
            variants=[CotrainVariant(name="x", variant="toy")],
        )
        assert cfg.publish_results is True

    def test_canonical_three_variants_config_validates(self):
        """The configs/cotrain_three_variants.json file ships in the
        repo as the canonical replacement for the deleted train_all.py.
        Lock it in against accidental schema drift.

        The JSON intentionally omits both local_checkpoints and
        hf_repo — BaseRunConfig requires exactly one, and omitting
        both lets the CLI caller pick either without colliding with a
        value baked into the JSON (scripts/train.py's _parse_cli
        merges JSON then CLI, so JSON values survive unless the CLI
        explicitly overrides them). Simulate the CLI merge here by
        supplying local_checkpoints on top of the loaded JSON."""
        import pathlib
        path = pathlib.Path(__file__).resolve().parents[2] / "configs" / "cotrain_three_variants.json"
        data = json.loads(path.read_text())
        # Drop the _comment field — it's documentation, not config.
        data.pop("_comment", None)
        # Ensure the JSON DOESN'T pin a checkpoint mode, so either side
        # of the mutual-exclusion can be chosen at launch time.
        assert "local_checkpoints" not in data, (
            "cotrain_three_variants.json must not pin local_checkpoints; "
            "it would collide with --hf-repo on the CLI"
        )
        assert "hf_repo" not in data, (
            "cotrain_three_variants.json must not pin hf_repo; "
            "it would collide with --local-checkpoints on the CLI"
        )
        # Simulate a local-checkpoints run.
        cfg_local = TypeAdapter(RunConfig).validate_python(
            {**data, "local_checkpoints": True},
        )
        assert isinstance(cfg_local, CotrainConfig)
        assert [v.name for v in cfg_local.variants] == ["small", "base", "large"]
        assert [v.variant for v in cfg_local.variants] == ["small", "base", "large"]
        # And simulate an hf-repo run — proves the same JSON can be
        # used with the CLAUDE.md-documented pod workflow.
        cfg_hf = TypeAdapter(RunConfig).validate_python(
            {**data, "hf_repo": "user/pawn-cotrain"},
        )
        assert isinstance(cfg_hf, CotrainConfig)
        assert cfg_hf.hf_repo == "user/pawn-cotrain"


@pytest.mark.unit
class TestCotrainVariant:
    def test_defaults(self):
        v = CotrainVariant(name="test")
        assert v.variant == "base"
        assert v.d_model is None
        assert v.max_seq_len == 512
        assert v.resume is None

    def test_all_variant_presets(self):
        for preset in ("toy", "small", "base", "large"):
            v = CotrainVariant.model_validate({"name": preset, "variant": preset})
            assert v.variant == preset

    def test_invalid_variant_rejected(self):
        with pytest.raises(ValidationError):
            CotrainVariant.model_validate({"name": "x", "variant": "enormous"})


# ---------------------------------------------------------------------------
# Base fields shared across configs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBaseRunConfig:
    def test_base_defaults_on_pretrain(self):
        cfg = PretrainConfig(local_checkpoints=True)
        assert cfg.elo_min is None
        assert cfg.elo_max is None
        assert cfg.max_games is None
        assert cfg.val_games == 512  # PretrainConfig overrides BaseRunConfig's 50K
        assert cfg.min_ply == 10
        assert cfg.batch_size == 256
        assert cfg.lr == 3e-4
        assert cfg.weight_decay == 0.0
        assert cfg.warmup_frac == 0.05
        assert cfg.warmup_steps is None
        assert cfg.max_grad_norm == 1.0
        assert cfg.patience is None
        assert cfg.log_interval == 100
        assert cfg.mate_boost == 0.0
        assert cfg.prepend_outcome is False
        assert cfg.discard_ply_limit is False
        assert cfg.no_compile is False
        assert cfg.sdpa_math is False
        assert cfg.num_workers == 4
        assert cfg.device == "cuda"
        assert cfg.wandb is False
        assert cfg.resume is None

    def test_base_run_config_not_accepted_by_discriminated_union(self):
        """BaseRunConfig is directly instantiable but NOT accepted by RunConfig union."""
        # BaseRunConfig can be instantiated on its own
        cfg = BaseRunConfig(local_checkpoints=True)
        assert cfg.local_checkpoints is True

        # But a BaseRunConfig dict (without run_type) must NOT validate as RunConfig
        adapter = TypeAdapter(RunConfig)
        with pytest.raises(ValidationError):
            adapter.validate_python(cfg.model_dump())

        # Even if we manually add a bogus run_type, it should be rejected
        data = cfg.model_dump()
        data["run_type"] = "base"
        with pytest.raises(ValidationError):
            adapter.validate_python(data)


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDiscriminatedUnion:
    def test_union_dispatches_pretrain(self):
        adapter = TypeAdapter(RunConfig)
        cfg = adapter.validate_python({
            "run_type": "pretrain",
            "local_checkpoints": True,
        })
        assert isinstance(cfg, PretrainConfig)
        assert cfg.run_type == "pretrain"

    def test_union_dispatches_adapter(self):
        adapter = TypeAdapter(RunConfig)
        cfg = adapter.validate_python({
            "run_type": "adapter",
            "local_checkpoints": True,
            "strategy": "lora",
        })
        assert isinstance(cfg, AdapterConfig)
        assert cfg.run_type == "adapter"
        assert cfg.strategy == "lora"

    def test_union_dispatches_cotrain(self):
        adapter = TypeAdapter(RunConfig)
        cfg = adapter.validate_python({
            "run_type": "cotrain",
            "local_checkpoints": True,
            "variants": [{"name": "s", "variant": "small"}],
        })
        assert isinstance(cfg, CotrainConfig)
        assert cfg.run_type == "cotrain"
        assert len(cfg.variants) == 1

    def test_union_missing_run_type(self):
        adapter = TypeAdapter(RunConfig)
        with pytest.raises(ValidationError):
            adapter.validate_python({"local_checkpoints": True})

    def test_union_invalid_run_type(self):
        adapter = TypeAdapter(RunConfig)
        with pytest.raises(ValidationError):
            adapter.validate_python({
                "run_type": "unknown",
                "local_checkpoints": True,
            })


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSerialization:
    def test_pretrain_json_roundtrip(self):
        cfg = PretrainConfig(
            local_checkpoints=True,
            variant="small",
            total_steps=1000,
            batch_size=64,
        )
        data = cfg.model_dump()
        cfg2 = PretrainConfig(**data)
        assert cfg == cfg2

    def test_adapter_json_roundtrip(self):
        cfg = AdapterConfig(
            local_checkpoints=True,
            strategy="lora",
            lora_rank=4,
            lora_targets="qkvo",
        )
        data = cfg.model_dump()
        cfg2 = AdapterConfig(**data)
        assert cfg == cfg2

    def test_bottleneck_n_hidden_default_zero(self):
        cfg = AdapterConfig(
            local_checkpoints=True, strategy="bottleneck", bottleneck_dim=8,
        )
        assert cfg.bottleneck_n_hidden == 0

    def test_bottleneck_n_hidden_roundtrip(self):
        cfg = AdapterConfig(
            local_checkpoints=True,
            strategy="bottleneck",
            bottleneck_dim=8,
            bottleneck_n_hidden=2,
        )
        data = cfg.model_dump()
        cfg2 = AdapterConfig(**data)
        assert cfg == cfg2
        assert cfg2.bottleneck_n_hidden == 2

    def test_pretrain_json_string_roundtrip(self):
        cfg = PretrainConfig(local_checkpoints=True, variant="large")
        s = cfg.model_dump_json()
        d = json.loads(s)
        cfg2 = PretrainConfig(**d)
        assert cfg == cfg2

    def test_cotrain_json_roundtrip(self):
        cfg = CotrainConfig(
            local_checkpoints=True,
            total_steps=500,
            variants=[
                CotrainVariant(name="a", variant="toy"),
                CotrainVariant(name="b", variant="small", d_model=128),
            ],
        )
        data = cfg.model_dump()
        cfg2 = CotrainConfig(**data)
        assert cfg == cfg2

    def test_cotrain_json_string_roundtrip(self):
        cfg = CotrainConfig(
            local_checkpoints=True,
            variants=[CotrainVariant(name="x", variant="toy")],
        )
        s = cfg.model_dump_json()
        d = json.loads(s)
        cfg2 = CotrainConfig(**d)
        assert cfg == cfg2

    def test_pretrain_json_schema_generates(self):
        schema = PretrainConfig.model_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "variant" in schema["properties"]
        assert "run_type" in schema["properties"]

    def test_adapter_json_schema_generates(self):
        schema = AdapterConfig.model_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "strategy" in schema["properties"]
        assert "run_type" in schema["properties"]

    def test_run_type_in_schema_has_literal_value(self):
        schema = PretrainConfig.model_json_schema()
        rt = schema["properties"]["run_type"]
        # Literal renders as either const or enum with single value
        assert (
            rt.get("const") == "pretrain"
            or rt.get("enum") == ["pretrain"]
            or rt.get("default") == "pretrain"
        )


# ---------------------------------------------------------------------------
# Bad types
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTypeValidation:
    def test_batch_size_must_be_int(self):
        with pytest.raises(ValidationError):
            PretrainConfig.model_validate(
                {"local_checkpoints": True, "batch_size": "many"}
            )

    def test_lr_must_be_float(self):
        with pytest.raises(ValidationError):
            PretrainConfig.model_validate(
                {"local_checkpoints": True, "lr": "fast"}
            )

    def test_prepend_outcome_flag(self):
        cfg = PretrainConfig(local_checkpoints=True, prepend_outcome=True)
        assert cfg.prepend_outcome is True
        # Default is off so existing runs keep pure-move sequences.
        default_cfg = PretrainConfig(local_checkpoints=True)
        assert default_cfg.prepend_outcome is False

    def test_prepend_outcome_propagates_to_cotrain(self):
        cfg = CotrainConfig(
            local_checkpoints=True,
            prepend_outcome=True,
            variants=[CotrainVariant(name="base", variant="base")],
        )
        assert cfg.prepend_outcome is True


@pytest.mark.unit
class TestCustomVariant:
    """variant='custom' forces explicit architecture — no fallback to a preset."""

    def test_pretrain_custom_requires_all_arch_fields(self):
        with pytest.raises(ValidationError) as exc_info:
            PretrainConfig(local_checkpoints=True, variant="custom")
        msg = str(exc_info.value)
        assert "d_model" in msg
        assert "n_layers" in msg
        assert "n_heads" in msg
        assert "d_ff" in msg

    def test_pretrain_custom_with_partial_fields_errors(self):
        with pytest.raises(ValidationError) as exc_info:
            PretrainConfig(
                local_checkpoints=True, variant="custom",
                d_model=128, n_layers=4,  # missing n_heads, d_ff
            )
        msg = str(exc_info.value)
        assert "n_heads" in msg
        assert "d_ff" in msg
        assert "d_model" not in msg  # already provided

    def test_pretrain_custom_accepts_complete_arch(self):
        cfg = PretrainConfig(
            local_checkpoints=True, variant="custom",
            d_model=128, n_layers=4, n_heads=4, d_ff=512,
        )
        assert cfg.variant == "custom"
        assert cfg.d_model == 128

    def test_cotrain_variant_custom_requires_all_arch_fields(self):
        with pytest.raises(ValidationError) as exc_info:
            CotrainVariant(name="tiny", variant="custom")
        msg = str(exc_info.value)
        for field in ("d_model", "n_layers", "n_heads", "d_ff"):
            assert field in msg
        assert "tiny" in msg  # mentions the offending variant name

    def test_cotrain_variant_custom_accepts_complete_arch(self):
        v = CotrainVariant(
            name="tiny", variant="custom",
            d_model=96, n_layers=3, n_heads=4, d_ff=384,
        )
        assert v.variant == "custom"

    def test_preset_variants_do_not_require_arch_fields(self):
        """Sanity: custom's validator must not apply to the named presets."""
        for preset in ("toy", "small", "base", "large"):
            PretrainConfig(local_checkpoints=True, variant=preset)
            CotrainVariant(name=preset, variant=preset)

    def test_pretrain_custom_rejects_incompatible_head_dim(self):
        """d_model must be divisible by n_heads; otherwise attention
        crashes at first forward. Validator catches it upfront."""
        with pytest.raises(ValidationError) as exc_info:
            PretrainConfig(
                local_checkpoints=True, variant="custom",
                d_model=96, n_layers=4, n_heads=5, d_ff=384,  # 96 % 5 = 1
            )
        msg = str(exc_info.value)
        assert "divisible" in msg
        assert "n_heads" in msg
        assert "d_model" in msg

    def test_pretrain_custom_accepts_compatible_head_dim(self):
        cfg = PretrainConfig(
            local_checkpoints=True, variant="custom",
            d_model=96, n_layers=4, n_heads=4, d_ff=384,  # 96 % 4 = 0
        )
        assert cfg.d_model == 96
        assert cfg.n_heads == 4

    def test_pretrain_custom_rejects_nonpositive(self):
        with pytest.raises(ValidationError) as exc_info:
            PretrainConfig(
                local_checkpoints=True, variant="custom",
                d_model=0, n_layers=4, n_heads=4, d_ff=512,
            )
        assert "positive" in str(exc_info.value)

    def test_cotrain_variant_custom_rejects_incompatible_head_dim(self):
        with pytest.raises(ValidationError) as exc_info:
            CotrainVariant(
                name="weird", variant="custom",
                d_model=96, n_layers=4, n_heads=5, d_ff=384,
            )
        msg = str(exc_info.value)
        assert "divisible" in msg
        assert "weird" in msg  # names the offending variant

    def test_cotrain_variant_custom_accepts_compatible_head_dim(self):
        v = CotrainVariant(
            name="ok", variant="custom",
            d_model=128, n_layers=4, n_heads=4, d_ff=512,  # 128 % 4 = 0
        )
        assert v.d_model == 128

    def test_extra_fields_forbidden(self):
        """Unknown fields are rejected so stale CLI flags (e.g. legacy
        `--legacy-vocab`) don't silently run with wrong architecture."""
        with pytest.raises(ValidationError, match="bogus_field"):
            PretrainConfig.model_validate(
                {"local_checkpoints": True, "bogus_field": "value"}
            )

    def test_stale_legacy_vocab_rejected(self):
        """Regression guard: legacy_vocab was removed — passing it must fail."""
        with pytest.raises(ValidationError, match="legacy_vocab"):
            PretrainConfig.model_validate(
                {"local_checkpoints": True, "legacy_vocab": True}
            )
