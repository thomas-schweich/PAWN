"""Tests for :mod:`pawn.config`.

The config module is pure metadata — frozen dataclasses + module-level
constants — so tests stay light. They cover:

- The vocab + sequence constants match the engine's contract.
- ``ModelConfig`` rejects self-inconsistent dimensions at construction.
- The production / tiny supernet+variant pairs satisfy ``validate_nested``.
- ``validate_nested`` raises ``NestingError`` on every kind of mismatch.
- The module imports without dragging in JAX or torch.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from pawn.config import (
    HEAD_DIM,
    MAX_SEQ_LEN,
    N_PRETRAINING_OUTCOMES,
    N_TOTAL_OUTCOMES,
    NUM_ACTIONS,
    OUTCOME_TOKEN_BASE,
    PAD_TOKEN,
    ROPE_BASE,
    SUPERNET,
    TINY_SUPERNET,
    TINY_VARIANTS,
    VARIANTS,
    VOCAB_SIZE,
    ModelConfig,
    NestingError,
    validate_nested,
)


# ---------------------------------------------------------------------------
# Vocab + sequence constants (these are the durable user-visible contract)
# ---------------------------------------------------------------------------


def test_vocab_constants_match_engine_contract() -> None:
    """The token-layout numbers are the contract the Rust engine asserts.

    1968 actions + 1 PAD + 11 outcomes = 1980 total. PAD == 1968, outcome
    range starts at 1969. Changing any of these requires a coordinated
    engine update and a new pre-vocab-transition tag.
    """
    assert NUM_ACTIONS == 1968
    assert PAD_TOKEN == 1968
    assert OUTCOME_TOKEN_BASE == 1969
    assert N_PRETRAINING_OUTCOMES == 5
    assert N_TOTAL_OUTCOMES == 11
    # A.2 (narrow): VOCAB_SIZE is the lm_head output width — action
    # tokens + PAD. Outcome columns (1969..1979) were trimmed since
    # outcomes are inputs only and never appear in targets. Saves
    # ~0.6% of lm_head FLOPs at zero correctness risk; the broader
    # PAD-column trim is deferred (needs paired generation refactor).
    from pawn.config import N_INPUT_TOKENS
    assert VOCAB_SIZE == NUM_ACTIONS + 1 == 1969
    assert N_INPUT_TOKENS == NUM_ACTIONS + 1 + N_TOTAL_OUTCOMES == 1980


def test_sequence_constants() -> None:
    assert MAX_SEQ_LEN == 512
    assert ROPE_BASE == 10000.0
    assert HEAD_DIM == 64


# ---------------------------------------------------------------------------
# ModelConfig: self-consistency
# ---------------------------------------------------------------------------


def test_model_config_valid_construction() -> None:
    """A coherent ModelConfig constructs cleanly."""
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=1, d_ff=256)
    assert cfg.d_model == 64
    assert cfg.head_dim == HEAD_DIM
    assert cfg.vocab_size == VOCAB_SIZE
    assert cfg.max_seq_len == MAX_SEQ_LEN


def test_model_config_is_frozen() -> None:
    """The dataclass is frozen — mutating a field raises `AttributeError`
    (the specific `FrozenInstanceError` is a subclass of it). Narrowed to
    AttributeError so a future `__setattr__` override raising the wrong
    error type would fail loudly instead of being silently accepted."""
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=1, d_ff=256)
    with pytest.raises(AttributeError):
        cfg.d_model = 128  # type: ignore[misc]


def test_model_config_accepts_d_ff_equal_to_d_model() -> None:
    """The guard is `d_ff < d_model`, so equality is allowed. The SwiGLU
    layer's intermediate dimension is allowed to match d_model exactly."""
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=1, d_ff=64)
    assert cfg.d_ff == 64


def test_model_config_rejects_d_model_not_multiple_of_head_dim() -> None:
    """If d_model isn't head_dim × n_heads, the model is malformed."""
    with pytest.raises(ValueError, match="d_model"):
        ModelConfig(d_model=100, n_layers=2, n_heads=1, d_ff=400)  # 100 ≠ 1×64


def test_model_config_rejects_zero_d_model() -> None:
    with pytest.raises(ValueError, match="d_model"):
        ModelConfig(d_model=0, n_layers=2, n_heads=1, d_ff=256)


def test_model_config_rejects_zero_n_layers() -> None:
    with pytest.raises(ValueError, match="n_layers"):
        ModelConfig(d_model=64, n_layers=0, n_heads=1, d_ff=256)


def test_model_config_rejects_zero_n_heads() -> None:
    with pytest.raises(ValueError, match="n_heads"):
        ModelConfig(d_model=64, n_layers=2, n_heads=0, d_ff=256)


def test_model_config_rejects_negative_n_heads() -> None:
    with pytest.raises(ValueError, match="n_heads"):
        ModelConfig(d_model=64, n_layers=2, n_heads=-1, d_ff=256)


def test_model_config_rejects_zero_d_ff() -> None:
    with pytest.raises(ValueError, match="d_ff"):
        ModelConfig(d_model=64, n_layers=2, n_heads=1, d_ff=0)


def test_model_config_rejects_zero_head_dim() -> None:
    with pytest.raises(ValueError, match="head_dim"):
        ModelConfig(d_model=64, n_layers=2, n_heads=1, d_ff=256, head_dim=0)


def test_model_config_rejects_zero_vocab_size() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        ModelConfig(d_model=64, n_layers=2, n_heads=1, d_ff=256, vocab_size=0)


def test_model_config_rejects_zero_max_seq_len() -> None:
    with pytest.raises(ValueError, match="max_seq_len"):
        ModelConfig(d_model=64, n_layers=2, n_heads=1, d_ff=256, max_seq_len=0)


def test_model_config_rejects_negative_n_outcomes() -> None:
    with pytest.raises(ValueError, match="n_outcomes"):
        ModelConfig(d_model=64, n_layers=2, n_heads=1, d_ff=256, n_outcomes=-1)


def test_model_config_rejects_d_ff_smaller_than_d_model() -> None:
    """d_ff is the SwiGLU intermediate width; smaller than d_model is a typo."""
    with pytest.raises(ValueError, match=r"d_ff.*d_model"):
        ModelConfig(d_model=64, n_layers=2, n_heads=1, d_ff=32)


def test_model_config_rejects_non_positive_rope_base() -> None:
    with pytest.raises(ValueError, match="rope_base"):
        ModelConfig(d_model=64, n_layers=2, n_heads=1, d_ff=256, rope_base=0.0)


# ---------------------------------------------------------------------------
# Production constants
# ---------------------------------------------------------------------------


def test_supernet_dimensions() -> None:
    """The plan §5 pins SUPERNET to d=640 / 10 layers / 10 heads / head_dim=64.

    ``d_ff`` was originally 4 × d_model (2560); round-3 perf work
    (commit landing this test edit) flipped to ~8/3 × d_model rounded
    up to a 128-multiple = 1792. The new ratio matches Llama-1/2/3
    SwiGLU sizing and cuts ~30% of FFN compute. See
    :data:`pawn.config.SUPERNET` for the rationale comment.
    """
    assert SUPERNET.d_model == 640
    assert SUPERNET.n_layers == 10
    assert SUPERNET.n_heads == 10
    assert SUPERNET.head_dim == 64
    assert SUPERNET.d_ff == 1792  # ≈ 8/3 × d_model, rounded to 128-multiple


def test_variants_dimensions() -> None:
    """small / base / large match the plan §5 figures."""
    assert set(VARIANTS.keys()) == {"small", "base", "large"}
    assert VARIANTS["small"].d_model == 256
    assert VARIANTS["small"].n_heads == 4
    assert VARIANTS["base"].d_model == 512
    assert VARIANTS["base"].n_heads == 8
    assert VARIANTS["large"] is SUPERNET  # large IS the supernet


def test_tiny_supernet_dimensions() -> None:
    """TINY_SUPERNET = d=192 / 4 layers / 3 heads per plan §5."""
    assert TINY_SUPERNET.d_model == 192
    assert TINY_SUPERNET.n_layers == 4
    assert TINY_SUPERNET.n_heads == 3
    assert TINY_SUPERNET.head_dim == 64
    assert TINY_SUPERNET.d_ff == 768


def test_tiny_variants_dimensions() -> None:
    assert set(TINY_VARIANTS.keys()) == {"small", "base", "large"}
    assert TINY_VARIANTS["small"].d_model == 64
    assert TINY_VARIANTS["small"].n_heads == 1
    assert TINY_VARIANTS["base"].d_model == 128
    assert TINY_VARIANTS["base"].n_heads == 2
    assert TINY_VARIANTS["large"] is TINY_SUPERNET


# ---------------------------------------------------------------------------
# validate_nested
# ---------------------------------------------------------------------------


def test_validate_nested_accepts_production_pairs() -> None:
    """The module-level constants satisfy their own nesting check.

    (Also verified at import via `_check_constants_nest_at_import`; this
    test makes the contract explicit.)
    """
    for cfg in VARIANTS.values():
        validate_nested(cfg, SUPERNET)
    for cfg in TINY_VARIANTS.values():
        validate_nested(cfg, TINY_SUPERNET)


def test_validate_nested_accepts_supernet_as_its_own_variant() -> None:
    """A supernet trivially nests within itself (large variant case)."""
    validate_nested(SUPERNET, SUPERNET)


def test_validate_nested_rejects_head_dim_mismatch() -> None:
    """A v1-style large (d=640, 8 heads, head_dim=80) cannot nest under the
    v2 supernet (head_dim=64). This is precisely what the v2 architecture
    fixes — the test pins it."""
    v1_style = ModelConfig(d_model=640, n_layers=10, n_heads=8, d_ff=2560, head_dim=80)
    with pytest.raises(NestingError, match="head_dim mismatch"):
        validate_nested(v1_style, SUPERNET)


def test_validate_nested_rejects_layer_mismatch() -> None:
    """Variants must share the supernet's depth — they're width slices,
    not depth slices."""
    shallower = ModelConfig(d_model=256, n_layers=4, n_heads=4, d_ff=1024)
    with pytest.raises(NestingError, match="n_layers"):
        validate_nested(shallower, SUPERNET)


def test_validate_nested_rejects_oversized_d_model() -> None:
    too_wide = ModelConfig(d_model=896, n_layers=10, n_heads=14, d_ff=3584)
    with pytest.raises(NestingError, match="d_model"):
        validate_nested(too_wide, SUPERNET)


def test_validate_nested_oversized_n_heads_caught_by_d_model_check() -> None:
    """With head_dim fixed, `n_heads > supernet.n_heads` is *implied by*
    `d_model > supernet.d_model` — the variant's d_model = head_dim ×
    n_heads can't be ≤ super.d_model while n_heads exceeds super.n_heads.
    The d_model branch catches it; no separate n_heads check is needed.

    `tighter_super` here has fewer heads (9) than `wider_variant` (10),
    but its d_model (9×64=576) is smaller too, so the d_model branch
    fires before the conceptual n_heads check would matter."""
    wider_variant = ModelConfig(d_model=640, n_layers=10, n_heads=10, d_ff=2560)
    tighter_super = ModelConfig(d_model=576, n_layers=10, n_heads=9, d_ff=2304)
    with pytest.raises(NestingError, match="d_model"):
        validate_nested(wider_variant, tighter_super)


def test_validate_nested_rejects_oversized_d_ff() -> None:
    """Inner FFN width must fit in the supernet's FFN slab."""
    fat_ffn = ModelConfig(d_model=256, n_layers=10, n_heads=4, d_ff=4096)
    with pytest.raises(NestingError, match="d_ff"):
        validate_nested(fat_ffn, SUPERNET)


def test_validate_nested_rejects_vocab_size_mismatch() -> None:
    """vocab_size must match exactly — a variant with a different vocab
    wouldn't share embedding tables with the supernet."""
    cfg = ModelConfig(d_model=256, n_layers=10, n_heads=4, d_ff=1024, vocab_size=2000)
    with pytest.raises(NestingError, match="vocab_size"):
        validate_nested(cfg, SUPERNET)


def test_validate_nested_rejects_max_seq_len_mismatch() -> None:
    cfg = ModelConfig(d_model=256, n_layers=10, n_heads=4, d_ff=1024, max_seq_len=1024)
    with pytest.raises(NestingError, match="max_seq_len"):
        validate_nested(cfg, SUPERNET)


def test_validate_nested_rejects_rope_base_mismatch() -> None:
    """RoPE phase tables are baked from rope_base; a variant with a
    different base would see different positional encoding."""
    cfg = ModelConfig(d_model=256, n_layers=10, n_heads=4, d_ff=1024, rope_base=500.0)
    with pytest.raises(NestingError, match="rope_base"):
        validate_nested(cfg, SUPERNET)


def test_validate_nested_rejects_n_outcomes_mismatch() -> None:
    """A variant with a different outcome-token count can't share the
    supernet's outcome embedding bank."""
    cfg = ModelConfig(d_model=256, n_layers=10, n_heads=4, d_ff=1024, n_outcomes=5)
    with pytest.raises(NestingError, match="n_outcomes"):
        validate_nested(cfg, SUPERNET)


def test_variants_dict_is_immutable() -> None:
    """VARIANTS is wrapped in MappingProxyType so an accidental
    `VARIANTS["small"] = ...` raises rather than silently swapping out a
    config."""
    with pytest.raises(TypeError):
        VARIANTS["small"] = SUPERNET  # type: ignore[index]
    with pytest.raises(TypeError):
        del VARIANTS["small"]  # type: ignore[attr-defined]


def test_tiny_variants_dict_is_immutable() -> None:
    with pytest.raises(TypeError):
        TINY_VARIANTS["small"] = TINY_SUPERNET  # type: ignore[index]


def test_nesting_error_is_value_error_subclass() -> None:
    """`NestingError` is a `ValueError` so existing `except ValueError`
    sites still catch it."""
    assert issubclass(NestingError, ValueError)


# ---------------------------------------------------------------------------
# Lightweight import contract
# ---------------------------------------------------------------------------


def test_config_module_imports_without_jax_or_torch() -> None:
    """`pawn.config` must be importable in a lightweight context (e.g. by
    the sweep driver, the dashboard reader, the legacy converter at the
    point it hasn't yet built the model)."""
    probe = textwrap.dedent(
        """
        import sys
        import pawn.config  # noqa: F401
        heavy = [m for m in sys.modules if m.split('.')[0] in ('jax', 'jaxlib', 'torch', 'equinox', 'optax')]
        if heavy:
            print('LEAKED:' + ','.join(sorted(heavy)))
            raise SystemExit(1)
        print('OK')
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"importing pawn.config dragged in heavy modules:\n"
        f"  stdout: {result.stdout.strip()}\n"
        f"  stderr: {result.stderr.strip()}"
    )
    assert result.stdout.strip() == "OK"
