"""Tests for pawn/data_utils.py (unpack_grid)."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from pawn.data_utils import _unpack_bits_cache, unpack_grid


class TestUnpackGrid:
    @pytest.mark.unit
    def test_output_shape(self):
        packed = torch.zeros(4, 64, dtype=torch.long)
        out = unpack_grid(packed)
        assert out.shape == (4, 64, 64)

    @pytest.mark.unit
    def test_all_zero_input(self):
        packed = torch.zeros(2, 64, dtype=torch.long)
        out = unpack_grid(packed)
        assert (out == 0.0).all()

    @pytest.mark.unit
    def test_output_dtype(self):
        packed = torch.zeros(1, 64, dtype=torch.long)
        out = unpack_grid(packed)
        assert out.dtype == torch.float32

    @pytest.mark.unit
    def test_single_bit_set(self):
        """Setting bit k in a row should produce entry k==1.0, rest 0.0."""
        packed = torch.zeros(1, 64, dtype=torch.long)
        packed[0, 3] = 1 << 5  # bit 5 set in row 3
        out = unpack_grid(packed)
        assert out[0, 3, 5].item() == 1.0
        assert out[0, 3, 4].item() == 0.0
        assert out[0, 3, 6].item() == 0.0
        # Other rows entirely zero
        assert (out[0, :3] == 0.0).all()
        assert (out[0, 4:] == 0.0).all()

    @pytest.mark.unit
    def test_all_bits_set(self):
        """All 64 bits set (row of -1, two's-complement = all bits) gives all-ones."""
        packed = torch.full((1, 64), -1, dtype=torch.long)  # all bits set
        out = unpack_grid(packed)
        # In two's complement int64, -1 has all 64 bits set. The signed right
        # shift preserves the sign bit, but the `& 1` mask always extracts the
        # LSB of the shifted value. So result should be all-ones.
        # (Verify: (-1 >> k) & 1 == 1 for k in 0..63 in signed arithmetic.)
        assert (out == 1.0).all()

    @pytest.mark.unit
    def test_mixed_bits(self):
        """Test a known bit pattern."""
        packed = torch.zeros(1, 64, dtype=torch.long)
        # Set bits 0, 1, 2 in row 7
        packed[0, 7] = 0b111
        out = unpack_grid(packed)
        assert out[0, 7, 0].item() == 1.0
        assert out[0, 7, 1].item() == 1.0
        assert out[0, 7, 2].item() == 1.0
        assert out[0, 7, 3].item() == 0.0

    @pytest.mark.unit
    def test_handles_higher_dim_input(self):
        """Works with (B, T, 64)."""
        packed = torch.zeros(3, 5, 64, dtype=torch.long)
        packed[0, 2, 10] = 1 << 20
        out = unpack_grid(packed)
        assert out.shape == (3, 5, 64, 64)
        assert out[0, 2, 10, 20].item() == 1.0

    @pytest.mark.unit
    def test_bit_cache_populated(self):
        packed = torch.zeros(1, 64, dtype=torch.long)
        _ = unpack_grid(packed)
        assert packed.device in _unpack_bits_cache

    @pytest.mark.unit
    def test_high_bit_62(self):
        """Bit 62 (near MSB) should unpack correctly."""
        packed = torch.zeros(1, 64, dtype=torch.long)
        packed[0, 0] = 1 << 62
        out = unpack_grid(packed)
        assert out[0, 0, 62].item() == 1.0
        assert out[0, 0, 61].item() == 0.0

    @pytest.mark.unit
    @given(
        bit_positions=st.lists(
            st.integers(min_value=0, max_value=62), min_size=0, max_size=8, unique=True
        ),
        row=st.integers(min_value=0, max_value=63),
    )
    @settings(max_examples=30, deadline=None)
    def test_property_unpack_single_row(self, bit_positions, row):
        """Property: unpacked bits at row == the positions we set.

        Bits 0..62 only — bit 63 cannot be assigned as a Python int to
        int64 tensor without ValueError (sign bit).
        """
        packed = torch.zeros(1, 64, dtype=torch.long)
        value = 0
        for b in bit_positions:
            value |= 1 << b
        packed[0, row] = value
        out = unpack_grid(packed)
        for b in range(64):
            expected = 1.0 if b in bit_positions else 0.0
            assert out[0, row, b].item() == expected

    @pytest.mark.unit
    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=10, deadline=None)
    def test_property_roundtrip(self, seed):
        """unpack_grid then re-pack via shift+OR should return original."""
        g = torch.Generator().manual_seed(seed)
        packed = torch.randint(0, 2**62, (1, 64), dtype=torch.long, generator=g)
        out = unpack_grid(packed)
        # Re-pack
        bits = torch.arange(64, dtype=torch.long)
        repacked = (out.long() << bits).sum(dim=-1)
        assert torch.equal(repacked, packed)
