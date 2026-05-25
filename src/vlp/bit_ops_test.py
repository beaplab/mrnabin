"""Tests for 128-bit (hi, lo) operations."""

import numpy as np

from .bit_ops import (
    MAX_U64,
    U64,
    iand_128,
    iand_hilo,
    ibits_128,
    ibits_hilo,
    ieor_128,
    ieor_hilo,
    ior_128,
    ior_hilo,
    ishft_128,
    ishft_hilo,
    make_128,
)


def _to_128(hi: int, lo: int) -> int:
    return (hi << 64) | lo


def _from_128(val: int) -> tuple[np.uint64, np.uint64]:
    return U64(val >> 64), U64(val & ((1 << 64) - 1))


class TestIshftHilo:
    def test_left_shift_small(self) -> None:
        hi, lo = ishft_hilo(U64(0), U64(1), 1)
        assert _to_128(hi, lo) == 2

    def test_left_shift_across_boundary(self) -> None:
        """Shifting lo bit 63 left by 1 should land in hi bit 0."""
        hi, lo = ishft_hilo(U64(0), U64(1 << 63), 1)
        assert hi == 1
        assert lo == 0

    def test_left_shift_by_64(self) -> None:
        hi, lo = ishft_hilo(U64(0), U64(0xFF), 64)
        assert hi == 0xFF
        assert lo == 0

    def test_left_shift_by_128(self) -> None:
        hi, lo = ishft_hilo(U64(0xFF), U64(0xFF), 128)
        assert hi == 0
        assert lo == 0

    def test_right_shift_small(self) -> None:
        hi, lo = ishft_hilo(U64(0), U64(4), -1)
        assert _to_128(hi, lo) == 2

    def test_right_shift_across_boundary(self) -> None:
        """Shifting hi bit 0 right by 1 should land in lo bit 63."""
        hi, lo = ishft_hilo(U64(1), U64(0), -1)
        assert hi == 0
        assert lo == (1 << 63)

    def test_right_shift_by_64(self) -> None:
        hi, lo = ishft_hilo(U64(0xFF), U64(0), -64)
        assert hi == 0
        assert lo == 0xFF

    def test_right_shift_by_128(self) -> None:
        hi, lo = ishft_hilo(U64(0xFF), U64(0xFF), -128)
        assert hi == 0
        assert lo == 0

    def test_shift_zero(self) -> None:
        hi, lo = ishft_hilo(U64(0xAB), U64(0xCD), 0)
        assert hi == 0xAB
        assert lo == 0xCD

    def test_left_shift_preserves_bits(self) -> None:
        """Shift a known 128-bit value left by 3."""
        val = (0x123 << 64) | 0x456
        hi, lo = ishft_hilo(*_from_128(val), 3)
        assert _to_128(hi, lo) == val << 3

    def test_right_shift_preserves_bits(self) -> None:
        """Shift a known 128-bit value right by 3."""
        val = (0x123 << 64) | 0x456
        hi, lo = ishft_hilo(*_from_128(val), -3)
        assert _to_128(hi, lo) == val >> 3

    def test_left_shift_max_u64(self) -> None:
        """Left-shifting all-ones by 3 should clear the bottom 3 bits."""
        hi, lo = ishft_hilo(MAX_U64, MAX_U64, 3)
        expected = ((1 << 128) - 1) << 3 & ((1 << 128) - 1)
        assert _to_128(hi, lo) == expected

    def test_right_shift_max_u64(self) -> None:
        """Right-shifting all-ones by 3 should clear the top 3 bits."""
        hi, lo = ishft_hilo(MAX_U64, MAX_U64, -3)
        expected = ((1 << 128) - 1) >> 3
        assert _to_128(hi, lo) == expected

    def test_left_shift_max_u64_by_64(self) -> None:
        hi, lo = ishft_hilo(MAX_U64, MAX_U64, 64)
        assert hi == MAX_U64
        assert lo == 0

    def test_right_shift_max_u64_by_64(self) -> None:
        hi, lo = ishft_hilo(MAX_U64, MAX_U64, -64)
        assert hi == 0
        assert lo == MAX_U64

    def test_left_shift_max_u64_by_1(self) -> None:
        hi, lo = ishft_hilo(MAX_U64, MAX_U64, 1)
        assert hi == MAX_U64
        assert lo == U64(0xFFFFFFFFFFFFFFFE)

    def test_right_shift_max_u64_by_1(self) -> None:
        hi, lo = ishft_hilo(MAX_U64, MAX_U64, -1)
        assert hi == U64(0x7FFFFFFFFFFFFFFF)
        assert lo == MAX_U64


class TestIbitsHilo:
    def test_extract_low_bits(self) -> None:
        """Extract bits 0..7 from lo."""
        hi, lo = ibits_hilo(U64(0), U64(0xFF), 0, 8)
        assert _to_128(hi, lo) == 0xFF

    def test_extract_low_bits_masked(self) -> None:
        """Extract 4 bits from a value with more bits set."""
        hi, lo = ibits_hilo(U64(0), U64(0xFF), 0, 4)
        assert _to_128(hi, lo) == 0x0F

    def test_extract_from_middle(self) -> None:
        """Extract bits 4..7 from lo."""
        hi, lo = ibits_hilo(U64(0), U64(0xF0), 4, 4)
        assert _to_128(hi, lo) == 0x0F

    def test_extract_across_boundary(self) -> None:
        """Extract bits that span the hi/lo boundary."""
        # Set bit 63 in lo and bit 0 in hi
        val = (1 << 64) | (1 << 63)
        hi, lo = ibits_hilo(*_from_128(val), 63, 2)
        assert _to_128(hi, lo) == 3

    def test_extract_from_hi(self) -> None:
        """Extract bits entirely from hi half."""
        hi, lo = ibits_hilo(U64(0xAB), U64(0), 64, 8)
        assert _to_128(hi, lo) == 0xAB

    def test_pos_beyond_128(self) -> None:
        hi, lo = ibits_hilo(U64(0xFF), U64(0xFF), 128, 8)
        assert hi == 0
        assert lo == 0

    def test_length_zero(self) -> None:
        """Extracting 0 bits should return 0."""
        hi, lo = ibits_hilo(U64(0xFF), U64(0xFF), 0, 0)
        assert hi == 0
        assert lo == 0

    def test_length_128(self) -> None:
        """Extracting all 128 bits from pos 0 returns the original value."""
        hi, lo = ibits_hilo(U64(0xAB), U64(0xCD), 0, 128)
        assert hi == 0xAB
        assert lo == 0xCD

    def test_max_u64_extract_low_3(self) -> None:
        """Extract bottom 3 bits from all-ones gives 0b111 = 7."""
        hi, lo = ibits_hilo(MAX_U64, MAX_U64, 0, 3)
        assert _to_128(hi, lo) == 7

    def test_max_u64_extract_low_64(self) -> None:
        hi, lo = ibits_hilo(MAX_U64, MAX_U64, 0, 64)
        assert hi == 0
        assert lo == MAX_U64

    def test_max_u64_extract_top_3(self) -> None:
        """Extract top 3 bits (pos=125, len=3) from all-ones gives 7."""
        hi, lo = ibits_hilo(MAX_U64, MAX_U64, 125, 3)
        assert _to_128(hi, lo) == 7

    def test_max_u64_extract_across_boundary(self) -> None:
        """Extract 8 bits spanning the hi/lo boundary from all-ones."""
        hi, lo = ibits_hilo(MAX_U64, MAX_U64, 60, 8)
        assert _to_128(hi, lo) == 0xFF

    def test_max_u64_extract_from_hi(self) -> None:
        """Extract 8 bits entirely from hi half of all-ones."""
        hi, lo = ibits_hilo(MAX_U64, MAX_U64, 64, 8)
        assert _to_128(hi, lo) == 0xFF

    def test_max_u64_extract_120(self) -> None:
        """Extract 120 bits from pos 0. Should be 2^120 - 1."""
        hi, lo = ibits_hilo(MAX_U64, MAX_U64, 0, 120)
        assert _to_128(hi, lo) == (1 << 120) - 1

    def test_max_u64_extract_all_128(self) -> None:
        hi, lo = ibits_hilo(MAX_U64, MAX_U64, 0, 128)
        assert hi == MAX_U64
        assert lo == MAX_U64


class TestIandHilo:
    def test_and_masks_bits(self) -> None:
        hi, lo = iand_hilo(U64(0xFF), U64(0xFF), U64(0x0F), U64(0xF0))
        assert hi == 0x0F
        assert lo == 0xF0

    def test_and_with_zero(self) -> None:
        hi, lo = iand_hilo(U64(0xFF), U64(0xFF), U64(0), U64(0))
        assert hi == 0
        assert lo == 0

    def test_and_with_all_ones(self) -> None:
        MAX = U64(0xFFFFFFFFFFFFFFFF)
        hi, lo = iand_hilo(U64(0xAB), U64(0xCD), MAX, MAX)
        assert hi == 0xAB
        assert lo == 0xCD


class TestIorHilo:
    def test_or_sets_bits(self) -> None:
        hi, lo = ior_hilo(U64(0xF0), U64(0x0F), U64(0x0F), U64(0xF0))
        assert hi == 0xFF
        assert lo == 0xFF

    def test_or_with_zero(self) -> None:
        hi, lo = ior_hilo(U64(0xAB), U64(0xCD), U64(0), U64(0))
        assert hi == 0xAB
        assert lo == 0xCD


class TestIeorHilo:
    def test_xor_flips_bits(self) -> None:
        hi, lo = ieor_hilo(U64(0xFF), U64(0xFF), U64(0x0F), U64(0xF0))
        assert hi == 0xF0
        assert lo == 0x0F

    def test_xor_with_self_is_zero(self) -> None:
        hi, lo = ieor_hilo(U64(0xAB), U64(0xCD), U64(0xAB), U64(0xCD))
        assert hi == 0
        assert lo == 0

    def test_xor_with_zero(self) -> None:
        hi, lo = ieor_hilo(U64(0xAB), U64(0xCD), U64(0), U64(0))
        assert hi == 0xAB
        assert lo == 0xCD


class TestMake128:
    def test_values(self) -> None:
        arr = make_128(U64(0xAB), U64(0xCD))
        assert arr[0] == 0xAB
        assert arr[1] == 0xCD

    def test_zeros(self) -> None:
        arr = make_128(U64(0), U64(0))
        assert arr[0] == 0
        assert arr[1] == 0


class TestIshft128:
    def test_matches_hilo(self) -> None:
        arr = make_128(U64(0x123), U64(0x456))
        result = ishft_128(arr, 3)
        hi, lo = ishft_hilo(U64(0x123), U64(0x456), 3)
        assert result[0] == hi
        assert result[1] == lo

    def test_right_shift(self) -> None:
        arr = make_128(U64(0x123), U64(0x456))
        result = ishft_128(arr, -3)
        hi, lo = ishft_hilo(U64(0x123), U64(0x456), -3)
        assert result[0] == hi
        assert result[1] == lo

    def test_shift_zero(self) -> None:
        arr = make_128(U64(0xAB), U64(0xCD))
        result = ishft_128(arr, 0)
        assert result[0] == 0xAB
        assert result[1] == 0xCD


class TestIbits128:
    def test_matches_hilo(self) -> None:
        arr = make_128(U64(0xFF), U64(0xFF))
        result = ibits_128(arr, 4, 8)
        hi, lo = ibits_hilo(U64(0xFF), U64(0xFF), 4, 8)
        assert result[0] == hi
        assert result[1] == lo

    def test_across_boundary(self) -> None:
        arr = U64(1), U64(1 << 63)
        result = ibits_128(arr, 63, 2)
        assert _to_128(result[0], result[1]) == 3


class TestIand128:
    def test_matches_hilo(self) -> None:
        a = make_128(U64(0xFF), U64(0xFF))
        b = make_128(U64(0x0F), U64(0xF0))
        result = iand_128(a, b)
        assert result[0] == 0x0F
        assert result[1] == 0xF0

    def test_with_zero(self) -> None:
        a = make_128(U64(0xFF), U64(0xFF))
        b = make_128(U64(0), U64(0))
        result = iand_128(a, b)
        assert result[0] == 0
        assert result[1] == 0


class TestIor128:
    def test_matches_hilo(self) -> None:
        a = make_128(U64(0xF0), U64(0x0F))
        b = make_128(U64(0x0F), U64(0xF0))
        result = ior_128(a, b)
        assert result[0] == 0xFF
        assert result[1] == 0xFF

    def test_with_zero(self) -> None:
        a = make_128(U64(0xAB), U64(0xCD))
        b = make_128(U64(0), U64(0))
        result = ior_128(a, b)
        assert result[0] == 0xAB
        assert result[1] == 0xCD


class TestIeor128:
    def test_matches_hilo(self) -> None:
        a = make_128(U64(0xFF), U64(0xFF))
        b = make_128(U64(0x0F), U64(0xF0))
        result = ieor_128(a, b)
        assert result[0] == 0xF0
        assert result[1] == 0x0F

    def test_self_is_zero(self) -> None:
        a = make_128(U64(0xAB), U64(0xCD))
        result = ieor_128(a, a)
        assert result[0] == 0
        assert result[1] == 0
