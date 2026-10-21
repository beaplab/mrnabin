"""Tests for 128-bit (hi, lo) operations."""

import operator
from collections.abc import Callable

import numpy as np
import pytest

from .bit_ops import (
    MAX_U64,
    U64,
    equals_zero_128,
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
    max_128,
)

MASK128 = (1 << 128) - 1
ALL_ONES = int(MAX_U64) << 64 | int(MAX_U64)


def _to_128(hi: int, lo: int) -> int:
    return (int(hi) << 64) | int(lo)


def _from_128(val: int) -> tuple[np.uint64, np.uint64]:
    return U64(val >> 64), U64(val & ((1 << 64) - 1))


def _shift_ref(val: int, n: int) -> int:
    """Python reference for ishft: positive n = left, negative n = right, truncated to 128 bits."""
    return (val << n) & MASK128 if n >= 0 else val >> -n


class TestIshftHilo:
    @pytest.mark.parametrize(
        "val, n",
        [
            (1, 1),
            (1 << 63, 1),  # lo bit 63 lands in hi bit 0
            (0xFF, 64),
            (0xFF << 64 | 0xFF, 128),
            (4, -1),
            (1 << 64, -1),  # hi bit 0 lands in lo bit 63
            (0xFF << 64, -64),
            (0xFF << 64 | 0xFF, -128),
            (0xAB << 64 | 0xCD, 0),
            (0x123 << 64 | 0x456, 3),
            (0x123 << 64 | 0x456, -3),
            (ALL_ONES, 1),
            (ALL_ONES, -1),
            (ALL_ONES, 3),
            (ALL_ONES, -3),
            (ALL_ONES, 64),
            (ALL_ONES, -64),
        ],
    )
    def test_matches_python_int(self, val: int, n: int) -> None:
        hi, lo = ishft_hilo(*_from_128(val), n)
        assert _to_128(hi, lo) == _shift_ref(val, n)


class TestIbitsHilo:
    @pytest.mark.parametrize(
        "val, pos, length",
        [
            (0xFF, 0, 8),
            (0xFF, 0, 4),
            (0xF0, 4, 4),
            ((1 << 64) | (1 << 63), 63, 2),  # spans the hi/lo boundary
            (0xAB << 64, 64, 8),
            (0xFF << 64 | 0xFF, 128, 8),
            (0xFF << 64 | 0xFF, 0, 0),
            (0xAB << 64 | 0xCD, 0, 128),
            (ALL_ONES, 0, 3),
            (ALL_ONES, 0, 64),
            (ALL_ONES, 125, 3),
            (ALL_ONES, 60, 8),
            (ALL_ONES, 64, 8),
            (ALL_ONES, 0, 120),
            (ALL_ONES, 0, 128),
        ],
    )
    def test_matches_python_int(self, val: int, pos: int, length: int) -> None:
        hi, lo = ibits_hilo(*_from_128(val), pos, length)
        assert _to_128(hi, lo) == (val >> pos) & ((1 << length) - 1)


_HILO_OPS: list[tuple[Callable, Callable[[int, int], int]]] = [
    (iand_hilo, operator.and_),
    (ior_hilo, operator.or_),
    (ieor_hilo, operator.xor),
]
_BINARY_INPUTS = [
    (0xFF << 64 | 0xFF, 0x0F << 64 | 0xF0),
    (0xF0 << 64 | 0x0F, 0x0F << 64 | 0xF0),
    (0xAB << 64 | 0xCD, 0),
    (0xAB << 64 | 0xCD, 0xAB << 64 | 0xCD),
    (0xAB << 64 | 0xCD, ALL_ONES),
]


class TestBinaryOpsHilo:
    @pytest.mark.parametrize("a, b", _BINARY_INPUTS)
    @pytest.mark.parametrize("hilo_op, py_op", _HILO_OPS)
    def test_matches_python_int(self, hilo_op: Callable, py_op: Callable, a: int, b: int) -> None:
        hi, lo = hilo_op(*_from_128(a), *_from_128(b))
        assert _to_128(hi, lo) == py_op(a, b)


class TestMake128:
    def test_values(self) -> None:
        arr = make_128(U64(0xAB), U64(0xCD))
        assert arr[0] == 0xAB
        assert arr[1] == 0xCD


def test_max_128_is_all_ones() -> None:
    assert _to_128(*max_128()) == ALL_ONES


@pytest.mark.parametrize("val, expected", [(0, True), (1, False), (1 << 64, False), (ALL_ONES, False)])
def test_equals_zero_128(val: int, expected: bool) -> None:
    """Zero only if both halves are; a set bit in either half is enough to fail."""
    assert equals_zero_128(_from_128(val)) is expected


_WRAPPER_INPUTS = [0x123 << 64 | 0x456, ALL_ONES, (1 << 64) | (1 << 63)]


class Test128Wrappers:
    @pytest.mark.parametrize("val", _WRAPPER_INPUTS)
    def test_match_hilo(self, val: int) -> None:
        """Each *_128 tuple wrapper returns the same result as its *_hilo counterpart."""
        u = _from_128(val)
        other = _from_128(0x0F << 64 | 0xF0)
        for n in (3, -3, 64, 0):
            assert ishft_128(u, n) == ishft_hilo(u[0], u[1], n)
        for pos, length in ((4, 8), (63, 2), (0, 128)):
            assert ibits_128(u, pos, length) == ibits_hilo(u[0], u[1], pos, length)
        for hilo_op, op128 in ((iand_hilo, iand_128), (ior_hilo, ior_128), (ieor_hilo, ieor_128)):
            assert op128(u, other) == hilo_op(u[0], u[1], other[0], other[1])
