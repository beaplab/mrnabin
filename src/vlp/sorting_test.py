"""Tests for sorting utilities."""

import numpy as np
import pytest

from .sorting import argsort_128, binary_search_128, sort_128


class TestNpSort128:
    @pytest.mark.parametrize(
        "rows, expected_idx",
        [
            ([[1, 0], [2, 0], [3, 0]], [0, 1, 2]),  # already sorted
            ([[3, 30], [2, 20], [1, 10]], [2, 1, 0]),  # reverse sorted
            ([[5, 30], [5, 10], [5, 20]], [1, 2, 0]),  # same hi, different lo
            ([[42, 99]], [0]),  # single element
            ([[2, 0], [1, 999]], [1, 0]),  # hi takes precedence over lo
            ([[7, 3], [1, 99], [7, 1], [3, 50], [1, 2]], [4, 1, 3, 2, 0]),
        ],
    )
    def test_sort_and_argsort(self, rows: list[list[int]], expected_idx: list[int]) -> None:
        arr = np.array(rows, dtype=np.uint64)
        expected_sorted = arr[expected_idx]
        np.testing.assert_array_equal(argsort_128(arr), expected_idx)
        sort_128(arr)
        np.testing.assert_array_equal(arr, expected_sorted)


class TestNpArgsort128:
    def test_stable(self) -> None:
        """Equal keys should preserve original order."""
        arr = np.array([[5, 10], [5, 10], [5, 10]], dtype=np.uint64)
        idx = argsort_128(arr)
        np.testing.assert_array_equal(idx, [0, 1, 2])

    def test_does_not_modify_input(self) -> None:
        arr = np.array([[3, 30], [1, 10], [2, 20]], dtype=np.uint64)
        original = arr.copy()
        argsort_128(arr)
        np.testing.assert_array_equal(arr, original)


_KEYS3 = ([1, 2, 3], [10, 20, 30])


class TestBinarySearch128:
    @pytest.mark.parametrize(
        "keys, val, expected",
        [
            (_KEYS3, (2, 20), 1),  # found
            (_KEYS3, (2, 25), -1),  # not found
            (_KEYS3, (1, 10), 0),  # first element
            (_KEYS3, (3, 30), 2),  # last element
            (([5], [42]), (5, 42), 0),  # single element found
            (([5], [42]), (5, 43), -1),  # single element not found
            (([1, 1, 1], [10, 20, 30]), (1, 20), 1),  # same hi, different lo
            (([1, 1, 1], [10, 20, 30]), (1, 15), -1),
        ],
    )
    def test_search(self, keys: tuple[list[int], list[int]], val: tuple[int, int], expected: int) -> None:
        keys_arr = np.array(keys, dtype=np.uint64)  # shape (2, n): row 0 = hi, row 1 = lo
        assert binary_search_128(keys_arr, np.array(val, dtype=np.uint64)) == expected
