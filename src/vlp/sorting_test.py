"""Tests for sorting utilities."""

import numpy as np
import numpy.typing as npt

from .sorting import argsort_128, binary_search_128, sort_128


class TestNpSort128:
    def test_already_sorted(self) -> None:
        arr = np.array([[1, 0], [2, 0], [3, 0]], dtype=np.uint64)
        sort_128(arr)
        np.testing.assert_array_equal(arr[:, 0], [1, 2, 3])
        np.testing.assert_array_equal(arr[:, 1], [0, 0, 0])

    def test_reverse_sorted(self) -> None:
        arr = np.array([[3, 30], [2, 20], [1, 10]], dtype=np.uint64)
        sort_128(arr)
        np.testing.assert_array_equal(arr[:, 0], [1, 2, 3])
        np.testing.assert_array_equal(arr[:, 1], [10, 20, 30])

    def test_same_hi_different_lo(self) -> None:
        arr = np.array([[5, 30], [5, 10], [5, 20]], dtype=np.uint64)
        sort_128(arr)
        np.testing.assert_array_equal(arr[:, 0], [5, 5, 5])
        np.testing.assert_array_equal(arr[:, 1], [10, 20, 30])

    def test_single_element(self) -> None:
        arr = np.array([[42, 99]], dtype=np.uint64)
        sort_128(arr)
        np.testing.assert_array_equal(arr[:, 0], [42])
        np.testing.assert_array_equal(arr[:, 1], [99])


class TestNpArgsort128:
    def test_reverse_sorted(self) -> None:
        arr = np.array([[3, 30], [2, 20], [1, 10]], dtype=np.uint64)
        idx = argsort_128(arr)
        np.testing.assert_array_equal(idx, [2, 1, 0])

    def test_same_hi_different_lo(self) -> None:
        arr = np.array([[5, 30], [5, 10], [5, 20]], dtype=np.uint64)
        idx = argsort_128(arr)
        np.testing.assert_array_equal(idx, [1, 2, 0])

    def test_single_element(self) -> None:
        arr = np.array([[42, 99]], dtype=np.uint64)
        idx = argsort_128(arr)
        np.testing.assert_array_equal(idx, [0])

    def test_already_sorted(self) -> None:
        arr = np.array([[1, 0], [2, 0], [3, 0]], dtype=np.uint64)
        idx = argsort_128(arr)
        np.testing.assert_array_equal(idx, [0, 1, 2])

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

    def test_consistent_with_sort(self) -> None:
        """Indexing by argsort should give the same result as sorting."""
        arr = np.array([[7, 3], [1, 99], [7, 1], [3, 50], [1, 2]], dtype=np.uint64)
        idx = argsort_128(arr)
        sorted_arr = arr.copy()
        sort_128(sorted_arr)
        np.testing.assert_array_equal(arr[idx], sorted_arr)

    def test_hi_takes_precedence(self) -> None:
        """Higher hi should sort after lower hi regardless of lo."""
        arr = np.array([[2, 0], [1, 999]], dtype=np.uint64)
        idx = argsort_128(arr)
        np.testing.assert_array_equal(idx, [1, 0])


class TestBinarySearch128:
    def _make_keys(self, hi: list[int], lo: list[int]) -> npt.NDArray[np.uint64]:
        keys = np.empty((2, len(hi)), dtype=np.uint64)
        keys[0] = np.array(hi, dtype=np.uint64)
        keys[1] = np.array(lo, dtype=np.uint64)
        return keys

    def _make_val(self, hi: int, lo: int) -> npt.NDArray[np.uint64]:
        return np.array([hi, lo], dtype=np.uint64)

    def test_found(self) -> None:
        keys = self._make_keys([1, 2, 3], [10, 20, 30])
        assert binary_search_128(keys, self._make_val(2, 20)) == 1

    def test_not_found(self) -> None:
        keys = self._make_keys([1, 2, 3], [10, 20, 30])
        assert binary_search_128(keys, self._make_val(2, 25)) == -1

    def test_first_element(self) -> None:
        keys = self._make_keys([1, 2, 3], [10, 20, 30])
        assert binary_search_128(keys, self._make_val(1, 10)) == 0

    def test_last_element(self) -> None:
        keys = self._make_keys([1, 2, 3], [10, 20, 30])
        assert binary_search_128(keys, self._make_val(3, 30)) == 2

    def test_single_element_found(self) -> None:
        keys = self._make_keys([5], [42])
        assert binary_search_128(keys, self._make_val(5, 42)) == 0

    def test_single_element_not_found(self) -> None:
        keys = self._make_keys([5], [42])
        assert binary_search_128(keys, self._make_val(5, 43)) == -1

    def test_same_hi_different_lo(self) -> None:
        keys = self._make_keys([1, 1, 1], [10, 20, 30])
        assert binary_search_128(keys, self._make_val(1, 20)) == 1
        assert binary_search_128(keys, self._make_val(1, 15)) == -1
