"""Sorting utilities for 128-bit integers."""

import numba
import numpy as np
import numpy.typing as npt


@numba.njit(cache=True)
def _radix_sort_128(arr: npt.NDArray[np.uint64]) -> None:
    """LSD byte-wise radix sort on (N, 2) uint64 array.
    Sorts arr in place by (hi, lo) lexicographic order."""
    n = arr.shape[0]
    if n < 2:
        return
    a_hi = arr[:, 0].copy()
    a_lo = arr[:, 1].copy()
    b_hi = np.empty(n, dtype=np.uint64)
    b_lo = np.empty(n, dtype=np.uint64)
    count = np.empty(256, dtype=np.int64)

    for p in range(16):
        use_hi = p >= 8
        shift = np.uint64((p & 7) * 8)

        for i in range(256):
            count[i] = 0
        for i in range(n):
            v = a_hi[i] if use_hi else a_lo[i]
            count[(v >> shift) & np.uint64(0xFF)] += 1

        total = 0
        for i in range(256):
            c = count[i]
            count[i] = total
            total += c

        for i in range(n):
            v = a_hi[i] if use_hi else a_lo[i]
            b = (v >> shift) & np.uint64(0xFF)
            pos = count[b]
            b_hi[pos] = a_hi[i]
            b_lo[pos] = a_lo[i]
            count[b] = pos + 1

        a_hi, b_hi = b_hi, a_hi
        a_lo, b_lo = b_lo, a_lo

    for i in range(n):
        arr[i, 0] = a_hi[i]
        arr[i, 1] = a_lo[i]


def sort_128(arr: npt.NDArray[np.uint64]) -> None:
    """Sort (N, 2) uint64 array in place by (hi, lo) lexicographic order."""
    _radix_sort_128(arr)


@numba.njit(cache=True)
def argsort_128(arr: npt.NDArray[np.uint64]) -> npt.NDArray[np.int32]:
    """LSD byte-wise radix argsort on (N, 2) uint64 array.
    Returns indices that would sort arr by (hi, lo) lexicographic order."""
    n = arr.shape[0]
    idx = np.arange(n, dtype=np.int32)
    if n < 2:
        return idx
    buf = np.empty(n, dtype=np.int32)
    count = np.empty(256, dtype=np.int64)

    for p in range(16):
        use_hi = p >= 8
        shift = np.uint64((p & 7) * 8)

        for i in range(256):
            count[i] = 0
        for i in range(n):
            v = arr[idx[i], 0] if use_hi else arr[idx[i], 1]
            count[(v >> shift) & np.uint64(0xFF)] += 1

        total = 0
        for i in range(256):
            c = count[i]
            count[i] = total
            total += c

        for i in range(n):
            v = arr[idx[i], 0] if use_hi else arr[idx[i], 1]
            b = (v >> shift) & np.uint64(0xFF)
            pos = count[b]
            buf[pos] = idx[i]
            count[b] = pos + 1

        idx, buf = buf, idx

    return idx


@numba.njit(cache=True)
def binary_search_128(
    keys: npt.NDArray[np.uint64],
    val: npt.NDArray[np.uint64],
) -> int:
    """Binary search for val in sorted keys[0..n-1].
    keys is (2, N) where keys[0] is hi and keys[1] is lo.
    val is (2,) where val[0] is hi and val[1] is lo.
    Returns index, or -1 if not found."""
    val_hi = val[0]
    val_lo = val[1]
    lo = 0
    hi = keys.shape[1] - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        mid_hi = keys[0, mid]
        mid_lo = keys[1, mid]
        if mid_hi == val_hi and mid_lo == val_lo:
            return mid
        elif mid_hi < val_hi or (mid_hi == val_hi and mid_lo < val_lo):
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
