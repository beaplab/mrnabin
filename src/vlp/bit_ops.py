"""128-bit integer operations using (hi, lo) uint64 pairs — all @numba.njit."""

import numba
import numpy as np

U64 = np.uint64
MAX_U64 = U64(0xFFFFFFFFFFFFFFFF)
U128 = tuple[np.uint64, np.uint64]


@numba.njit(cache=True)
def ishft_hilo(hi: np.uint64, lo: np.uint64, n: int) -> tuple[np.uint64, np.uint64]:
    """Logical shift on 128-bit (hi, lo). Positive n = left, negative n = right."""
    hi = U64(hi)
    lo = U64(lo)
    if n > 0:
        if n >= 128:
            return U64(0), U64(0)
        elif n >= 64:
            return U64(lo << U64(n - 64)), U64(0)
        else:
            return U64((hi << U64(n)) | (lo >> U64(64 - n))), U64(lo << U64(n))
    elif n < 0:
        nn = -n
        if nn >= 128:
            return U64(0), U64(0)
        elif nn >= 64:
            return U64(0), U64(hi >> U64(nn - 64))
        else:
            return U64(hi >> U64(nn)), U64((lo >> U64(nn)) | (hi << U64(64 - nn)))
    return hi, lo


@numba.njit(cache=True)
def ibits_hilo(hi: np.uint64, lo: np.uint64, pos: int, length: int) -> tuple[np.uint64, np.uint64]:
    """Extract 'length' bits starting at bit 'pos' from 128-bit (hi, lo). Returns (hi, lo)."""
    hi = U64(hi)
    lo = U64(lo)
    # Shift right by pos
    if pos >= 128:
        rhi = U64(0)
        rlo = U64(0)
    elif pos >= 64:
        rhi = U64(0)
        rlo = U64(hi >> U64(pos - 64))
    elif pos > 0:
        rhi = U64(hi >> U64(pos))
        rlo = U64((lo >> U64(pos)) | (hi << U64(64 - pos)))
    else:
        rhi = hi
        rlo = lo
    # Mask to length bits
    if length >= 128:
        return rhi, rlo
    elif length >= 64:
        mhi = U64((U64(1) << U64(length - 64)) - U64(1))
        return U64(rhi & mhi), rlo
    else:
        mlo = U64((U64(1) << U64(length)) - U64(1))
        return U64(0), U64(rlo & mlo)


@numba.njit(cache=True)
def iand_hilo(h1: np.uint64, l1: np.uint64, h2: np.uint64, l2: np.uint64) -> tuple[np.uint64, np.uint64]:
    return U64(U64(h1) & U64(h2)), U64(U64(l1) & U64(l2))


@numba.njit(cache=True)
def ior_hilo(h1: np.uint64, l1: np.uint64, h2: np.uint64, l2: np.uint64) -> tuple[np.uint64, np.uint64]:
    return U64(U64(h1) | U64(h2)), U64(U64(l1) | U64(l2))


@numba.njit(cache=True)
def ieor_hilo(h1: np.uint64, l1: np.uint64, h2: np.uint64, l2: np.uint64) -> tuple[np.uint64, np.uint64]:
    return U64(U64(h1) ^ U64(h2)), U64(U64(l1) ^ U64(l2))


@numba.njit(cache=True)
def make_128(hi: U64, lo: U64) -> U128:
    return (hi, lo)


@numba.njit(cache=True)
def max_128() -> U128:
    return (MAX_U64, MAX_U64)


@numba.njit(cache=True)
def equals_zero_128(u: U128) -> bool:
    return u[0] == U64(0) and u[1] == U64(0)


@numba.njit(cache=True)
def ishft_128(u: U128, n: int) -> U128:
    hi, lo = ishft_hilo(u[0], u[1], n)
    return (hi, lo)


@numba.njit(cache=True)
def ibits_128(u: U128, pos: int, length: int) -> U128:
    hi, lo = ibits_hilo(u[0], u[1], pos, length)
    return (hi, lo)


@numba.njit(cache=True)
def iand_128(a1: U128, a2: U128) -> U128:
    return iand_hilo(a1[0], a1[1], a2[0], a2[1])


@numba.njit(cache=True)
def ior_128(a1: U128, a2: U128) -> U128:
    return ior_hilo(a1[0], a1[1], a2[0], a2[1])


@numba.njit(cache=True)
def ieor_128(a1: U128, a2: U128) -> U128:
    return ieor_hilo(a1[0], a1[1], a2[0], a2[1])


def str_128(u: U128) -> str:
    return f"{u[0]:064b}{u[1]:064b}"
