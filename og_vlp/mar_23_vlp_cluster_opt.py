#!/usr/bin/env python3
"""
Optimized Python port of mar_23_vlp_cluster_fixed_size_gfortran.F90
Uses NumPy structured arrays for fast 128-bit integer sorting.
Produces identical printed output to the Fortran version.
"""

from dataclasses import dataclass, field
from typing import Any

import numba
import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[Any]

# ===========================================================================
# #define constants
# ===========================================================================
file_data = "sample_data/aa_bb_contigs.fasta"
file_in3 = ""
file_out = ""
file_sym = ""

MAX_LEN_DATA = 114000000
MAX_NUM_RECS = 117000
MAX_SEQ_LEN = 56000
MAX_NUM_STATES = 2000000
POLY_THRESH = 50
CLUSTER_MIN_SEQ_LEN = 3000
MIN_CLUSTER_SIZE = 6
XCHI_THRESH = 255.0
THRESH_NEXT = 0
XTHRESH_REMOVE = -3.0
KKIND = 16  # bytes => 128 bits
CKIND = 1
LineLen = 100
print_it = False

# Derived constants
max_al = 4
nb = 3
nbr = 6
nc = 1 + (KKIND * 8 - nbr) // nb  # = 41

# NumPy structured dtype for 128-bit integers (high, low uint64)
uint128_dt = np.dtype([("hi", np.uint64), ("lo", np.uint64)])

# ===========================================================================
# Module-level globals (vlp_stuff)
# ===========================================================================
ac = np.full(256, max_al, dtype=np.int32)  # ac(0:255)
al = 4
a_char = ["."] * 128  # a(0:127)
num_rec = 0
num_long = np.zeros(4, dtype=np.int32)  # num_long(0:3)
tot_train = 0
len_thresh = np.array([1, 1000, 2000, 3000], dtype=np.int32)  # len_thresh(0:3)

# Large arrays - allocated later
c: NDArray = None  # type: ignore[assignment]  # integer*1 array
cc: NDArray = None  # type: ignore[assignment]  # integer*1 array
vlp_ptr: NDArray = None  # type: ignore[assignment]  # integer*4 array (0:max_al, MAX_NUM_STATES)
vlp_model: NDArray = None  # type: ignore[assignment]  # real array (0:max_al, MAX_NUM_STATES)
vlp_state_hi: NDArray = None  # type: ignore[assignment]  # uint64 array (high 64 bits of 128-bit state)
vlp_state_lo: NDArray = None  # type: ignore[assignment]  # uint64 array (low 64 bits of 128-bit state)
ns = 0

# Score arrays - allocated after ncl is known
score: NDArray = None  # type: ignore[assignment]  # real(0:1, ncl, num_rec)
tmp_score: NDArray = None  # type: ignore[assignment]  # real(0:1, ncl)
mean: NDArray = None  # type: ignore[assignment]
var: NDArray = None  # type: ignore[assignment]


@dataclass
class Record:
    id: str = ""
    c_beg: int = 0
    c_end: int = 0
    c_len: int = 0
    cluster: int = 0
    direction: int = 0
    use_rec: int = 0
    cnts: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.float32))
    percent: float = 0.0


rec: list[Record] = []


# ===========================================================================
# (hi, lo) uint64 bit operations — all @numba.njit
# ===========================================================================
U64 = np.uint64


@numba.njit("UniTuple(uint64, 2)(uint64, uint64, int64)", cache=True)
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


@numba.njit("UniTuple(uint64, 2)(uint64, uint64, int64, int64)", cache=True)
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


@numba.njit("UniTuple(uint64, 2)(uint64, uint64, uint64, uint64)", cache=True)
def iand_hilo(h1: int, l1: int, h2: int, l2: int) -> tuple[np.uint64, np.uint64]:
    return U64(U64(h1) & U64(h2)), U64(U64(l1) & U64(l2))


@numba.njit("UniTuple(uint64, 2)(uint64, uint64, uint64, uint64)", cache=True)
def ior_hilo(h1: int, l1: int, h2: int, l2: int) -> tuple[np.uint64, np.uint64]:
    return U64(U64(h1) | U64(h2)), U64(U64(l1) | U64(l2))


@numba.njit("UniTuple(uint64, 2)(uint64, uint64, uint64, uint64)", cache=True)
def ieor_hilo(h1: int, l1: int, h2: int, l2: int) -> tuple[np.uint64, np.uint64]:
    return U64(U64(h1) ^ U64(h2)), U64(U64(l1) ^ U64(l2))


# ===========================================================================
# acgt_set
# ===========================================================================
def acgt_set() -> None:
    global al, a_char
    b = "ACGT"
    al = 4
    a_char = ["."] * 128
    ac[:] = al
    for i in range(1, al + 1):
        ch = b[i - 1]
        ac[ord(ch)] = i - 1
        a_char[i - 1] = ch


# ===========================================================================
# get_datam
# ===========================================================================
def get_datam() -> None:
    global num_rec

    num_rec = 0
    # rec[0] is index 1 in Fortran; we use 1-based indexing via rec[0] unused
    # Actually, let's use rec as 1-based: rec[1..N]
    # Pre-allocate with dummy at index 0
    rec.clear()
    rec.append(Record())  # rec[0] placeholder

    next_c_beg = 1  # 1-based position in c array

    with open(file_data, "r") as f:
        lines = f.readlines()

    line_idx = 0
    while line_idx < len(lines):
        # Read next line, look for '>'
        line = lines[line_idx].rstrip("\n\r")
        line_idx += 1

        # Fortran: read(7, *, IOSTAT=reason) idm  -- list-directed read gets first token
        tokens = line.split()
        if not tokens:
            continue
        idm = tokens[0]
        if not idm.startswith(">"):
            continue

        # We found a '>' header
        mlen = 0
        c_beg = next_c_beg

        while line_idx < len(lines):
            line = lines[line_idx]
            # Fortran reads exactly 100 chars as (100a1)
            # Check if this line starts a new record
            stripped = line.rstrip("\n\r")
            if stripped.startswith(">"):
                # backspace - don't advance line_idx
                break
            line_idx += 1
            # Process characters: Fortran reads 100 chars padded with spaces
            ccline = stripped
            for t in range(len(ccline)):
                ch = ccline[t]
                if ch == " ":
                    break
                mlen += 1
                c[c_beg + mlen] = ac[ord(ch)]  # 1-based: c_beg + mlen

        num_rec += 1
        # Ensure rec list is large enough
        while len(rec) <= num_rec + 1:
            rec.append(Record())

        rec[num_rec].id = idm[1:].strip()[:25]  # trim(idm(2:)), max 25 chars
        rec[num_rec].c_beg = c_beg
        rec[num_rec].c_end = c_beg + mlen - 1
        rec[num_rec].c_len = mlen
        rec[num_rec].cluster = 0
        rec[num_rec].direction = 0
        rec[num_rec].use_rec = 0
        next_c_beg = rec[num_rec].c_end + 1
        rec[num_rec + 1] = Record()
        rec[num_rec + 1].c_beg = next_c_beg

    # Fortran: print max seq len, total, num_rec
    max_clen = max(rec[n].c_len for n in range(1, num_rec + 1))
    total_end = rec[num_rec].c_end
    print(f"max_obs_seq_len:{max_clen:7d} total_num_obs{total_end:10d}  num recs{num_rec:9d}")

    if file_sym == "":
        return

    for n in range(1, num_rec + 1):
        rec[n].percent = 0.0

    with open(file_sym, "r") as f:
        for line in f:
            if len(line) < 25:
                continue
            idm = line[:25].strip()
            try:
                xper = float(line[25:33])
            except ValueError, IndexError:
                continue
            for n in range(1, num_rec + 1):
                if idm == rec[n].id:
                    rec[n].percent = xper
                    break

    for n in range(1, num_rec + 1):
        rec[n].use_rec = 1

    # print counts for clusters 0..9
    vals = []
    for i in range(10):
        cnt = sum(1 for n in range(1, num_rec + 1) if rec[n].use_rec > 0 and rec[n].cluster == i)
        vals.append(cnt)
    print("".join(f"{v:6d}" for v in vals))

    vals = []
    for i in range(10):
        cnt = sum(1 for n in range(1, num_rec + 1) if rec[n].use_rec > 0 and rec[n].cluster == i and rec[n].percent > 0)
        vals.append(cnt)
    print("".join(f"{v:6d}" for v in vals))

    for n in range(1, num_rec + 1):
        rec[n].use_rec = 0


# ===========================================================================
# count_tetragraphs
# ===========================================================================
@numba.njit("void(int8[:], int64, int64, int64, float32[:])", cache=True)
def _count_tetragraphs_inner(c_arr: NDArray, c_beg: int, c_end: int, al_val: int, cnts: NDArray) -> None:
    valid = 15
    tet = 0
    for i in range(256):
        cnts[i] = 0.0
    for t in range(c_beg, c_end + 1):
        cv = c_arr[t]
        tet = ((tet << 2) | cv) & 255
        valid = (valid << 1) & 15
        if cv == al_val:
            valid = valid | 1
        if valid == 0:
            cnts[tet] += 1.0


def count_tetragraphs() -> None:
    for n in range(1, num_rec + 1):
        _count_tetragraphs_inner(c, rec[n].c_beg, rec[n].c_end, al, rec[n].cnts)


# ===========================================================================
# ctcs - chi-squared contingency table test
# ===========================================================================
def ctcs(tab: NDArray, nr: int, ncc: int) -> tuple[int, float]:
    """
    tab: (nr, ncc) float array
    Returns (dof, xchi)
    """
    rsum = tab.sum(axis=1)
    csum = tab.sum(axis=0)
    tr = np.count_nonzero(rsum > 0.5)
    tc = np.count_nonzero(csum > 0.5)
    dof = (tr - 1) * (tc - 1)
    xtsum = csum.sum()
    xchi = 0.0
    for i in range(nr):
        if rsum[i] < 0.5:
            continue
        for j in range(ncc):
            if csum[j] < 0.5:
                continue
            xe = rsum[i] * csum[j] / xtsum
            xchi += (tab[i, j] - xe) ** 2 / max(0.01, xe)
    return int(dof), xchi


# ===========================================================================
# reverse_complement
# ===========================================================================
@numba.njit("void(int8[:], int64, int8[:])", cache=True)
def _reverse_complement_rev(c_in: NDArray, c_len: int, c_out: NDArray) -> None:
    for t in range(1, c_len + 1):
        v = 3 - c_in[t]
        if v < 0:
            v = 4
        c_out[c_len + 1 - t] = v


def reverse_complement(c_in: NDArray, c_len: int, rev: int) -> NDArray:
    """
    c_in: 1-based array slice of length c_len
    rev: 0 or 1
    Returns c_out as numpy array (1-based indexing, length c_len+1)
    """
    c_out = np.empty(c_len + 1, dtype=np.int8)
    if rev == 1:
        _reverse_complement_rev(c_in, c_len, c_out)
    else:
        c_out[1 : c_len + 1] = c_in[1 : c_len + 1]
    return c_out


# ===========================================================================
# np_sort_128 - sort 128-bit ints using NumPy structured array
# ===========================================================================
def np_sort_128(n: int, window_hi: NDArray, window_lo: NDArray) -> None:
    """
    Sort parallel arrays window_hi[0..n-1], window_lo[0..n-1] (0-based)
    using NumPy structured array sort on (hi, lo).
    """
    arr = np.empty(n, dtype=uint128_dt)
    arr["hi"] = window_hi[:n]
    arr["lo"] = window_lo[:n]
    arr.sort()
    window_hi[:n] = arr["hi"]
    window_lo[:n] = arr["lo"]


# ===========================================================================
# argsort (descending, 1-based)
# ===========================================================================
def argsort_desc(r: NDArray, n: int) -> list[int]:
    """
    Fortran argsort: merge sort, descending by r[d[i]].
    r: 1-based array, d: 1-based result permutation.
    Returns d as list (1-based).
    """
    d = list(range(n + 1))  # d[0] unused, d[1..n] = 1..n
    for i in range(1, n + 1):
        d[i] = i
    if n == 1:
        return d

    il = [0] * (n + 1)
    stepsize = 1
    while stepsize < n:
        left = 1
        while left <= n - stepsize:
            i = left
            j = left + stepsize
            ksize = min(stepsize * 2, n - left + 1)
            k = 1
            while i < left + stepsize and j < left + ksize:
                if r[d[i]] > r[d[j]]:
                    il[k] = d[i]
                    i += 1
                    k += 1
                else:
                    il[k] = d[j]
                    j += 1
                    k += 1
            if i < left + stepsize:
                for kk in range(k, ksize + 1):
                    il[kk] = d[i + kk - k]
            else:
                for kk in range(k, ksize + 1):
                    il[kk] = d[j + kk - k]
            for kk in range(1, ksize + 1):
                d[left + kk - 1] = il[kk]
            left += stepsize * 2
        stepsize *= 2
    return d


# ===========================================================================
# vlp_xscore
# ===========================================================================
@numba.njit("UniTuple(float64, 2)(int8[:], int64, float32[:,:], int32[:,:], int64)", cache=True)
def _vlp_xscore_jit(c_data: NDArray, cl: int, vlp_model: NDArray, vlp_ptr: NDArray, ns: int) -> tuple[float, float]:
    xscore = 0.0
    xvar = 0.0
    s = ns
    for t in range(1, cl + 1):
        cv = c_data[t]
        sc = vlp_model[cv, s]
        xscore += sc
        xvar += sc * sc
        s = vlp_ptr[cv, s]
    return xscore, xvar


def vlp_xscore(c_data: NDArray, cl: int) -> tuple[float, float]:
    """
    c_data: 1-based int8 array, indices 1..cl
    Returns (score, xvar)
    """
    return _vlp_xscore_jit(c_data, cl, vlp_model, vlp_ptr, ns)


# ===========================================================================
# JIT helper: build window inner loop
# ===========================================================================
@numba.njit("int64(int8[:], int64, uint64[:], uint64[:], int64, int64, int64)", cache=True)
def _build_window_inner(
    cc_arr: NDArray, c_len: int, window_hi: NDArray, window_lo: NDArray, offset: int, nb_val: int, nc_val: int
) -> int:
    """Build window entries for one sequence. Returns new offset."""
    pos_end_bits = nb_val * (nc_val - 1)
    w_hi = U64(0xFFFFFFFFFFFFFFFF)
    w_lo = U64(0xFFFFFFFFFFFFFFFF)
    for t in range(1, c_len + 1):
        # part1 = ishft(ibits(w, nb*2, nb*(nc-2)), nb)
        p1_hi, p1_lo = ibits_hilo(w_hi, w_lo, nb_val * 2, nb_val * (nc_val - 2))
        p1_hi, p1_lo = ishft_hilo(p1_hi, p1_lo, nb_val)
        # part2 = ishft(ibits(w, 0, nb), nb*(nc-1))
        p2_hi, p2_lo = ibits_hilo(w_hi, w_lo, 0, nb_val)
        p2_hi, p2_lo = ishft_hilo(p2_hi, p2_lo, pos_end_bits)
        # val = part1 | part2 | cc_arr[t]
        r_hi, r_lo = ior_hilo(p1_hi, p1_lo, p2_hi, p2_lo)
        r_hi, r_lo = ior_hilo(r_hi, r_lo, U64(0), U64(cc_arr[t]))
        window_hi[offset] = r_hi
        window_lo[offset] = r_lo
        w_hi = r_hi
        w_lo = r_lo
        offset += 1
    return offset


# ===========================================================================
# JIT: counting loop — the main hotspot in make_vlp_model
# ===========================================================================
_COUNT_STATES_SIG = (
    "int64(uint64[:], uint64[:], int64, uint64[:], uint64[:],"
    " float32[:,:], uint64[:], uint64[:], int64, int64, int64, int64, int64, int64)"
)


@numba.njit(_COUNT_STATES_SIG, cache=True)
def _count_states_jit(
    window_hi: NDArray,
    window_lo: NDArray,
    tot_train: int,
    vmask_hi: NDArray,
    vmask_lo: NDArray,
    vlp_model: NDArray,
    vlp_state_hi: NDArray,
    vlp_state_lo: NDArray,
    nc_val: int,
    nb_val: int,
    nbr_val: int,
    al_val: int,
    max_al_val: int,
    poly_thresh_val: int,
) -> int:
    """Count states from sorted window. Returns ns."""
    org_count = np.zeros((max_al_val + 1, nc_val + 2), dtype=np.float64)

    _, p_lo = ibits_hilo(window_hi[0], window_lo[0], 0, nb_val)
    p = numba.int64(p_lo)
    for lev in range(-1, nc_val + 1):
        org_count[p, lev + 1] += 1.0

    ns = 0
    pos_end = nb_val * (nc_val - 1)

    for t in range(1, tot_train + 1):
        xhi, xlo = ieor_hilo(window_hi[t], window_lo[t], window_hi[t - 1], window_lo[t - 1])
        k = nc_val
        for kk in range(1, nc_val):
            mhi, mlo = iand_hilo(xhi, xlo, vmask_hi[kk], vmask_lo[kk])
            if mhi != U64(0) or mlo != U64(0):
                k = kk
                break

        for lev in range(nc_val - 1, k - 1, -1):
            count_sum = 0.0
            for aa in range(al_val):
                count_sum += org_count[aa, lev + 1]
            if count_sum > poly_thresh_val:
                flag = 1
                wm_hi, wm_lo = iand_hilo(window_hi[t - 1], window_lo[t - 1], vmask_hi[lev], vmask_lo[lev])
                for pos in range(nb_val, pos_end + 1, nb_val):
                    _, bits_lo = ibits_hilo(wm_hi, wm_lo, pos, nb_val)
                    if numba.int64(bits_lo) >= al_val:
                        flag = 0
                if flag == 1:
                    ns += 1
                    for aa in range(al_val):
                        vlp_model[aa, ns] = np.float32(org_count[aa, lev + 1])
                    sh_hi, sh_lo = ishft_hilo(wm_hi, wm_lo, nbr_val - nb_val)
                    st_hi, st_lo = ior_hilo(sh_hi, sh_lo, U64(0), U64(lev))
                    vlp_state_hi[ns] = st_hi
                    vlp_state_lo[ns] = st_lo
            for aa in range(max_al_val + 1):
                org_count[aa, lev + 1] = 0.0

        if t == tot_train:
            break

        _, p_lo = ibits_hilo(window_hi[t], window_lo[t], 0, nb_val)
        p = numba.int64(p_lo)
        for lev in range(-1, nc_val + 1):
            org_count[p, lev + 1] += 1.0

    ns += 1
    vlp_state_hi[ns] = U64(0)
    vlp_state_lo[ns] = U64(0)
    for aa in range(al_val):
        vlp_model[aa, ns] = np.float32(org_count[aa, 1])

    return ns


# ===========================================================================
# JIT: flatten model + log weights
# ===========================================================================
@numba.njit("void(float32[:,:], uint64[:], int64, int64, int64, int64)", cache=True)
def _flatten_and_log_jit(
    vlp_model: NDArray, vlp_state_lo: NDArray, ns: int, al_val: int, nc_val: int, max_al_val: int
) -> None:
    """Flatten the VLP model and compute log weights. Modifies vlp_model in-place."""
    org_count = np.zeros((max_al_val + 1, nc_val + 2), dtype=np.float64)
    vlp_count = np.zeros((max_al_val + 1, nc_val + 1), dtype=np.float64)
    level_state = np.zeros(nc_val + 2, dtype=np.int32)
    flat = np.zeros((max_al_val + 1, nc_val + 2), dtype=np.float64)

    inv_al = 1.0 / al_val
    for aa in range(al_val):
        flat[aa, 0] = inv_al  # fl_idx(-1) = 0
    for s in range(1, ns + 1):
        vlp_model[al_val, s] = np.float32(inv_al)

    old_lev = -1

    for s in range(ns, -1, -1):
        new_lev = 0
        if s > 0:
            new_lev = numba.int32(vlp_state_lo[s] & U64(0x3F))

        for lev in range(old_lev, new_lev - 1, -1):
            ss = level_state[lev + 1]
            xsum = 0.0
            for aa in range(al_val):
                xsum += vlp_count[aa, lev + 1]
            if xsum < 1.0:
                xsum = 1.0
            xlam = 1.0 / xsum
            for aa in range(al_val):
                vlp_model[aa, ss] = np.float32((1.0 - xlam) * vlp_count[aa, lev + 1] / xsum + xlam * flat[aa, lev + 1])

        if s < 1:
            break

        old_lev = new_lev
        level_state[new_lev + 1] = s
        for aa in range(al_val):
            v = float(vlp_model[aa, s])
            org_count[aa, new_lev + 1] = v
            vlp_count[aa, new_lev + 1] = v

        org_sum = 0.0
        for aa in range(al_val):
            org_sum += org_count[aa, new_lev + 1]
        xlam = 1.0 / org_sum
        for aa in range(al_val):
            flat[aa, new_lev + 1] = (1.0 - xlam) * org_count[aa, new_lev + 1] / org_sum + xlam * flat[aa, new_lev]

        xsum_org = 0.0
        for aa in range(al_val):
            xsum_org += org_count[aa, new_lev + 1]
        for aa in range(al_val):
            flat[aa, new_lev + 1] = flat[aa, new_lev + 1] / xsum_org
        final_lev = new_lev - 1
        for aa in range(al_val):
            vlp_model[aa, s] = np.float32(flat[aa, final_lev + 1])

    # Make log weights
    log2 = np.log(2.0)
    for s in range(1, ns + 1):
        for aa in range(al_val + 1):
            v = float(vlp_model[aa, s])
            if v < 0.01:
                v = 0.01
            vlp_model[aa, s] = np.float32(np.log(al_val * v) / log2)


# ===========================================================================
# make_vlp_model
# ===========================================================================
def make_vlp_model(num_cluster: int) -> None:
    global ns, tot_train

    # vmask array: vmask_hi/lo[0..nc-1]
    vmask_hi = np.zeros(nc, dtype=np.uint64)
    vmask_lo = np.zeros(nc, dtype=np.uint64)
    for i in range(1, nc):
        bits_val_lo = np.uint64((1 << min(nb * i, 64)) - 1) if nb * i <= 64 else np.uint64(0xFFFFFFFFFFFFFFFF)
        bits_val_hi = np.uint64(0) if nb * i <= 64 else np.uint64((1 << (nb * i - 64)) - 1)
        vmask_hi[i], vmask_lo[i] = ishft_hilo(bits_val_hi, bits_val_lo, nb * (nc - i))

    vmskshft_hi = np.zeros(nc, dtype=np.uint64)
    vmskshft_lo = np.zeros(nc, dtype=np.uint64)
    for i in range(nc):
        vmskshft_hi[i], vmskshft_lo[i] = ishft_hilo(vmask_hi[i], vmask_lo[i], nbr - nb)

    # Count total training length for pre-allocation
    total_len = 0
    for n in range(1, num_rec + 1):
        if rec[n].cluster == num_cluster and rec[n].use_rec != 0:
            total_len += rec[n].c_len

    # Build window of data into pre-allocated arrays (0-based)
    window_hi = np.empty(total_len + 1, dtype=np.uint64)
    window_lo = np.empty(total_len + 1, dtype=np.uint64)
    tot_train = 0

    for n in range(1, num_rec + 1):
        if rec[n].cluster != num_cluster or rec[n].use_rec == 0:
            continue
        c_in_slice = np.empty(rec[n].c_len + 1, dtype=np.int8)
        c_in_slice[1 : rec[n].c_len + 1] = c[rec[n].c_beg : rec[n].c_beg + rec[n].c_len]
        cc_arr = reverse_complement(c_in_slice, rec[n].c_len, rec[n].direction)
        tot_train = _build_window_inner(cc_arr, rec[n].c_len, window_hi, window_lo, tot_train, nb, nc)

    # Sort
    np_sort_128(tot_train, window_hi, window_lo)
    # Append sentinel (all-ones = -1 unsigned) at index tot_train (0-based)
    window_hi[tot_train] = np.uint64(0xFFFFFFFFFFFFFFFF)
    window_lo[tot_train] = np.uint64(0xFFFFFFFFFFFFFFFF)

    # Count states (JIT)
    vlp_model[:, :] = 0.0
    ns = _count_states_jit(
        window_hi,
        window_lo,
        tot_train,
        vmask_hi,
        vmask_lo,
        vlp_model,
        vlp_state_hi,
        vlp_state_lo,
        nc,
        nb,
        nbr,
        al,
        max_al,
        POLY_THRESH,
    )

    # Flatten model + log weights (JIT)
    _flatten_and_log_jit(vlp_model, vlp_state_lo, ns, al, nc, max_al)

    # Connect model — use dict for O(1) lookup (fast, ~0.2s total)
    pos_end = nb * (nc - 1)
    vlp2_dict = {}
    for i in range(1, ns + 1):
        vlp2_dict[(int(vlp_state_hi[i]), int(vlp_state_lo[i]))] = i

    vlp_ptr[:, :] = 0
    vlp_ptr[al, :] = ns

    for s in range(1, ns + 1):
        nlev = int(vlp_state_lo[s]) & 0x3F
        st_hi, st_lo = iand_hilo(vlp_state_hi[s], vlp_state_lo[s], vmskshft_hi[nlev], vmskshft_lo[nlev])
        st_hi = U64(st_hi)
        st_lo = U64(st_lo)
        for k in range(al):
            ns_hi, ns_lo = ishft_hilo(st_hi, st_lo, -nb)
            ns_hi = U64(ns_hi)
            ns_lo = U64(ns_lo)
            k_hi, k_lo = ishft_hilo(U64(0), U64(k), pos_end + nbr - nb)
            k_hi = U64(k_hi)
            k_lo = U64(k_lo)
            ns_hi, ns_lo = ieor_hilo(ns_hi, ns_lo, k_hi, k_lo)
            ns_hi = U64(ns_hi)
            ns_lo = U64(ns_lo)
            for lev in range(min(nc - 1, nlev + 1), -1, -1):
                c_hi, c_lo = iand_hilo(ns_hi, ns_lo, vmskshft_hi[lev], vmskshft_lo[lev])
                c_hi = U64(c_hi)
                c_lo = U64(c_lo)
                c_hi, c_lo = ieor_hilo(c_hi, c_lo, U64(0), U64(lev))
                j = vlp2_dict.get((int(c_hi), int(c_lo)), 0)
                if j == 0:
                    continue
                vlp_ptr[k, s] = j
                break


# ===========================================================================
# make_clusters
# ===========================================================================
def make_clusters(minlen: int) -> int:

    rev_map = np.zeros(256, dtype=np.int32)
    for p in range(0, 7, 2):
        for i in range(256):
            rev_map[i] = (rev_map[i] << 2) | ((255 - i) >> p) & 3

    ntot = sum(1 for n in range(1, num_rec + 1) if rec[n].c_len >= minlen)
    print(f"\nMAKING: clusters based on tetragraph counts{minlen:9d}{ntot:9d}")

    ct = np.zeros((ntot + 1, ntot + 1), dtype=np.int8)  # 1-based
    cluster = np.zeros(ntot + 1, dtype=np.int32)
    cluster_member = np.zeros(ntot + 1, dtype=np.int32)

    # Map from 1..ntot to rec indices
    rec_map = [0]  # 1-based
    for n in range(1, num_rec + 1):
        if rec[n].c_len >= minlen:
            rec_map.append(n)

    # Pairwise chi-squared
    tab = np.zeros((256, 2), dtype=np.float32)
    n1 = 0
    for n in range(1, num_rec + 1):
        if rec[n].c_len < minlen:
            continue
        n1 += 1
        ct[n1, n1] = 1
        tab[:, 0] = rec[n].cnts[:]
        n2 = n1
        for nn in range(n + 1, num_rec + 1):
            if rec[nn].c_len < minlen:
                continue
            n2 += 1
            tab[:, 1] = rec[nn].cnts[:]
            _, xchi0 = ctcs(tab, 256, 2)
            tab[:, 1] = rec[nn].cnts[rev_map[:]]
            _, xchi1 = ctcs(tab, 256, 2)
            xchi = min(xchi0, xchi1)
            if xchi > XCHI_THRESH:
                continue
            ct[n1, n2] = 1
            ct[n2, n1] = 1

    # Chain together to make clusters
    for n in range(1, num_rec + 1):
        rec[n].cluster = 0

    num_cluster = 0
    while True:
        cluster[:] = 0
        for n in range(1, ntot + 1):
            cluster[n] = int(ct[:, n].sum())
        nmax = np.argmax(cluster[1 : ntot + 1]) + 1  # 1-based index
        cluster_vals = ct[:, nmax].copy()
        cluster_int = np.zeros(ntot + 1, dtype=np.int32)
        cluster_int[1 : ntot + 1] = cluster_vals[1 : ntot + 1]
        if cluster_int[1 : ntot + 1].sum() < 2:
            break
        ct[:, nmax] = 0
        while True:
            old_sum = int(cluster_int[1 : ntot + 1].sum())
            for n in range(1, ntot + 1):
                if cluster_int[n] == 1:
                    cluster_int[1 : ntot + 1] = np.maximum(
                        cluster_int[1 : ntot + 1], ct[1 : ntot + 1, n].astype(np.int32)
                    )
                    ct[:, n] = 0
            if old_sum == int(cluster_int[1 : ntot + 1].sum()):
                break

        csum = int(cluster_int[1 : ntot + 1].sum())
        if csum > 2:
            print(f"cluster_density {' ' * 18}{num_cluster:7d}{csum:7d}")
        if csum < MIN_CLUSTER_SIZE:
            continue
        num_cluster += 1
        cluster_member[:] = 0
        nmem = 0
        n1 = 0
        for n in range(1, num_rec + 1):
            if rec[n].c_len < minlen:
                continue
            n1 += 1
            if cluster_int[n1] == 1:
                nmem += 1
                cluster_member[nmem] = n
                rec[n].cluster = num_cluster
                rec[n].use_rec = 1

        n = cluster_member[1]
        rec[n].direction = 0
        tab[:, 0] = rec[n].cnts[:]

        while True:
            change = 0
            for nn in range(1, nmem + 1):
                n = cluster_member[nn]
                tab[:, 1] = rec[n].cnts[:]
                _, xchi0 = ctcs(tab, 256, 2)
                tab[:, 1] = rec[n].cnts[rev_map[:]]
                _, xchi1 = ctcs(tab, 256, 2)
                direc = 0
                if xchi1 < xchi0:
                    direc = 1
                if rec[n].direction != direc:
                    change += 1
                rec[n].direction = direc
            if change == 0:
                break
            if nmem >= MIN_CLUSTER_SIZE:
                # Fortran: print"(3i5,3x,100i1/(18x,100i1))"
                dirs = [rec[cluster_member[nn]].direction for nn in range(1, nmem + 1)]
                line = f"{num_cluster:5d}{nmem:5d}{change:5d}   "
                for idx, d in enumerate(dirs):
                    if idx > 0 and idx % 100 == 0:
                        line += "\n" + " " * 18
                    line += str(d)
                print(line)
            tab[:, :] = 0
            for nn in range(1, nmem + 1):
                n = cluster_member[nn]
                if rec[n].direction == 0:
                    tab[:, 0] += rec[n].cnts[:]
                if rec[n].direction == 1:
                    tab[:, 0] += rec[n].cnts[rev_map[:]]

        if csum >= MIN_CLUSTER_SIZE:
            make_vlp_model(num_cluster)
            max_lev = 0
            for ss in range(1, ns + 1):
                lv = int(vlp_state_lo[ss]) & 0x3F
                if lv > max_lev:
                    max_lev = lv
            cnt_cluster = sum(1 for nn in range(1, num_rec + 1) if rec[nn].cluster == num_cluster)
            print(
                f"cluster_density {' ' * 18}{num_cluster:7d}{csum:7d}{cnt_cluster:7d}{max_lev:7d}{ns:9d}{tot_train:9d}"
            )

    print(f"num_cluster{num_cluster:5d}")
    return num_cluster


# ===========================================================================
# compute_scores
# ===========================================================================
def compute_scores(ncl: int, iprint: int) -> None:

    if iprint == 1:
        print("\nBUILDING: VLP model")
    score[:, :, :] = -99999.0
    mean[:] = 0
    var[:] = 0

    for k in range(1, ncl + 1):
        mtot = 0
        num_cluster_k = sum(1 for n in range(1, num_rec + 1) if rec[n].cluster == k)
        if num_cluster_k < 3:
            for n in range(1, num_rec + 1):
                if rec[n].cluster == k:
                    rec[n].cluster = 0
            continue
        make_vlp_model(k)
        for n in range(1, num_rec + 1):
            if rec[n].use_rec == 0:
                continue
            # Forward score
            c_in = np.empty(rec[n].c_len + 1, dtype=np.int8)
            c_in[1 : rec[n].c_len + 1] = c[rec[n].c_beg : rec[n].c_beg + rec[n].c_len]
            xsc0, xvar0 = vlp_xscore(c_in, rec[n].c_len)
            score[0, k, n] = xsc0
            # Reverse complement score
            cc_arr = reverse_complement(c_in, rec[n].c_len, 1)
            xsc1, xvar1 = vlp_xscore(cc_arr, rec[n].c_len)
            score[1, k, n] = xsc1
            if rec[n].cluster == k:
                mtot += rec[n].c_len
                mean[k] += max(xsc0, xsc1)
                if xsc1 > xsc0:
                    rec[n].direction = 1
                    var[k] += xvar1
                else:
                    var[k] += xvar0
                    rec[n].direction = 0

        if mtot > 0:
            mean[k] = mean[k] / mtot
            var[k] = (var[k] / mtot) - mean[k] * mean[k]

        if iprint == 1:
            max_lev = 0
            for ss in range(1, ns + 1):
                lv = int(vlp_state_lo[ss]) & 0x3F
                if lv > max_lev:
                    max_lev = lv
            print(
                f"cluster{k:7d}{num_cluster_k:7d}{max_lev:7d}{ns:9d}{tot_train:9d}{mtot:9d}{mean[k]:9.5f}{var[k]:9.5f}"
            )

        cnt_k = sum(1 for n in range(1, num_rec + 1) if rec[n].cluster == k)
        cnt_k_sym = sum(1 for n in range(1, num_rec + 1) if rec[n].cluster == k and rec[n].percent > 0)
        max_lev = 0
        for ss in range(1, ns + 1):
            lv = int(vlp_state_lo[ss]) & 0x3F
            if lv > max_lev:
                max_lev = lv
        print(f"{k:2d}{cnt_k:10d}{cnt_k_sym:10d}{max_lev:10d}{ns:10d}{mtot:10d}{mean[k]:9.5f}{var[k]:9.5f}")

        if iprint == 0:
            continue

        minv = -10
        maxv = 10
        hist1 = np.zeros(maxv - minv + 1, dtype=np.int32)
        hist2 = np.zeros(maxv - minv + 1, dtype=np.int32)

        for n in range(1, num_rec + 1):
            if rec[n].use_rec == 0:
                continue
            max_score_kn = max(score[0, k, n], score[1, k, n])
            sig = np.sqrt(rec[n].c_len * var[k]) if var[k] > 0 else 1.0
            m_val = int((max_score_kn - rec[n].c_len * mean[k]) / sig)
            m_val = max(minv, min(maxv, m_val))
            idx = m_val - minv
            if k == rec[n].cluster:
                hist1[idx] += 1
            if k != rec[n].cluster:
                hist2[idx] += 1

        cnt_use = sum(1 for n in range(1, num_rec + 1) if rec[n].use_rec == 1)
        print(
            f"Histogram_train: num_msg{cnt_use:7d}  mean - var - sig"
            f"{mean[k]:10.5f}{var[k]:10.5f}{np.sqrt(var[k]):10.5f}"
        )
        print("".join(f"{i:6d}" for i in range(minv, maxv + 1)))
        print("".join(f"{hist1[i]:6d}" for i in range(maxv - minv + 1)) + f"{hist1.sum():6d}")
        print("".join(f"{hist2[i]:6d}" for i in range(maxv - minv + 1)) + f"{hist2.sum():6d}")

    for n in range(1, num_rec + 1):
        if rec[n].cluster > ncl:
            rec[n].cluster = 0


# ===========================================================================
# find_best_scores
# ===========================================================================
def find_best_scores(ncl: int, iprint: int) -> None:

    itr = 0
    ichange = 999
    cnt_use = sum(1 for n in range(1, num_rec + 1) if rec[n].use_rec > 0)
    print(f"FIND: best scores. num clus{ncl:3d}   recs in play{cnt_use:9d}")

    vals = []
    for i in range(ncl + 1):
        vals.append(sum(1 for n in range(1, num_rec + 1) if rec[n].cluster == i and rec[n].use_rec > 0))
    print("rec per clust" + "".join(f"{v:7d}" for v in vals))

    vals = []
    for i in range(ncl + 1):
        vals.append(
            sum(1 for n in range(1, num_rec + 1) if rec[n].use_rec > 0 and rec[n].cluster == i and rec[n].percent > 0)
        )
    print("sym per clust" + "".join(f"{v:7d}" for v in vals))

    f_counts = [0] * (ncl + 1)
    for i in range(1, ncl + 1):
        f_counts[i] = sum(1 for n in range(1, num_rec + 1) if rec[n].cluster == i)

    while ichange > 0:
        itr += 1
        compute_scores(ncl, 0)
        ichange = 0
        for n in range(1, num_rec + 1):
            if rec[n].use_rec == 0:
                continue
            kc = rec[n].cluster
            kd = rec[n].direction
            # maxl2 = maxloc(score(:,:,n)) -> shape (2, ncl) in Fortran (0:1, 1:ncl)
            # Find the (direction, cluster) with max score
            best_val = -1e30
            kcbest = 1
            kdbest = 0
            for kk in range(1, ncl + 1):
                for dd in range(2):
                    if score[dd, kk, n] > best_val:
                        best_val = score[dd, kk, n]
                        kcbest = kk
                        kdbest = dd
            rec[n].cluster = kcbest
            rec[n].direction = kdbest
            if kc != kcbest or kd != kdbest:
                ichange += 1

        # xsum = sum of max scores across all (direction, cluster) for each rec with use_rec > 0
        xsum = 0.0
        nz = 0
        for n in range(1, num_rec + 1):
            if rec[n].use_rec <= 0:
                continue
            max_val = -1e30
            for kk in range(1, ncl + 1):
                for dd in range(2):
                    if score[dd, kk, n] > max_val:
                        max_val = score[dd, kk, n]
            xsum += max_val
            if max_val < 0:
                nz += 1

        cnt_pos = sum(1 for i in range(1, ncl + 1) if f_counts[i] > 0)
        print(f"ITER:{itr:9d}{ichange:9d}{cnt_pos:9d}" + " " * 21 + f"SCORE:{xsum:15.2f}{nz:9d}{ncl:9d}")

    vals = []
    for i in range(ncl + 1):
        vals.append(sum(1 for n in range(1, num_rec + 1) if rec[n].cluster == i and rec[n].use_rec > 0))
    print("rec per clust" + "".join(f"{v:7d}" for v in vals))

    vals = []
    for i in range(ncl + 1):
        vals.append(
            sum(1 for n in range(1, num_rec + 1) if rec[n].use_rec > 0 and rec[n].cluster == i and rec[n].percent > 0)
        )
    print("sym per clust" + "".join(f"{v:7d}" for v in vals))


# ===========================================================================
# remove_next_cluster
# ===========================================================================
def remove_next_cluster(ncl: int) -> tuple[int, float]:

    compute_scores(ncl, 1)

    cluster_tot = np.zeros(ncl + 1, dtype=np.int32)  # 1-based
    cluster_score = np.zeros(ncl + 1, dtype=np.float64)

    for n in range(1, num_rec + 1):
        if rec[n].use_rec == 0:
            continue
        k = rec[n].cluster
        cluster_tot[k] += 1
        # tmp_score = score(:,:,n) with cluster k zeroed out
        best_other = -99999.0
        for kk in range(1, ncl + 1):
            if kk == k:
                continue
            for dd in range(2):
                if score[dd, kk, n] > best_other:
                    best_other = score[dd, kk, n]
        cluster_score[k] += best_other

    for k in range(1, ncl + 1):
        if cluster_tot[k] == 0:
            cluster_score[k] = -99999.0
        else:
            cluster_score[k] = cluster_score[k] / max(cluster_tot[k], 1)

    kbest = int(np.argmax(cluster_score[1 : ncl + 1]) + 1)
    xbest_score = float(cluster_score[kbest])

    cnt_use = sum(1 for n in range(1, num_rec + 1) if rec[n].use_rec > 0)
    print(f"\nREMOVE: next cluster{cnt_use:9d}{kbest:6d}{xbest_score:9.2f}")

    vals = []
    for i in range(ncl + 1):
        vals.append(sum(1 for n in range(1, num_rec + 1) if rec[n].cluster == i and rec[n].use_rec > 0))
    print("rec per clust" + "".join(f"{v:7d}" for v in vals))

    vals = []
    for i in range(ncl + 1):
        vals.append(
            sum(1 for n in range(1, num_rec + 1) if rec[n].use_rec > 0 and rec[n].cluster == i and rec[n].percent > 0)
        )
    print("sym per clust" + "".join(f"{v:7d}" for v in vals))

    remove_vals = [int(xbest_score)]
    for i in range(1, ncl + 1):
        remove_vals.append(int(cluster_score[i]))
    print("remove clust " + "".join(f"{v:7d}" for v in remove_vals))

    if xbest_score < THRESH_NEXT:
        return kbest, xbest_score

    for n in range(1, num_rec + 1):
        if rec[n].cluster == kbest:
            rec[n].cluster = 0

    vals = []
    for i in range(ncl + 1):
        vals.append(sum(1 for n in range(1, num_rec + 1) if rec[n].cluster == i and rec[n].use_rec > 0))
    print("rec per clust" + "".join(f"{v:7d}" for v in vals))

    vals = []
    for i in range(ncl + 1):
        vals.append(
            sum(1 for n in range(1, num_rec + 1) if rec[n].use_rec > 0 and rec[n].cluster == i and rec[n].percent > 0)
        )
    print("sym per clust" + "".join(f"{v:7d}" for v in vals))

    return kbest, xbest_score


# ===========================================================================
# main
# ===========================================================================
def main() -> None:
    global c, cc, vlp_ptr, vlp_model, vlp_state_hi, vlp_state_lo, rec
    global score, tmp_score, mean, var

    c = np.zeros(MAX_LEN_DATA + 1, dtype=np.int8)  # 1-based
    cc = np.zeros(MAX_SEQ_LEN + 1, dtype=np.int8)  # 1-based
    vlp_ptr = np.zeros((max_al + 1, MAX_NUM_STATES + 1), dtype=np.int32)  # (0:max_al, 1:MAX_NUM_STATES)
    vlp_model = np.zeros((max_al + 1, MAX_NUM_STATES + 1), dtype=np.float32)  # (0:max_al, 1:MAX_NUM_STATES)
    vlp_state_hi = np.zeros(MAX_NUM_STATES + 1, dtype=np.uint64)  # high 64 bits; 1-based
    vlp_state_lo = np.zeros(MAX_NUM_STATES + 1, dtype=np.uint64)  # low 64 bits; 1-based
    rec = [Record() for _ in range(MAX_NUM_RECS + 2)]  # 1-based with room

    acgt_set()
    get_datam()

    # print"(2i8)", (v, count(rec(1:num_rec)%c_len >= v), v = 0, 10000, 500)
    for v in range(0, 10001, 500):
        cnt = sum(1 for n in range(1, num_rec + 1) if rec[n].c_len >= v)
        print(f"{v:8d}{cnt:8d}")

    if file_in3 != "":
        with open(file_in3, "r") as f:
            for n in range(1, num_rec + 1):
                line = f.readline()
                # read format: (25x, 2i3) => skip 25 chars, read two 3-char integers
                rec[n].cluster = int(line[25:28])
                rec[n].direction = int(line[28:31])
        for n in range(1, num_rec + 1):
            rec[n].use_rec = 1
        ncl = 16
        score = np.zeros((2, ncl + 1, num_rec + 1), dtype=np.float32)
        tmp_score = np.zeros((2, ncl + 1), dtype=np.float32)
        mean = np.zeros(ncl + 1, dtype=np.float32)
        var = np.zeros(ncl + 1, dtype=np.float32)
        for n in range(1, num_rec + 1):
            rec[n].use_rec = 0
        for n in range(1, num_rec + 1):
            if rec[n].c_len > 1000:
                rec[n].use_rec = 1
        cnt_sym = sum(1 for n in range(1, num_rec + 1) if rec[n].percent > 0)
        print(f"num_rec - sym{num_rec:9d}{cnt_sym:9d}")
        vals = []
        for i in range(ncl + 1):
            vals.append(sum(1 for n in range(1, num_rec + 1) if rec[n].cluster == i and rec[n].use_rec > 0))
        print("rec per clust" + "".join(f"{v:7d}" for v in vals))
        vals = []
        for i in range(ncl + 1):
            vals.append(
                sum(
                    1
                    for n in range(1, num_rec + 1)
                    if rec[n].use_rec > 0 and rec[n].cluster == i and rec[n].percent > 0
                )
            )
        print("sym per clust" + "".join(f"{v:7d}" for v in vals))
        compute_scores(ncl, 1)
        cnt_sym = sum(1 for n in range(1, num_rec + 1) if rec[n].percent > 0)
        print(f"num_rec - sym{num_rec:9d}{cnt_sym:9d}")
        vals = []
        for i in range(ncl + 1):
            vals.append(sum(1 for n in range(1, num_rec + 1) if rec[n].cluster == i and rec[n].use_rec > 0))
        print("rec per clust" + "".join(f"{v:7d}" for v in vals))
        vals = []
        for i in range(ncl + 1):
            vals.append(
                sum(
                    1
                    for n in range(1, num_rec + 1)
                    if rec[n].use_rec > 0 and rec[n].cluster == i and rec[n].percent > 0
                )
            )
        print("sym per clust" + "".join(f"{v:7d}" for v in vals))
        print(f"remove msg if score <{XTHRESH_REMOVE:6.2f}")
        for n in range(1, num_rec + 1):
            k = rec[n].cluster
            if k == 0:
                continue
            xs = (score[rec[n].direction, k, n] - (rec[n].c_len * mean[k])) / np.sqrt(rec[n].c_len * var[k])
            if xs < XTHRESH_REMOVE:
                rec[n].cluster = 0
        cnt_sym = sum(1 for n in range(1, num_rec + 1) if rec[n].percent > 0)
        print(f"num_rec - sym{num_rec:9d}{cnt_sym:9d}")
        vals = []
        for i in range(ncl + 1):
            vals.append(sum(1 for n in range(1, num_rec + 1) if rec[n].cluster == i and rec[n].use_rec > 0))
        print("rec per clust" + "".join(f"{v:7d}" for v in vals))
        vals = []
        for i in range(ncl + 1):
            vals.append(
                sum(
                    1
                    for n in range(1, num_rec + 1)
                    if rec[n].use_rec > 0 and rec[n].cluster == i and rec[n].percent > 0
                )
            )
        print("sym per clust" + "".join(f"{v:7d}" for v in vals))
        find_best_scores(ncl, 1)
        cnt_sym = sum(1 for n in range(1, num_rec + 1) if rec[n].percent > 0)
        cnt_use = sum(1 for n in range(1, num_rec + 1) if rec[n].use_rec > 0)
        cnt_use_sym = sum(1 for n in range(1, num_rec + 1) if rec[n].use_rec > 0 and rec[n].percent > 0)
        print(f"num_rec - sym{num_rec:9d}{cnt_sym:9d}{cnt_use:9d}{cnt_use_sym:9d}")
        vals = []
        for i in range(ncl + 1):
            vals.append(sum(1 for n in range(1, num_rec + 1) if rec[n].cluster == i and rec[n].use_rec > 0))
        print("rec per clust" + "".join(f"{v:7d}" for v in vals))
        vals = []
        for i in range(ncl + 1):
            vals.append(
                sum(
                    1
                    for n in range(1, num_rec + 1)
                    if rec[n].use_rec > 0 and rec[n].cluster == i and rec[n].percent > 0
                )
            )
        print("sym per clust" + "".join(f"{v:7d}" for v in vals))
        return

    count_tetragraphs()
    ncl = make_clusters(CLUSTER_MIN_SEQ_LEN)
    print(f"num_cluster{ncl:6d}")
    score = np.zeros((2, ncl + 1, num_rec + 1), dtype=np.float32)
    tmp_score = np.zeros((2, ncl + 1), dtype=np.float32)
    mean = np.zeros(ncl + 1, dtype=np.float32)
    var = np.zeros(ncl + 1, dtype=np.float32)
    compute_scores(ncl, 1)
    find_best_scores(ncl, 1)

    for nl in range(3, -1, -1):
        for n in range(1, num_rec + 1):
            if rec[n].c_len >= len_thresh[nl]:
                rec[n].use_rec = 1
        print(f"\nMin_msg_len:{len_thresh[nl]:7d}")
        find_best_scores(ncl, 1)
        while True:
            kbest, xbest_score = remove_next_cluster(ncl)
            if xbest_score < THRESH_NEXT:
                break
            find_best_scores(ncl, 1)

    if file_out == "":
        return

    with open(file_out, "w") as f:
        for n in range(1, num_rec + 1):
            d = rec[n].direction
            k = rec[n].cluster
            s = score[d, k, n] if k > 0 else 0.0
            f.write(f"{rec[n].id:25s}{rec[n].cluster:3d}{rec[n].direction:3d}{s:11.2f}\n")


if __name__ == "__main__":
    main()
