"""
VLP model building, scoring, and JIT helpers.

TODO:

* remove level information from vlp state (why shift to make room?!)

* do something about wasted five left bits in the context

* first row of vlp states and model are unused
"""

import numba
import numpy as np
import numpy.typing as npt

from lib.encoded_sequences import Sequence

from .bit_ops import (
    MAX_U64,
    U64,
    U128,
    equals_zero_128,
    iand_128,
    ibits_128,
    ieor_128,
    ior_128,
    ishft_128,
    make_128,
    max_128,
)
from .sorting import argsort_128, binary_search_128, sort_128

ALPHA_SIZE = 4
BASE_BITS = 3  # we use 3 bits to encode A, C, G, T, and N (unknown)
LEVEL_BITS = 6  # number of reserved bits at the bottom of the 128-bit key storing the trie level
MAX_DEPTH = 1 + (128 - LEVEL_BITS) // BASE_BITS  # = 41 max number of bases that fit in 128 bits


@numba.njit(cache=True)
def _build_training_set_inner(
    sequence: npt.NDArray[np.int8],
    training: npt.NDArray[np.uint64],
    offset: int,
) -> int:
    """Build training set for given sequence. Returns new offset.

    Maintains a sliding 128-bit window. Layout:
      slot 0 (LSB): the current base (the one being predicted)
      slot 1: the oldest base (MAX_DEPTH - 2 steps ago)
      slot 2: the second oldest base
      ...
      slot MAX_DEPTH - 2: 2 steps ago
      slot MAX_DEPTH - 1 (MSB): the most recent base (1 step ago)

    Each iteration shifts the window down by one slot (dropping the
    oldest at slot 1), moves the previous predicted base (slot 0) to
    the top (slot MAX_DEPTH - 1) and inserts the newest base at slot 0.

    i.e., abcdefghijklmnopqrstuvwxy as xwvutsrqponmlkjihgfedcba.y
          bcdefghijklmnopqrstuvwxyz as yxwvutsrqponmlkjihgfedcb.z

    00000sss...sssbbb (5 unused zero bits, 40 slots, base in slot 0)
    """
    prev: U128 = max_128()  # start with an "empty" context
    for t in range(len(sequence)):
        p1 = ibits_128(prev, BASE_BITS * 2, BASE_BITS * (MAX_DEPTH - 2))  # extract last 39 bases
        p1 = ishft_128(p1, BASE_BITS)  # shift them to slots 1...39
        p2 = ibits_128(prev, 0, BASE_BITS)  # extract first base from slot 0
        p2 = ishft_128(p2, BASE_BITS * (MAX_DEPTH - 1))  # shift it to slot 40
        r = ior_128(p1, p2)
        r = ior_128(r, make_128(0, sequence[t]))  # add next base to slot 0
        training[offset, 0], training[offset, 1] = r[0], r[1]  # add 41 base window to training set
        prev = r
        offset += 1
    return offset


def _build_training_set(sequences: list[Sequence], num_cluster: int) -> npt.NDArray[np.uint64]:
    """Build set of encoded sequence data for VLP model training."""
    total_len = sum(len(s.seq) for s in sequences if s.cluster == num_cluster and s.active)
    training = np.empty((total_len, 2), dtype=np.uint64)
    offset = 0
    for s in sequences:
        if s.cluster != num_cluster or not s.active:
            continue
        offset = _build_training_set_inner(s.oriented(), training, offset)
    return training


@numba.njit(cache=True)
def _build_vmask() -> npt.NDArray[np.uint64]:
    """Build vmask array for the VLP trie.
    vmask[i] masks the top (i+1) slots of the 128-bit context window.
    """
    num_levels = MAX_DEPTH - 1
    vmask = np.zeros((num_levels, 2), dtype=np.uint64)
    for i in range(num_levels):
        n = i + 1  # number of slots to mask
        bits_val_lo = U64((1 << min(BASE_BITS * n, 64)) - 1) if BASE_BITS * n <= 64 else MAX_U64
        bits_val_hi = U64(0) if BASE_BITS * n <= 64 else U64((1 << (BASE_BITS * n - 64)) - 1)
        vmask[i] = ishft_128(make_128(bits_val_hi, bits_val_lo), BASE_BITS * (MAX_DEPTH - n))
    return vmask


@numba.njit(cache=True)
def _create_and_count_states_jit(
    training: npt.NDArray[np.uint64],
    count_thresh: int,
    max_states: int,
) -> tuple[int, npt.NDArray[np.uint64], npt.NDArray[np.uint8], npt.NDArray[np.uint64]]:
    """Count states from sorted training set. Main hotspot.

    Creates states based on shifted training data
    00sss...sssllllll (2 unused zero bits, 40 slots, level encdoded in bottom 6 bits
    """
    states = np.zeros((max_states + 1, 2), dtype=np.uint64)
    # level 0 holds global counts
    base_counts = np.zeros((ALPHA_SIZE + 1, MAX_DEPTH), dtype=np.uint64)
    model_counts = np.zeros((ALPHA_SIZE, max_states + 1), dtype=np.uint64)
    model_levels = np.zeros(max_states + 1, dtype=np.uint8)

    num_states = 0
    full = False
    vmask = _build_vmask()  # vmask[i] masks i + 1 slots
    for idx in range(training.shape[0]):
        # Accumulate predicted base from entry into counts at all levels
        base128 = ibits_128(training[idx], 0, BASE_BITS)
        base = numba.int64(base128[1])
        for lev in range(MAX_DEPTH):
            base_counts[base, lev] += 1

        if idx == training.shape[0] - 1:
            # at the end so no next entry to xor with
            break

        if full:
            continue

        # xor with next entry to find level where contexts diverge.
        # If k == MAX_DEPTH, then entries only differ in the predicted base at slot 0
        xored = ieor_128(training[idx], training[idx + 1])
        k = MAX_DEPTH
        for level in range(1, MAX_DEPTH):
            masked = iand_128(xored, vmask[level - 1])
            if not equals_zero_128(masked):
                k = level
                break

        for lev in range(MAX_DEPTH - 1, k - 1, -1):
            count_sum = np.sum(base_counts[:ALPHA_SIZE, lev])
            if count_sum > count_thresh:
                valid = True
                masked_window = iand_128(training[idx], vmask[lev - 1])
                # check for EOS
                for pos in range(BASE_BITS, BASE_BITS * (MAX_DEPTH - 1) + 1, BASE_BITS):
                    _, bits = ibits_128(masked_window, pos, BASE_BITS)
                    if numba.int64(bits) >= ALPHA_SIZE:
                        valid = False
                if valid:
                    num_states += 1
                    if num_states >= max_states:
                        num_states -= 1
                        full = True
                        break
                    for aa in range(ALPHA_SIZE):
                        model_counts[aa, num_states] = base_counts[aa, lev]
                    # writing over slot 0 so only need to shift by L-B
                    shifted = ishft_128(masked_window, LEVEL_BITS - BASE_BITS)
                    state = ior_128(shifted, make_128(0, lev))
                    model_levels[num_states] = lev
                    states[num_states] = state
            for aa in range(ALPHA_SIZE + 1):
                base_counts[aa, lev] = 0

    # root state — always added, needed for global base probabilities
    num_states += 1
    states[num_states] = make_128(0, 0)
    for aa in range(ALPHA_SIZE):
        model_counts[aa, num_states] = base_counts[aa, 0]

    return num_states, model_counts, model_levels, states


@numba.njit(cache=True)
def _flatten_jit(
    model_counts: npt.NDArray[np.uint64], model_levels: npt.NDArray[np.uint8], ns: int
) -> npt.NDArray[np.float32]:
    """Convert counts to probabilities"""
    vlp_model = np.zeros((ALPHA_SIZE + 1, model_levels.shape[0]), dtype=np.float32)

    # work space variables
    level_state = np.zeros(MAX_DEPTH, dtype=np.int32)  # 0-indexed
    flat = np.zeros((ALPHA_SIZE, MAX_DEPTH + 1), dtype=np.float64)  # 1-indexed (0 holds uniform)

    alpha_inv = 1.0 / ALPHA_SIZE
    for aa in range(ALPHA_SIZE):
        flat[aa, 0] = alpha_inv
    for s in range(1, ns + 1):
        vlp_model[ALPHA_SIZE, s] = np.float32(alpha_inv)

    old_lev = -1
    for s in range(ns, -1, -1):
        new_lev = 0
        if s > 0:
            new_lev = numba.int32(model_levels[s])

        for lev in range(old_lev + 1, new_lev, -1):
            # walk up trie states from deepest to shallowest
            ss = level_state[lev - 1]
            for aa in range(ALPHA_SIZE):
                # write prediction probabilities
                vlp_model[aa, ss] = flat[aa, lev]

        if s < 1:
            break

        old_lev = new_lev
        level_state[new_lev] = s
        sum_inv = 1.0 / float(np.sum(model_counts[:, s]))
        for aa in range(ALPHA_SIZE):
            flat[aa, new_lev + 1] = (1.0 - sum_inv) * float(model_counts[aa, s]) * sum_inv + sum_inv * flat[aa, new_lev]

    return vlp_model


@numba.njit(cache=True)
def _log_weights_jit(vlp_model: npt.NDArray[np.float32], ns: int) -> None:
    # log weights in-place
    log2 = np.log(2.0)
    for s in range(1, ns + 1):
        for aa in range(ALPHA_SIZE + 1):
            v = max(1.0 / 64, vlp_model[aa, s])
            vlp_model[aa, s] = np.float32(np.log(ALPHA_SIZE * v) / log2)


@numba.njit(cache=True)
def _build_vmskshft() -> npt.NDArray[np.uint64]:
    """Build vmskshft array for the VLP trie.
    vmskshft[i] is vmask[i] shifted left by (LEVEL_BITS - BASE_BITS) to align with the state key layout.
    """
    vmask = _build_vmask()
    num_levels = MAX_DEPTH - 1
    vmskshft = np.zeros((num_levels, 2), dtype=np.uint64)
    for i in range(num_levels):
        vmskshft[i] = ishft_128(vmask[i], LEVEL_BITS - BASE_BITS)
    return vmskshft


@numba.njit(cache=True)
def _connect_model_jit(
    vlp_states: npt.NDArray[np.uint64],
    model_levels: npt.NDArray[np.uint8],
    ns: int,
) -> npt.NDArray[np.int32]:
    """JIT: Connect VLP model states using sorted array + binary search."""
    vlp_ptr = np.zeros((ALPHA_SIZE + 1, vlp_states.shape[0]), dtype=np.int32)
    vlp_ptr[ALPHA_SIZE, :] = ns
    vmskshft = _build_vmskshft()

    # Sort states and build index mapping
    idx = argsort_128(vlp_states[1 : ns + 1])
    sorted_keys = np.empty((2, ns), dtype=np.uint64)
    sorted_keys[0] = vlp_states[idx + 1, 0]
    sorted_keys[1] = vlp_states[idx + 1, 1]
    sorted_idx = idx + 1

    pos_end = BASE_BITS * (MAX_DEPTH - 1)

    for s in range(1, ns + 1):
        nlev = model_levels[s]
        st = iand_128(vlp_states[s], vmskshft[nlev - 1])
        for k in range(ALPHA_SIZE):
            next_state = ishft_128(st, -BASE_BITS)
            shifted_k = ishft_128(make_128(0, k), pos_end + LEVEL_BITS - BASE_BITS)
            next_state = ieor_128(next_state, shifted_k)
            for lev in range(min(MAX_DEPTH - 1, nlev + 1), 0, -1):
                c = iand_128(next_state, vmskshft[lev - 1])
                c = ieor_128(c, make_128(0, lev))
                j = binary_search_128(sorted_keys, c)
                if j < 0:
                    continue
                vlp_ptr[k, s] = sorted_idx[j]
                break
    return vlp_ptr


@numba.njit(cache=True)
def _score_jit(
    sequence: npt.NDArray[np.int8], model: npt.NDArray[np.float32], trans_ptr: npt.NDArray[np.int32], last_state: int
) -> tuple[float, float]:
    """
    Walks the trie one nucleotide at a time, accumulating the log-likelihood score.

    Starting at the root state (s = num_states), for each base at the
    current state (model[base, s]), add it to score, then follow the trie transition
    to the next state (trans_ptr[base, s])

    The trie transitions mean longer matching contexts get more
    specific (higher or lower) weights.
    """
    score = 0.0
    var = 0.0
    state = last_state
    for t in range(len(sequence)):
        nucleobase = sequence[t]
        s = model[nucleobase, state]
        score += s
        var += s * s
        state = trans_ptr[nucleobase, state]
    return score, var


class VLPTrie:
    """
    Variable Length Pattern Trie
    The 5th row (index 4) stores unknown/ambiguous bases.
    In connect_model, trans_ptr[4, s] is set to always transition back to root
    model[4, s] gets set during _flatten_jit
    """

    def __init__(self) -> None:
        # VLP trie transition pointers: trans_ptr[alphabet_code, state] -> next state
        self.trans_ptr: npt.NDArray[np.int32] | None = None
        # VLP model log-likelihood weights: model[alphabet_code, state] -> log score
        self.model: npt.NDArray[np.float32] | None = None
        # Number of VLP states in the current model
        self.num_states: int = 0
        self.max_level: int = 0

    def state_string(self) -> str:
        return f"max_lvl:{self.max_level:7d} num_states:{self.num_states:9d}"

    def make_model(
        self,
        num_cluster: int,
        sequences: list[Sequence],
        verbose: bool = False,
        count_thresh: int = 50,
        max_states: int = 2000000,
    ) -> None:
        training = _build_training_set(sequences, num_cluster)
        sort_128(training)
        if verbose:
            print(f"    training set size:{training.shape[0]:9d}")

        self.num_states, model_counts, model_levels, states = _create_and_count_states_jit(
            training, count_thresh, max_states
        )
        self.max_level = np.max(model_levels)

        # convert counts to probabilities, then to log probabilities
        self.model = _flatten_jit(model_counts, model_levels, self.num_states)
        _log_weights_jit(self.model, self.num_states)

        # wire up trie transitions
        self.trans_ptr = _connect_model_jit(states, model_levels, self.num_states)

    def score(self, sequence: npt.NDArray[np.int8]) -> tuple[float, float]:
        """Score a sequence against the current VLP model.

        Returns (score, var) where score is the total log-likelihood
        and xvar is the sum of squared per-base scores.
        """
        return _score_jit(sequence, self.model, self.trans_ptr, self.num_states)
