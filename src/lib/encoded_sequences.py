"""Encoded nucleotide sequence representation and operations.

Sequences are stored as int8 arrays where A=0, C=1, G=2, T=3, unknown=4.
"""

from dataclasses import dataclass, field

import numba
import numpy as np
import numpy.typing as npt

_NUCLEOTIDE_CHARS = "ACGTN"
_NUCLEOTIDE_MAP: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}
_UNKNOWN_BASE = len(_NUCLEOTIDE_MAP)

_NUCLEOTIDE_LOOK_UP = np.full(256, _UNKNOWN_BASE, dtype=np.int8)
for _ch, _v in _NUCLEOTIDE_MAP.items():
    _NUCLEOTIDE_LOOK_UP[ord(_ch)] = _v


@dataclass
class Sequence:
    seq_id: str = ""
    seq: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    cluster: int = 0
    reverse: bool = False
    active: bool = False
    # tetragraph counts
    cnts: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.float32))

    def oriented(self) -> npt.NDArray[np.int8]:
        """if the sequence is the sense strand return; else return reverse complement"""
        if self.reverse:
            return reverse_complement(self.seq, True)
        return self.seq


def encode_nucleotide(ch: str) -> int:
    """Map a nucleotide character to 0-3, or _UNKNOWN_BASE (4) for anything else."""
    return _NUCLEOTIDE_MAP.get(ch, _UNKNOWN_BASE)


def encode_seq(seq: str) -> npt.NDArray[np.int8]:
    """Encode a nucleotide string to an int8 array (ACGT=0..3, else=4)."""
    return _NUCLEOTIDE_LOOK_UP[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def decode_seq(s: Sequence) -> str:
    return "".join(_NUCLEOTIDE_CHARS[b] for b in s.seq)


@numba.njit(cache=True)
def _count_tetragraphs_inner(seq: npt.NDArray[np.int8], cnts: npt.NDArray[np.float32]) -> None:
    """Count overlapping tetragraphs into cnts[256] using a 2-bit-per-base
    rolling index. Windows containing any unknown base (value 4) are skipped.
    EX: CGAC -> 0b01100001 = 97
    """

    valid = 15
    tet = 0
    # reset so callers may reuse Sequence.cnts across runs
    for i in range(256):
        cnts[i] = 0.0
    for t in range(len(seq)):
        cv = seq[t]
        tet = ((tet << 2) | cv) & 255
        valid = (valid << 1) & 15
        if cv == _UNKNOWN_BASE:
            valid = valid | 1
        if valid == 0:
            # no unknown bases in tetragraph
            cnts[tet] += 1.0


def count_tetragraphs(sequences: list[Sequence]) -> None:
    for s in sequences:
        _count_tetragraphs_inner(s.seq, s.cnts)


def add_cluster_info(
    sequences: list[Sequence], cluster: npt.NDArray[np.int8], cluster_num: int, minlen: int
) -> list[int]:
    # adds cluster info to sequences and return a reverse mapping
    cluster_members = []
    cluster_i = -1
    for i, s in enumerate(sequences):
        if len(s.seq) < minlen:
            continue
        cluster_i += 1
        if cluster[cluster_i] == 1:
            cluster_members.append(i)
            s.cluster = cluster_num
            s.active = True
    return cluster_members


@numba.njit(cache=True)
def _reverse_complement_rev(c_in: npt.NDArray[np.int8], c_out: npt.NDArray[np.int8]) -> None:
    c_len = len(c_in)
    for t in range(c_len):
        v = 3 - c_in[t]
        if v < 0:
            v = 4
        c_out[c_len - 1 - t] = v


def reverse_complement(c_in: npt.NDArray[np.int8], rev: bool) -> npt.NDArray[np.int8]:
    """
    EX: CGAC = 0b01100001 = 97 -> GTCG = 0b10110110 = 182
    """
    if not rev:
        return c_in
    c_out = np.empty(len(c_in), dtype=np.int8)
    _reverse_complement_rev(c_in, c_out)
    return c_out


def reverse_complement_map() -> npt.NDArray[np.uint8]:
    # maps tetragraph index to reverse complement index
    # EX 97 -> 182
    rev_map = np.zeros(256, dtype=np.uint8)
    for p in range(0, 7, 2):
        for i in range(256):
            rev_map[i] = (rev_map[i] << 2) | ((255 - i) >> p) & 3
    return rev_map


def read_mrna_file_encoded(path: str) -> list[Sequence]:
    sequences: list[Sequence] = []
    with open(path) as f:
        for line in f:
            label, sequence = line.strip().split()
            sequences.append(Sequence(seq_id=label, seq=encode_seq(sequence)))
    return sequences
