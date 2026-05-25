from itertools import product

import numpy as np
import numpy.typing as npt

NUCLEOBASES = ("A", "C", "G", "T")


def _flip(nucleotide: str) -> str:
    match nucleotide:
        case "A":
            return "T"
        case "T":
            return "A"
        case "C":
            return "G"
        case "G":
            return "C"
        case _:
            raise Exception("not a nucleotide")


def reverse_complement(seq: str) -> str:
    return "".join([_flip(n) for n in seq[::-1]])


def seq_to_indices(seq: str, k: int, index_map: dict[tuple[str, ...], int]) -> list[int]:
    # map a sequence's consecutive kmers to indices using a kmer_map
    indices = []
    for kmer in [seq[i : i + k] for i in range(len(seq) - k + 1)]:
        indices.append(index_map[tuple(list(kmer))])
    return indices


def seq_to_kmer_array(seq: str, k: int, kmer_map: dict[tuple[str, ...], int]) -> npt.NDArray[np.float64]:
    kmer_array = np.bincount(seq_to_indices(seq, k, kmer_map))
    if len(kmer_array) < len(kmer_map):
        kmer_array = np.pad(kmer_array, (0, len(kmer_map) - len(kmer_array)))
    return np.astype(kmer_array / np.sum(kmer_array), np.float64)


def encoded_seq_to_kmer_array(seq: npt.NDArray[np.int8], k: int, n_kmers: int) -> npt.NDArray[np.float64]:
    """Vectorized k-mer distribution for int8-encoded sequences (ACGT=0..3, N=4).

    Windows containing any value >= 4 are skipped. K-mer index ordering matches
    `generate_kmer_map` (A=0, C=1, G=2, T=3, big-endian).
    """
    n_windows = len(seq) - k + 1
    if n_windows <= 0:
        return np.zeros(n_kmers, dtype=np.float64)
    s = seq.astype(np.int64, copy=False)
    indices = np.zeros(n_windows, dtype=np.int64)
    valid = np.ones(n_windows, dtype=bool)
    for j in range(k):
        w = s[j : j + n_windows]
        indices = indices * 4 + w
        valid &= w < 4
    counts = np.bincount(indices[valid], minlength=n_kmers).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


def generate_kmer_map(k: int) -> dict[tuple[str, ...], int]:
    # maps kmers to indices ; 4^k indices
    return {kmer: i for i, kmer in enumerate(product(NUCLEOBASES, repeat=k))}


def decode(indices: list[int], k: int, index_map: dict[tuple[str, ...], int]) -> str:
    _reversed_kmap = {v: i for i, v in index_map.items()}
    s = "".join(_reversed_kmap[indices[0]])
    for i in indices[1:]:
        addition = _reversed_kmap[i]
        if addition[: k - 1] != tuple(s[-(k - 1) :]):
            raise Exception(
                "next kmer not a continuation of the previous: ",
                f"{s[-k:]} {addition} {tuple(s[-(k - 1) :])} {addition[:k-1]}",
            )
        s += addition[-1]
    return s
