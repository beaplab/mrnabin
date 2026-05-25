import numpy as np

from distbin.kmers import (
    decode,
    generate_kmer_map,
    reverse_complement,
    seq_to_indices,
    seq_to_kmer_array,
)


def test_reverse_complement() -> None:
    assert reverse_complement("TGTCGCGTGTTTTGGATGA") == "TCATCCAAAACACGCGACA"


def test_generate_kmer_map() -> None:
    kmer_map = generate_kmer_map(4)
    assert len(set(kmer_map.values())) == 256


def test_seq_to_indices() -> None:
    seq = "TGTCGCGTGTTTTGGATGA"
    kmer_map = generate_kmer_map(4)
    forward = seq_to_indices(seq, 4, kmer_map)
    assert forward is not None
    # reverse = seq_to_indices(rc_seq, 4, kmer_map)
    # assert tuple(forward[:8]) == tuple(reverse[-8:][::-1])
    # assert tuple(forward[-8:][::-1]) == tuple(reverse[:8])


def test_seq_to_kmer_array() -> None:
    seqs = [
        "TGTCGCGTGTTTTGGATGA",
        "TGTCGCGTGTTTTGTCACA",
    ]
    k = 4
    kmer_map = generate_kmer_map(k)
    kmer_arrays = [seq_to_kmer_array(s, k, kmer_map) for s in seqs]
    assert len(kmer_arrays[0]) == len(kmer_map)
    assert len(kmer_arrays[1]) == len(kmer_map)
    assert np.abs(sum(kmer_arrays[0]) - 1.0) < 1e-6
    assert np.abs(sum(kmer_arrays[1]) - 1.0) < 1e-6
    assert set(kmer_arrays[0]) == set([np.float64(1 / 16.0), np.float64(0.0)])
    assert set(kmer_arrays[1]) == set([np.float64(1 / 8.0), np.float64(1 / 16.0), np.float64(0.0)])
    assert len(np.nonzero(kmer_arrays[0])[0]) == 16
    assert len(np.nonzero(kmer_arrays[1])[0]) == 15


def test_encode_decode() -> None:
    for k in [3, 4, 5, 6]:
        kmer_map = generate_kmer_map(k)
        s = "ACGTACGTTGCA"
        assert decode(seq_to_indices(s, k, kmer_map), k, kmer_map) == s
        s = "AAAAAAAAAAAC"
        assert decode(seq_to_indices(s, k, kmer_map), k, kmer_map) == s
