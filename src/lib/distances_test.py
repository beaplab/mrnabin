import numpy as np

from lib.distances import kmer_distributions, stack_distributions, tv_cdist, tv_distance_matrix
from lib.encoded_sequences import encode_seq

SEQUENCES = {
    "a": "ACGTACGTACGTACGTACGT",
    "b": "ACGTACGTACGTACGTACGT",
    "c": "AAAACCCCGGGGTTTTAAAA",
}


def test_kmer_distributions_sum_to_one() -> None:
    dists = kmer_distributions(SEQUENCES, 4)
    assert set(dists) == set(SEQUENCES)
    for d in dists.values():
        assert len(d) == 256
        assert np.isclose(d.sum(), 1.0)


def test_kmer_distributions_match_for_encoded_input() -> None:
    from_str = kmer_distributions(SEQUENCES, 4)
    encoded = {l: encode_seq(s) for l, s in SEQUENCES.items()}
    from_encoded = kmer_distributions(encoded, 4)
    for label in SEQUENCES:
        assert np.allclose(from_str[label], from_encoded[label])


def test_tv_distance_matrix_is_a_metric_in_zero_one() -> None:
    matrix = tv_distance_matrix(stack_distributions(kmer_distributions(SEQUENCES, 4), ["a", "b", "c"]))
    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.allclose(matrix, matrix.T)
    assert np.all((matrix >= 0) & (matrix <= 1))
    # identical sequences are at distance 0, different ones are not
    assert matrix[0, 1] == 0.0
    assert matrix[0, 2] > 0.0


def test_tv_cdist_agrees_with_the_full_matrix() -> None:
    dists = kmer_distributions(SEQUENCES, 4)
    stack = stack_distributions(dists, ["a", "b", "c"])
    assert np.allclose(tv_cdist(stack, stack), tv_distance_matrix(stack))
