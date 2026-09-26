"""k-mer distributions and the total variation distance between them.

All distances here are total variation: half the L1 distance between two k-mer
frequency distributions, so they lie in [0, 1].  This is the distance DistBin,
VLP and DBV all use; the helpers were factored out of
`distbin.binner.Binner` so that a new algorithm does not have to import
DistBin just to featurise its input.
"""

from typing import Mapping, cast

import numpy as np
import numpy.typing as npt
import scipy

from distbin.kmers import encoded_seq_to_kmer_array, generate_kmer_map, seq_to_kmer_array


def kmer_distributions(
    sequences: Mapping[str, str] | Mapping[str, npt.NDArray[np.int8]], k: int
) -> dict[str, npt.NDArray[np.float64]]:
    """k-mer frequency distribution per label, for str or int8-encoded input."""
    sample = next(iter(sequences.values()), None)
    if isinstance(sample, np.ndarray):
        n_kmers = 4**k
        encoded = cast(Mapping[str, npt.NDArray[np.int8]], sequences)
        return {l: encoded_seq_to_kmer_array(s, k, n_kmers) for l, s in encoded.items()}
    kmer_map = generate_kmer_map(k)
    strs = cast(Mapping[str, str], sequences)
    return {l: seq_to_kmer_array(s, k, kmer_map) for l, s in strs.items()}


def stack_distributions(
    distributions: Mapping[str, npt.NDArray[np.float64]], labels: list[str]
) -> npt.NDArray[np.float64]:
    return np.stack([distributions[l] for l in labels])


def tv_distance_matrix(stack: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Square matrix of total variation distances between the rows of `stack`."""
    dists = np.astype(scipy.spatial.distance.pdist(stack, "cityblock") / 2, np.float64)
    return cast(npt.NDArray[np.float64], scipy.spatial.distance.squareform(dists))


def tv_cdist(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Total variation distances between every row of `a` and every row of `b`."""
    return np.astype(scipy.spatial.distance.cdist(a, b, "cityblock") / 2, np.float64)
