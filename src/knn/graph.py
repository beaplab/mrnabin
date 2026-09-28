"""Mutual k-nearest-neighbour graph.

A single global distance threshold cannot separate species, because different
species have different intrinsic k-mer spread: in one EukProt pair the
within-species total variation distances averaged 0.201 for one organism and
0.286 for the other, while the *cross*-species distances started at 0.125.  Any
constant that keeps the second organism together also merges it with the first.

So build the neighbourhood from distance *ranks* instead of distance values,
which adapts to whatever the local scale happens to be:

1. `knn_indices` takes each contig's `n_neighbors` nearest others.
2. `mutual_knn_edges` keeps an edge only if both endpoints chose each other,
   which discards most of the accidental proximity.

The result is fed to Leiden in `knn.binner`.

This used to reweight each edge by the Jaccard overlap of the two
neighbourhoods (the standard SNN step) and prune below a threshold.  Both were
measured and dropped: the threshold was inert across {0.02 .. 0.2} at every k
tried, and the weighting was worth nothing on separable pairs -- unit weights
matched it to within 0.001 purity at equal coverage.  It did buy 0.02-0.04
purity on pairs whose species overlap, so if those matter later it is worth
restoring; it cost a blocked sparse product that dominated runtime at large k.
Note that unit weights roughly double every internal edge sum, so CPM
resolutions are ~2x what they were with Jaccard weights.
"""

import math

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from lib.distances import tv_cdist

# rows per pass in the blocked routines; bounds peak memory, not the result
BLOCK_SIZE = 512


def auto_n_neighbors(n: int) -> int:
    """The O(log n) rule, floored to keep the graph from disintegrating.

    Note that log n comes from wanting the kNN graph to be *connected*, which
    is the opposite of what binning wants, and at n = 4000 it gives k = 9.
    `Binner` defaults to a
    fixed 20 instead; this is here to make the comparison easy to run.
    """
    return max(5, math.ceil(math.log(max(n, 2))))


def _top_k(block: npt.NDArray[np.float64], row_offset: int, k: int) -> npt.NDArray[np.int64]:
    """Nearest k columns for each row of a (rows, n) distance block, nearest first.

    `row_offset` is the index of the block's first row in the full matrix, so
    that each row's own column can be masked out.  Mutates `block`.
    """
    rows = np.arange(block.shape[0])
    block[rows, row_offset + rows] = np.inf
    nearest = np.argpartition(block, kth=k - 1, axis=1)[:, :k]
    r = rows[:, None]
    # argpartition leaves the k unordered; sort so ties resolve deterministically
    order = np.argsort(block[r, nearest], axis=1, kind="stable")
    return np.astype(nearest[r, order], np.int64)


def knn_indices(distances: npt.NDArray[np.float64], n_neighbors: int) -> npt.NDArray[np.int64]:
    """Indices of each row's `n_neighbors` nearest others, nearest first.

    Self is excluded, so a row of an n x n matrix yields at most n - 1 entries.
    Needs the whole distance matrix; use `knn_indices_from_stack` above a few
    thousand contigs.
    """
    n = distances.shape[0]
    k = min(n_neighbors, n - 1)
    if k < 1:
        return np.zeros((n, 0), dtype=np.int64)
    return _top_k(distances.astype(np.float64, copy=True), 0, k)


def knn_indices_from_stack(
    stack: npt.NDArray[np.float64], n_neighbors: int, block_size: int = BLOCK_SIZE
) -> npt.NDArray[np.int64]:
    """`knn_indices` without ever materialising the n x n distance matrix.

    A dense matrix over a whole mixture is out of reach -- 41k contigs would
    need 13.7 GB -- but only k columns per row are ever kept, so the distances
    are computed a row-block at a time and thrown away.
    """
    n = stack.shape[0]
    k = min(n_neighbors, n - 1)
    if k < 1:
        return np.zeros((n, 0), dtype=np.int64)

    indices = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        indices[start:stop] = _top_k(tv_cdist(stack[start:stop], stack), start, k)
    return indices


def mutual_knn_edges(knn_idx: npt.NDArray[np.int64], mutual: bool = True) -> npt.NDArray[np.int64]:
    """Undirected edge list from a kNN index array, as an (m, 2) array of i < j.

    With `mutual`, an edge survives only if both endpoints listed each other,
    which on these mixtures discards roughly 70% of the directed edges and most
    of the accidental proximity.  Otherwise the directed lists are symmetrised.
    """
    n, k = knn_idx.shape
    if n == 0 or k == 0:
        return np.zeros((0, 2), dtype=np.int64)

    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    ones = np.ones(n * k, dtype=np.int8)
    adjacency = sp.csr_matrix((ones, (rows, knn_idx.ravel())), shape=(n, n))

    kept = adjacency.multiply(adjacency.T) if mutual else adjacency.maximum(adjacency.T)
    edge_i, edge_j = sp.triu(kept, k=1).nonzero()
    return np.stack([edge_i, edge_j], axis=1).astype(np.int64)


def graph_stats(n: int, edges: npt.NDArray[np.int64]) -> str:
    if edges.shape[0] == 0:
        return f"nodes:{n} edges:0 isolated:{n}"
    degree = np.bincount(edges.ravel(), minlength=n)
    return (
        f"nodes:{n} "
        f"edges:{edges.shape[0]} "
        f"isolated:{int(np.sum(degree == 0))} "
        f"mean_degree:{np.mean(degree):.1f} "
        f"median_degree:{np.median(degree):.0f} "
    )


def merge_events(
    edges: npt.NDArray[np.int64], before: npt.NDArray[np.int64], after: npt.NDArray[np.int64], min_part: int
) -> list[tuple[int, npt.NDArray[np.int64], npt.NDArray[np.int64], float]]:
    """Pairs of earlier communities that a later partition put together.

    Each community of `after` is split into the communities of `before` its
    nodes came from; every pair of those parts with at least `min_part` nodes
    each is returned as (community of `after`, nodes of part a, nodes of part b,
    rel), where

        rel = density(a, b) / sqrt(density(a) * density(b))

    is the edge density between the parts relative to their own internal
    densities.  CPM merges two parts once density(a, b) clears the resolution,
    which says nothing about how they compare with the parts' own cohesion.
    """
    # a part is a (community after, community before) pair
    key = after.astype(np.int64) * (int(before.max()) + 1) + before
    keys, part_of, sizes = np.unique(key, return_inverse=True, return_counts=True)
    community_of = keys // (int(before.max()) + 1)
    big = np.flatnonzero(sizes >= min_part)
    if big.size < 2:
        return []

    inside = after[edges[:, 0]] == after[edges[:, 1]]
    pu, pv = part_of[edges[inside, 0]], part_of[edges[inside, 1]]
    pair_keys, pair_counts = np.unique(np.minimum(pu, pv) * len(keys) + np.maximum(pu, pv), return_counts=True)
    counts = dict(zip(pair_keys.tolist(), pair_counts.tolist()))

    def density(a: int, b: int) -> float:
        possible = sizes[a] * (sizes[a] - 1) / 2 if a == b else sizes[a] * sizes[b]
        return counts.get(min(a, b) * len(keys) + max(a, b), 0) / possible

    events = []
    for community in np.unique(community_of[big]):
        parts = big[community_of[big] == community]
        for i, a in enumerate(parts):
            for b in parts[i + 1 :]:
                internal = density(a, a) * density(b, b)
                rel = density(a, b) / math.sqrt(internal) if internal > 0 else 0.0
                events.append((int(community), np.flatnonzero(part_of == a), np.flatnonzero(part_of == b), float(rel)))
    return events
