"""Mutual k-nearest-neighbour graph, fed to Leiden in `knn.binner`.

Neighbourhoods come from distance ranks rather than a global threshold, since
species differ in intrinsic k-mer spread.  Edges are unweighted; Jaccard (SNN)
weighting was dropped as it only helped on pairs whose species overlap.
"""

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from lib.distances import tv_cdist

# rows per pass in the blocked routines; bounds peak memory, not the result
BLOCK_SIZE = 512


def capped_n_neighbors(n_neighbors: int, n: int, divisor: int) -> int:
    """`n_neighbors`, cut to n // divisor (floor 5); a divisor <= 0 disables it.

    Keeps small inputs from getting a near-complete graph.
    """
    if divisor <= 0:
        return n_neighbors
    return min(n_neighbors, max(5, n // divisor))


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
    """`knn_indices` computed a row-block at a time, never building the n x n matrix."""
    n = stack.shape[0]
    k = min(n_neighbors, n - 1)
    if k < 1:
        return np.zeros((n, 0), dtype=np.int64)

    indices = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        indices[start:stop] = _top_k(tv_cdist(stack[start:stop], stack), start, k)
    return indices


def mutual_knn_edges(knn_idx: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Edges whose endpoints listed each other, as an (m, 2) array of i < j."""
    n, k = knn_idx.shape
    if n == 0 or k == 0:
        return np.zeros((0, 2), dtype=np.int64)

    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    ones = np.ones(n * k, dtype=np.int8)
    adjacency = sp.csr_matrix((ones, (rows, knn_idx.ravel())), shape=(n, n))

    edge_i, edge_j = sp.triu(adjacency.multiply(adjacency.T), k=1).nonzero()
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
