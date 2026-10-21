import numpy as np

from knn.graph import (
    capped_n_neighbors,
    graph_stats,
    knn_indices,
    knn_indices_from_stack,
    mutual_knn_edges,
)
from lib.distances import tv_distance_matrix


def _two_cluster_stack() -> np.ndarray:
    """Two tight groups of 20 distributions, far apart on the simplex."""
    rng = np.random.default_rng(1)
    dim = 32
    a = np.full(dim, 1.0 / dim)
    b = np.zeros(dim)
    b[: dim // 2] = 2.0 / dim
    points = []
    for center in (a, b):
        for _ in range(20):
            p = np.abs(center + rng.normal(0, 0.002, dim))
            points.append(p / p.sum())
    return np.stack(points)


def test_knn_indices_excludes_self_and_orders_by_distance() -> None:
    distances = np.array(
        [
            [0.0, 0.1, 0.3, 0.2],
            [0.1, 0.0, 0.4, 0.5],
            [0.3, 0.4, 0.0, 0.6],
            [0.2, 0.5, 0.6, 0.0],
        ]
    )
    idx = knn_indices(distances, 2)
    assert idx.shape == (4, 2)
    assert idx[0].tolist() == [1, 3]
    assert idx[2].tolist() == [0, 1]
    # k is capped at the n - 1 others
    assert knn_indices(distances, 10).shape == (4, 3)


def test_blocked_knn_matches_the_full_matrix() -> None:
    stack = _two_cluster_stack()
    expected = knn_indices(tv_distance_matrix(stack), 6)
    for block_size in (1, 7, 40, 1000):
        assert np.array_equal(knn_indices_from_stack(stack, 6, block_size), expected)


def test_mutual_knn_edges() -> None:
    idx = knn_indices_from_stack(_two_cluster_stack(), 8)
    edges = mutual_knn_edges(idx)
    # each undirected pair once, as i < j
    assert np.all(edges[:, 0] < edges[:, 1])
    assert len({tuple(e) for e in edges.tolist()}) == edges.shape[0]
    # mutual: each endpoint lists the other
    lists = [set(row) for row in idx.tolist()]
    assert all(j in lists[i] and i in lists[j] for i, j in edges.tolist())
    # and no edge crosses between the two groups
    assert not [(i, j) for i, j in edges.tolist() if (i < 20) != (j < 20)]
    assert mutual_knn_edges(np.zeros((0, 0), dtype=np.int64)).shape == (0, 2)


def test_capped_n_neighbors() -> None:
    assert capped_n_neighbors(1024, 100_000, 8) == 1024
    assert capped_n_neighbors(1024, 1467, 8) == 183
    assert capped_n_neighbors(1024, 10, 8) == 5
    assert capped_n_neighbors(3, 10, 8) == 3
    assert capped_n_neighbors(1024, 1467, 0) == 1024


def test_graph_stats_reports_isolated_nodes() -> None:
    assert "isolated:2" in graph_stats(4, np.array([[0, 1]], dtype=np.int64))
