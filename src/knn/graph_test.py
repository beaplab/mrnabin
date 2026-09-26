import numpy as np

from knn.graph import auto_n_neighbors, graph_stats, knn_indices, knn_indices_from_stack, mutual_knn_edges
from lib.distances import tv_distance_matrix


def _two_cluster_distances() -> np.ndarray:
    # two tight groups of five, far apart on the simplex
    rng = np.random.default_rng(0)
    dim = 32
    a = np.full(dim, 1.0 / dim)
    b = np.zeros(dim)
    b[: dim // 2] = 2.0 / dim
    points = []
    for center in (a, b):
        for _ in range(5):
            p = np.abs(center + rng.normal(0, 0.0005, dim))
            points.append(p / p.sum())
    return tv_distance_matrix(np.stack(points))


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
    assert not any(i in row for i, row in enumerate(idx.tolist()))
    assert idx[0].tolist() == [1, 3]
    assert idx[2].tolist() == [0, 1]


def test_knn_indices_caps_at_available_neighbors() -> None:
    distances = np.array([[0.0, 0.5], [0.5, 0.0]])
    assert knn_indices(distances, 10).shape == (2, 1)


def test_mutual_knn_edges_returns_upper_triangle_pairs() -> None:
    idx = knn_indices(_two_cluster_distances(), 3)
    edges = mutual_knn_edges(idx, mutual=True)
    assert edges.shape[1] == 2
    assert np.all(edges[:, 0] < edges[:, 1])
    # each undirected pair appears exactly once
    assert len({tuple(e) for e in edges.tolist()}) == edges.shape[0]


def test_mutual_knn_edges_separates_two_clusters() -> None:
    idx = knn_indices(_two_cluster_distances(), 3)
    edges = mutual_knn_edges(idx, mutual=True)
    # no edge should cross between indices 0-4 and 5-9
    crossing = [(i, j) for i, j in edges.tolist() if (i < 5) != (j < 5)]
    assert crossing == []


def test_mutual_filter_keeps_fewer_edges_than_directed() -> None:
    idx = knn_indices(_two_cluster_distances(), 4)
    assert mutual_knn_edges(idx, mutual=True).shape[0] <= mutual_knn_edges(idx, mutual=False).shape[0]


def test_mutual_knn_edges_handles_empty_input() -> None:
    assert mutual_knn_edges(np.zeros((0, 0), dtype=np.int64)).shape == (0, 2)


def test_auto_n_neighbors_grows_with_log_n() -> None:
    assert auto_n_neighbors(2) == 5
    assert auto_n_neighbors(4000) == 9
    assert auto_n_neighbors(100000) < 15


def test_graph_stats_reports_isolated_nodes() -> None:
    assert "isolated:2" in graph_stats(4, np.array([[0, 1]], dtype=np.int64))


def _two_cluster_stack() -> np.ndarray:
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


def test_blocked_knn_matches_the_full_matrix() -> None:
    stack = _two_cluster_stack()
    expected = knn_indices(tv_distance_matrix(stack), 6)
    for block_size in (1, 7, 40, 1000):
        assert np.array_equal(knn_indices_from_stack(stack, 6, block_size), expected)


def test_blocked_knn_excludes_self_across_block_boundaries() -> None:
    stack = _two_cluster_stack()
    idx = knn_indices_from_stack(stack, 5, block_size=7)
    assert not any(i in row for i, row in enumerate(idx.tolist()))


def test_mutual_knn_edges_match_the_directed_lists() -> None:
    """Every mutual edge must appear in both endpoints' neighbour lists."""
    idx = knn_indices_from_stack(_two_cluster_stack(), 8)
    lists = [set(row) for row in idx.tolist()]
    for i, j in mutual_knn_edges(idx, mutual=True).tolist():
        assert j in lists[i] and i in lists[j]
