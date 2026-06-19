"""Tests for dbv scoring utilities."""

import numpy as np

from lib.encoded_sequences import Sequence, encode_seq

from .scoring import ClusterScore, _relative_max_pct, compute_scores, reassign, weakest_cluster


def _make_seq(s: str, cluster: int = -1, active: bool = True) -> Sequence:
    return Sequence(seq_id=s[:8], seq=encode_seq(s), cluster=cluster, active=active)


class TestComputeScores:
    def test_dissolves_small_clusters_to_minus_one(self) -> None:
        """Clusters with fewer than 3 members are dissolved (set to -1, not 0)."""
        sequences = [
            _make_seq("ACGT" * 50, cluster=0),
            _make_seq("ACGT" * 50, cluster=0),
            *[_make_seq("CCCC" * 50, cluster=1) for _ in range(4)],
        ]
        compute_scores(2, sequences, verbose=False)
        assert sequences[0].cluster == -1
        assert sequences[1].cluster == -1
        for s in sequences[2:]:
            assert s.cluster == 1

    def test_returns_score_per_cluster(self) -> None:
        sequences = [_make_seq("ACGT" * 50, cluster=0) for _ in range(4)]
        scores = compute_scores(1, sequences, verbose=False)
        assert len(scores) == 1
        for n in range(len(sequences)):
            assert scores[0].scores[n] > -np.inf


class TestReassign:
    def test_assigns_to_best_cluster(self) -> None:
        sequences = [_make_seq("AAAA", cluster=-1) for _ in range(3)]
        scores = [ClusterScore(3), ClusterScore(3)]
        scores[0].scores[:] = [10.0, 5.0, 3.0]
        scores[1].scores[:] = [20.0, 15.0, 12.0]
        reassign(scores, sequences)
        for s in sequences:
            assert s.cluster == 1

    def test_negative_best_score_yields_minus_one(self) -> None:
        sequences = [_make_seq("AAAA", cluster=0)]
        scores = [ClusterScore(1)]
        scores[0].scores[0] = -5.0
        reassign(scores, sequences)
        assert sequences[0].cluster == -1

    def test_inactive_skipped(self) -> None:
        s = _make_seq("AAAA", cluster=0, active=False)
        scores = [ClusterScore(1)]
        scores[0].scores[0] = -5.0
        reassign(scores, [s])
        assert s.cluster == 0


class TestWeakestCluster:
    def _two_cluster_seqs(self) -> list[Sequence]:
        return [_make_seq("A" * 100, cluster=0) for _ in range(3)] + [_make_seq("C" * 100, cluster=1) for _ in range(3)]

    def test_returns_minus_one_when_healthy(self) -> None:
        sequences = self._two_cluster_seqs()
        scores = [ClusterScore(6), ClusterScore(6)]
        scores[0].scores[:] = [100.0, 100.0, 100.0, -50.0, -50.0, -50.0]
        scores[1].scores[:] = [-50.0, -50.0, -50.0, 100.0, 100.0, 100.0]
        result = weakest_cluster(scores, sequences, absorb_thresh=0.5, cv_outlier_ratio=1.25, verbose=False)
        assert result == -1

    def test_picks_bad_model_via_mean_norm(self) -> None:
        """When per-base mean falls below 0.01, picks argmin(mean_norm_scores)."""
        sequences = self._two_cluster_seqs()
        scores = [ClusterScore(6), ClusterScore(6)]
        # cluster 0 has near-zero per-base avg (0.1/100 = 0.001); cluster 1 is healthy
        scores[0].scores[:] = [0.1, 0.1, 0.1, -100.0, -100.0, -100.0]
        scores[1].scores[:] = [-100.0, -100.0, -100.0, 100.0, 100.0, 100.0]
        result = weakest_cluster(scores, sequences, absorb_thresh=0.5, cv_outlier_ratio=1.25, verbose=False)
        assert result == 0

    def test_keeps_last_cluster_despite_bad_model(self) -> None:
        """With only one live cluster, never cull it — degrade to 1 bin, not 0."""
        # Single live cluster (cluster 0); cluster 1 has no members. Cluster 0's
        # per-base mean is below 0.01, which would otherwise trip the bad-model
        # branch and collapse the binning to zero.
        sequences = [_make_seq("A" * 100, cluster=0) for _ in range(3)]
        scores = [ClusterScore(3), ClusterScore(3)]
        scores[0].scores[:] = [0.1, 0.1, 0.1]
        # cluster 1 keeps its all -inf init (no members)
        result = weakest_cluster(scores, sequences, absorb_thresh=0.5, cv_outlier_ratio=1.25, verbose=False)
        assert result == -1

    def test_picks_absorbable_cluster(self) -> None:
        """When max absorbability exceeds threshold, picks argmax(absorbability)."""
        sequences = self._two_cluster_seqs()
        scores = [ClusterScore(6), ClusterScore(6)]
        # cluster 1's members fit cluster 0's model nearly as well as their own
        scores[0].scores[:] = [100.0, 100.0, 100.0, 95.0, 95.0, 95.0]
        scores[1].scores[:] = [10.0, 10.0, 10.0, 100.0, 100.0, 100.0]
        result = weakest_cluster(scores, sequences, absorb_thresh=0.5, cv_outlier_ratio=1.25, verbose=False)
        assert result == 1


class TestRelativeMaxPct:
    def test_empty_returns_zero(self) -> None:
        assert _relative_max_pct([]) == 0.0

    def test_singleton_returns_zero(self) -> None:
        assert _relative_max_pct([5.0]) == 0.0

    def test_second_max_zero_returns_zero(self) -> None:
        assert _relative_max_pct([5.0, 0.0]) == 0.0

    def test_ratio(self) -> None:
        assert _relative_max_pct([10.0, 2.0, 1.0]) == 5.0
