"""Tests for scoring utilities."""

import numpy as np

from lib.encoded_sequences import Sequence, count_tetragraphs, encode_seq

from .binner import VLPBinner
from .scoring import VLPScore
from .scoring import _compute_scores as compute_scores
from .scoring import find_best_scores, remove_next_cluster


def _make_clustered_binner(cluster_seqs: dict[int, list[str]]) -> tuple[VLPBinner, int]:
    """Create a VLPBinner with sequences assigned to clusters.

    Args:
        cluster_seqs: mapping from cluster number to list of sequence strings.
    Returns:
        (binner, max_cluster_num)
    """
    binner = VLPBinner()
    max_k = 0
    for k, seqs in cluster_seqs.items():
        max_k = max(max_k, k)
        for s in seqs:
            binner.sequences.append(Sequence(seq=encode_seq(s), cluster=k, active=True))
    count_tetragraphs(binner.sequences)
    return binner, max_k


class TestVLPScore:
    def test_initialization_shape(self) -> None:
        """Score arrays should have correct shapes."""
        vs = VLPScore(num_clusters=3, num_sequences=10)
        assert vs.score.shape == (2, 4, 10)
        assert vs.mean.shape == (4,)
        assert vs.var.shape == (4,)

    def test_initialization_values(self) -> None:
        """Score array should be initialized to -np.inf, mean/var to 0."""
        vs = VLPScore(num_clusters=2, num_sequences=5)
        assert np.all(vs.score == -np.inf)
        assert np.all(vs.mean == 0.0)
        assert np.all(vs.var == 0.0)


class TestComputeScores:
    def test_dissolves_small_clusters(self) -> None:
        """Clusters with fewer than 3 members are dissolved (set to cluster 0)."""
        binner, ncl = _make_clustered_binner(
            {
                1: ["ACGT" * 50, "ACGT" * 50],  # only 2 members -> dissolved
                2: ["GGCC" * 50] * 4,
            }
        )
        compute_scores(ncl, binner.sequences, verbose=False)
        # The two sequences originally in cluster 1 should now be in cluster 0
        for s in binner.sequences:
            if s.seq[0] == 0:  # A-starting sequences were cluster 1
                assert s.cluster == 0

    def test_keeps_large_clusters(self) -> None:
        """Clusters with 3+ members are kept."""
        binner, ncl = _make_clustered_binner(
            {
                1: ["ACGT" * 50] * 4,
            }
        )
        compute_scores(ncl, binner.sequences, verbose=False)
        for s in binner.sequences:
            assert s.cluster == 1

    def test_scores_both_orientations(self) -> None:
        """Score matrix should have values for both forward (0) and reverse (1) orientations."""
        binner, ncl = _make_clustered_binner(
            {
                1: ["ACGT" * 50] * 4,
            }
        )
        vlp_score = compute_scores(ncl, binner.sequences, verbose=False)
        for n in range(len(binner.sequences)):
            assert vlp_score.score[0, 1, n] > -np.inf
            assert vlp_score.score[1, 1, n] > -np.inf

    def test_mean_and_var_set(self) -> None:
        """Mean and variance should be computed for clusters with members."""
        binner, ncl = _make_clustered_binner(
            {
                1: ["ACGT" * 50] * 4,
            }
        )
        vlp_score = compute_scores(ncl, binner.sequences, verbose=False)
        # Mean should be non-zero for cluster 1
        assert vlp_score.mean[1] != 0.0

    def test_sets_direction(self) -> None:
        """compute_scores should set direction for sequences in each cluster."""
        binner, ncl = _make_clustered_binner(
            {
                1: ["ACGT" * 50] * 4,
            }
        )
        compute_scores(ncl, binner.sequences, verbose=False)
        for s in binner.sequences:
            assert isinstance(s.reverse, bool)

    def test_returns_vlp_score(self) -> None:
        """Should return a VLPScore object."""
        binner, ncl = _make_clustered_binner(
            {
                1: ["ACGT" * 50] * 4,
            }
        )
        result = compute_scores(ncl, binner.sequences, verbose=False)
        assert isinstance(result, VLPScore)

    def test_inactive_sequences_skipped(self) -> None:
        """Inactive sequences should not be scored."""
        binner, ncl = _make_clustered_binner(
            {
                1: ["ACGT" * 50] * 4,
            }
        )
        binner.sequences[0].active = False
        vlp_score = compute_scores(ncl, binner.sequences, verbose=False)
        # Inactive sequence should still have the init value
        assert vlp_score.score[0, 1, 0] == -np.inf


class TestFindBestScores:
    def test_converges(self) -> None:
        """find_best_scores should terminate (convergence)."""
        binner, ncl = _make_clustered_binner(
            {
                1: ["AAAA" * 50] * 4,
                2: ["CCCC" * 50] * 4,
            }
        )
        # Should not loop forever
        find_best_scores(ncl, binner.sequences, verbose=False)

    def test_reassigns_to_best_cluster(self) -> None:
        """A sequence should end up in the cluster whose model scores it highest."""
        binner, ncl = _make_clustered_binner(
            {
                1: ["AAAA" * 50] * 4,
                2: ["CCCC" * 50] * 4,
            }
        )
        # Put an A-rich sequence in cluster 2 (wrong cluster)
        misplaced = Sequence(seq=encode_seq("AAAA" * 50), cluster=2, active=True)
        count_tetragraphs([misplaced])
        binner.sequences.append(misplaced)

        find_best_scores(ncl, binner.sequences, verbose=False)
        # The A-rich sequence should move to cluster 1
        assert binner.sequences[-1].cluster == 1

    def test_stable_assignment_no_change(self) -> None:
        """If sequences are already in their best cluster, nothing should change."""
        binner, ncl = _make_clustered_binner(
            {
                1: ["AAAA" * 50] * 5,
                2: ["CCCC" * 50] * 5,
            }
        )
        # Record initial assignments
        initial = [s.cluster for s in binner.sequences]
        find_best_scores(ncl, binner.sequences, verbose=False)
        final = [s.cluster for s in binner.sequences]
        assert initial == final

    def test_sets_direction(self) -> None:
        """After find_best_scores, all active sequences should have a direction."""
        binner, ncl = _make_clustered_binner(
            {
                1: ["ACGT" * 50] * 4,
                2: ["GGCC" * 50] * 4,
            }
        )
        find_best_scores(ncl, binner.sequences, verbose=False)
        for s in binner.sequences:
            if s.active:
                assert isinstance(s.reverse, bool)


class TestRemoveNextCluster:
    def test_removes_weakest_cluster(self) -> None:
        """The cluster whose members score best elsewhere should be removed."""
        # Make two very similar clusters - one should be absorbable
        binner, ncl = _make_clustered_binner(
            {
                1: ["ACGT" * 50] * 5,
                2: ["ACGT" * 50] * 3,  # smaller, similar cluster -> likely removable
            }
        )
        kbest, xbest_score = remove_next_cluster(binner, ncl, thresh_next=-99999, verbose=False)
        assert kbest in (1, 2)
        # The removed cluster's members should now be in cluster 0
        removed_count = sum(1 for s in binner.sequences if s.cluster == kbest)
        assert removed_count == 0

    def test_no_removal_below_threshold(self) -> None:
        """If the best removal score is below thresh_next, no cluster is removed."""
        binner, ncl = _make_clustered_binner(
            {
                1: ["AAAA" * 50] * 5,
                2: ["CCCC" * 50] * 5,
            }
        )
        # Very high threshold -> no removal
        kbest, xbest_score = remove_next_cluster(binner, ncl, thresh_next=99999, verbose=False)
        assert xbest_score < np.inf
        # Both clusters should still have members
        c1 = sum(1 for s in binner.sequences if s.cluster == 1)
        c2 = sum(1 for s in binner.sequences if s.cluster == 2)
        assert c1 > 0
        assert c2 > 0

    def test_returns_tuple(self) -> None:
        """Should return (cluster_id, removal_score)."""
        binner, ncl = _make_clustered_binner(
            {
                1: ["ACGT" * 50] * 4,
            }
        )
        result = remove_next_cluster(binner, ncl, thresh_next=0, verbose=False)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)

    def test_removal_unassigns_members(self) -> None:
        """Removed cluster's members should be set to cluster 0."""
        binner, ncl = _make_clustered_binner(
            {
                1: ["ACGT" * 50] * 10,
                2: ["ACGT" * 50] * 3,  # small similar cluster
            }
        )
        kbest, _ = remove_next_cluster(binner, ncl, thresh_next=-99999, verbose=False)
        for s in binner.sequences:
            assert s.cluster != kbest or s.cluster == 0
