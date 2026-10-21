"""Tests for scoring utilities."""

import numpy as np

from lib.encoded_sequences import Sequence, count_tetragraphs, encode_seq, reverse_complement

from .binner import VLPBinner
from .scoring import _compute_scores as compute_scores
from .scoring import find_best_scores, remove_next_cluster


def _make_clustered_binner(cluster_seqs: dict[int, list[str]]) -> tuple[VLPBinner, int]:
    """A VLPBinner with sequences in the given clusters, and the max cluster number."""
    binner = VLPBinner()
    max_k = 0
    for k, seqs in cluster_seqs.items():
        max_k = max(max_k, k)
        for s in seqs:
            binner.sequences.append(Sequence(seq=encode_seq(s), cluster=k, active=True))
    count_tetragraphs(binner.sequences)
    return binner, max_k


def _skewed(n: int, probs: list[float], seed: int, length: int = 400) -> list[str]:
    """n random sequences with base composition `probs` (A, C, G, T)."""
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list("ACGT"), size=length, p=probs)) for _ in range(n)]


# A-rich and not strand-symmetric, so a sequence and its reverse complement score differently
A_RICH = [0.4, 0.3, 0.2, 0.1]
# strand-symmetric and unlike A_RICH in either orientation
GC_RICH = [0.1, 0.4, 0.4, 0.1]


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
        assert [s.cluster for s in binner.sequences] == [0, 0, 2, 2, 2, 2]

    def test_large_cluster_kept_and_scored(self) -> None:
        """A 3+ member cluster is kept, scored in both orientations, and gets a per-base mean."""
        binner, ncl = _make_clustered_binner(
            {
                1: ["ACGT" * 50] * 4,
            }
        )
        vlp_score = compute_scores(ncl, binner.sequences, verbose=False)
        for n, s in enumerate(binner.sequences):
            assert s.cluster == 1
            assert vlp_score.score[0, 1, n] > -np.inf
            assert vlp_score.score[1, 1, n] > -np.inf
        assert vlp_score.mean[1] != 0.0

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

    def test_reverse_strand_joins_its_cluster_flipped(self) -> None:
        """A reverse-complemented member is found in its own cluster's model on the other strand."""
        forward = _skewed(5, A_RICH, seed=0)
        binner, ncl = _make_clustered_binner({1: forward[:4], 2: _skewed(4, GC_RICH, seed=1)})
        # forward[4]'s reverse complement, misfiled in cluster 2 and marked forward
        flipped = Sequence(seq=reverse_complement(encode_seq(forward[4]), True), cluster=2, active=True, reverse=False)
        count_tetragraphs([flipped])
        binner.sequences.append(flipped)

        find_best_scores(ncl, binner.sequences, verbose=False)
        assert (flipped.cluster, flipped.reverse) == (1, True)
        assert [(s.cluster, s.reverse) for s in binner.sequences[:4]] == [(1, False)] * 4

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


class TestRemoveNextCluster:
    def test_removes_weakest_cluster(self) -> None:
        """The cluster whose members score best elsewhere is the one removed.

        Cluster 3 is a subset of cluster 1, so cluster 1 absorbs it.
        """
        a_rich = _skewed(5, A_RICH, seed=0)
        binner, ncl = _make_clustered_binner({1: a_rich, 2: _skewed(5, GC_RICH, seed=1), 3: a_rich[:3]})
        kbest, _ = remove_next_cluster(binner, ncl, thresh_next=-99999, verbose=False)
        assert kbest == 3
        assert [s.cluster for s in binner.sequences] == [1] * 5 + [2] * 5 + [0] * 3

    def test_no_removal_below_threshold(self) -> None:
        """If the best removal score is below thresh_next, no cluster is removed."""
        binner, ncl = _make_clustered_binner(
            {
                1: ["AAAA" * 50] * 5,
                2: ["CCCC" * 50] * 5,
            }
        )
        remove_next_cluster(binner, ncl, thresh_next=99999, verbose=False)
        assert [s.cluster for s in binner.sequences] == [1] * 5 + [2] * 5
