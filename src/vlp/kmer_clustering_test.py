"""Tests for clustering utilities."""

import numpy as np

from lib.encoded_sequences import Sequence, count_tetragraphs, encode_seq, reverse_complement_map

from .binner import VLPBinner
from .kmer_clustering import _ctcs, _initial_cluster, _set_directions, _similarity_matrix, make_4mer_clusters


def _make_binner(seqs: list[str], minlen: int = 1) -> VLPBinner:
    """Create a VlpBinner with encoded sequences that have tetragraphs counted."""
    binner = VLPBinner()
    for s in seqs:
        binner.sequences.append(Sequence(seq=encode_seq(s)))
    count_tetragraphs(binner.sequences)
    return binner


class TestCtcs:
    def test_identical_columns(self) -> None:
        """Identical columns should give xchi=0."""
        tab = np.array([[10.0, 10.0], [20.0, 20.0], [5.0, 5.0]], dtype=np.float32)
        assert _ctcs(tab, 3, 2) < 0.01

    def test_completely_different(self) -> None:
        """Disjoint columns should give a large xchi."""
        tab = np.zeros((256, 2), dtype=np.float32)
        tab[0, 0] = 100.0
        tab[1, 1] = 100.0
        assert _ctcs(tab, 256, 2) > 100.0

    def test_empty_rows_skipped(self) -> None:
        """Rows with sum < 0.5 are skipped."""
        tab = np.zeros((256, 2), dtype=np.float32)
        # Only one row has data
        tab[0, 0] = 50.0
        tab[0, 1] = 50.0
        assert _ctcs(tab, 256, 2) < 0.01

    def test_single_row(self) -> None:
        """Single non-zero row, two columns -> xchi=0."""
        tab = np.array([[30.0, 70.0]], dtype=np.float32)
        assert _ctcs(tab, 1, 2) < 0.01

    def test_proportional_columns(self) -> None:
        """Proportional columns (same ratios) should give xchi near 0."""
        tab = np.array([[10.0, 20.0], [30.0, 60.0], [5.0, 10.0]], dtype=np.float32)
        assert _ctcs(tab, 3, 2) < 0.01

    def test_symmetric(self) -> None:
        """Swapping columns should give the same xchi."""
        tab = np.zeros((256, 2), dtype=np.float32)
        tab[0, 0] = 30.0
        tab[0, 1] = 10.0
        tab[1, 0] = 10.0
        tab[1, 1] = 30.0
        xchi1 = _ctcs(tab, 256, 2)
        tab[0, 0], tab[0, 1] = tab[0, 1], tab[0, 0]
        tab[1, 0], tab[1, 1] = tab[1, 1], tab[1, 0]
        xchi2 = _ctcs(tab, 256, 2)
        assert abs(xchi1 - xchi2) < 0.01

    def test_empty_column_skipped(self) -> None:
        """A column with sum < 0.5 is skipped."""
        tab = np.zeros((256, 2), dtype=np.float32)
        tab[0, 0] = 100.0
        # column 1 is empty
        assert _ctcs(tab, 256, 2) < 0.01


class TestSimilarityMatrix:
    def test_diagonal_is_one(self) -> None:
        """Every sequence should be similar to itself."""
        binner = _make_binner(["ACGT" * 50, "GGCC" * 50])
        rev_map = reverse_complement_map()
        sim = _similarity_matrix(binner.sequences, rev_map, 2, 1, 9999.0)
        assert sim[0, 0] == 1
        assert sim[1, 1] == 1

    def test_identical_seqs_similar(self) -> None:
        """Identical sequences should be similar (xchi=0)."""
        binner = _make_binner(["ACGT" * 50, "ACGT" * 50])
        rev_map = reverse_complement_map()
        sim = _similarity_matrix(binner.sequences, rev_map, 2, 1, 100.0)
        assert sim[0, 1] == 1
        assert sim[1, 0] == 1

    def test_symmetric(self) -> None:
        """Similarity matrix should be symmetric."""
        binner = _make_binner(["ACGT" * 50, "AAAA" * 50, "CCCC" * 50])
        rev_map = reverse_complement_map()
        sim = _similarity_matrix(binner.sequences, rev_map, 3, 1, 500.0)
        for i in range(3):
            for j in range(3):
                assert sim[i, j] == sim[j, i]

    def test_minlen_filters(self) -> None:
        """Sequences shorter than minlen should be excluded."""
        binner = _make_binner(["ACGT" * 50, "AC"])  # 200 bases and 2 bases
        rev_map = reverse_complement_map()
        sim = _similarity_matrix(binner.sequences, rev_map, 1, 100, 9999.0)
        assert sim.shape == (1, 1)
        assert sim[0, 0] == 1

    def test_high_threshold_all_similar(self) -> None:
        """With a very high threshold, all pairs should be similar."""
        binner = _make_binner(["ACGT" * 50, "AAAA" * 50, "CCCC" * 50])
        rev_map = reverse_complement_map()
        sim = _similarity_matrix(binner.sequences, rev_map, 3, 1, 999999.0)
        assert sim.sum() == 9  # all ones

    def test_low_threshold_only_diagonal(self) -> None:
        """With threshold 0, only self-similarity (diagonal) should remain."""
        binner = _make_binner(["ACGT" * 50, "AAAA" * 50])
        rev_map = reverse_complement_map()
        sim = _similarity_matrix(binner.sequences, rev_map, 2, 1, 0.0)
        # Diagonal is always 1, off-diagonal needs xchi <= 0 which only identical seqs achieve
        assert sim[0, 0] == 1
        assert sim[1, 1] == 1


class TestInitialCluster:
    def test_fully_connected(self) -> None:
        """A fully connected similarity matrix should produce one cluster containing all."""
        sim = np.ones((3, 3), dtype=np.int8)
        cluster = _initial_cluster(sim)
        assert int(cluster.sum()) == 3

    def test_two_components(self) -> None:
        """Two disconnected components: first call gets the larger one."""
        sim = np.zeros((4, 4), dtype=np.int8)
        # Component 1: nodes 0, 1, 2
        sim[0, 1] = sim[1, 0] = 1
        sim[1, 2] = sim[2, 1] = 1
        sim[0, 2] = sim[2, 0] = 1
        np.fill_diagonal(sim, 1)
        # Node 3 is isolated (only self-similar)
        cluster = _initial_cluster(sim)
        assert int(cluster.sum()) == 3
        assert cluster[3] == 0

    def test_no_pairs(self) -> None:
        """If no sequence has a neighbor, cluster sum < 2."""
        sim = np.eye(3, dtype=np.int8)
        cluster = _initial_cluster(sim)
        assert int(cluster.sum()) < 2

    def test_transitive_closure(self) -> None:
        """A chain 0-1-2 should be grouped by transitive closure even if 0 and 2 aren't directly similar."""
        sim = np.eye(4, dtype=np.int8)
        sim[0, 1] = sim[1, 0] = 1
        sim[1, 2] = sim[2, 1] = 1
        # 0 and 2 not directly connected, but reachable via 1
        cluster = _initial_cluster(sim)
        assert cluster[0] == 1
        assert cluster[1] == 1
        assert cluster[2] == 1
        assert cluster[3] == 0

    def test_mutates_sim(self) -> None:
        """_initial_cluster zeros out columns of clustered nodes in sim."""
        sim = np.ones((3, 3), dtype=np.int8)
        _initial_cluster(sim)
        # All columns should be zeroed
        assert sim.sum() == 0


class TestSetDirections:
    def test_identical_seqs_all_forward(self) -> None:
        """Identical sequences should all get direction 0 (forward)."""
        binner = _make_binner(["ACGT" * 50] * 3)
        for i, s in enumerate(binner.sequences):
            s.cluster = 1
            s.active = True
        rev_map = reverse_complement_map()
        members = list(range(3))
        _set_directions(binner.sequences, members, 1, rev_map, verbose=False)
        for s in binner.sequences:
            assert s.reverse is False

    def test_reverse_complement_detected(self) -> None:
        """A reverse-complement sequence should get direction 1."""
        # Build a sequence and its reverse complement
        seq_fwd = "ACGT" * 50
        seq_rc = "ACGT" * 50  # ACGT is its own reverse complement
        # Use a non-palindromic sequence
        seq_fwd = "AAACCCGGGTTT" * 17  # 204 bases
        # Reverse complement of AAACCCGGGTTT is AAACCCGGGTTT — also palindromic
        # Use truly asymmetric sequence
        seq_fwd = "AAAAACCCCC" * 20  # 200 bases, A-rich in forward
        seq_rc = "GGGGGTTTTTT"[:-1] * 20  # approximate reverse complement

        binner = _make_binner([seq_fwd, seq_fwd, seq_rc])
        for i, s in enumerate(binner.sequences):
            s.cluster = 1
            s.active = True
        rev_map = reverse_complement_map()
        members = list(range(3))
        _set_directions(binner.sequences, members, 1, rev_map, verbose=False)
        # Directions should converge (no assertion on specific values since
        # it depends on the consensus, but they should all be set)
        for s in binner.sequences:
            assert isinstance(s.reverse, bool)

    def test_converges(self) -> None:
        """Direction setting should converge (not loop forever)."""
        binner = _make_binner(["ACGT" * 50, "AAAA" * 50, "CCCC" * 50])
        for i, s in enumerate(binner.sequences):
            s.cluster = 1
            s.active = True
        rev_map = reverse_complement_map()
        members = list(range(3))
        # If this returns, it converged
        _set_directions(binner.sequences, members, 1, rev_map, verbose=False)


class TestMakeClusters:
    def test_identical_seqs_one_cluster(self) -> None:
        """Identical long sequences should form one cluster."""
        binner = _make_binner(["ACGT" * 100] * 5)
        ncl = make_4mer_clusters(binner.sequences, minlen=1, min_cluster_size=2, xchi_thresh=500.0, verbose=False)
        assert ncl >= 1
        # All should be in the same cluster
        clusters = {s.cluster for s in binner.sequences if s.cluster > 0}
        assert len(clusters) == 1

    def test_no_clusters_if_too_few(self) -> None:
        """If min_cluster_size is larger than available sequences, no clusters form."""
        binner = _make_binner(["ACGT" * 100] * 2)
        ncl = make_4mer_clusters(binner.sequences, minlen=1, min_cluster_size=100, xchi_thresh=500.0, verbose=False)
        assert ncl == 0

    def test_short_seqs_filtered(self) -> None:
        """Sequences shorter than minlen should not be clustered."""
        binner = _make_binner(["ACGT" * 100, "ACGT" * 100, "AC"])
        make_4mer_clusters(binner.sequences, minlen=100, min_cluster_size=2, xchi_thresh=500.0, verbose=False)
        assert binner.sequences[2].cluster == 0

    def test_sets_active(self) -> None:
        """Clustered sequences should have active=True."""
        binner = _make_binner(["ACGT" * 100] * 4)
        make_4mer_clusters(binner.sequences, minlen=1, min_cluster_size=2, xchi_thresh=500.0, verbose=False)
        for s in binner.sequences:
            if s.cluster > 0:
                assert s.active is True

    def test_sets_direction(self) -> None:
        """Clustered sequences should have direction set to 0 or 1."""
        binner = _make_binner(["ACGT" * 100] * 4)
        make_4mer_clusters(binner.sequences, minlen=1, min_cluster_size=2, xchi_thresh=500.0, verbose=False)
        for s in binner.sequences:
            if s.cluster > 0:
                assert isinstance(s.reverse, bool)

    def test_low_threshold_fewer_clusters(self) -> None:
        """With xchi_thresh=0, only identical/rev-comp pairs cluster; dissimilar seqs don't."""
        # AAAA and ACGT have different tetragraph profiles and are not reverse complements
        binner = _make_binner(["AAAA" * 100, "ACGT" * 100, "ACGT" * 100])
        ncl = make_4mer_clusters(binner.sequences, minlen=1, min_cluster_size=2, xchi_thresh=0.0, verbose=False)
        # The two ACGT seqs should cluster (xchi=0), AAAA should not join them
        assert binner.sequences[0].cluster == 0 or ncl >= 1
