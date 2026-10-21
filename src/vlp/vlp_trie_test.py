"""Tests for VLP model building, scoring, and helpers."""

import numpy as np

from lib.encoded_sequences import Sequence, count_tetragraphs, encode_seq

from .binner import VLPBinner
from .sorting import sort_128
from .vlp_trie import (
    ALPHA_SIZE,
    BASE_BITS,
    LEVEL_BITS,
    MAX_DEPTH,
    VLPTrie,
    _build_training_set,
    _build_vmask,
    _create_and_count_states_jit,
)


def _make_binner_with_cluster(seqs: list[str], cluster: int = 1) -> VLPBinner:
    """Create a VLPBinner with sequences assigned to a cluster, ready for model building."""
    binner = VLPBinner()
    for s in seqs:
        binner.sequences.append(Sequence(seq=encode_seq(s), cluster=cluster, active=True))
    count_tetragraphs(binner.sequences)
    return binner


def _popcount_128(hi: int, lo: int) -> int:
    return bin(hi).count("1") + bin(lo).count("1")


class TestBuildVmask:
    def test_vmask_length(self) -> None:
        vmask = _build_vmask()
        assert vmask.shape == (MAX_DEPTH - 1, 2)

    def test_vmask_i_masks_i_plus_1_slots(self) -> None:
        """vmask[i] should mask exactly (i+1) * BASE_BITS bits."""
        vmask = _build_vmask()
        for i in range(MAX_DEPTH - 1):
            bits = _popcount_128(int(vmask[i, 0]), int(vmask[i, 1]))
            assert bits == (i + 1) * BASE_BITS

    def test_vmask_superset(self) -> None:
        """vmask[i+1] should be a superset of vmask[i]."""
        vmask = _build_vmask()
        for i in range(MAX_DEPTH - 2):
            assert int(vmask[i, 0]) & int(vmask[i + 1, 0]) == int(vmask[i, 0])
            assert int(vmask[i, 1]) & int(vmask[i + 1, 1]) == int(vmask[i, 1])

    def test_vmskshft_is_vmask_shifted(self) -> None:
        """vmskshft[i] should be vmask[i] shifted left by LEVEL_BITS - BASE_BITS."""
        from .bit_ops import ishft_128
        from .vlp_trie import _build_vmskshft

        vmask = _build_vmask()
        vmskshft = _build_vmskshft()
        shift = LEVEL_BITS - BASE_BITS
        for i in range(MAX_DEPTH - 1):
            expected = ishft_128(vmask[i], shift)
            assert vmskshft[i, 0] == expected[0]
            assert vmskshft[i, 1] == expected[1]


class TestBuildTrainingWindow:
    def test_window_length(self) -> None:
        """Window should contain one entry per base in the cluster."""
        binner = _make_binner_with_cluster(["ACGTACGT", "GGGGAAAA"])
        training = _build_training_set(binner.sequences, 1)
        assert training.shape[0] == 16  # 8 + 8

    def test_excludes_other_clusters(self) -> None:
        """Sequences not in the target cluster are excluded."""
        binner = _make_binner_with_cluster(["ACGTACGT"])
        binner.sequences.append(Sequence(seq=encode_seq("GGGG"), cluster=2, active=True))
        training = _build_training_set(binner.sequences, 1)
        assert training.shape[0] == 8

    def test_excludes_inactive(self) -> None:
        """Inactive sequences are excluded."""
        binner = _make_binner_with_cluster(["ACGTACGT", "GGGG"])
        binner.sequences[1].active = False
        training = _build_training_set(binner.sequences, 1)
        assert training.shape[0] == 8


def _prepare_count_states(seqs: list[str], cluster: int = 1) -> tuple:
    """Build sorted training set, ready for _create_and_count_states_jit."""
    binner = _make_binner_with_cluster(seqs, cluster)
    training = _build_training_set(binner.sequences, cluster)
    training_size = training.shape[0]
    sort_128(training)
    max_states = training_size + 1
    return training, max_states, training_size


class TestCountStatesJit:
    def test_root_state_has_zero_key(self) -> None:
        """The root state (last state) has key (0, 0)."""
        training, max_states, _ = _prepare_count_states(["ACGT" * 50])
        ns, _, _, states = _create_and_count_states_jit(training, 50, max_states)
        assert states[ns, 0] == 0
        assert states[ns, 1] == 0

    def test_root_model_sums_to_training_size(self) -> None:
        """Root state's model values should sum to the training set size."""
        training, max_states, ts = _prepare_count_states(["ACGT" * 50, "GGCC" * 50])
        ns, model_counts, _, _ = _create_and_count_states_jit(training, 50, max_states)
        root_sum = sum(int(model_counts[aa, ns]) for aa in range(ALPHA_SIZE))
        assert root_sum == ts

    def test_high_threshold_only_root(self) -> None:
        """A threshold higher than training size should produce only the root."""
        training, max_states, ts = _prepare_count_states(["ACGT" * 20])
        ns, *_ = _create_and_count_states_jit(training, ts + 1, max_states)
        assert ns == 1

    def test_lower_threshold_more_states(self) -> None:
        """A lower count threshold keeps more contexts, so more states."""
        seqs = ["ACGT" * 100, "GGCC" * 100, "AACG" * 100]
        training_lo, max_states_lo, _ = _prepare_count_states(seqs)
        ns_lo, *_ = _create_and_count_states_jit(training_lo, 1, max_states_lo)
        training_hi, max_states_hi, _ = _prepare_count_states(seqs)
        ns_hi, *_ = _create_and_count_states_jit(training_hi, 1000, max_states_hi)
        assert ns_lo > ns_hi

    def test_state_invariants(self) -> None:
        """Non-root states have nonzero keys and levels 1..MAX_DEPTH-1; counts are non-negative."""
        training, max_states, _ = _prepare_count_states(["ACGT" * 100, "GGCC" * 100])
        ns, model_counts, model_levels, states = _create_and_count_states_jit(training, 10, max_states)
        for s in range(1, ns):
            assert states[s, 0] != 0 or states[s, 1] != 0
            assert 1 <= model_levels[s] <= MAX_DEPTH - 1
        assert np.all(model_counts[:ALPHA_SIZE, 1 : ns + 1] >= 0)


class TestFlattenAndLogJit:
    def test_output_is_log_scaled(self) -> None:
        """Weights are log2(ALPHA_SIZE * p), with p clamped to >= 1/64 and a uniform (zero) last row."""
        binner = _make_binner_with_cluster(["ACGT" * 50, "GGCC" * 50])
        vlp_trie = VLPTrie()
        vlp_trie.make_model(1, binner.sequences)
        assert vlp_trie.model is not None
        weights = vlp_trie.model[:, 1 : vlp_trie.num_states + 1].astype(np.float64)
        np.testing.assert_array_equal(weights[ALPHA_SIZE], 0.0)
        assert np.all(weights >= np.log2(ALPHA_SIZE / 64) - 1e-6)
        assert np.all(weights <= np.log2(ALPHA_SIZE) + 1e-6)
        # Undo the log: probabilities sum to 1, plus at most ALPHA_SIZE/64 added by the clamp
        prob_sums = (2.0 ** weights[:ALPHA_SIZE] / ALPHA_SIZE).sum(axis=0)
        assert np.all(prob_sums >= 1 - 1e-5)
        assert np.all(prob_sums <= 1 + ALPHA_SIZE / 64)


class TestVlpScore:
    def test_score_own_cluster_higher(self) -> None:
        """A sequence should score higher against its own cluster's model."""
        sequences: list[Sequence] = []
        # Cluster 1: A-rich sequences
        for _ in range(3):
            sequences.append(Sequence(seq=encode_seq("A" * 200), cluster=1, active=True))
        # Cluster 2: C-rich sequences
        for _ in range(3):
            sequences.append(Sequence(seq=encode_seq("C" * 200), cluster=2, active=True))
        count_tetragraphs(sequences)

        # Build model for cluster 1
        vlp_trie = VLPTrie()
        vlp_trie.make_model(1, sequences)
        a_seq = encode_seq("A" * 100)
        score_a1, _ = vlp_trie.score(a_seq)

        c_seq = encode_seq("C" * 100)
        score_c1, _ = vlp_trie.score(c_seq)

        # A-rich sequence should score higher on A-rich model
        assert score_a1 > score_c1

    def test_empty_sequence_zero_score(self) -> None:
        """An empty sequence should return score 0."""
        binner = _make_binner_with_cluster(["ACGT" * 50, "ACGT" * 50, "ACGT" * 50])
        vlp_trie = VLPTrie()
        vlp_trie.make_model(1, binner.sequences)
        seq = np.array([], dtype=np.int8)
        score, xvar = vlp_trie.score(seq)
        assert score == 0.0
        assert xvar == 0.0


class TestConnectModel:
    def test_root_state_fallback(self) -> None:
        """All transitions from the root state (ns) should point somewhere valid."""
        binner = _make_binner_with_cluster(["ACGT" * 50, "GGCC" * 50, "AACG" * 50])
        vlp_trie = VLPTrie()
        vlp_trie.make_model(1, binner.sequences)
        assert vlp_trie.trans_ptr is not None
        for k in range(4):
            target = vlp_trie.trans_ptr[k, vlp_trie.num_states]
            assert 0 <= target <= vlp_trie.num_states

    def test_unknown_base_transition(self) -> None:
        """Transition for unknown base (index 4) should always go to root."""
        binner = _make_binner_with_cluster(["ACGT" * 50, "GGCC" * 50, "AACG" * 50])
        vlp_trie = VLPTrie()
        vlp_trie.make_model(1, binner.sequences)
        assert vlp_trie.trans_ptr is not None
        for s in range(1, vlp_trie.num_states + 1):
            assert vlp_trie.trans_ptr[4, s] == vlp_trie.num_states
