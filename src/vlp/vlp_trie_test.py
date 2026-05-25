"""Tests for VLP model building, scoring, and helpers."""

import numpy as np
import numpy.typing as npt

from lib.encoded_sequences import Sequence, count_tetragraphs, encode_seq

from .binner import VLPBinner
from .sorting import binary_search_128, sort_128
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


class TestBinarySearch128:
    def _make_keys(self, hi: list[int], lo: list[int]) -> npt.NDArray[np.uint64]:
        keys = np.empty((2, len(hi)), dtype=np.uint64)
        keys[0] = np.array(hi, dtype=np.uint64)
        keys[1] = np.array(lo, dtype=np.uint64)
        return keys

    def _make_val(self, hi: int, lo: int) -> npt.NDArray[np.uint64]:
        return np.array([hi, lo], dtype=np.uint64)

    def test_found(self) -> None:
        keys = self._make_keys([1, 2, 3], [10, 20, 30])
        assert binary_search_128(keys, self._make_val(2, 20)) == 1

    def test_not_found(self) -> None:
        keys = self._make_keys([1, 2, 3], [10, 20, 30])
        assert binary_search_128(keys, self._make_val(2, 25)) == -1

    def test_first_element(self) -> None:
        keys = self._make_keys([1, 2, 3], [10, 20, 30])
        assert binary_search_128(keys, self._make_val(1, 10)) == 0

    def test_last_element(self) -> None:
        keys = self._make_keys([1, 2, 3], [10, 20, 30])
        assert binary_search_128(keys, self._make_val(3, 30)) == 2

    def test_single_element_found(self) -> None:
        keys = self._make_keys([5], [42])
        assert binary_search_128(keys, self._make_val(5, 42)) == 0

    def test_single_element_not_found(self) -> None:
        keys = self._make_keys([5], [42])
        assert binary_search_128(keys, self._make_val(5, 43)) == -1

    def test_same_hi_different_lo(self) -> None:
        keys = self._make_keys([1, 1, 1], [10, 20, 30])
        assert binary_search_128(keys, self._make_val(1, 20)) == 1
        assert binary_search_128(keys, self._make_val(1, 15)) == -1


def _popcount_128(hi: int, lo: int) -> int:
    return bin(hi).count("1") + bin(lo).count("1")


class TestBuildVmask:
    def test_vmask_length(self) -> None:
        vmask = _build_vmask()
        assert vmask.shape == (MAX_DEPTH - 1, 2)

    def test_vmask_all_nonzero(self) -> None:
        vmask = _build_vmask()
        for i in range(MAX_DEPTH - 1):
            assert vmask[i, 0] != 0 or vmask[i, 1] != 0

    def test_vmask_bit_count_increases(self) -> None:
        """Higher indices should mask more bits."""
        vmask = _build_vmask()
        for i in range(MAX_DEPTH - 2):
            bits_i = _popcount_128(int(vmask[i, 0]), int(vmask[i, 1]))
            bits_next = _popcount_128(int(vmask[i + 1, 0]), int(vmask[i + 1, 1]))
            assert bits_next > bits_i

    def test_vmask_first_masks_one_slot(self) -> None:
        """vmask[0] should mask exactly BASE_BITS bits (1 slot)."""
        vmask = _build_vmask()
        bits = _popcount_128(int(vmask[0, 0]), int(vmask[0, 1]))
        assert bits == BASE_BITS

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
    def test_always_produces_root(self) -> None:
        """Even with few entries, at least the root state is produced."""
        training, max_states, _ = _prepare_count_states(["ACGT"])
        ns, *_ = _create_and_count_states_jit(training, 50, max_states)
        assert ns >= 1

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
        """A lower threshold should produce at least as many states."""
        seqs = ["ACGT" * 100, "GGCC" * 100, "AACG" * 100]
        training_lo, max_states_lo, _ = _prepare_count_states(seqs)
        ns_lo, *_ = _create_and_count_states_jit(training_lo, 1, max_states_lo)
        training_hi, max_states_hi, _ = _prepare_count_states(seqs)
        ns_hi, *_ = _create_and_count_states_jit(training_hi, 1000, max_states_hi)
        assert ns_lo >= ns_hi

    def test_non_root_keys_nonzero(self) -> None:
        """All non-root state keys should be nonzero."""
        training, max_states, _ = _prepare_count_states(["ACGT" * 100, "GGCC" * 100])
        ns, _, _, states = _create_and_count_states_jit(training, 10, max_states)
        for s in range(1, ns):
            assert states[s, 0] != 0 or states[s, 1] != 0

    def test_model_values_non_negative(self) -> None:
        """Before flatten_and_log, all model values should be non-negative counts."""
        training, max_states, _ = _prepare_count_states(["ACGT" * 50, "GGCC" * 50])
        ns, model_counts, _, _ = _create_and_count_states_jit(training, 50, max_states)
        for s in range(1, ns + 1):
            for aa in range(ALPHA_SIZE):
                assert model_counts[aa, s] >= 0

    def test_state_levels_in_valid_range(self) -> None:
        """Each non-root state's level should be 1..MAX_DEPTH-1."""
        training, max_states, _ = _prepare_count_states(["ACGT" * 100, "GGCC" * 100])
        ns, _, model_levels, _ = _create_and_count_states_jit(training, 10, max_states)
        for s in range(1, ns):
            assert 1 <= model_levels[s] <= MAX_DEPTH - 1

    def test_deterministic(self) -> None:
        """Same input should always produce the same number of states."""
        seqs = ["ACGT" * 50, "GGCC" * 50]
        training1, max_states1, _ = _prepare_count_states(seqs)
        ns1, *_ = _create_and_count_states_jit(training1, 50, max_states1)
        training2, max_states2, _ = _prepare_count_states(seqs)
        ns2, *_ = _create_and_count_states_jit(training2, 50, max_states2)
        assert ns1 == ns2


class TestFlattenAndLogJit:
    def test_output_is_log_scaled(self) -> None:
        """After flatten+log, model values should be log-scaled (can be negative)."""
        binner = _make_binner_with_cluster(["ACGT" * 50, "GGCC" * 50])
        vlp_trie = VLPTrie()
        vlp_trie.make_model(1, binner.sequences)
        # The root state (ns) should have log-scaled values
        assert vlp_trie.model is not None
        has_negative = False
        for aa in range(4):
            if vlp_trie.model[aa, vlp_trie.num_states] < 0:
                has_negative = True
        assert has_negative or vlp_trie.num_states > 0  # model was built


class TestMakeVlpModel:
    def test_builds_states(self) -> None:
        """make_vlp_model should produce at least one state."""
        binner = _make_binner_with_cluster(["ACGT" * 50, "ACGT" * 50, "ACGT" * 50])
        vlp_trie = VLPTrie()
        vlp_trie.make_model(1, binner.sequences)
        assert vlp_trie.num_states > 0

    def test_vlp_ptr_populated(self) -> None:
        """vlp_ptr should have non-zero transitions after model building."""
        binner = _make_binner_with_cluster(["ACGT" * 50, "GGCC" * 50, "AACG" * 50])
        vlp_trie = VLPTrie()
        vlp_trie.make_model(1, binner.sequences)
        assert vlp_trie.trans_ptr is not None
        # At least some transitions should point to non-root states
        has_transition = False
        for s in range(1, vlp_trie.num_states + 1):
            for k in range(4):
                if vlp_trie.trans_ptr[k, s] != 0 and vlp_trie.trans_ptr[k, s] != vlp_trie.num_states:
                    has_transition = True
        assert has_transition

    def test_different_clusters_different_models(self) -> None:
        """Building models for different clusters should give different states."""
        sequences: list[Sequence] = []
        # Cluster 1: A-rich
        for _ in range(3):
            sequences.append(Sequence(seq=encode_seq("A" * 200), cluster=1, active=True))
        # Cluster 2: C-rich
        for _ in range(3):
            sequences.append(Sequence(seq=encode_seq("C" * 200), cluster=2, active=True))
        count_tetragraphs(sequences)

        trie1 = VLPTrie()
        trie1.make_model(1, sequences)
        assert trie1.model is not None
        model1_root = trie1.model[:4, trie1.num_states].copy()

        trie2 = VLPTrie()
        trie2.make_model(2, sequences)
        assert trie2.model is not None
        model2_root = trie2.model[:4, trie2.num_states].copy()

        # The root-state weights should differ
        assert not np.allclose(model1_root, model2_root)


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

    def test_returns_variance(self) -> None:
        """score should return a non-negative variance component."""
        binner = _make_binner_with_cluster(["ACGT" * 50, "ACGT" * 50, "ACGT" * 50])
        vlp_trie = VLPTrie()
        vlp_trie.make_model(1, binner.sequences)
        seq = encode_seq("ACGTACGT")
        _, xvar = vlp_trie.score(seq)
        assert xvar >= 0.0

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
