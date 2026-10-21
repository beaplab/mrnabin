import math
from typing import Any

import numpy as np

from knn.binner_test import _mixture
from knn.vlp_merge import KNNVLPBinner, score_matrix
from lib.encoded_sequences import Sequence, encode_seq
from lib.evaluate import bins_by_id
from vlp.vlp_trie import VLPTrie


def _sequences(n_per_org: int = 30) -> list[Sequence]:
    return [Sequence(seq_id=l, seq=encode_seq(s)) for l, s in _mixture(n_per_org=n_per_org).items()]


def _binner(**params: Any) -> KNNVLPBinner:
    return KNNVLPBinner(
        n_neighbors=15,
        selection_percentiles_low=0.0,
        selection_percentiles_high=100.0,
        assign_remaining=False,
        **params,
    )


def test_knn_seeds_are_capped_and_the_rest_unbinned() -> None:
    binner = _binner(vlp_seed_bins=1)
    binner.sequences = _sequences()
    assert binner._run_distbin() == 1
    bins, unbinned = bins_by_id(binner.sequences)
    assert len(bins) == 1
    assert len(set(l[:2] for l in bins[0])) == 1
    assert bins[0] | unbinned == {s.seq_id for s in binner.sequences}


def test_held_out_scores_come_from_the_other_fold() -> None:
    sequences = _sequences(n_per_org=10)[:10]
    for s in sequences:
        s.cluster, s.active = 0, True
    folds = np.array([0, 1] * 5, dtype=np.int8)
    scores, _ = score_matrix(sequences, 1, folds)
    other = VLPTrie()
    other.make_model(0, [s for s, f in zip(sequences, folds) if f == 1])
    assert scores[0, 0] == other.score(sequences[0].oriented())[0]


def test_held_out_merge_keeps_organisms_apart() -> None:
    binner = _binner()
    binner.sequences = _sequences()
    binner.run()
    bins, unbinned = bins_by_id(binner.sequences)
    assert bins
    for b in bins:
        assert len(set(l[:2] for l in b)) == 1
    assert set().union(*bins) | unbinned == {s.seq_id for s in binner.sequences}


def test_tolerance_controls_how_far_merging_goes() -> None:
    def n_bins(tolerance: float) -> int:
        binner = _binner(resolution=0.5, vlp_tolerance=tolerance)
        binner.sequences = _sequences()
        binner.run()
        return len(bins_by_id(binner.sequences)[0])

    # a negative tolerance refuses every merge; an infinite one merges to the end.
    # 0.002 must not dissolve an organism's last bin: its contigs would fit
    # nowhere, which the all-contig fit counts as a loss
    assert n_bins(-1.0) >= n_bins(0.002) == 2
    assert n_bins(math.inf) == 1


def test_no_surviving_seed_leaves_everything_unbinned() -> None:
    # at resolution 0.9 every seed is too small to split into two modelled folds
    binner = _binner(resolution=0.9)
    binner.sequences = _sequences()
    binner.run()
    bins, unbinned = bins_by_id(binner.sequences)
    assert bins == [] and len(unbinned) == len(binner.sequences)
