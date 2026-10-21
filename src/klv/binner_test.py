import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import klv.binner
from klv.binner import KLVBinner, score_matrix
from knn.binner_test import _mixture
from lib.encoded_sequences import Sequence, encode_seq
from lib.evaluate import bins_by_id
from vlp.vlp_trie import VLPTrie


def _sequences(n_per_org: int = 30) -> list[Sequence]:
    return [Sequence(seq_id=l, seq=encode_seq(s)) for l, s in _mixture(n_per_org=n_per_org).items()]


def _binner(**params: Any) -> KLVBinner:
    params.setdefault("min_bin_size", 3)
    return KLVBinner(
        n_neighbors=15,
        # 60 contigs: the cap would cut k to 5 and shatter the seeds; these test the merge
        neighbor_cap=0,
        selection_percentiles_low=0.0,
        selection_percentiles_high=100.0,
        **params,
    )


def test_knn_seeds_are_capped_and_the_rest_unbinned() -> None:
    binner = _binner(vlp_seed_bins=1)
    binner.sequences = _sequences()
    assert binner._run_distbin() == 1
    bins, unbinned = bins_by_id(binner.sequences)
    assert len(bins) == 1
    assert len(set(l[:2] for l in bins[0])) == 1
    assert unbinned


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
    bins, _ = bins_by_id(binner.sequences)
    assert bins
    for b in bins:
        assert len(set(l[:2] for l in b)) == 1


def test_tolerance_controls_how_far_merging_goes() -> None:
    def n_bins(tolerance: float) -> int:
        binner = _binner(vlp_tolerance=tolerance)
        binner.sequences = _sequences()
        binner.run()
        return len(bins_by_id(binner.sequences)[0])

    # negative refuses every merge, infinite merges to the end; 0.002 keeps one bin per organism
    assert n_bins(-1.0) >= n_bins(0.002) == 2
    assert n_bins(math.inf) == 1


def test_no_surviving_seed_leaves_everything_unbinned() -> None:
    # every KNN bin is below min_bin_size, so KNN hands over no seeds
    binner = _binner(min_bin_size=1000)
    binner.sequences = _sequences()
    binner.run()
    bins, unbinned = bins_by_id(binner.sequences)
    assert bins == [] and len(unbinned) == len(binner.sequences)


def test_merge_that_leaves_no_cluster_is_undone(monkeypatch: pytest.MonkeyPatch) -> None:
    sequences = _sequences(n_per_org=3)
    for i, s in enumerate(sequences):
        s.cluster, s.active = i // 3, True
    before = [s.cluster for s in sequences]
    calls = 0

    def fake_scores(
        seqs: list[Sequence], n_clusters: int, folds: Any, cache: Any = None
    ) -> tuple[npt.NDArray[np.float64], list[int]]:
        # first call: two live clusters; after the dissolve, every cluster is
        # too small to model, as score_matrix would report it
        nonlocal calls
        calls += 1
        if calls > 1:
            for s in seqs:
                s.cluster = -1
            return np.full((n_clusters, len(seqs)), -np.inf), [0] * n_clusters
        scores = np.array([[1.0 if c == k else 0.5 for c in before] for k in range(n_clusters)])
        return scores, [1] * n_clusters

    monkeypatch.setattr(klv.binner, "score_matrix", fake_scores)
    binner = _binner()
    binner.sequences = sequences
    binner._merge(2, np.zeros(len(sequences), dtype=np.int8))
    assert calls == 2
    assert [s.cluster for s in sequences] == before


def test_cached_rows_skip_retraining_and_match(monkeypatch: pytest.MonkeyPatch) -> None:
    sequences = _sequences(n_per_org=10)
    for i, s in enumerate(sequences):
        s.cluster, s.active = i // 10, True
    folds = np.array([0, 1] * 10, dtype=np.int8)
    cache: klv.binner.RowCache = {}
    expected, _ = score_matrix(sequences, 2, folds, cache)

    trained: list[int] = []
    make_model = VLPTrie.make_model

    def counting(self: VLPTrie, k: int, *args: Any, **kwargs: Any) -> None:
        trained.append(k)
        make_model(self, k, *args, **kwargs)

    monkeypatch.setattr(VLPTrie, "make_model", counting)
    scores, _ = score_matrix(sequences, 2, folds, cache)
    assert trained == [] and np.array_equal(scores, expected)
    # moving a contig retrains the two clusters it touched, one trie per fold
    sequences[0].cluster = 1
    score_matrix(sequences, 2, folds, cache)
    assert trained == [0, 0, 1, 1]
