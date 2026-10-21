"""Tests for dbv binner utilities."""

import pytest

from lib.encoded_sequences import Sequence, encode_seq

from .binner import DBVBinner


def _make_seq(seq_id: str, s: str, cluster: int = -1, active: bool = False) -> Sequence:
    return Sequence(seq_id=seq_id, seq=encode_seq(s), cluster=cluster, active=active)


class TestUpdateSequences:
    def test_bins_activate_and_unbinned_deactivate(self) -> None:
        binner = DBVBinner()
        binner.sequences = [_make_seq(i, "ACGT", active=i == "d") for i in "abcd"]
        binner._update_sequences([{"a", "b"}, {"c"}], {"d"})
        assert [(s.cluster, s.active) for s in binner.sequences] == [(0, True), (0, True), (1, True), (-1, False)]

    def test_missing_sequence_raises(self) -> None:
        binner = DBVBinner()
        binner.sequences = [_make_seq("a", "ACGT")]
        with pytest.raises(Exception, match="not found"):
            binner._update_sequences([], set())


class TestRemoveCluster:
    def test_unassigns_matching_members(self) -> None:
        binner = DBVBinner()
        binner.sequences = [
            _make_seq("a", "ACGT", cluster=0, active=True),
            _make_seq("b", "ACGT", cluster=1, active=True),
            _make_seq("c", "ACGT", cluster=0, active=True),
        ]
        binner._remove_cluster(0)
        assert binner.sequences[0].cluster == -1
        assert binner.sequences[1].cluster == 1
        assert binner.sequences[2].cluster == -1


class TestLenThreshs:
    def test_returns_percentiles_and_one(self) -> None:
        binner = DBVBinner()
        binner.sequences = [_make_seq(f"s{i}", "A" * (i + 1)) for i in range(100)]
        threshs = binner._len_threshs()
        assert len(threshs) == 4
        assert threshs[-1] == 1
        assert threshs[0] > threshs[1] > threshs[2]
