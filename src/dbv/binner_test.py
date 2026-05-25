"""Tests for dbv binner utilities."""

import pytest

from lib.encoded_sequences import Sequence, encode_seq
from lib.evaluate import bins_by_id

from .binner import DBVBinner


def _make_seq(seq_id: str, s: str, cluster: int = -1, active: bool = False) -> Sequence:
    return Sequence(seq_id=seq_id, seq=encode_seq(s), cluster=cluster, active=active)


class TestBinsById:
    def test_groups_by_cluster(self) -> None:
        sequences = [
            _make_seq("a", "ACGT", cluster=0, active=True),
            _make_seq("b", "ACGT", cluster=0, active=True),
            _make_seq("c", "ACGT", cluster=1, active=True),
        ]
        bins, unbinned = bins_by_id(sequences)
        assert bins == [{"a", "b"}, {"c"}]
        assert unbinned == set()

    def test_unbinned_includes_inactive_and_minus_one(self) -> None:
        sequences = [
            _make_seq("a", "ACGT", cluster=0, active=True),
            _make_seq("b", "ACGT", cluster=-1, active=True),
            _make_seq("c", "ACGT", cluster=0, active=False),
        ]
        bins, unbinned = bins_by_id(sequences)
        assert bins == [{"a"}]
        assert unbinned == {"b", "c"}


class TestUpdateSequences:
    def test_assigns_cluster_and_activates(self) -> None:
        binner = DBVBinner()
        binner.sequences = [
            _make_seq("a", "ACGT"),
            _make_seq("b", "ACGT"),
            _make_seq("c", "ACGT"),
        ]
        binner._update_sequences([{"a", "b"}, {"c"}], set())
        assert binner.sequences[0].cluster == 0
        assert binner.sequences[0].active is True
        assert binner.sequences[1].cluster == 0
        assert binner.sequences[2].cluster == 1

    def test_unbinned_get_minus_one_and_inactive(self) -> None:
        binner = DBVBinner()
        binner.sequences = [_make_seq("a", "ACGT")]
        binner._update_sequences([], {"a"})
        assert binner.sequences[0].cluster == -1
        assert binner.sequences[0].active is False

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
