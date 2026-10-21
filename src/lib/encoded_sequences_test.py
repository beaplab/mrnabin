"""Tests for encoded sequence utilities."""

import tempfile

import numpy as np
import pytest

from .encoded_sequences import (
    Sequence,
    count_tetragraphs,
    decode_seq,
    encode_seq,
    read_mrna_file_encoded,
    reverse_complement,
    reverse_complement_map,
)


class TestReadMrnaFile:
    def test_reads_labels_and_sequences(self) -> None:
        """read_mrna_file_encoded should assign seq_id from the label column and encode the nucleotide column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("gene1 ACGT\ngene2 TTAA\n")
            f.flush()
            seqs = read_mrna_file_encoded(f.name)

        assert len(seqs) == 2
        assert seqs[0].seq_id == "gene1"
        assert list(seqs[0].seq) == [0, 1, 2, 3]  # A=0, C=1, G=2, T=3
        assert seqs[1].seq_id == "gene2"
        assert list(seqs[1].seq) == [3, 3, 0, 0]


def test_decode_seq_roundtrips_including_unknown_bases() -> None:
    """decode_seq(encode_seq(s)) returns s; unknown bases (value 4) decode to 'N'."""
    assert decode_seq(Sequence(seq=encode_seq("ACGTANGT"))) == "ACGTANGT"


class TestCountTetragraphs:
    @pytest.mark.parametrize(
        "seq, expected",
        [
            ("ACG", {}),  # shorter than 4 bases: nothing
            ("CCCC", {85: 1.0}),  # 0b01010101
            ("AAAAAAA", {0: 4.0}),  # 7 bases, 4 windows, all AAAA
            ("ACGTAAG", {27: 1.0, 108: 1.0, 176: 1.0, 194: 1.0}),  # ACGT CGTA GTAA TAAG
            # N resets the window: ANCG and NCGT are skipped, CGTA counts
            ("ANCGTA", {108: 1.0}),
        ],
    )
    def test_counts(self, seq: str, expected: dict[int, float]) -> None:
        sequences = [Sequence(seq=encode_seq(seq))]
        count_tetragraphs(sequences)
        cnts = sequences[0].cnts
        assert {int(i): float(cnts[i]) for i in np.flatnonzero(cnts)} == expected

    def test_multiple_records(self) -> None:
        """Two records get independent tetragraph counts."""
        sequences = [
            Sequence(seq=encode_seq("AAAA")),  # 1 tetragraph at index 0
            Sequence(seq=encode_seq("CCCC")),  # 1 tetragraph at index 85
        ]
        count_tetragraphs(sequences)
        assert sequences[0].cnts[0] == 1.0
        assert sequences[0].cnts.sum() == 1.0
        assert sequences[1].cnts[85] == 1.0
        assert sequences[1].cnts.sum() == 1.0


@pytest.mark.parametrize(
    "seq, rev, expected",
    [
        ("AACG", False, [0, 0, 1, 2]),  # unchanged
        ("AACG", True, [1, 2, 3, 3]),  # CGTT; not a palindrome, so reversal is visible
        ("A", True, [3]),
        ("N", True, [4]),  # unknown stays unknown
    ],
)
def test_reverse_complement(seq: str, rev: bool, expected: list[int]) -> None:
    np.testing.assert_array_equal(reverse_complement(encode_seq(seq), rev), expected)


class TestReverseComplementMap:
    @pytest.mark.parametrize(
        "index, expected",
        [
            (97, 182),  # CGAC -> GTCG
            (0, 255),  # AAAA -> TTTT
            (27, 27),  # ACGT is its own reverse complement
        ],
    )
    def test_known_values(self, index: int, expected: int) -> None:
        assert reverse_complement_map()[index] == expected

    def test_double_application_is_identity(self) -> None:
        """Applying the map twice returns the original index."""
        rev_map = reverse_complement_map()
        for i in range(256):
            assert rev_map[rev_map[i]] == i
