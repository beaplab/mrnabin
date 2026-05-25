"""Tests for encoded sequence utilities."""

import tempfile

import numpy as np

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


class TestDecodeSeq:
    def test_roundtrip(self) -> None:
        """decode_seq(encode_seq(s)) should return the original string."""
        s = Sequence(seq=encode_seq("ACGTACGT"))
        assert decode_seq(s) == "ACGTACGT"

    def test_unknown_base(self) -> None:
        """Unknown bases (value 4) should decode to 'N'."""
        s = Sequence(seq=encode_seq("ANGT"))
        assert decode_seq(s) == "ANGT"


class TestCountTetragraphs:
    def _seq_list(self, seq: str) -> list[Sequence]:
        return [Sequence(seq=encode_seq(seq))]

    def test_short_sequence_no_tetragraphs(self) -> None:
        """Sequences shorter than 4 bases produce no tetragraph counts."""
        sequences = self._seq_list("ACG")
        count_tetragraphs(sequences)
        assert sequences[0].cnts.sum() == 0.0

    def test_exactly_four_bases(self) -> None:
        """A 4-base sequence produces exactly one tetragraph count."""
        sequences = self._seq_list("ACGT")
        count_tetragraphs(sequences)
        assert sequences[0].cnts.sum() == 1.0

    def test_homopolymer(self) -> None:
        """AAAAAAA (7 bases) should produce 4 tetragraphs, all at index 0 (0b00000000)."""
        sequences = self._seq_list("AAAAAAA")
        count_tetragraphs(sequences)
        assert sequences[0].cnts.sum() == 4.0
        assert sequences[0].cnts[0] == 4.0  # AAAA = 0b00000000 = 0

    def test_unknown_char_resets_valid(self) -> None:
        """A non-ACGT character (mapped to al=4) should reset the valid window."""
        sequences = self._seq_list("ACGT")
        # Overwrite the 2nd base with unknown (al=4)
        sequences[0].seq[1] = 4
        count_tetragraphs(sequences)
        assert sequences[0].cnts.sum() == 0.0

    def test_known_tetragraph_value(self) -> None:
        """CCCC should map to tetragraph index 0b01010101 = 85."""
        sequences = self._seq_list("CCCC")
        count_tetragraphs(sequences)
        assert sequences[0].cnts[85] == 1.0
        assert sequences[0].cnts.sum() == 1.0

    def test_acgtaag(self) -> None:
        """ACGTAAG has 4 tetragraphs: ACGT(27), CGTA(108), GTAA(176), TAAG(194)."""
        sequences = self._seq_list("ACGTAAG")
        count_tetragraphs(sequences)
        assert sequences[0].cnts.sum() == 4.0
        assert sequences[0].cnts[27] == 1.0  # ACGT
        assert sequences[0].cnts[108] == 1.0  # CGTA
        assert sequences[0].cnts[176] == 1.0  # GTAA
        assert sequences[0].cnts[194] == 1.0  # TAAG

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


class TestReverseComplement:
    def test_no_reverse(self) -> None:
        """rev=False returns the input array unchanged."""
        c_in = np.array([0, 1, 2, 3], dtype=np.int8)  # A C G T
        result = reverse_complement(c_in, False)
        np.testing.assert_array_equal(result, [0, 1, 2, 3])

    def test_reverse_complement_acgt(self) -> None:
        """rev=True: ACGT -> reverse complement ACGT (T=3->A=0, G=2->C=1, C=1->G=2, A=0->T=3)."""
        c_in = np.array([0, 1, 2, 3], dtype=np.int8)  # A C G T
        result = reverse_complement(c_in, True)
        # Reversed: T G C A -> complement: A(0) C(1) G(2) T(3)
        np.testing.assert_array_equal(result, [0, 1, 2, 3])

    def test_reverse_complement_single(self) -> None:
        """Single base A(0) -> complement T(3)."""
        c_in = np.array([0], dtype=np.int8)
        result = reverse_complement(c_in, True)
        assert result[0] == 3

    def test_reverse_complement_unknown(self) -> None:
        """Unknown base (value 4) maps to 3-4=-1 < 0, so gets set to 4."""
        c_in = np.array([4], dtype=np.int8)
        result = reverse_complement(c_in, True)
        assert result[0] == 4


class TestReverseComplementMap:
    def test_cgac_maps_to_gtcg(self) -> None:
        """CGAC (97) -> reverse complement GTCG (182)."""
        rev_map = reverse_complement_map()
        assert rev_map[97] == 182

    def test_aaaa_maps_to_tttt(self) -> None:
        """AAAA (0) -> TTTT (255)."""
        rev_map = reverse_complement_map()
        assert rev_map[0] == 255

    def test_double_application_is_identity(self) -> None:
        """Applying the map twice returns the original index."""
        rev_map = reverse_complement_map()
        for i in range(256):
            assert rev_map[rev_map[i]] == i

    def test_acgt_maps_to_acgt(self) -> None:
        """ACGT (27) is its own reverse complement."""
        rev_map = reverse_complement_map()
        assert rev_map[27] == 27
