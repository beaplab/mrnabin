from unittest.mock import MagicMock, mock_open, patch

import pytest

from lib.fasta import normalize_sequences, process_contigs, process_peptides, read_mrna_file

mock_peptides = (
    ">DATA DATA DATA ... DATA DATA DATA id1:0-100(-)\n"
    ">DATA DATA DATA ... DATA DATA DATA id1:101-120(+)\n"
    ">DATA DATA DATA ... DATA DATA DATA id2:11-12(+)\n"
)

mock_contigs = (
    ">id1 DATA DATA\n"
    "GAAACAGAAT\n"
    "CAGCATACTC\n"
    "AAAGGTTCGG\n"
    ">id2 DATA DATA\n"
    "AACAGCAACA\n"
    "ACAACAACAA\n"
    ">id3 DATA DATA\n"
    "TCTGTTGACA\n"
    "AAGTAAGCAG\n"
)


def test_process_peptides() -> None:
    with patch("lib.fasta.open", mock_open(read_data=mock_peptides)) as m:
        orientations = process_peptides("mock_path")
        assert len(orientations) == 2
        assert orientations["id1"] is False
        assert orientations["id2"] is True
    m.assert_called_with("mock_path")


def test_process_contigs() -> None:
    with patch("lib.fasta.open", mock_open(read_data=mock_contigs)) as m:
        contigs = list(process_contigs("mock_path", 0))
        assert len(contigs) == 3
        assert contigs[0][0] == "id1"
        assert len(contigs[0][1]) == 30
        assert contigs[1][0] == "id2"
        assert len(contigs[1][1]) == 20
    m.assert_called_with("mock_path")


@patch("lib.fasta.process_peptides", return_value={"id1": False, "id2": True})
def test_normalize_sequences(mock_process_peptides: MagicMock) -> None:
    print(type(mock_process_peptides))
    with patch("lib.fasta.open", mock_open(read_data=mock_contigs)) as m:
        sequences = list(normalize_sequences("mock_path_1", "mock_path_2", "AA"))
        assert sequences[0][0] == "AAid1"
        assert sequences[1][0] == "AAid2"
        assert sequences[0][1] == "CCGAACCTTTGAGTATGCTGATTCTGTTTC"
        assert sequences[1][1] == "AACAGCAACAACAACAACAA"
    m.assert_called_with("mock_path_2")


def test_process_peptides_malformed_header_raises() -> None:
    with patch("lib.fasta.open", mock_open(read_data=">this header has no coords\n")):
        with pytest.raises(ValueError, match="malformed peptide header"):
            process_peptides("mock_path")


def test_read_mrna_file() -> None:
    data = "id1\tACGT\nid2\tTTTT\n"
    with patch("lib.fasta.open", mock_open(read_data=data)):
        seqs = read_mrna_file("mock_path")
    assert seqs == {"id1": "ACGT", "id2": "TTTT"}


def test_read_mrna_file_duplicate_warns_and_keeps_first(capsys: pytest.CaptureFixture[str]) -> None:
    data = "id1\tACGT\nid1\tGGGG\n"
    with patch("lib.fasta.open", mock_open(read_data=data)):
        seqs = read_mrna_file("mock_path")
    assert seqs == {"id1": "ACGT"}
    assert "duplicate label" in capsys.readouterr().err
