import sys
from typing import Generator

from distbin.kmers import reverse_complement
from lib.encoded_sequences import Sequence, encode_seq


def process_peptides(file_path: str) -> dict[str, bool]:
    """Process peptide FASTA file and gather orientation information.

    If the same identifier has multiple entries, we prefer the
    orientation of the longest.

    files are assumed to have headers with the following format:

    >DATA DATA DATA ... DATA DATA DATA identifier:start-end(orientation)

    * identifier: str
    * start, end: int, int
    * orientation: +/-

    identifiers should match the identifiers in the contig files
    """
    seq_info: dict[str, tuple[int, str]] = {}
    with open(file_path) as f:
        for line in f:
            if line[0] == ">":
                try:
                    identifier, info = line.strip().split()[-1].split(":")
                    positions, orientation = info.split("(")
                    orientation = orientation.strip(")")
                    start, end = positions.split("-")
                    length = int(end) - int(start)
                except (ValueError, IndexError) as e:
                    raise ValueError(f"malformed peptide header: {line.strip()!r}") from e
                if identifier in seq_info:
                    prev_length = seq_info[identifier][0]
                    if prev_length > length:
                        continue
                seq_info[identifier] = (length, orientation)
    return {k: v[1] == "+" for k, v in seq_info.items()}


def process_contigs(path: str, label_col: int) -> Generator[tuple[str, str], None, None]:
    """Process mRNA FASTA file and yield labeled sequences

    mRNA files are assumed to have the following format:

    >identifier DATA DATA ...
    CONTIG_LINE_1
    CONTIG_LINE_2
    ...
    >identifier DATA DATA ...
    CONTIG_LINE_1
    CONTIG_LINE_2
    ...
    """
    with open(path) as f:
        label = ""
        seq_parts: list[str] = []
        for line in f:
            line = line.strip()
            if not line or line[0] == ";":
                continue
            if line[0] == ">":
                if label and seq_parts:
                    yield label, "".join(seq_parts)
                label = line.split()[label_col].lstrip(">")
                seq_parts = []
            else:
                seq_parts.append(line)
        if label and seq_parts:
            yield label, "".join(seq_parts)


def normalize_sequences(
    peptide_path: str,
    contig_path: str,
    prefix: str | None = None,
    label_col: int = 0,
) -> Generator[tuple[str, str], None, None]:
    seq_orientation = process_peptides(peptide_path)
    kept, skipped = 0, 0
    for label, sequence in process_contigs(contig_path, label_col):
        if label not in seq_orientation:
            skipped += 1
            continue
        if not seq_orientation[label]:
            sequence = reverse_complement(sequence)
        if prefix:
            label = prefix + label
        kept += 1
        yield (label, sequence)
    msg = f"normalize_sequences({contig_path}): {kept} contig(s) with orientation, {skipped} skipped"
    if skipped:
        msg += f" (no peptide orientation in {peptide_path})"
    print(msg, file=sys.stderr)


def read_mrna_file(path: str) -> dict[str, str]:
    seqs: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            label, sequence = line.strip().split()
            if label in seqs:
                print(f"warning: duplicate label in {path}: {label!r}, keeping first", file=sys.stderr)
                continue
            seqs[label] = sequence
    return seqs


def _read_seq_file(data_path: str, label_col: int) -> list[Sequence]:
    # seq_id capped at 25 chars to keep display output readable downstream
    return [
        Sequence(seq_id=label[:25], seq=encode_seq(seq_str)) for label, seq_str in process_contigs(data_path, label_col)
    ]


def _orient_and_tag_sequences(
    peptide_path: str, sequences: list[Sequence], prefix: str | None = None
) -> list[Sequence]:
    seq_orientation = process_peptides(peptide_path)
    oriented = []
    for sequence in sequences:
        if sequence.seq_id not in seq_orientation:
            continue
        if not seq_orientation[sequence.seq_id]:
            sequence.reverse = True
        if prefix:
            sequence.seq_id = prefix + sequence.seq_id
        oriented.append(sequence)
    return oriented


def get_sequences(contig_path: str, peptide_path: str, prefix: str | None = None, label_col: int = 0) -> list[Sequence]:
    sequences = _read_seq_file(contig_path, label_col)
    return _orient_and_tag_sequences(peptide_path, sequences, prefix)
