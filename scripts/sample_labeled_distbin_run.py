import os

from distbin.binner import Binner
from lib.evaluate import grade_bins
from lib.fasta import normalize_sequences


def main() -> None:
    path = "sample_data"
    data = [
        ("aa_peptides.fasta", "aa_contigs.fasta", "aa"),
        ("bb_peptides.fasta", "bb_contigs.fasta", "bb"),
    ]

    sequences = {}
    for peptide_file, contig_file, label_prefix in data:
        peptide_path = os.path.join(path, peptide_file)
        contig_path = os.path.join(path, contig_file)
        for label, sequence in normalize_sequences(peptide_path, contig_path, label_prefix):
            sequences[label] = sequence

    binner = Binner(sequences, verbose=True, compact=True, labeled=True)
    bins, unbinned = binner.run()
    print(grade_bins(bins, unbinned))


if __name__ == "__main__":
    main()
