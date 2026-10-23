import os

from knn.binner import Binner
from lib.args import get_args
from lib.evaluate import grade_bins
from lib.fasta import normalize_sequences

DEFAULT_CONFIG = "config/base_knn_config.yaml"


def main() -> None:
    args = get_args([DEFAULT_CONFIG])

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

    binner = Binner(sequences, **vars(args))
    bins, unbinned = binner.run()
    print(grade_bins(bins, unbinned))


if __name__ == "__main__":
    main()
