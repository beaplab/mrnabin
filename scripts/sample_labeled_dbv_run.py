import os

from dbv.binner import DBVBinner
from lib.args import get_args
from lib.encoded_sequences import Sequence
from lib.evaluate import bins_by_id, grade_bins
from lib.fasta import get_sequences

DEFAULT_CONFIG = "config/base_dbv_config.yaml"


def main() -> None:
    args = get_args([DEFAULT_CONFIG])
    binner = DBVBinner(**vars(args))

    sequences: list[Sequence] = []
    path = "sample_data"
    data = [
        ("aa_peptides.fasta", "aa_contigs.fasta", "aa"),
        ("bb_peptides.fasta", "bb_contigs.fasta", "bb"),
    ]
    for peptide_file, contig_file, label_prefix in data:
        contig_path = os.path.join(path, contig_file)
        peptide_path = os.path.join(path, peptide_file)
        seqs = get_sequences(contig_path, peptide_path, label_prefix)
        sequences += seqs

    binner.sequences = sequences
    binner.run()

    bins, unbinned = bins_by_id(binner.sequences)
    print(grade_bins(bins, unbinned))


if __name__ == "__main__":
    main()
