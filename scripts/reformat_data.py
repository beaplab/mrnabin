import argparse
import os

from lib.fasta import normalize_sequences


def main(args: argparse.Namespace) -> None:
    protein_files = sorted(os.listdir(args.protein_path))
    protein_file = os.path.join(args.protein_path, protein_files[args.org_id])

    contig_files = sorted(os.listdir(args.contig_path))
    contig_file = os.path.join(args.contig_path, contig_files[args.org_id])

    label_prefix = contig_files[args.org_id][:3]  # should be dd_ for d in 0-9
    for label, sequence in normalize_sequences(protein_file, contig_file, label_prefix):
        print(label, sequence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("contig_path", type=str, help="directory of fasta files for contigs")
    parser.add_argument("protein_path", type=str, help="directory of fasta files for proteins")
    parser.add_argument("org_id", type=int, help="fasta file number to process")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
