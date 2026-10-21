import argparse

from knn.binner import Binner
from lib.args import get_args
from lib.evaluate import dump_bins
from lib.fasta import normalize_sequences

KNN_CONFIG = "config/base_knn_config.yaml"


def main(args: argparse.Namespace) -> None:
    sequences = dict(normalize_sequences(args.peptide_file, args.contig_file))
    bins, unbinned = Binner(sequences, **vars(args)).run()
    print(f"Bin sizes: {[len(b) for b in bins]} ; Unbinned: {len(unbinned)}")
    with open(args.output_file, "w") as f:
        for contig, bin_label in dump_bins(bins, unbinned):
            print("\t".join([contig, bin_label]), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("contig_file", type=str)
    parser.add_argument("peptide_file", type=str)
    parser.add_argument("output_file", type=str)
    args = get_args([KNN_CONFIG], parser)
    main(args)
