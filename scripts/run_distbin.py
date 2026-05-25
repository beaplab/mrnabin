import argparse

from distbin.binner import Binner
from lib.args import get_args
from lib.evaluate import dump_bins
from lib.fasta import normalize_sequences

DEFAULT_CONFIG = "config/base_distbin_config.yaml"


def main(args: argparse.Namespace) -> None:
    sequences = {l: s for l, s in normalize_sequences(args.peptide_file, args.contig_file)}
    binner = Binner(sequences, **vars(args))
    bins, unbinned = binner.run()
    print(f"Bin sizes: {[len(b) for b in bins]} ; Unbinned: {len(unbinned)}")
    with open(args.output_file, "w") as f:
        for contig, bin_label in dump_bins(bins, unbinned):
            print("\t".join([contig, bin_label]), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("contig_file", type=str)
    parser.add_argument("peptide_file", type=str)
    parser.add_argument("output_file", type=str)
    args = get_args([DEFAULT_CONFIG], parser)
    main(args)
