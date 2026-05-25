import argparse

from lib.args import get_args
from lib.evaluate import bins_by_id, dump_bins
from lib.fasta import get_sequences
from vlp.binner import VLPBinner

DEFAULT_CONFIG = "config/base_vlp_config.yaml"


def main(args: argparse.Namespace) -> None:
    binner = VLPBinner(**vars(args))
    sequences = get_sequences(args.contig_file, args.peptide_file)
    binner.sequences = sequences
    binner.run()
    bins, unbinned = bins_by_id(binner.sequences)
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
