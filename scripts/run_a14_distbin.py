from distbin.binner import Binner
from lib.a14 import grade_bins_a14
from lib.args import get_args
from lib.fasta import normalize_sequences

DISTBIN_CONFIG = "config/base_distbin_config.yaml"
A14_CONFIG = "config/a14_config.yaml"


def main() -> None:
    args = get_args([DISTBIN_CONFIG, A14_CONFIG])

    sequences = {l: s for l, s in normalize_sequences(args.peptide_path, args.mrna_path)}
    binner = Binner(
        sequences,
        k=args.k,
        verbose=args.verbose,
        selection_percentiles=(args.selection_percentiles_low, args.selection_percentiles_high),
        creation_threshold=args.creation_threshold,
        selection_threshold=args.selection_threshold,
        cluster_thresholds=[float(t) for t in args.cluster_thresholds.split(",")],
        reduction_constant=args.reduction_constant,
        final_threshold=args.final_threshold,
    )
    bins, unbinned = binner.run()
    print(grade_bins_a14(args.comp_path, args.match_path, bins, unbinned))


if __name__ == "__main__":
    main()
