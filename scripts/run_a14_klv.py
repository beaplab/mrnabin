from klv.binner import KLVBinner
from lib.a14 import grade_bins_a14, symbiont_bin_details
from lib.args import get_args
from lib.encoded_sequences import Sequence, encode_seq
from lib.evaluate import bins_by_id
from lib.fasta import normalize_sequences

DBV_CONFIG = "config/base_dbv_config.yaml"
KNN_CONFIG = "config/base_knn_config.yaml"
KLV_CONFIG = "config/base_klv_config.yaml"
A14_CONFIG = "config/a14_config.yaml"


def main() -> None:
    args = get_args([DBV_CONFIG, KNN_CONFIG, KLV_CONFIG, A14_CONFIG])
    sequences = [
        Sequence(seq_id=label, seq=encode_seq(seq))
        for label, seq in normalize_sequences(args.peptide_path, args.mrna_path)
    ]

    binner = KLVBinner(**vars(args))
    binner.sequences = sequences
    binner.run()

    bins, unbinned = bins_by_id(binner.sequences)
    grade_str, best_bin = grade_bins_a14(args.comp_path, args.match_path, bins, unbinned)
    print(grade_str)

    if args.deets:
        deets = symbiont_bin_details(args.homology_path, args.comp_path, bins[best_bin])
        print(f"\nDetails for Bin {best_bin}")
        print("#protein\tsymbiont_pct\tsymbiont_match_len\teuk_prot_match\teuk_prot_score")
        for d in deets:
            print(f"{d[0]}\t{d[1]}\t{d[2]}\t{d[3]}\t{d[4]}\t")


if __name__ == "__main__":
    main()
