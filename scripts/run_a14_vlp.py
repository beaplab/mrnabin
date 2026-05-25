from lib.a14 import grade_bins_a14
from lib.args import get_args
from lib.encoded_sequences import Sequence, encode_seq
from lib.evaluate import bins_by_id
from lib.fasta import normalize_sequences
from vlp.binner import VLPBinner

VLP_CONFIG = "config/base_vlp_config.yaml"
A14_CONFIG = "config/a14_config.yaml"


def main() -> None:
    args = get_args([VLP_CONFIG, A14_CONFIG])

    sequences = [
        Sequence(seq_id=label, seq=encode_seq(seq))
        for label, seq in normalize_sequences(args.peptide_path, args.mrna_path)
    ]

    binner = VLPBinner(**vars(args))
    binner.sequences = sequences
    binner.run()

    bins, unbinned = bins_by_id(binner.sequences)
    print(grade_bins_a14(args.comp_path, args.match_path, bins, unbinned))


if __name__ == "__main__":
    main()
