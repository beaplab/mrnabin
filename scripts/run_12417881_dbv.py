import os

from dbv.binner import DBVBinner
from lib.args import get_args
from lib.evaluate import bins_by_id, grade_bins
from lib.fasta import get_sequences

DISTBIN_CONFIG = "config/base_distbin_config.yaml"
DBV_CONFIG = "config/base_dbv_config.yaml"
PREFIX_LEN = 7
transcriptomes_path = "data/12417881/assembled_transcriptomes"
protein_path = "data/12417881/proteins"


def main() -> None:
    args = get_args([DISTBIN_CONFIG, DBV_CONFIG])
    args.prefix_len = PREFIX_LEN
    orgs = sorted(os.listdir(transcriptomes_path))
    for i in range(len(orgs)):
        for j in range(len(orgs)):
            if i >= j:
                continue
            sequences = []
            for org_id in [i, j]:
                contig_file = os.path.join(transcriptomes_path, orgs[org_id])
                peptide_file = os.path.join(protein_path, orgs[org_id])
                sequences += get_sequences(contig_file, peptide_file, prefix=orgs[org_id][:PREFIX_LEN], label_col=1)
            binner = DBVBinner(**vars(args))
            binner.sequences = sequences
            binner.run()
            bins, unbinned = bins_by_id(binner.sequences)
            print(i, j, grade_bins(bins, unbinned, prefix_len=PREFIX_LEN), flush=True)


if __name__ == "__main__":
    main()
