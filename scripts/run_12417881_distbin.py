import os

from distbin.binner import Binner
from lib.args import get_args
from lib.evaluate import grade_bins
from lib.fasta import normalize_sequences

DEFAULT_CONFIG = "config/base_distbin_config.yaml"
PREFIX_LEN = 7
transcriptomes_path = "data/12417881/assembled_transcriptomes"
protein_path = "data/12417881/proteins"


def main() -> None:
    args = get_args([DEFAULT_CONFIG])
    args.prefix_len = PREFIX_LEN
    orgs = sorted(os.listdir(transcriptomes_path))
    for i in range(len(orgs)):
        for j in range(len(orgs)):
            if i >= j:
                continue
            sequences = {}
            for org_id in [i, j]:
                contig_file = os.path.join(transcriptomes_path, orgs[org_id])
                peptide_file = os.path.join(protein_path, orgs[org_id])
                sequences.update(
                    {
                        l: s
                        for l, s in normalize_sequences(
                            peptide_file, contig_file, orgs[org_id][:PREFIX_LEN], label_col=1
                        )
                    }
                )
            binner = Binner(sequences, **vars(args))
            bins, unbinned = binner.run()
            print(i, j, grade_bins(bins, unbinned, prefix_len=PREFIX_LEN), flush=True)


if __name__ == "__main__":
    main()
