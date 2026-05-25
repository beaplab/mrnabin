import os

from distbin.binner import Binner
from lib.args import get_args
from lib.evaluate import grade_bins
from lib.fasta import read_mrna_file

DEFAULT_CONFIG = "config/base_distbin_config.yaml"
DATA_PATH = "data/5686984/m_choanoflagellate_transcriptomes.mrnabin"


def main() -> None:
    args = get_args([DEFAULT_CONFIG])
    orgs = sorted(os.listdir(DATA_PATH))
    for i in range(len(orgs)):
        for j in range(len(orgs)):
            if i >= j:
                continue
            sequences = {}
            for org_id in [i, j]:
                sequences.update(read_mrna_file(os.path.join(DATA_PATH, orgs[org_id])))

            binner = Binner(sequences, **vars(args))

            bins, unbinned = binner.run()
            print(i, j, grade_bins(bins, unbinned), flush=True)


if __name__ == "__main__":
    main()
