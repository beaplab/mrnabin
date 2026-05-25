import os
import random

from dbv.binner import DBVBinner
from lib.args import get_args
from lib.encoded_sequences import Sequence, read_mrna_file_encoded
from lib.evaluate import bins_by_id, grade_bins

DISTBIN_CONFIG = "config/base_distbin_config.yaml"
DBV_CONFIG = "config/base_dbv_config.yaml"
DATA_PATH = "data/5686984/m_choanoflagellate_transcriptomes.mrnabin"


def main() -> None:
    args = get_args([DISTBIN_CONFIG, DBV_CONFIG])
    orgs = sorted(os.listdir(DATA_PATH))
    for i in range(len(orgs)):
        for j in range(len(orgs)):
            if i >= j:
                continue
            sequences: list[Sequence] = []
            for org_id in [i, j]:
                sequences += read_mrna_file_encoded(os.path.join(DATA_PATH, orgs[org_id]))
            random.shuffle(sequences)
            binner = DBVBinner(**vars(args))
            binner.sequences = sequences
            binner.run()
            bins, unbinned = bins_by_id(binner.sequences)
            print(i, j, grade_bins(bins, unbinned), flush=True)


if __name__ == "__main__":
    main()
