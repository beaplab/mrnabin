import os

from lib.args import get_args
from lib.encoded_sequences import Sequence, read_mrna_file_encoded
from lib.evaluate import bins_by_id, grade_bins
from vlp.binner import VLPBinner

DEFAULT_CONFIG = "config/base_vlp_config.yaml"
DATA_PATH = "data/5686984/m_choanoflagellate_transcriptomes.mrnabin"


def main() -> None:
    args = get_args([DEFAULT_CONFIG])
    orgs = sorted(os.listdir(DATA_PATH))
    for i in range(len(orgs)):
        for j in range(len(orgs)):
            if i >= j:
                continue

            sequences: list[Sequence] = []
            for org_id in [i, j]:
                sequences += read_mrna_file_encoded(os.path.join(DATA_PATH, orgs[org_id]))

            binner = VLPBinner(**vars(args))
            binner.sequences = sequences
            binner.run()

            bins, unbinned = bins_by_id(binner.sequences)
            print(i, j, grade_bins(bins, unbinned), flush=True)


if __name__ == "__main__":
    main()
