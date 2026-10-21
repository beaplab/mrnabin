import os

# Pin each worker's math libraries to a single thread BEFORE numpy/numba are
# imported. Otherwise every process spins up its own BLAS/OpenMP/numba thread
# pool and N processes x C cores oversubscribes the machine into thrashing.
# VECLIB_MAXIMUM_THREADS is the one that matters on macOS: numpy links Apple's
# Accelerate (vecLib) BLAS here, which ignores the OMP/OPENBLAS/MKL vars.
for _v in (
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

from dbv.binner import DBVBinner  # noqa: E402
from lib.args import get_args  # noqa: E402
from lib.evaluate import bins_by_id, grade_bins  # noqa: E402
from lib.fasta import get_sequences  # noqa: E402

DISTBIN_CONFIG = "config/base_distbin_config.yaml"
DBV_CONFIG = "config/base_dbv_config.yaml"
PREFIX_LEN = 7
transcriptomes_path = "data/12417881/assembled_transcriptomes"
protein_path = "data/12417881/proteins"

ORGS = sorted(os.listdir(transcriptomes_path))


def _load_org(org_id: int) -> list:
    name = ORGS[org_id]
    contig_file = os.path.join(transcriptomes_path, name)
    peptide_file = os.path.join(protein_path, name)
    return get_sequences(contig_file, peptide_file, prefix=name[:PREFIX_LEN], label_col=1)


def _run_pair(task: tuple) -> tuple:
    args_dict, i, j = task
    # Reload per task: the binner mutates Sequence objects (.cluster/.active),
    # so each pair needs its own fresh copies — sharing would corrupt results.
    sequences = _load_org(i) + _load_org(j)
    binner = DBVBinner(**args_dict)
    binner.sequences = sequences
    binner.run()
    bins, unbinned = bins_by_id(binner.sequences)
    return i, j, grade_bins(bins, unbinned, prefix_len=PREFIX_LEN)


def main() -> None:
    args = get_args([DISTBIN_CONFIG, DBV_CONFIG])
    args.prefix_len = PREFIX_LEN
    args.verbose = False  # interleaved per-pair logs are useless in parallel
    args_dict = vars(args)

    pairs = [(args_dict, i, j) for i in range(len(ORGS)) for j in range(i + 1, len(ORGS))]

    workers = int(os.environ.get("DBV_WORKERS", os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_run_pair, p) for p in pairs]
        for fut in as_completed(futures):
            i, j, grade = fut.result()
            print(i, j, grade, flush=True)


if __name__ == "__main__":
    main()
