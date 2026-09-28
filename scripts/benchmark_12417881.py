"""Run KNN or DBV over a fixed random sample of EukProt mixtures and keep the bins.

The hand-picked pairs in knn/HISTORY.md were chosen from DBV's weak tail, so they
say nothing about typical mixtures.  This draws a seeded sample of 2- and
3-species mixtures instead, runs one algorithm and config on every mixture in
parallel, and writes each mixture's bins to `runs/bench/<name>/<mixture>.tsv` so
`bench_summary.py` can score them any way later without rerunning.

    uv run scripts/benchmark_12417881.py knn default
    uv run scripts/benchmark_12417881.py knn coarse --set resolution=0.1 n_neighbors=1024
    uv run scripts/benchmark_12417881.py dbv dbv --workers 12

`--set` values are parsed as YAML, so lists and floats work: `length_stages=[75,50,25,0]`.
"""

import os

# Pin math libraries to one thread per worker before numpy is imported;
# see run_parallel_12417881_dbv.py.
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from typing import Any  # noqa: E402

import yaml  # noqa: E402

from lib.evaluate import budget_scores, dump_bins  # noqa: E402

PREFIX_LEN = 7
TRANSCRIPTOMES = "data/12417881/assembled_transcriptomes"
PROTEINS = "data/12417881/proteins"
OUT_ROOT = "runs/bench"
CONFIGS = {
    "knn": ["config/base_knn_config.yaml"],
    "dbv": ["config/base_distbin_config.yaml", "config/base_dbv_config.yaml"],
    # KNN seeds, then DBV's VLP phase; DBV's keys do not collide with KNN's
    "knnv": ["config/base_dbv_config.yaml", "config/base_knn_config.yaml"],
}
# the EukProt pairs HISTORY.md measured on, kept as a separate hard set
HARD = [
    ("EP00463", "EP00495"),
    ("EP00244", "EP00245"),
    ("EP00245", "EP00246"),
    ("EP00178", "EP00657"),
    ("EP00252", "EP00262"),
    ("EP00002", "EP00271"),
    ("EP00002", "EP00178"),
]


def sample_mixtures(seed: int, n_pairs: int, n_triples: int) -> list[tuple[str, ...]]:
    orgs = [o[:PREFIX_LEN] for o in sorted(os.listdir(TRANSCRIPTOMES))]
    rng = random.Random(seed)
    mixtures: list[tuple[str, ...]] = []
    for size, count in ((2, n_pairs), (3, n_triples)):
        seen: set[tuple[str, ...]] = set()
        while len(seen) < count:
            seen.add(tuple(sorted(rng.sample(orgs, size))))
        mixtures += sorted(seen)
    return mixtures


def mixture_id(mixture: tuple[str, ...]) -> str:
    return "+".join(mixture)


def _files(org: str) -> tuple[str, str]:
    (name,) = [o for o in os.listdir(TRANSCRIPTOMES) if o.startswith(org)]
    return os.path.join(TRANSCRIPTOMES, name), os.path.join(PROTEINS, name)


def _run_knn(mixture: tuple[str, ...], params: dict[str, Any]) -> tuple[list[set[str]], set[str]]:
    from knn.binner import Binner
    from lib.fasta import normalize_sequences

    sequences: dict[str, str] = {}
    for org in mixture:
        contigs, peptides = _files(org)
        sequences.update(dict(normalize_sequences(peptides, contigs, org, label_col=1)))
    return Binner(sequences, **params).run()


def _run_dbv(
    mixture: tuple[str, ...], params: dict[str, Any], knn_seeded: bool = False
) -> tuple[list[set[str]], set[str]]:
    from dbv.binner import DBVBinner
    from knn.vlp_merge import KNNVLPBinner
    from lib.evaluate import bins_by_id
    from lib.fasta import get_sequences

    sequences = []
    for org in mixture:
        contigs, peptides = _files(org)
        sequences += get_sequences(contigs, peptides, prefix=org, label_col=1)
    binner = KNNVLPBinner(**params) if knn_seeded else DBVBinner(**params)
    binner.sequences = sequences
    binner.run()
    return bins_by_id(binner.sequences)


def _run_one(algo: str, mixture: tuple[str, ...], params: dict[str, Any], out_dir: str) -> str:
    start = time.time()
    if algo == "knn":
        bins, unbinned = _run_knn(mixture, params)
    else:
        bins, unbinned = _run_dbv(mixture, params, knn_seeded=algo == "knnv")
    seconds = time.time() - start
    with open(os.path.join(out_dir, mixture_id(mixture) + ".tsv"), "w") as f:
        f.write(f"#seconds\t{seconds:.1f}\n")
        for contig, label in dump_bins(sorted(bins, key=len, reverse=True), unbinned):
            f.write(f"{contig}\t{label}\n")
    p16, r16 = budget_scores(bins, unbinned, (16,), PREFIX_LEN)[16]
    return f"{mixture_id(mixture)} bins:{len(bins)} p@16:{p16:.3f} r@16:{r16:.3f} {seconds:.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("algo", choices=sorted(CONFIGS))
    parser.add_argument("name", help="output directory under runs/bench")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides, key=value")
    parser.add_argument("-c", "--config", default=None, help="extra YAML applied over the base config")
    parser.add_argument("--sets", nargs="+", default=["random"], choices=["random", "hard"])
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--triples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=[], help="run only these mixture ids")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    params: dict[str, Any] = {}
    for path in CONFIGS[args.algo] + ([args.config] if args.config else []):
        with open(path) as f:
            params.update(yaml.safe_load(f) or {})
    for item in args.set:
        key, value = item.split("=", 1)
        params[key] = yaml.safe_load(value)
    params.update(verbose=False, compact=False, labeled=False, prefix_len=PREFIX_LEN)

    mixtures: list[tuple[str, ...]] = []
    if "random" in args.sets:
        mixtures += sample_mixtures(args.seed, args.pairs, args.triples)
    if "hard" in args.sets:
        mixtures += HARD
    if args.only:
        mixtures = [m for m in mixtures if mixture_id(m) in args.only]

    out_dir = os.path.join(OUT_ROOT, args.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "params.yaml"), "w") as f:
        yaml.safe_dump({"algo": args.algo, **params}, f)

    todo = [m for m in mixtures if not os.path.exists(os.path.join(out_dir, mixture_id(m) + ".tsv"))]
    print(f"{len(todo)} of {len(mixtures)} mixtures to run -> {out_dir}", flush=True)
    # largest mixtures first so a slow one does not start last
    todo.sort(key=len, reverse=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_one, args.algo, m, params, out_dir): m for m in todo}
        for fut in as_completed(futures):
            try:
                print(fut.result(), flush=True)
            # leidenalg raises bare BaseException; keep the sweep going, the missing .tsv marks the failure
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt):
                    raise
                print(f"{mixture_id(futures[fut])} FAILED {e!r}", flush=True)


if __name__ == "__main__":
    main()
