"""Run any binner on any dataset.

    uv run scripts/run_binner.py ALGO DATASET [DATASET ARGS] [BINNER FLAGS]

ALGO is one of distbin, knn, dbv, vlp, klv. DATASET is one of:

    file CONTIGS PEPTIDES OUTPUT   bin one transcriptome, write a contig -> bin TSV
    a14                            the A14 symbiont data (paths in config/a14_config.yaml)
    5686984 [A B]                  choanoflagellate mixtures
    12417881 [A B]                 EukProt mixtures

The last two are labeled mixtures of two organisms: with no A B every pair is
swept, printing `i j grade` per pair, or with --names the codes the two
filenames start with (EP00002, 07) in place of i j; A B picks one pair, by
index into the sorted organism list, EukProt id, or any unique part of a
filename. --parallel [N] spreads a sweep over N processes (10 if N is omitted).
After the grade, each pair's line carries scores that fragmenting into many
small bins cannot inflate (see lib.evaluate.fragment_scores).

    uv run scripts/run_binner.py klv 5686984 --parallel
    uv run scripts/run_binner.py knn 12417881 EP00178 EP00657 --verbose --resolution_min 0.02
    uv run scripts/run_binner.py dbv a14 --deets
    uv run scripts/run_binner.py distbin file contigs.fa peptides.fa bins.tsv

Sequences are shuffled before binning unless --no-shuffle is given. --seed
(default 0) fixes the shuffle, and knn's own seed for knn and klv, so runs are
reproducible.
"""

import argparse
import multiprocessing
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from dbv.binner import DBVBinner
from distbin.binner import Binner as DistBinner
from klv.binner import KLVBinner
from knn.binner import Binner as KNNBinner
from lib.a14 import grade_bins_a14, symbiont_bin_details
from lib.args import get_args
from lib.encoded_sequences import Sequence, encode_seq, read_mrna_file_encoded
from lib.evaluate import bins_by_id, dump_bins, fragment_scores, grade_bins
from lib.fasta import get_sequences, normalize_sequences, read_mrna_file
from vlp.binner import VLPBinner

# Either a {label: sequence} dict (distbin, knn) or encoded Sequences (dbv, vlp, klv).
Seqs = dict[str, str] | list[Sequence]
Bins = tuple[list[set[str]], set[str]]

DISTBIN_CONFIG = "config/base_distbin_config.yaml"
DBV_CONFIG = "config/base_dbv_config.yaml"
KNN_CONFIG = "config/base_knn_config.yaml"
KLV_CONFIG = "config/base_klv_config.yaml"
VLP_CONFIG = "config/base_vlp_config.yaml"
A14_CONFIG = "config/a14_config.yaml"

DEFAULT_WORKERS = 10

# Pinned to one thread per worker in --parallel runs: otherwise every process
# spins up its own BLAS/OpenMP/numba thread pool and N processes x C cores
# oversubscribes the machine into thrashing. VECLIB_MAXIMUM_THREADS is the one
# that matters on macOS, where numpy links Apple's Accelerate BLAS.
THREAD_VARS = (
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


@dataclass(frozen=True)
class Algo:
    configs: list[str]
    encoded: bool  # takes list[Sequence] rather than a {label: sequence} dict
    binner: Callable[..., Any]


ALGOS = {
    "distbin": Algo([DISTBIN_CONFIG], False, DistBinner),
    "knn": Algo([KNN_CONFIG], False, KNNBinner),
    "dbv": Algo([DISTBIN_CONFIG, DBV_CONFIG], True, DBVBinner),
    "vlp": Algo([VLP_CONFIG], True, VLPBinner),
    "klv": Algo([DBV_CONFIG, KNN_CONFIG, KLV_CONFIG], True, KLVBinner),
}


def run_binner(algo: Algo, sequences: Seqs, params: dict[str, Any]) -> Bins:
    if algo.encoded:
        binner = algo.binner(**params)
        binner.sequences = sequences
        binner.run()
        return bins_by_id(binner.sequences)
    bins, unbinned = algo.binner(sequences, **params).run()
    return bins, unbinned


def shuffled(sequences: Seqs, seed: int) -> Seqs:
    rng = random.Random(seed)
    if isinstance(sequences, list):
        return rng.sample(sequences, len(sequences))
    return dict(rng.sample(list(sequences.items()), len(sequences)))


def merge(a: Seqs, b: Seqs) -> Seqs:
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    if isinstance(a, dict) and isinstance(b, dict):
        return a | b
    raise TypeError("cannot merge a dict with a list of Sequences")


# --- unlabeled datasets -------------------------------------------------------


def load_fasta(contig_file: str, peptide_file: str, encoded: bool) -> Seqs:
    if encoded:
        return get_sequences(contig_file, peptide_file)
    return dict(normalize_sequences(peptide_file, contig_file))


def load_a14(mrna_path: str, peptide_path: str, encoded: bool) -> Seqs:
    pairs = normalize_sequences(peptide_path, mrna_path)
    if encoded:
        return [Sequence(seq_id=label, seq=encode_seq(seq)) for label, seq in pairs]
    return dict(pairs)


# --- labeled two-organism mixtures --------------------------------------------

MRNABIN_PATH = "data/5686984/m_choanoflagellate_transcriptomes.mrnabin"
EUKPROT_CONTIGS = "data/12417881/assembled_transcriptomes"
EUKPROT_PEPTIDES = "data/12417881/proteins"
EUKPROT_PREFIX_LEN = 7


def load_5686984(org: str, encoded: bool) -> Seqs:
    path = os.path.join(MRNABIN_PATH, org)
    return read_mrna_file_encoded(path) if encoded else read_mrna_file(path)


def load_12417881(org: str, encoded: bool) -> Seqs:
    contigs = os.path.join(EUKPROT_CONTIGS, org)
    peptides = os.path.join(EUKPROT_PEPTIDES, org)
    prefix = org[:EUKPROT_PREFIX_LEN]
    if encoded:
        return get_sequences(contigs, peptides, prefix=prefix, label_col=1)
    return dict(normalize_sequences(peptides, contigs, prefix, label_col=1))


@dataclass(frozen=True)
class Mixtures:
    list_orgs: Callable[[], list[str]]
    load: Callable[[str, bool], Seqs]
    prefix_len: int = 2


MIXTURES = {
    "5686984": Mixtures(lambda: sorted(os.listdir(MRNABIN_PATH)), load_5686984),
    "12417881": Mixtures(lambda: sorted(os.listdir(EUKPROT_CONTIGS)), load_12417881, EUKPROT_PREFIX_LEN),
}

DATASETS = ["file", "a14", *MIXTURES]


def org_name(org: str) -> str:
    """The code an organism's filename starts with, e.g. EP00002 or 07."""
    return org.split("_")[0]


def resolve_org(orgs: list[str], spec: str) -> int:
    """Map an index, an EukProt id, or a filename fragment to an index into orgs."""
    if spec.isdigit() and int(spec) < len(orgs):
        return int(spec)
    matches = [i for i, o in enumerate(orgs) if spec in o]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"no organism matching {spec!r}")
    raise SystemExit(f"{spec!r} matches several organisms: {', '.join(orgs[i] for i in matches)}")


def run_pair(task: tuple[str, str, dict[str, Any], int, int, str, str]) -> tuple[int, int, str]:
    algo_name, dataset, params, i, j, org_a, org_b = task
    algo, mix = ALGOS[algo_name], MIXTURES[dataset]
    # Reload per pair: the encoded binners mutate Sequence objects, so pairs
    # can't share them.
    sequences = merge(mix.load(org_a, algo.encoded), mix.load(org_b, algo.encoded))
    if params["shuffle"]:
        sequences = shuffled(sequences, params["seed"])
    bins, unbinned = run_binner(algo, sequences, params)
    grade = grade_bins(bins, unbinned, prefix_len=mix.prefix_len)
    return i, j, f"{grade} | {fragment_scores(bins, unbinned, prefix_len=mix.prefix_len)}"


def run_mixtures(args: argparse.Namespace) -> None:
    mix = MIXTURES[args.dataset]
    orgs = mix.list_orgs()
    args.prefix_len = mix.prefix_len
    args.labeled = True

    if args.org_a is not None:
        if args.org_b is None:
            raise SystemExit("give two organisms, or none to sweep every pair")
        i, j = sorted([resolve_org(orgs, args.org_a), resolve_org(orgs, args.org_b)])
        pairs = [(i, j)]
        print("Mixing:", " + ".join(orgs[i] for i in pairs[0]), file=sys.stderr, flush=True)
    else:
        pairs = [(i, j) for i in range(len(orgs)) for j in range(i + 1, len(orgs))]

    if args.parallel:
        args.verbose = False  # interleaved per-pair logs are useless in parallel
    tasks = [(args.algo, args.dataset, vars(args), i, j, orgs[i], orgs[j]) for i, j in pairs]

    def show(i: int, j: int, grade: str) -> None:
        print(*((org_name(orgs[i]), org_name(orgs[j])) if args.names else (i, j)), grade, flush=True)

    if not args.parallel:
        for task in tasks:
            show(*run_pair(task))
        return

    # Spawned workers import numpy fresh, so they see these; the parent's
    # already-imported libraries are unaffected.
    for v in THREAD_VARS:
        os.environ.setdefault(v, "1")
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.parallel, mp_context=ctx) as ex:
        futures = [ex.submit(run_pair, task) for task in tasks]
        for fut in as_completed(futures):
            show(*fut.result())


# --- entry point --------------------------------------------------------------


def run_file(args: argparse.Namespace) -> None:
    algo = ALGOS[args.algo]
    sequences = load_fasta(args.contig_file, args.peptide_file, algo.encoded)
    if args.shuffle:
        sequences = shuffled(sequences, args.seed)
    bins, unbinned = run_binner(algo, sequences, vars(args))
    print(f"Bin sizes: {[len(b) for b in bins]} ; Unbinned: {len(unbinned)}")
    with open(args.output_file, "w") as f:
        for contig, bin_label in dump_bins(bins, unbinned):
            print("\t".join([contig, bin_label]), file=f)


def run_a14(args: argparse.Namespace) -> None:
    algo = ALGOS[args.algo]
    sequences = load_a14(args.mrna_path, args.peptide_path, algo.encoded)
    if args.shuffle:
        sequences = shuffled(sequences, args.seed)
    bins, unbinned = run_binner(algo, sequences, vars(args))
    grade_str, best_bin = grade_bins_a14(args.comp_path, args.match_path, bins, unbinned)
    print(grade_str)

    if args.deets:
        deets = symbiont_bin_details(args.homology_path, args.comp_path, bins[best_bin])
        print(f"\nDetails for Bin {best_bin}")
        print("#protein\tsymbiont_pct\tsymbiont_match_len\teuk_prot_match\teuk_prot_score")
        for d in deets:
            print(f"{d[0]}\t{d[1]}\t{d[2]}\t{d[3]}\t{d[4]}\t")


def parse_args() -> argparse.Namespace:
    # ALGO and DATASET decide which configs supply the binner flags, so they
    # must come first and are read before the full parse.
    usage = "run_binner.py {%s} {%s} ..." % (",".join(ALGOS), ",".join(DATASETS))
    if len(sys.argv) < 3 or sys.argv[1] not in ALGOS or sys.argv[2] not in DATASETS:
        raise SystemExit(f"usage: {usage}\n\n{__doc__}")
    algo, dataset = sys.argv[1], sys.argv[2]

    # "resolve" lets base_knn_config.yaml's seed replace the --seed below, so
    # one flag drives both the shuffle and knn's own seed.
    parser = argparse.ArgumentParser(usage=usage, conflict_handler="resolve")
    parser.add_argument("algo", choices=ALGOS)
    parser.add_argument("dataset", choices=DATASETS)
    if dataset == "file":
        parser.add_argument("contig_file", type=str)
        parser.add_argument("peptide_file", type=str)
        parser.add_argument("output_file", type=str)
    elif dataset in MIXTURES:
        parser.add_argument("org_a", type=str, nargs="?")
        parser.add_argument("org_b", type=str, nargs="?")
        parser.add_argument("--parallel", type=int, nargs="?", const=DEFAULT_WORKERS, default=0, metavar="N")
        parser.add_argument(
            "--names", action="store_true", help="print organism codes (EP00002, 07) instead of indices"
        )
    parser.add_argument("--shuffle", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=0)

    configs = ALGOS[algo].configs + ([A14_CONFIG] if dataset == "a14" else [])
    return get_args(configs, parser)


def main() -> None:
    args = parse_args()
    if args.dataset == "file":
        run_file(args)
    elif args.dataset == "a14":
        run_a14(args)
    else:
        run_mixtures(args)


if __name__ == "__main__":
    main()
