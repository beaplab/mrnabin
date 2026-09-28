"""Record a full VLP merge path per mixture, for choosing a stopping criterion.

KNN seeds VLP with its `vlp_seed_bins` largest bins, as in KNNVLPBinner.  DBV's
length schedule then activates and reassigns contigs without culling.  From
there clusters are dissolved one at a time -- always the most absorbable, i.e.
the one whose members score best elsewhere relative to at home -- and the
contigs reassigned, all the way down to one cluster.  No stopping rule is
applied: the point is to record every step so rules can be compared offline
(`merge_path_analysis.py`).

Each mixture writes `runs/path/<name>/<mixture>.npz` holding the contig ids,
the membership after every step (-1 = unassigned), and per-step label-free
statistics:

  n_live        live clusters
  ll_per_base   total own-cluster score / assigned bases (DBV's cull guard)
  total_score   total own-cluster score, in the trie's log units
  assigned      assigned contigs; assigned_bases likewise in bases
  states        total trie states over live clusters (model size)
  absorb_max    absorbability of the cluster dissolved next

With --holdout every score is out-of-fold (see knn.vlp_merge.score_matrix), so total_score
is a held-out likelihood and needs no complexity penalty.

    uv run scripts/merge_path_12417881.py mypath --sets random hard --workers 12
"""

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from benchmark_12417881 import (  # type: ignore[import-not-found]  # noqa: E402  # sibling script
    CONFIGS,
    HARD,
    PREFIX_LEN,
    _files,
    mixture_id,
    sample_mixtures,
)

from lib.encoded_sequences import Sequence  # noqa: E402

OUT_ROOT = "runs/path"


def _run_one(mixture: tuple[str, ...], params: dict[str, Any], out_dir: str, holdout: bool) -> str:
    from knn.vlp_merge import KNNVLPBinner, absorbability, assignment_stats, dissolve, reassign, score_matrix
    from lib.fasta import get_sequences

    start = time.time()
    sequences: list[Sequence] = []
    for org in mixture:
        contigs, peptides = _files(org)
        sequences += get_sequences(contigs, peptides, prefix=org, label_col=1)
    binner = KNNVLPBinner(**params)
    binner.sequences = sequences
    n_clusters = binner._run_distbin()
    folds = np.random.default_rng(0).integers(0, 2, len(sequences)).astype(np.int8) if holdout else None

    # DBV's length schedule, reassignment only
    for threshold in binner._len_threshs():
        for s in sequences:
            if len(s.seq) >= threshold:
                s.active = True
        scores, states = score_matrix(sequences, n_clusters, folds)
        reassign(sequences, scores)

    memberships, stats = [], []
    while True:
        scores, states = score_matrix(sequences, n_clusters, folds)
        reassign(sequences, scores)
        row = assignment_stats(sequences, scores, states)
        absorb = absorbability(sequences, scores)
        row["absorb_max"] = float(absorb.max()) if np.isfinite(absorb).any() else np.nan
        memberships.append(np.array([s.cluster if s.active else -1 for s in sequences], dtype=np.int16))
        stats.append(row)
        if row["n_live"] <= 1:
            break
        dissolve(sequences, scores, int(np.argmax(absorb)))

    arrays: dict[str, Any] = {k: np.array([r[k] for r in stats]) for k in stats[0]}
    arrays["ids"] = np.array([s.seq_id for s in sequences])
    arrays["membership"] = np.stack(memberships)
    np.savez_compressed(os.path.join(out_dir, mixture_id(mixture) + ".npz"), **arrays)
    return f"{mixture_id(mixture)} steps:{len(stats)} {time.time() - start:.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides, key=value")
    parser.add_argument("--sets", nargs="+", default=["random"], choices=["random", "hard"])
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--holdout", action="store_true", help="out-of-fold scoring: every score is held out")
    args = parser.parse_args()

    params: dict[str, Any] = {}
    for path in CONFIGS["knnv"]:
        with open(path) as f:
            params.update(yaml.safe_load(f) or {})
    params["vlp_seed_bins"] = 32
    for item in args.set:
        key, value = item.split("=", 1)
        params[key] = yaml.safe_load(value)
    params.update(verbose=False, compact=False, labeled=False, prefix_len=PREFIX_LEN)

    mixtures: list[tuple[str, ...]] = []
    if "random" in args.sets:
        mixtures += sample_mixtures(0, 30, 6)
    if "hard" in args.sets:
        mixtures += HARD
    if args.only:
        mixtures = [m for m in mixtures if mixture_id(m) in args.only]

    out_dir = os.path.join(OUT_ROOT, args.name)
    os.makedirs(out_dir, exist_ok=True)
    todo = [m for m in mixtures if not os.path.exists(os.path.join(out_dir, mixture_id(m) + ".npz"))]
    todo.sort(key=len, reverse=True)
    print(f"{len(todo)} of {len(mixtures)} mixtures to run -> {out_dir}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_one, m, params, out_dir, args.holdout): m for m in todo}
        for fut in as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt):
                    raise
                print(f"{mixture_id(futures[fut])} FAILED {e!r}", flush=True)


if __name__ == "__main__":
    main()
