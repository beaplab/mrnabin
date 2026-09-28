"""Score and compare the bin dumps written by benchmark_12417881.py.

    uv run scripts/bench_summary.py dbv default coarse
    uv run scripts/bench_summary.py dbv default --per-mixture

Only mixtures present in every named run are compared, so a run still in
progress is compared on what it has finished.

Columns:
  bins     total bins emitted
  n90      largest bins needed to hold 90% of binned contigs
  cov      fraction of contigs binned at all
  acc      plurality accuracy over all bins (the old, fragment-flattered score)
  r@N      correct contigs in the N largest bins, over all contigs -- the
           headline: fragmenting cannot raise it, lumping species lowers it
  p@N      accuracy within the N largest bins
  worst1   lowest per-organism fraction held by that organism's single best bin
  worst4   the same, over its best 4 bins
"""

import argparse
import os
from collections import defaultdict

import numpy as np

from lib.evaluate import bins_to_cover, budget_scores, species_completeness

PREFIX_LEN = 7
OUT_ROOT = "runs/bench"
BUDGETS = (4, 16, 64)


def load(path: str) -> tuple[list[set[str]], set[str], float]:
    by_label: dict[str, set[str]] = defaultdict(set)
    seconds = float("nan")
    with open(path) as f:
        for line in f:
            key, value = line.rstrip("\n").split("\t")
            if key == "#seconds":
                seconds = float(value)
            else:
                by_label[value].add(key)
    unbinned = by_label.pop("bin_none", set())
    return list(by_label.values()), unbinned, seconds


def score(path: str) -> dict[str, float]:
    bins, unbinned, seconds = load(path)
    total = sum(len(b) for b in bins) + len(unbinned)
    budgets = budget_scores(bins, unbinned, BUDGETS + (len(bins),), PREFIX_LEN)
    row = {
        "bins": len(bins),
        "n90": bins_to_cover(bins, 0.9),
        "cov": 1 - len(unbinned) / total,
        "acc": budgets[len(bins)][0],
    }
    for n in BUDGETS:
        row[f"r@{n}"] = budgets[n][1]
    for n in BUDGETS:
        row[f"p@{n}"] = budgets[n][0]
    row["worst1"] = min(species_completeness(bins, unbinned, 1, PREFIX_LEN).values())
    row["worst4"] = min(species_completeness(bins, unbinned, 4, PREFIX_LEN).values())
    row["sec"] = seconds
    return row


def fmt(key: str, value: float) -> str:
    if key in ("bins", "n90", "sec"):
        return f"{value:>7.0f}"
    return f"{value:>7.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--per-mixture", action="store_true")
    parser.add_argument("--species", type=int, default=0, help="only mixtures of this many organisms (0 = all)")
    args = parser.parse_args()

    files = {n: {f[:-4] for f in os.listdir(os.path.join(OUT_ROOT, n)) if f.endswith(".tsv")} for n in args.names}
    common = sorted(set.intersection(*files.values()))
    if args.species:
        common = [m for m in common if m.count("+") + 1 == args.species]
    print(f"{len(common)} mixtures in common\n")

    scores = {n: {m: score(os.path.join(OUT_ROOT, n, m + ".tsv")) for m in common} for n in args.names}
    keys = list(next(iter(next(iter(scores.values())).values())).keys()) if common else []
    width = max(len(n) for n in args.names + ["mixture"])

    if args.per_mixture:
        for m in common:
            print(m)
            for n in args.names:
                print(f"  {n:<{width}}", "".join(fmt(k, scores[n][m][k]) for k in keys))
        print()

    print(f"{'mean':<{width + 2}}", "".join(f"{k:>7}" for k in keys))
    for n in args.names:
        means = {k: float(np.nanmean([scores[n][m][k] for m in common])) for k in keys}
        print(f"  {n:<{width}}", "".join(fmt(k, means[k]) for k in keys))
    print(f"{'median':<{width + 2}}", "".join(f"{k:>7}" for k in keys))
    for n in args.names:
        medians = {k: float(np.nanmedian([scores[n][m][k] for m in common])) for k in keys}
        print(f"  {n:<{width}}", "".join(fmt(k, medians[k]) for k in keys))


if __name__ == "__main__":
    main()
