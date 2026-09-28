"""Compare label-free stopping rules on merge paths from merge_path_12417881.py.

For each mixture the *right* stop is the fewest-bins step before the first
merge that costs precision: the last step whose precision is within `damage` of
the path's starting precision, before any step falls below that.  A rule picks
a step from label-free statistics alone; it is scored on how often it goes past
the right stop (crossed), how its bin count compares with the right one, and
the r@4 / r@16 / precision of the partition it picks.

Parameters are chosen on half the mixtures (by name order, even positions) and
reported on the other half, so the tuned numbers are not in-sample.

    uv run scripts/merge_path_analysis.py path32
"""

import argparse
import json
import os
from collections import Counter
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

PREFIX_LEN = 7
ROOT = "runs/path"
Path = dict[str, npt.NDArray[np.float64]]
Rule = Callable[[Path], int]


def load(path: str, total_bases: float | None = None) -> Path:
    data = np.load(path)
    org = np.array([i[:PREFIX_LEN] for i in data["ids"]])
    steps = len(data["n_live"])
    labelled: dict[str, list[float]] = {"prec": [], "r4": [], "r16": [], "cov": []}
    for t in range(steps):
        member = data["membership"][t]
        bins = [np.flatnonzero(member == k) for k in np.unique(member[member >= 0])]
        bins.sort(key=len, reverse=True)
        correct = [Counter(org[b]).most_common(1)[0][1] for b in bins]
        assigned = sum(len(b) for b in bins)
        labelled["prec"].append(sum(correct) / assigned if assigned else 0.0)
        labelled["r4"].append(sum(correct[:4]) / len(member))
        labelled["r16"].append(sum(correct[:16]) / len(member))
        labelled["cov"].append(assigned / len(member))
    out = {k: np.asarray(data[k], dtype=np.float64) for k in data.files if k not in ("ids", "membership")}
    out.update({k: np.array(v) for k, v in labelled.items()})
    out["n_species"] = np.array([len(set(org))])
    if total_bases is not None:
        # held-out fit per base over *all* contigs, unassigned ones counting as
        # background (0): dissolving a species' last cluster then costs fit,
        # where ll_per_base (over assigned contigs only) just loses them
        out["ll_all"] = out["total_score"] / total_bases
    return out


def right_stop(p: Path, damage: float) -> int:
    floor = p["prec"][0] - damage
    below = np.flatnonzero(p["prec"] < floor)
    return int(below[0]) - 1 if below.size else len(p["prec"]) - 1


def first_trigger(stop: npt.NDArray[np.bool_]) -> int:
    """Index of the first True -- stop there, before the next merge -- else the last step."""
    hits = np.flatnonzero(stop)
    return int(hits[0]) if hits.size else len(stop) - 1


def guard(tol: float) -> Rule:
    """DBV's rule with a noise tolerance: stop before a merge that drops per-base LL by > tol."""

    def rule(p: Path) -> int:
        ll = p["ll_per_base"]
        return first_trigger(np.append(ll[1:] < ll[:-1] - tol, True))

    return rule


def guard_all(tol: float) -> Rule:
    """As guard, on fit per base over all contigs (needs total_bases.json)."""

    def rule(p: Path) -> int:
        ll = p["ll_all"]
        return first_trigger(np.append(ll[1:] < ll[:-1] - tol, True))

    return rule


def relative_guard(c: float) -> Rule:
    """Stop before a per-base LL drop larger than c x the median |step change| so far."""

    def rule(p: Path) -> int:
        ll = p["ll_per_base"]
        change = np.diff(ll)
        stop = np.zeros(len(ll), dtype=bool)
        for t in range(len(change)):
            noise = np.median(np.abs(change[:t])) if t >= 3 else np.inf
            stop[t] = -change[t] > c * noise
        stop[-1] = True
        return first_trigger(stop)

    return rule


def absorb(a0: float) -> Rule:
    """Stop once the most absorbable cluster is below a0 (DBV's absorb_thresh)."""

    def rule(p: Path) -> int:
        return first_trigger(np.nan_to_num(p["absorb_max"], nan=-np.inf) < a0)

    return rule


def mdl(lam: float) -> Rule:
    """Fewest-bits step: minimise -total_score + lam x trie states, over the whole path."""

    def rule(p: Path) -> int:
        return int(np.argmin(-p["total_score"] + lam * p["states"]))

    return rule


def bic(lam: float) -> Rule:
    """As mdl, but a flat cost per cluster scaled by log(assigned bases)."""

    def rule(p: Path) -> int:
        return int(np.argmin(-p["total_score"] + lam * p["n_live"] * np.log(p["assigned_bases"])))

    return rule


def biggest_drop(p: Path) -> int:
    """Parameter-free: stop just before the largest single-step fall in per-base LL."""
    return int(np.argmin(np.diff(p["ll_per_base"])))


def evaluate(rule: Rule, paths: dict[str, Path], damage: float) -> dict[str, float]:
    rows = []
    for p in paths.values():
        t, right = rule(p), right_stop(p, damage)
        rows.append(
            (
                t > right,
                np.log2(p["n_live"][t] / p["n_live"][right]),
                p["n_live"][t],
                p["r4"][t],
                p["r16"][t],
                p["prec"][t],
            )
        )
    a = np.array(rows, dtype=np.float64)
    return {
        "crossed": a[:, 0].mean(),
        "log2bins": float(np.median(a[:, 1])),
        "bins": float(np.median(a[:, 2])),
        "r@4": a[:, 3].mean(),
        "r@16": a[:, 4].mean(),
        "prec": a[:, 5].mean(),
    }


FAMILIES: dict[str, tuple[Callable[[float], Rule], list[float]]] = {
    "guard": (guard, [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]),
    "relative_guard": (relative_guard, [3, 5, 10, 20, 40]),
    "guard_all": (guard_all, [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]),
    "absorb": (absorb, [0.7, 0.5, 0.3, 0.2, 0.1, 0.0, -0.1]),
    "mdl": (mdl, [0.0, 0.5, 1, 2, 4, 8, 16]),
    "bic": (bic, [0, 100, 1000, 1500, 2000, 3000, 4000, 6000, 10000, 30000]),
}


OBJECTIVES: dict[str, Callable[[dict[str, float]], float]] = {
    # few bins without crossing into damage
    "safe": lambda m: m["r@4"] - m["crossed"],
    # few, pure bins: a cross-species merge lowers r@4, extra bins cannot raise it
    "r4": lambda m: m["r@4"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--damage", type=float, default=0.01)
    parser.add_argument("--sweep", action="store_true", help="print every parameter value")
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), default="r4")
    parser.add_argument("--swap", action="store_true", help="tune on the odd half, test on the even")
    args = parser.parse_args()

    directory = os.path.join(ROOT, args.name)
    names = sorted(f[:-4] for f in os.listdir(directory) if f.endswith(".npz"))
    bases_file = os.path.join(directory, "total_bases.json")
    bases = json.load(open(bases_file)) if os.path.exists(bases_file) else {}
    paths = {n: load(os.path.join(directory, n + ".npz"), bases.get(n)) for n in names}
    if not bases:
        del FAMILIES["guard_all"]
    halves = (names[0::2], names[1::2])[:: -1 if args.swap else 1]
    tune = {n: paths[n] for n in halves[0]}
    test = {n: paths[n] for n in halves[1]}
    print(f"{len(paths)} paths: tune on {len(tune)}, test on {len(test)} (damage {args.damage})\n")

    header = f"{'rule':<24}{'crossed':>8}{'log2bins':>9}{'bins':>6}{'r@4':>7}{'r@16':>7}{'prec':>7}"

    def show(label: str, m: dict[str, float]) -> None:
        print(
            f"{label:<24}{m['crossed']:>8.2f}{m['log2bins']:>9.2f}{m['bins']:>6.0f}"
            f"{m['r@4']:>7.3f}{m['r@16']:>7.3f}{m['prec']:>7.3f}"
        )

    right: Rule = lambda p: right_stop(p, args.damage)  # noqa: E731
    first: Rule = lambda p: 0  # noqa: E731
    print("reference, on the test half\n" + header)
    show("right stop (labels)", evaluate(right, test, args.damage))
    best_r4: Rule = lambda p: int(np.argmax(p["r4"]))  # noqa: E731
    show("best r@4 (labels)", evaluate(best_r4, test, args.damage))
    show("no merging", evaluate(first, test, args.damage))
    show("biggest LL drop", evaluate(biggest_drop, test, args.damage))

    print("\ntuned on one half, scored on the other\n" + header)
    for family, (make, values) in FAMILIES.items():
        objective = OBJECTIVES[args.objective]
        scored = [(objective(evaluate(make(v), tune, args.damage)), v) for v in values]
        if args.sweep:
            for v in values:
                show(f"  {family}({v:g}) tune", evaluate(make(v), tune, args.damage))
        best = max(scored)[1]
        show(f"{family}({best:g})", evaluate(make(best), test, args.damage))


if __name__ == "__main__":
    main()
