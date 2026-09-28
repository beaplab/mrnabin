from collections import Counter, defaultdict
from typing import Generator

import numpy as np

from lib.encoded_sequences import Sequence


def bins_by_id(sequences: list[Sequence]) -> tuple[list[set[str]], set[str]]:
    # returns list of bins and the unbinned set
    bins_by_cluster_id = defaultdict(list)
    for s in sequences:
        if s.active and s.cluster != -1:
            bins_by_cluster_id[s.cluster].append(s.seq_id)
    bins = [set(b) for b in bins_by_cluster_id.values()]
    unbinned = set(s.seq_id for s in sequences if not s.active or s.cluster == -1)
    return bins, unbinned


def dump_bins(bins: list[set[str]], unassigned: set[str]) -> Generator[tuple[str, str]]:
    for i, b in enumerate(bins):
        bin_label = f"bin_{i+1:04d}"
        for contig in b:
            yield (contig, bin_label)
    for contig in unassigned:
        yield (contig, "bin_none")


def _bin_counter(binn: set[str], prefix_len: int) -> Counter[str]:
    """count labels; assumed to be a prefix_len char prefix"""
    return Counter(b[:prefix_len] for b in binn)


def _purity(binn: set[str], prefix_len: int) -> float:
    counter = _bin_counter(binn, prefix_len)
    if not counter:
        return 0.0
    correct = counter.most_common()[0][1]
    return correct / counter.total()


def _bin_label(binn: set[str], prefix_len: int) -> str:
    counter = _bin_counter(binn, prefix_len)
    return counter.most_common()[0][0] if counter else "?"


def _bin_stats(bins: list[set[str]], prefix_len: int) -> str:
    return "[" + ", ".join([f"{_bin_label(b, prefix_len)} {_purity(b, prefix_len):.2} {len(b)}" for b in bins]) + "]"


def grade_bins(bins: list[set[str]], unassigned: set[str], limit: int = 5, prefix_len: int = 2) -> str:
    num_correct = []
    num_wrong = []
    for i, b in enumerate(bins):
        bin_count = _bin_counter(b, prefix_len)
        if not bin_count:
            num_correct.append(0)
            num_wrong.append(0)
            continue
        top_label, top_count = bin_count.most_common()[0]
        num_correct.append(top_count)
        num_wrong.append(bin_count.total() - top_count)

    total_unassigned = len(unassigned)
    total_correct = sum(num_correct)
    total_wrong = sum(num_wrong)
    total_assigned = total_correct + total_wrong
    total_total = total_assigned + total_unassigned
    accuracy = total_correct / total_assigned if total_assigned else 0.0
    unassigned_frac = total_unassigned / total_total if total_total else 0.0
    output = f"Num bins: {len(bins)} "
    output += f"Correct: {total_correct}/{total_assigned} "
    output += f"({accuracy:.3}) "
    output += f"Unassigned: {total_unassigned} ({unassigned_frac:.3}) "
    output += "Top bins: " + _bin_stats(bins[:limit], prefix_len)
    return output


def all_pure(bins: list[set[str]], prefix_len: int) -> str:
    if set([_purity(b, prefix_len) for b in bins]) == {1.0}:
        return "\U00002705"
    else:
        return "\U0000274c"


def sequence_stats(values: list[int]) -> str:
    return (
        f"count:{len(values)} "
        f"mean:{np.mean(values):.0f} "
        f"min:{np.min(values)} "
        f"p01:{np.percentile(values, 1):.0f} "
        f"p25:{np.percentile(values, 25):.0f} "
        f"p50:{np.percentile(values, 50):.0f} "
        f"p75:{np.percentile(values, 75):.0f} "
        f"p90:{np.percentile(values, 90):.0f} "
        f"p95:{np.percentile(values, 95):.0f} "
        f"p99:{np.percentile(values, 99):.0f} "
        f"max:{np.max(values)} "
    )


def _species_totals(bins: list[set[str]], unassigned: set[str], prefix_len: int) -> Counter[str]:
    totals: Counter[str] = Counter(u[:prefix_len] for u in unassigned)
    for b in bins:
        totals.update(_bin_counter(b, prefix_len))
    return totals


def budget_scores(
    bins: list[set[str]], unassigned: set[str], budgets: tuple[int, ...] = (4, 8, 16, 32, 64), prefix_len: int = 2
) -> dict[int, tuple[float, float]]:
    """(precision, recall) when only the `budget` largest bins are reported.

    Everything outside those bins counts as unbinned.  Precision credits each
    kept bin with its plurality organism, as `grade_bins` does; recall is
    correct contigs over *all* contigs.  With a fixed budget, fragmenting into
    tiny pure bins no longer pays: it raises precision but starves recall.
    Largest-first is the choice a user without labels would make.
    """
    ordered = sorted(bins, key=len, reverse=True)
    total = sum(len(b) for b in bins) + len(unassigned)
    scores = {}
    for budget in budgets:
        kept = ordered[:budget]
        assigned = sum(len(b) for b in kept)
        correct = sum(_bin_counter(b, prefix_len).most_common()[0][1] for b in kept if b)
        precision = correct / assigned if assigned else 0.0
        recall = correct / total if total else 0.0
        scores[budget] = (precision, recall)
    return scores


def species_completeness(
    bins: list[set[str]], unassigned: set[str], top: int = 1, prefix_len: int = 2
) -> dict[str, float]:
    """Per organism, the fraction of its contigs in its `top` bins that it leads.

    Only bins where the organism is the plurality count, so a 50/50 bin cannot
    make both organisms look complete.  top=1 asks how much of each organism a
    single bin captures, which is the ideal output.
    """
    totals = _species_totals(bins, unassigned, prefix_len)
    held: dict[str, list[int]] = defaultdict(list)
    for b in bins:
        counter = _bin_counter(b, prefix_len)
        if counter:
            label, count = counter.most_common()[0]
            held[label].append(count)
    return {s: sum(sorted(held[s], reverse=True)[:top]) / n for s, n in totals.items()}


def bins_to_cover(bins: list[set[str]], fraction: float = 0.9) -> int:
    """How many of the largest bins hold `fraction` of all binned contigs.

    A size-distribution summary that ignores the long tail of tiny bins, where
    a raw bin count is dominated by it.
    """
    sizes = sorted((len(b) for b in bins), reverse=True)
    needed = fraction * sum(sizes)
    held = 0
    for count, size in enumerate(sizes, 1):
        held += size
        if held >= needed:
            return count
    return len(sizes)
