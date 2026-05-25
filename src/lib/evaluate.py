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
