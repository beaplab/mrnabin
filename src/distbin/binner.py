"""
For a given list of LabeledKmerCounters of length n,
`distances` is the n x n matrix whose entries
d_{i, j} are the distance btw the counters in positions i and j of the list

Bins are sets of indices whose corresponding counters are all close in distance
"""

import time
from collections.abc import Sequence
from copy import deepcopy
from typing import cast

import numpy as np
import numpy.typing as npt
import scipy

from distbin.evaluate import (
    distance_stats,
    selected_label_stats,
)
from distbin.kmers import encoded_seq_to_kmer_array, generate_kmer_map, seq_to_kmer_array
from lib.evaluate import all_pure, grade_bins, sequence_stats


class Binner:
    def __init__(
        self,
        sequences: dict[str, str] | dict[str, npt.NDArray[np.int8]],
        k: int = 4,
        selection_percentiles: tuple[float, float] = (85.0, 95.0),
        creation_threshold: float = 0.1,
        selection_threshold: float = 0.175,
        cluster_thresholds: Sequence[float] = (0.02, 0.03125, 0.04, 0.05, 0.0625),
        reduction_constant: float = 16.0,
        final_threshold: float = 0.5,
        verbose: bool = False,
        compact: bool = False,
        minimal: bool = False,
        labeled: bool = False,
        prefix_len: int = 2,
        **kwargs: str,
    ):
        self.k = k
        self.kmer_map = generate_kmer_map(k)
        self.sequences = sequences
        self.verbose = verbose
        self.labeled = labeled
        self.compact = compact
        if self.compact:
            self.verbose = True
        sample = next(iter(sequences.values()), None)
        if isinstance(sample, np.ndarray):
            n_kmers = len(self.kmer_map)
            encoded = cast(dict[str, npt.NDArray[np.int8]], sequences)
            self.kmer_distributions = {l: encoded_seq_to_kmer_array(s, k, n_kmers) for l, s in encoded.items()}
        else:
            strs = cast(dict[str, str], sequences)
            self.kmer_distributions = {l: seq_to_kmer_array(s, k, self.kmer_map) for l, s in strs.items()}
        self.selection_percentiles = selection_percentiles
        self.creation_threshold = creation_threshold
        self.selection_threshold = selection_threshold
        self.cluster_thresholds = cluster_thresholds
        self.reduction_constant = reduction_constant
        self.final_threshold = final_threshold
        self.minimal = minimal
        self.prefix_len = prefix_len

    def run(self) -> tuple[list[set[str]], set[str]]:
        """Bin all sequences and return (bins, unbinned).

        If `minimal`, stops after the initial bin-creation and
        closest-bin-by-contig pass on the length-percentile selection;
        unbinned then includes every sequence outside the selection.
        Otherwise, also combines/reduces bins and assigns the remaining
        sequences by closest-bin-median.
        """
        start = time.time()
        all_labels = list(self.sequences.keys())
        selection = self._select_labels(self.selection_percentiles[0], self.selection_percentiles[1])

        distance_matrix = self._get_distance_matrix(selection)

        # create a large set of bins where we are confident that they are all single-species
        initial_bins = self._create_bins(selection, distance_matrix, self.creation_threshold)

        # add contigs to the closest bins (min dist to a contig in the bin)
        selection_bins = self._add_to_closest_bin(selection, distance_matrix, initial_bins)

        if self.minimal:
            binned = list(set.union(*selection_bins))
            unbinned = set([l for l in all_labels if l not in binned])
            return selection_bins, unbinned

        # reduce the number of bins by combining them
        combined_bins = self._combine_bins(selection_bins)

        # only keep the large bins
        reduced_bins = self._reduce_bins(combined_bins)

        # try binning the selection again
        bin_bins, _ = self._add_to_closest_bin_median(selection, reduced_bins, self.selection_threshold)

        final_bins, unbinned = self._add_to_closest_bin_median(all_labels, bin_bins, self.final_threshold)

        if self.verbose:
            print(f"TOTAL BIN TIME: {int(time.time()-start)} s", flush=True)

        return final_bins, unbinned

    def _select_labels(self, low_pct: float, high_pct: float) -> list[str]:
        low_idx = int(len(self.sequences) * low_pct / 100)
        high_idx = int(len(self.sequences) * high_pct / 100)
        selection = sorted([(len(s), l) for l, s in self.sequences.items()])[low_idx:high_idx]
        if self.verbose:
            print("Overall sequence length stats:")
            print(sequence_stats([len(s) for s in self.sequences.values()]))
            print(f"Selected {len(selection)} contigs ranging in length from {selection[0][0]} to {selection[-1][0]}")
            print(sequence_stats([len(self.sequences[l]) for (_, l) in selection]))
        return [l for (_, l) in selection]

    def _get_distance_matrix(self, labels: list[str]) -> npt.NDArray:
        stack = np.stack([self.kmer_distributions[l] for l in labels])
        dists = np.astype(scipy.spatial.distance.pdist(stack, "cityblock") / 2, np.float64)
        matrix = scipy.spatial.distance.squareform(dists)

        if self.verbose:
            print(f"overall stats for all {len(labels)} contigs:")
            print(distance_stats(matrix))
            if self.labeled:
                selected_label_stats(labels, matrix, self.kmer_distributions, self.prefix_len)

        return matrix

    def _one_line_bin_stats(self, bins: list[set[str]], unbinned: set[str] = set()) -> str:
        num_binned = len(set([x for b in bins for x in b]))
        total = num_binned + len(unbinned)
        stats = f"{len(bins)} bins ({num_binned}/{total}) "
        if self.labeled:
            return stats + grade_bins(bins, unbinned, prefix_len=self.prefix_len)
        else:
            return stats + str([len(b) for b in bins[:10]])

    def _create_bins(self, labels: list[str], distance_matrix: npt.NDArray, threshold: float) -> list[set[str]]:
        label_array = np.array(labels)
        bins: list[set[str]] = []
        for idx in range(distance_matrix.shape[0]):
            close_to_idx = np.where(distance_matrix[idx] < threshold)
            close_labels: list[str] = cast(list[str], label_array[close_to_idx].tolist())
            _add_to_bins(bins, set(close_labels + [labels[idx]]))
        bins = sorted([b for b in bins if len(b) > 1], reverse=True, key=lambda x: len(x))

        if self.verbose:
            verb = f"Creating bins by binning contigs that are within {threshold:.3} of each other.\n"
            unbinned = set([l for l in labels if l not in set.union(*bins)])
            print(
                f"S.0. {threshold if self.compact else verb}",
                self._one_line_bin_stats(bins, unbinned),
                all_pure(bins, self.prefix_len) if self.labeled else " ",
            )
        return bins

    def _add_to_closest_bin(
        self, labels: list[str], distance_matrix: npt.NDArray, bins: list[set[str]]
    ) -> list[set[str]]:
        new_bins = [deepcopy(b) for b in bins]
        binned = list(set.union(*bins))
        label_to_idx = {l: i for i, l in enumerate(labels)}
        binned_cols = [label_to_idx[l] for l in binned]
        sub = distance_matrix[:, binned_cols]
        closest_idx = np.argmin(sub, axis=1)
        closest_dist = np.min(sub, axis=1)

        for i, l in enumerate(labels):
            if l in binned:
                continue
            if closest_dist[i] > self.selection_threshold:
                continue
            _add_to_bins(new_bins, set([labels[i], binned[closest_idx[i]]]))

        new_bins = sorted(new_bins, key=lambda b: len(b), reverse=True)

        if self.verbose:
            verb = (
                "Binned selected contigs to closest bin, "
                f"except those > {self.selection_threshold} away from any bin.\n"
            )
            unbinned = set([l for l in labels if l not in set.union(*new_bins)])
            print(
                f"S.1. {self.selection_threshold if self.compact else verb}",
                self._one_line_bin_stats(new_bins, unbinned),
            )

        return new_bins

    def _combine_bins(self, bins: list[set[str]]) -> list[set[str]]:
        start = time.time()
        new_bins = [deepcopy(b) for b in bins]
        for t_num, threshold in enumerate(self.cluster_thresholds):
            n = len(new_bins)
            median_stack = np.stack(
                [np.median(np.stack([self.kmer_distributions[l] for l in b]), axis=0) for b in new_bins]
            )
            dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(median_stack, "cityblock") / 2)

            clustered: set[int] = set()
            for i in range(n):
                if i in clustered:
                    continue
                close_bins = [j for j in range(n) if j != i and j not in clustered and dists[i, j] < threshold]
                if not close_bins:
                    continue
                new_bins[i] |= set.union(*[new_bins[j] for j in close_bins])
                clustered |= set(close_bins)
            new_bins = sorted(
                [b for i, b in enumerate(new_bins) if i not in clustered], key=lambda x: len(x), reverse=True
            )
            if self.verbose and not self.compact:
                print(f"S.2.{t_num} Combining bins...")
                print(f" {threshold} Top 10 bin sizes: ", [len(b) for b in new_bins[:10]])

        if self.verbose:
            verb = f"Combined_bins {self.cluster_thresholds}\n"
            print(f"S.2. {self.cluster_thresholds if self.compact else verb}", end=" ")
            print(self._one_line_bin_stats(new_bins))
            if not self.compact:
                print(f"time: {time.time()-start:.0f} s")

        return new_bins

    def _reduce_bins(self, bins: list[set[str]]) -> list[set[str]]:
        total = sum([len(b) for b in bins])
        sizes = [len(b) for b in bins]
        indices = [i for i in range(len(sizes)) if total < sizes[i] ** 2]
        if indices:
            index = indices[-1] + 1
            core_bins = bins[:index]
        else:
            index = 1
            core_bins = bins[:1]

        reduction_constant = min(self.reduction_constant, total)

        while sum([len(b) for b in core_bins]) ** 2 < reduction_constant * total:
            # TODO: this is very arbitrary; find a better criterion
            index += 1
            core_bins = bins[:index]

        if self.verbose:
            verb = f"Only keeping top {len(core_bins)} bins.\n"
            unbinned = set()
            if len(bins) > index:
                unbinned = set.union(*bins[index:])
            print(f"S.3. {sum([len(b) for b in core_bins]) if self.compact else verb}", end=" ")
            print(self._one_line_bin_stats(core_bins, unbinned))

        return core_bins

    def _add_to_closest_bin_median(
        self, labels: list[str], bins: list[set[str]], threshold: float
    ) -> tuple[list[set[str]], set[str]]:
        new_bins = [deepcopy(b) for b in bins]
        binned = set.union(*bins)
        median_stack = np.stack(
            [np.median(np.stack([self.kmer_distributions[l] for l in b]), axis=0) for b in new_bins]
        )
        unbinned: set[str] = set()
        candidates = [l for l in labels if l not in binned]
        if candidates:
            cand_stack = np.stack([self.kmer_distributions[l] for l in candidates])
            dists = scipy.spatial.distance.cdist(cand_stack, median_stack, "cityblock") / 2
            closest_index = np.argmin(dists, axis=1)
            closest_dist = dists[np.arange(len(candidates)), closest_index]
            for i, l in enumerate(candidates):
                if closest_dist[i] > threshold:
                    unbinned.add(l)
                else:
                    new_bins[closest_index[i]].add(l)

        new_bins = sorted(new_bins, key=lambda b: len(b), reverse=True)

        if self.verbose:
            verb = f"Binned contigs to closest bin median up to {threshold}.\n"
            print(f"*. {threshold if self.compact else verb}", end=" ")
            print(self._one_line_bin_stats(new_bins, unbinned))

        return new_bins, unbinned


def _add_to_bins(bins: list[set[str]], new_bin: set[str]) -> None:
    # check if new_bin overlaps with any existing bins
    bin_nums = []
    for i, b in enumerate(bins):
        if new_bin & b:
            bin_nums.append(i)

    if len(bin_nums) >= 1:
        # merge other overlapping bins into first overlapping bin
        while len(bin_nums) > 1:
            i = bin_nums.pop()
            b = bins.pop(i)
            bins[bin_nums[0]] |= b
        # merge new_bin into first overlapping bin
        bins[bin_nums[0]] |= new_bin
    else:
        # no overlaps, just add new_bin to collection
        bins.append(new_bin)
