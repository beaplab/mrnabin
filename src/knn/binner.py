"""Mutual-kNN + Leiden binning.

K.0 build a mutual-kNN graph over the length-percentile selection and
K.1 partition the graph with Leiden
K.2 drop bins below `min_bin_size` back to unbinned
K.3 assign the remaining contigs by a vote among their nearest neighbours in the binned core
"""

import math
import time
from collections.abc import Sequence

import igraph as ig
import leidenalg
import numpy as np
import numpy.typing as npt

from distbin.evaluate import distance_stats, selected_label_stats
from knn.graph import BLOCK_SIZE, auto_n_neighbors, graph_stats, knn_indices_from_stack, mutual_knn_edges
from lib.distances import kmer_distributions, stack_distributions, tv_cdist, tv_distance_matrix
from lib.evaluate import all_pure, grade_bins
from lib.selection import select_by_length_percentile, selection_report

PARTITION_TYPES = {
    "cpm": leidenalg.CPMVertexPartition,
    "modularity": leidenalg.RBConfigurationVertexPartition,
}

SPREAD_FLOOR = 1e-9


class Binner:
    def __init__(
        self,
        sequences: dict[str, str] | dict[str, npt.NDArray[np.int8]],
        k: int = 4,
        selection_percentiles_low: float = 45.0,
        selection_percentiles_high: float = 95.0,
        n_neighbors: int = 512,
        mutual: bool = True,
        quality: str = "cpm",
        resolution: float = 0.5,
        n_iterations: int = 2,
        seed: int = 0,
        min_bin_size: int = 3,
        assign_remaining: bool = True,
        assign_top_bins: int = 5,
        assign_sample: int = 30,
        max_bins: int = 0,
        target_coverage: float = 1.0,
        length_stages: Sequence[float] = (),
        stage_admit: float = 0.75,
        recluster_residual: bool = False,
        residual_min_bin_size: int = 8,
        z_max: float = math.inf,
        block_size: int = BLOCK_SIZE,
        stats_max_nodes: int = 6000,
        verbose: bool = False,
        compact: bool = False,
        labeled: bool = False,
        prefix_len: int = 2,
        **kwargs: str,
    ):
        if quality not in PARTITION_TYPES:
            raise ValueError(f"quality must be one of {sorted(PARTITION_TYPES)}, got {quality!r}")
        self.k = k
        self.sequences = sequences
        self.kmer_distributions = kmer_distributions(sequences, k)
        self.selection_percentiles = (selection_percentiles_low, selection_percentiles_high)
        self.n_neighbors = n_neighbors
        self.mutual = mutual
        self.quality = quality
        self.resolution = resolution
        self.n_iterations = n_iterations
        self.seed = seed
        self.min_bin_size = min_bin_size
        self.assign_remaining = assign_remaining
        self.assign_top_bins = assign_top_bins
        self.assign_sample = assign_sample
        self.max_bins = max_bins
        self.target_coverage = target_coverage
        # CLI list arguments arrive as strings
        self.length_stages = tuple(float(p) for p in length_stages)
        self.stage_admit = stage_admit
        self.recluster_residual = recluster_residual
        self.residual_min_bin_size = residual_min_bin_size
        self.z_max = z_max
        self.block_size = block_size
        self.stats_max_nodes = stats_max_nodes
        self.verbose = verbose
        self.compact = compact
        if self.compact:
            self.verbose = True
        self.labeled = labeled
        self.prefix_len = prefix_len

    def run(self) -> tuple[list[set[str]], set[str]]:
        """Bin all sequences and return (bins, unbinned).

        K.0-K.2 build the graph and partition it; `max_bins` optionally keeps
        only the largest bins as seeds.  Everything else is assigned by K.3,
        either in one pass or -- with `length_stages` set -- in length tranches
        that recompute each bin's envelope as it grows.  `recluster_residual`
        then lets whatever is still unassigned form its own bins, which is the
        only way a species the seeding missed can get one.  Whatever remains
        after all of that is returned unbinned.
        """
        start = time.time()
        all_labels = list(self.sequences.keys())
        selection = self._select_labels()

        edges = self._build_graph(selection)
        bins = self._partition(selection, edges)
        bins, _ = self._drop_small_bins(bins)
        bins = self._keep_largest(bins)

        if self.assign_remaining:
            # staging replaces the single pass rather than preceding it: the
            # contigs it declines are what `recluster_residual` works on, and a
            # full pass afterwards would leave nothing behind.  Either way
            # _assign_by_fit sees every label, so contigs from bins dropped at
            # K.2 get a second chance and `small` must not be re-added.
            if self.length_stages:
                bins = self._assign_staged(all_labels, bins)
            else:
                bins, _ = self._assign_by_fit(all_labels, bins)
            if self.recluster_residual:
                bins = self._recluster_residual(all_labels, bins)

        binned = set.union(*bins) if bins else set()
        unbinned = set(l for l in all_labels if l not in binned)

        if self.verbose:
            print(f"TOTAL BIN TIME: {int(time.time()-start)} s", flush=True)

        return bins, unbinned

    def _select_labels(self) -> list[str]:
        selection = select_by_length_percentile(
            self.sequences, self.selection_percentiles[0], self.selection_percentiles[1]
        )
        if self.verbose and not self.compact:
            print(selection_report(self.sequences, selection))
        return selection

    def _report_distance_stats(self, labels: list[str], stack: npt.NDArray[np.float64]) -> None:
        """The pairwise-distance summary DistBin prints, when n allows it.

        Guarded by `stats_max_nodes` because it needs the dense n x n matrix
        the rest of the pipeline is careful never to build.
        """
        if len(labels) > self.stats_max_nodes:
            print(f"skipping distance stats for {len(labels)} contigs (over stats_max_nodes)")
            return
        matrix = tv_distance_matrix(stack)
        print(f"overall stats for all {len(labels)} contigs:")
        print(distance_stats(matrix))
        if self.labeled:
            selected_label_stats(labels, matrix, self.kmer_distributions, self.prefix_len)

    def _one_line_bin_stats(self, bins: list[set[str]], unbinned: set[str] = set()) -> str:
        num_binned = len(set([x for b in bins for x in b]))
        total = num_binned + len(unbinned)
        stats = f"{len(bins)} bins ({num_binned}/{total}) "
        if self.labeled:
            return stats + grade_bins(bins, unbinned, prefix_len=self.prefix_len)
        else:
            return stats + str([len(b) for b in bins[:10]])

    def _build_graph(self, labels: list[str]) -> npt.NDArray[np.int64]:
        n = len(labels)
        stack = stack_distributions(self.kmer_distributions, labels)
        if self.verbose and not self.compact:
            self._report_distance_stats(labels, stack)

        n_neighbors = self.n_neighbors if self.n_neighbors > 0 else auto_n_neighbors(n)
        knn_idx = knn_indices_from_stack(stack, n_neighbors, self.block_size)
        edges = mutual_knn_edges(knn_idx, mutual=self.mutual)

        if self.verbose:
            kind = "mutual" if self.mutual else "directed"
            verb = f"Built {kind} {n_neighbors}-NN graph.\n"
            print(f"K.0. {n_neighbors if self.compact else verb}", graph_stats(n, edges))

        return edges

    def _partition(self, labels: list[str], edges: npt.NDArray[np.int64]) -> list[set[str]]:
        # .tolist() rather than a comprehension: there can be millions of edges
        graph = ig.Graph(n=len(labels), edges=edges.tolist(), directed=False)
        partition = leidenalg.find_partition(
            graph,
            PARTITION_TYPES[self.quality],
            resolution_parameter=self.resolution,
            n_iterations=self.n_iterations,
            seed=self.seed,
        )
        bins = [set(labels[i] for i in community) for community in partition]
        bins = sorted([b for b in bins if b], key=len, reverse=True)

        if self.verbose:
            verb = f"Leiden ({self.quality}, resolution {self.resolution}).\n"
            print(
                f"K.1. {self.resolution if self.compact else verb}",
                self._one_line_bin_stats(bins),
                all_pure(bins, self.prefix_len) if self.labeled else " ",
            )
        return bins

    def _drop_small_bins(self, bins: list[set[str]]) -> tuple[list[set[str]], set[str]]:
        """Leiden on a mutual-kNN graph leaves isolated nodes as singletons."""
        kept = [b for b in bins if len(b) >= self.min_bin_size]
        dropped = set().union(*[b for b in bins if len(b) < self.min_bin_size]) if len(kept) < len(bins) else set()

        if self.verbose:
            verb = f"Dropped bins smaller than {self.min_bin_size}.\n"
            print(f"K.2. {self.min_bin_size if self.compact else verb}", self._one_line_bin_stats(kept, dropped))
        return kept, dropped

    def _bin_references(self, bins: list[set[str]]) -> tuple[
        npt.NDArray[np.float64],
        list[npt.NDArray[np.float64]],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Centroid, member sample, and internal-distance scale for each bin.

        The internal distances are what put every bin on its own scale: a tight
        bin and a diffuse one are then comparable, which a raw distance
        comparison never is.
        """
        rng = np.random.default_rng(self.seed)
        members = [sorted(b) for b in bins]
        samples = [rng.choice(m, size=min(self.assign_sample, len(m)), replace=False) for m in members]
        stacks = [stack_distributions(self.kmer_distributions, list(s)) for s in samples]
        centroids = np.stack([np.median(stack_distributions(self.kmer_distributions, m), axis=0) for m in members])

        intra = [tv_cdist(s, s)[np.triu_indices(len(s), 1)] for s in stacks]
        mean = np.array([i.mean() if i.size else 0.0 for i in intra])
        spread = np.array([i.std() if i.size else 0.0 for i in intra]) + SPREAD_FLOOR
        widest = np.array([i.max() if i.size else np.inf for i in intra])
        return centroids, stacks, mean, spread, widest

    def _fit_scores(
        self, chunk_stack: npt.NDArray[np.float64], top: npt.NDArray[np.int64], stacks: list[npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        """Mean distance from each candidate to the sampled members of its top bins.

        Looped over bins rather than candidates: each bin appears in far fewer
        shortlists than there are candidates, so this is a handful of large
        cdist calls instead of one tiny call per (candidate, bin) pair.
        """
        d = np.empty(top.shape, dtype=np.float64)
        for column, bin_index in enumerate(np.unique(top)):
            rows, slots = np.where(top == bin_index)
            if rows.size == 0:
                continue
            d[rows, slots] = tv_cdist(chunk_stack[rows], stacks[bin_index]).mean(axis=1)
        return d

    def _keep_largest(self, bins: list[set[str]]) -> list[set[str]]:
        """Keep only the `max_bins` largest bins as assignment seeds.

        Selection, not merging: the discarded bins' contigs are not lost, they
        go back into the candidate pool and K.3 re-judges each one individually.
        That is why this is cheap where merging is ruinous -- reducing 3463 bins
        to 64 by merging costs 0.43 accuracy, by selection 0.006.

        Ordering by size beat six alternatives (size/radius, size/sd,
        sqrt-size/radius, two non-redundancy rules, farthest-point) across 13
        mixtures: it wins or ties on most and is never badly wrong, where the
        diversity-based rules lose by up to 0.23.  Seeds want to be populated,
        not distinctive.
        """
        if self.max_bins <= 0 or len(bins) <= self.max_bins:
            return bins
        kept = sorted(bins, key=len, reverse=True)[: self.max_bins]

        if self.verbose:
            verb = f"Kept the {self.max_bins} largest bins as seeds.\n"
            print(f"K.2b. {self.max_bins if self.compact else verb}", self._one_line_bin_stats(kept))
        return kept

    def _score_candidates(
        self,
        candidates: list[str],
        centroids: npt.NDArray[np.float64],
        stacks: list[npt.NDArray[np.float64]],
        mean: npt.NDArray[np.float64],
        spread: npt.NDArray[np.float64],
        top_n: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Best-fitting bin per candidate, as (z, d, bin index).

        Scored in full before any admission decision, because a coverage target
        is a quantile of z and cannot be decided chunk by chunk.
        """
        best_z = np.empty(len(candidates), dtype=np.float64)
        best_d = np.empty(len(candidates), dtype=np.float64)
        winners = np.empty(len(candidates), dtype=np.int64)
        for start in range(0, len(candidates), self.block_size):
            stop = min(start + self.block_size, len(candidates))
            chunk_stack = stack_distributions(self.kmer_distributions, candidates[start:stop])
            # shortlist by centroid first; scoring every bin in full is wasted work
            top = np.astype(np.argsort(tv_cdist(chunk_stack, centroids), axis=1)[:, :top_n], np.int64)
            d = self._fit_scores(chunk_stack, top, stacks)
            z = (d - mean[top]) / spread[top]
            rows = np.arange(stop - start)
            pick = np.argmin(z, axis=1)
            best_z[start:stop] = z[rows, pick]
            best_d[start:stop] = d[rows, pick]
            winners[start:stop] = top[rows, pick]
        return best_z, best_d, winners

    def _assign_staged(self, labels: list[str], bins: list[set[str]]) -> list[set[str]]:
        """Assign in length tranches, longest first, recomputing envelopes between.

        Each stage takes the contigs at least as long as a length percentile,
        admits the best `stage_admit` fraction by z, and defers the rest to a
        later stage where the enlarged bins may judge them better.

        The point is that seeds cover only 6-10% of a mixture, so mu_C and
        sigma_C start out estimated from a thin slice; every stage improves them
        for the next.  Ordering by length matters because a short contig's
        distances are mostly sampling noise -- populate the bins with reliable
        long contigs first.

        Measured against a single pass: +0.060 on the pair whose two organisms
        differ systematically in contig length (its recall parity jumps 0.35 ->
        0.96 exactly at the p75 -> p50 boundary), and within 0.004 on five of the
        other seven.  So it is free-or-better, but the gain is specific to
        length-skewed mixtures rather than general.
        """
        lengths = {l: len(self.sequences[l]) for l in labels}
        all_lengths = np.fromiter(lengths.values(), dtype=np.float64, count=len(lengths))
        for stage, pct in enumerate(self.length_stages, 1):
            assigned = set.union(*bins) if bins else set()
            floor = float(np.percentile(all_lengths, pct))
            tranche = [l for l in labels if l not in assigned and lengths[l] >= floor]
            if not tranche:
                continue
            centroids, stacks, mean, spread, _ = self._bin_references(bins)
            top_n = min(self.assign_top_bins, len(bins))
            z, _, winners = self._score_candidates(tranche, centroids, stacks, mean, spread, top_n)
            keep = max(1, int(self.stage_admit * len(tranche)))
            cut = np.partition(z, keep - 1)[keep - 1]
            for i, label in enumerate(tranche):
                if z[i] <= cut:
                    bins[winners[i]].add(label)
            if self.verbose:
                verb = f"Stage {stage}: length >= p{pct:g}, {len(tranche)} candidates.\n"
                print(f"K.3.{stage} {pct if self.compact else verb}", self._one_line_bin_stats(bins))
        return bins

    def _recluster_residual(self, labels: list[str], bins: list[set[str]]) -> list[set[str]]:
        """Re-run the graph on whatever is still unassigned and keep what clusters.

        A species whose bins never reached the seed threshold has no seed at all,
        so the assignment steps can only misfile it.  This gives it a chance to
        form its own bins instead.  It is the one component here that can create
        a bin for a species the seeding missed.

        Measured on eight pairs it recovered 5.0-8.0% of each mixture at
        0.717-0.958 purity, and on two pairs the new bins were *purer* than the
        main assignment.  It never hurt.
        """
        assigned = set.union(*bins) if bins else set()
        residual = [l for l in labels if l not in assigned]
        if len(residual) < max(2 * self.residual_min_bin_size, 200):
            return bins

        stack = stack_distributions(self.kmer_distributions, residual)
        n_neighbors = self.n_neighbors if self.n_neighbors > 0 else auto_n_neighbors(len(residual))
        edges = mutual_knn_edges(knn_indices_from_stack(stack, n_neighbors, self.block_size), mutual=self.mutual)
        fresh = [c for c in self._partition(residual, edges) if len(c) >= self.residual_min_bin_size]

        if self.verbose:
            verb = f"Reclustered {len(residual)} residual contigs into {len(fresh)} bins.\n"
            print(f"K.4. {len(fresh) if self.compact else verb}", self._one_line_bin_stats(bins + fresh))
        return bins + fresh

    def _assign_by_fit(self, labels: list[str], bins: list[set[str]]) -> tuple[list[set[str]], set[str]]:
        """Assign each remaining contig to the bin whose own scale it best fits.

        For candidate p and bin C, z_C = (mean d(p, C) - mu_C) / sigma_C, where
        mu_C and sigma_C come from C's own internal distances.  p joins argmin z.

        Two ways to refuse a poor fit, and they are not equivalent:

        `target_coverage` admits the best-fitting candidates until that fraction
        of all contigs is binned.  This is the portable form.  Measured over 13
        mixtures, gating to 90% coverage raised mean accuracy from 0.834 to
        0.847 and the mean margin over DBV from +0.012 to +0.025.

        `z_max` refuses anything above a fixed z (and beyond C's widest internal
        distance).  Kept for control, but z is *not* comparable across datasets:
        reaching 90% coverage needed z between 3.3 and 6.1 on EukProt pairs and
        between 11.8 and 16.9 on choanoflagellate ones.  A fixed z=6 leaves
        ~95% coverage on the former and ~60% on the latter, which is why early
        fixed-threshold experiments looked like the gate always lost.
        """
        new_bins = [set(b) for b in bins]
        if not new_bins:
            return new_bins, set(labels)

        centroids, stacks, mean, spread, widest = self._bin_references(new_bins)
        binned = set.union(*new_bins)
        candidates = [l for l in labels if l not in binned]
        top_n = min(self.assign_top_bins, len(new_bins))

        best_z, best_d, winners = self._score_candidates(candidates, centroids, stacks, mean, spread, top_n)

        admissible = np.ones(len(candidates), dtype=bool)
        if self.target_coverage < 1.0:
            room = int(round(self.target_coverage * len(labels))) - len(binned)
            if room <= 0:
                admissible[:] = False
            elif room < len(candidates):
                admissible = best_z <= np.partition(best_z, room - 1)[room - 1]
        elif math.isfinite(self.z_max):
            admissible = (best_z <= self.z_max) & (best_d <= widest[winners])

        unbinned: set[str] = set()
        for i, label in enumerate(candidates):
            if admissible[i]:
                new_bins[winners[i]].add(label)
            else:
                unbinned.add(label)

        new_bins = sorted(new_bins, key=len, reverse=True)

        if self.verbose:
            how = f"target coverage {self.target_coverage:.0%}" if self.target_coverage < 1.0 else f"z <= {self.z_max}"
            verb = f"Assigned remaining contigs to the best-fitting bin ({how}).\n"
            print(f"K.3. {how if self.compact else verb}", self._one_line_bin_stats(new_bins, unbinned))

        return new_bins, unbinned
