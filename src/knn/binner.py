"""Mutual-kNN + Leiden binning.

K.0 build a mutual-kNN graph over the length-percentile selection
K.1 partition it with Leiden (CPM), sweeping resolution down to `resolution_min`
K.2 drop bins below `min_bin_size`; optionally keep only the `max_bins` largest
K.3 assign the remaining contigs to the bin whose own scale they best fit, in
    length tranches
K.4 recluster whatever is still unassigned into new bins
"""

import time

import igraph as ig
import leidenalg
import numpy as np
import numpy.typing as npt

from distbin.evaluate import distance_stats, selected_label_stats
from knn.graph import (
    BLOCK_SIZE,
    capped_n_neighbors,
    graph_stats,
    knn_indices_from_stack,
    mutual_knn_edges,
)
from lib.distances import kmer_distributions, stack_distributions, tv_cdist, tv_distance_matrix
from lib.evaluate import all_pure, grade_bins
from lib.selection import select_by_length_percentile, selection_report

SPREAD_FLOOR = 1e-9

# Leiden: CPM, swept down from RESOLUTION_START by SWEEP_FACTOR to `resolution_min`
RESOLUTION_START = 0.5
SWEEP_FACTOR = 0.7
N_ITERATIONS = 2

# K.3: shortlist each candidate's ASSIGN_TOP_BINS nearest centroids, then score it
# by its mean distance to its ASSIGN_NEAREST nearest of ASSIGN_SAMPLE sampled
# members per bin
ASSIGN_TOP_BINS = 5
ASSIGN_SAMPLE = 200
ASSIGN_NEAREST = 5

# K.3 tranches, as length percentiles, longest first; each admits the best
# STAGE_ADMIT fraction by z
LENGTH_STAGES = (75.0, 50.0, 25.0, 0.0)
STAGE_ADMIT = 0.75

# K.4: residual bins smaller than this are dropped
RESIDUAL_MIN_BIN_SIZE = 8

# verbose pairwise-distance stats need a dense n x n matrix; skipped above this
STATS_MAX_NODES = 6000


class Binner:
    def __init__(
        self,
        sequences: dict[str, str] | dict[str, npt.NDArray[np.int8]],
        k: int = 4,
        selection_percentiles_low: float = 45.0,
        selection_percentiles_high: float = 95.0,
        n_neighbors: int = 1024,
        neighbor_cap: int = 8,
        resolution_min: float = 0.002,
        seed: int = 0,
        min_bin_size: int = 32,
        max_bins: int = 0,
        verbose: bool = False,
        compact: bool = False,
        labeled: bool = False,
        prefix_len: int = 2,
        **kwargs: str,
    ):
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
        self.k = k
        self.sequences = sequences
        self.kmer_distributions = kmer_distributions(sequences, k)
        self.selection_percentiles = (selection_percentiles_low, selection_percentiles_high)
        self.n_neighbors = n_neighbors
        self.neighbor_cap = neighbor_cap
        # k actually used for the main graph, after the cap; set by _build_graph
        self._graph_k = n_neighbors
        self.resolution_min = resolution_min
        self.seed = seed
        self.min_bin_size = min_bin_size
        self.max_bins = max_bins
        self.verbose = verbose
        self.compact = compact
        if self.compact:
            self.verbose = True
        self.labeled = labeled
        self.prefix_len = prefix_len

    def run(self) -> tuple[list[set[str]], set[str]]:
        """Bin all sequences and return (bins, unbinned)."""
        start = time.time()
        all_labels = list(self.sequences.keys())
        selection = self._select_labels()

        edges = self._build_graph(selection)
        bins = self._partition(selection, edges)
        bins, _ = self._drop_small_bins(bins)
        bins = self._keep_largest(bins)
        # staging sees every label, so contigs from bins dropped at K.2 get a
        # second chance; the contigs it declines are what K.4 works on
        bins = self._assign_staged(all_labels, bins)
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
        """The pairwise-distance summary DistBin prints, up to STATS_MAX_NODES."""
        if len(labels) > STATS_MAX_NODES:
            print(f"skipping distance stats for {len(labels)} contigs (over {STATS_MAX_NODES})")
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

        n_neighbors = self._graph_k = capped_n_neighbors(self.n_neighbors, n, self.neighbor_cap)
        edges = mutual_knn_edges(knn_indices_from_stack(stack, n_neighbors, BLOCK_SIZE))

        if self.verbose:
            verb = f"Built mutual {n_neighbors}-NN graph.\n"
            print(f"K.0. {n_neighbors if self.compact else verb}", graph_stats(n, edges))

        return edges

    def _partition(self, labels: list[str], edges: npt.NDArray[np.int64], sweep: bool = True) -> list[set[str]]:
        # .tolist() rather than a comprehension: there can be millions of edges
        graph = ig.Graph(n=len(labels), edges=edges.tolist(), directed=False)
        if sweep:
            membership, resolution = self._sweep_resolution(graph)
        else:
            resolution = RESOLUTION_START
            membership = self._leiden(graph, resolution).membership
        communities: dict[int, set[str]] = {}
        for i, community in enumerate(membership):
            communities.setdefault(community, set()).add(labels[i])
        bins = sorted(communities.values(), key=len, reverse=True)

        if self.verbose:
            verb = f"Leiden (CPM, resolution {resolution:.3g}).\n"
            print(
                f"K.1. {resolution if self.compact else verb}",
                self._one_line_bin_stats(bins),
                all_pure(bins, self.prefix_len) if self.labeled else " ",
            )
        return bins

    def _leiden(
        self, graph: ig.Graph, resolution: float, initial: list[int] | None = None
    ) -> leidenalg.VertexPartition:
        return leidenalg.find_partition(
            graph,
            leidenalg.CPMVertexPartition,
            initial_membership=initial,
            resolution_parameter=resolution,
            n_iterations=N_ITERATIONS,
            seed=self.seed,
        )

    def _sweep_resolution(self, graph: ig.Graph) -> tuple[list[int], float]:
        """Coarsen by lowering resolution from RESOLUTION_START to `resolution_min`.

        Each step warm-starts Leiden from the previous membership, so the
        partition coarsens continuously rather than being re-drawn.
        """
        resolution = RESOLUTION_START
        membership = self._leiden(graph, resolution).membership
        while resolution * SWEEP_FACTOR >= self.resolution_min:
            resolution *= SWEEP_FACTOR
            membership = self._leiden(graph, resolution, membership).membership

        if self.verbose and not self.compact:
            print(f"Swept resolution to {resolution:.3g}.")
        return membership, resolution

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
    ]:
        """Centroid, member sample, and internal-distance scale for each bin.

        The scale is over each sampled member's mean distance to its
        ASSIGN_NEAREST nearest fellow members, which stays local in multi-modal bins.
        """
        rng = np.random.default_rng(self.seed)
        members = [sorted(b) for b in bins]
        samples = [rng.choice(m, size=min(ASSIGN_SAMPLE, len(m)), replace=False) for m in members]
        stacks = [stack_distributions(self.kmer_distributions, list(s)) for s in samples]
        centroids = np.stack([np.median(stack_distributions(self.kmer_distributions, m), axis=0) for m in members])

        intra = [_nearest_mean(tv_cdist(s, s), ASSIGN_NEAREST, exclude_self=True) for s in stacks]
        mean = np.array([i.mean() if i.size else 0.0 for i in intra])
        spread = np.array([i.std() if i.size else 0.0 for i in intra]) + SPREAD_FLOOR
        return centroids, stacks, mean, spread

    def _fit_scores(
        self, chunk_stack: npt.NDArray[np.float64], top: npt.NDArray[np.int64], stacks: list[npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        """Mean distance from each candidate to its ASSIGN_NEAREST nearest sampled
        members in each of its top bins.  Looped over bins for fewer, larger cdists.
        """
        d = np.empty(top.shape, dtype=np.float64)
        for column, bin_index in enumerate(np.unique(top)):
            rows, slots = np.where(top == bin_index)
            if rows.size == 0:
                continue
            block = tv_cdist(chunk_stack[rows], stacks[bin_index])
            d[rows, slots] = _nearest_mean(block, ASSIGN_NEAREST)
        return d

    def _keep_largest(self, bins: list[set[str]]) -> list[set[str]]:
        """Keep only the `max_bins` largest bins as assignment seeds.

        Selection, not merging: the discarded bins' contigs go back into the
        candidate pool for K.3.
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
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Best-fitting bin per candidate, as (z, bin index)."""
        best_z = np.empty(len(candidates), dtype=np.float64)
        winners = np.empty(len(candidates), dtype=np.int64)
        for start in range(0, len(candidates), BLOCK_SIZE):
            stop = min(start + BLOCK_SIZE, len(candidates))
            chunk_stack = stack_distributions(self.kmer_distributions, candidates[start:stop])
            # shortlist by centroid first; scoring every bin in full is wasted work
            top = np.astype(np.argsort(tv_cdist(chunk_stack, centroids), axis=1)[:, :top_n], np.int64)
            d = self._fit_scores(chunk_stack, top, stacks)
            z = (d - mean[top]) / spread[top]
            rows = np.arange(stop - start)
            pick = np.argmin(z, axis=1)
            best_z[start:stop] = z[rows, pick]
            winners[start:stop] = top[rows, pick]
        return best_z, winners

    def _assign_staged(self, labels: list[str], bins: list[set[str]]) -> list[set[str]]:
        """Assign in length tranches, longest first, recomputing envelopes between.

        For candidate p and bin C, z_C = (d(p, C) - mu_C) / sigma_C, from C's
        own internal distances; p goes to argmin z.  Each stage admits the best
        STAGE_ADMIT fraction of its tranche and defers the rest.
        """
        if not bins:
            return bins
        lengths = {l: len(self.sequences[l]) for l in labels}
        all_lengths = np.fromiter(lengths.values(), dtype=np.float64, count=len(lengths))
        for stage, pct in enumerate(LENGTH_STAGES, 1):
            assigned = set.union(*bins) if bins else set()
            floor = float(np.percentile(all_lengths, pct))
            tranche = [l for l in labels if l not in assigned and lengths[l] >= floor]
            if not tranche:
                continue
            centroids, stacks, mean, spread = self._bin_references(bins)
            top_n = min(ASSIGN_TOP_BINS, len(bins))
            z, winners = self._score_candidates(tranche, centroids, stacks, mean, spread, top_n)
            keep = max(1, int(STAGE_ADMIT * len(tranche)))
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

        This is the only way a species the seeding missed can get a bin.
        """
        assigned = set.union(*bins) if bins else set()
        residual = [l for l in labels if l not in assigned]
        if len(residual) < max(2 * RESIDUAL_MIN_BIN_SIZE, 200):
            return bins

        stack = stack_distributions(self.kmer_distributions, residual)
        # k follows the main graph's cap; capping on the residual over-splits it
        edges = mutual_knn_edges(knn_indices_from_stack(stack, self._graph_k, BLOCK_SIZE))
        # unswept: the residual is cut finely
        fresh = [c for c in self._partition(residual, edges, sweep=False) if len(c) >= RESIDUAL_MIN_BIN_SIZE]

        if self.verbose:
            verb = f"Reclustered {len(residual)} residual contigs into {len(fresh)} bins.\n"
            print(f"K.4. {len(fresh) if self.compact else verb}", self._one_line_bin_stats(bins + fresh))
        return bins + fresh


def _nearest_mean(d: npt.NDArray[np.float64], m: int, exclude_self: bool = False) -> npt.NDArray[np.float64]:
    """Mean of each row's m smallest distances, m capped at columns - 1."""
    m = min(m, max(d.shape[1] - 1, 1))
    if exclude_self:
        if d.shape[1] < 2:
            return np.zeros(0)
        d = d.copy()
        np.fill_diagonal(d, np.inf)
    return np.partition(d, m - 1, axis=1)[:, :m].mean(axis=1)
