"""KLV: KNN + Leiden bins refined and merged by VLP trie models.

KNN seeds VLP with its `vlp_seed_bins` largest bins; the rest of their contigs
start unbinned.  DBV's length schedule then activates and reassigns contigs,
and clusters are merged by dissolving one at a time into the others.

Two ways to run the merge phase:

`vlp_holdout` (default): each cluster gets one trie per fold of a random split,
and each contig is scored by the other fold's trie.  Each step dissolves the
most absorbable cluster; the merge stops, undoing the last step, when held-out
log-likelihood per base over all active contigs falls by more than
`vlp_tolerance`.  Otherwise, DBV's own in-sample merge phase runs unchanged.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from dbv.binner import DBVBinner
from knn.binner import Binner
from lib.encoded_sequences import Sequence
from vlp.vlp_trie import VLPTrie

# clusters with fewer active members (per fold, when held out) are dissolved,
# as DBV's compute_scores does
MIN_CLUSTER = 3


# per cluster: the members its tries were trained on, its score row, its trie sizes
RowCache = dict[int, tuple[list[int], npt.NDArray[np.float64], int]]


def score_matrix(
    sequences: list[Sequence],
    n_clusters: int,
    folds: npt.NDArray[np.int8] | None = None,
    cache: RowCache | None = None,
) -> tuple[npt.NDArray[np.float64], list[int]]:
    """Every active contig's score under every live cluster's trie, and trie sizes.

    With `folds` (0 or 1 per contig), each contig is scored by the trie trained
    on the other fold.  Clusters too small to model are dissolved (members -> -1).
    With `cache`, a cluster whose members are unchanged since the last call
    reuses its row instead of retraining; only valid while the active set and
    the folds stay fixed.
    """
    scores = np.full((n_clusters, len(sequences)), -np.inf)
    states = [0] * n_clusters
    active = [i for i, s in enumerate(sequences) if s.active]
    oriented = {i: sequences[i].oriented() for i in active}
    for k in range(n_clusters):
        members = [i for i in active if sequences[i].cluster == k]
        if cache is not None and k in cache and cache[k][0] == members:
            _, scores[k], states[k] = cache[k]
            continue
        parts = [members] if folds is None else [[i for i in members if folds[i] == f] for f in (0, 1)]
        if min(len(part) for part in parts) < MIN_CLUSTER:
            for i in members:
                sequences[i].cluster = -1
            continue
        tries = []
        for part in parts:
            trie = VLPTrie()
            trie.make_model(k, [sequences[i] for i in part])
            states[k] += trie.num_states
            tries.append(trie)
        for i in active:
            trie = tries[0] if folds is None else tries[1 - folds[i]]
            scores[k, i] = trie.score(oriented[i])[0]
        if cache is not None:
            cache[k] = (members, scores[k].copy(), states[k])
    return scores, states


def reassign(sequences: list[Sequence], scores: npt.NDArray[np.float64]) -> None:
    """Each active contig to its best-scoring cluster, or -1 if none beats background."""
    if scores.shape[0] == 0:
        for s in sequences:
            if s.active:
                s.cluster = -1
        return
    best = np.argmax(scores, axis=0)
    best_val = scores[best, np.arange(scores.shape[1])]
    for i, s in enumerate(sequences):
        if s.active:
            s.cluster = int(best[i]) if best_val[i] > 0.0 else -1


def absorbability(sequences: list[Sequence], scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Per cluster, mean best-other score over mean own score, as DBV's weakest_cluster."""
    cluster = np.array([s.cluster if s.active else -1 for s in sequences])
    absorb = np.full(scores.shape[0], -np.inf)
    for k in np.unique(cluster[cluster >= 0]):
        members = np.flatnonzero(cluster == k)
        own = scores[k, members].mean()
        others = np.delete(scores[:, members], k, axis=0)
        alt = others.max(axis=0).mean() if others.size else -np.inf
        absorb[k] = alt / max(own, 1.0)
    return absorb


def assignment_stats(sequences: list[Sequence], scores: npt.NDArray[np.float64], states: list[int]) -> dict[str, float]:
    """Label-free summary of the current assignment."""
    cluster = np.array([s.cluster if s.active else -1 for s in sequences])
    assigned = np.flatnonzero(cluster >= 0)
    own = scores[cluster[assigned], assigned]
    bases = sum(len(sequences[i].seq) for i in assigned)
    active_bases = sum(len(s.seq) for s in sequences if s.active)
    live = np.unique(cluster[assigned])
    return {
        "n_live": float(live.size),
        "total_score": float(own.sum()),
        "assigned": float(assigned.size),
        "assigned_bases": float(bases),
        "ll_per_base": float(own.sum() / bases) if bases else -np.inf,
        # unassigned contigs count as background (0): losing a species' last
        # cluster then costs fit, where ll_per_base would just drop them
        "ll_all": float(own.sum() / active_bases) if active_bases else -np.inf,
        "states": float(sum(states[k] for k in live)),
    }


def dissolve(sequences: list[Sequence], scores: npt.NDArray[np.float64], victim: int) -> None:
    """Unassign the victim's members and reassign them among the other clusters."""
    for s in sequences:
        if s.cluster == victim:
            s.cluster = -1
    scores[victim, :] = -np.inf
    reassign(sequences, scores)


class KLVBinner(DBVBinner):
    def __init__(
        self,
        vlp_seed_bins: int = 32,
        vlp_holdout: bool = True,
        vlp_tolerance: float = 0.002,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.vlp_seed_bins = vlp_seed_bins
        self.vlp_holdout = vlp_holdout
        self.vlp_tolerance = vlp_tolerance
        self.knn_params = params
        self.seed = int(params.get("seed", 0))

    def _run_distbin(self) -> int:
        """Seed from KNN; named for the DBVBinner step it replaces."""
        sequences = {s.seq_id: s.oriented() for s in self.sequences}
        bins, _ = Binner(sequences, **self.knn_params).run()
        seeds = sorted(bins, key=len, reverse=True)[: self.vlp_seed_bins]
        seeded = set().union(*seeds) if seeds else set()
        self._update_sequences(seeds, set(sequences) - seeded)
        if self.verbose:
            print(f"KNN seeds: {[len(b) for b in seeds]}, {len(sequences) - len(seeded)} unbinned", flush=True)
        return len(seeds)

    def run(self) -> None:
        if not self.vlp_holdout:
            super().run()
            return

        n_clusters = self._run_distbin()
        folds = np.random.default_rng(self.seed).integers(0, 2, len(self.sequences)).astype(np.int8)
        for threshold in self._len_threshs():
            for s in self.sequences:
                if len(s.seq) >= threshold:
                    s.active = True
            scores, _ = score_matrix(self.sequences, n_clusters, folds)
            reassign(self.sequences, scores)
        self._merge(n_clusters, folds)

    def _merge(self, n_clusters: int, folds: npt.NDArray[np.int8]) -> None:
        """Dissolve the most absorbable cluster until held-out fit drops by > vlp_tolerance.

        Stops without doing anything if no cluster survived to this point, which
        happens when every seed is too small to split into two modelled folds.
        A dissolve that leaves no modellable cluster is undone.
        """
        previous: list[int] | None = None
        previous_ll = -np.inf
        # every contig is active by now, so a cluster's row changes only with its members
        cache: RowCache = {}
        while True:
            scores, states = score_matrix(self.sequences, n_clusters, folds, cache)
            reassign(self.sequences, scores)
            stats = assignment_stats(self.sequences, scores, states)
            if stats["n_live"] == 0:
                if previous is not None:
                    for s, cluster in zip(self.sequences, previous):
                        s.cluster = cluster
                return
            if previous is not None and stats["ll_all"] < previous_ll - self.vlp_tolerance:
                for s, cluster in zip(self.sequences, previous):
                    s.cluster = cluster
                if self.verbose:
                    print(f"Stopped: held-out ll/base {previous_ll:.5f} -> {stats['ll_all']:.5f}", flush=True)
                return
            if self.verbose:
                print(f"{stats['n_live']:.0f} clusters, held-out ll/base {stats['ll_all']:.5f}", flush=True)
            previous = [s.cluster for s in self.sequences]
            previous_ll = stats["ll_all"]
            if stats["n_live"] <= 1:
                return
            dissolve(self.sequences, scores, int(np.argmax(absorbability(self.sequences, scores))))
