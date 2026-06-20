"""
DistBin + Variable Length Pattern clustering
"""

import numpy as np

from distbin.binner import Binner as DBBinner
from lib.encoded_sequences import Sequence
from lib.evaluate import bins_by_id, grade_bins

from .scoring import ClusterScore, compute_scores, reassign, weakest_cluster

DEFAULT_CLUSTER_THRESHOLDS = (0.02, 0.03125, 0.04, 0.05, 0.0625)


class DBVBinner:
    def __init__(
        self,
        min_seq_len: int = 3000,
        min_cluster_size: int = 6,
        xchi_thresh: float = 255.0,
        absorb_thresh: float = 0.5,
        cv_outlier_ratio: float = 1.25,
        skip_remove: bool = False,
        k: int = 4,
        selection_percentiles_low: float = 85.0,
        selection_percentiles_high: float = 95.0,
        creation_threshold: float = 0.1,
        selection_threshold: float = 0.175,
        cluster_thresholds: tuple[float, ...] = DEFAULT_CLUSTER_THRESHOLDS,
        reduction_constant: float = 16.0,
        final_threshold: float = 0.175,
        cull_ll_tolerance: float = 0.0,
        minimal: bool = False,
        verbose: bool = False,
        labeled: bool = False,
        prefix_len: int = 2,
        **kwargs: str,
    ) -> None:
        self.min_seq_len = min_seq_len
        self.min_cluster_size = min_cluster_size
        self.xchi_thresh = xchi_thresh
        self.absorb_thresh = absorb_thresh
        self.cv_outlier_ratio = cv_outlier_ratio
        self.skip_remove = skip_remove
        self.k = k
        self.selection_percentiles_low = selection_percentiles_low
        self.selection_percentiles_high = selection_percentiles_high
        self.creation_threshold = creation_threshold
        self.selection_threshold = selection_threshold
        self.cluster_thresholds = cluster_thresholds
        self.reduction_constant = reduction_constant
        self.final_threshold = final_threshold
        self.cull_ll_tolerance = cull_ll_tolerance
        self.minimal = minimal
        self.verbose = verbose
        self.labeled = labeled
        self.prefix_len = prefix_len
        self.sequences: list[Sequence] = []

    def _update_sequences(self, bins: list[set[str]], unbinned: set[str]) -> None:
        cluster_by_id = {seq_id: cid for cid, binn in enumerate(bins) for seq_id in binn}
        for s in self.sequences:
            if s.seq_id in cluster_by_id:
                s.cluster = cluster_by_id[s.seq_id]
                s.active = True
            elif s.seq_id in unbinned:
                s.cluster = -1
                s.active = False
            else:
                raise Exception(f"seq {s.seq_id} not found!")

    def _run_distbin(self) -> int:
        sequences = {s.seq_id: s.oriented() for s in self.sequences}
        dbb = DBBinner(
            sequences,
            k=self.k,
            verbose=self.verbose,
            selection_percentiles=(self.selection_percentiles_low, self.selection_percentiles_high),
            creation_threshold=self.creation_threshold,
            selection_threshold=self.selection_threshold,
            cluster_thresholds=self.cluster_thresholds,
            reduction_constant=self.reduction_constant,
            final_threshold=self.final_threshold,
            labeled=self.labeled,
            minimal=self.minimal,
            prefix_len=self.prefix_len,
        )
        bins, unbinned = dbb.run()
        if self.verbose:
            print([len(b) for b in bins], len(unbinned))
            if self.labeled:
                print("GRADE FOR DISTBIN:", grade_bins(bins, unbinned, prefix_len=self.prefix_len), flush=True)
        self._update_sequences(bins, unbinned)
        return len(bins)

    def _len_threshs(self) -> list[int]:
        percentiles = [50, 25, 10]
        lengths = [len(s.seq) for s in self.sequences]
        return [int(np.percentile(lengths, p)) for p in percentiles] + [1]

    def _remove_cluster(self, cluster_id: int) -> None:
        for s in self.sequences:
            if s.cluster == cluster_id:
                s.cluster = -1

    def _per_base_loglik(self, scores: list[ClusterScore]) -> float:
        """Mean per-base log-likelihood of the current assignment.

        Sum each assigned, active sequence's score under its own cluster model,
        divided by the total assigned bases. Merging two distinct species forces
        one species' members onto a foreign model where they score worse per base,
        so this value drops sharply on a bad cull and stays flat/rises on a good
        one (consolidating fragments of the same species).
        """
        total_ll = 0.0
        total_bases = 0
        for i, s in enumerate(self.sequences):
            if not s.active or s.cluster == -1:
                continue
            sc = float(scores[s.cluster].scores[i])
            if sc == -np.inf:
                continue
            total_ll += sc
            total_bases += len(s.seq)
        if total_bases == 0:
            return -np.inf
        return total_ll / total_bases

    def run(self) -> None:
        num_clusters = self._run_distbin()
        for len_thresh in self._len_threshs():
            if self.verbose:
                print(f"\nLENGTH THRESHOLD: {len_thresh}")
            for s in self.sequences:
                if len(s.seq) >= len_thresh:
                    s.active = True

            scores = compute_scores(num_clusters, self.sequences, self.verbose)
            reassign(scores, self.sequences, self.verbose)

            if self.verbose and self.labeled:
                print(
                    "GRADE AFTER S&R",
                    grade_bins(*bins_by_id(self.sequences), limit=20, prefix_len=self.prefix_len),
                    flush=True,
                )

            scores = compute_scores(num_clusters, self.sequences, self.verbose)
            while not self.skip_remove:
                weakest_id = weakest_cluster(
                    scores, self.sequences, self.absorb_thresh, self.cv_outlier_ratio, self.verbose
                )
                if weakest_id == -1:
                    break

                # Tentatively cull the weakest cluster, then accept only if the
                # assignment's per-base log-likelihood does not drop by more than
                # cull_ll_tolerance. A good cull (consolidating same-species
                # fragments) holds the likelihood flat or improves it; a bad cull
                # (merging two species) tanks it, so we roll back and stop.
                before_ll = self._per_base_loglik(scores)
                saved = [s.cluster for s in self.sequences]
                self._remove_cluster(weakest_id)
                scores[weakest_id] = ClusterScore(len(self.sequences))  # re-init the scores for this cluster
                reassign(scores, self.sequences, self.verbose)
                new_scores = compute_scores(num_clusters, self.sequences, self.verbose)
                after_ll = self._per_base_loglik(new_scores)

                if after_ll < before_ll - self.cull_ll_tolerance:
                    for s, c in zip(self.sequences, saved):
                        s.cluster = c
                    if self.verbose:
                        print(
                            f"Rejecting cull of {weakest_id}: "
                            f"per-base loglik {before_ll:.5f} -> {after_ll:.5f}",
                            flush=True,
                        )
                    break

                scores = new_scores
                if self.verbose:
                    print(f"Accepted cull of {weakest_id}: per-base loglik {before_ll:.5f} -> {after_ll:.5f}")
                if self.verbose and self.labeled:
                    print(
                        "GRADE AFTER CULL",
                        grade_bins(*bins_by_id(self.sequences), prefix_len=self.prefix_len, limit=20),
                        flush=True,
                    )

            if self.verbose and self.labeled:
                print(
                    "GRADE AFTER S&W&R",
                    grade_bins(*bins_by_id(self.sequences), limit=20, prefix_len=self.prefix_len),
                    flush=True,
                )
