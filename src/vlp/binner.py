"""
Variable Length Pattern clustering
Uses NumPy structured arrays for fast 128-bit integer sorting.
"""

from lib.encoded_sequences import Sequence
from lib.evaluate import sequence_stats

from .kmer_clustering import make_4mer_clusters
from .scoring import find_best_scores, remove_next_cluster


class VLPBinner:
    def __init__(
        self,
        min_seq_len: int = 3000,
        min_cluster_size: int = 6,
        xchi_thresh: float = 255.0,
        thresh_next: int = 0,
        verbose: bool = False,
        **kwargs: str,
    ) -> None:
        self.min_seq_len = min_seq_len
        self.min_cluster_size = min_cluster_size
        self.xchi_thresh = xchi_thresh
        self.thresh_next = thresh_next
        self.verbose = verbose
        self.sequences: list[Sequence] = []

    def run(
        self,
    ) -> None:
        """Group sequences with similar tetragraph profiles using
        chi-squared similarity and transitive closure.

        Gradually lower the minimum sequence length threshold
        activating shorter sequences at each step and re-optimizing
        cluster assignments.

        At each threshold level, repeatedly remove the weakest cluster
        (the one whose members score best under a different cluster)
        and reassign its sequences, until no more clusters are worth
        removing.
        """
        if self.verbose:
            print("Sequence Lengths: ", sequence_stats([len(s.seq) for s in self.sequences]))

        num_clusters = make_4mer_clusters(
            self.sequences, self.min_seq_len, self.min_cluster_size, self.xchi_thresh, verbose=self.verbose
        )

        # Minimum sequence length thresholds for successive clustering rounds
        for len_thresh in [3000, 2000, 1000, 1]:
            if self.verbose:
                print(f"\nLENGTH THRESHOLD: {len_thresh}")
            for n in range(len(self.sequences)):
                if len(self.sequences[n].seq) >= len_thresh:
                    self.sequences[n].active = True
            find_best_scores(num_clusters, self.sequences, False)

            while True:
                kbest, xbest_score = remove_next_cluster(self, num_clusters, self.thresh_next, self.verbose)

                if xbest_score < self.thresh_next:
                    break
                find_best_scores(num_clusters, self.sequences, verbose=False)
