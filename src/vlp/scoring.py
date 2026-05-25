"""Score computation, best-score finding, and cluster removal."""

from typing import TYPE_CHECKING

import numpy as np

from lib.encoded_sequences import Sequence, reverse_complement

from .vlp_trie import VLPTrie

if TYPE_CHECKING:
    from .binner import VLPBinner


class VLPScore:
    """Per-cluster VLP model scores for every sequence.

    Attributes:
        score: Log-likelihood scores, shape (2, num_clusters+1, num_sequences).
            Axis 0 is orientation (0=forward, 1=reverse-complement).
            Axis 1 is cluster index (1-based; index 0 is for unassigned and unused).
            Axis 2 is sequence index.
        mean: Per-cluster mean score per base, shape (num_clusters+1,).
        var: Per-cluster score variance per base, shape (num_clusters+1,).
    """

    def __init__(self, num_clusters: int, num_sequences: int):
        self.score = np.zeros((2, num_clusters + 1, num_sequences), dtype=np.float32)
        self.mean = np.zeros(num_clusters + 1, dtype=np.float32)
        self.var = np.zeros(num_clusters + 1, dtype=np.float32)
        self.score[:, :, :] = -np.inf

    def score_cluster(
        self,
        k: int,
        vlp_trie: VLPTrie,
        sequences: list[Sequence],
    ) -> None:
        """score every active sequence against cluster k's model
        returns number of bases in cluster k
        """
        for i in range(len(sequences)):
            if not sequences[i].active:
                continue
            seq = sequences[i].seq

            # Score in forward orientation
            score_f, var_f = vlp_trie.score(seq)
            self.score[0, k, i] = score_f

            # Score in reverse-complement orientation
            cc_arr = reverse_complement(seq, True)
            score_r, var_r = vlp_trie.score(cc_arr)
            self.score[1, k, i] = score_r

            # For sequences in this cluster, accumulate stats and set direction
            if sequences[i].cluster == k:
                self.mean[k] += max(score_f, score_r)
                if score_r > score_f:
                    sequences[i].reverse = True
                    self.var[k] += var_r
                else:
                    self.var[k] += var_f
                    sequences[i].reverse = False

    def print_histogram(self, k: int, sequences: list[Sequence]) -> None:
        # Print histogram of z-scores for cluster k's model:
        # hist1 = sequences belonging to cluster k, hist2 = all others
        minv = -10
        maxv = 10
        hist1 = np.zeros(maxv - minv + 1, dtype=np.int32)
        hist2 = np.zeros(maxv - minv + 1, dtype=np.int32)

        for i in range(len(sequences)):
            if not sequences[i].active:
                continue
            max_score_k_i = max(self.score[0, k, i], self.score[1, k, i])
            seq_len = len(sequences[i].seq)
            sig = np.sqrt(seq_len * self.var[k]) if self.var[k] > 0 else 1.0
            m_val = int((max_score_k_i - seq_len * self.mean[k]) / sig)
            m_val = max(minv, min(maxv, m_val))
            idx = m_val - minv
            if k == sequences[i].cluster:
                hist1[idx] += 1
            if k != sequences[i].cluster:
                hist2[idx] += 1

        cnt_use = len([s for s in sequences if s.active])
        print(
            f"Histogram_train: num_msg{cnt_use:7d}  mean - var - sig"
            f"{self.mean[k]:10.5f}{self.var[k]:10.5f}{np.sqrt(self.var[k]):10.5f}"
        )
        print("".join(f"{i:6d}" for i in range(minv, maxv + 1)))
        print("".join(f"{hist1[i]:6d}" for i in range(maxv - minv + 1)) + f"{hist1.sum():6d}")
        print("".join(f"{hist2[i]:6d}" for i in range(maxv - minv + 1)) + f"{hist2.sum():6d}")


def _cluster_size(cluster_num: int, sequences: list[Sequence]) -> int:
    return len([s for s in sequences if s.cluster == cluster_num and s.active])


def _cluster_bases(cluster_num: int, sequences: list[Sequence]) -> int:
    return sum([len(s.seq) for s in sequences if s.cluster == cluster_num and s.active])


def _compute_scores(num_clusters: int, sequences: list[Sequence], verbose: bool) -> VLPScore:
    """Build a VLP model for each cluster and score every active sequence against it.

    For each cluster:
      1. Build a VLP trie model from the sequences currently assigned to it.
      2. Score every active sequence against that model in both forward and
         reverse-complement orientations.
      3. For sequences belonging to the cluster, accumulate per-base mean score
         and variance, and set direction to whichever orientation scored higher.

    Clusters with fewer than 3 members are dissolved (members set to cluster 0).
    """
    if verbose:
        print("\n    BUILDING: VLP model")
    vlp_score = VLPScore(num_clusters, len(sequences))

    for k in range(1, num_clusters + 1):
        # Dissolve clusters that are too small to build a meaningful model
        if _cluster_size(k, sequences) < 3:
            for n in range(len(sequences)):
                if sequences[n].cluster == k:
                    sequences[n].cluster = 0
            continue

        vlp_trie = VLPTrie()
        vlp_trie.make_model(k, sequences, verbose=verbose)

        vlp_score.score_cluster(k, vlp_trie, sequences)

        # Normalize mean and variance to per-base values
        num_bases = _cluster_bases(k, sequences)
        if num_bases > 0:
            vlp_score.mean[k] = vlp_score.mean[k] / num_bases
            vlp_score.var[k] = (vlp_score.var[k] / num_bases) - vlp_score.mean[k] * vlp_score.mean[k]

        if verbose:
            print(
                f"    cluster:{k:2d} size:{_cluster_size(k, sequences):7d} "
                + vlp_trie.state_string()
                + f" num_bases:{num_bases:9d} mean:{vlp_score.mean[k]:9.5f} var:{vlp_score.var[k]:9.5f}"
            )
            # vlp_score.print_histogram(k, sequences)

    # Unassign any sequences with invalid cluster numbers
    for n in range(len(sequences)):
        if sequences[n].cluster > num_clusters:
            sequences[n].cluster = 0

    return vlp_score


def find_best_scores(num_clusters: int, sequences: list[Sequence], verbose: bool) -> None:
    """Build a VLPTrie; iteratively reassign each active sequence to
    its highest-scoring cluster/direction until no assignments change
    (expectation-maximization style convergence).
    """

    if verbose:
        print(f"\nFIND: best scores. num clusters:  {num_clusters:3d}   ", end="")
        print(f"seqs in play: {sum(1 for s in sequences if s.active):9d}")
        vals = [_cluster_size(i, sequences) for i in range(num_clusters + 1)]
        print("      seqs per clust" + "".join(f"{v:7d}" for v in vals))

    itr = 0
    reassigned = 999
    while reassigned > 0:
        itr += 1
        vlp_score = _compute_scores(num_clusters, sequences, verbose=verbose)
        reassigned = 0

        # Reassign each active sequence to its highest-scoring (cluster, direction)
        for n in range(len(sequences)):
            if not sequences[n].active:
                continue
            kc = sequences[n].cluster
            kd = sequences[n].reverse
            best_val = -1e30
            kcbest = 1
            kdbest = False
            for kk in range(1, num_clusters + 1):
                for dd in range(2):
                    if vlp_score.score[dd, kk, n] > best_val:
                        best_val = vlp_score.score[dd, kk, n]
                        kcbest = kk
                        kdbest = dd == 1
            sequences[n].cluster = kcbest
            sequences[n].reverse = kdbest
            if kc != kcbest or kd != kdbest:
                reassigned += 1

        # Compute total score and count sequences with negative best scores
        xsum = 0.0
        num_negative = 0
        for n in range(len(sequences)):
            if not sequences[n].active:
                continue
            max_val = -1e30
            for kk in range(1, num_clusters + 1):
                for dd in range(2):
                    if vlp_score.score[dd, kk, n] > max_val:
                        max_val = vlp_score.score[dd, kk, n]
            xsum += max_val
            if max_val < 0:
                num_negative += 1

        if verbose:
            print("\nFIND: best scores. ")
            print("      ITER" + " " * 5 + "reassigned " + " " * 9 + "SCORE " + " " * 1 + "num neg   num clusters ")
            print(f"{itr:10d}{reassigned:15d}{xsum:15.2f}{num_negative:9d}{num_clusters:15d}")

    # Print final cluster distribution
    if verbose:
        vals = [_cluster_size(i, sequences) for i in range(num_clusters + 1)]
        print("\nFIND: best scores. ")
        print(" final seqs per clust" + "".join(f"{v:7d}" for v in vals))


def remove_next_cluster(binner: VLPBinner, num_clusters: int, thresh_next: int, verbose: bool) -> tuple[int, float]:
    """Find the weakest cluster and remove it by unassigning its members.

    A cluster's "removal score" is the average best score its members would get
    under a *different* cluster. The cluster with the highest such score is the
    easiest to absorb elsewhere, making it the best candidate for removal.

    If the best removal score is below thresh_next, no cluster is removed.
    Returns (cluster_id, removal_score).
    """

    if verbose:
        print("\nREMOVE: score sequences against every cluster")
    # Score every sequence against every cluster's VLP model
    vlp_score = _compute_scores(num_clusters, binner.sequences, verbose)

    # For each cluster, compute the average "best alternative score" —
    # i.e. how well its members would be explained by their next-best cluster.
    cluster_tot = np.zeros(num_clusters + 1, dtype=np.int32)
    cluster_score = np.zeros(num_clusters + 1, dtype=np.float64)

    for n in range(len(binner.sequences)):
        if not binner.sequences[n].active:
            continue
        k = binner.sequences[n].cluster
        cluster_tot[k] += 1
        # Find this sequence's best score under any cluster other than its own
        best_other = -np.inf
        for kk in range(1, num_clusters + 1):
            if kk == k:
                continue
            for dd in range(2):
                if vlp_score.score[dd, kk, n] > best_other:
                    best_other = vlp_score.score[dd, kk, n]
        cluster_score[k] += best_other

    # Normalize to get the average best-alternative score per cluster
    for k in range(1, num_clusters + 1):
        if cluster_tot[k] == 0:
            cluster_score[k] = -np.inf
        else:
            cluster_score[k] = cluster_score[k] / max(cluster_tot[k], 1)

    # The cluster whose members score best elsewhere is the removal candidate
    kbest = int(np.argmax(cluster_score[1 : num_clusters + 1]) + 1)
    xbest_score = float(cluster_score[kbest])

    if verbose:
        cnt_use = sum(1 for n in range(len(binner.sequences)) if binner.sequences[n].active)
        print(f"\nREMOVE: next cluster{cnt_use:9d}{kbest:6d}{xbest_score:9.2f}")
        _print_cluster_dist(num_clusters, binner.sequences)

        remove_vals = [xbest_score]
        for i in range(1, num_clusters + 1):
            remove_vals.append(cluster_score[i])
        print("remove clust " + "".join(f"{v:7.0f}" for v in remove_vals))

    # If the best removal score is below the threshold, keep all clusters
    if xbest_score < thresh_next:
        return kbest, xbest_score

    # Remove the cluster by unassigning all its members (set cluster to 0)
    for n in range(len(binner.sequences)):
        if binner.sequences[n].cluster == kbest:
            binner.sequences[n].cluster = 0

    # Print cluster distribution after removal
    if verbose:
        _print_cluster_dist(num_clusters, binner.sequences)

    return kbest, xbest_score


def _print_cluster_dist(num_clusters: int, sequences: list[Sequence]) -> None:
    vals = [_cluster_size(i, sequences) for i in range(num_clusters + 1)]
    print("rec per clust" + "".join(f"{v:7d}" for v in vals))
