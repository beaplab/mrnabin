import numpy as np

from lib.encoded_sequences import Sequence
from vlp.vlp_trie import VLPTrie


class ClusterScore:

    def __init__(self, length: int):
        self.scores = np.full(length, -np.inf)
        self.mean = 0.0
        self.var = 0.0


def _score_cluster(
    k: int,
    vlp_trie: VLPTrie,
    sequences: list[Sequence],
    verbose: bool = False,
) -> ClusterScore:
    """score every active sequence against cluster k's model"""
    cs = ClusterScore(len(sequences))
    total = 0.0
    sumsq = 0.0
    for i in range(len(sequences)):
        if not sequences[i].active:
            continue
        seq = sequences[i].seq
        score, var = vlp_trie.score(seq)
        cs.scores[i] = score

        # For sequences in this cluster, accumulate stats
        if sequences[i].cluster == k:
            total += score
            sumsq += var

    # Normalize mean and variance to per-base values
    num_bases = sum(len(s.seq) for s in sequences if s.cluster == k and s.active)
    if num_bases > 0:
        cs.mean = total / num_bases
        cs.var = sumsq / num_bases - cs.mean * cs.mean

    if verbose:
        print(
            f"    cluster:{k:2d} size:{_cluster_size(k, sequences):7d} "
            + vlp_trie.state_string()
            + f" num_bases:{num_bases:9d} mean:{cs.mean:9.5f} var:{cs.var:9.5f}"
        )
    return cs


def _cluster_size(cluster_num: int, sequences: list[Sequence]) -> int:
    return len([s for s in sequences if s.cluster == cluster_num and s.active])


def compute_scores(num_clusters: int, sequences: list[Sequence], verbose: bool) -> list[ClusterScore]:
    """Build a VLP model for each cluster and score every active sequence against it."""
    cluster_scores = []
    for k in range(num_clusters):
        # Ignore and dissolve clusters that are too small to build a meaningful model
        if _cluster_size(k, sequences) < 3:
            for n in range(len(sequences)):
                if sequences[n].cluster == k:
                    sequences[n].cluster = -1
            cluster_scores.append(ClusterScore(len(sequences)))  # add empty score
            continue

        vlp_trie = VLPTrie()
        vlp_trie.make_model(k, sequences, verbose=False)
        cluster_score = _score_cluster(k, vlp_trie, sequences, verbose=False)
        cluster_scores.append(cluster_score)
    return cluster_scores


def reassign(cluster_scores: list[ClusterScore], sequences: list[Sequence], verbose: bool = False) -> None:
    """
    for every active sequence, assign it to the cluster whose model gives it the best score
    * assigned: sequences that have yet to be assigned a cluster that are now in one
    * reassigned: sequences that change clusters
    * num_negative: sequences who's max score is negative (no cluster is good)
    """
    assigned = 0
    reassigned = 0
    stayed = 0
    num_negative = 0
    deassigned = 0
    for i, s in enumerate(sequences):
        if not s.active:
            continue
        cluster_id = s.cluster
        best_val = -np.inf
        best_cluster_id = cluster_id
        for j, cs in enumerate(cluster_scores):
            if cs.scores[i] > best_val:
                best_val = cs.scores[i]
                best_cluster_id = j
        if verbose:
            if best_val <= 0.0:
                num_negative += 1
                if cluster_id != -1:
                    deassigned += 1
            elif cluster_id == -1:
                assigned += 1
            elif cluster_id != best_cluster_id:
                reassigned += 1
            else:
                stayed += 1
        if best_val > 0.0:
            s.cluster = best_cluster_id
        else:
            s.cluster = -1

    if verbose:
        print(f"    newly assigned: {assigned} ; reassigned: {reassigned} ; stayed: {stayed} ;", end=" ")
        print(f"num negative: {num_negative} ; deassigned: {deassigned}", end=" ")
        print()


def weakest_cluster(
    cluster_scores: list[ClusterScore],
    sequences: list[Sequence],
    absorb_thresh: float,
    cv_outlier_ratio: float,
    verbose: bool,
) -> int:
    """Find the weakest cluster and remove it by unassigning its members.

    Uses coefficient of variation (std / mean) of per-sequence length-normalized
    scores within each cluster. Mixed clusters tend to have bimodal score
    distributions (majority members fit the model well, minority members fit
    poorly), giving a high CV. Pure clusters have tightly-clustered scores and
    low CV.
    """
    num_clusters = len(cluster_scores)
    cluster_totals = [0] * num_clusters
    alt_scores = [0.0] * num_clusters
    true_scores = [0.0] * num_clusters
    norm_scores: list[list[float]] = [[] for _ in range(num_clusters)]
    cv = [0.0] * num_clusters
    for i, s in enumerate(sequences):
        if not s.active or s.cluster == -1:
            continue
        cid = s.cluster
        cluster_totals[cid] += 1
        true_scores[cid] += cluster_scores[cid].scores[i]
        norm_scores[cid].append(float(cluster_scores[cid].scores[i]) / max(len(s.seq), 1))
        best_other = -np.inf
        for k in range(num_clusters):
            if k == cid:
                continue
            if cluster_scores[k].scores[i] > best_other:
                best_other = cluster_scores[k].scores[i]
        alt_scores[cid] += best_other

    for i in range(num_clusters):
        # true_scores, alt_scores, cv all start at 0.0 — skip leaves them at 0
        if cluster_totals[i] == 0:
            continue
        alt_scores[i] /= cluster_totals[i]
        true_scores[i] /= cluster_totals[i]
        if cluster_totals[i] < 2:
            continue
        arr = np.asarray(norm_scores[i])
        m = arr.mean()
        if m == 0:
            continue
        cv[i] = float(arr.std() / m)

    # Never propose removing the last surviving cluster. With only one live
    # cluster there is nothing to absorb its members into, so culling it
    # collapses the binning to zero bins. A monopolized seed set should
    # degrade to 1 bin, not 0. (Without this guard the mean_norm_scores
    # branch below picks the sole real cluster, since empty peers score inf.)
    num_live = sum(1 for t in cluster_totals if t > 0)
    if num_live <= 1:
        if verbose:
            print(f"Only {num_live} live cluster(s); skipping cull to avoid collapse to 0")
        return -1

    # TODO: consider switching to additive: absorbability = alt_per_base - true_per_base
    # (log-likelihood difference; no zero-floor needed)
    absorbability = [a / max(t, 1.0) for a, t in zip(alt_scores, true_scores)]
    mean_norm_scores = [float(np.mean(s)) if s else np.inf for s in norm_scores]
    weakest_id = -1
    if np.min(mean_norm_scores) < 0.01:
        # check to see if there is a cluster with a bad model
        weakest_id = int(np.argmin(mean_norm_scores))
    else:
        if _relative_max_pct(cv) > cv_outlier_ratio:
            weakest_id = int(np.argmax(cv))
        elif max(absorbability) > absorb_thresh:
            weakest_id = int(np.argmax(absorbability))

    if verbose:
        if weakest_id == -1:
            print("No weakest cluster found")
        else:
            print(
                f"Weakest cluster: {weakest_id} (size: {cluster_totals[weakest_id]}) "
                f"CV: {cv[weakest_id]:.3f} CV ratio: {_relative_max_pct(cv):.2f} "
                f"absorbability: {absorbability[weakest_id]:.2f}"
            )
        print("    true:     ", [f"{s:.2f}" for s in true_scores])
        print("    alt:      ", [f"{s:.2f}" for s in alt_scores])
        print("    absorb %: ", [f"{s:.2f}" for s in absorbability])
        print("    mean_norm:", [f"{c:.3f}" for c in mean_norm_scores])
        print("    cv:       ", [f"{c:.3f}" for c in cv])

    return weakest_id


def _relative_max_pct(nums: list[float]) -> float:
    sorted_nums = sorted(nums, reverse=True)
    if len(nums) <= 1 or sorted_nums[1] <= 0.0:
        return 0.0
    max_num = sorted_nums[0]
    second_num = sorted_nums[1]
    return max_num / second_num
