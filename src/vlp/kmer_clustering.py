"""Cluster formation from tetragraph chi-squared distances."""

import numba
import numpy as np
import numpy.typing as npt

from lib.encoded_sequences import Sequence, add_cluster_info, count_tetragraphs, reverse_complement_map


@numba.njit(cache=True)
def _ctcs(tab: npt.NDArray[np.float32], nr: int, ncc: int) -> float:
    """
    Chi-squared contingency table test.
    tab: (nr, ncc) float array
    Returns xchi statistic.
    """
    rsum = np.zeros(nr, dtype=np.float32)
    cluster_size = np.zeros(ncc, dtype=np.float32)
    for i in range(nr):
        for j in range(ncc):
            rsum[i] += tab[i, j]
            cluster_size[j] += tab[i, j]
    xtsum = np.float32(0.0)
    for j in range(ncc):
        xtsum += cluster_size[j]
    xchi = 0.0
    for i in range(nr):
        if rsum[i] < 0.5:
            continue
        for j in range(ncc):
            if cluster_size[j] < 0.5:
                continue
            xe = rsum[i] * cluster_size[j] / xtsum
            xchi += (tab[i, j] - xe) ** 2 / max(0.01, xe)
    return xchi


def _similarity_matrix(
    sequences: list[Sequence], rev_map: npt.NDArray[np.uint8], total: int, minlen: int, xchi_thresh: float
) -> npt.NDArray[np.int8]:
    sim = np.zeros((total, total), dtype=np.int8)  # 0/1 similarity matrix
    # Pairwise chi-squared
    tab = np.zeros((256, 2), dtype=np.float32)
    n1 = -1
    for n in range(len(sequences)):
        if len(sequences[n].seq) < minlen:
            continue
        n1 += 1
        sim[n1, n1] = 1
        tab[:, 0] = sequences[n].cnts[:]
        n2 = n1
        for nn in range(n + 1, len(sequences)):
            if len(sequences[nn].seq) < minlen:
                continue
            n2 += 1
            tab[:, 1] = sequences[nn].cnts[:]
            xchi0 = _ctcs(tab, 256, 2)
            tab[:, 1] = sequences[nn].cnts[rev_map[:]]
            xchi1 = _ctcs(tab, 256, 2)
            xchi = min(xchi0, xchi1)
            if xchi > xchi_thresh:
                continue
            sim[n1, n2] = 1
            sim[n2, n1] = 1
    return sim


def _set_directions(
    sequences: list[Sequence],
    cluster_members: list[int],
    cluster_num: int,
    rev_map: npt.NDArray[np.uint8],
    verbose: bool = True,
) -> int:
    """
    Determine whether each sequence in the cluster should be read forward or reverse-complemented.
    Iterate until stable:

    * Compare each member's tetragraph counts against a reference profile in both orientations
    * Pick the direction with the lower chi-squared
    * If any direction changed, rebuild the reference profile by summing all member counts and repeat
    """
    cluster_size = len(cluster_members)
    tab = np.zeros((256, 2), dtype=np.float32)
    s_idx = cluster_members[0]
    tab[:, 0] = sequences[s_idx].cnts[:]
    sequences[s_idx].reverse = False
    num_flipped = 0
    while True:
        # set sequence directions
        change = 0
        for nn in range(cluster_size):
            s_idx = cluster_members[nn]
            tab[:, 1] = sequences[s_idx].cnts[:]
            xchi0 = _ctcs(tab, 256, 2)
            tab[:, 1] = sequences[s_idx].cnts[rev_map[:]]
            xchi1 = _ctcs(tab, 256, 2)
            rev = xchi1 < xchi0
            if sequences[s_idx].reverse != rev:
                change += 1
            sequences[s_idx].reverse = rev

        if change == 0:
            break

        num_flipped = change

        tab[:, :] = 0
        for nn in range(cluster_size):
            s_idx = cluster_members[nn]
            if not sequences[s_idx].reverse:
                tab[:, 0] += sequences[s_idx].cnts[:]
            else:
                tab[:, 0] += sequences[s_idx].cnts[rev_map[:]]
    return num_flipped


def _initial_cluster(sim: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    # start cluster with seq with most similar seqs
    nmax = np.argmax(sim.sum(axis=1))
    cluster = sim[:, nmax].copy()
    if cluster.sum() < 2:
        return cluster
    sim[:, nmax] = 0

    # grow cluster by transitive closure
    while True:
        old_sum = int(cluster.sum())
        members = cluster == 1
        cluster = np.maximum(cluster, sim[:, members].max(axis=1))
        sim[:, members] = 0
        if old_sum == int(cluster.sum()):
            # no more growth
            return cluster


def make_4mer_clusters(
    sequences: list[Sequence], minlen: int, min_cluster_size: int, xchi_thresh: float, verbose: bool = True
) -> int:
    """
    * compare 4-mer distributions of each pair of long-enough sequences using chi-squared test.
    * find sequence with most similar neighbors. create clusters using greedy transitive closure.
    * repeat extracting clusters until no cluster of size >= 2 can be formed.
    * skip clusters smaller than min_cluster_size.
    * within each cluster, iteratively determine orientation
    """
    count_tetragraphs(sequences)

    rev_map = reverse_complement_map()

    total = len([s for s in sequences if len(s.seq) >= minlen])
    if verbose:
        print(f"4-mer clustering {total:9d} sequences of length at least {minlen:9d}")

    sim = _similarity_matrix(sequences, rev_map, total, minlen, xchi_thresh)
    cluster_num = 0
    while True:
        cluster = _initial_cluster(sim)
        cluster_size = int(cluster.sum())
        if cluster_size < 2:
            break
        if cluster_size < min_cluster_size:
            continue
        cluster_num += 1
        cluster_members = add_cluster_info(sequences, cluster, cluster_num, minlen)

        flipped = _set_directions(sequences, cluster_members, cluster_num, rev_map, verbose)

        if verbose:
            # dirs = "".join([str(int(sequences[cluster_members[nn]].reverse)) for nn in range(cluster_size)])
            dirs = ""
            print(f"    cluster num {cluster_num:4d} size: {cluster_size:7d} flipped: {flipped:5d} {dirs}")

    return cluster_num
