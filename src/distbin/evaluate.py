from itertools import combinations

import numpy as np
import numpy.typing as npt
import scipy


def _label_org(label: str, prefix_len: int) -> str:
    return label[:prefix_len]


def distance_stats(values: npt.NDArray[np.float64]) -> str:
    return (
        f"mean:{np.mean(values):.3f} "
        f"min:{np.min(values):.3f} "
        f"p.25:{np.percentile(values, 0.25):.3f} "
        f"p.5:{np.percentile(values, 0.5):.3f} "
        f"p01:{np.percentile(values, 1):.3f} "
        f"p05:{np.percentile(values, 5):.3f} "
        f"p10:{np.percentile(values, 10):.3f} "
        f"p50:{np.percentile(values, 50):.3f} "
        f"p99:{np.percentile(values, 99):.3f} "
        f"p99.9:{np.percentile(values, 99.9):.3f} "
        f"max:{np.max(values):.3f} "
    )


def selected_label_stats(
    labels: list[str],
    distances: npt.NDArray[np.float64],
    kmer_distributions: dict[str, npt.NDArray[np.float64]],
    prefix_len: int,
) -> None:
    orgs = sorted(list(set([_label_org(l, prefix_len) for l in labels])))
    org_stacks = []
    org_dists = []
    for org in orgs:
        org_stack = np.stack([kmer_distributions[l] for l in labels if l.startswith(org)])
        org_dist = np.astype(scipy.spatial.distance.pdist(org_stack, "cityblock") / 2, np.float64)
        org_stacks.append(org_stack)
        org_dists.append(org_dist)
        print(f"org: {org} ; {org_stack.shape[0]} contigs")
        print(distance_stats(org_dist))

    if len(orgs) > 1:
        min_cross_dist = 1.0
        print("cross species kmer distance stats:")
        for i, j in combinations(range(len(orgs)), 2):
            print(f"org: {orgs[i]} vs. orgs: {orgs[j]}")
            cross_dists = np.astype(
                scipy.spatial.distance.cdist(org_stacks[i], org_stacks[j], "cityblock") / 2, np.float64
            )
            print(distance_stats(cross_dists))
            if np.min(cross_dists) < min_cross_dist:
                min_cross_dist = np.min(cross_dists)

        for i, org_dist in enumerate(org_dists):
            p = np.mean(org_dist < min_cross_dist) * 100
            print(f"org: {orgs[i]} ; min cross dist is in p{p:.2f} overall")
    print()
