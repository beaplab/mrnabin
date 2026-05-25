import numpy as np

from distbin.binner import Binner, _add_to_bins


def test_add_to_bins() -> None:
    bins = [set(["a", "b", "c"]), set(["d", "e"]), set(["f", "g", "h"])]
    labels = set(["c", "h", "j"])
    _add_to_bins(bins, labels)
    bins.sort(key=lambda x: len(x))
    assert bins[0] == {"d", "e"}
    assert bins[1] == {"a", "b", "c", "f", "g", "h", "j"}


def _make_binner(kdists: dict[str, np.ndarray], **overrides: object) -> Binner:
    """Build a Binner bypassing __init__ so tests can supply kmer distributions directly."""
    b = Binner.__new__(Binner)
    b.kmer_distributions = kdists
    b.verbose = False
    b.compact = False
    b.labeled = False
    b.cluster_thresholds = ()
    for k, v in overrides.items():
        setattr(b, k, v)
    return b


def test_combine_bins_merges_close_bins() -> None:
    kdists = {
        "a": np.array([1.0, 0.0, 0.0, 0.0]),
        "b": np.array([0.9, 0.1, 0.0, 0.0]),
        "c": np.array([0.0, 1.0, 0.0, 0.0]),
        "d": np.array([0.95, 0.05, 0.0, 0.0]),
    }
    binner = _make_binner(kdists, cluster_thresholds=(0.06, 0.5))
    bins = [{"a"}, {"b"}, {"c"}, {"d"}]
    # threshold 0.06: TV(a,d)=0.05 merges; TV(a,b)=0.1, TV(b,c)=0.9, TV(a,c)=1.0 don't
    # threshold 0.5: median({a,d})=[.975,.025,0,0]; TV to b = .075 merges; TV to c = .975 doesn't
    result = binner._combine_bins(bins)
    sorted_result = sorted([frozenset(b) for b in result], key=len, reverse=True)
    assert sorted_result == [frozenset({"a", "b", "d"}), frozenset({"c"})]


def test_combine_bins_no_merges() -> None:
    kdists = {
        "a": np.array([1.0, 0.0, 0.0, 0.0]),
        "b": np.array([0.0, 1.0, 0.0, 0.0]),
        "c": np.array([0.0, 0.0, 1.0, 0.0]),
    }
    binner = _make_binner(kdists, cluster_thresholds=(0.1,))
    bins = [{"a"}, {"b"}, {"c"}]
    result = binner._combine_bins(bins)
    assert sorted([frozenset(b) for b in result]) == sorted([frozenset({"a"}), frozenset({"b"}), frozenset({"c"})])


def test_add_to_closest_bin_median_assigns_and_unbins() -> None:
    kdists = {
        "a": np.array([1.0, 0.0, 0.0, 0.0]),
        "c": np.array([0.0, 1.0, 0.0, 0.0]),
        "b": np.array([0.9, 0.1, 0.0, 0.0]),
        "e": np.array([0.0, 0.95, 0.0, 0.05]),
        "f": np.array([0.25, 0.25, 0.25, 0.25]),
    }
    binner = _make_binner(kdists)
    bins = [{"a"}, {"c"}]
    new_bins, unbinned = binner._add_to_closest_bin_median(["a", "b", "c", "e", "f"], bins, threshold=0.5)
    # b → bin with a (TV=0.1), e → bin with c (TV=0.05), f equidistant 0.75 from both → unbinned
    assert unbinned == {"f"}
    by_member = {next(iter(b & {"a", "c"})): b for b in new_bins}
    assert by_member["a"] == {"a", "b"}
    assert by_member["c"] == {"c", "e"}


def test_add_to_closest_bin_median_no_candidates() -> None:
    kdists = {
        "a": np.array([1.0, 0.0, 0.0, 0.0]),
        "b": np.array([0.0, 1.0, 0.0, 0.0]),
    }
    binner = _make_binner(kdists)
    bins = [{"a"}, {"b"}]
    new_bins, unbinned = binner._add_to_closest_bin_median(["a", "b"], bins, threshold=0.5)
    assert unbinned == set()
    assert sorted([frozenset(b) for b in new_bins]) == sorted([frozenset({"a"}), frozenset({"b"})])
