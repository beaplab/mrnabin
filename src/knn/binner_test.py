from typing import Any

import numpy as np
import pytest

import knn.binner
from knn.binner import Binner, _nearest_mean


def _mixture(n_per_org: int = 40, length: int = 4000, seed: int = 0) -> dict[str, str]:
    """Two organisms with different base composition, labelled by prefix."""
    rng = np.random.default_rng(seed)
    sequences = {}
    for org, probs in (("aa", [0.4, 0.1, 0.1, 0.4]), ("bb", [0.1, 0.4, 0.4, 0.1])):
        for i in range(n_per_org):
            draw = rng.choice(list("ACGT"), size=length, p=probs)
            sequences[f"{org}_{i:03d}"] = "".join(draw.tolist())
    return sequences


def _run(sequences: dict[str, str], **overrides: Any) -> tuple[list[set[str]], set[str]]:
    """Run a Binner sized for these small mixtures, checking every contig is accounted for once."""
    settings: dict[str, Any] = {
        "selection_percentiles_low": 0.0,
        "selection_percentiles_high": 100.0,
        "n_neighbors": 15,
        "min_bin_size": 3,
    }
    bins, unbinned = Binner(sequences, **(settings | overrides)).run()
    assigned = [l for b in bins for l in b]
    assert len(assigned) == len(set(assigned)), "a contig landed in two bins"
    assert not set(assigned) & unbinned, "a contig is both binned and unbinned"
    assert set(assigned) | unbinned == set(sequences)
    return bins, unbinned


def _mixed(bins: list[set[str]]) -> list[set[str]]:
    return [b for b in bins if len(set(l[:2] for l in b)) > 1]


@pytest.mark.parametrize("high", [100.0, 60.0])
def test_separates_two_compositions(high: float) -> None:
    """Pure bins whether the graph sees every contig or K.3 assigns 40% of them."""
    sequences = _mixture()
    bins, unbinned = _run(sequences, selection_percentiles_high=high)
    assert bins and not _mixed(bins)
    assert len(unbinned) < 0.5 * len(sequences)


@pytest.mark.parametrize(
    "overrides",
    [
        # narrow band: most contigs reach a bin only through K.3
        {"selection_percentiles_low": 85.0, "selection_percentiles_high": 95.0},
        # K.2 drops bins and K.3 re-assigns their contigs; once a contig was
        # counted both binned and unbinned
        {"selection_percentiles_high": 60.0, "min_bin_size": 6},
    ],
)
def test_every_contig_is_accounted_for_once(overrides: dict[str, Any]) -> None:
    _run(_mixture(n_per_org=30), **overrides)


def test_max_bins_keeps_the_largest_and_reassigns_the_rest() -> None:
    """Selection, not merging: discarded seeds' contigs still get binned."""
    sequences = _mixture(n_per_org=30)
    bins, unbinned = _run(sequences, selection_percentiles_high=60.0, max_bins=2)
    assert len(bins) <= 2
    assert len(unbinned) < 0.5 * len(sequences)


def test_neighbor_cap_prevents_a_complete_graph() -> None:
    """With k = n the graph goes complete and the organisms merge; the default cap prevents it."""
    sequences = _mixture(n_per_org=20)
    uncapped, _ = _run(sequences, n_neighbors=len(sequences), neighbor_cap=0)
    capped, _ = _run(sequences, n_neighbors=len(sequences))
    assert _mixed(uncapped)
    assert capped and not _mixed(capped)


@pytest.mark.parametrize(
    "param, value", [("n_neighbors", 0), ("resolution_min", 0.0), ("resolution_min", -1.0), ("resolution_min", np.inf)]
)
def test_rejects_bad_params(param: str, value: Any) -> None:
    # a resolution floor of 0 or below never stops the sweep
    with pytest.raises(ValueError, match=param):
        Binner(_mixture(n_per_org=5), **{param: value})


@pytest.mark.parametrize("n", [0, 1, 2])
def test_empty_length_band_leaves_everything_unbinned(n: int) -> None:
    sequences = dict(list(_mixture(n_per_org=1).items())[:n])
    assert _run(sequences, selection_percentiles_low=99.0, selection_percentiles_high=99.5) == ([], set(sequences))


def test_residual_graph_is_capped_without_a_main_graph() -> None:
    """Uncapped, 240 residual contigs get k = 239: a complete graph and one mixed bin."""
    binner = Binner(_mixture(n_per_org=120, length=1000))
    bins = binner._recluster_residual(list(binner.sequences), [])
    assert len(bins) > 1 and not _mixed(bins)


def test_unmeasurable_bins_borrow_the_typical_spread() -> None:
    sequences = _mixture(n_per_org=20)
    binner = Binner(sequences)
    labels = sorted(sequences)
    _, _, _, spread = binner._bin_references([set(labels[:2]), set(labels[2:20]), set(labels[20:])])
    # two members give two equal internal distances: no sigma of their own
    assert spread[0] == np.median(spread[1:])


def test_tied_scores_admit_only_stage_admit(monkeypatch: pytest.MonkeyPatch) -> None:
    sequences = _mixture(n_per_org=10)
    duplicate = sequences["bb_000"]
    sequences |= {f"dup_{i}": duplicate for i in range(8)}
    monkeypatch.setattr(knn.binner, "LENGTH_STAGES", (0.0,))
    binner = Binner(sequences)
    seed = {l for l in sequences if l.startswith("aa")}
    bins = binner._assign_staged(sorted(seed) + [f"dup_{i}" for i in range(8)], [seed])
    # 8 identical candidates: STAGE_ADMIT 0.75 admits 6, not all 8 tied at the cut
    assert sum(l.startswith("dup") for l in bins[0]) == 6


def test_nearest_mean_excludes_self() -> None:
    d = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 4.0], [2.0, 4.0, 0.0]])
    assert _nearest_mean(d, 2, exclude_self=True).tolist() == [1.5, 2.5, 3.0]
    # outsiders are capped at the same m as members: sample size - 1
    assert _nearest_mean(d[:1], 5).tolist() == [0.5]
