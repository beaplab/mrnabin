import numpy as np

from knn.binner import Binner


def _mixture(n_per_org: int = 40, length: int = 4000, seed: int = 0) -> dict[str, str]:
    """Two organisms with different base composition, labelled by prefix."""
    rng = np.random.default_rng(seed)
    sequences = {}
    for org, probs in (("aa", [0.4, 0.1, 0.1, 0.4]), ("bb", [0.1, 0.4, 0.4, 0.1])):
        for i in range(n_per_org):
            draw = rng.choice(list("ACGT"), size=length, p=probs)
            sequences[f"{org}_{i:03d}"] = "".join(draw.tolist())
    return sequences


def test_run_separates_two_compositions() -> None:
    binner = Binner(
        _mixture(),
        selection_percentiles_low=0.0,
        selection_percentiles_high=100.0,
        n_neighbors=15,
        min_bin_size=3,
    )
    bins, unbinned = binner.run()
    assert bins
    for b in bins:
        prefixes = set(l[:2] for l in b)
        assert len(prefixes) == 1, f"bin mixed organisms: {prefixes}"
    assert len(unbinned) < 0.5 * len(binner.sequences)


def test_n_neighbors_near_n_merges_everything() -> None:
    """k saturates at n - 1, the graph goes complete, and the organisms merge.

    The default n_neighbors is sized for tens of thousands of contigs, so it
    degenerates on small inputs.  Pinned so the failure mode stays visible.
    """
    sequences = _mixture(n_per_org=20)
    bins, _ = Binner(
        sequences,
        selection_percentiles_low=0.0,
        selection_percentiles_high=100.0,
        n_neighbors=len(sequences),
        min_bin_size=3,
    ).run()
    assert any(len(set(l[:2] for l in b)) == 2 for b in bins)


def test_graph_only_leaves_everything_else_unbinned() -> None:
    sequences = _mixture()
    binner = Binner(sequences, selection_percentiles_low=85.0, selection_percentiles_high=95.0)
    bins, unbinned = binner.run()
    binned = set().union(*bins) if bins else set()
    assert binned | unbinned == set(sequences)
    assert not binned & unbinned


def test_every_sequence_is_binned_or_unbinned_exactly_once() -> None:
    sequences = _mixture(n_per_org=25)
    bins, unbinned = Binner(
        sequences, selection_percentiles_low=0.0, selection_percentiles_high=100.0, assign_remaining=True
    ).run()
    assigned = [l for b in bins for l in b]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) | unbinned == set(sequences)
    assert not set(assigned) & unbinned


def test_rejects_unknown_quality() -> None:
    try:
        Binner(_mixture(n_per_org=5), quality="louvain")
    except ValueError as e:
        assert "quality" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown quality")


def test_auto_n_neighbors_path_runs() -> None:
    sequences = _mixture(n_per_org=25)
    bins, _ = Binner(sequences, n_neighbors=0, selection_percentiles_low=0.0, selection_percentiles_high=100.0).run()
    assert bins


def _seeded(sequences: dict[str, str], **overrides: object) -> Binner:
    """Binner sized for the small synthetic mixtures in this file."""
    settings: dict[str, object] = {
        "selection_percentiles_low": 0.0,
        "selection_percentiles_high": 60.0,
        "n_neighbors": 15,
        "min_bin_size": 3,
    }
    settings.update(overrides)
    return Binner(sequences, **settings)  # type: ignore[arg-type]


def test_assign_remaining_increases_coverage() -> None:
    sequences = _mixture(n_per_org=30)
    _, without = _seeded(sequences, assign_remaining=False).run()
    _, with_ = _seeded(sequences, assign_remaining=True).run()
    assert len(with_) < len(without)


def test_tighter_z_max_assigns_fewer() -> None:
    sequences = _mixture(n_per_org=30)
    loose = len(_seeded(sequences, assign_remaining=True, z_max=10.0).run()[1])
    tight = len(_seeded(sequences, assign_remaining=True, z_max=-1.0).run()[1])
    assert tight > loose


def test_assignment_respects_organism_boundaries() -> None:
    sequences = _mixture(n_per_org=30)
    bins, _ = _seeded(sequences, assign_remaining=True).run()
    for b in bins:
        assert len(set(l[:2] for l in b)) == 1, "assignment mixed organisms"


def test_dropped_small_bins_are_not_double_counted() -> None:
    """K.2 drops bins, K.3 may re-assign their contigs; the two must agree.

    The earlier `unbinned |= small` put such a contig in a bin *and* in the
    unbinned set, inflating both totals past the number of input sequences.
    """
    sequences = _mixture(n_per_org=30)
    bins, unbinned = _seeded(sequences, assign_remaining=True, min_bin_size=6).run()
    assigned = [l for b in bins for l in b]
    assert len(assigned) == len(set(assigned)), "a contig landed in two bins"
    assert not set(assigned) & unbinned, "a contig is both binned and unbinned"
    assert len(assigned) + len(unbinned) == len(sequences)


def test_max_bins_keeps_the_largest_and_reassigns_the_rest() -> None:
    """Selection, not merging: discarded seeds' contigs still get binned."""
    sequences = _mixture(n_per_org=30)
    bins, unbinned = _seeded(sequences, assign_remaining=True, max_bins=2).run()
    assert len(bins) <= 2
    assigned = [l for b in bins for l in b]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) | unbinned == set(sequences)


def test_target_coverage_binds() -> None:
    sequences = _mixture(n_per_org=30)
    bins, _ = _seeded(sequences, assign_remaining=True, target_coverage=0.6).run()
    covered = sum(len(b) for b in bins) / len(sequences)
    assert 0.5 <= covered <= 0.7, covered


def test_target_coverage_takes_priority_over_z_max() -> None:
    """A coverage target is a quantile of z, so a fixed z_max must not also fire."""
    sequences = _mixture(n_per_org=30)
    bins, _ = _seeded(sequences, assign_remaining=True, target_coverage=0.8, z_max=0.0).run()
    covered = sum(len(b) for b in bins) / len(sequences)
    assert covered > 0.7, covered


def test_length_stages_assign_everything() -> None:
    sequences = _mixture(n_per_org=30)
    bins, unbinned = _seeded(sequences, assign_remaining=True, length_stages=(50.0, 0.0)).run()
    assigned = [l for b in bins for l in b]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) | unbinned == set(sequences)


def test_length_stages_do_not_lose_contigs() -> None:
    """Deferred contigs must reach a later stage, not vanish."""
    sequences = _mixture(n_per_org=40)
    staged, staged_unb = _seeded(sequences, assign_remaining=True, length_stages=(75.0, 50.0, 25.0, 0.0)).run()
    single, single_unb = _seeded(sequences, assign_remaining=True).run()
    assert sum(len(b) for b in staged) + len(staged_unb) == len(sequences)
    assert sum(len(b) for b in single) + len(single_unb) == len(sequences)


def test_recluster_residual_only_adds_bins() -> None:
    sequences = _mixture(n_per_org=40)
    without, _ = _seeded(sequences, assign_remaining=True, target_coverage=0.6).run()
    with_, _ = _seeded(sequences, assign_remaining=True, target_coverage=0.6, recluster_residual=True).run()
    assert len(with_) >= len(without)
    assigned = [l for b in with_ for l in b]
    assert len(assigned) == len(set(assigned)), "a contig landed in two bins"
