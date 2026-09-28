from lib.evaluate import (
    all_pure,
    bins_to_cover,
    budget_scores,
    dump_bins,
    grade_bins,
    sequence_stats,
    species_completeness,
)


def test_dump_bins() -> None:
    bins = [{"a", "b"}, {"c"}]
    unassigned = {"d"}
    rows = sorted(dump_bins(bins, unassigned))
    assert rows == [("a", "bin_0001"), ("b", "bin_0001"), ("c", "bin_0002"), ("d", "bin_none")]


def test_grade_bins_mixed_purity() -> None:
    bins = [{"AAfoo", "AAbar", "BBquux"}, {"BBx", "BBy"}]
    out = grade_bins(bins, unassigned={"CCorphan"})
    assert "Num bins: 2" in out
    assert "Correct: 4/5" in out
    assert "Unassigned: 1" in out


def test_grade_bins_empty_bin_does_not_crash() -> None:
    out = grade_bins([set(), {"AAx", "AAy"}], unassigned=set())
    assert "Correct: 2/2" in out


def test_all_pure() -> None:
    assert all_pure([{"AAx", "AAy"}, {"BBx"}], prefix_len=2) == "\U00002705"
    assert all_pure([{"AAx", "BBy"}], prefix_len=2) == "\U0000274c"


def test_sequence_stats() -> None:
    out = sequence_stats([1, 2, 3, 4, 5])
    assert "count:5" in out
    assert "min:1" in out
    assert "max:5" in out


def test_budget_scores_penalises_fragments() -> None:
    bins = [{"AA1", "AA2", "AA3", "BB1"}, {"BB2"}, {"BB3"}]
    scores = budget_scores(bins, unassigned={"AA4"}, budgets=(1, 3))
    assert scores[1] == (0.75, 3 / 7)
    assert scores[3] == (5 / 6, 5 / 7)


def test_species_completeness_uses_led_bins_only() -> None:
    bins = [{"AA1", "AA2", "BB1"}, {"BB2", "BB3"}, {"BB4"}]
    one = species_completeness(bins, unassigned={"AA3"}, top=1)
    assert one == {"AA": 2 / 3, "BB": 2 / 4}
    assert species_completeness(bins, set(), top=2)["BB"] == 3 / 4


def test_bins_to_cover() -> None:
    bins = [set(map(str, range(90))), set(map(str, range(5))), set(map(str, range(5)))]
    assert bins_to_cover(bins, 0.9) == 1
    assert bins_to_cover(bins, 0.95) == 2
    assert bins_to_cover([], 0.9) == 0
