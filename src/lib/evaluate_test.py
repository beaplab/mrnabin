from lib.evaluate import all_pure, dump_bins, grade_bins, sequence_stats


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
