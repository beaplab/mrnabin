from lib.selection import select_by_length_percentile, selection_report

SEQUENCES = {f"s{i:02d}": "A" * (100 * (i + 1)) for i in range(100)}


def test_select_by_length_percentile_takes_the_band() -> None:
    selection = select_by_length_percentile(SEQUENCES, 85.0, 95.0)
    assert len(selection) == 10
    assert selection[0] == "s85"
    assert selection[-1] == "s94"


def test_select_by_length_percentile_orders_by_length() -> None:
    selection = select_by_length_percentile(SEQUENCES, 0.0, 100.0)
    lengths = [len(SEQUENCES[l]) for l in selection]
    assert lengths == sorted(lengths)


def test_select_by_length_percentile_can_be_empty() -> None:
    assert select_by_length_percentile(SEQUENCES, 50.0, 50.0) == []


def test_selection_report_handles_empty_selection() -> None:
    assert "Selected 0 contigs" in selection_report(SEQUENCES, [])


def test_selection_report_mentions_the_length_range() -> None:
    report = selection_report(SEQUENCES, select_by_length_percentile(SEQUENCES, 85.0, 95.0))
    assert "Selected 10 contigs ranging in length from 8600 to 9500" in report
