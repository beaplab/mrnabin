"""Choosing which contigs an algorithm should look at first.

A k-mer distribution estimated from a short contig is mostly sampling noise:
the total variation distance between two contigs drawn from the *same* profile
falls off as 1/sqrt(length), so it is ~0.36 at 561 bp and ~0.12 at 5.5 kb.
Every algorithm here therefore starts from a band of longer contigs and only
assigns the rest once that band has been binned.

Extracted verbatim from `distbin.binner.Binner._select_labels` so that other
algorithms can select the same way without depending on DistBin.
"""

from collections.abc import Sized
from typing import Mapping

from lib.evaluate import sequence_stats


def select_by_length_percentile(sequences: Mapping[str, Sized], low_pct: float, high_pct: float) -> list[str]:
    """Labels of the contigs whose lengths fall between two percentiles.

    Percentiles are taken over the rank order of the contigs, not over the
    length values, so the size of the selection depends only on the width of
    the band.
    """
    low_idx = int(len(sequences) * low_pct / 100)
    high_idx = int(len(sequences) * high_pct / 100)
    selection = sorted([(len(s), l) for l, s in sequences.items()])[low_idx:high_idx]
    return [l for (_, l) in selection]


def selection_report(sequences: Mapping[str, Sized], selection: list[str]) -> str:
    """The length stats DistBin prints for a selection, as a string."""
    lengths = [len(sequences[l]) for l in selection]
    report = "Overall sequence length stats:\n"
    report += sequence_stats([len(s) for s in sequences.values()]) + "\n"
    if not selection:
        return report + "Selected 0 contigs"
    report += f"Selected {len(selection)} contigs ranging in length from {min(lengths)} to {max(lengths)}\n"
    return report + sequence_stats(lengths)
