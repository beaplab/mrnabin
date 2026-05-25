from typing import Generator

SYMBIONT = "Haptophyta"
HOST = "Rhizaria"


def _read_comparison_file(file_path: str) -> Generator[tuple[str, str, float, int]]:
    """Process file comparing transcriptomes of host+symbiont and symbiont alone.

    Columns:
    * Query sequence (from host+symbiont)
    * Target sequence (from symbiont alone)
    * Percent identity
    * Length of match
    * others...
    """
    with open(file_path) as f:
        for line in f:
            fields = line.strip().split()
            yield (fields[0], fields[1], float(fields[2]), int(fields[3]))


def _symbiont_label_pcts(file_path: str, match_len_thresh: int = 100) -> dict[str, float]:
    """skip matches with length < match_len_thresh
    assign highest pct identity of all matches to query"""
    label_pcts: dict[str, float] = dict()
    for query, _, pct_id, match_len in _read_comparison_file(file_path):
        if match_len < match_len_thresh:
            continue
        if query not in label_pcts or pct_id > label_pcts[query]:
            label_pcts[query] = pct_id
    return label_pcts


def grade_bins_a14(
    file_path: str, match_path: str, bins: list[set[str]], unassigned: set[str], pct_thresh: float = 99.0
) -> tuple[str, int]:
    label_pcts = _symbiont_label_pcts(file_path)
    match_dict = _get_hap_rhi_matches(match_path)

    no_pct = 0
    bin_pct_sums = []
    bin_counts = []
    over_thresh_pcts = []
    for i, b in enumerate(bins):
        bin_pct = 0.0
        bin_count = 0
        over_thresh_pct = 0
        for label in b:
            if label in label_pcts:
                pct_id = label_pcts[label]
                bin_count += 1
                bin_pct += pct_id
                if pct_id > pct_thresh:
                    over_thresh_pct += 1
            else:
                no_pct += 1
        bin_counts.append(bin_count)
        bin_pct_sums.append(bin_pct)
        over_thresh_pcts.append(over_thresh_pct)

    total_unassigned = len(unassigned)
    total_total = sum(bin_counts) + no_pct
    unassigned_frac = total_unassigned / total_total if total_total else 0.0
    no_pct_frac = no_pct / total_total if total_total else 0.0
    output = f"Num bins: {len(bins)} "
    output += f"Unassigned: {total_unassigned} ({unassigned_frac:.2f}) "
    output += f"NoPctId: {no_pct} ({no_pct_frac:.2f})\n"
    for i in range(len(bins)):
        avg_pct_id = bin_pct_sums[i] / bin_counts[i] if bin_counts[i] else 0.0
        output += (
            f"size: {len(bins[i]):5d} ; comparable: {bin_counts[i]:4d} over_99_pct_id: {over_thresh_pcts[i]:4d} ; "
        )
        output += f"avg_pct_id: {avg_pct_id:.1f}% ; "
        host_matches, symb_matches, other_matches = _symb_host_match(match_dict, bins[i])
        total_matches = host_matches + symb_matches + other_matches
        if total_matches:
            output += f"host_matches: {host_matches:4d} ({100*host_matches/total_matches:4.1f}%) "
            output += f"symb_matches: {symb_matches:4d} ({100*symb_matches/total_matches:4.1f}%) "
            output += f"other_matches: {other_matches:4d} ({100*other_matches/total_matches:4.1f}%) "
        else:
            output += "host_matches:    0 ( -- %) symb_matches:    0 ( -- %) other_matches:    0 ( -- %) "
        output += "\n"

    best_bin = over_thresh_pcts.index(max(over_thresh_pcts))
    return output, best_bin


def _get_homologies(file_path: str) -> dict[str, tuple[str, float]]:
    """Read BLAST output 8 format: query, subject, %id, aln_len, mismatches,
    gap_openings, q_start, q_end, s_start, s_end, e_value, bit_score."""
    homologues: dict[str, tuple[str, float]] = dict()
    with open(file_path) as f:
        for line in f:
            fields = line.strip().split("\t")
            query, hit, bitscore = fields[0], fields[1], float(fields[11])
            query = query.split(".")[0]
            if query not in homologues or homologues[query][1] < bitscore:
                homologues[query] = (hit, bitscore)
    return homologues


def symbiont_bin_details(
    homologue_path: str, symbiont_path: str, binn: set[str]
) -> list[tuple[str, float | None, int | None, str | None, float | None]]:
    homologues = _get_homologies(homologue_path)
    symbiont_matches = {
        query: (pct_id, match_len) for query, _, pct_id, match_len in _read_comparison_file(symbiont_path)
    }
    output = []
    for label in binn:
        pct_id, match_len, hit, bitscore = None, None, None, None
        if label in symbiont_matches:
            pct_id, match_len = symbiont_matches[label]
        if label in homologues:
            hit, bitscore = homologues[label]
        output.append((label, pct_id, match_len, hit, bitscore))
    output.sort(key=lambda x: (x[1] is not None, x[1], x[2] is not None, x[2], x[4] is not None, x[4]), reverse=True)
    return output


def _get_hap_rhi_matches(match_path: str) -> dict[str, dict[str, int]]:
    match_dict = {}
    with open(match_path) as f:
        header = next(f).strip().split()
        for line in f:
            fields = line.strip().split()
            match_dict[fields[0]] = dict(zip(header[1:], [int(f) for f in fields[1:]]))
    return match_dict


def _symb_host_match(match_dict: dict[str, dict[str, int]], binn: set[str]) -> tuple[int, int, int]:
    """In practice anything that is zero for Haptophyta and non-zero
    for Rhizaria is probably a host contig, and anything that is
    non-zero for Haptophyta and zero for Rhizaria is probably a
    symbiont contig.
    """
    host_contigs = 0
    symb_contigs = 0
    other_contigs = 0
    for label in binn:
        if label in match_dict:
            if match_dict[label][SYMBIONT] == 0 and match_dict[label][HOST] > 0:
                host_contigs += 1
            elif match_dict[label][SYMBIONT] > 0 and match_dict[label][HOST] == 0:
                symb_contigs += 1
            else:
                other_contigs += 1
    return host_contigs, symb_contigs, other_contigs
