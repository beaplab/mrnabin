"""Run the kNN/Leiden binner on a mixture of two specified EukProt organisms.

`run_12417881_distbin.py` sweeps every pair; this runs one pair so a parameter
can be tried without waiting for 7000 mixtures.

    uv run scripts/run_12417881_knn.py EP00178 EP00657 --verbose
    uv run scripts/run_12417881_knn.py 11 35 --verbose --resolution 0.02

Organisms may be given as an EukProt id, any unique part of a filename, or an
index into the sorted organism list (the numbering used by the sweep scripts).
"""

import argparse
import os

from knn.binner import Binner
from lib.args import get_args
from lib.evaluate import grade_bins
from lib.fasta import normalize_sequences

DEFAULT_CONFIG = "config/base_knn_config.yaml"
PREFIX_LEN = 7
transcriptomes_path = "data/12417881/assembled_transcriptomes"
protein_path = "data/12417881/proteins"


def resolve_org(orgs: list[str], spec: str) -> str:
    """Map an index, an EukProt id, or a filename fragment to a filename."""
    if spec.isdigit() and int(spec) < len(orgs):
        return orgs[int(spec)]
    matches = [o for o in orgs if spec in o]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"no organism matching {spec!r} in {transcriptomes_path}")
    raise SystemExit(f"{spec!r} matches several organisms: {', '.join(matches)}")


def load_mixture(org_files: list[str]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    for org_file in org_files:
        contig_file = os.path.join(transcriptomes_path, org_file)
        peptide_file = os.path.join(protein_path, org_file)
        sequences.update(
            {l: s for l, s in normalize_sequences(peptide_file, contig_file, org_file[:PREFIX_LEN], label_col=1)}
        )
    return sequences


def main(args: argparse.Namespace) -> None:
    orgs = sorted(os.listdir(transcriptomes_path))
    org_files = [resolve_org(orgs, args.org_a), resolve_org(orgs, args.org_b)]
    print("Mixing:", " + ".join(org_files), flush=True)

    args.prefix_len = PREFIX_LEN
    args.labeled = True
    sequences = load_mixture(org_files)
    binner = Binner(sequences, **vars(args))
    bins, unbinned = binner.run()
    print(grade_bins(bins, unbinned, prefix_len=PREFIX_LEN), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("org_a", type=str)
    parser.add_argument("org_b", type=str)
    args = get_args([DEFAULT_CONFIG], parser)
    main(args)
