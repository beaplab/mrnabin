"""
uv run scripts/prepare_data.py -o data/5686984/m_choanoflagellate_transcriptomes.mixes -p all_k1_mix -k 1
data/5686984/m_choanoflagellate_transcriptomes.mrnabin/*.mrnabin
"""

# std
import argparse
import os
import random

# 3p
import numpy as np

from distbin.kmers import generate_kmer_map, seq_to_indices

# proj
from lib.fasta import read_mrna_file


def main(args: argparse.Namespace) -> None:
    sequences = {}
    for path in args.mrna_files:
        sequences.update(read_mrna_file(path))
    if args.verbose:
        print(f"{len(sequences)} sequences.")
    seqs = list(sequences.values())

    # create the train and validate splits
    random.shuffle(seqs)
    n = len(seqs)
    split_point = int(n * args.s)
    train_data = seqs[:split_point]
    val_data = seqs[split_point:]

    # encode and write to file
    k = args.k
    kmap = generate_kmer_map(k)
    for data, split_type in ((train_data, "train"), (val_data, "val")):
        ids = [seq_to_indices(s, k, kmap) for s in data]
        lens = np.array([len(i) for i in ids], dtype=np.uint16)
        flat = np.concatenate([np.array(i, dtype=np.uint16) for i in ids])
        flat.tofile(os.path.join(args.outdir, f"{args.prefix}_{split_type}_data.bin"))
        lens.tofile(os.path.join(args.outdir, f"{args.prefix}_{split_type}_lengths.bin"))
        print(f"{split_type} has {np.sum(lens):,} tokens")
        print(f"{split_type} has {len(lens):,} sequences")
        print(f"First 20 sequence lengths: {lens[:20]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mrna_files", nargs="+", type=str, help="list of mrna file paths")
    parser.add_argument("-k", type=int, default=4, help="kmer length")
    parser.add_argument("-s", type=float, default=0.5, help="train/val split")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("-o", "--outdir", type=str, help="output dir")
    parser.add_argument("-p", "--prefix", type=str, help="output file prefix")
    args = parser.parse_args()
    main(args)
