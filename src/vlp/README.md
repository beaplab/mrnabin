# VLP

Variable Length Pattern clustering for mRNA contigs.

## Getting Started

`uv run scripts/sample_labeled_vlp_run.py`

To run on your own data:

`uv run scripts/run_vlp.py contig_filepath peptide_filepath output_filepath`

## Algorithm Description

VLP clusters mRNA contigs by building context-dependent nucleotide
models (tries) and iteratively reassigning sequences to the
best-scoring cluster.

### Phase 1: Initial clustering (4-mer profiles)

1. Build a pairwise similarity matrix using chi-squared tests on each
contig's 4-mer distributions, comparing each pair in both forward and
reverse-complement orientations.

2. Extract clusters by greedy transitive closure: start with the
most-connected sequence, grow the cluster by adding all transitively
similar sequences, and repeat until no cluster of size >= 2 can be
formed. Skip clusters smaller than `MIN_CLUSTER_SIZE`.

3. Within each cluster, iteratively determine strand orientation by
comparing each member's 4-mer profile against the cluster consensus.

### Phase 2: VLP model refinement

4. Gradually lower the minimum sequence length threshold (3000, 2000,
1000, 1), activating shorter sequences at each step.

5. At each threshold, run an EM-style convergence loop:
   - Build a VLP trie model for each cluster from its member
     sequences. The trie encodes context-dependent nucleotide
     probabilities using a sliding 128-bit window of up to 41 bases.
   - Score every active sequence against every cluster model in both
     orientations.
   - Reassign each sequence to its highest-scoring cluster and
     orientation.
   - Repeat until no reassignments occur.

6. After convergence, repeatedly identify and remove the weakest
cluster (the one whose members score best under a different cluster's
model) and re-run the EM loop, until no more clusters are worth
removing.

### VLP Trie

The VLP trie represents a variable-length context model over
nucleotide sequences. Each state in the trie corresponds to a context
of up to 41 preceding bases, encoded as a 128-bit integer. Training
data is sorted lexicographically by context; divergence points between
consecutive entries determine new states. Counts are smoothed via
hierarchical backing-off (deeper states blend toward shallower ones)
and converted to log-probabilities. Scoring walks the trie one base at
a time, accumulating log-likelihood.

## Parameters

* `min_seq_len = 3000` (minimum contig length for initial
  4-mer clustering)

* `min_cluster_size = 6` (clusters smaller than this are discarded)

* `xchi_thresh = 255.0` (chi-squared threshold for 4-mer similarity;
  pairs above this are considered dissimilar)

* `thresh_next = 0` (removal score threshold; clusters scoring below
  this are kept)

## History

The VLP algorithm is based on
[mRNAbinner/mar_23_vlp_cluster.F90](https://github.com/beaplab/mRNAbinner/blob/81ee1802444308ab13f727c1b30591d7a704d691/mar_23_vlp_cluster.F90).
This Python port uses NumPy arrays and Numba JIT compilation for the
performance-critical paths (128-bit key manipulation, sorting, trie
construction, and scoring). See [og_vlp/README.md](../../og_vlp/README.md)
for the original Fortran code and a direct Python translation.
