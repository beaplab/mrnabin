# DBV

DistBin + Variable Length Pattern clustering for mRNA contigs.

## Getting Started

`uv run scripts/sample_labeled_dbv_run.py`

To run on your own data:

`uv run scripts/run_dbv.py contig_filepath peptide_filepath output_filepath`

## Algorithm Description

DBV bootstraps clusters with [DistBin](../distbin/README.md) (a kmer-distribution
heuristic on a length-percentile selection), then refines them with
[VLP](../vlp/README.md) trie models — gradually activating shorter contigs and
culling weak clusters.

### Phase 1: Initial clustering (DistBin)

1. Run DistBin on the input contigs. This produces an initial set of bins and a
set of unbinned sequences. See [distbin/README.md](../distbin/README.md) for
the heuristic.

2. Sequences in bins are activated and assigned a cluster id; unbinned
sequences are marked `cluster = -1` and inactive.

### Phase 2: VLP model refinement

3. Gradually lower the minimum sequence length threshold (median, 25th
percentile, 10th percentile, and 1), activating shorter sequences at each
step.

4. At each threshold:
   - Build a VLP trie model for each cluster from its active members.
   - Score every active sequence against every cluster's model.
   - Reassign each sequence to its highest-scoring cluster; if no model gives
     a positive score, set `cluster = -1`.

5. After the reassignment pass, repeatedly identify the weakest cluster and
remove it (set its members' `cluster = -1`, then reassign). Stop when no
cluster qualifies for removal.

### Weakest-cluster selection

For each active cluster, three diagnostics are computed:

* `mean_norm_scores[k]` — average per-base log-likelihood of cluster k's
  members under cluster k's own model. Low values mean the cluster's model is
  a poor fit even for its own members.

* `cv[k]` — coefficient of variation of per-base scores within the cluster.
  High CV suggests a bimodal distribution (mixed-composition cluster).

* `absorbability[k]` — ratio of the average best-other-cluster score to the
  average own-cluster score. High values mean members would fit nearly as
  well elsewhere.

Selection (in order):
1. If `min(mean_norm_scores) < 0.01`, remove the cluster with the worst model
   (`argmin(mean_norm_scores)`).
2. Otherwise, if the highest CV is more than `cv_outlier_ratio` times the
   second-highest, remove that outlier (`argmax(cv)`).
3. Otherwise, if any cluster's absorbability exceeds `absorb_thresh`, remove
   the most absorbable cluster (`argmax(absorbability)`).
4. Otherwise, stop.

## Parameters

DBV-specific:

* `absorb_thresh = 0.5` (absorbability threshold for the cluster-removal
  branch; lower values are more aggressive about merging clusters)

* `cv_outlier_ratio = 1.25` (minimum ratio of highest CV to second-highest
  before the CV branch fires)

* `skip_remove = False` (if True, only run reassignment passes — never
  remove clusters)

Inherited from [VLP](../vlp/README.md):

* `min_seq_len = 3000`, `min_cluster_size = 6`, `xchi_thresh = 255.0`

Inherited from [DistBin](../distbin/README.md):

* `k = 4`, `selection_percentiles = (85.0, 95.0)`, `creation_threshold = 0.1`,
  `selection_threshold = 0.175`,
  `cluster_thresholds = (0.02, 0.03125, 0.04, 0.05, 0.0625)`,
  `reduction_constant = 16.0`, `final_threshold = 0.175`, `prefix_len = 2`

## History

DBV combines DistBin's kmer-distribution clustering with the VLP refinement
loop from the [original Fortran
implementation](https://github.com/beaplab/mRNAbinner/blob/81ee1802444308ab13f727c1b30591d7a704d691/mar_23_vlp_cluster.F90).
DistBin replaces VLP's 4-mer chi-squared transitive-closure step for the
initial clustering; the VLP trie model then drives the iterative refinement
and weakest-cluster removal.
