# DistBin

A simple heuristic for clustering mRNA contigs.

## Getting Started

A small example that outputs the performance on a mix of two different species:

`uv run scripts/sample_labeled_distbin_run.py`

In general, the input should be a FASTA file of mRNA contigs along
with a peptide file used to orient the contigs.

`uv run scripts/run_distbin.py contig_filepath peptide_filepath output_filepath`

The script orients the contigs (via `lib.fasta.normalize_sequences`)
before constructing the `Binner`; `Binner` itself just takes the
already-oriented sequences as a `{label: sequence}` dict.

## Algorithm Description

The hypothesis is that the kmer-distribution for mRNA contigs coming
from the same species is sufficiently different from the
kmer-distribution for mRNA contigs coming from another species.
Furthermore, we hypothesize that longer contigs contain more useful
information, but that the longest contigs are more likely to contain
noise from sequencing artifacts.  Thus, the general strategy is to
select a set of longer contigs; bin them carefully so that each bin is
likely to be single-species, while trying to limit the number of bins.
Once a binning for the selection of longer contigs has been
established, assign the rest of the contigs to the closest bin.

More precisely:

0. For the selection, we pick the contigs whose length are between
`selection_percentiles[0]` and `selection_percentiles[1]`.

1. We create a large number of initial bins out of all the selected
contigs that are within `creation_threshold` of each other.

2. For those selected contigs that are not within `creation_threshold`
of any other contig, we assign them to the bin that contains the
closest contig as long as they are within `selection_threshold` of
each other.

3. We then try to reduce the number of bins by combining bins whose
  median centroids are within `cluster_thresholds[i]` of each
  other. We do several rounds of this, increasing the threshold each
  time. This step is the slowest and depends on the number of bins
  created by the first two steps.

4. From the reduced bins, we pick out the largest few.

5. The remaining unbinned selected contigs are added to the bins with
the closest median centroid as long as they are within
`selection_threshold`.

6. Finally, all the unbinned contigs are added to the bins with the
closest median centroid as long as they are within
`final_threshold`. Any contig that is further than this away from all
the bins remains unbinned.

## Parameters

[Warning] Most of these parameters were tuned on a dataset of fairly
similar species, and with $k=4$.

* `k = 4` (longer kmers don't seem to offer more information, while
  exponentially increasing the size of the distributions)

* `selection_percentiles = (85.0, 95.0)` (we choose 10% of the long
  contigs to focus on)

* `creation_threshold = 0.1` (we have seen contigs from different
  species get binned together if this constant is any higher)

* `selection_threshold = 0.175` (this threshold should be higher than
  the `creation_threshold`)

* `cluster_thresholds = [0.02, 0.03125, 0.04, 0.05, 0.0625]`

* `reduction_constant = 16.0` (controls how many top bins are kept
  after reduction: bins are accumulated until
  `sum_of_bin_sizes ** 2 >= reduction_constant * total_contigs`. Higher
  values keep more bins.)

* `final_threshold = 0.5` (The higher this treshold is, the fewer
  contigs remain unbinned at the cost of accuracy.)
