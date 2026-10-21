# KNN

Graph clustering of mRNA contigs: a mutual k-nearest-neighbour graph over the
4-mer distributions, partitioned with Leiden, then every remaining contig
assigned to the bin whose own internal scale it best fits.

## Getting Started

`uv run scripts/run_knn.py contig_file peptide_file output_file`

On a labelled EukProt pair:

`uv run scripts/run_12417881_knn.py EP00178 EP00657 --verbose`

Organisms may be given as an EukProt id, a filename fragment, or an index into
the sorted organism list.

In general the input is a `{label: sequence}` dict of oriented contigs, as
produced by `lib.fasta.normalize_sequences`, and `Binner.run()` returns
`(bins, unbinned)` — the same contract as DistBin and DBV.

## Algorithm

Distances throughout are total variation: half the L1 distance between two k-mer
frequency distributions, so they lie in [0, 1].

0. Select the contigs whose lengths fall between
`selection_percentiles_low` and `selection_percentiles_high`. Take
each one's `n_neighbors` nearest others by distance, then keep an edge
only where both endpoints chose each other. 

1. Run Leiden (CPM) on the unweighted graph at resolution 0.5, from
`seed`. Then lower the resolution by a factor of 0.7 per step down to
`resolution_min`, starting each step from the previous membership so
the partition coarsens continuously instead of being redrawn.

2. Discard bins below `min_bin_size`. Optionally, only keep the
`max_bins` largest bins. Discarded bins' contigs return to the
candidate pool rather than being dropped.

3. For each contig not in a bin, assign it to the bin whose internal
scale fits it best (see [Assignment](#assignment)), in length tranches
from longest to shortest. Each tranche admits its best-fitting 75% and
defers the rest to the next.

4. Any leftover contigs are run through the graph again and may form
their own bins (of at least 8).

## Parameters

**KNN**

* `k = 4` - k-mer size. 4-mers give a 256-dimensional distribution per contig.
* `selection_percentiles_low = 45.0`, `selection_percentiles_high = 95.0` — the
  length percentile band the graph is built over.
* `n_neighbors = 1024` — neighbours per contig. Raising it adds edges,
  so bins grow and coverage rises while purity falls. It also sets a
  hard *floor* on the bin count: Leiden cannot merge across connected
  components, and the component count falls as `n_neighbors` rises. If
  `resolution_min` stops reducing the bin count, this is why.  The default
  config is tuned for mixtures of tens of thousands of contigs. On
  much smaller inputs `n_neighbors: 1024` can approach the contig
  count.
* `neighbor_cap = 8` - cut `n_neighbors` to n / 8, n being the main graph's
  node count (floor 5); `0` disables. Useful for small n.

**Leiden**

* `resolution_min = 0.002` — where the resolution sweep stops; the coarseness
  knob. Under CPM, a node joins a community of size `n_c` only if it has more
  than `resolution * n_c` edges into it, so lower values give fewer, larger
  bins. (N.B., resolution and `n_neighbors` are effectively one knob. Accuracy
  depends only on the bin count they jointly produce, not on which of them
  produced it. Prefer the smallest `n_neighbors` that can reach it, since
  larger values cost O(n * n_neighbors) memory in the mutual filter.)
* `seed = 0` — passed to Leiden, and to the member sampling in step 3.
* `min_bin_size = 32` — bins smaller than this are dropped at step 2.
* `max_bins = 0` — keep only the N largest bins as seeds (0 = keep all).

**Output**

* `verbose`, `compact` — per-step logging, in DistBin's format.
* `labeled`, `prefix_len` — grade bins against organism labels taken as the
  first `prefix_len` characters of each contig id.

## What was tried

**The resolution sweep.** Re-running Leiden cold at each resolution
produced a "cliff" where purity collapsed in one step. Starting each
step from the previous membership removed it, but cross-species merges
still happen late at mixture-dependent resolutions, hence the fixed
floor. Two stopping rules failed:
* a jump in the largest community;
* a merge veto (removed)

**Graph parameters.**
* `n_neighbors` and `resolution` are one knob: accuracy depends only on the
  bin count they produce.
* The floor on bin count is the number of connected components.
* Large k with low resolution is the worst combination.
* The length band and min_bin_size are not organism-neutral: minority-organism bins are
smaller.


For each contig not in a bin, shortlist the 5 nearest bin centroids. Score
each shortlisted bin C as

```
z_C = (d(p, C) − μ_C) / σ_C
```

where `d(p, C)` is the mean distance from the contig to its 5 nearest of 200
sampled members of C, and μ_C and σ_C are the mean and standard deviation of
the same statistic for C's own members. The contig joins `argmin z` — the bin
whose internal scale it fits best. Dividing by σ_C is what makes a tight small
bin and a loose large one comparable. Nothing is refused on z; contigs left
over are those each length tranche deferred past the last one.
