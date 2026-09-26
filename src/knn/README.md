# KNN

Graph clustering of mRNA contigs: a mutual k-nearest-neighbour graph over the
4-mer distributions, partitioned with Leiden, then every remaining contig
assigned to the bin whose own internal scale it best fits.

See [HISTORY.md](HISTORY.md) for the alternatives that were tried and the
measurements behind the default values.

## Getting Started

`uv run scripts/run_12417881_knn.py EP00178 EP00657 --verbose`

Organisms may be given as an EukProt id, a filename fragment, or an index into
the sorted organism list.

In general the input is a `{label: sequence}` dict of oriented contigs, as
produced by `lib.fasta.normalize_sequences`, and `Binner.run()` returns
`(bins, unbinned)` — the same contract as DistBin and DBV.

## Algorithm

Distances throughout are total variation: half the L1 distance between two k-mer
frequency distributions, so they lie in [0, 1].

**K.0 — build the graph.** Select the contigs whose lengths fall between
`selection_percentiles_low` and `selection_percentiles_high`. Take each one's
`n_neighbors` nearest others by distance, then keep an edge only where both
endpoints chose each other (`mutual`). `knn_indices_from_stack` computes the
neighbour lists a row-block at a time, so the n x n distance matrix is never
materialised.

**K.1 — partition.** Run Leiden on the unweighted graph with the `quality`
objective at `resolution`, for `n_iterations` from `seed`.

**K.2 — drop small bins.** Discard bins below `min_bin_size`; a mutual-kNN graph
strands isolated nodes that Leiden returns as singletons.

**K.2b — keep the largest bins** (optional, `max_bins`). Discarded bins' contigs
return to the candidate pool rather than being dropped.

**K.3 — assign the rest.** For each contig not in a bin, shortlist the
`assign_top_bins` nearest bin centroids. Score each shortlisted bin C as

```
z_C = (mean d(p, C) − μ_C) / σ_C
```

where `d(p, C)` is the mean distance from the contig to `assign_sample` sampled
members of C, and μ_C and σ_C are the mean and standard deviation of C's *own*
internal pairwise distances. The contig joins `argmin z` — the bin whose
internal scale it fits best. Dividing by σ_C is what makes a tight small bin and
a loose large one comparable.

If `z_max` is finite an admissibility gate also applies: the contig joins its
best bin only if `z ≤ z_max` and its distance is no greater than C's widest
internal distance, and is otherwise left unbinned. With `z_max` infinite (the
default) nothing is refused and every contig is binned.

**K.4 — recluster the residual** (optional, `recluster_residual`). Whatever K.3
declined is run through the graph again and may form its own bins.

Set `assign_remaining: false` to stop after K.2 and leave everything outside the
graph's bins unbinned. Everything still unassigned at the end is returned as the
noise set, which `lib.evaluate.dump_bins` writes as `bin_none`.

## Parameters

**Features and selection**

* `k = 4` — k-mer size. 4-mers give a 256-dimensional distribution per contig.
* `selection_percentiles_low = 45.0`, `selection_percentiles_high = 95.0` — the
  length percentile band the graph is built over. Widening the band admits more
  contigs and more short-contig noise; narrowing it biases against species whose
  contigs assemble shorter.

**Graph**

* `n_neighbors = 512` — neighbours per contig. Raising it adds edges, so bins
  grow and coverage rises while purity falls. `0` selects the O(log n) rule in
  `graph.auto_n_neighbors`. It also sets a hard *floor* on the bin count:
  Leiden cannot merge across connected components, and the component count
  falls as `n_neighbors` rises (1566 components at 512, 118 at 2048 on one
  mixture). If `resolution` stops reducing the bin count, this is why.
* `mutual = true` — require reciprocated neighbours. Setting it false
  symmetrises the directed lists instead, giving a much denser graph.

**Leiden**

* `quality = "cpm"` — `"cpm"` has no resolution limit; `"modularity"`
  (RBConfiguration) does, and merges small communities into large ones.
* `resolution = 0.5` — under CPM, a node joins a community of size `n_c` only if
  it has more than `resolution * n_c` edges into it, so the bar rises with
  community size. Raising it cuts more finely: more bins, higher purity, lower
  coverage. Values >= 1 produce no bins at all.

  **`resolution` and `n_neighbors` are effectively one knob.** Accuracy depends
  only on the bin count they jointly produce, not on which of them produced it:
  (2048, 0.005) gives 373 bins at 0.959 and (4096, 0.02) gives 365 at 0.958.
  Tune them together toward a bin count, and prefer the smallest `n_neighbors`
  that can reach it, since larger values cost O(n * n_neighbors) memory in the
  mutual filter. See HISTORY.md for how far each mixture can be coarsened
  before its bins start spanning species.
* `n_iterations = 2`, `seed = 0` — passed to Leiden.
* `min_bin_size = 3` — bins smaller than this are dropped at K.2.

**Assignment**

* `assign_remaining = true` — run K.3.
* `assign_top_bins = 5` — how many nearest centroids to score per contig.
* `assign_sample = 30` — members sampled per bin for `d(p, C)`, μ_C and σ_C.
* `z_max = .inf` — admissibility gate on a fixed z. Infinite means no gate.
  Prefer `target_coverage`: z is not comparable across datasets, so the same
  `z_max` refuses very different fractions on different data.
* `target_coverage = 1.0` — admit the best-fitting candidates until this
  fraction of all contigs is binned. The portable form of the gate; 1.0 disables
  it, and `z_max` then applies instead.
* `max_bins = 0` — keep only the N largest core bins as seeds (0 = keep all).
  Selection, not merging: discarded bins' contigs are re-assigned individually.
  An explicit reporting budget — nothing in the data suggests a value.

**Staged assignment** (all optional; empty/false reproduces a single pass)

* `length_stages = []` — length percentiles, longest tranche first, e.g.
  `[75, 50, 25, 0]`. Each stage assigns the contigs at least that long and
  recomputes every bin's envelope before the next, so bins are populated by
  reliable long contigs before short noisy ones are judged. When set, staging
  *replaces* the single pass rather than preceding it.
* `stage_admit = 0.75` — fraction of each tranche admitted by best z; the rest
  are deferred to a later stage where the enlarged bins may judge them better.
* `recluster_residual = false` — rebuild the graph on whatever is still
  unassigned and let it form new bins. The only way a species that never reached
  `min_bin_size` in the main graph can get a bin of its own.
* `residual_min_bin_size = 8` — floor for those new bins. Lower than
  `min_bin_size` on purpose: a species that failed the main threshold is exactly
  what this stage is for.

**Execution**

* `block_size = 512` — rows per pass in the blocked routines. Bounds peak memory
  without changing results.
* `stats_max_nodes = 6000` — the verbose distance summary needs a dense n x n
  matrix, and is skipped above this size.

**Output**

* `verbose`, `compact` — per-step logging, in DistBin's format.
* `labeled`, `prefix_len` — grade bins against organism labels taken as the
  first `prefix_len` characters of each contig id.

## Shared code

`lib.selection.select_by_length_percentile` and `lib.distances`
(`kmer_distributions`, `tv_distance_matrix`, `tv_cdist`), both lifted from
`distbin.binner.Binner`, which still has its own copies.
