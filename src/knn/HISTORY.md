# KNN — what was tried, and why the defaults are what they are

Everything here was measured on pairwise mixtures from `data/12417881` (EukProt)
and `data/5686984` (choanoflagellates). DistBin and DBV numbers are from the
stored sweeps in `runs/`, not re-run.

## The original problem

DistBin decides whether two contigs belong together by comparing their total
variation distance to a constant. No constant is right for every species: in one
EukProt pair the within-species distances averaged 0.201 for one organism and
0.286 for the other, while *cross*-species distances started at 0.125. A
threshold loose enough to hold the second organism together also merges it with
the first.

Distance *ranks* adapt to the local scale in a way distance values do not, which
was the motivation for a kNN graph.

**How that turned out.** The graph did not remove the threshold — it relocated
it. CPM's `resolution` is a threshold on internal edge density, and it matters
enormously. What the graph actually bought was structural: Leiden replaces
DistBin's greedy, irreversible bin-merging with a global optimisation. DistBin
scores 0.732 on EP00002+EP00271 while emitting only 3 bins; its *seeds* were
already ~0.997 pure and the merge step collapsed them. That is where most of the
improvement on DistBin's weak pairs comes from, not from ranks beating distances.

## Current numbers, after the DBV orientation fix

DBV's `_score_cluster` scored the raw contig while its tries were trained on
the oriented one (fixed on main in ea0718c). `get_sequences` leaves
reverse-strand contigs as read -- about half of every EukProt transcriptome --
so every DBV and KNN -> VLP number measured before the fix scored half the
contigs on the wrong strand. **All DBV and KNN -> VLP figures in the sections
below this one are pre-fix.** KNN alone was unaffected: its loader orients.

43 mixtures (seeded random 30 pairs + 6 triples, plus the 7 hard pairs):

| config | bins | n90 | cov | r@4 | r@16 | p@4 | worst4 | sec |
|---|---|---|---|---|---|---|---|---|
| DBV | 7 | 5 | 0.876 | 0.614 | 0.763 | 0.862 | 0.498 | 157 |
| KNN -> VLP, DBV-style merge, 16 seeds | 15 | 10 | 0.970 | 0.608 | **0.902** | **0.941** | 0.672 | 553 |
| **KNN -> VLP, held-out merge (default)** | 9 | **3** | 0.932 | **0.822** | 0.868 | 0.930 | **0.769** | 1258 |

(bins, n90 and sec are medians; the rest means. The held-out run used 12
workers against 6 for the others, which inflates its time.)

The fix alone moved DBV's r@16 from 0.659 to 0.763 (better on 39 of 43) and its
coverage from 0.755 to 0.876, and the DBV-style KNN -> VLP from 0.773 to 0.902
(better on all 43). The stored DBV EukProt sweep and figure on main predate it.

The held-out merge beats DBV at r@4 on 40 of 43 mixtures (worst loss -0.14).
Against the DBV-style merge it trades r@16 (0.868 vs 0.902: it merges harder
and leaves 7% unbinned against 3%) for r@4 (0.822 vs 0.608).

**How the held-out merge was chosen.** `scripts/merge_path_12417881.py
--holdout` records full merge paths with out-of-fold scoring;
`scripts/merge_path_analysis.py` compares stopping rules, tuned on one half of
the mixtures and scored on the other. On the corrected paths:

| rule | bins | r@4 | prec |
|---|---|---|---|
| best r@4, with labels | 8-9 | 0.85 | 0.92-0.95 |
| guard on fit over all contigs, 0.002 | 8-11 | 0.81-0.84 | 0.92-0.94 |
| guard on fit over assigned contigs, 0.002 | 6-7 | 0.78-0.83 | 0.92-0.93 |
| BIC | 3-6 | 0.77 | 0.90-0.92 |
| no merging | 32 | 0.44 | 0.94-0.97 |

(ranges are the two halves.) With correct scoring the tolerant guard beats BIC
and needs no fitted scale; its good range was 0.002-0.003 on both halves.
Averaging fit over assigned contigs only has a failure mode a synthetic test
caught: dissolving a species' last cluster leaves its contigs unassigned, which
drops them from the average rather than lowering it.

Held-out likelihood alone does not pick a bin count -- it is highest with no
merging at all, because separate models for a species' sub-populations really
do predict held-out contigs better. What held-out scoring buys is a better
merge path: the best achievable r@4 rose from 0.70 to 0.745 before the fix,
likely because in-sample a contig always looks good in the cluster whose model
it trained.

## Scored with a bin budget: the fragmentation problem

Everything below this section was measured with `grade_bins`, which credits
each bin's plurality organism, so thousands of tiny bins score as well as a
handful of large ones. `lib.evaluate.budget_scores` scores only the N largest
bins and counts everything else as unbinned: `r@N` is correct contigs in those
bins over *all* contigs, which fragmentation cannot raise.

Measured with `scripts/benchmark_12417881.py` on a seeded random sample (30
EukProt pairs, 6 triples) plus the 7 hard pairs used below -- 43 mixtures.

| config | bins (median) | r@4 | r@16 | r@64 | p@16 (median) | sec (median) |
|---|---|---|---|---|---|---|
| DBV (current config) | 5 | **0.558** | **0.659** | 0.659 | 0.907 | 164 |
| KNN, old defaults | 3228 | 0.035 | 0.084 | 0.196 | 0.988 | 176 |
| KNN, sweep to 0.002 (new defaults) | 160 | 0.411 | 0.630 | **0.822** | **0.957** | 471 |
| same, sweep with veto to 1e-4 | 177 | 0.265 | 0.502 | 0.783 | 0.955 | 501 |
| same, mean-fit assignment | 160 | 0.417 | 0.633 | 0.815 | 0.946 | 440 |

(means, except where marked median)

* **The old defaults' accuracy was almost all fragmentation.** Their 16
  largest bins hold 8% of a mixture correctly.
* **DBV emits 3-18 bins under its current config**, not the 384-1435 quoted
  below, which came from an older sweep. It is far ahead at 4-16 bins, and it
  sometimes collapses (2 bins at 0.44 precision on a triple).
* **The warm-started sweep closes most of the gap**: r@16 0.084 -> 0.630 at
  higher precision than DBV, and ahead of it from roughly 64 bins up.
* **Nearest-member assignment (`assign_nearest`) is neutral** at this
  coarseness.

### KNN seeds merged by VLP

`knn.vlp_merge.KNNVLPBinner`: KNN's largest `vlp_seed_bins` bins seed DBV's VLP
phase in place of DistBin's. Same 43 mixtures:

| config | bins (mean) | r@4 | r@16 | p@16 | worst4 | sec (median) |
|---|---|---|---|---|---|---|
| DBV | 7 | 0.558 | 0.659 | 0.864 | 0.429 | 164 |
| KNN, sweep to 0.002 | 206 | 0.411 | 0.630 | 0.916 | 0.445 | 471 |
| KNN -> VLP, 32 seeds | 23 | 0.458 | 0.777 | 0.930 | 0.533 | 981 |
| KNN -> VLP, 16 seeds | 11 | **0.562** | **0.773** | 0.923 | **0.627** | 583 |

Against DBV per mixture, 16 seeds: r@16 better on 37, worse on 2 (worst
-0.06); r@4 better on 19, worse on 23; worst4 better on 32. The largest gains
are on mixtures where DBV collapsed (EP00178+EP00657: r@16 0.05 -> 0.71).

This reverses the earlier "VLP merging loses" result below. That merged ~1600
fine bins before assignment, where one wrong merge is catastrophic; here VLP
starts from 16 bins that already hold most of a mixture, and its likelihood guard
stops at the first cull that lowers the fit. With 32 seeds VLP stopped at ~23
bins, which is why fewer seeds did better at small budgets.

### Choosing the number of bins

`scripts/merge_path_12417881.py` records, per mixture, a full VLP merge path:
KNN's 32 largest bins seed VLP, DBV's length schedule reassigns without culling,
then the most absorbable cluster is dissolved and its contigs reassigned, one
step at a time down to a single cluster. `scripts/merge_path_analysis.py` then
compares label-free stopping rules along those paths, tuning each rule's
parameter on half the mixtures and scoring it on the other half.

**Two regimes.** On clean mixtures (starting precision >= 0.97, about half),
precision stays flat down to roughly one bin per species and then collapses in
one step, and the largest per-base log-likelihood drop sits exactly on that step
(e.g. EP00163+EP01103, EP00610+EP01127, EP01111+EP01128 at 3 bins). On messier
mixtures precision erodes gradually from the first dissolution; there is no
clean species boundary to find.

**DBV's likelihood guard is too sensitive.** Per-base log-likelihood wanders by
~0.001 between harmless steps, so a zero-tolerance guard stops at the first
wobble -- which is why KNN -> VLP halted at 11-23 bins.

Scored on r@4 (a cross-species merge lowers it, extra bins cannot raise it),
held-out half, damage threshold 0.03:

| rule | bins (median) | r@4 | prec |
|---|---|---|---|
| best r@4, with labels | 9 | 0.696 | 0.928 |
| BIC, lambda = 3000 | 9 | 0.628 | 0.922 |
| guard, tolerance 0.002 | 9 | 0.606 | 0.913 |
| stop before biggest LL drop | 4 | 0.622 | 0.876 |
| absorbability < 0.3 | 9 | 0.603 | 0.859 |
| MDL on trie states | 3 | 0.493 | 0.797 |
| no merging (32 bins) | 32 | 0.384 | 0.944 |

BIC picks the step minimising `-total_score + lambda * n_bins * log(assigned
bases)`. Its tuning curve has a plateau at lambda 2000-4000; with the halves
swapped it scores 0.633 / 0.898, where guard and absorbability reach similar
r@4 (0.64-0.65) at precision ~0.86. It recovers ~90% of the labelled optimum.
Charging for trie states instead (MDL) does worse: state counts barely change
as clusters merge, because a merged trie is about as large as the two it
replaces.

Not yet built into `KNNVLPBinner`; the victim order (most absorbable first) was
fixed throughout, and only the stopping point was chosen.

### The both-signals veto

Requiring both signals (`veto_both`) fragments less than requiring either, but
sweeping to 1e-4 then lets cross-species merges through and precision falls
below the fixed floor's. Neither threshold pair beat the floor:

| config | r@4 | r@16 | r@64 | p@16 |
|---|---|---|---|---|
| floor 0.002, no veto | **0.411** | **0.630** | **0.822** | **0.916** |
| either signal, to 1e-4 | 0.265 | 0.502 | 0.783 | 0.923 |
| both, rel 0.08 / dist 1.15 | 0.318 | 0.532 | 0.788 | 0.904 |
| both, rel 0.05 / dist 1.2 | 0.382 | 0.582 | 0.801 | 0.893 |

### The warm-started sweep, and why the cliff was an artifact

Each step starts Leiden from the previous step's membership. On the clean pair
EP00002+EP00271 at k=1024 the two largest communities converged on the two
organisms: 2 bins >= 32 at 0.991 purity covering 93% of the graph by
resolution 1.4e-4, with no cross-species merge on the way. The collapse
described under "Every mixture has a cliff" came from re-running Leiden cold at
each resolution.

Cross-species merges do still happen warm, but late and at mixture-dependent
resolutions: 0.007 on a triple, 0.0017 on EP00178+EP00657, 0.0008 on
EP00002+EP00178, never on EP00002+EP00271. Hence a fixed floor (0.002) for now.

**The "largest community jumps" stopping rule does not work.** Within-species
steps grew the largest community by up to 1.5x (early) and 1.24x (late, purity
unchanged); cross-species ones by 1.26-1.33x.

**Merge statistics that do separate them**, over 285 merges of two >= 90%-pure
parts of >= 100 nodes on six mixtures (11 cross-species):

| statistic | cross (median) | within (median) | AUC |
|---|---|---|---|
| rel = inter-part edge density / sqrt(internal densities) | 0.031 | 0.162 | 0.87 |
| conductance of the cut | 0.046 | 0.338 | 0.86 |
| 4-mer centroid distance / within-part distance | 1.24 | 1.09 | 0.88 |

Vetoing on `rel < 0.08 or distance ratio > 1.15` catches 11/11 cross merges but
also 31% of within-species ones -- and a vetoed pair is re-proposed and
re-vetoed at every later step, so species stay split. End to end it lost
(r@16 0.502 vs 0.630). The thresholds were fitted on 11 events, three of the six
mixtures are in the benchmark sample, and a stricter rule (fewer false vetoes)
was not tried.

## Results

All 11 tested mixtures, current defaults.

| pair | bins | cov | acc | ratio | sec | DistBin | DBV |
|---|---|---|---|---|---|---|---|
| EP00463+EP00495 | 837 | 100% | 0.832 | 1.00 | 11 | 0.734 | 0.729 |
| EP00244+EP00245 | 1649 | 100% | 0.765 | 1.00 | 33 | 0.613 | 0.672 |
| EP00245+EP00246 | 1639 | 100% | 0.771 | 1.00 | 39 | 0.666 | 0.721 |
| EP00178+EP00657 | 1525 | 100% | 0.930 | 1.00 | 50 | - | - |
| EP00252+EP00262 | 2682 | 100% | 0.785 | 1.00 | 70 | 0.659 | 0.745 |
| EP00002+EP00271 | 3463 | 100% | 0.977 | 1.00 | 113 | 0.732 | 0.959 |
| EP00002+EP00178 | 3769 | 100% | 0.921 | 1.00 | 127 | 0.809 | 0.930 |
| cho1+8 | 2422 | 100% | 0.765 | 1.00 | 107 | - | 0.647 |
| cho1+17 | 3675 | 100% | 0.918 | 1.00 | 180 | - | 0.771 |
| cho1+7 | 3087 | 100% | 0.917 | 1.00 | 127 | - | 0.795 |
| cho1+14 | 2153 | 100% | 0.872 | 1.00 | 95 | - | 0.867 |

Beats DBV on 8 of the 9 pairs with a baseline, ties cho1+14 (+0.005), loses
EP00002+EP00178 by 0.009 — there at 100% coverage against DBV's 90%.

`ratio` is minority-organism recall over majority-organism recall. It is 1.00
everywhere.

**Three caveats.** Bin counts are 837-3769 where DBV emits 384-1435, and
`grade_bins` credits each bin's plurality organism, so more bins flatter the
score by an amount that has not been quantified. The baselines were not re-run.

And most importantly, **this set is not representative**. Across the 2608 pairs
in the stored sweep, DistBin's median is 0.882 and DBV's is 0.982; every pair in
the table above sits below both, most of them far below. The set was assembled
by picking pairs DistBin or DBV scored badly on, so "beats DBV on 8 of 9" is a
statement about DBV's weak tail, not about typical mixtures. The one pair drawn
from the bulk of the distribution (median EP00161+EP01094, DBV 0.980) scores
0.984 -- roughly parity. A fair comparison needs a random sample of pairs.

## The length band

A length percentile is not a species-neutral filter, because contig lengths are
not comparable across species. On EP00178+EP00657:

| band | n | EP00178 | EP00657 | minority |
|---|---|---|---|---|
| p85-p95 (DistBin's) | 4,140 | 4,045 | 95 | 2.3% |
| p75-p100 | 10,352 | 8,807 | 1,545 | 14.9% |
| p45-p95 (default) | 20,703 | 13,231 | 7,472 | 36.1% |
| whole mixture | 41,406 | 28,382 | 13,024 | 31.5% |

EP00657 has 13,024 contigs and 95 of them clear p85 — 0.7% of the organism. No
clustering can find a species the selection has already discarded, and this was
the failure that motivated widening the band.

p45-p95 beat p70-p95 on every pair tested, on coverage, purity and skew
simultaneously. The top 5% is still dropped, following DistBin's reasoning that
the longest contigs carry more assembly artefacts.

**Still unsolved.** The band is not neutral in general — on EP00002+EP00178 a
59/41 mixture yields an 83/17 selection, and the narrower band is marginally
*better* there, so which direction helps is pair-dependent. A stratified sample
across length deciles would be neutral by construction and is the obvious next
change to `lib/selection.py`.

## n_neighbors and resolution

These are largely redundant: **both trade coverage against purity in the same
direction.** Raising `n_neighbors` or lowering `resolution` each adds coverage
and costs purity, so they have to be considered together. Sweeping one while
holding the other at an extreme produces contradictory conclusions, which
happened repeatedly here.

Measured over k in {4 .. 1024} x resolution in {4, 1, 0.5, 0.25, 0.125, 0.0039}
on all 11 pairs:

* **Resolution >= 1 produces zero bins at every k.** Not just on sparse graphs.
* **Large k with low resolution is the worst combination available** — core
  purity 0.63 at k=1024, resolution 0.0039. The old defaults (800 / 0.005) sat
  in exactly this corner.
* On separable pairs, large k is free: top-800 neighbourhoods are 97%
  same-species, so more neighbours only help connectivity.
* On overlapping pairs, large k floods the neighbourhood — the median rank of
  the first cross-species neighbour is 5.

For a purity floor of 0.99 across the separable pairs, the best setting was
k=512 / resolution 0.25 (~31% of the mixture in core bins); for 0.98,
k=1024 / 0.125 (~40%).

**k=4 was a trap.** Ranked purely on worst-case core purity it wins (0.997
across all 11 pairs), but the maximum achievable cluster size at k=4 is **7** —
with four neighbours per node the mutual graph has no larger component. Those
are cliques, not bins, and they are unusable as seeds: σ_C from a 3-member bin
rests on 3 distances. Ranking settings by purity without a bin-size constraint
selects whatever fragments hardest.

## min_bin_size

Raising it is **not organism-neutral**. Minority-organism bins are
systematically smaller, so a high floor deletes them preferentially: recall
parity falls from 0.90 at floor 10 to 0.63 at floor 30 on the reference pair,
for no purity gain (purity is flat at ~0.97 across floors 1-500 — small bins are
trivially pure and inflate nothing).

An early attempt to couple `min_bin_size` to `n_neighbors` via the worst-case
Jaccard bound `2J(k+1)/(1+J)` was measured and abandoned: the bound is correct
as a worst case but not predictive, and bin count above a given floor is roughly
k-invariant at k=200-800.

## Removed: SNN Jaccard reweighting

The standard SNN step weighted each edge by the Jaccard overlap of the two
neighbourhoods and pruned below a threshold.

* **The threshold was inert** across {0.02, 0.05, 0.1, 0.2} at every k tried.
* **The weighting was worth nothing on separable pairs** — unit weights matched
  it to within 0.001 purity at equal coverage — and 0.02-0.04 purity on
  overlapping ones.
* It cost a blocked sparse product that dominated runtime at large k, and was
  the most intricate code in the module.

Worth restoring if overlapping pairs become the priority. Note unit weights
roughly double every internal edge sum, so CPM resolutions are ~2x what Jaccard
weights needed — the two are not interchangeable at a fixed resolution.

## The admissibility gate: corrected

K.3 can refuse a contig whose best bin still fits badly. The first version did
this with a fixed `z_max`, *and* silently required `d <= widest_C` whenever
`z_max` was finite -- so setting `z_max` switched on a second, much harsher
condition at the same time.

That coupling produced a wrong conclusion, recorded here for a while: measured
on four pairs the gate appeared to cost ~50 points of coverage for 0.02-0.09
accuracy and to skew refusals towards the minority organism, so it was disabled
by default. What was actually being measured was `d <= widest_C` firing while
`z_max` looked responsible.

**Gating on z alone helps on 13 of 13 pairs**, monotonically:

| z_max | mean accuracy | mean coverage | mean parity |
|---|---|---|---|
| inf | 0.836 | 100% | 1.00 |
| 6 | 0.860 | 84% | 0.92 |
| 4 | 0.876 | 71% | 0.87 |
| 2 | 0.904 | 43% | 0.77 |

**But z is not a portable unit.** Reaching 90% coverage needs z between 3.3 and
6.1 on EukProt pairs and between 11.8 and 16.9 on choanoflagellate ones -- the z
distributions differ by ~3x between datasets. A fixed z=6 leaves ~95% coverage
on the former and ~60% on the latter. It is not even portable within one
dataset: 90% coverage needed z=3.31, 4.28 and 5.36 on three comparable EukProt
pairs.

So the gate is exposed as `target_coverage`, a quantile of z rather than a value
of it. Gating to 90% coverage raised mean accuracy from 0.834 to 0.847 and the
mean margin over DBV from +0.012 to +0.025 across 13 pairs, and lands on the
requested coverage exactly. `z_max` is retained for control.

Parity still degrades under gating on some pairs (0.87-0.94 at the 90% target on
four of thirteen) and that is a property of the mixture, not of how the
threshold is expressed -- the coverage-targeted form does not fix it. Worth
reporting per run rather than assuming.

## Rejected: merging bins before assignment

K.2 emits many small bins (837-3769), which is more than DBV's 384-1435 and
looks like something to consolidate. Two merge criteria were built and measured,
both applied to the core bins before K.3.

**Distance-based.** For bin A let r_A be the mean distance from A's centroid to
its members and s_A the sd of those; merge A and B when each centroid sits
inside the other's envelope, `(d(cA,cB) - r_A)/s_A <= t` in both directions.

**VLP-based.** One `compute_scores` pass gives the per-base log-likelihood of
every core contig under every bin's trie. Merge when each model explains the
other's members nearly as well as their own does:
`delta_A = mean_{i in A}[L(i|A) - L(i|B)]`, merge if `max(delta_A, delta_B) <= t`.

Both lose, on both a separable and an overlapping pair, at every threshold:

| pair | setting | bins | core purity | final acc |
|---|---|---|---|---|
| EP00244+EP00245 | no merge | 1649 | 0.866 | 0.765 |
| | VLP t=0.001 | 1128 | 0.803 | 0.740 |
| | distance t=0 | 1241 | 0.759 | 0.694 |
| EP00002+EP00271 | no merge | 3463 | 0.999 | 0.977 |
| | VLP t=0.0001 | 2610 | 0.973 | 0.966 |
| | distance t=-1 | 3059 | 0.840 | 0.900 |

Three things worth keeping from this:

**VLP's trie models make genuinely better merge decisions than the k-mer
distance** -- 0.803 at 1128 bins against 0.759 at 1241, i.e. more merging for
less damage. It is the one place in this work where VLP's richer representation
measurably beat the 4-mer distance. It still was not enough.

**The failure is leverage, not accuracy.** A misassigned contig costs one
contig; a merge that crosses a species boundary costs both bins entirely plus
everything K.3 subsequently routes into them. With ~1600 bins there are tens of
thousands of merge opportunities per round, so a handful being wrong is enough.
Better statistics on each decision do not compensate. This is also a sharper
account of DistBin's 0.732 on EP00002+EP00271: its seeds were ~0.997 pure and
its merge step is the same lever with no safety on it.

**The most confident merges are the harmful ones.** On the separable pair,
thresholds from 0.0001 to 0.001 all produce ~2600 bins at 0.972 -- a fixed set
of merges with `delta` near zero fires regardless of threshold, and it is that
set which costs the purity. Same inversion as the assignment gate's confidence
statistic, where accuracy fell as the gate tightened.

So the many-small-bins output is not a defect to clean up afterwards; it is what
the data supports. If fewer bins are wanted, lower `resolution` at K.1, where
Leiden optimises globally and no decision is irreversible.

Cost, for completeness: the VLP pass was 56 s (1649 clusters x 11,142 core
contigs) and 499 s (3463 x 25,044), against 33 s and 113 s for the entire
current pipeline on those pairs.

## Seed selection

`max_bins` keeps only the N largest core bins as assignment seeds. Selection,
not merging: the discarded bins' contigs go back into the candidate pool and K.3
re-judges each individually. Reducing 3463 bins to 64 costs 0.43 accuracy by
merging and 0.006 by selection, because merging needs ~3400 irreversible
decisions and selection needs none.

**Ordering by size beat six alternatives** end to end -- size/radius, size/sd,
sqrt-size/radius, two non-redundancy rules, farthest-point -- across 13 pairs.
It wins or ties on most and is never badly wrong, where the diversity-based
rules lose by up to 0.23. Seeds want to be *populated*, not *distinctive*: a bin
far from everything is more likely an artifact, and making an artifact a seed
lets it attract contigs that belong elsewhere.

**That comparison was measured through K.3, which confounds it.** Judged on the
seeds alone the criteria are nearly identical on purity (within 0.01 on six of
eight pairs) and differ almost entirely in contig mass -- the same 64 bins hold
520 contigs under farthest-point and 5039 under size. So the end-to-end ranking
was measuring seed *mass*, transmitted through the assignment step, not seed
quality. `farthest-point` actually has the highest seed purity on the two pairs
where purity separates at all; it loses because 64 tiny pure bins are poor
targets.

Seed parity is also poor for every criterion on half the pairs (0.31 on skew,
0.44 on cho1+17, 0.50 on medhard for `size`) and K.3 recovers it to 1.00
afterwards. Any evaluation that stops at the core is measuring something that
gets substantially rewritten downstream.

**No data-based cutoff for the seed count exists.** Eleven rules were tested --
seven rank-based (elbow on the size distribution, biggest gap, cumulative
fraction, elbow on size/radius, non-redundancy, contigs-claimed-after-assignment,
assignment fit z) and four absolute size thresholds. Every one either lands at an
endpoint or tracks a quantity that improves monotonically with seed count. The
size distribution is close to a power law, so curvature-based rules hit the ends;
fit quality rises with more seeds, so z measures closeness rather than
correctness.

The accuracy elbow is real -- 0.856 at 17 seeds, 0.970 at 32 on the median pair
-- but only visible with labels, and its location varies (16 seeds on cho1+17,
beyond 512 on skew) without correlating with anything measurable. It sits at
roughly 8-16 modes per organism, which is plausibly codon-usage and expression
strata: a fact about the biology, not about the partition.

So `max_bins` is an explicit reporting budget, not a tuning knob. Truncation cost
from keeping ~everything to keeping 64: 0.010-0.062, and it is *regressive* --
cheapest on the pairs that already separate well (cho1+17 0.010, median 0.012)
and most expensive on the hard ones (skew 0.062, 0244+0245 0.062).

## Assignment methods compared

From one shared core per pair, so the only variable is the assignment step.

| method | EP00463+EP00495 | EP00002+EP00271 | EP00244+EP00245 | cho1+8 |
|---|---|---|---|---|
| argmin z, no gate | **0.832** @ 100% | **0.977** @ 100% | **0.765** @ 100% | 0.765 @ 100% |
| VLP | 0.731 @ 100% | 0.960 @ 100% | 0.720 @ 98% | **0.768** @ 100% |
| argmin z, z<=2 | 0.888 @ 53% | 0.999 @ 51% | 0.837 @ 46% | 0.878 @ 34% |
| argmin d, z<=2 | 0.903 @ 46% | 0.999 @ 48% | 0.850 @ 42% | 0.890 @ 31% |
| neighbour vote | 0.903 @ 51% | 0.999 @ 46% | 0.852 @ 39% | 0.884 @ 29% |

**VLP** (DBV's trie-based reassignment) is 4-11x slower and loses on three of
four pairs, tying on the hardest. Given 0.90-0.999 pure seeds it pulls results
down toward its own ceiling rather than building on them, which suggests its
gains inside DBV come from repairing bad DistBin seeds. Its cost is superlinear
in (clusters x contigs): 23 s for 837 x 16,601, 911 s for 3,463 x 65,025.

**The neighbour vote** — K.3's original form, a majority vote among a contig's
nearest already-binned neighbours — scored 0.688 against a 0.685 base rate on
the *old* core of a few large bins, because a purely relative rule against an
unbalanced core returns the majority organism by construction. Against the
current core of many small pure bins it is competitive (0.903 where the fit test
also scored 0.903). It was replaced for the wrong reason but the replacement
still wins on coverage and parity.

## Reducing the bin count

K.2 emits 800-5000 bins. Getting that down turns out to be governed by one
relationship and one hard limit.

**n_neighbors and resolution are a single knob.** Accuracy depends only on the
bin count they jointly produce, not on which of them produced it. Two different
paths to the same count give the same result:

| k | res | bins | core purity | final |
|---|---|---|---|---|
| 2048 | 0.005 | 373 | 0.976 | 0.959 |
| 4096 | 0.02 | 365 | 0.975 | 0.958 |

So there is one accuracy-vs-bin-count curve per mixture and (k, res) only
selects a point on it. Given a target, use the smallest k that reaches it --
larger k costs O(n*k) memory in the mutual filter and buys nothing more.

Mechanically: CPM admits a node to a community of size n_c only if it has more
than `res * n_c` edges into it, so the bar rises with community size. Raising k
adds edges (bigger numerator); lowering res lowers the bar. Both grow
communities. They differ in *quality* because k changes which contigs are
adjacent at all -- importing cross-species neighbours as neighbourhoods reach
further -- while res only relaxes the threshold on edges already present. Large
k with low res is therefore the worst corner available (core purity 0.63).

**The floor is graph connectivity, not resolution.** Below some res the bin
count simply stops falling, because Leiden cannot merge across components. The
floor tracks component count closely: at k=512 the hard pair's graph has 1566
components and floors at ~1030 bins; at k=2048 it has 118 and floors at ~373.
Raising k is therefore a prerequisite for res to have room to work.

| k | edges | components | giant component |
|---|---|---|---|
| 16 | 20,243 | 8,573 | 27.6% |
| 512 | 731,443 | 1,566 | 88.7% |
| 2048 | 4,918,856 | 118 | 99.1% |

**Every mixture has a cliff.** Coarsening is safe until one community spans both
organisms, then core purity collapses in a single step (0.976 -> 0.711 on
medhard). Past the cliff further merging is nearly free -- 254 down to 31 bins
costs only 0.064 -- because the damage is already done. Below the cliff res also
stops controlling the count monotonically, so a bisection search for a target
must stay above it.

**Reduction tolerance is predicted by core purity at the default setting**, with
a threshold near 0.995 (not 0.95, which is what intuition suggests). Using
"deepest setting whose core purity stays within 0.03 of default":

| pair | DBV | core@def | default | best reduced | ratio | cost |
|---|---|---|---|---|---|---|
| EP00002+EP00271 | 0.959 | 0.999 | 3463 @ 0.977 | 142 @ 0.915 | 24x | -0.062 |
| median EP00161+EP01094 | 0.980 | 0.997 | 2895 @ 0.984 | 129 @ 0.943 | 22x | -0.041 |
| medhard EP00088+EP01125 | 0.929 | 0.996 | 5338 @ 0.975 | 373 @ 0.959 | 14x | -0.016 |
| EP00002+EP00178 | 0.930 | 0.997 | 3769 @ 0.921 | 1290 @ 0.906 | 2.9x | -0.015 |
| cho1+7 | 0.795 | 0.991 | 3087 @ 0.917 | 1160 @ 0.906 | 2.7x | -0.011 |
| cho1+17 | 0.771 | 0.993 | 3675 @ 0.918 | 1764 @ 0.905 | 2.1x | -0.013 |
| EP00178+EP00657 | - | 0.986 | 1525 @ 0.930 | 810 @ 0.929 | 1.9x | -0.001 |
| 5 feature-limited pairs | 0.647-0.867 | 0.866-0.978 | - | none | 1.0x | - |

Three bands, nearly non-overlapping: 0.996-0.999 tolerates 14-24x, 0.986-0.993
tolerates 2-3x, below 0.91 tolerates none. cho1+14 (core 0.978, no reduction) is
the one case that breaks the banding, so treat it as an indicator, not a rule.

**A ~64-bin target is not reachable.** Every pair crosses its cliff before
getting there; the cost is 0.13-0.21 accuracy and lands below DBV. The
achievable range on separable mixtures is 129-373 bins.

**A label-free stopping rule exists.** A cross-species merge shows up as a
discontinuous jump in the largest community size, where within-species merging
grows it smoothly. Sweeping res downward and stopping at the last value before
that jump picks the right level per mixture without ground truth, which a fixed
default cannot do -- the right answer ranges from 1764 bins to 129 across these
pairs.

**At low bin counts K.3 becomes the constraint on clean pairs.** On
EP00002+EP00271 at 142 bins the core is 0.991 but the final is 0.915 -- the
partition holds the information and the assignment step loses it. The cause is
not sample size: `assign_sample` 30 -> 100 recovers 0.024 but 300 is worse than
100, and widening `assign_top_bins` to 15 hurts (0.950 -> 0.933). A coarse bin
is a union of several fine ones, so its distance distribution is multi-modal and
a single (mu, sigma) envelope describes it badly; sampling it more thoroughly
just inflates sigma and admits contigs it should refuse. The targeted fix is to
score against a bin's *nearest* members rather than a uniform sample.

## Staged assignment, and two graph-based alternatives

All measured from the same start: seeds = core bins with >= 32 members, which is
17-99 bins covering 6-10% of a mixture. Compared against a single
`_assign_by_fit` pass from those same seeds.

**Length-staged + confidence-ordered + residual reclustering** is what shipped.
Tranches at length percentiles 75/50/25/0, admitting the best 75% of each by z
and recomputing envelopes between stages, then reclustering the leftovers:

| pair | staged | fit alone | delta | DBV | bins | parity |
|---|---|---|---|---|---|---|
| skew (EP00178+EP00657) | 0.893 | 0.833 | +0.060 | 0.943 | 135 | 0.98 |
| median (EP00161+EP01094) | 0.975 | 0.972 | +0.003 | 0.980 | 189 | 1.00 |
| clean (EP00002+EP00271) | 0.966 | 0.963 | +0.003 | 0.959 | 204 | 0.97 |
| 0244+0245 | 0.673 | 0.673 | 0.000 | 0.672 | 101 | 0.99 |
| cho1+8 | 0.714 | 0.715 | -0.001 | 0.647 | 228 | 0.99 |
| 0463+0495 | 0.769 | 0.773 | -0.004 | 0.729 | 34 | 0.99 |
| medhard (EP00088+EP01125) | 0.961 | 0.965 | -0.004 | 0.929 | 338 | 0.99 |
| cho1+17 | 0.892 | 0.907 | -0.015 | 0.771 | 268 | 1.00 |

Free-or-better on seven of eight, coverage 96.7-99.4%, parity 0.97-1.00
everywhere -- the most uniformly balanced output measured. 2-10 s on top of the
graph build.

The +0.060 is specific to length-skewed mixtures: skew's parity jumps 0.35 ->
0.96 exactly at the p75 -> p50 boundary, i.e. the minority organism's contigs sit
in the shorter half. Pairs whose organisms have similar length distributions have
nothing for the tranches to separate.

`recluster_residual` pays unconditionally: 5.0-8.0% recovered per pair at
0.717-0.958 purity, and on 0244+0245 and cho1+8 the new bins were *purer* than
the main assignment (0.717 vs 0.669; 0.737 vs 0.713).

`stage_admit` is unvalidated -- set to 0.75 and never varied. Its main visible
effect is fixing the residual at ~8.2% of every mixture (0.75^4 compounding),
which is a parameter artifact, not a property of the data.

## Graph propagation and label spreading: tried, not adopted

Both spread seed labels along the kNN graph instead of comparing centroids.
Attractive because the noise bin falls out for free -- nodes with no path to a
seed are never labelled, 2.9-16.2% by pair -- and because hop and vote counts are
portable where z is not.

**Majority propagation** (each unlabelled node takes its neighbours' majority,
one hop at a time) converges in ~8 rounds, under a second. Purity decays per hop
as errors spread transitively: on skew 0.993 (seeds) -> 0.947 (1 hop) -> 0.922
(2) -> 0.915 (converged, 83.7% of the band). It matched the fit test on two pairs
and beat it on skew (0.915 vs 0.855).

**A confidence requirement** recovers much of that purity: >= 5 labelled
neighbours with a bare majority gave 0.940 purity and 0.99 parity at 59% of the
band on skew. Requiring *several neighbours to have an opinion* helps far more
than requiring them to *agree strongly* -- raising the required share from 0.5 to
1.0 costs 27-36 points of coverage for ~0.008 purity, while raising the required
vote count from 1 to 3 costs 10 points and buys 0.004 purity plus 0.13 parity. A
node with one labelled neighbour is unreliable however unanimous that vote is.

**Damped label spreading** (seeds re-injected each iteration, so propagated
labels never become sources) was expected to fix error spread at its root. It did
not, and the result diagnoses the mechanism: heavy damping is much *worse*. At
the loosest gate, alpha=0.5 gives 0.811 purity on 0463+0495 where alpha=0.99
gives 0.567; 0.779 vs 0.553 on cho1+8. Since alpha changes only which labels win
and never how many nodes are reachable, those are the same contigs relabelled.
Errors therefore come from signal travelling *too far*, not from labels
amplifying themselves.

All three controls -- hop count, vote threshold, damping -- land on nearly the
same purity/coverage frontier, so they are three parameterisations of one
tradeoff rather than distinct mechanisms.

Why none shipped: propagation is confined to the p45-p95 band, so even at its
best settings it labels 16-48% of a whole mixture and still needs a second stage.
The staged pipeline above reaches 96.7-99.4% and beats it. The free noise bin and
the portable dials are the reasons to revisit.

## Where the remaining headroom is

Not in these parameters. On closely-related pairs — charophyte algae, a hornwort
against a gymnosperm, radiolarians, loricate choanoflagellates — the median rank
of the first cross-species neighbour is **5**, against 800 on separable pairs.
The five nearest contigs are already contaminated, so no threshold, rank or
resolution recovers them. That is a property of the 4-mer representation.

Dimension reduction (PCA on the k-mer vectors) is unlikely to help: a linear
projection removes noise directions but cannot separate genuinely overlapping
distributions, and with no labels nothing steers the projection toward the
discriminating axis. Cheaply falsifiable if wanted — PCA, then re-measure top-10
neighbour purity.

Frame-aware features are the only lever that adds information rather than
reorganising it, and ORF coordinates are already available from the peptide
files. The cheap test is whether in-frame codon usage or GC3 moves top-10
neighbour purity (0.754 on EP00244+EP00245) — no clustering or parameters
required.

## Open questions

Roughly in order of how much they would change the picture.

* **Everything here is two-species mixtures.** The stated use case is an unknown
  number of species. `max_bins` is calibrated for two organisms and would
  under-seed a ten-species sample; the recall-parity metric is undefined past
  two. 120 EukProt transcriptomes are on disk, so 3/5/10/20-species mixtures are
  straightforward to build and `prefix_len=7` grading already handles any number.
  Untested, and it is the case the tool is for.

* **The benchmark set is not representative.** Across the 2608 pairs in the
  stored sweep DistBin's median is 0.882 and DBV's is 0.982; every pair tested
  here sits below both, most far below, because the set was assembled by picking
  pairs those two scored badly on. The one pair drawn from the middle (median,
  DBV 0.980) scores 0.975-0.984. A random sample of pairs would say what the real
  comparison is.

* **Bin counts are not comparable to DBV's.** We emit 34-3769 where DBV emits
  2-1081, and `grade_bins` credits each bin's plurality organism, so the accuracy
  column flatters us by an unquantified amount. Scoring both at a matched bin
  count would settle it, and `max_bins` makes that possible.

* **The length band is still not species-neutral.** p45-p95 turns a 59/41
  mixture into an 83/17 selection on EP00002+EP00178. A stratified sample across
  length deciles would be neutral by construction -- a small change to
  `lib/selection.py`, untested.

* **K.3's single-(mu, sigma) envelope misdescribes large bins.** At 142 bins on
  EP00002+EP00271 the core holds 0.991 while the final is 0.915. Not a sampling
  problem: `assign_sample` 30 -> 100 recovers 0.024 but 300 is worse than 100, and
  widening `assign_top_bins` to 15 hurts. A coarse bin is a union of fine ones,
  so its distance distribution is multi-modal. Scoring against a bin's *nearest*
  members rather than a uniform sample is the targeted fix.

* **Frame-aware features are the only untried lever that adds information.** On
  closely-related pairs the median rank of the first cross-species neighbour is 5
  (against 800 on separable pairs), so no threshold, rank or resolution recovers
  them -- that is a property of the 4-mer representation. ORF coordinates are
  already available from the peptide files. The cheap test is whether in-frame
  codon usage or GC3 moves top-10 neighbour purity (0.754 on EP00244+EP00245); no
  clustering or parameters required. Dimension reduction is unlikely to help: a
  linear projection cannot separate genuinely overlapping distributions, and with
  no labels nothing steers it toward the discriminating axis.

* **DBV emits 384-1435 bins while DistBin hands it 3-22 and VLP only culls
  clusters.** Never reconciled. It bears on whether seeding DBV from the KNN core
  is worthwhile -- VLP scored below ungated `argmin z` on three of four pairs, so
  the answer may be that the core is better used without VLP.

* **No data-based seed cutoff exists** (see Seed selection). Eleven rules tested.
  The one untried signal is stability across resampling or feature splits, which
  measures reproducibility rather than fit or size -- the only candidate that
  could mark the accuracy elbow. Costs several full runs per setting.

## Reproducing any of this

The experiment scripts lived in session-temp and are gone; the numbers above are
the record. Nothing in `scripts/` reproduces the sweeps. `run_12417881_knn.py`
runs one pair with any config override, which is enough to re-derive a single
row; the multi-pair comparisons would need rebuilding.
