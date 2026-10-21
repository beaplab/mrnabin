# KLV

KNN + Leiden + VLP: the [KNN](../knn/README.md) graph clustering proposes bins,
and VLP trie models merge them down to a handful per mixture.

## Getting Started

`uv run scripts/run_binner.py klv file contig_file peptide_file output_file`

The output is the usual TSV of contig id and bin label (`bin_none` for
unbinned). In code, `KLVBinner` follows DBV's contract: set
`binner.sequences` to the list of `Sequence` from `lib.fasta.get_sequences`,
call `run()`, and read the bins with `lib.evaluate.bins_by_id`.

## Algorithm

1. Run KNN + Leiden (all of `config/base_knn_config.yaml`) and keep
   its `vlp_seed_bins` largest bins as clusters. Contigs in the other bins start
   unbinned.
2. Randomly split the contigs into two halves (folds), and assign
   every contig to one of the two folds, fixed for the run. Each
   cluster gets one VLP trie per fold, trained on that fold's members,
   and every contig is scored by the trie of the fold it is *not*
   in. All scores are held out, so a contig never looks good in a
   cluster just because it trained that cluster's model.
3. DBV's length schedule (median, p25, p10, then everything) activates
   contigs and reassigns each to its best-scoring cluster, or leaves
   it unbinned if no model works.
4. Repeatedly dissolve the most absorbable cluster -- the one whose
   members score best elsewhere relative to at home -- and reassign its contigs.
   Stop, undoing the last step, when held-out log-likelihood per base, over all
   active contigs with unbinned ones counting as background, falls by more than
   `vlp_tolerance`.

## Parameters

`config/base_klv_config.yaml`, applied over `base_dbv_config.yaml` and
`base_knn_config.yaml`:

* `vlp_seed_bins = 32` — KNN bins passed on as clusters.
* `vlp_holdout = true` — the held-out merge above. `false` runs DBV's own VLP
  phase on the seeds instead
* `vlp_tolerance = 0.002` — the merge stops when held-out fit per base falls by
  more than this in one step.
