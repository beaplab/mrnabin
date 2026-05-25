# The OG VLP

The VLP library is based on [mRNAbinner/mar_23_vlp_cluster.F90](https://github.com/beaplab/mRNAbinner/blob/81ee1802444308ab13f727c1b30591d7a704d691/mar_23_vlp_cluster.F90) (also available at: [mar_23_vlp_cluster.F90](mar_23_vlp_cluster.F90)).

Two scripts are direct ports of the original Fortran code. The output should be very similar to the original.

### [mar_23_vlp_cluster_fixed_size_gfortran.F90](mar_23_vlp_cluster_fixed_size_gfortran.F90)
A version of the code modified to compile on a Macbook running OS X. Instead of using giant fixed-size arrays, this version allocates them manually.

`brew install gcc` will also install `gfortran`.

To compile: `gfortran -cpp -O2 -fdec -o mar_23_vlp_cluster_fixed_size_gfortran mar_23_vlp_cluster_fixed_size_gfortran.F90`.

From the project root, run on [sample_data/aa_bb_contigs.fasta](../sample_data/aa_bb_contigs.fasta): `./og_vlp/mar_23_vlp_cluster_fixed_size_gfortran`

### [mar_23_vlp_cluster_opt.py](mar_23_vlp_cluster_opt.py)
A very direct and literal Python version of the code with some JIT added to run in a reasonable amount of time.

From the project root, run on [sample_data/aa_bb_contigs.fasta](../sample_data/aa_bb_contigs.fasta): `uv run og_vlp/mar_23_vlp_cluster_opt.py`

