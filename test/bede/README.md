# GPU benchmarks on Bede (N8 CIR)

Slurm scripts that time the CUDA version of two commits ("before" and "after") on both Bede
node types, profile `hamr-perf` on both GPUs, and record the software available on the nodes:

| node type | CPU | GPU | partitions | submitted with |
|---|---|---|---|---|
| ppc64le | POWER9 | V100 32 GB (sm_70) | `gpu`, `test` | `sbatch` |
| aarch64 | Grace | GH200 H100 96/144 GB (sm_90) | `gh`, `ghtest` | `ghbatch` |

Binaries are built separately for each node type (`$WORK/bin/<arch>/`).

0. Replace the Slurm account placeholder with your project code:

       sed -i 's/BEDE_ACCOUNT/<project-code>/' test/bede/*.job

1. Check that jobs, modules and the GPUs work (each prints `CHECK PASSED` at the end of its
   `slurm-<id>.out`):

       sbatch  --partition=test   test/bede/check.job
       ghbatch --partition=ghtest test/bede/check.job

2. Submit the benchmark set (default: before = `cf6946e8`, after = `HEAD`):

       test/bede/submit.sh [before-commit] [after-commit]

3. When all jobs have finished, collect the results into one archive:

       test/bede/collect.sh

The benchmark cases are the same as on Viking (see `test/viking/README.md`). Everything is
written to `$WORK` (default `~/vampire-benchmarks`, under 1 GB). Modules default to
`gcc/12.2` with `cuda/12.4.1` (ppc64le) or `cuda/12.6.1` (aarch64); override with `MODULES`.
