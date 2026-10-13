# GPU benchmarks on Viking (University of York)

Slurm scripts that time the CUDA version of two commits ("before" and "after") on the
A40 (`gpu`) and H100 (`gpuplus`) partitions, profile `hamr-perf` on both GPUs, and record
the software available on the GPU nodes.

1. Replace the Slurm account placeholder with your project code:

       sed -i 's/VIKING_ACCOUNT/<project-code>/' test/viking/*.job

2. Submit (default: before = `cf6946e8`, after = `HEAD`):

       test/viking/submit.sh [before-commit] [after-commit]

3. When all jobs have finished, collect the results into one archive:

       test/viking/collect.sh

Benchmark runs are the validation cases `hamr-perf` and `thermal-perf` (~168k atoms), plus
`thermal-perf` with lateral sizes of 5, 10, 20 and 100 nm (about 3.8k to 1.36M atoms), each
in double and single precision with 3 repeats. `bench.csv` lists, per run, the wall time and
the simulation time reported by vampire (which excludes system generation).

Everything is written to `$WORK` (default `~/scratch/vampire-benchmarks`). The CUDA module
can be chosen with `CUDA_MODULE` (default: the module system's default `CUDA`).
