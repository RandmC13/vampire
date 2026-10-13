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

After `submit.sh` has run (it exports the sources), two further jobs can be submitted from the
checkout with `REPO` and `WORK` exported (`export REPO=$PWD WORK=~/scratch/vampire-benchmarks`):

    # physics validation suite on each GPU type (under an hour each)
    sbatch --partition=gpu     --output=$WORK/logs/%x-gpu-%j.log     test/viking/validate.job
    sbatch --partition=gpuplus --output=$WORK/logs/%x-gpuplus-%j.log test/viking/validate.job
    # GPU interconnect and copy bandwidth of a full node
    sbatch --partition=gpu     --gres=gpu:3 --output=$WORK/logs/%x-gpu-%j.log     test/viking/topology.job
    sbatch --partition=gpuplus --gres=gpu:2 --output=$WORK/logs/%x-gpuplus-%j.log test/viking/topology.job

Everything is written to `$WORK` (default `~/scratch/vampire-benchmarks`). The CUDA module
can be chosen with `CUDA_MODULE` (default: the module system's default `CUDA`).
