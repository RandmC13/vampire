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

## Multi-GPU version

The multi-GPU version (`make cuda-multigpu`, one MPI rank per GPU) is checked and timed with

    test/viking/mgpu-submit.sh [commit]      # default: HEAD
    test/viking/mgpu-collect.sh              # when all jobs have finished

which submits, on two A40 nodes (6 GPUs; `NODES` changes the number of nodes) and one H100
node (2 GPUs):

- `mgpu-check.job`: `developers/test/validation/mgpu_check.sh` on 1, 2, one node's and all GPUs
  (bit-identical spins on any number of GPUs, comparison with the CPU versions)
- `mgpu-scaling.job` (after the check succeeded): strong scaling of `hamr-perf` and
  `thermal-perf` at 100 x 100 and 200 x 200 nm, and weak scaling of `thermal-perf` at
  100 x 100 nm per GPU, in double and single precision, on 1 GPU up to all GPUs, including 2
  GPUs on two nodes. Results in `$WORK/mgpu/scaling/<partition>.csv`, with the time each run
  spent on halo exchange and the part of it the GPU waited for, i.e. not hidden behind the
  core-atom fields (reported by vampire for every rank).

Ranks are started by `mgpu-launch.sh`, which gives every rank one GPU and binds it to the
CPU cores and memory of that GPU's NUMA node. Modules default to `CUDA/12.9.0`,
`OpenMPI/4.1.5-GCC-12.3.0` and `SciPy-bundle/2023.07-gfbf-2023a`; override with `MODULES`.

## HAMR chain at production size

`h3/` holds the three stages of a HAMR simulation (200 x 60 x 20 nm granular FePt film with a
sensor layer, about 9M atoms): a Monte Carlo relaxation, the HAMR write and a one step dipole
read-back, each restarted from the checkpoint of the previous stage.

    test/viking/h3-submit.sh --dry-run       # first: the whole chain on a small system
    test/viking/h3-collect.sh                # check the archive, then delete $WORK/h3
    test/viking/h3-submit.sh                 # the full chain and its benchmarks
    test/viking/h3-collect.sh

submits the relaxation on 10 CPU nodes (CPU MPI version), then the complete write in single
precision on 6 A40 GPUs (multi-GPU version, `gpu:skip-non-magnetic-atoms`), the read-back on 10
CPU nodes, and benchmarks of the write: restarted from the relaxed state and stopped after 5
minutes of simulation, on 1, 2, 3 and 6 A40 and 1 and 2 H100 GPUs (double and single precision,
with and without skipping the non-magnetic atoms) and with the CPU MPI version on 1, 4 and 10
nodes. `$WORK/h3/bench/bench.csv` lists for each run the set-up time, the time steps per
second, the hours the whole write (4.29M steps) would take, and the peak host and GPU memory
per rank. Every stage after the relaxation loads a checkpoint written on another number of
processes.

All files go to `$WORK/h3`, which must not be on the file system of the home directory (the
complete write saves about 90 GB of spin snapshots); the scripts stop otherwise.
`h3-collect.sh` packs everything except the checkpoints and the spin snapshots other than the
last one of the write.

`h3-profile.job` (submitted by hand once the relaxation has finished, see its header) profiles
the write on the H100 node: the original version (cf6946e8) on `hamr-perf` and timed at
production size, and the multi-GPU version on 1 and 2 GPUs under Nsight Systems (CUDA and MPI
calls of each rank). Its results go to `$WORK/h3/profile` and into the `h3-collect.sh` archive
(without the `.nsys-rep` files, which stay in scratch for the Nsight Systems GUI).
