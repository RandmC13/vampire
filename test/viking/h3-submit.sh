#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Submits the HAMR chain (inputs in test/viking/h3) and its benchmarks on Viking:
#
#   h3-relax        Monte Carlo relaxation, CPU MPI version on 10 nodes
#   h3-bench-gpu    timed writes on 2 A40 nodes (6 GPUs) and 1 H100 node (2 GPUs)
#   h3-bench-cpu    timed writes with the CPU MPI version on 1, 4 and 10 nodes
#   h3-write        complete write, single precision, 6 A40 GPUs
#   h3-read         read-back from the end of the write, CPU MPI version on 10 nodes
#
# Every stage after the relaxation starts from a checkpoint written on another
# number of processes. With --dry-run the same jobs run on a 24 x 24 x 20 nm
# system with short runs on the test and gpu_short partitions (and 1 H100),
# to check the chain and where its files go before the full runs.
#
# Usage (from the vampire checkout on a Viking login node):
#   test/viking/h3-submit.sh [--dry-run] [commit]      (default: HEAD)
# Environment: WORK (default ~/scratch/vampire-benchmarks; must not be on the
# file system of the home directory), MODULES.
#------------------------------------------------------------------------------
set -eu
cd "$(dirname "$0")"
dry=0
[ "${1:-}" = "--dry-run" ] && { dry=1; shift; }
if grep -l VIKING_ACCOUNT h3-*.job; then
   echo "Replace the VIKING_ACCOUNT placeholder in the files above with your project account first."
   exit 1
fi

export REPO=$(git rev-parse --show-toplevel)
export WORK=${WORK:-$HOME/scratch/vampire-benchmarks}
export H3_DRY=$dry
h3=$WORK/h3
mkdir -p $h3
if [ "$(df -P $h3 | awk 'NR == 2 { print $6 }')" = "$(df -P $HOME | awk 'NR == 2 { print $6 }')" ]; then
   echo "Error: $h3 is on the file system of the home directory; set WORK to a directory in scratch"
   exit 1
fi
if [ -n "$(ls -A $h3 2> /dev/null | grep -v '^logs$')" ]; then
   echo "Error: $h3 is not empty; move or delete the results of the previous chain first"
   exit 1
fi
mkdir -p $h3/logs
commit=$(git rev-parse ${1:-HEAD})
echo "$commit" > $h3/commit.txt
echo "$dry" > $h3/dry-run.txt
git -C $REPO archive $commit | (mkdir -p $h3/src && tar -x -C $h3/src)
echo "commit $commit, results in $h3"

opts="--parsable --export=ALL,REPO=$REPO,WORK=$WORK,H3_DRY=$dry --output=$h3/logs/%x-%j.log"
if [ $dry = 1 ]; then
   cpu="--partition=test --nodes=1 --ntasks-per-node=8 --time=00:30:00"
   a40="--partition=gpu_short --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --time=00:30:00"
   h100="--partition=gpuplus --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --time=00:30:00"
   write=$a40; read=$cpu; relax=$cpu; bench_cpu=$cpu; bench_a40=$a40; bench_h100=$h100
else
   relax="--partition=nodes --nodes=10 --ntasks-per-node=96 --time=12:00:00"
   bench_cpu="--partition=nodes --nodes=10 --ntasks-per-node=96 --time=04:00:00"
   bench_a40="--partition=gpu --nodes=2 --ntasks-per-node=3 --gres=gpu:3 --time=10:00:00"
   bench_h100="--partition=gpuplus --nodes=1 --ntasks-per-node=2 --gres=gpu:2 --time=05:00:00"
   write="--partition=gpu --nodes=2 --ntasks-per-node=3 --gres=gpu:3 --time=24:00:00"
   read="--partition=nodes --nodes=10 --ntasks-per-node=96 --time=04:00:00"
fi
relax_id=$(sbatch $opts $relax h3-relax.job)
after="--dependency=afterok:$relax_id"
sbatch $opts $bench_a40 $after h3-bench-gpu.job
sbatch $opts $bench_h100 $after h3-bench-gpu.job
sbatch $opts $bench_cpu $after h3-bench-cpu.job
write_id=$(sbatch $opts $write $after h3-write.job)
sbatch $opts $read --dependency=afterok:$write_id h3-read.job
echo "Submitted. Monitor with: squeue --me    Collect with: test/viking/h3-collect.sh"
