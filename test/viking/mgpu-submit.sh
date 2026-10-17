#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Submits the multi-GPU jobs (one GPU per rank) on Viking:
#
#   mgpu-check     multi-GPU validation (developers/test/validation/mgpu_check.sh)
#   mgpu-scaling   strong and weak scaling timings, after the check succeeded
#
# on NODES A40 nodes (gpu, 3 GPUs each; default 2 nodes = 6 GPUs, the most a
# user may have) and one H100 node (gpuplus, 2 GPUs).
#
# Usage (from the vampire checkout on a Viking login node):
#   test/viking/mgpu-submit.sh [commit]      (default: HEAD)
# Environment: WORK (default ~/scratch/vampire-benchmarks), NODES, MODULES.
#------------------------------------------------------------------------------
set -eu
cd "$(dirname "$0")"
if grep -l VIKING_ACCOUNT mgpu-*.job; then
   echo "Replace the VIKING_ACCOUNT placeholder in the files above with your project account first."
   exit 1
fi

export REPO=$(git rev-parse --show-toplevel)
export WORK=${WORK:-$HOME/scratch/vampire-benchmarks}
mgpu=$WORK/mgpu
mkdir -p $mgpu/logs
commit=$(git rev-parse ${1:-HEAD})
echo "$commit" > $mgpu/commit.txt
rm -rf $mgpu/src; mkdir -p $mgpu/src
git -C $REPO archive $commit | tar -x -C $mgpu/src
echo "commit $commit"

nodes=${NODES:-2}
opts="--parsable --export=ALL,REPO=$REPO,WORK=$WORK --output=$mgpu/logs/%x-%j.log"
a40="--partition=gpu --nodes=$nodes --ntasks-per-node=3 --gres=gpu:3"
h100="--partition=gpuplus --nodes=1 --ntasks-per-node=2 --gres=gpu:2"
for shape in "$a40" "$h100"; do
   check=$(sbatch $opts $shape mgpu-check.job)
   sbatch $opts $shape --dependency=afterok:$check mgpu-scaling.job
done
echo "Submitted. Monitor with: squeue --me    Collect with: test/viking/mgpu-collect.sh"
