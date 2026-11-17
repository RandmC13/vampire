#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Submits the capped three-track HAMR run (inputs in test/viking/capped) on
# Viking:
#
#   capped-write    the write on 2 A40 nodes (6 GPUs), as SEGMENTS jobs of 3 days
#                   (default 4), each continuing the previous one (afterany)
#   capped-read     the read-back after each of the three tracks, CPU MPI
#                   version on 10 nodes, after the last write segment
#
# A write segment that starts after the write has finished does nothing. With
# --dry-run the same jobs run on a 40 nm long film with a fast head on the
# gpu_short (one A40) and test partitions, to check the chain first. With
# --test they write two tracks on a 100 x 98 nm film with the physics of the
# run (one write job of at most 4 hours, about 2.5 hours expected), on one
# node of 2 H100 (gpuplus), or of 6 A40 on 2 nodes with TEST_GPUS=a40.
#
# Usage (from the vampire checkout on a Viking login node):
#   test/viking/capped-submit.sh [--dry-run | --test] [commit]      (default: HEAD)
# Environment: WORK (default ~/scratch/vampire-benchmarks; not on the file
# system of the home directory), SEGMENTS, TEST_GPUS, MODULES.
#------------------------------------------------------------------------------
set -eu
cd "$(dirname "$0")"
mode=full
case "${1:-}" in --dry-run) mode=dry-run; shift;; --test) mode=test; shift;; esac
if grep -l VIKING_ACCOUNT capped-*.job; then
   echo "Replace the VIKING_ACCOUNT placeholder in the files above with your project account first."
   exit 1
fi
export REPO=$(git rev-parse --show-toplevel)
export WORK=${WORK:-$HOME/scratch/vampire-benchmarks}
export CAP_MODE=$mode
cap=$WORK/capped
mkdir -p $cap
if [ "$(df -P $cap | awk 'NR == 2 { print $6 }')" = "$(df -P $HOME | awk 'NR == 2 { print $6 }')" ]; then
   echo "Error: $cap is on the file system of the home directory; set WORK to a directory in scratch"
   exit 1
fi
if [ -n "$(ls -A $cap 2> /dev/null | grep -v '^logs$')" ]; then
   echo "Error: $cap is not empty; move or delete the results of the previous run first"
   exit 1
fi
mkdir -p $cap/logs
commit=$(git rev-parse ${1:-HEAD})
echo "$commit" > $cap/commit.txt
echo "$mode" > $cap/mode.txt
git -C $REPO archive $commit | (mkdir -p $cap/src && tar -x -C $cap/src)
echo "commit $commit, results in $cap"
opts="--parsable --export=ALL,REPO=$REPO,WORK=$WORK,CAP_MODE=$mode --output=$cap/logs/%x-%j.log"
if [ $mode = dry-run ]; then
   write="--partition=gpu_short --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --time=00:30:00"
   read="--partition=test --nodes=1 --ntasks-per-node=8 --time=00:30:00"
   segments=${SEGMENTS:-2}
elif [ $mode = test ]; then
   if [ "${TEST_GPUS:-h100}" = a40 ]; then
      write="--partition=gpu --nodes=2 --ntasks-per-node=3 --gres=gpu:3 --time=04:00:00"
   else
      write="--partition=gpuplus --nodes=1 --ntasks-per-node=2 --gres=gpu:2 --time=04:00:00"
   fi
   read="--partition=nodes --nodes=1 --ntasks-per-node=96 --time=01:00:00"
   segments=${SEGMENTS:-1}
else
   write="--partition=gpu --nodes=2 --ntasks-per-node=3 --gres=gpu:3 --time=3-00:00:00"
   read="--partition=nodes --nodes=10 --ntasks-per-node=96 --time=06:00:00"
   segments=${SEGMENTS:-4}
fi
id=$(sbatch $opts $write capped-write.job)
echo "write segment 1: job $id"
for ((s = 2; s <= segments; s++)); do
   id=$(sbatch $opts $write --dependency=afterany:$id capped-write.job)
   echo "write segment $s: job $id"
done
id=$(sbatch $opts $read --dependency=afterany:$id capped-read.job)
echo "read: job $id"
echo "Submitted. Monitor with: squeue --me    Progress: tail $cap/write/log    Collect with: test/viking/capped-collect.sh"
