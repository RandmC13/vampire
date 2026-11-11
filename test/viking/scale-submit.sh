#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Submits the check of hamrvis at scale on Viking: the HAMR chain of
# test/viking/h3 (200 x 60 x 20 nm, 8.7M atoms) run on GPUs, twice from one
# relaxation, then reduced to hamrvis bundles and checked against the raw files.
#
#   scale-build          CPU and GPU binaries (CPU node)
#   scale-relax          Monte Carlo relaxation, 1 H100
#   scale-write window   write with hamr:active-window, 1 H100   } at the same time
#   scale-write full     write with complete snapshots, 1 H100   }
#   scale-read  window   read-back after each write, 2 CPU nodes
#   scale-read  full
#   scale-reduce window  hamrvis reduce + bundle check of each chain, 1 CPU node
#   scale-reduce full
#
# The raw files stay in $WORK/h3/scale until the checks have run. With --dry-run
# the same jobs run on a 24 x 24 x 20 nm system with 30 minute limits on the
# nodes and gpu (A40) partitions (the test and gpu_short partitions allow too
# few jobs per user for the chain), to check it before the full run. If a
# submission fails, the jobs already submitted are cancelled.
#
# Usage (from the vampire checkout on a Viking login node):
#   test/viking/scale-submit.sh [--dry-run] [vampire commit] [hamrvis commit]
#     vampire commit   default HEAD of this checkout
#     hamrvis commit   default the hamr-vis branch of origin (fetched)
# Environment: WORK (default ~/scratch/vampire-benchmarks; not on the file
# system of the home directory), MODULES.
#------------------------------------------------------------------------------
set -eu
cd "$(dirname "$0")"
dry=0
[ "${1:-}" = "--dry-run" ] && { dry=1; shift; }
if grep -l VIKING_ACCOUNT scale-*.job; then
   echo "Replace the VIKING_ACCOUNT placeholder in the files above with your project account first."
   exit 1
fi
export REPO=$(git rev-parse --show-toplevel)
export WORK=${WORK:-$HOME/scratch/vampire-benchmarks}
export H3_DRY=$dry
S=$WORK/h3/scale
mkdir -p $S
if [ "$(df -P $S | awk 'NR == 2 { print $6 }')" = "$(df -P $HOME | awk 'NR == 2 { print $6 }')" ]; then
   echo "Error: $S is on the file system of the home directory; set WORK to a directory in scratch"
   exit 1
fi
if [ -n "$(ls -A $S 2> /dev/null | grep -v '^logs$')" ]; then
   echo "Error: $S is not empty; move or delete the results of the previous check first"
   exit 1
fi
mkdir -p $S/logs
commit=$(git rev-parse ${1:-HEAD})
if [ -n "${2:-}" ]; then
   hv=$(git -C $REPO rev-parse $2)
else
   git -C $REPO fetch -q origin hamr-vis
   hv=$(git -C $REPO rev-parse FETCH_HEAD)
fi
echo "$commit" > $S/commit.txt
echo "$hv" > $S/hamrvis-commit.txt
echo "$dry" > $S/dry-run.txt
git -C $REPO archive $commit | (mkdir -p $S/src && tar -x -C $S/src)
git -C $REPO archive $hv | (mkdir -p $S/hamrvis && tar -x -C $S/hamrvis)
echo "vampire $commit, hamrvis $hv, results in $S"

exp="--export=ALL,REPO=$REPO,WORK=$WORK,H3_DRY=$dry"
opts="--parsable --output=$S/logs/%x-%j.log"
if [ $dry = 1 ]; then
   cpu="--partition=nodes --time=00:30:00"
   gpu="--partition=gpu --time=00:30:00"
   read="--partition=nodes --nodes=1 --ntasks-per-node=8 --time=00:30:00"
   small="--cpus-per-task=8 --mem=16G"
else
   cpu="--partition=nodes"
   gpu="--partition=gpuplus"
   read="--partition=nodes --nodes=2 --ntasks-per-node=96 --time=01:00:00"
   small=""
fi
# submit <sbatch arguments>: prints the job id; if sbatch fails, cancels the jobs of this chain submitted so far
submit(){
   local id
   id=$(sbatch "$@") || { echo "Submission failed: cancelling the jobs of this chain submitted so far: $(tr '\n' ' ' < $S/jobs.txt)" >&2; [ -s $S/jobs.txt ] && scancel $(cat $S/jobs.txt); exit 1; }
   echo $id >> $S/jobs.txt
   echo $id
}
: > $S/jobs.txt
build=$(submit $opts $exp $cpu $small $([ $dry = 1 ] || echo --time=01:00:00) scale-build.job)
relax=$(submit $opts $exp $gpu $([ $dry = 1 ] || echo --time=02:00:00) --dependency=afterok:$build scale-relax.job)
for kind in window full; do
   t=$([ $kind = window ] && echo 03:00:00 || echo 04:00:00)
   write=$(submit $opts $exp,KIND=$kind $gpu $([ $dry = 1 ] || echo --time=$t) --job-name=vampire-scale-write-$kind --dependency=afterok:$relax scale-write.job)
   rd=$(submit $opts $exp,KIND=$kind $read --job-name=vampire-scale-read-$kind --dependency=afterok:$write scale-read.job)
   red=$(submit $opts $exp,KIND=$kind $cpu $small $([ $dry = 1 ] || echo --time=04:00:00) --job-name=vampire-scale-reduce-$kind --dependency=afterok:$rd scale-reduce.job)
   echo "$kind: write $write, read $rd, reduce + check $red"
done
echo "build $build, relax $relax"
echo "Submitted. Monitor with: squeue --me    Collect with: test/viking/scale-collect.sh"
