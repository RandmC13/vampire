#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Submits the GPU benchmark set on Bede, for both node types:
#
#   recon     software/hardware survey (test, ghtest)
#   build     CUDA binaries (dp, sp) of the two benchmarked commits (test, ghtest)
#   bench     before/after timings on V100 (gpu) and GH200 (gh), 72 runs each
#   profile   Nsight profiles of hamr-perf on V100 and GH200 (after version)
#
# Usage (from the vampire checkout on a Bede login node):
#   test/bede/submit.sh [before-commit] [after-commit]
# Defaults: before = cf6946e8 (validated GPU code before the performance work),
#           after  = HEAD.
#
# Environment: WORK (default ~/vampire-benchmarks).
# The job files contain the placeholder BEDE_ACCOUNT for the project code,
# which must be replaced first.
#------------------------------------------------------------------------------
set -eu
cd "$(dirname "$0")"
if grep -l BEDE_ACCOUNT *.job; then
   echo "Replace the BEDE_ACCOUNT placeholder in the files above with your project code first."
   exit 1
fi

export REPO=$(git rev-parse --show-toplevel)
export WORK=${WORK:-$HOME/vampire-benchmarks}
mkdir -p $WORK/logs
commits=$(git rev-parse ${1:-cf6946e8} ${2:-HEAD})
printf "before %s\nafter %s\n" $commits > $WORK/commits.txt
cat $WORK/commits.txt
while read version commit; do
   rm -rf $WORK/src-$version; mkdir -p $WORK/src-$version
   git -C $REPO archive $commit | tar -x -C $WORK/src-$version
done < $WORK/commits.txt

# POWER9 jobs are submitted with sbatch, Grace Hopper jobs with ghbatch
opts="--parsable --export=ALL,REPO=$REPO,WORK=$WORK"
for cmd in sbatch ghbatch; do
   if [ $cmd = sbatch ]; then short=test; long=gpu; else short=ghtest; long=gh; fi
   $cmd $opts --output=$WORK/logs/%x-$short-%j.log --partition=$short recon.job
   build=$($cmd $opts --output=$WORK/logs/%x-$short-%j.log --partition=$short build.job | cut -d';' -f1)
   $cmd $opts --output=$WORK/logs/%x-$long-%A_%a.log --dependency=afterok:$build --partition=$long --array=0-71 bench.job
   $cmd $opts --output=$WORK/logs/%x-$long-%j.log --dependency=afterok:$build --partition=$long profile.job
done
echo "Submitted. Monitor with: squeue --me    Collect with: test/bede/collect.sh"
