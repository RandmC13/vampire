#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Submits the GPU benchmark set on Viking:
#
#   recon     software/hardware survey of an A40 and an H100 node
#   build     CUDA binaries (dp, sp) of the two benchmarked commits
#   bench     before/after timings on A40 (gpu) and H100 (gpuplus), 72 runs each
#   profile   Nsight profiles of hamr-perf on A40 and H100 (after version)
#
# Usage (from the vampire checkout on Viking):
#   test/viking/submit.sh [before-commit] [after-commit]
# Defaults: before = cf6946e8 (validated GPU code before the performance work),
#           after  = HEAD.
#
# Environment: WORK (default ~/scratch/vampire-benchmarks), CUDA_MODULE.
# The job files contain the placeholder VIKING_ACCOUNT for the Slurm project
# account, which must be replaced with the project code first.
#------------------------------------------------------------------------------
set -eu
cd "$(dirname "$0")"
if grep -l VIKING_ACCOUNT *.job; then
   echo "Replace the VIKING_ACCOUNT placeholder in the files above with your project account first."
   exit 1
fi

export REPO=$(git rev-parse --show-toplevel)
export WORK=${WORK:-$HOME/scratch/vampire-benchmarks}
mkdir -p $WORK/logs
commits=$(git rev-parse ${1:-cf6946e8} ${2:-HEAD})
printf "before %s\nafter %s\n" $commits > $WORK/commits.txt
cat $WORK/commits.txt

sub(){ sbatch --parsable --output=$WORK/logs/%x-%j.log "$@"; }

sub --partition=gpu_short recon.job
sub --partition=gpuplus   recon.job
build=$(sub build.job)
for partition in gpu gpuplus; do
   sbatch --parsable --output=$WORK/logs/%x-$partition-%A_%a.log \
          --dependency=afterok:$build --partition=$partition --array=0-71 bench.job
   sub --dependency=afterok:$build --partition=$partition profile.job
done
echo "Submitted. Monitor with: squeue --me    Collect with: test/viking/collect.sh"
