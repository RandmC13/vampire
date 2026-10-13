#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Settings shared by the Viking job scripts (sourced, not run).
#
#   REPO         vampire checkout the jobs were submitted from (set by submit.sh)
#   WORK         directory for sources, binaries, runs and results
#                (default: ~/scratch/vampire-benchmarks)
#   CUDA_MODULE  CUDA module to load (default: the module system's default CUDA)
#------------------------------------------------------------------------------
module purge
module load ${CUDA_MODULE:-CUDA}
set -u   # after the module commands, which may use unset variables
: ${REPO:?REPO is not set, submit the jobs with test/viking/submit.sh}
WORK=${WORK:-$HOME/scratch/vampire-benchmarks}

# GPU architectures compiled into the benchmark binaries: A40 (sm_86) and H100 (sm_90)
GPU_ARCH="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

# benchmark versions: label -> commit (full hashes written by submit.sh)
VERSIONS=(before after)
