#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Settings shared by the multi-GPU job scripts (sourced, not run).
#
#   REPO     vampire checkout the jobs were submitted from (set by mgpu-submit.sh)
#   WORK     directory for sources, binaries, runs and results
#            (default: ~/scratch/vampire-benchmarks); multi-GPU files go to $WORK/mgpu
#   MODULES  modules to load instead of the defaults below: CUDA, an MPI built
#            with a GCC the CUDA compiler supports, and python3 with numpy
#            built with the same GCC
#------------------------------------------------------------------------------
module purge
module load ${MODULES:-CUDA/12.9.0 OpenMPI/4.1.5-GCC-12.3.0 SciPy-bundle/2023.07-gfbf-2023a}
set -u   # after the module commands, which may use unset variables
: ${REPO:?REPO is not set, submit the jobs with test/viking/mgpu-submit.sh}
WORK=${WORK:-$HOME/scratch/vampire-benchmarks}
MGPU=$WORK/mgpu
export MPIRUN=$REPO/test/viking/mgpu-launch.sh
partition=$SLURM_JOB_PARTITION
nodes=$SLURM_JOB_NUM_NODES
per_node=$SLURM_NTASKS_PER_NODE   # ranks (= GPUs) per node
total=$SLURM_NTASKS

{ module list; nvcc --version | tail -n 2; mpirun --version | head -n 1; mpicxx --version | head -n 1
  python3 -c "import numpy; print('numpy', numpy.__version__)"; echo "nodes: $SLURM_JOB_NODELIST"; nvidia-smi -L; } 2>&1

# private copy of the exported sources (builds clean in place)
src=$MGPU/build-$SLURM_JOB_ID
rm -rf $src; cp -r $MGPU/src $src
