#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Settings shared by the Bede job scripts (sourced, not run). Bede has two
# node types, so modules, GPU architecture and binaries depend on the CPU:
#    ppc64le  POWER9 + V100   (partitions gpu, test)    sm_70
#    aarch64  Grace + GH200   (partitions gh, ghtest)   sm_90
#
#   REPO     vampire checkout the jobs were submitted from (set by submit.sh)
#   WORK     directory for sources, binaries, runs and results
#   MODULES  modules to load instead of the defaults below
#------------------------------------------------------------------------------
ARCH=$(uname -m)
case $ARCH in
   ppc64le) modules="gcc/12.2 cuda/12.4.1"; GPU_ARCH="-gencode arch=compute_70,code=sm_70" ;;
   aarch64) modules="gcc/12.2 cuda/12.6.1"; GPU_ARCH="-gencode arch=compute_90,code=sm_90" ;;
   *) echo "unexpected architecture $ARCH"; exit 1 ;;
esac
module purge
module load ${MODULES:-$modules}
set -u   # after the module commands, which may use unset variables
: ${REPO:?REPO is not set, submit the jobs with test/bede/submit.sh}
WORK=${WORK:-$HOME/vampire-benchmarks}   # under 1 GB, fits the home quota

# benchmark versions: label -> commit (full hashes written by submit.sh)
VERSIONS=(before after)
