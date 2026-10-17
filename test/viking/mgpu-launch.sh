#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# MPI launcher for the multi-GPU version with one GPU per rank.
#
#    mgpu-launch.sh -np N program [arguments]
#
# starts N ranks with mpirun (extra mpirun options from MPI_OPTS) and runs
# each rank through this script again, which gives the rank one GPU of its
# node (by node-local rank, as the only visible device) and binds the rank to
# the CPU cores and memory of that GPU's NUMA node if numactl can. Each rank
# prints its binding to stderr. Used as MPIRUN for developers/test/validation/vtest.py.
#------------------------------------------------------------------------------
if [ "${1:-}" = "-np" ]; then
   exec mpirun ${MPI_OPTS:-} --bind-to none -np "$2" "$0" "${@:3}"
fi

rank=${OMPI_COMM_WORLD_RANK:-0}
local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:-0}

# GPUs of this job on this node (all GPUs nvidia-smi lists, restricted to
# CUDA_VISIBLE_DEVICES when the scheduler sets it by index or UUID)
mapfile -t all < <(nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader | tr -d ' ')
gpus=()
for g in "${all[@]}"; do
   IFS=, read -r index uuid bus <<< "$g"
   if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || [[ ",$CUDA_VISIBLE_DEVICES," == *",$index,"* ]] || [[ ",$CUDA_VISIBLE_DEVICES," == *",$uuid,"* ]]; then
      gpus+=("$uuid,$bus")
   fi
done
if [ ${#gpus[@]} = 0 ]; then
   echo "rank $rank on $(hostname -s): no GPU found" >&2
   exec "$@"
fi
IFS=, read -r uuid bus <<< "${gpus[local_rank % ${#gpus[@]}]}"
export CUDA_VISIBLE_DEVICES=$uuid

# NUMA node of the GPU (sysfs uses a 4 digit PCI domain, nvidia-smi 8)
bus=$(echo ${bus: -12} | tr A-F a-f)
numa=$(cat /sys/bus/pci/devices/$bus/numa_node 2> /dev/null || echo -1)
bind=()
if [ "$numa" -ge 0 ] && command -v numactl > /dev/null && numactl --cpunodebind=$numa --membind=$numa true 2> /dev/null; then
   bind=(numactl --cpunodebind=$numa --membind=$numa)
fi
echo "rank $rank on $(hostname -s): GPU $bus (NUMA node $numa), bound: ${bind[*]:-no}" >&2
exec "${bind[@]}" "$@"
