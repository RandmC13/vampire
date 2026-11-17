#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# A short demonstration of the capped HAMR write on one local GPU, from the
# inputs of the Viking run (test/viking/capped/write): the same film, cap, grains,
# bits (14 x 32 nm), laser, head field and active window, but two tracks of
# 6 bits on an 85 x 98 nm film and a head at 90 m/s (field rise and fall
# 30 ps), with a 20 ps equilibration and a spin snapshot every 50 ps. Track 1
# (one bit per transition) is written first, then its neighbour, track 2 (two
# bits per transition). About 2.2 hours on an RTX 3070 in single precision.
#
#   test/local/capped-demo.sh <vampire-cuda binary> [run directory]
#
# Build the binary with: make cuda CUDA_PR=-DCUDA_SP CUDA_ARCH=-arch=native
# The run directory (default capped-demo) must not exist yet.
#------------------------------------------------------------------------------
set -eu
exe=$(realpath "${1:?usage: capped-demo.sh <vampire-cuda binary> [run directory]}")
dir=${2:-capped-demo}
[ -e "$dir" ] && { echo "Error: $dir exists"; exit 1; }
mkdir -p "$dir"
cp "$(dirname "$(realpath "$0")")"/../viking/capped/write/* "$dir"/
printf '%s\n' "" "# demonstration: two tracks of 6 bits on an 85 x 98 nm film, head at 90 m/s" \
   "dimensions:system-size-x = 85 !nm" "dimensions:system-size-y = 98 !nm" \
   "hamr:head-speed = 90 !m/s" "hamr:field-rise-time = 30e-12 !s" "hamr:field-fall-time = 30e-12 !s" \
   "hamr:number-of-bits = 12" "hamr:bit-sequence = 1,-1,1,-1,1,-1,1,1,-1,-1,1,1" "hamr:track-order = 1,2" \
   "sim:equilibration-time-steps = 20000" "config:atoms-output-rate = 50000" "gpu:skip-non-magnetic-atoms" >> "$dir/input"
cd "$dir"
echo "running in $PWD (progress: tail -f log)"
start=$(date +%s)
"$exe" > stdout 2>&1 || { tail -n 30 stdout; exit 1; }
echo "done in $(( ($(date +%s) - start) / 60 )) min"
