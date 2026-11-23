#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Read-back of the local capped demonstration (capped-demo.sh): for the state
# after each track, the dipole field in the sensor layer (program dipole-field,
# tensor solver with periodic images, probe cells only), from the inputs of the
# Viking read (test/viking/capped/read) on the demonstration's 85 x 98 nm film.
# Results in <run directory>/read-track<n>/dipole-field.txt.
#
#   test/local/capped-demo-read.sh <vampire-parallel or vampire-serial> [run directory]
#
# Build with make parallel (or make serial). With the parallel version the read
# runs on NP processes (default: the number of physical cores, which is what
# mpirun allows without oversubscribing) with mpirun.
#------------------------------------------------------------------------------
set -eu
exe=$(realpath "${1:?usage: capped-demo-read.sh <vampire binary> [run directory]}")
run=$(realpath "${2:-capped-demo}")
inputs=$(dirname "$(realpath "$0")")/../viking/capped/read
launch=""
cores=$(lscpu -p=Core,Socket 2> /dev/null | grep -v "^#" | sort -u | wc -l)
case $(basename "$exe") in *parallel*) launch="mpirun -np ${NP:-${cores:-1}}";; esac
found=0
for track in 1 2 3; do
   ls "$run"/vampire*-track$track.chk > /dev/null 2>&1 || continue
   found=1
   dir=$run/read-track$track
   mkdir -p "$dir"
   cp "$inputs"/* "$dir"/
   printf '%s\n' "" "# demonstration film" "dimensions:system-size-x = 85 !nm" "dimensions:system-size-y = 98 !nm" >> "$dir/input"
   for f in "$run"/vampire*-track$track.chk; do cp "$f" "$dir/$(basename "$f" | sed "s/-track$track//")"; done
   cd "$dir"
   start=$(date +%s)
   $launch "$exe" > stdout 2>&1 || { tail -n 30 stdout; exit 1; }
   rm -f vampire*.chk   # copies of the write's checkpoint files
   echo "read after track $track: $(( $(date +%s) - start )) s, field in $dir/dipole-field.txt"
done
[ $found = 1 ] || { echo "Error: no track checkpoints (vampire*-track<n>.chk) in $run"; exit 1; }
