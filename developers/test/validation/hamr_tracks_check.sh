#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Checks hamr:track-order and hamr:save-track-states on a HAMR write of three
# tracks of 3 bits at T = 0 (the film of the hamr-t0 case, widened to 26 nm),
# written in the order 2, 1, 3 with a different bit pattern on each track:
#
#   1. the log reports the tracks in the order 2, 1, 3, and every process
#      writes a checkpoint file vampire<rank>-track<n>.chk for each track
#   2. the spin snapshot after each track agrees with the one of the serial
#      CPU version (REF, default ./vampire-serial): largest difference of any
#      spin component below TOL (default 1e-8; use about 1e-3 for single
#      precision GPU builds)
#   3. the checkpoint files after track 2, renamed to vampire<rank>.chk and
#      loaded by the serial CPU version, give exactly the spins of the snapshot
#      written after track 2
#
#    developers/test/validation/hamr_tracks_check.sh [vampire binary]
#
# The binary defaults to ./vampire-serial (CUDA binaries run on the GPU). With
# RANKS=N the run uses mpirun -np N, or $MPIRUN -np N (MPI and multi-GPU
# versions). Results go to a temporary directory, which is removed unless
# KEEP=1. A few minutes on one CPU.
#------------------------------------------------------------------------------
set -eu

exe=$(realpath "${1:-./vampire-serial}")
ref=$(realpath "${REF:-./vampire-serial}")
tol=${TOL:-1e-8}
mpi=""
[ -n "${RANKS:-}" ] && mpi="${MPIRUN:-mpirun} -np $RANKS"
here=$(dirname "$(realpath "$0")")
case_dir=$here/cases/hamr-t0
work=$(mktemp -d)
[ "${KEEP:-0}" = 1 ] && echo "results in $work" || trap 'rm -rf "$work"' EXIT
fail=0

# writes the input of the three-track write into directory $1
write_input(){
   mkdir -p "$1"
   cp "$case_dir/hamr.mat" "$case_dir/input" "$1/"
   printf '%s\n' "dimensions:system-size-y = 26.0 !nm" "hamr:number-of-bits = 9" \
      "hamr:bit-sequence-type = user-defined" "hamr:bit-sequence = 1,-1,1,-1,-1,1,1,1,-1" \
      "hamr:track-order = 2,1,3" "hamr:save-track-states" >> "$1/input"
}

write_input "$work/ref"
write_input "$work/run"
case $(basename "$exe") in *multigpu*) echo "gpu:decomposition-independent-order" >> "$work/run/input";; esac
(cd "$work/ref" && "$ref" > stdout 2>&1) || { echo "FAIL: reference run (see $work/ref/stdout)"; exit 1; }
(cd "$work/run" && $mpi "$exe" > stdout 2>&1) || { echo "FAIL: run (see $work/run/stdout)"; exit 1; }

# 1. order and files
order=$(grep -h "Tracks written in the order" "$work/run/log" | sed 's/.*order://' | tr -d ' ')
[ "$order" = 213 ] && echo "PASS 1: tracks written in the order 2, 1, 3" || { echo "FAIL 1: track order '$order'"; fail=1; }
ranks=$(ls "$work/run"/vampire*-track2.chk | wc -l)
for n in 1 2 3; do
   [ "$(ls "$work/run"/vampire*-track$n.chk | wc -l)" = "$ranks" ] || { echo "FAIL 1: checkpoint files of track $n missing"; fail=1; }
done
echo "      $ranks process(es), checkpoint files for tracks 1, 2 and 3"

# largest difference of any spin component between snapshot $2 of run directory $1 and snapshot $4 of run
# directory $3, with the atoms matched by position (parallel runs write the atoms in a different order)
spin_difference(){
   python3 - "$@" <<'EOF'
import sys
import numpy as np
def spins(run, snapshot):
   coords = np.loadtxt(run + "/atoms-coords.data", skiprows=1)
   s = np.loadtxt(run + "/spins-%08d.data" % int(snapshot), skiprows=1)
   return s[np.lexsort((coords[:, 4], coords[:, 3], coords[:, 2]))]
a, b = spins(sys.argv[1], sys.argv[2]), spins(sys.argv[3], sys.argv[4])
print(np.abs(a - b).max() if a.shape == b.shape else "shape")
EOF
}

# 2. snapshots after each track (written in the order 2, 1, 3: files 0, 1, 2) against the serial CPU version
for i in 0 1 2; do
   diff=$(spin_difference "$work/ref" $i "$work/run" $i)
   if [ "$diff" != shape ] && python3 -c "import sys; sys.exit(0 if float('$diff') <= float('$tol') else 1)"; then
      echo "PASS 2: snapshot $i (after track $(echo 213 | cut -c$((i+1)))) agrees with the serial CPU version, largest difference $diff"
   else
      echo "FAIL 2: snapshot $i differs from the serial CPU version ($diff)"; fail=1
   fi
done

# 3. the checkpoint after track 2 loads (serial CPU version) with exactly the spins of the snapshot after track 2
load=$work/load
mkdir -p "$load"
cp "$case_dir/hamr.mat" "$load/"
grep -v "^sim:program\|^sim:equilibration-time-steps\|^hamr:" "$work/run/input" > "$load/input"
printf '%s\n' "sim:program = time-series" "sim:temperature = 0" "sim:applied-field-strength = 0" \
   "sim:equilibration-time-steps = 0" "sim:total-time-steps = 1" "sim:load-checkpoint = restart" \
   "config:atoms" "config:atoms-output-rate = 1" >> "$load/input"
for f in "$work/run"/vampire*-track2.chk; do cp "$f" "$load/$(basename "$f" | sed 's/-track2//')"; done
(cd "$load" && "$ref" > stdout 2>&1) || { echo "FAIL 3: loading the checkpoint (see $load/stdout)"; exit 1; }
if [ "$(spin_difference "$load" 0 "$work/run" 0)" = 0.0 ]; then
   echo "PASS 3: checkpoint after track 2 loads with the spins of the snapshot after track 2"
else
   echo "FAIL 3: spins loaded from the checkpoint after track 2 differ from the snapshot"; fail=1
fi

exit $fail
