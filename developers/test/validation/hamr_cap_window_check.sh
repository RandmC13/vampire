#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Checks hamr:active-window on a granular FePt film with a continuous cap
# (2 nm grains of the hamr-granular case under a 0.5 nm Co3Pt-like layer,
# exchange coupled to the grains), one track along a 60 x 8 nm film, head at 3000 m/s:
#
#   1. identity: with hamr:active-window-temperature = 0 every grain, and so
#      the whole cap, is integrated: output identical to the run without the
#      window (on a 12 x 40 nm film, which up to 6 processes divide across the
#      track with and without the window, so that the MPI and multi-GPU
#      versions compare the same division)
#   2. damping buffer: with the buffer damping equal to the cap's own (0.1),
#      a 5 nm margin and a 15 nm buffer give output identical to a 20 nm
#      margin without a buffer (the per-atom damping and noise are those of
#      the material)
#   3. buffer temperature: at a constant 300 K (no laser heating, no field) the
#      window with a buffer over most of the cap (1 nm margin, 30 nm buffer,
#      buffer damping 3) keeps the cap at the magnetisation of the run without
#      the window: the time averages of the cap's magnetisation length differ
#      by less than DM (default 0.01); a buffer whose thermal noise did not
#      follow its damping is colder and differs by about 0.02
#   4. with SEEDS=N (at least 8): the write on a 100 x 8 nm film with and
#      without the window: z-scores of the seed-mean system magnetisation at
#      every output, of the final magnetisation of every grain and of the
#      cap's magnetisation; FAIL if any |z| exceeds ZMAX (default 4)
#
#    developers/test/validation/hamr_cap_window_check.sh [vampire binary]
#
# The binary defaults to ./vampire-serial (CUDA binaries run on the GPU). With
# RANKS=N every run uses mpirun -np N, or $MPIRUN -np N (MPI and multi-GPU
# versions); JOBS runs at a time (default 6). The multi-GPU version
# runs with gpu:decomposition-independent-order. Results go to a temporary
# directory, which is removed unless KEEP=1, or to the (new, empty) directory
# WORK, which is kept. Stages 1-3 take minutes on a GPU.
#------------------------------------------------------------------------------
set -eu

exe=$(realpath "${1:-./vampire-serial}")
seeds=${SEEDS:-0}
jobs=${JOBS:-6}
zmax=${ZMAX:-4}
dm=${DM:-0.01}
mpi=""
[ -n "${RANKS:-}" ] && mpi="${MPIRUN:-mpirun} -np $RANKS"
mgpu=0
case $(basename "$exe") in *multigpu*) mgpu=1;; esac
here=$(dirname "$(realpath "$0")")
if [ -n "${WORK:-}" ]; then
   work=$(realpath -m "$WORK"); mkdir -p "$work"; KEEP=1
else
   work=$(mktemp -d)
fi
[ "${KEEP:-0}" = 1 ] && echo "results in $work" || trap 'rm -rf "$work"' EXIT

# material file: the three FePt materials of the granular HAMR case (heights 0-0.8) and three cap materials
# (heights 0.8-1, continuous): Tc about 1100 K, moment 2.07 muB, anisotropy for Hk = 1.2 T at 300 K, damping
# 0.1, coupled to the grains with 20 % of the FePt exchange (the lattice bonds category 3 to categories 1 and 2)
{
   grep -v '^material\[[45]\]' "$here/cases/hamr-granular/FePt_sensor.mat" | sed \
      -e 's/^material:num-materials *= *5/material:num-materials=6/' \
      -e 's/maximum-height=0.5/maximum-height=0.8/' \
      -e 's/initial-spin-direction = 0.3,0,1/initial-spin-direction = 0,0,1/'
   jint=1.12e-21; jcap=1.05e-20
   printf '%s\n' "material[1]:exchange-matrix-1st-nn[6] = $jint" "material[2]:exchange-matrix-1st-nn[6] = $jint" \
      "material[3]:exchange-matrix-1st-nn[4] = $jint" "material[3]:exchange-matrix-1st-nn[5] = $jint"
   for m in 4 5 6; do
      printf '%s\n' "material[$m]:material-name=cap$((m-3))" "material[$m]:damping-constant=0.1" \
         "material[$m]:atomic-spin-moment=2.07 !muB" "material[$m]:uniaxial-anisotropy-constant=1.43e-23" \
         "material[$m]:material-element=Co" "material[$m]:minimum-height=0.8" "material[$m]:maximum-height=1.0" \
         "material[$m]:unit-cell-category = $((m-3))" "material[$m]:initial-spin-direction = 0,0,1" \
         "material[$m]:continuous = true"
   done
   printf '%s\n' "material[4]:exchange-matrix-1st-nn[3] = $jint" "material[4]:exchange-matrix-1st-nn[6] = $jcap" \
      "material[5]:exchange-matrix-1st-nn[3] = $jint" "material[5]:exchange-matrix-1st-nn[6] = $jcap" \
      "material[6]:exchange-matrix-1st-nn[1] = $jint" "material[6]:exchange-matrix-1st-nn[2] = $jint" \
      "material[6]:exchange-matrix-1st-nn[4] = $jcap" "material[6]:exchange-matrix-1st-nn[5] = $jcap"
} > "$work/capped.mat"

run(){ # name length (nm) seed [extra input lines]; width (nm, default 8) from the environment
   local dir="$work/$1" length=$2 seed=$3
   shift 3
   mkdir "$dir"
   cp "$work/capped.mat" "$dir/FePt_sensor.mat"
   sed -e "s/system-size-x = 8 !nm/system-size-x = $length !nm/" \
       -e "s/system-size-y = 8 !nm/system-size-y = ${width:-8} !nm/" \
       -e 's/system-size-z = 20 !nm/system-size-z = 2.5 !nm/' \
       -e 's/grain-height = 10 !nm/grain-height = 2 !nm/' \
       -e 's/laser-FWHM-\([xy]\) = 5/laser-FWHM-\1 = 8/' \
       -e 's/head-speed = 5000/head-speed = 3000/' \
       -e 's/bit-size = 4 !nm/bit-size = 6 !nm/' \
       -e "s/number-of-bits = 1\$/number-of-bits = $((length/6))/" \
       -e 's/equilibration-time-steps = 0/equilibration-time-steps = 5000/' \
       -e 's/output-rate = 500/output-rate = 1000/' \
       "$here/cases/hamr-granular/input" > "$dir/input"
   echo "sim:integrator-random-seed = $seed" >> "$dir/input"
   [ "$mgpu" = 1 ] && echo "gpu:decomposition-independent-order" >> "$dir/input"
   for line in "$@"; do echo "$line" >> "$dir/input"; done
   (cd "$dir" && $mpi "$exe" > stdout 2>&1) || { tail -n 20 "$dir/stdout"; return 1; }
}
export -f run
export work exe here mpi mgpu

same(){ # run directory a, run directory b, label
   local ok=1 f
   for f in output grain.txt; do cmp -s <(grep -v '^#' "$work/$1/$f") <(grep -v '^#' "$work/$2/$f") || ok=0; done
   [ $ok = 1 ] && echo "PASS $3: output and grain.txt identical" || { echo "FAIL $3: output or grain.txt differs"; fail=1; }
}
fail=0

window_used(){ # run directory, label: the window held some but not all atoms after it started
   python3 - "$work/$1/log" "$2" << 'EOF' || fail=1
import sys, re
log, label = sys.argv[1], sys.argv[2]
text = open(log).read()
total = int(re.search(r'(?:Number of atoms generated|Total number of atoms \(all CPUs\)): (\d+)', text).group(1))
n = [int(m) for m in re.findall(r'HAMR active window: \d+ grains, (\d+) atoms integrated', text)]
used = [k for k in n[1:] if 0 < k < total]
print(('PASS' if used else 'FAIL') + f' {label}: the window held {min(used) if used else 0} to {max(used) if used else 0} of {total} atoms ({len(n)} changes)')
sys.exit(int(not used))
EOF
}

# the runs of stages 1-3, JOBS at a time
cold='"sim:maximum-temperature = 300.0" "sim:maximum-applied-field-strength = 0.0 !T"'
printf '%s\n' \
   'width=40 run id-off 12 1' \
   'width=40 run id-window 12 1 hamr:active-window "hamr:active-window-temperature = 0"' \
   'run buffer-own 60 1 hamr:active-window "hamr:active-window-buffer-damping = 0.1"' \
   'run margin-only 60 1 hamr:active-window "hamr:active-window-margin = 20 !nm" "hamr:active-window-buffer = 0 !nm"' \
   "run temp-off 60 1 $cold" \
   "run temp-window 60 1 $cold hamr:active-window \"hamr:active-window-margin = 1 !nm\" \"hamr:active-window-buffer = 30 !nm\" \"hamr:active-window-buffer-damping = 3.0\"" | xargs -d '\n' -P "$jobs" -I{} bash -c '{}'

# 1. identity
same id-off id-window "identity (every grain and the whole cap integrated)"
grep -h 'decomposed into' "$work/id-off/log" "$work/id-window/log" | sed 's/.*decomposed into //' | uniq | sed -n 2p | grep -q . && { echo "FAIL identity: the runs with and without the window divide the film differently"; fail=1; }

# 2. damping buffer with the cap's own damping
same buffer-own margin-only "damping buffer (buffer damping equal to the cap's)"
grep -q "continuous layers integrated within" "$work/buffer-own/log" || { echo "FAIL damping buffer: the window found no continuous layer"; fail=1; }
window_used buffer-own "damping buffer"

# 3. buffer temperature at a constant 300 K
window_used temp-window "buffer temperature"
python3 - "$work" "$dm" << 'EOF' || fail=1
import sys, numpy as np
work, dm = sys.argv[1], float(sys.argv[2])
def cap_m(run):
    lines = open(f'{work}/{run}/output').read().splitlines()
    rows = np.array([l.split() for l in lines if l[:1].isdigit()], float)
    # columns: time, temperature, field, system mx, my, mz, |m|, then mx, my, mz, |m| of materials 1-6
    m = rows[:, 7 + 4*3 + 3:7 + 4*6:4]
    return m[len(m)//2:].mean()
off, win = cap_m('temp-off'), cap_m('temp-window')
d = abs(off - win)
print(('PASS' if d < dm else 'FAIL') + f' buffer temperature: cap magnetisation length {off:.4f} without and {win:.4f} with the window (difference {d:.4f}, limit {dm})')
sys.exit(int(d >= dm))
EOF

# 4. statistics
if [ "$seeds" -ge 1 ]; then
   for s in $(seq 1 "$seeds"); do
      echo "run off-$s 100 $s"
      echo "run window-$s 100 $s hamr:active-window"
   done | xargs -d '\n' -P "$jobs" -I{} bash -c '{}'
   python3 - "$work" "$seeds" "$zmax" << 'EOF' || fail=1
import sys, numpy as np
work, seeds, zmax = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
rows = lambda f: np.array([l.split() for l in open(f) if l[:1].isdigit()], float)
def z(a, b):
    se = np.sqrt(a.var(0, ddof=1)/len(a) + b.var(0, ddof=1)/len(b))
    return np.where(se > 0, (b.mean(0) - a.mean(0))/np.where(se > 0, se, 1), 0)
o = {m: np.array([rows(f'{work}/{m}-{s}/output') for s in range(1, seeds+1)]) for m in ('off', 'window')}
mz = {m: o[m][:, :, 5]*o[m][:, :, 6] for m in o}
zs = z(mz['off'], mz['window'])
cap = {m: o[m][:, :, 7 + 4*3 + 2:7 + 4*6:4].mean(2) for m in o}
zc = z(cap['off'], cap['window'])
g = {m: np.array([rows(f'{work}/{m}-{s}/grain.txt')[-1] for s in range(1, seeds+1)]) for m in o}
ngrains = int(open(f'{work}/off-1/grain-coordinates.txt').read().split('Number of grains:')[1].split()[0])
gmz = {m: np.array([g[m][:, 2+4*i+2]*g[m][:, 2+4*i+3] for i in range(ngrains)]).T for m in g}
zg = z(gmz['off'], gmz['window'])
print(f'system mz at {len(zs)} outputs: max |z| {np.abs(zs).max():.2f}; final {mz["off"][:, -1].mean():+.4f} (off) {mz["window"][:, -1].mean():+.4f} (window)')
print(f'cap mz at {len(zc)} outputs: max |z| {np.abs(zc).max():.2f}; final {cap["off"][:, -1].mean():+.4f} (off) {cap["window"][:, -1].mean():+.4f} (window)')
print(f'final mz of {ngrains} grains: max |z| {np.abs(zg).max():.2f}, sum z^2 {np.sum(zg**2):.1f}')
worst = max(np.abs(zs).max(), np.abs(zc).max(), np.abs(zg).max())
if seeds < 8:
    print(f'NOTE statistics: largest |z| {worst:.2f} with {seeds} seeds, not judged (at least 8 needed)')
    sys.exit(0)
print(('PASS' if worst <= zmax else 'FAIL') + f' statistics: largest |z| {worst:.2f} (limit {zmax}, {seeds} seeds)')
sys.exit(int(worst > zmax))
EOF
fi

exit $fail
