#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Checks hamr:active-window on a HAMR write of 9 bits along a granular FePt
# film (60 x 8 nm, 2 nm grains height, laser FWHM 8 nm, 1000 m/s):
#
#   1. identity: on a film of magnetic grains only, the window with
#      hamr:active-window-temperature = 0 (every grain integrated) gives output
#      identical to the run without the window
#   2. statistics: on the film with a non-magnetic fill and sensor layer, SEEDS
#      seeds with and without the window (threshold 1 K): z-scores of the
#      seed-mean system magnetisation at every output and of the final
#      magnetisation of every grain; FAIL if any |z| exceeds ZMAX (default 4)
#      and the statistics of one window run calculated from the spins on the
#      host agree with the device statistics (largest difference below TOL,
#      default 1e-10)
#   3. window snapshots: one window run with spin snapshots every 5 ps, written
#      as window snapshots and in full: the output is identical, and every
#      snapshot rebuilt from the window snapshots is identical to the full one
#   4. the run times with and without the window
#   5. with PERF=<seconds>: the speed on a 300 x 20 x 20 nm film (4.3 million
#      atoms), each run stopped after PERF seconds (600 is enough on a laptop GPU)
#
#    developers/test/validation/hamr_window_check.sh [vampire binary]
#
# The binary defaults to ./vampire-serial (CUDA binaries run on the GPU).
# SEEDS defaults to 8, JOBS (runs at a time) to 1. Results go to a temporary
# directory, which is removed unless KEEP=1.
#------------------------------------------------------------------------------
set -eu

exe=$(realpath "${1:-./vampire-serial}")
seeds=${SEEDS:-8}
jobs=${JOBS:-1}
zmax=${ZMAX:-4}
here=$(dirname "$(realpath "$0")")
work=$(mktemp -d)
[ "${KEEP:-0}" = 1 ] && echo "results in $work" || trap 'rm -rf "$work"' EXIT

# material file of the granular HAMR case: low density fill, thin sensor layer,
# and a version with the three FePt materials only
sed -e 's/^material\[4\]:density = 1.0/material[4]:density = 0.01/' \
    -e 's/initial-spin-direction = 0.3,0,1/initial-spin-direction = 0,0,1/' \
    -e 's/^material\[5\]:minimum-height=0.5/material[5]:minimum-height=0.85/' \
    "$here/cases/hamr-granular/FePt_sensor.mat" > "$work/full.mat"
grep -v '^material\[[45]\]' "$work/full.mat" | sed 's/^material:num-materials *= *5/material:num-materials=3/' > "$work/grains.mat"
grep -q '^material:num-materials=3' "$work/grains.mat"

run(){ # name material seed [extra input lines]
   local dir="$work/$1" mat=$2 seed=$3; shift 3
   mkdir "$dir"
   cp "$work/$mat" "$dir/FePt_sensor.mat"
   sed -e 's/system-size-x = 8 !nm/system-size-x = 60 !nm/' \
       -e 's/system-size-z = 20 !nm/system-size-z = 4 !nm/' \
       -e 's/grain-height = 10 !nm/grain-height = 2 !nm/' \
       -e 's/laser-FWHM-\([xy]\) = 5/laser-FWHM-\1 = 8/' \
       -e 's/head-speed = 5000/head-speed = 1000/' \
       -e 's/bit-size = 4 !nm/bit-size = 6 !nm/' \
       -e 's/number-of-bits = 1$/number-of-bits = 9/' \
       -e 's/equilibration-time-steps = 0/equilibration-time-steps = 20000/' \
       -e 's/output-rate = 500/output-rate = 1000/' \
       "$here/cases/hamr-granular/input" > "$dir/input"
   echo "sim:integrator-random-seed = $seed" >> "$dir/input"
   for line in "$@"; do echo "$line" >> "$dir/input"; done
   local start=$(date +%s.%N)
   (cd "$dir" && "$exe" > stdout 2>&1) || { tail -n 20 "$dir/stdout"; return 1; }
   echo "$(date +%s.%N) - $start" | bc > "$dir/seconds"
}
export -f run
export work exe here

fail=0

# 1. identity
run id-off grains.mat 1
run id-window grains.mat 1 "hamr:active-window" "hamr:active-window-temperature = 0"
for f in output grain.txt; do
   if cmp -s <(grep -v '^#' "$work/id-off/$f") <(grep -v '^#' "$work/id-window/$f"); then
      echo "PASS identity: $f identical with every grain in the window"
   else
      echo "FAIL identity: $f differs with every grain in the window"; fail=1
   fi
done

# 2. statistics
for s in $(seq 1 "$seeds"); do
   echo "run off-$s full.mat $s"
   echo "run window-$s full.mat $s hamr:active-window"
done | xargs -P "$jobs" -I{} bash -c '{}'

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
# grain file: time, temperature, then mx, my, mz, |m| of every grain
g = {m: np.array([rows(f'{work}/{m}-{s}/grain.txt')[-1] for s in range(1, seeds+1)]) for m in o}
ngrains = int(open(f'{work}/off-1/grain-coordinates.txt').read().split('Number of grains:')[1].split()[0])
gmz = {m: np.array([g[m][:, 2+4*i+2]*g[m][:, 2+4*i+3] for i in range(ngrains)]).T for m in g}
zg = z(gmz['off'], gmz['window'])
print(f'system mz at {len(zs)} outputs: max |z| {np.abs(zs).max():.2f}; final {mz["off"][:, -1].mean():+.4f} (off) {mz["window"][:, -1].mean():+.4f} (window)')
w = np.abs(zs).argmax()
print(f'   largest at t = {o["off"][0, w, 0]:.3g} s: off {mz["off"][:, w].mean():+.4f} sd {mz["off"][:, w].std(ddof=1):.4f}, window {mz["window"][:, w].mean():+.4f} sd {mz["window"][:, w].std(ddof=1):.4f}')
print(f'final mz of {ngrains} grains: max |z| {np.abs(zg).max():.2f}, sum z^2 {np.sum(zg**2):.1f}')
worst = max(np.abs(zs).max(), np.abs(zg).max())
print(('PASS' if worst <= zmax else 'FAIL') + f' statistics: largest |z| {worst:.2f} (limit {zmax}, {seeds} seeds)')
sys.exit(int(worst > zmax))
EOF

# statistics of a window run calculated from the spins on the host: the same
# trajectory, so the output must agree with the device statistics to rounding
run window-cpustats-1 full.mat 1 "hamr:active-window" "gpu:calculate-statistics-on-cpu=true"
python3 - "$work" "${TOL:-1e-10}" << 'EOF' || fail=1
import sys, numpy as np
work, tol = sys.argv[1], float(sys.argv[2])
rows = lambda f: np.array([l.split() for l in open(f) if l[:1].isdigit()], float)
d = max(np.abs(rows(f'{work}/window-1/{f}') - rows(f'{work}/window-cpustats-1/{f}')).max() for f in ('output', 'grain.txt'))
print(('PASS' if d <= tol else 'FAIL') + f' window statistics: device and host statistics differ by at most {d:.2g} (limit {tol})')
sys.exit(int(d > tol))
EOF

# 3. window snapshots
snapshots=(config:atoms "config:output-format = binary" "config:atoms-output-rate = 5000")
run snap-window-1 full.mat 1 "hamr:active-window" "${snapshots[@]}"
run snap-full-1 full.mat 1 "hamr:active-window" "hamr:active-window-full-snapshots" "${snapshots[@]}"
python3 - "$work" << 'EOF' || fail=1
import sys, os, re, numpy as np
work = sys.argv[1]
rows = lambda f: [l for l in open(f) if not l.startswith('#')]
def series(run):
    state, i = None, 0
    while os.path.exists(meta := f'{run}/spins-{i:08d}.meta'):
        text = open(meta).read()
        data = f'{run}/' + re.search(r'^(spins-\S+\.data)$', text, re.M).group(1)
        with open(data, 'rb') as f:
            if 'Spin data: window' in text:
                n, r = np.fromfile(f, np.uint64, 2)
                ranges = np.fromfile(f, np.uint64, 2 * int(r)).astype(int).reshape(-1, 2)
                s = np.fromfile(f, np.float64).reshape(-1, 3)
                pos = np.concatenate([np.arange(a, a + c) for a, c in ranges]) if len(ranges) else []
                state[pos] = s
            else:
                n = int(np.fromfile(f, np.uint64, 1)[0])
                state = np.fromfile(f, np.float64).reshape(n, 3)
        yield 'Spin data: window' in text, os.path.getsize(data), state.copy()
        i += 1
ok = all(rows(f'{work}/snap-window-1/{f}') == rows(f'{work}/snap-full-1/{f}') for f in ('output', 'grain.txt'))
w, f = list(series(f'{work}/snap-window-1')), list(series(f'{work}/snap-full-1'))
same = len(w) == len(f) and all(np.array_equal(a[2], b[2]) for a, b in zip(w, f))
nw = sum(a[0] for a in w)
print(('PASS' if ok and same and nw > 0 else 'FAIL') + f' window snapshots: {len(w)} snapshots ({nw} window), output '
      f'{"identical" if ok else "differs"}, rebuilt snapshots {"identical" if same else "differ"}; spin data '
      f'{sum(a[1] for a in w) / 1e6:.1f} MB instead of {sum(b[1] for b in f) / 1e6:.1f} MB')
sys.exit(int(not (ok and same and nw > 0)))
EOF

# 4. run times
for m in off window; do
   echo "$m: $(cat "$work"/$m-[0-9]*/seconds | awk '{t += $1} END {printf "%.1f s per run", t/NR}')"
done

# 5. speed (PERF=<seconds> per run): a 300 x 20 x 20 nm film (FePt 10 nm high,
#    sensor layer above, laser FWHM 30 nm), written for PERF seconds with and
#    without the window; wall time per bit from the log, from bit 4 on (all
#    grains are integrated for the first 10 ps)
if [ "${PERF:-0}" != 0 ]; then
   sed -e 's/^material\[4\]:density = 1.0/material[4]:density = 0.01/' \
       -e 's/initial-spin-direction = 0.3,0,1/initial-spin-direction = 0,0,1/' \
       "$here/cases/hamr-granular/FePt_sensor.mat" > "$work/perf.mat"
   for m in off window; do
      mkdir "$work/perf-$m"
      cp "$work/perf.mat" "$work/perf-$m/FePt_sensor.mat"
      sed -e 's/system-size-x = 8 !nm/system-size-x = 300 !nm/' \
          -e 's/system-size-y = 8 !nm/system-size-y = 20 !nm/' \
          -e 's/laser-FWHM-\([xy]\) = 5/laser-FWHM-\1 = 30/' \
          -e 's/head-speed = 5000/head-speed = 1000/' \
          -e 's/head-field-x = 5/head-field-x = 14/' \
          -e 's/head-field-y = 6/head-field-y = 18/' \
          -e 's/bit-size = 4 !nm/bit-size = 6 !nm/' \
          -e 's/track-size = 7 !nm/track-size = 18 !nm/' \
          -e 's/number-of-bits = 1$/number-of-bits = 49/' \
          -e 's/output-rate = 500/output-rate = 1000/' \
          "$here/cases/hamr-granular/input" > "$work/perf-$m/input"
      [ $m = window ] && echo "hamr:active-window" >> "$work/perf-$m/input"
      (cd "$work/perf-$m" && timeout "$PERF" "$exe" > stdout 2>&1) || true
   done
   python3 - "$work" << 'EOF'
import sys, re, datetime
for m in ('off', 'window'):
   log = open(f'{sys.argv[1]}/perf-{m}/log').read()
   atoms = re.search(r'Number of atoms generated: (\d+)', log)
   bits = [(int(b), datetime.datetime.strptime(t, '%d-%m-%Y %H:%M:%S')) for t, b in re.findall(r'(\S+ \S+)\] Written bit (\d+) ', log.replace('[', ''))]
   bits = [b for b in bits if b[0] >= 4]
   window = [int(a) for a in re.findall(r'HAMR active window: \d+ grains, (\d+) atoms', log)]
   rate = f'{(bits[-1][1]-bits[0][1]).total_seconds()/(bits[-1][0]-bits[0][0]):.1f} s per bit (bits {bits[0][0]}-{bits[-1][0]})' if len(bits) > 1 else 'too few bits written, increase PERF'
   extra = f', integrated atoms {min(window)}-{max(window)}' if window else ''
   print(f'speed {m}: {rate}, {atoms.group(1) if atoms else "?"} atoms{extra}')
EOF
fi

exit $fail
