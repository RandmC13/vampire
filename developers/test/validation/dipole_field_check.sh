#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Checks sim:program = dipole-field, dipole:field-cells = probe and the periodic
# images of the tensor solver on a simple cubic Co film (2.5 A lattice, 2 nm
# thick, 1 nm cells) with a non-magnetic probe layer 2.5 nm above it
# (cells:probe-non-magnetic-cells):
#
#   1. on a film that is not periodic, the program gives the fields of the
#      time-series program with sim:total-time-steps = 0 (dipole-field.txt
#      identical)
#   2. a uniformly magnetised film periodic in x and y: every cell of a layer
#      has the same field (largest spread below TOL, default 1e-6 T), and the
#      field in the probe layer is below TOL; the film without periodic
#      boundaries has an edge field there, for comparison
#   3. perpendicular stripes of 4 nm (written into a checkpoint): the field in
#      the probe layer agrees with the analytic field of the periodic stripes
#      (largest difference below STRIPES_TOL, default 1 % of the largest
#      field)
#   4. dipole:field-cells = probe gives the probe cells the fields of all cells
#      (identical) and the other cells zero
#   5. with an MPI binary and RANKS (default 2): the stripe fields of RANKS
#      processes agree with one process to rounding (below 1e-12 T)
#
#    developers/test/validation/dipole_field_check.sh [vampire binary]
#
# The binary defaults to ./vampire-serial; stage 5 runs when the binary name
# contains "parallel". Results go to a temporary directory, which is removed
# unless KEEP=1. Seconds on one CPU.
#------------------------------------------------------------------------------
set -eu

exe=$(realpath "${1:-./vampire-serial}")
tol=${TOL:-1e-6}
stripes_tol=${STRIPES_TOL:-0.01}
ranks=${RANKS:-2}
mpi_exe=0
case $(basename "$exe") in *parallel*) mpi_exe=1;; esac
work=$(mktemp -d)
[ "${KEEP:-0}" = 1 ] && echo "results in $work" || trap 'rm -rf "$work"' EXIT
fail=0

cat > "$work/film.mat" << 'EOF'
material:num-materials=2
material[1]:material-name=Co
material[1]:damping-constant=1.0
material[1]:exchange-matrix[1]=11.2e-21
material[1]:atomic-spin-moment=1.72 !muB
material[1]:material-element=Co
material[1]:initial-spin-direction=0,0,1
material[1]:minimum-height=0.0
material[1]:maximum-height=0.4
material[2]:material-name=probe
material[2]:non-magnetic=keep
material[2]:atomic-spin-moment=1.72 !muB
material[2]:material-element=Ag
material[2]:minimum-height=0.8
material[2]:maximum-height=1.0
EOF

run(){ # name program periodic(0/1) ranks [extra input lines]
   local dir="$work/$1" program=$2 periodic=$3 np=$4
   shift 4
   mkdir -p "$dir"
   cp "$work/film.mat" "$dir/"
   cat > "$dir/input" << EOF
dimensions:unit-cell-size = 2.5 !A
dimensions:system-size-x = 8 !nm
dimensions:system-size-y = 4 !nm
dimensions:system-size-z = 5 !nm
material:file = film.mat
cells:macro-cell-size = 1 !nm
cells:probe-non-magnetic-cells
dipole:solver = tensor
dipole:output-dipole-field
sim:temperature = 0.0
sim:time-step = 1 !fs
sim:total-time-steps = 0
sim:program = $program
sim:integrator = llg-heun
output:real-time
EOF
   [ "$periodic" = 1 ] && echo "create:periodic-boundaries = xy" >> "$dir/input"
   for line in "$@"; do echo "$line" >> "$dir/input"; done
   if [ "$np" -gt 1 ]; then
      (cd "$dir" && ${MPIRUN:-mpirun} -np "$np" "$exe" > stdout 2>&1) || { tail -n 20 "$dir/stdout"; exit 1; }
   else
      (cd "$dir" && "$exe" > stdout 2>&1) || { tail -n 20 "$dir/stdout"; exit 1; }
   fi
}

# 1. program against time-series
run ts-open time-series 0 1
run df-open dipole-field 0 1
if cmp -s "$work/ts-open/dipole-field.txt" "$work/df-open/dipole-field.txt"; then
   echo "PASS 1: dipole-field gives the fields of time-series (no periodic boundaries)"
else
   echo "FAIL 1: dipole-field and time-series give different fields"; fail=1
fi

# 2. uniform film, periodic and not
run uniform dipole-field 1 1
python3 - "$work" "$tol" << 'EOF' || fail=1
import sys
import numpy as np
work, tol = sys.argv[1], float(sys.argv[2])
p = np.loadtxt(work + "/uniform/dipole-field.txt")
o = np.loadtxt(work + "/df-open/dipole-field.txt")
ok = True
spread = 0.0
for z in np.unique(np.round(p[p[:, 1] > 0][:, 4], 6)):
   layer = p[(p[:, 1] > 0) & (np.abs(p[:, 4] - z) < 1e-6)]
   spread = max(spread, np.abs(layer[:, 6:9] - layer[:, 6:9].mean(0)).max())
probe = lambda d: d[d[:, 4] > 35.0]
bp, bo = np.abs(probe(p)[:, 6:9]).max(), np.abs(probe(o)[:, 6:9]).max()
ok = spread < tol and bp < tol
print(f"{'PASS' if ok else 'FAIL'} 2: uniform periodic film: spread of the field within a layer {spread:.2e} T, "
      f"largest field in the probe layer {bp:.2e} T (tolerance {tol:g} T); without periodic boundaries {bo:.3e} T")
sys.exit(0 if ok else 1)
EOF

# 3. stripes: a checkpoint of the film with mz = +1 for x < 4 nm and -1 above, loaded by dipole-field
run prepare time-series 1 1 "sim:save-checkpoint = end" "config:atoms" "config:output-format = text" "config:atoms-output-rate = 1" \
   "sim:total-time-steps = 1"
python3 - "$work/prepare" << 'EOF'
import sys, struct
import numpy as np
run = sys.argv[1]
coords = np.loadtxt(run + "/atoms-coords.data", skiprows=1)
data = bytearray(open(run + "/vampire0.chk", "rb").read())
n = struct.unpack("<Q", data[:8])[0]
assert n == len(coords)
offset = 8*5 + 8*3 + 2 + 8*3 + 4 + 624*4   # header, then the x, y and z spin arrays
mz = np.where(coords[:, 2] < 40.0, 1.0, -1.0)
zero = np.zeros(n)
for k, a in enumerate((zero, zero, mz)):
   data[offset + 8*n*k:offset + 8*n*(k+1)] = a.astype("<f8").tobytes()
open(run + "/vampire0.chk", "wb").write(bytes(data))
EOF
mkdir -p "$work/stripes"; cp "$work/prepare/vampire0.chk" "$work/stripes/"
run stripes dipole-field 1 1 "sim:load-checkpoint = restart"
python3 - "$work" "$stripes_tol" << 'EOF' || fail=1
import sys
import numpy as np
work, rtol = sys.argv[1], float(sys.argv[2])
d = np.loadtxt(work + "/stripes/dipole-field.txt")
probe = d[d[:, 4] > 35.0]
# analytic: atoms of 2.5 A as cells of 2.5 A, so the slab spans z = -1.25 to 18.75 A and the stripe +M spans
# x = -1.25 to 38.75 A; field of the periodic square wave of period 80 A above the slab
mu0, muB, a = 4e-7*np.pi, 9.274009994e-24, 2.5e-10
M = 1.72*muB/a**3
lam, x0, zb, zt = 80.0, -1.25, -1.25, 18.75
x, z = probe[:, 2], probe[:, 4]
bz = np.zeros_like(x)
for n in range(1, 400, 2):
   k = 2*np.pi*n/lam
   bz += (4/(np.pi*n)) * (mu0*M/2) * (np.exp(-k*(z - zt)) - np.exp(-k*(z - zb))) * np.sin(k*(x - x0))
diff = np.abs(probe[:, 8] - bz).max()
scale = np.abs(bz).max()
ok = diff <= rtol*scale
print(f"{'PASS' if ok else 'FAIL'} 3: stripes: largest difference from the analytic field {diff:.3e} T, "
      f"{100*diff/scale:.2f} % of the largest field {scale:.4f} T (tolerance {100*rtol:g} %)")
sys.exit(0 if ok else 1)
EOF

# 4. probe cells only
mkdir -p "$work/stripes-probe"; cp "$work/prepare/vampire0.chk" "$work/stripes-probe/"
run stripes-probe dipole-field 1 1 "sim:load-checkpoint = restart" "dipole:field-cells = probe"
python3 - "$work" << 'EOF' || fail=1
import sys
import numpy as np
work = sys.argv[1]
a = np.loadtxt(work + "/stripes/dipole-field.txt")
p = np.loadtxt(work + "/stripes-probe/dipole-field.txt")
probe = a[:, 4] > 35.0
same = np.array_equal(a[probe, 6:9], p[probe, 6:9])
others = np.abs(p[~probe, 6:9]).max()
ok = same and others == 0.0
print(f"{'PASS' if ok else 'FAIL'} 4: field-cells = probe: probe cells {'identical' if same else 'different'}, largest field of the other cells {others:g} T")
sys.exit(0 if ok else 1)
EOF

# 5. processes
if [ "$mpi_exe" = 1 ]; then
   mkdir -p "$work/stripes-mpi"; cp "$work/prepare/vampire0.chk" "$work/stripes-mpi/"
   run stripes-mpi dipole-field 1 "$ranks" "sim:load-checkpoint = restart"
   python3 - "$work" "$ranks" << 'EOF' || fail=1
import sys
import numpy as np
work, ranks = sys.argv[1], sys.argv[2]
a = np.loadtxt(work + "/stripes/dipole-field.txt")
b = np.loadtxt(work + "/stripes-mpi/dipole-field.txt")
diff = np.abs(a[:, 6:9] - b[:, 6:9]).max()
ok = diff < 1e-12
print(f"{'PASS' if ok else 'FAIL'} 5: {ranks} processes against 1: largest field difference {diff:.2e} T")
sys.exit(0 if ok else 1)
EOF
fi

exit $fail
