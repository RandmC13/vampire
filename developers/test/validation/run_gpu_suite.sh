#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Builds the CUDA version of vampire in double and single precision, runs the
# validation suite with each build and compares against the CPU references.
# Each build is run with statistics calculated on the GPU (default) and on the
# CPU (gpu:calculate-statistics-on-cpu=true). Also checks that the CPU code
# path of the CUDA binary (gpu:acceleration=off) reproduces the serial code,
# and times the hamr-perf benchmark.
#
# Usage (from anywhere):  developers/test/validation/run_gpu_suite.sh [label-prefix]
#
# Results are written to developers/test/validation/results/<prefix>-* and packed, with
# build logs, comparison reports and system information, into
# developers/test/validation/results/gpu-suite-<host>-<date>.tar.gz
#
# Note: the CUDA and serial builds share object files, so this script runs
# 'make clean' before each build and again at the end.
#------------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/../../.."

prefix=${1:-gpu}
host=$(hostname -s)
name=gpu-suite-$host-$(date +%Y%m%d-%H%M)
results=developers/test/validation/results
out=$results/$name
vt="python3 developers/test/validation/vtest.py"
mkdir -p $out

# record system information
{ nvidia-smi; nvcc --version; g++ --version; echo "git: $(git rev-parse HEAD)"; git status --short; } > $out/system.txt 2>&1

labels=""
for pr in dp sp; do

   echo "=== Building CUDA version ($pr) ==="
   make clean
   if ! make -j$(nproc) cuda CUDA_PR=-DCUDA_$(echo $pr | tr a-z A-Z) > $out/build-$pr.log 2>&1; then
      echo "Build failed, see $out/build-$pr.log"
      continue
   fi
   cp vampire-cuda vampire-cuda-$pr

   echo "=== Running validation suite ($pr) ==="
   $vt run --exe ./vampire-cuda-$pr --label $prefix-$pr --precision $pr
   $vt compare --test $prefix-$pr --ref cpu --verbose > $out/compare-$pr.txt

   echo "=== Running validation suite ($pr, statistics on CPU) ==="
   $vt run --exe ./vampire-cuda-$pr --label $prefix-$pr-cpustats --precision $pr --extra "gpu:calculate-statistics-on-cpu=true"
   $vt compare --test $prefix-$pr-cpustats --ref cpu --verbose > $out/compare-$pr-cpustats.txt
   # same dynamics as the default run, only the statistics summation order differs
   $vt compare --test $prefix-$pr --ref $prefix-$pr-cpustats --verbose > $out/compare-$pr-vs-cpustats.txt

   echo "=== Running hamr-perf benchmark ($pr) ==="
   $vt run --exe ./vampire-cuda-$pr --label $prefix-$pr-bench --precision $pr --cases hamr-perf
   $vt compare --test $prefix-$pr-bench --ref cpu --cases hamr-perf > $out/benchmark-$pr.txt
   $vt run --exe ./vampire-cuda-$pr --label $prefix-$pr-cpustats-bench --precision $pr --cases hamr-perf --extra "gpu:calculate-statistics-on-cpu=true"
   $vt compare --test $prefix-$pr-cpustats-bench --ref cpu --cases hamr-perf > $out/benchmark-$pr-cpustats.txt

   labels="$labels $prefix-$pr $prefix-$pr-bench $prefix-$pr-cpustats $prefix-$pr-cpustats-bench"
done

# the CPU path of the CUDA binary should be bit-identical to vampire-serial
if [ -x vampire-cuda-dp ]; then
   echo "=== Running validation suite (CUDA binary, gpu:acceleration=off) ==="
   $vt run --exe ./vampire-cuda-dp --label $prefix-cpupath --extra "gpu:acceleration=off" --jobs $(nproc)
   $vt compare --test $prefix-cpupath --ref cpu --verbose > $out/compare-cpupath.txt
   labels="$labels $prefix-cpupath"
   cp vampire-cuda-dp vampire-cuda   # leave the default (double precision) build in place
fi
make clean

# configuration files must never contain NaN: on the GPU the spins on the host are
# overwritten with NaN whenever they are not valid, so NaN here means invalid spins were used
grep -rli nan --include='*.data' --include='*.cfg' $(for l in $labels; do echo $results/$l; done) > $out/config-nan-check.txt 2>/dev/null
echo "configuration files containing NaN: $(wc -l < $out/config-nan-check.txt)" >> $out/config-nan-check.txt

# pack results (atomic configuration files are excluded to keep the archive small)
tar czf $results/$name.tar.gz --exclude='*.data' --exclude='*.meta' -C $results $name $labels

echo
echo "=== Summary ==="
for f in $out/compare-*.txt $out/benchmark-*.txt $out/config-nan-check.txt; do
   [ -f $f ] && echo "$(basename $f): $(tail -n 1 $f)"
done
echo "Archive: $results/$name.tar.gz"
