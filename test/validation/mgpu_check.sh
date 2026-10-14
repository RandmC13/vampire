#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Checks the multi-GPU version (make cuda-multigpu) on 1, 2 and 4 MPI ranks,
# which may share one GPU:
#
#   1. builds the CPU MPI version and runs the zero temperature cases on the
#      same numbers of ranks (same decomposition and initial spins as the GPU
#      runs, since the random initial spins of most cases depend on the rank)
#   2. builds the multi-GPU version in double and single precision and runs
#      the zero temperature cases on each number of ranks
#   3. compares 1-rank runs with the serial CPU references, multi-rank runs
#      with the CPU MPI runs, and t0-spin-spiral (whose initial state does not
#      depend on the decomposition) with the serial references on any number
#      of ranks
#   4. checks that the final spin configurations of t0-spin-spiral are bit for
#      bit identical on every number of ranks (vtest.py configs); this needs
#      gpu:decomposition-independent-order, which the GPU runs here enable
#
# Usage (from anywhere):  test/validation/mgpu_check.sh [ranks ...]   (default: 1 2 4)
#
# Environment variables:
#   JOBS       parallel build jobs (default: 4)
#   CUDA_ARCH  GPU architecture flags (default: -arch=native)
#   MPIRUN     MPI launcher (default: mpirun --oversubscribe)
#
# Results are packed into test/validation/results/mgpu-check-<host>-<date>.tar.gz
#
# Note: the builds share object directories with the other versions; this
# script runs 'make clean' before each build and at the end.
#------------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/../.."

ranks=${*:-1 2 4}
jobs=${JOBS:-10}
arch=${CUDA_ARCH:--arch=native}
export MPIRUN=${MPIRUN:-mpirun --oversubscribe}
name=mgpu-check-$(hostname -s)-$(date +%Y%m%d-%H%M)
results=test/validation/results
out=$results/$name
mkdir -p $out
vt="python3 test/validation/vtest.py"

spiral=t0-spin-spiral
t0=$(ls test/validation/cases | grep '^t0-' | grep -v "^$spiral\$" | paste -sd,)
# bit for bit identical results on any number of ranks need exchange sums in atom key order
configs=(--extra "config:atoms=end" --extra "config:output-format=binary" --extra "gpu:decomposition-independent-order")

{
	echo "ranks: $ranks, MPIRUN: $MPIRUN"
	nvidia-smi
	nvcc --version
	mpirun --version
	echo "git: $(git rev-parse HEAD)"
	git status --short
} >$out/system.txt 2>&1

echo "=== Building CPU MPI version ==="
make clean
if ! make -j$jobs parallel >$out/build-parallel.log 2>&1; then
	echo "Build failed, see $out/build-parallel.log"
	exit 1
fi
for n in $ranks; do
	[ $n = 1 ] && continue
	echo "=== CPU MPI version, $n ranks ==="
	$vt run --exe ./vampire-parallel --label $name-parallel-np$n --ranks $n --cases $t0
done

for pr in dp sp; do
	echo "=== Building multi-GPU version ($pr) ==="
	make clean
	if ! make -j$jobs cuda-multigpu CUDA_ARCH="$arch" CUDA_PR=-DCUDA_$(echo $pr | tr a-z A-Z) >$out/build-mgpu-$pr.log 2>&1; then
		echo "Build failed, see $out/build-mgpu-$pr.log"
		exit 1
	fi
	for n in $ranks; do
		echo "=== Multi-GPU version ($pr), $n ranks ==="
		$vt run --exe ./vampire-cuda-multigpu --label $name-$pr-np$n --ranks $n --precision $pr --cases $t0,$spiral "${configs[@]}"
	done
done
make clean
rm -f vampire-parallel vampire-cuda-multigpu

#------------------------------------------------------------------------------
# comparisons
#------------------------------------------------------------------------------
first=$(echo $ranks | awk '{print $1}')
for pr in dp sp; do
	for n in $ranks; do
		if [ $n = 1 ]; then
			$vt compare --test $name-$pr-np$n --ref cpu --cases $t0,$spiral --verbose >$out/compare-$pr-np$n.txt
		else
			$vt compare --test $name-$pr-np$n --ref $name-parallel-np$n --cases $t0 --verbose >$out/compare-$pr-np$n.txt
			$vt compare --test $name-$pr-np$n --ref cpu --cases $spiral --verbose >>$out/compare-$pr-np$n.txt
		fi
		[ $n != $first ] && $vt configs --test $name-$pr-np$n --ref $name-$pr-np$first --cases $spiral >$out/configs-$pr-np$n.txt
	done
done

tar czf $results/$name.tar.gz --exclude='*.data' -C $results $name $(cd $results && ls -d $name-*)

echo
echo "=== Summary (known GPU anisotropy failures: t0-anis-biaxial, -triaxial, -cubic-rotated, 5 each) ==="
for f in $out/compare-*.txt; do
	echo "$(basename $f .txt): $(grep FAIL $f | grep -vc OVERALL) failing quantities; other than the known cases:"
	grep FAIL $f | grep -v "biaxial\|triaxial\|cubic-rotated\|OVERALL" | sed 's/^/   /'
done
for f in $out/configs-*.txt; do echo "$(basename $f .txt) vs $first rank(s): $(grep -h "$spiral" $f | awk '{print $3}')"; done
echo "Archive: $results/$name.tar.gz"
