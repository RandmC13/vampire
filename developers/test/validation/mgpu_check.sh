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
#   5. runs the other LLG cases whose initial state does not depend on the
#      decomposition (finite temperature, HAMR and program cases) with one seed
#      on each number of ranks and checks that their final spin configurations
#      are bit for bit identical on every number of ranks
#   6. runs those cases with all their seeds on the first number of ranks and
#      compares them with the serial CPU references (statistically for finite
#      temperature cases: the thermal noise sequence differs from the CPU's)
#   7. reruns the time series and a HAMR case of stage 5 saving a checkpoint
#      part way, and continues them from it, on each number of ranks, and checks
#      that their final spins are bit for bit those of the uninterrupted runs
#   8. restarts a one step time series from the checkpoint saved at the end of
#      the granular HAMR writes of stage 5 (sim:load-checkpoint=restart, binary
#      file-per-node configuration output) on each number of ranks, and checks
#      that its final spins are bit for bit identical on every number of ranks;
#      the deterministic granular write is also compared with the first number
#      of ranks, including its grain statistics
#   9. reruns the granular HAMR writes with gpu:skip-non-magnetic-atoms on each
#      number of ranks: output and grain files must be identical to those of
#      stage 5, and so must the final spins of every magnetic atom
#  10. runs the chain of hamr_chain.sh (granular medium with grain rounding) on
#      each number of ranks: the CPU MPI version relaxes it by Monte Carlo and
#      writes it from the relaxed state as a reference; the multi-GPU version
#      writes it from the same relaxed state (compared with the reference within
#      the deterministic tolerance), and the CPU MPI version loads the multi-GPU
#      checkpoint, whose spins must be bit for bit those of the multi-GPU write
#  11. continues the runs of stage 7 on each number of ranks from the checkpoints
#      saved part way on the previous number of ranks of the list (cyclically,
#      so that each number of ranks loads checkpoints of another): the atoms are
#      found by their keys, and the final spins must be bit for bit those of the
#      uninterrupted runs
#  12. runs the chain of stage 10 on a medium with a 1 % dense fill between the
#      grains (as in production runs) and create:decomposition-independent-random,
#      which makes the fill independent of the decomposition, with each stage on
#      a different number of ranks: the multi-GPU write on n ranks starts from
#      the relaxation on the previous number of ranks of the list (compared with
#      the CPU write on those ranks within the deterministic tolerance), and the
#      CPU MPI version loads its checkpoint on the next number of ranks
#
# Usage (from anywhere):  developers/test/validation/mgpu_check.sh [ranks ...]   (default: 1 2 4)
#
# Environment variables:
#   JOBS       parallel build jobs (default: number of cores)
#   CUDA_ARCH  GPU architecture flags (default: -arch=native)
#   MPIRUN     MPI launcher (default: mpirun --oversubscribe --bind-to none)
#   GPU_JOBS   GPU processes running at the same time (default: 4): runs on n ranks
#              are started GPU_JOBS/n at a time; results do not depend on it
#
# Results are packed into developers/test/validation/results/mgpu-check-<host>-<date>.tar.gz
#
# Note: the builds share object directories with the other versions; this
# script runs 'make clean' before each build and at the end.
#------------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/../../.."

ranks=${*:-1 2 4}
jobs=${JOBS:-$(nproc)}
arch=${CUDA_ARCH:--arch=native}
# no binding: several runs at a time would otherwise bind their ranks to the same cores
export MPIRUN=${MPIRUN:-mpirun --oversubscribe --bind-to none}
gpu_jobs=${GPU_JOBS:-4}
# concurrent runs on <ranks> ranks sharing the GPU (gpu_runs) or the CPU cores (cpu_runs)
gpu_runs(){ echo $(( gpu_jobs / $1 > 0 ? gpu_jobs / $1 : 1 )); }
cpu_runs(){ echo $(( $(nproc) / $1 > 0 ? $(nproc) / $1 : 1 )); }
name=mgpu-check-$(hostname -s)-$(date +%Y%m%d-%H%M)
results=developers/test/validation/results
out=$results/$name
mkdir -p $out
vt="python3 developers/test/validation/vtest.py"
source developers/test/validation/hamr_chain.sh
ranks_list=($ranks)
# the number of ranks after (next_ranks) or before (prev_ranks) <n> in the list, cyclically
next_ranks(){ local i; for i in ${!ranks_list[@]}; do [ ${ranks_list[$i]} = $1 ] && echo ${ranks_list[$(( (i + 1) % ${#ranks_list[@]} ))]}; done; }
prev_ranks(){ local i; for i in ${!ranks_list[@]}; do [ ${ranks_list[$i]} = $1 ] && echo ${ranks_list[$(( (i + ${#ranks_list[@]} - 1) % ${#ranks_list[@]} ))]}; done; }
# medium of stage 12: the chain case with a 1 % dense fill, whose atoms are drawn at random
sparse_case=$out/sparse-case/hamr-granular-t0
mkdir -p $sparse_case
cp $chain_case/* $sparse_case/
sed -i 's/^material\[4\]:density = 1.0$/material[4]:density = 0.01/' $sparse_case/FePt_sensor.mat
grep -q '^material\[4\]:density = 0.01$' $sparse_case/FePt_sensor.mat || { echo "Fill density of $chain_case not found"; exit 1; }
keyed="create:decomposition-independent-random"

spiral=t0-spin-spiral
t0=$(ls developers/test/validation/cases | grep '^t0-' | grep -v "^$spiral\$" | paste -sd,)
# LLG cases (other than t0-*) whose initial spins do not depend on the decomposition; the
# programs of ft_cont never reach the end-of-run configuration output, so for them a
# configuration is written at every output instead (they have only a few outputs)
ft_end=analytic-thermal,hamr-small,hamr-t0,prog-field-cool,prog-fmr,prog-hysteresis,prog-laser-pulse,timeseries-T300
ft_cont=curie-llg,prog-partial-hysteresis
# granular HAMR writes, which also save a checkpoint at the end for the read-back of stage 8
readback=hamr-granular,hamr-granular-t0
ft=$ft_end,$ft_cont,$readback
# bit for bit identical results on any number of ranks need exchange sums in atom key order
configs=(--extra "config:atoms=end" --extra "config:output-format=binary" --extra "gpu:decomposition-independent-order")
# cases continued from a checkpoint (stage 7)
continued=timeseries-T300,hamr-small
configs_cont=(--extra "config:atoms" --extra "config:atoms-output-rate=1" --extra "config:output-format=binary" --extra "gpu:decomposition-independent-order")

{
	echo "ranks: $ranks, MPIRUN: $MPIRUN"
	nvidia-smi
	nvcc --version
	mpirun --version
	echo "git: $(git rev-parse HEAD)"
	git status --short
} >$out/system.txt 2>&1

# continue_runs <exe> <ranks> <label of uninterrupted runs> <new label>: reruns each run of the
# continued cases with one checkpoint saved part way (the checkpoint rate is 60 % of the time steps of
# the run, read from its last output row), then continues a second run from that checkpoint
continue_runs(){
	for c in ${continued//,/ }; do
		for src in $results/$3/$c/*/; do
			d=$results/$4/$c/$(basename $src)
			rm -rf $d; mkdir -p $d
			cp developers/test/validation/cases/$c/* $d/; rm $d/case.json
			cp $src/input $d/input.full   # input of the uninterrupted run (with its extra lines)
			dt=$(awk -F= '/^sim:time-step *=/ { print $2 + 0 }' $d/input.full)
			rate=$(awk -v dt=$dt '!/^#/ && $1 + 0 == $1 { t = $1 } END { printf "%d", int(0.6 * int(t / dt + 0.5) / 1000) * 1000 }' $src/output)
			{ cat $d/input.full; echo "sim:save-checkpoint=continuous"; echo "sim:save-checkpoint-rate=$rate"; } >$d/input
			(cd $d && $MPIRUN -np $2 $1 >stdout-first 2>&1)
			{ cat $d/input.full; echo "sim:load-checkpoint=continue"; } >$d/input
			(cd $d && $MPIRUN -np $2 $1 >stdout 2>&1) || echo "$c $(basename $src): continuation failed (see $d/stdout)"
			grep -q "Checkpoint file loaded at sim::time $rate\." $d/log* || echo "$c $(basename $src): checkpoint not loaded at time step $rate (see $d/log)"
		done
	done
}

# xcontinue_runs <exe> <ranks> <label of continued runs on other ranks> <new label>: continues each
# run of those labels from the checkpoint it saved part way
xcontinue_runs(){
	for c in ${continued//,/ }; do
		for src in $results/$3/$c/*/; do
			d=$results/$4/$c/$(basename $src)
			rm -rf $d; mkdir -p $d
			cp developers/test/validation/cases/$c/* $src/vampire*.chk $d/; rm $d/case.json
			cp $src/input $d/input
			(cd $d && $MPIRUN -np $2 $1 >stdout 2>&1) || echo "$c $(basename $src): continuation failed (see $d/stdout)"
		done
	done
}

# key_loaded <label> <from ranks> <to ranks>: reports runs of <label> that did not load their checkpoint
# by atom key, which they must when the numbers of ranks differ
key_loaded(){
	[ $2 = $3 ] || grep -L "loaded by atom key" $results/$1/*/*/log | sed 's/^/FAIL checkpoint not loaded by atom key: /'
}

# readback_runs <exe> <ranks> <label of the writes> <new label>: restarts a one step 300 K time
# series from the checkpoint saved at the end of each run of the read-back cases
readback_runs(){
	for c in ${readback//,/ }; do
		for src in $results/$3/$c/*/; do
			d=$results/$4/$c/$(basename $src)
			rm -rf $d; mkdir -p $d
			cp developers/test/validation/cases/$c/* $src/vampire*.chk $d/; rm $d/case.json
			{
				sed 's/^sim:program *=.*/sim:program = time-series/' $src/input
				echo "sim:temperature = 300.0"
				echo "sim:total-time-steps = 1"
				echo "sim:time-steps-increment = 1"
				echo "sim:load-checkpoint = restart"
				echo "config:output-mode = file-per-node"
				echo "config:output-nodes = 20"
			} >$d/input
			(cd $d && $MPIRUN -np $2 $1 >stdout 2>&1) || echo "$c $(basename $src): read-back failed (see $d/stdout)"
			grep -q "Checkpoint file loaded" $d/log* || echo "$c $(basename $src): checkpoint not loaded (see $d/log)"
		done
	done
}

echo "=== Building CPU MPI version ==="
make clean
if ! make -j$jobs parallel >$out/build-parallel.log 2>&1; then
	echo "Build failed, see $out/build-parallel.log"
	exit 1
fi
for n in $ranks; do
	[ $n = 1 ] && continue
	echo "=== CPU MPI version, $n ranks ==="
	$vt run --exe ./vampire-parallel --label $name-parallel-np$n --ranks $n --cases $t0 --jobs $(cpu_runs $n)
done
echo "=== CPU MPI version, HAMR chain: Monte Carlo relaxation and reference write ==="
for n in $ranks; do
	(
		chain_run relax - $name-parallel-np$n-relax dp "" $MPIRUN -np $n $PWD/vampire-parallel
		chain_run write $name-parallel-np$n-relax $name-parallel-np$n-chain dp "" $MPIRUN -np $n $PWD/vampire-parallel
	) &
	(
		chain_case=$sparse_case chain_run relax - $name-parallel-np$n-xrelax dp "$keyed" $MPIRUN -np $n $PWD/vampire-parallel
		chain_case=$sparse_case chain_run write $name-parallel-np$n-xrelax $name-parallel-np$n-xchain dp "$keyed" $MPIRUN -np $n $PWD/vampire-parallel
	) &
done
wait

for pr in dp sp; do
	echo "=== Building multi-GPU version ($pr) ==="
	make clean
	if ! make -j$jobs cuda-multigpu CUDA_ARCH="$arch" CUDA_PR=-DCUDA_$(echo $pr | tr a-z A-Z) >$out/build-mgpu-$pr.log 2>&1; then
		echo "Build failed, see $out/build-mgpu-$pr.log"
		exit 1
	fi
	for n in $ranks; do
		echo "=== Multi-GPU version ($pr), $n ranks ==="
		$vt run --exe ./vampire-cuda-multigpu --label $name-$pr-np$n --ranks $n --jobs $(gpu_runs $n) --precision $pr --cases $t0,$spiral "${configs[@]}"
		echo "=== Multi-GPU version ($pr), $n ranks, finite temperature and program cases, one seed ==="
		$vt run --exe ./vampire-cuda-multigpu --label $name-$pr-np$n-ft --ranks $n --jobs $(gpu_runs $n) --precision $pr --cases $ft_end --seeds 1 "${configs[@]}"
		$vt run --exe ./vampire-cuda-multigpu --label $name-$pr-np$n-ft --ranks $n --jobs $(gpu_runs $n) --precision $pr --cases $readback --seeds 1 "${configs[@]}" --extra "sim:save-checkpoint=end"
		$vt run --exe ./vampire-cuda-multigpu --label $name-$pr-np$n-ft --ranks $n --jobs $(gpu_runs $n) --precision $pr --cases $ft_cont --seeds 1 "${configs_cont[@]}"
		echo "=== Multi-GPU version ($pr), $n ranks, continuation from a checkpoint ==="
		continue_runs $PWD/vampire-cuda-multigpu $n $name-$pr-np$n-ft $name-$pr-np$n-continue
		echo "=== Multi-GPU version ($pr), $n ranks, read-back restarted from the end of the granular writes ==="
		readback_runs $PWD/vampire-cuda-multigpu $n $name-$pr-np$n-ft $name-$pr-np$n-readback
		echo "=== Multi-GPU version ($pr), $n ranks, granular writes with gpu:skip-non-magnetic-atoms ==="
		$vt run --exe ./vampire-cuda-multigpu --label $name-$pr-np$n-skip --ranks $n --jobs $(gpu_runs $n) --precision $pr --cases $readback --seeds 1 "${configs[@]}" --extra "gpu:skip-non-magnetic-atoms"
		echo "=== Multi-GPU version ($pr), $n ranks, HAMR chain: write from the CPU relaxation, CPU load of its checkpoint ==="
		chain_run write $name-parallel-np$n-relax $name-$pr-np$n-chain $pr "" $MPIRUN -np $n $PWD/vampire-cuda-multigpu
		chain_run load $name-$pr-np$n-chain $name-$pr-np$n-chainload $pr "" $MPIRUN -np $n $PWD/vampire-parallel
	done
	for n in $ranks; do
		p=$(prev_ranks $n)
		echo "=== Multi-GPU version ($pr), $n ranks, continuation from the checkpoints saved on $p ranks ==="
		xcontinue_runs $PWD/vampire-cuda-multigpu $n $name-$pr-np$p-continue $name-$pr-np$n-xcontinue
		echo "=== Multi-GPU version ($pr), $n ranks, 1 % fill HAMR chain: write from the CPU relaxation on $p ranks, CPU load on $(next_ranks $n) ranks ==="
		chain_case=$sparse_case chain_run write $name-parallel-np$p-xrelax $name-$pr-np$n-xchain $pr "$keyed" $MPIRUN -np $n $PWD/vampire-cuda-multigpu
		chain_case=$sparse_case chain_run load $name-$pr-np$n-xchain $name-$pr-np$n-xchainload $pr "$keyed" $MPIRUN -np $(next_ranks $n) $PWD/vampire-parallel
	done
	first=$(echo $ranks | awk '{print $1}')
	echo "=== Multi-GPU version ($pr), $first rank(s), finite temperature and program cases, all seeds ==="
	$vt run --exe ./vampire-cuda-multigpu --label $name-$pr-stat --ranks $first --jobs $(gpu_runs $first) --precision $pr --cases $ft
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
		[ $n != $first ] && $vt configs --test $name-$pr-np$n-ft --ref $name-$pr-np$first-ft --cases $ft >$out/configs-ft-$pr-np$n.txt
		[ $n != $first ] && $vt configs --test $name-$pr-np$n-readback --ref $name-$pr-np$first-readback --cases $readback >$out/configs-readback-$pr-np$n.txt
		[ $n != $first ] && $vt compare --test $name-$pr-np$n-ft --ref $name-$pr-np$first-ft --cases hamr-granular-t0 --verbose >$out/compare-granular-$pr-np$n.txt
		$vt configs --test $name-$pr-np$n-continue --ref $name-$pr-np$n-ft --cases $continued >$out/continue-$pr-np$n.txt
		{
			$vt identical --test $name-$pr-np$n-skip --ref $name-$pr-np$n-ft --cases $readback
			$vt configs --test $name-$pr-np$n-skip --ref $name-$pr-np$n-ft --cases $readback --magnetic-only
			grep -L "Not integrating" $results/$name-$pr-np$n-skip/*/*/log | sed 's/^/FAIL skipping not active: /'
		} >$out/skip-$pr-np$n.txt
		{
			$vt compare --test $name-$pr-np$n-chain --ref $name-parallel-np$n-chain --cases hamr-granular-t0 --verbose
			$vt configs --test $name-$pr-np$n-chainload --ref $name-$pr-np$n-chain --cases hamr-granular-t0
		} >$out/chain-$pr-np$n.txt
		p=$(prev_ranks $n)
		{
			$vt configs --test $name-$pr-np$n-xcontinue --ref $name-$pr-np$n-ft --cases $continued
			key_loaded $name-$pr-np$n-xcontinue $p $n
		} >$out/xcontinue-$pr-np$n.txt
		{
			$vt compare --test $name-$pr-np$n-xchain --ref $name-parallel-np$p-xchain --cases hamr-granular-t0 --verbose
			$vt configs --test $name-$pr-np$n-xchainload --ref $name-$pr-np$n-xchain --cases hamr-granular-t0
			key_loaded $name-$pr-np$n-xchain $p $n
			key_loaded $name-$pr-np$n-xchainload $n $(next_ranks $n)
		} >$out/xchain-$pr-np$n.txt
	done
	$vt compare --test $name-$pr-stat --ref cpu --cases $ft --verbose --summary >$out/compare-stat-$pr.txt
done

tar czf $results/$name.tar.gz --exclude='*.data' --exclude='*.chk' -C $results $name $(cd $results && ls -d $name-*)

echo
echo "=== Summary (known GPU anisotropy failures: t0-anis-biaxial, -triaxial, -cubic-rotated, 5 each) ==="
for f in $out/compare-*.txt; do
	echo "$(basename $f .txt): $(grep FAIL $f | grep -vc OVERALL) failing quantities; other than the known cases:"
	grep FAIL $f | grep -v "biaxial\|triaxial\|cubic-rotated\|OVERALL" | sed 's/^/   /'
done
for f in $out/compare-stat-*.txt; do
	echo "$(basename $f .txt): $(grep -E '^   all|fraction' $f | sed 's/^ *//' | paste -sd';')"
done
for f in $out/configs-*.txt; do
	echo "$(basename $f .txt) vs $first rank(s): $(grep -c IDENTICAL $f) identical, $(grep 'DIFFERENT\|FAIL' $f | grep -vc OVERALL) different or missing"
	grep "DIFFERENT\|FAIL" $f | grep -v OVERALL | sed 's/^/   /'
done
for f in $out/skip-*.txt; do
	echo "$(basename $f .txt) vs without skipping: $(grep -c IDENTICAL $f) identical, $(grep 'DIFFERENT\|FAIL' $f | grep -vc OVERALL) different or missing"
	grep "DIFFERENT\|FAIL" $f | grep -v OVERALL | sed 's/^/   /'
done
for f in $out/continue-*.txt $out/xcontinue-*.txt; do
	echo "$(basename $f .txt) vs uninterrupted run: $(grep -c IDENTICAL $f) identical, $(grep 'DIFFERENT\|FAIL' $f | grep -vc OVERALL) different or missing"
	grep "DIFFERENT\|FAIL" $f | grep -v OVERALL | sed 's/^/   /'
done
for f in $out/chain-*.txt $out/xchain-*.txt; do
	echo "$(basename $f .txt) (write vs CPU write; CPU load vs write): $(grep OVERALL $f | paste -sd';')"
	grep "FAIL\|DIFFERENT" $f | grep -v OVERALL | sed 's/^/   /'
done
echo "Archive: $results/$name.tar.gz"
