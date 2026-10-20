#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Settings and functions shared by the HAMR chain jobs (sourced, not run).
#
#   REPO     vampire checkout the jobs were submitted from (set by h3-submit.sh)
#   WORK     directory for sources, binaries, runs and results; it must not be
#            on the file system of the home directory (default:
#            ~/scratch/vampire-benchmarks); HAMR chain files go to $WORK/h3
#   H3_DRY   1 for the dry run (small system, short runs; set by h3-submit.sh)
#   MODULES  modules to load instead of the defaults below
#------------------------------------------------------------------------------
module purge
module load ${MODULES:-CUDA/12.9.0 OpenMPI/4.1.5-GCC-12.3.0 SciPy-bundle/2023.07-gfbf-2023a}
set -u   # after the module commands, which may use unset variables
: ${REPO:?REPO is not set, submit the jobs with test/viking/h3-submit.sh}
WORK=${WORK:-$HOME/scratch/vampire-benchmarks}
H3=$WORK/h3
DRY=${H3_DRY:-0}
INPUTS=$REPO/test/viking/h3
SAMPLE=$REPO/test/viking/h3-sample.sh
# length of each timed benchmark run once the simulation has started (seconds)
WINDOW=$([ $DRY = 1 ] && echo 60 || echo 300)

# check_scratch <directory>: stops unless the directory is on another file system than
# the home directory, so that the large outputs cannot fill the home quota
check_scratch(){
   mkdir -p "$1"
   if [ "$(df -P "$1" | awk 'NR == 2 { print $6 }')" = "$(df -P "$HOME" | awk 'NR == 2 { print $6 }')" ]; then
      echo "Error: $1 is on the file system of the home directory; set WORK to a directory in scratch"
      exit 1
   fi
}
check_scratch $H3
mkdir -p $H3/bench

{ module list; echo "nodes: ${SLURM_JOB_NODELIST:-}"; nvidia-smi -L 2> /dev/null; } 2>&1

# build <make target> <binary> <destination> [make arguments]: builds a private copy of the exported sources
# (or of the source tree in BUILD_SRC)
build(){
   local target=$1 binary=$2 dest=$3
   shift 3
   local src=$H3/build-$SLURM_JOB_ID
   rm -rf $src; cp -r ${BUILD_SRC:-$H3/src} $src
   mkdir -p $(dirname $dest)
   make -C $src -j$(nproc) $target "$@" > $dest.build.log 2>&1 || { tail -n 30 $dest.build.log; exit 1; }
   mv $src/$binary $dest
   rm -rf $src
}

# stage_dir <stage> <directory>: a new run directory with the input and material file of
# <stage> (relax, write or read); the dry run shrinks the system and shortens the runs
stage_dir(){
   rm -rf $2; mkdir -p $2
   cp $INPUTS/$1/* $2/
   if [ $DRY = 1 ]; then
      printf '%s\n' "" "# dry run: small system, fast head, short relaxation" \
         "dimensions:system-size-x = 24 !nm" "dimensions:system-size-y = 24 !nm" \
         "hamr:track-size = 22 !nm" "hamr:head-speed = 2000 !m/s" "config:atoms-output-rate = 1000" >> $2/input
      [ $1 = relax ] && printf '%s\n' "sim:total-time-steps = 200" "sim:equilibration-time-steps = 100" >> $2/input
   fi
}

# link_checkpoint <run directory> <directory>: links the checkpoint files of a run (as many as
# the processes in its ranks.txt) into a directory
link_checkpoint(){
   local n=$(cat $1/ranks.txt) r
   for ((r = 0; r < n; r++)); do ln -sf $1/vampire$r.chk $2/; done
}

# bench_run <directory> <command ...>: a write restarted from the relaxed state that runs
# for $WINDOW seconds after its first output row and is then stopped; prints
# "setup_s,steps,window_s,steps_per_s,peak_rss_max_mib,peak_rss_sum_mib,peak_gpu_max_mib,exit"
bench_run(){
   local dir=$1
   shift
   # never save a checkpoint: the relaxed state is linked, not copied
   sed -i '/^sim:save-checkpoint/d' $dir/input
   link_checkpoint $H3/relax $dir
   cd $dir
   local start=$(date +%s.%N) t rows t0= rows0= t1= rows1=0 status
   setsid "$@" > stdout 2>&1 &
   local pid=$!
   while kill -0 $pid 2> /dev/null; do
      sleep 5
      t=$(date +%s.%N)
      rows=$(grep -vc '^#' output 2> /dev/null)
      [ "${rows:-0}" -gt 0 ] || continue
      [ -z "$t0" ] && { t0=$t; rows0=$rows; }
      t1=$t; rows1=$rows
      echo "$t $rows" >> progress.txt
      if awk -v a=$t0 -v b=$t -v w=$WINDOW 'BEGIN { exit !(b - a >= w) }'; then
         kill -TERM -- -$pid 2> /dev/null
         break
      fi
   done
   wait $pid; status=$?
   rm -f *.data *.meta
   cd - > /dev/null
   # rows are written every 100 steps (output:output-rate)
   awk -v s=$start -v a=${t0:-0} -v b=${t1:-0} -v r0=${rows0:-0} -v r1=$rows1 -v st=$status \
       -v mem="$(cat $dir/memory-rank*.txt 2> /dev/null)" 'BEGIN {
      n = split(mem, m, "\n"); for (i = 1; i <= n; i++) { split(m[i], f, " "); rss += f[2]; if (f[2] > rmax) rmax = f[2]; if (f[3] > gmax) gmax = f[3] }
      steps = 100 * (r1 - r0); w = b - a
      printf "%.1f,%d,%.1f,%.3f,%.0f,%.0f,%.0f,%d\n", a - s, steps, w, (w > 0 ? steps / w : 0), rmax / 1024, rss / 1024, gmax, st }'
}

# full_write_h <steps per second>: hours for the whole write (4.29M steps)
full_write_h(){ awk -v r=${1:-0} 'BEGIN { printf "%.2f", (r > 0 ? 4290000 / r / 3600 : 0) }'; }
