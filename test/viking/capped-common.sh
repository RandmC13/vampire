#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Settings and functions shared by the capped three-track HAMR jobs (sourced,
# not run).
#
#   REPO     vampire checkout the jobs were submitted from (set by capped-submit.sh)
#   WORK     directory for sources, binaries and runs, not on the file system of
#            the home directory (default ~/scratch/vampire-benchmarks); the files
#            of this run go to $WORK/capped
#   CAP_MODE full (the run), dry-run (40 nm long film, fast head) or test (two
#            tracks on a 98 nm wide film, TEST_LENGTH nm long, default 100);
#            set by capped-submit.sh
#   MODULES  modules to load instead of the defaults below
#------------------------------------------------------------------------------
module purge
module load ${MODULES:-CUDA/12.9.0 OpenMPI/4.1.5-GCC-12.3.0 SciPy-bundle/2023.07-gfbf-2023a}
set -u   # after the module commands, which may use unset variables
: ${REPO:?REPO is not set, submit the jobs with test/viking/capped-submit.sh}
WORK=${WORK:-$HOME/scratch/vampire-benchmarks}
CAP=$WORK/capped
MODE=${CAP_MODE:-full}
INPUTS=$REPO/test/viking/capped
SAMPLE=$REPO/test/viking/h3-sample.sh

{ module list; echo "nodes: ${SLURM_JOB_NODELIST:-}"; nvidia-smi -L 2> /dev/null; } 2>&1

# build <make target> <binary> <destination> [make arguments]: builds a private copy of the exported sources
build(){
   local target=$1 binary=$2 dest=$3
   shift 3
   local src=$CAP/build-$SLURM_JOB_ID
   rm -rf "${src:?}"; cp -r $CAP/src $src
   mkdir -p $(dirname $dest)
   make -C $src -j$(nproc) $target "$@" > $dest.build.log 2>&1 || { tail -n 30 $dest.build.log; exit 1; }
   mv $src/$binary $dest
   rm -rf "${src:?}"
}

# stage_dir <stage> <directory>: a new run directory with the input and material file of <stage> (write or
# read); the dry run shortens the film to 40 nm (two bits per track), moves the head at 2000 m/s and writes
# checkpoints and snapshots more often; the test writes two tracks on a 98 nm wide film (2 tracks of 32 nm and
# the guard bands) of TEST_LENGTH nm (default 100: 7 bits per track), track 1 (one bit per transition) and then
# its neighbour, track 2 (two bits per transition), with the physics of the run, and a spin snapshot every 0.25 ns
stage_dir(){
   mkdir -p $2
   cp $INPUTS/$1/* $2/
   if [ $MODE = test ]; then
      local length=${TEST_LENGTH:-100}
      # bits per track as the hamr module counts them: floor((length - 1 A)/bit size)
      local n=$(( (length*10 - 1) / 140 )) i seq=""
      for ((i = 0; i < n; i++)); do seq+="$(( i % 2 ? -1 : 1 )),"; done
      for ((i = 0; i < n; i++)); do seq+="$(( i / 2 % 2 ? -1 : 1 )),"; done
      printf '%s\n' "" "# test: two tracks of $n bits on a $length x 98 nm film" \
         "dimensions:system-size-x = $length !nm" "dimensions:system-size-y = 98 !nm" >> $2/input
      [ $1 = write ] && printf '%s\n' "hamr:number-of-bits = $((2*n))" "hamr:bit-sequence = ${seq%,}" \
         "hamr:track-order = 1,2" "config:atoms-output-rate = 250000" >> $2/input
   fi
   if [ $MODE = dry-run ]; then
      printf '%s\n' "" "# dry run: 40 nm long film, fast head, short equilibration" \
         "dimensions:system-size-x = 40 !nm" "hamr:head-speed = 2000 !m/s" "sim:equilibration-time-steps = 5000" \
         "sim:save-checkpoint-rate = 20000" "config:atoms-output-rate = 20000" >> $2/input
   fi
}

# checkpoints_consistent <directory> <ranks>: every process has a checkpoint file and all are at the same step
# (a checkpoint is written under a temporary name and renamed once all processes have written theirs)
checkpoints_consistent(){
   python3 - "$1" "$2" << 'EOF'
import sys, struct, os
d, n = sys.argv[1], int(sys.argv[2])
steps = set()
for r in range(n):
   f = f"{d}/vampire{r}.chk"
   if not os.path.exists(f):
      print(f"Error: {f} is missing"); sys.exit(1)
   with open(f, "rb") as h:
      steps.add(struct.unpack("<q", h.read(16)[8:16])[0])
if len(steps) != 1:
   print(f"Error: the checkpoint files in {d} are at different steps: {sorted(steps)}"); sys.exit(1)
print(f"checkpoint of {n} processes at step {steps.pop()}")
EOF
}
