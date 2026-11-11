#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Packs the small results of the check of hamrvis at scale into one archive
# (run once the jobs have finished): commits, logs, run times, the reduce and
# check logs and results, and the small files of every run (not the spin
# snapshots, checkpoints or bundles). Prints the check results and the space
# used, and the commands that copy the bundles home. The raw files are left in
# place for further checks; delete $WORK/h3/scale when done.
#------------------------------------------------------------------------------
set -eu
WORK=${WORK:-$HOME/scratch/vampire-benchmarks}
cd $WORK/h3/scale
echo "Space used:"; du -sh relax write-* read-* bundle-* 2> /dev/null || true
cat */time.txt time-reduce-*.txt 2> /dev/null || true
for k in window full; do [ -f check-$k.log ] && { echo "== check $k"; grep -v '^  snapshot ' check-$k.log; }; done
files=(commit.txt hamrvis-commit.txt dry-run.txt logs $(ls -d reduce-*.log check-*.log check-*.json time-reduce-*.txt bin/*.build.log 2> /dev/null || true))
for run in relax write-window write-full read-window read-full; do
   [ -d $run ] && files+=($(cd $run && find . -maxdepth 1 -type f ! -name '*.chk' ! -name '*.data' ! -name '*.meta' ! -name 'atomistic*' | sed "s|^\.|$run|"))
done
archive=$WORK/viking-scale-$(date +%Y%m%d-%H%M).tar.gz
tar czf $archive "${files[@]}"
echo "Archive: $archive ($(du -h $archive | cut -f1))"
echo "Copy the bundles home (into the hamrvis checkout's bundles/ folder) with:"
for k in window full; do [ -d bundle-$k ] && echo "   rsync -av --info=progress2 <viking>:$PWD/bundle-$k/ bundles/h3-scale-$k/"; done
