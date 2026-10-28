#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Packs the results of the HAMR chain into one archive (run once the jobs have
# finished): benchmark table, logs, time series, grain and memory files of every
# stage, the last spin snapshot of the write and the read-back fields; not the
# checkpoints or the other spin snapshots. Prints the space used first.
#------------------------------------------------------------------------------
set -eu
WORK=${WORK:-$HOME/scratch/vampire-benchmarks}
cd $WORK/h3
echo "Space used:"; du -sh relax write read bench profile gpu-relax window 2> /dev/null || true
[ -f bench/bench.csv ] && echo "bench.csv: $(($(wc -l < bench/bench.csv) - 1)) runs, $(awk -F, 'NR > 1 && $10 == 0' bench/bench.csv | wc -l) with no time steps measured"
cat */time.txt 2> /dev/null || true

files=(commit.txt dry-run.txt logs bench bin/*.build.log)
[ -d profile ] && files+=(profile)
[ -d gpu-relax ] && files+=(gpu-relax)
[ -d window ] && files+=(window)
for stage in relax write read; do
   [ -d $stage ] && files+=($(cd $stage && find . -maxdepth 1 -type f ! -name '*.chk' ! -name '*.data' ! -name '*.meta' | sed "s|^\.|$stage|"))
done
if [ -d write ]; then
   last=$(ls write | sed -n 's/^spins-\([0-9]\{8\}\).*/\1/p' | sort | tail -n 1)
   [ -n "$last" ] && files+=($(ls -d write/spins-$last* write/atoms-coords* write/non-magnetic-atoms* 2> /dev/null || true))
fi
archive=$WORK/viking-h3-$(date +%Y%m%d-%H%M).tar.gz
tar czf $archive --exclude='*.chk' --exclude='bench/runs/*/*.data' --exclude='*.nsys-rep' --exclude='profile/src-original' --exclude='gpu-relax/src-*' --exclude='gpu-relax/*/*/*.data' --exclude='gpu-relax/*/*/*.meta' --exclude='window/src-*' --exclude='window/*/*.data' --exclude='window/*/*.meta' "${files[@]}"
echo "Archive: $archive ($(du -h $archive | cut -f1))"
