#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Packs the results of the capped three-track HAMR run into one archive: logs,
# the time series and logs of every write segment, grain files, the meta files
# of the spin snapshots and the read-back fields; not the checkpoints or the
# spin data (which stay in $WORK/capped/write for the visualisation). Prints
# the space used first.
#------------------------------------------------------------------------------
set -eu
WORK=${WORK:-$HOME/scratch/vampire-benchmarks}
cd $WORK/capped
echo "Space used:"; du -sh write read-track* 2> /dev/null || true
cat write/time.txt read-track*/time.txt 2> /dev/null || true

files=(commit.txt mode.txt logs bin/*.build.log)
for d in write read-track1 read-track2 read-track3; do
   [ -d $d ] && files+=($(cd $d && find . -maxdepth 1 -type f ! -name '*.chk' ! -name '*.chk.tmp' ! -name '*.data' ! -name '*.meta' | sed "s|^\.|$d|"))
done
# the meta files of the spin snapshots (time, field, temperature of each) and of the coordinates
[ -d write ] && files+=($(ls -d write/*.meta 2> /dev/null || true))
archive=$WORK/viking-capped-$(date +%Y%m%d-%H%M).tar.gz
tar czf $archive --exclude='*.chk' --exclude='*.chk.tmp' "${files[@]}"
echo "Archive: $archive ($(du -h $archive | cut -f1))"
