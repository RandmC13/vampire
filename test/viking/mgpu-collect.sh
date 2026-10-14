#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Packs the multi-GPU check archives, scaling results, run logs and job logs
# into one archive (run once the jobs have finished).
#------------------------------------------------------------------------------
set -eu
WORK=${WORK:-$HOME/scratch/vampire-benchmarks}
cd $WORK/mgpu
for f in scaling/*.csv; do
   [ -f $f ] && echo "$f: $(($(wc -l < $f) - 1)) runs, $(awk -F, 'NR > 1 && $14 != 0' $f | wc -l) with non-zero exit code"
done
archive=$WORK/viking-mgpu-benchmarks-$(date +%Y%m%d-%H%M).tar.gz
tar czf $archive --exclude='*.data' $(ls -d commit.txt check scaling/*.csv scaling/runs scaling/bin-*/*.log logs 2> /dev/null)
echo "Archive: $archive"
