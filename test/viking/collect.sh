#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Gathers the benchmark results into $WORK/bench.csv and packs results, logs,
# profiles and the node survey into one archive (run once the jobs finish).
#------------------------------------------------------------------------------
set -eu
WORK=${WORK:-$HOME/scratch/vampire-benchmarks}
cd $WORK
{ echo "partition,gpu,version,commit,precision,case,atoms,repeat,wall_s,sim_s,exit"; cat results/*/*.csv; } > bench.csv
echo "$(($(wc -l < bench.csv) - 1)) of 144 benchmark runs recorded; non-zero exit codes:"
awk -F, 'NR > 1 && $11 != 0' bench.csv
archive=$WORK/viking-benchmarks-$(date +%Y%m%d-%H%M).tar.gz
tar czf $archive $(ls -d commits.txt bench.csv results logs profiles recon bin/*.log 2> /dev/null)
echo "Archive: $archive"
