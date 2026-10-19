#!/bin/bash
#------------------------------------------------------------------------------
#
#   This file is part of the VAMPIRE open source package under the
#   Free BSD licence (see licence file for details).
#
#------------------------------------------------------------------------------
# Runs a program in place of this process and records its peak memory every
# 5 s in memory-rank<rank>.txt: "<rank> <peak resident host memory in kB>
# <largest GPU memory seen in MiB>" (GPU memory only where nvidia-smi exists).
#
#    h3-sample.sh program [arguments]     (one per MPI rank)
#------------------------------------------------------------------------------
rank=${OMPI_COMM_WORLD_RANK:-0}
pid=$$
(
   gpu=0
   while kill -0 $pid 2> /dev/null; do
      rss=$(awk '/^VmHWM/ { print $2 }' /proc/$pid/status 2> /dev/null)
      if command -v nvidia-smi > /dev/null; then
         g=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2> /dev/null | awk -F', *' -v p=$pid '$1 == p { print $2 }')
         [ "${g:-0}" -gt $gpu ] && gpu=$g
      fi
      [ -n "$rss" ] && echo "$rank $rss $gpu" > memory-rank$rank.txt
      sleep 5
   done
) &
exec "$@"
