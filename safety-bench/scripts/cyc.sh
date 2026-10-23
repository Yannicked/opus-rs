#!/bin/bash
# Pinned, interleaved user-mode cycle counts for one case: A vs B.
# usage: cyc.sh BIN_A BIN_B case kind rounds
A=$1; B=$2; C=$3; K=$4; R=${5:-8}
W=/home/yannick/projects/opus-rs-safety/fixtures/answer_16k.wav
cnt() { taskset -c ${PIN:-6} nice -n 5 perf stat -e cycles:u -x, "$@" 2>&1 >/dev/null | grep cycles | cut -d, -f1; }
for r in $(seq 1 $R); do
  a0=$(cnt $A run $W $C $K 0); a=$(cnt $A run $W $C $K 10)
  b0=$(cnt $B run $W $C $K 0); b=$(cnt $B run $W $C $K 10)
  echo "$(( (a-a0)/10 )) $(( (b-b0)/10 ))"
done | python3 -c "
import sys,statistics
r=[tuple(map(int,l.split())) for l in sys.stdin]
a=[x for x,_ in r]; b=[y for _,y in r]
ma,mb=statistics.median(a),statistics.median(b)
print(f'$C $K cycles/pass A {ma:.0f} B {mb:.0f} delta {100*(mb-ma)/ma:+.2f}%  (A {min(a)}..{max(a)}, B {min(b)}..{max(b)})')"
