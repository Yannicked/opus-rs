#!/bin/bash
# Build the criterion bench for the work tree and A/B it against the
# "safety" binary on filter $1, pinned, alternating, $2 rounds.
S=${SCRATCH:-/tmp/opus-rs-bench}
. $S/env.sh
export CARGO_TARGET_DIR=$S/tgt-bench
(cd /home/yannick/projects/opus-rs-safety && nice cargo +1.98.1 bench -j4 --bench opus_bench --no-run 2>&1 | grep -E "^error" -A5)
cp $S/tgt-bench/release/deps/opus_bench-c31b702b9df6e7f2 $S/bench-try
cd /home/yannick/projects/opus-rs-safety
for r in $(seq 1 ${2:-3}); do for v in safety try; do
  rm -rf $S/crit-tmp; CRITERION_HOME=$S/crit-tmp taskset -c 6 nice -n 5 $S/bench-$v --bench --noplot "$1" 2>/dev/null | python3 -c "
import re,sys
unit={'ps':1e-3,'ns':1,'µs':1e3,'ms':1e6,'s':1e9}
name=None
for l in sys.stdin:
    m=re.match(r'^(\S+)\s+time:\s+\[\S+ \S+ (\S+) (\S+)',l)
    if m: print('$v',m.group(1),float(m.group(2))*unit[m.group(3)]); continue
    m=re.match(r'^\s+time:\s+\[\S+ (?:ps|ns|µs|ms|s) (\S+) (ps|ns|µs|ms|s)',l)
    if m and name: print('$v',name,float(m.group(1))*unit[m.group(2)]); continue
    if l.strip() and not l.startswith(' ') and not l.startswith('Bench') and 'change' not in l and 'Warn' not in l and 'Found' not in l: name=l.strip()
"
done; done | python3 -c "
import sys,collections,statistics
d=collections.defaultdict(lambda: collections.defaultdict(list))
for l in sys.stdin:
    v,n,t=l.split(); d[n][v].append(float(t))
for n in sorted(d):
    a=d[n]['safety']; b=d[n]['try']
    if a and b: print(f'{n:50} safety {statistics.median(a):10.1f} try {statistics.median(b):10.1f} {100*(statistics.median(b)-statistics.median(a))/statistics.median(a):+6.1f}%  spread {100*(max(a)-min(a))/statistics.median(a):.1f}%')
"
