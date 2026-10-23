#!/bin/bash
# Kernel microbench (public fns): safety vs work tree, pinned, alternating.
S=${SCRATCH:-/tmp/opus-rs-bench}
. $S/env.sh
for v in safety new; do (cd $S/kb-$v && CARGO_TARGET_DIR=$S/tgt-kb$v nice cargo build --release -j4 2>&1 | grep -E "^error" -A5); done
for r in $(seq 1 ${1:-3}); do for v in safety new; do taskset -c 6 nice -n 5 $S/tgt-kb$v/release/kb-$v | sed "s/^/$v /"; done; done | python3 -c "
import sys,collections,statistics,re
d=collections.defaultdict(lambda: collections.defaultdict(list))
for l in sys.stdin:
    v,rest=l.split(' ',1); m=re.match(r'(.*?)\s+([\d.]+) ns',rest.strip()); d[m.group(1)][v].append(float(m.group(2)))
for n in d:
    a=statistics.median(d[n]['safety']); b=statistics.median(d[n]['new'])
    print(f'{n:28} safety {a:8.1f} new {b:8.1f} {100*(b-a)/a:+6.1f}%')
"
