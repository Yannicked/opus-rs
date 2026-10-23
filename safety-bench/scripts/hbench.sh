#!/bin/bash
# Interleaved A/B timing with the harness: upstream vs work tree.
# Each round runs both binaries back to back (order alternates) with REPS
# repetitions each; reports per-round minimum times.
ROUNDS=${1:-5}; REPS=${2:-5}
. ${SCRATCH:-/tmp/opus-rs-bench}/env.sh
(cd ${SCRATCH:-/tmp/opus-rs-bench}/h-up && CARGO_TARGET_DIR=${SCRATCH:-/tmp/opus-rs-bench}/tgt-hup nice cargo build --release -j4 2>&1 | grep -E "^error")
(cd ${SCRATCH:-/tmp/opus-rs-bench}/h-new && CARGO_TARGET_DIR=${SCRATCH:-/tmp/opus-rs-bench}/tgt-hnew nice cargo build --release -j4 2>&1 | grep -E "^error")
W=/home/yannick/projects/opus-rs-safety/fixtures/answer_16k.wav
out=${SCRATCH:-/tmp/opus-rs-bench}/hbench-$(date +%H%M%S).txt
for r in $(seq 1 $ROUNDS); do
  if [ $((r % 2)) = 1 ]; then order="up new"; else order="new up"; fi
  for v in $order; do
    vn=BIN_$v; B=${!vn:-${SCRATCH:-/tmp/opus-rs-bench}/tgt-h$v/release/h-$v}; taskset -c ${PIN:-6} nice -n 5 $B bench $W $REPS | sed "s/^/$v /" >> $out
  done
done
python3 - $out <<'PY'
import sys, collections, statistics
d = collections.defaultdict(list)
for line in open(sys.argv[1]):
    v, case, kind, mn, med = line.split()
    d[(case, kind, v)].append(float(mn.split('=')[1]))
cases = sorted({(c, k) for (c, k, v) in d})
print(f"{'case':14} {'kind':8} {'upstream ms':>12} {'new ms':>9} {'delta':>7}   (median of per-round minima; spread = min..max)")
for c, k in cases:
    a = d[(c, k, 'up')]; b = d[(c, k, 'new')]
    ma, mb = statistics.median(a), statistics.median(b)
    print(f"{c:14} {k:8} {ma:12.3f} {mb:9.3f} {100*(mb-ma)/ma:+6.1f}%   up {min(a):.2f}..{max(a):.2f}  new {min(b):.2f}..{max(b):.2f}")
PY
