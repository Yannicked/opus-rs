#!/bin/bash
# Instructions retired (user mode) per encode/decode pass, upstream vs new.
# Deterministic, so it is immune to the timing noise of the shared laptop;
# reported next to the wall-clock numbers, not instead of them.
. ${SCRATCH:-/tmp/opus-rs-bench}/env.sh
(cd ${SCRATCH:-/tmp/opus-rs-bench}/h-up && CARGO_TARGET_DIR=${SCRATCH:-/tmp/opus-rs-bench}/tgt-hup nice cargo build --release -j4 2>&1 | grep -E "^error")
(cd ${SCRATCH:-/tmp/opus-rs-bench}/h-new && CARGO_TARGET_DIR=${SCRATCH:-/tmp/opus-rs-bench}/tgt-hnew nice cargo build --release -j4 2>&1 | grep -E "^error")
W=/home/yannick/projects/opus-rs-safety/fixtures/answer_16k.wav
N=${N:-3}
cnt() { perf stat -e instructions:u -x, "$@" 2>&1 >/dev/null | grep instructions | cut -d, -f1; }
printf "%-14s %-8s %14s %14s %8s\n" case kind upstream new delta
for c in ${CASES:-silk_nb silk_wb hybrid_fb celt_fb_st celt_fb_10ms}; do for k in ${KINDS:-enc dec dec_plc}; do
  for v in up new; do
    vn=BIN_$v; B=${!vn:-${SCRATCH:-/tmp/opus-rs-bench}/tgt-h$v/release/h-$v}
    a=$(cnt $B run $W $c $k 0); b=$(cnt $B run $W $c $k $N)
    eval "r_$v=$(( (b - a) / N ))"
  done
  printf "%-14s %-8s %14d %14d %+7.2f%%\n" $c $k $r_up $r_new $(python3 -c "print(100*($r_new-$r_up)/$r_up)")
done; done
