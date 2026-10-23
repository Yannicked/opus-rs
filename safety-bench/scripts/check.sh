#!/bin/bash
# Build the harness against the work tree and compare with the reference
# (upstream 0.1.39 plus the compute_theta aliasing fix): encoder bytes on the
# test signals, and decoder output on the reference's packet corpus plus
# mutated and random packets.
. ${SCRATCH:-/tmp/opus-rs-bench}/env.sh
cd ${SCRATCH:-/tmp/opus-rs-bench}/h-new && CARGO_TARGET_DIR=${SCRATCH:-/tmp/opus-rs-bench}/tgt-hnew nice cargo build --release -j4 2>&1 | grep -E "^(warning|error)" -A5 | head -40
CORPUS_IN=${SCRATCH:-/tmp/opus-rs-bench}/corpus-ref.bin nice ${SCRATCH:-/tmp/opus-rs-bench}/tgt-hnew/release/h-new hash /home/yannick/projects/opus-rs-safety/fixtures/answer_16k.wav > ${SCRATCH:-/tmp/opus-rs-bench}/hash-new.txt
if diff -q ${SCRATCH:-/tmp/opus-rs-bench}/hash-ref.txt ${SCRATCH:-/tmp/opus-rs-bench}/hash-new.txt >/dev/null; then echo "HASH IDENTICAL: $(tail -1 ${SCRATCH:-/tmp/opus-rs-bench}/hash-new.txt)"; else echo "HASH DIFFERS"; diff ${SCRATCH:-/tmp/opus-rs-bench}/hash-ref.txt ${SCRATCH:-/tmp/opus-rs-bench}/hash-new.txt | head -20; exit 1; fi
