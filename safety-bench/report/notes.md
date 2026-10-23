# Notes for the opus-rs safety benchmark data

`data.json` has the before/safety/after numbers. Things a reader must know:

**Versions.** There are four:
- "before": upstream opus-rs 0.1.39 (`0f17280`);
- "safety": branch `safety/fewer-unsafe` at `f389dcf`, as of the first report;
- "round2": `a873931`, after the first plain-Rust round;
- "after": the branch head after the redesign round.

**Round 3 changed no output.** Every new commit hashes identical to the
reference for the encoder and the decoder.

**Outputs that changed (encoder only).** Two commits change encoder output on
purpose. Every other commit is bit-identical to them.

- `compute_theta` aliasing fix (`da393a5`). Upstream wrote through `&[f32]`
  cast to `*mut f32`. That is undefined behaviour, and release builds were
  miscompiled because of it. 24 of 200 harness configs (stereo CELT and
  hybrid encodes) change. The fixed release output equals upstream's own debug
  build in all 200 configs.
- Pulse-search fix (`a4bff32`). On all-zero bands with K > 4 and N not a
  multiple of 8, upstream wrote every pulse past the end of `y`. In 0.1.39,
  through `OpusEncoder`, the pulses land in spare stack slots and are lost.
  A few more encoder configs change.

So the stereo CELT encode numbers compare different work: "before" encodes
other bits. Decoder output is identical across all three versions, on the
same packet corpus plus 160,000 mutated or random packets.

**What "bit-identical" was checked against.** The reference is upstream plus
the two fixes. The check covers encoder bytes for 200 configs, and decoder
samples at native and other output rates, with lost packets and hostile
packets. It ran at every commit on the branch. The harness is
`scratchpad/harness-src/main.rs`; run it with `CORPUS_IN=` to decode the
reference packet set.

**Coverage gap.** This machine has AVX2, FMA and AVX-512, so the hash
exercises the AVX2/FMA paths only. The SSE-only and scalar fallbacks get the
same mechanical changes but are not covered by the hash. The NEON paths were
compiled and clippy-checked for `aarch64-apple-darwin` but never run, because
no ARM machine was available.

**Noise.** Timings are pinned to one fast core, under `nice`, interleaved in
rotating order. The machine was shared with other work, so a few runs have
outliers; the spreads in the JSON show them. The A/A comparison (one binary
against itself) is in `aa_noise`. Separately rebuilt binaries also differ by
about ±1% from code layout alone.

**Cycles, not wall time.** End-to-end numbers are user-mode cycles per pass,
minus a 0-pass run that removes setup. They don't depend on the clock
frequency.

**`no_std`.** In `no_std` builds, `f32::mul_add` comes from `libm` (software
FMA). The plain-Rust FMA kernels (comb filter) are only selected there when the
crate is built with AVX and FMA enabled at compile time, so this matters only
for that build. The results are still correct, but slower.

**Criterion `celt_autocorr` and the pitch microbenchmarks** use the `pitch.rs`
kernels, which keep intrinsics. Plain versions were 16–32% slower; see the
`kernels` section of `data.json`.

**Code layout moves single benchmarks.** `silk_burg_modified_fix/d10`
was 14–16% slower in the round-2 builds, though none of the code changed in
that round runs in Burg. A separate pinned A/B of the round-3 build shows
+0.3%. Treat an isolated Criterion jump in code a commit didn't touch as
layout, and confirm it with an A/B before acting on it.

**`celt_fb_st/dec`, "before".** One of its runs is an outlier; the 35% spread
comes from that run. Use the median.

**The exact-energy trick (round 3).** PVQ resynthesis now gets `sum(y^2)` as
an integer. Pulse counts are at most 128, so the sum is at most 16384, far
below 2^24. Every partial sum is then exact in f32 in any order, so the gain
is bit-identical to the old lane-ordered SIMD sum. The old kernels stay only
as a fallback past 2^24, which the codec never reaches.
