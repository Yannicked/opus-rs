# Safety work: harness and benchmarks (parked)

Kept with the `safety/fewer-unsafe` branch so the work can be picked up
later. None of it is part of the crate.

- `harness/`: hashes encoder bytes (200 configurations) and decoder
  samples (with losses, other output rates, and 160,000 mutated or random
  packets), to prove each commit is bit-identical to the reference.
- `kernel-bench/`: per-kernel microbenchmarks (renormalise, haar1,
  alg_unquant, inner products, MDCT, band energies).
- `scripts/`: pinned-core A/B timing (`kb.sh` kernels, `cb.sh` Criterion,
  `cyc.sh`/`hbench.sh` end to end with `perf stat`), and the build and
  check helpers. They expect a scratch directory in `$SCRATCH`
  (default `/tmp/opus-rs-bench`) holding one build per version.
- `pulse-search-repro/`: the AVX2 pulse search writing past `y` on an
  all-zero band (fixed on this branch).
- `report/`: `data.json` (before, safety, round 2, after) and `notes.md`
  with the caveats, as of head `023043c`.

To resume, rebase onto upstream (or onto the FEC and concealment PRs,
restsend/opus-rs#67 and #68). Expect conflicts mainly from the repo-wide
`cargo fmt` commit: drop it and re-run `cargo fmt` after the rebase. The
new `src/celt/plc.rs` from #68 needs the same audit.
