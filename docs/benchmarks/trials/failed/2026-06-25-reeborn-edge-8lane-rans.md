# Trial: REEBORN edge — 8-lane interleaved rANS decode

## Status

Failed (hypothesis rejected: more scalar lanes ≠ faster).

## Hypothesis

R152 reached 4-lane interleaved rANS (1.70× ILP over scalar); R153 was decode-bound at e2e
1.05× (ceiling 1.51×). The deferred lever was "8–32 lanes". Test: does going 4 → 8 scalar lanes
lift single-core rANS exponent-decode throughput?

## Scope

- Mode: experimental, REEBORN edge-decode (codec decode throughput)
- REE kernel: rtc-rans-v1 + new `rans_decode_interleaved8_into` (8 lane states in separate locals)
- Artifact: synthetic 40M-symbol exponent stream, realistic bf16 peak (70% on 118–120, 3.147 bits/sym)
- Target/profile: Apple Silicon (arm64, NEON), `cargo test --release`, single core
- Bottleneck tag: CPU arithmetic / register pressure

## Setup

```sh
cargo test -p rtc-codec reeborn_edge_lane_bench -- --ignored --nocapture --release
```

8-lane encode/decode added to `crates/rtc-codec/src/rans.rs`; round-trip verified bit-exact
(`rans_interleaved8_roundtrip_matches`, lengths 1..10003).

## Results

| decode | Gweight/s/core | vs scalar | vs 4-lane |
|---|---:|---:|---:|
| scalar (1 lane) | 0.170 | 1.00× | — |
| interleaved4 | 0.275 | 1.62× | 1.00× |
| **interleaved8** | **0.233** | 1.37× | **0.85× (REGRESSION)** |

8-lane decode stayed bit-exact but ran **15% SLOWER** than 4-lane.

## Analysis

4 lanes already saturate this core's scalar ILP for rANS decode (each lane = a divide-free state
update + a table-gather + a renorm branch). Eight independent lane states + pointers + streams
exceed the integer register file, so the 8-way body spills, and the extra state/stream cache
pressure costs more than the added independent work buys. "More scalar lanes" is a dead end past 4.

The real throughput lever is therefore NOT scalar lane count but one of:
1. **True NEON-SIMD rANS** — vectorize the state update across lanes (uint32x4). Hard part: the
   `slot2sym[x & mask]` lookup needs a gather, which NEON does poorly (vtbl only for ≤ 4×16 B tables).
2. **Coderless decode (REEBORN-FOR, E7)** — the frame-of-reference bit-packed exponent has NO table
   lookup and NO state machine, just fixed-width bit unpacking → branch-free and cleanly NEON-vectorizable.
   Likely the faster-decode path, and the natural partner of the raw-significand design (~0.6 b/w
   ratio cost vs rANS for a much simpler, SIMD-friendly decode).

## Decision

Rejected (8 scalar lanes regress). Keep 4-lane as the scalar rANS sweet spot.

Paper value: useful negative result — the edge throughput win must come from SIMD or a
table-free (FOR/bit-pack) decode, not from widening the scalar interleave.

## Next Experiment

Benchmark REEBORN-FOR exponent decode (coderless fixed-width unpack) vs 4-lane rANS decode, on the
same synthetic stream — expected: lower ratio (~3.2 vs 2.6 bits/exp) but much higher, SIMD-friendly
decode throughput. If confirmed, REEBORN-FOR is the edge codec; then NEON-vectorize it + fuse into
the streaming GEMV.
