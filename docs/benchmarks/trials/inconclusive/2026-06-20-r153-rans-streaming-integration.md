# R153 — REESTREAM-RANS streaming integration: lossless + best ratio, but speed MARGINAL (inconclusive)

- Date: 2026-06-20
- Kernel lineage: REESTREAM-RANS (rtc-rans-v1 R152 wired into REESTREAM-PAR R150a)
- Model: Gemma 3 1B IT lm-head ([262144×1152] bf16, w=6); 8 GB box, >RAM cold, F_NOCACHE
- Verdict: **INCONCLUSIVE** — lossless e2e ✓ and best-in-class ratio (34% < raw bf16,
  at the entropy floor) ✓, BUT the e2e capacity-bound **speed is only 1.05×** (far below
  the 1.51× byte-ratio ceiling, and *below* bit-plane's 1.15×) because the scalar rANS
  decode + reconstruct pipeline is compute-bound under the cold read.

## What was built

- **rtc-rans sidecar** "RLMR" v1 (`rans_stream.rs`): global exponent freq table +
  per-block 4-lane interleaved-rANS exponent stream + raw residual plane + a per-block
  byte-length table. `write_lmhead_sidecar_rans` / `build_rans_sidecar` /
  `read_rans_header` / `stream_lmhead_from_rans_sidecar`.
- **`streaming_rans_gemv_parallel`**: REESTREAM-PAR block partition; each worker seeks
  to its block range, reads + rANS-decodes the exponent lanes + reconstructs bf16
  (exp+residual) + dots. Tables built once per thread; read/exp buffers reused.
- `rans_decode_interleaved4_into` + `RansDecodeTables`: precomputed cum/slot tables, no
  per-call alloc (the per-block table rebuild was the first inefficiency found).

## Results

**Correctness — GO (lossless):**
- Synthetic parity (`rans_sidecar_streams_equal_to_reference`): streamed == resident
  bf16 GEMV bit-for-bit.
- Real Gemma (`r153_gemma_rans_lmhead_lossless`, #[ignore]): **262144 logits
  bit-identical** to resident. rtc-codec 52 / rllm-runtime lib 296, 0 warnings.

**Ratio — GO (best in class):** rANS sidecar = ~10.57 bits/weight (at the R151 floor)
⇒ **34% fewer bytes than raw bf16** (bit-plane was 12% for this w=6 tensor). Real
*capacity* win — fits 34% more model losslessly.

**Speed — MARGINAL (the inconclusive part):** `r153_rans_capacity_bound` (>RAM cold,
fair parallel-raw baseline, k=20 replicas, 6 cores):
```
raw bf16  parallel  12.08 GB -> 6958 ms  (1.74 GB/s)
rANS      parallel   7.98 GB -> 6597 ms  (1.21 GB/s, decode NOT fully hidden)
bytes: 34% fewer   SPEEDUP vs raw: 1.05x   (bit-plane R150a was 1.15x)
```
(The per-block table-reuse fix moved it 0.99x → 1.05x; still far below 1.51x.)

## Analysis — honest

- **The byte savings are wasted on slow decode.** rANS reads 34% fewer bytes but
  processes at only **1.21 GB/s** vs the raw cold read's **1.74 GB/s** → the path is
  compute-bound, not read-bound. The 34% smaller stream does not convert to speed.
- **rANS is a speed REGRESSION vs bit-plane (1.05x < 1.15x) despite a far better
  ratio.** R152's scout (decode-only, 1.9 Gw/s aggregate) was optimistic: the full e2e
  pipeline adds (a) the scalar exp+residual→bf16 reconstruct (join_bf16 per weight),
  (b) the dot, and (c) within each worker the read and the decode are SERIAL (not
  double-buffered), so per-thread time ≈ read + decode (additive), like R150a's nt=1.
  Bit-plane wins the e2e because its NEON tbl-gather decode keeps up with the read;
  rANS's scalar decode cannot.
- **This is the R143→R144 pattern again:** isolated decode throughput is an optimistic
  proxy; the full reconstruct+dot pipeline is the real wall.

## Decision

**INCONCLUSIVE.** The rANS streaming path is *correct and lossless* and delivers the
*best lossless ratio* (a real 34% capacity win over raw bf16), but the e2e *speed* in
the capacity-bound regime is marginal (1.05×) and currently worse than bit-plane —
the scalar decode+reconstruct pipeline is the bottleneck. The ratio/capacity result is
solid; the speed thesis for rANS is not yet proven.

## Next (R154 — make the decode pipeline keep up with the read)

1. **NEON-vectorize the reconstruct** (exp+residual→bf16): the scalar `join_bf16` loop
   over block_rows×hidden weights/block is a big additive cost.
2. **Double-buffer read+decode within each worker** (R148-style): overlap the cold read
   of block N+1 with the decode of block N so per-thread time → max(read, decode), not
   sum. Likely the biggest single lever (rANS decode ≈ read, so overlap ~halves it).
3. **Fuse reconstruct into the dot** (no full bf16 materialization).
4. 8-way interleaved rANS for more decode ILP.
Ceiling: the 1.51× byte ratio. If the pipeline reaches read-bound, rANS's 34% savings
should beat bit-plane's 12% (1.51× vs 1.15×).

## Verification status

- [x] Synthetic + real Gemma lossless (262144 logits identical).
- [x] Ratio 34% < raw bf16 (at the entropy floor).
- [~] Speed 1.05× (MARGINAL, below ceiling and below bit-plane) — pipeline compute-bound.
- [x] rtc-codec 52 / rllm-runtime lib 296, 0 warnings.
