# R149c — capacity-bound speed on the REAL Gemma lm-head: NO-GO (honest negative)

- Date: 2026-06-20
- Kernel lineage: REESTREAM (R148) + REEPLANE-W6 (R149b), reused
- Model: Gemma 3 1B IT (`gemma-3-1b-it-rawcodec.spsa`; lm-head [262144×1152] bf16, w=6)
- Device: Apple aarch64, **8 GB RAM**, fast internal NVMe; cold I/O via `F_NOCACHE`
- Verdict: **NO-GO (speed)** — with a *fair* pipelined-raw baseline, streaming the
  compressed lm-head is **0.71–0.83× (slower)** than reading raw bf16. The earlier
  "2.06×" was a measurement artifact (pipeline vs naive blocking read), now corrected.

## Hypothesis

On a real model's lm-head read cold, a pipelined bit-plane stream (read + REEPLANE-W6
decode + dot, ~12% fewer bytes) beats a cold raw-bf16 read — validating the
R143→R149 capacity-bound thesis on real weights through the real streamer.

## Method

`#[ignore]` bench `r149c_real_lmhead_capacity_bound`. Writes the real lm-head sidecar
(528 MB) + dumps raw bf16 (604 MB) to /tmp. Times three cold (`F_NOCACHE`) reads,
5 iters each, with a **lossless gate** (streamed == resident, 262144 logits identical):

1. raw bf16, **naive single blocking** `read_exact` (zero compute);
2. raw bf16, **pipelined** double-buffered reader, same block cadence, zero compute
   — the FAIR baseline (a real raw runtime pipelines its reads too);
3. bit-plane, **pipelined** `streaming_bitplane_gemv` (read + decode + dot).

The decomposition isolates the compression contribution (3 vs 2) from the pipeline
contribution (2 vs 1).

## Results (two runs, stable)

```
raw bf16   single-read  0.604 GB -> 127–133 ms  (4.5–4.8 GB/s, naive blocking)
raw bf16   PIPELINED    0.604 GB ->  49–56 ms   (10.8–12.4 GB/s, fair baseline)
bit-plane  PIPELINED    0.528 GB ->  67–68 ms   (7.8–7.9 GB/s, decode+dot pipelined)
bytes: 12% fewer (0.875 ratio)
compression effect (comp vs pipelined-raw): 0.71–0.83x   <- SLOWER
pipeline effect    (single -> pipelined raw): 2.4–2.6x
total vs naive raw: 1.9–2.0x
VERDICT: NO-GO (pipelined raw cold read still faster)
lossless gate: OK (262144 logits identical)
```

## Analysis (first-principles)

- **The decode is the wall, again.** The bit-plane consumer is decode-bound at
  ~7.8 GB/s; the pipelined raw reader sustains ~11–12 GB/s. With pipelining, total
  time ≈ `max(read_time, decode_time)`. Here `decode (67 ms) > raw_read (49–56 ms)`,
  so the 12% byte savings cannot win — decode is slower than the read it tries to
  hide under. This is the R144/R145 decode-throughput wall, now at the I/O layer.
- **Crossover condition.** Compression wins on speed iff `max(comp_read, decode) <
  raw_read`, i.e. only when **storage bandwidth < decode bandwidth** (slow disk,
  network/cloud-cold storage) OR decode is made faster/parallel. On fast local NVMe,
  read (~12 GB/s) > decode (~8 GB/s) → raw wins. Even in the best case the ceiling is
  the byte-savings ratio (~1.14× for bit-plane), achievable only if decode is fully
  hidden.
- **Revises R147/R148.** Those reported a GO (1.13×/1.32×) but their raw baseline was
  a **single-threaded** `read_exact` loop (verified in `streaming_gemv_capacity_bound_bench`),
  not pipelined — so their "win" was substantially the pipeline, not compression.
  R149c's fair baseline shows the compression-only contribution is **negative** on
  this hardware. A re-run of R147/R148 with a pipelined-raw baseline is the honest
  follow-up. (R148's regime was genuinely >RAM replicated data vs R149c's F_NOCACHE
  on 600 MB; the methodological point stands regardless.)
- **What still holds:** the bit-plane lm-head is **lossless** (262144 logits identical)
  and **12% smaller** — a real *capacity/RAM* win (run a model that wouldn't otherwise
  fit), just not a *speed* win on fast SSD.

## Decision

**NO-GO for speed** on fast local storage with a fair baseline. Recorded honestly;
not fudged into a GO by comparing against a naive baseline. The streaming lm-head
remains valuable for **lossless capacity** (R149a/R149b), and the `RLLM_STREAM_NOCACHE`
knob + this bench harness are kept for the regimes/levers below.

## Next (candidate levers, in priority order)

1. **Parallel streaming decode (R150).** Multi-thread the consumer so aggregate
   decode bandwidth > read bandwidth; the path then becomes read-bound and wins by
   the byte ratio (~1.14×). Modest but real, and the prerequisite for whole-model.
2. **Slow-storage / cold-cloud regime.** Where read bandwidth < decode (~8 GB/s),
   compression wins materially — measure on a throttled/network device to confirm.
3. **Higher-ratio codec** with decode still ≥ read bandwidth (more bytes saved per
   decode) — the dfloat Huffman ratio (33%) is bigger but its decode is far slower
   (R142), so net-negative here.

## Verification status

- [x] Lossless gate inside the bench (262144 logits identical).
- [x] Fair pipelined-raw baseline; compression-only effect isolated (0.71–0.83×).
- [x] Reproducible across runs (NO-GO stable).
- [x] rtc-codec 48 / rllm-runtime lib 294 green; 0 warnings.
