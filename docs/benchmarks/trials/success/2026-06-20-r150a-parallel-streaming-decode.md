# R150a — REESTREAM-PAR parallel streaming decode: capacity-bound speed win (GO)

- Date: 2026-06-20
- Kernel lineage: **REESTREAM-PAR** (parallel variant of R148 REESTREAM) + REEPLANE-W6 (R149b)
- Model: Gemma 3 1B IT lm-head ([262144×1152] bf16, w=6); 8 GB box, **>RAM cold** files + F_NOCACHE
- Verdict: **GO** — in the genuine capacity-bound regime, parallel-decoded bit-plane
  streaming beats a *fair* parallel-raw cold read by **1.15×** (= the 12% byte ratio),
  lossless. This is the win R149c's single-consumer path could not reach.

## Hypothesis

R149c showed the single-consumer streaming lm-head is **decode-bound** (~8 GB/s) and
loses. With **parallel decode** (N threads), aggregate decode rises above the read
bandwidth → the path becomes **read-bound** → compression wins by the byte ratio
(~1.14×).

## Method

- **REESTREAM-PAR** (`streaming_bitplane_gemv_parallel`): partition blocks across
  `n_threads` workers; each opens its own handle, seeks to its block range, reads +
  decodes (`decode_bitplane_row_into`) + dots its blocks into a disjoint `out` slice
  (`split_at_mut`, no unsafe, no channels). Shared `decode_dot_block` helper with the
  R148 single-consumer kernel. `stream_lmhead_from_sidecar` routes here with
  `n_threads = RLLM_STREAM_THREADS` (default = cores).
- **Lossless parity** (`streaming_parallel_matches_single_thread`): parallel
  (n ∈ {1,2,4}) == single-thread decode+dot, bit-for-bit, w=5 and w=6.
- **Capacity-bound bench** (`r150a_parallel_lmhead_capacity_bound`, #[ignore]): the
  REAL lm-head, replicated to **>8 GB** (raw 9.66 GB, comp 8.46 GB) so reads are
  genuinely cold (a 600 MB file fits 8 GB RAM → F_NOCACHE-variance noise, R149c's
  flaw). Three cold baselines + honest decomposition; lossless gate inside.

## Results — GO

```
lossless gate: OK (262144 logits identical)
raw bf16  1-reader   9.66 GB -> 7256 ms  (1.33 GB/s)
raw bf16  6-reader   9.66 GB -> 5568 ms  (1.74 GB/s, FAIR baseline)
bit-plane nt=1       8.46 GB -> 7712 ms  (1.10 GB/s)
bit-plane nt=6       8.46 GB -> 4859 ms  (1.74 GB/s)
bytes: 12% fewer
compression effect (comp vs parallel-raw): 1.15x   <- the honest win
concurrent-read effect (1->6 readers):     1.30x
VERDICT: GO
```

- **Parity:** parallel == single-thread, bit-for-bit, all thread counts and widths.
- **Real-Gemma lossless still holds with the parallel default:**
  `r149b_gemma_streaming_lmhead_lossless` (now exercising the parallel path) green —
  262144 logits identical to resident. Suites: rtc-codec 48 / rllm-runtime lib 295,
  0 warnings.

## Analysis (honest decomposition)

- **The byte ratio, recovered cleanly.** At 6 readers both raw and comp saturate the
  SSD at ~1.74 GB/s; comp wins **1.15×** purely by reading 12% fewer bytes — the
  decode is fully hidden by parallelism. This is exactly the #2 ceiling
  ("read-bound ⇒ wins by the byte ratio"), measured.
- **Parallel decode is the flip.** nt=1 bit-plane (7712 ms, 1.10 GB/s) is *slower*
  than even 1-reader raw — decode is additive without overlap (confirms R149c). nt=6
  (4859 ms) is the win. The lever is the parallelism, not the codec alone.
- **Two effects kept separate (lesson from R149c, applied):** the FAIR baseline is
  parallel-raw (same 6 readers), so the reported 1.15× is compression *only*. The
  separate concurrent-read effect (1→6 readers, 1.30×) is not credited to compression.
  Comparing comp-nt6 to 1-reader raw would have shown a misleading ~1.49×.
- **Regime matters.** This holds where reads are the wall (>RAM, cold ~1.3–1.7 GB/s
  SSD). On a fits-in-RAM file (R149c) F_NOCACHE variance swamps the signal and there
  is no capacity-bound win to measure. The honest ceiling is the byte ratio (~1.14×
  for bit-plane's 13 bits/weight); a higher-ratio codec with decode ≥ read bandwidth
  would widen it.

## Decision

**GO** — parallel streaming decode (REESTREAM-PAR) turns R149c's NO-GO into a clean
1.15× capacity-bound win, lossless, properly isolated from the concurrent-read effect.
The streaming path is now read-bound (decode no longer the bottleneck), which is the
prerequisite for R150b. `RLLM_STREAM_THREADS` knob added; parallel is the default for
the opt-in `RLLM_STREAM_LMHEAD` path.

## Next (R150b)

Stream the **transformer projections** (attention + MLP weight matrices), not just
the lm-head — the body is the bulk of per-token bytes. With decode now read-bound,
the whole-model capacity-bound win (run a model > RAM, ~byte-ratio faster + lossless)
becomes reachable. Wire pack → block-framed projections → REESTREAM-PAR in the
forward pass; correctness gate per projection; measure tok/s on a model > device RAM.

## Verification status

- [x] Parallel == single-thread, bit-for-bit (w=5, w=6; n ∈ {1,2,4}).
- [x] Real Gemma streaming lm-head == resident with parallel default (262144 identical).
- [x] >RAM cold bench, fair parallel-raw baseline, compression-only 1.15× (= byte ratio).
- [x] rtc-codec 48 / rllm-runtime lib 295 green; 0 warnings.
