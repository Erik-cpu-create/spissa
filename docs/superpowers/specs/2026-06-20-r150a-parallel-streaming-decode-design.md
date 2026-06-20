# Spec: R150a — parallel streaming bit-plane decode (REESTREAM-PAR)

Date: 2026-06-20
Status: **DONE (GO)** — see `docs/benchmarks/trials/success/2026-06-20-r150a-parallel-streaming-decode.md`. Parallel decode makes the streaming lm-head read-bound; compression-only win 1.15× (= byte ratio) vs a fair parallel-raw baseline, >RAM cold, lossless.
REE kernel: **REESTREAM-PAR** (parallel variant of R148 REESTREAM)

## Honest positioning (read first)

R149c's fair baseline found the single-consumer streaming lm-head is **decode-bound**
(~8 GB/s) and loses to a pipelined-raw cold read (~12 GB/s) on fast NVMe → NO-GO on
speed. The crossover: compression wins iff `max(comp_read, decode) < raw_read`. With
**parallel decode** (N consumer threads), aggregate decode rises above the read
bandwidth, the path becomes **read-bound**, and it wins by the byte ratio (~1.14× for
bit-plane's 12% fewer bytes). R150a builds that parallel kernel and re-runs the fair
bench. This is also the prerequisite for R150b (whole-model projection streaming):
the body must decode fast enough to keep up with the read.

**Expected honest ceiling:** the win is bounded by the byte-savings ratio (~1.14×),
and only materializes if (a) N-thread decode aggregate > read bandwidth and (b) the
reads themselves scale (concurrent F_NOCACHE reads). If the single-thread read can't
be beaten and decode already nearly matches it, the result may be MARGINAL — reported
honestly either way.

## Design

Partition the blocks across `n_threads` worker threads (embarrassingly parallel —
no channels, no mutex). Each thread opens its own file handle, seeks to its block
range, and sequentially reads + decodes + dots its blocks into a disjoint output
slice. Across threads, one thread's decode overlaps another's read; concurrent
F_NOCACHE reads can also exceed single-thread read bandwidth on NVMe.

- **`decode_dot_block`** (shared helper): decode + bf16-dot one block buffer
  (`[B×index ++ B×residual]`) into a `block_rows`-length output slice. Used by both
  the R148 single-consumer kernel and the new parallel one (removes duplication).
- **`streaming_bitplane_gemv_parallel(.., n_threads)`**: `std::thread::scope`;
  `out.split_at_mut` gives each worker a disjoint range (safe, no unsafe). Worker i
  seeks to `data_offset + blk_start*block_bytes`, reads `nblk` blocks, decode_dot each.
  `n_threads` clamped to `[1, num_blocks]`. Bit-identical to the single-thread
  decode+dot (each row is independent and deterministic; only thread assignment
  changes, not values).
- **`stream_lmhead_from_sidecar`** routes to the parallel kernel with
  `n_threads = RLLM_STREAM_THREADS` (default `available_parallelism`). The R148
  single-consumer `streaming_bitplane_gemv` is kept for its bench + as the n=1
  reference.

## Testing (TDD gates)

1. **Lossless parity:** `streaming_parallel_matches_single_thread` — parallel output
   (n_threads ∈ {1,2,4}) == single-thread `decode_neon`/dispatcher + dot reference,
   bit-for-bit, for w=5 and w=6 synthetic fixtures.
2. **Re-confirm R149b gate:** real Gemma identical-token generation still holds with
   the now-parallel default path (parallel is bit-identical, but verify honestly).
3. **Bench `r150a_parallel_lmhead_capacity_bound`:** real Gemma lm-head, cold
   (F_NOCACHE), fair pipelined-raw baseline vs parallel bit-plane (n_threads sweep
   {1,2,4,cores}). Report GB/s, speedup vs fair raw, verdict (GO if parallel bit-plane
   < fair pipelined raw). Lossless gate inside.
4. Existing suites green; default change verified bit-identical via (1)+(2).

## Non-goals

- Streaming the transformer projections (R150b — the next phase, gated on R150a GO).
- Parallel/async multi-reader beyond one handle per worker thread.
- A new container codec.

## Doctrine

Reuses R148 streaming + R149b REEPLANE-W6 decode + `bf16_row_dot_f32`. No new deps.
Lossless by default (parity-proven). Honest metrics: MARGINAL/NO-GO reported as-is;
the ~1.14× byte-ratio ceiling stated up front, not inflated.
