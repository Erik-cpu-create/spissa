# Spec: R149c — capacity-bound speed demo on a real model's lm-head (cold)

Date: 2026-06-20
Status: **DONE (NO-GO, honest negative)** — see `docs/benchmarks/trials/failed/2026-06-20-r149c-capacity-bound-real-lmhead.md`. With a fair pipelined-raw baseline, compression is 0.71–0.83× (slower) on fast SSD: decode (~8 GB/s) < pipelined read (~12 GB/s). RAM win holds; speed win does not.
REE kernel: REESTREAM (R148) + REEPLANE-W6 (R149b), reused — no new kernel

## Honest positioning (read first)

R143→R149b built and proved the streaming lm-head **lossless** (R149b: Gemma w=6,
262144 logits identical, tokens identical). The **speed** win — lossless compression
beating raw bf16 — has only ever been shown on a **synthetic replicated** >RAM file
(R148: 1.32× cold). R149c proves it on the **real model's lm-head**, read **cold**,
through the **real production streamer** (`write_lmhead_sidecar` →
`stream_lmhead_from_sidecar`), vs a cold raw-bf16 read+GEMV baseline.

The dev box has **8 GB RAM**, so cold I/O is forced with `F_NOCACHE` (the R147/R148
method) rather than needing a literally-larger-than-RAM file — F_NOCACHE makes each
read miss the page cache, reproducing the capacity-bound regime exactly.

**Expected honest outcome:** streaming the lm-head cold should win *in isolation*
(fewer bytes + decode pipelined under the read). End-to-end generation tok/s may be
**neutral or slightly worse**, because only the lm-head streams cold while the
transformer body stays resident/warm — which is the finding that motivates R150
(stream the projections). Both results are valid and reported honestly; a NO-GO or
MARGINAL is not fudged.

## Goal

1. A `#[ignore]` bench (`r149c_real_lmhead_capacity_bound`) that, on the real Gemma
   3 1B lm-head ([262144×1152] bf16, w=6):
   - writes the bit-plane sidecar (v2, ~504 MB) and dumps the raw bf16 lm-head
     (~604 MB) to /tmp;
   - times a **cold** (`F_NOCACHE`) raw-bf16 read (the strongest baseline — give raw
     zero compute, per R148) vs a **cold** pipelined `streaming_bitplane_gemv`
     (read + decode + dot, decode hidden under the read);
   - asserts the streamed logits equal the resident bf16 GEMV (lossless, already
     proven) and reports raw/comp GB, ms, GB/s, speedup, RAM delta, verdict.
2. A capacity-bound runtime knob: `RLLM_STREAM_NOCACHE=1` makes
   `stream_lmhead_from_sidecar` read with `F_NOCACHE`, so the real generation path
   can run in the capacity-bound regime without thrashing the page cache. Opt-in,
   default unchanged.

## Design

- **`stream_lmhead_from_sidecar`** (`bitplane_stream.rs`): read `RLLM_STREAM_NOCACHE`
  (truthy = "1"/"true"); pass it as `nocache` to `streaming_bitplane_gemv`. Default
  false → byte-identical behavior to today.
- **Bench** (`bitplane_stream.rs` tests, `#[ignore]`, aarch64):
  - `write_lmhead_sidecar(gemma-3-1b-it-rawcodec.spsa, embed, 256, sidecar)`.
  - dump raw bf16 via `with_raw_tensor(..|b| b.to_vec())` → `/tmp/gemma1b-lmhead-raw.bin`.
  - raw cold: `F_NOCACHE` open, `read_exact` the 604 MB, `black_box` (zero compute —
    the conservative baseline; a real raw path would add the dot on top).
  - comp cold: parse the sidecar header, `streaming_bitplane_gemv(.., nocache=true, ..)`
    over all blocks (read + REEPLANE-W6 decode + bf16 dot, pipelined).
  - lossless gate: streamed logits == `lm_head_logits_parallel_bf16` resident.
  - report GB, ms, GB/s, speedup (raw_ms/comp_ms), RAM (comp 504 vs raw 604 MB),
    verdict (GO if comp_ms < raw_ms).

## Testing / acceptance

- Lossless gate inside the bench (exact `assert_eq!` on the real logits).
- Verdict reported honestly: GO (comp faster cold), MARGINAL, or NO-GO. A negative
  result is a valid recorded outcome.
- Existing suites stay green; `RLLM_STREAM_NOCACHE` is opt-in, default off.

## Non-goals

- Streaming the transformer projections (R150) — this is lm-head only.
- End-to-end generation tok/s improvement (expected neutral until the body streams);
  if measured, reported as honest context, not as the headline.
- New codec/container format (`codec_for_id`); the sidecar stays a separate file.

## Originality & doctrine

Reuses R148 REESTREAM + R149b REEPLANE-W6 + `bf16_row_dot_f32`; the cold real-lm-head
bench + the `RLLM_STREAM_NOCACHE` knob are new. No new dependencies. Lossless by
default preserved. Honest metrics: cold-I/O methodology (F_NOCACHE) stated; raw given
the zero-compute benefit so a win is conservative.
