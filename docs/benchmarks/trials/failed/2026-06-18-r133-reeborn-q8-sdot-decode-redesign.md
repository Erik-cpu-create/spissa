# Trial: R133 REEBORN-Q8-SDOT decode fast-path (contiguous + row-parallel)

Date: 2026-06-18
Owner: RLLM
Status: rejected (no decode speedup — decode is memory-bound, not compute-bound)
Folder: failed

## Hypothesis

llama.cpp/ggml CPU q8 decode is fast because of structure, not a single trick:
it (1) ALWAYS quantizes the activation to int8 and uses `sdot`/`i8mm` (never
int8-weight × f32-activation), (2) quantizes the activation ONCE per matmul,
(3) reads each weight tensor as one contiguous mmap region, (4) parallelizes
across output rows with a persistent pool. RLLM had the pieces (sdot R130,
act-cache R127, row-parallel R132) but wired them in a per-chunk streaming
structure that defeats them: the DEFAULT decode used scalar `q8_0_dot_i8_f32`
(i8×f32, ~0.57 GMAC/s, ~100× below int8 peak), the int8-sdot path was
single-threaded, and each weight tensor was dispatched chunk-by-chunk
(~3366 `with_raw_chunk` calls/token). Restructuring the raw-codec q8 batch=1
decode to ggml's shape — one contiguous tensor view + quantize-once + int8 sdot
row-parallel — should close the bulk of the ~100× CPU decode gap vs llama.cpp,
without abandoning the chunk machinery (kept for compressed/low-RAM/integrity).

## Scope

- Mode: exact-lowram runtime (q8, near-exact int8 activation — quant-only diff, same as llama.cpp q8 inference)
- REE kernel: REEBORN-Q8-SDOT-DECODE (whole-tensor int8 sdot, row-parallel; name pending Erik's confirmation)
- Model/artifact: `models/gemma-3-4b-it-q8.rllm` (q8_transformer_keep_io, codec rtc-raw-v1)
- Architecture: Gemma 3 4B, Q8_0
- Target device/profile: Apple Silicon, CPU only
- Bottleneck tag: CPU arithmetic (scalar→sdot) + per-chunk dispatch structure

## Setup

New fast-path gated on `RLLM_Q8_ACTIVATION=1` + `dtype==Q8_0` + `batch==1`
(`streaming/linear.rs`): `LazyRllmModel::with_raw_tensor` (lazy.rs) returns the
whole tensor's contiguous raw chunks as ONE zero-copy mmap slice (`reader.read_span`
in rllm-container), integrity-checked once via `verify_tensor_checksum` (honors
integrity mode; mismatch → falls back to the chunk path). The slice feeds
`accumulate_q8_0_full_tensor_int8_batch1` (kernels.rs): quantize the activation
to int8 ONCE (`quantize_input_q8_blocks`), then `sdot_int8_batch1_rows_range`
across output rows split over `std::thread::scope` workers (4-row ILP via the
R130 `batch1_x4_ilp`, per-row `i8_dot32` remainder). The per-chunk path remains
the fallback for compressed codecs / low-RAM / strict integrity / non-aarch64 /
batch>1 prefill. Default behavior (env unset) unchanged.

## Results

- **Correctness:** preserved. `RLLM_Q8_ACTIVATION=1` on the real q8 model →
  "The capital of France is" → "Paris. Paris is a" (token ids unchanged).
- **Engagement:** the fast-path fires for ALL projections (q/k/v/o/gate/up/down,
  every layer) — `with_raw_tensor` confirmed `handled=YES`, i.e. the tensor's
  rtc-raw-v1 chunks ARE contiguous in the mmap (validated empirically).
- **Parity:** `r133_full_tensor_int8_batch1_matches_per_row_reference` —
  bit-identical to the per-row int8 reference (incl. non-÷4 rows + parallel
  split). Quant-only diff vs the f32-exact path (inherent to int8 activations,
  same as llama.cpp).
- **Decode speed (COOL machine, warm, steady-state, 12-token A/B): NO SPEEDUP.**
  scalar avg ~5.87s/token (steps 1–11) vs fast-path avg ~5.74s/token (steps 2–11;
  step 1 = 14s cold-cache outlier). Overall 0.12 tok/s both. `RLLM_THREADS=8` made
  no difference (~5.77s). Per-step instrumentation: the 7 projections = **99.8%**
  of the layer-loop time, and the run is mostly user-CPU-time (20s user / 4s system)
  — yet the matmul runs at **~0.56 GMAC/s = identical to scalar**, with 8 cores
  giving nothing.
- **Tests:** full runtime suite green (280 passed, 1 ignored).

## Analysis

The fast-path is correct, engages, and preserves output — but it does NOT speed up
decode, because **decode is memory-bound, not compute-bound.** Batch=1 decode has
arithmetic intensity 1 (each weight byte is used once), so the sdot/i8mm kernel and
8-core parallelism cannot help — 8 cores all stall waiting on the weight stream.
(Prefill got 1.6× from the same int8/i8mm precisely because batch>1 reuses each
weight byte → compute-bound.) This OVERTURNS the earlier "compute-bound / scalar
dot is the bottleneck" conclusion (which was thermally confounded). The ~0.56 GB/s
effective weight-read rate is ~90× below RAM bandwidth and far below llama.cpp on
the same weights/RAM — the prime suspect is residency: RLLM's working set (4.75 GB
q8 model + a **2.68 GB f32-decoded embedding** held resident for the tied LM head)
likely evicts the q8 weights, forcing slow re-reads, whereas llama.cpp keeps the
embedding quantized and fits. The lossless/streaming differentiator is intact
(chunk path kept; fast-path opt-in; default unchanged), and the fast-path code is
retained behind `RLLM_Q8_ACTIVATION=1` (harmless; may help once decode is no longer
memory-bound).

## Decision

failed (useful negative)

The compute lever does not move decode. Value: definitively re-classifies decode as
**memory-bound** (8-core sdot ≡ scalar), correcting the thermal-confounded
"compute-bound" read. Next lever shifts to the memory/residency axis — first kill
the 2.68 GB f32 embedding residency (keep it q8/bf16 + dequant-on-the-fly for lookup
+ LM head) so the q8 weights stay resident, then re-measure decode.
