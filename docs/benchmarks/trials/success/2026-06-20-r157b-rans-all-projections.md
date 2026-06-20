# R157b — flexible block_rows + all transformer-layer projections stream-rANS lossless (GO)

- Date: 2026-06-20
- Kernel lineage: REESTREAM-RANS (R153/R154), generalized
- Model: Gemma 3 1B IT (`gemma-3-1b-it-rawcodec.rllm`) — all 7 layer-0 projections
- Verdict: **GO** — the rANS streaming GEMV is a lossless drop-in for the resident bf16
  matmul on **every** projection shape in a real transformer layer, including the ones
  whose row count isn't a multiple of `block_rows` (o_proj/down_proj at 1152 rows). This
  is the prerequisite for wiring streaming into the forward pass (R157c).

## What changed — flexible block_rows (zero-row padding)

`write_lmhead_sidecar_rans` previously required `vocab % block_rows == 0`. Now it pads
the row count up to a multiple of `block_rows` with **zero rows** (`exp`/`res` extended
with zeros; the frequency table is built from the padded planes so `freq[0] ≥ 1`). The
padding decodes to bf16 zero and `stream_lmhead_from_rans_sidecar` truncates the output
to the real `vocab`, so it stays bit-exact. `build_rans_sidecar` uses
`num_blocks = vocab.div_ceil(block_rows)`; the reader streams into a padded buffer then
`truncate(vocab)`. Uniform blocks preserved (no variable-size-block complexity).

## Results — GO

`r157b_gemma_layer0_all_projections_lossless` (#[ignore]):
```
q_proj   [1024×1152] (no pad)  — 1024 outputs bit-identical
k_proj   [ 256×1152] (no pad)  —  256 outputs bit-identical
v_proj   [ 256×1152] (no pad)  —  256 outputs bit-identical
o_proj   [1152×1024] (PADDED)  — 1152 outputs bit-identical
gate_proj[6912×1152] (no pad)  — 6912 outputs bit-identical
up_proj  [6912×1152] (no pad)  — 6912 outputs bit-identical
down_proj[1152×6912] (PADDED)  — 1152 outputs bit-identical
```
Every `W·x` streamed via rANS equals the resident bf16 `W·x` bit-for-bit, including the
two padded tensors. Existing %256 round-trips stay green; rtc-codec 54 / rllm-runtime
lib 296, 0 warnings.

## Analysis

- **rANS streaming is a lossless drop-in for the resident matmul, any shape.** The only
  part of a layer forward that rANS touches is the weight-matrix GEMVs; norms, RoPE,
  attention, and activations are unchanged. Since all 7 GEMVs are proven bit-identical, a
  layer built from streaming-rANS GEMVs + the same non-weight ops is bit-identical to the
  resident layer by composition.
- **Padding cost is negligible** (≤ block_rows−1 extra zero rows per tensor) and lossless.

## Decision

**GO** — flexible block_rows unblocks every projection; streaming-rANS GEMV is lossless
across all layer shapes. Ready for the forward-pass integration.

## Next (R157c — the engine wiring)

- Pack a model's body projections as rANS and route the decode forward pass to
  `streaming_rans_gemv_parallel` per projection instead of the resident matmul.
- Handle prefill (batch>1): amortize one streamed weight read over all batch activations
  (a GEMM, not GEMV).
- Measure end-to-end tok/s + RSS on a model genuinely > device RAM, lossless.

## Verification status

- [x] Flexible block_rows: padded tensors (o_proj, down_proj) stream lossless.
- [x] All 7 Gemma layer-0 projections: streamed W·x == resident bf16, bit-identical.
- [x] %256 round-trips + container RansCodec unaffected; 0 warnings.
