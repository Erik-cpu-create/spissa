# R110: REEDOT Q8 Int8 SDOT Lab

Date: 2026-06-17
Owner: RLLM
Status: accepted lab
Folder: success

## Hypothesis

R109 suggested removing the per-chunk consumed-block allocation as R110. This
trial pivots instead. A single-thread Ollama re-baseline on the same device and
model showed an ~170x prefill and ~64x decode gap (see Evidence Inputs), and
R78–R109 (30+ stages) only moved Q8 prefill from `26.75s` to best `7.76s` while
every f32 NEON lab kernel plateaued at `4.6–5.3x` over the f32 baseline. The gap
is too large for further f32 micro-optimization. R110 tests whether a native ARM
`sdot` (int8×int8 → int32) kernel beats the existing f32 NEON kernels in the lab,
because RLLM currently has zero integer-SIMD matmul (it dequantizes Q8 weights to
f32 and does f32 FMA, while llama.cpp/ggml uses `sdot`/`smmla`).

## Scope

- Mode: exact-lowram lab
- REE kernel lineage: `REEDOT-LAB`
- Artifact shape: synthetic Llama 3.2 1B-like Q8 row
- Batch: 55
- Input features: 2048
- Blocks per row: 64
- Bottleneck tag: CPU arithmetic / Q8 integer-SIMD dot
- Implementation: int8 `sdot` via inline asm (the `vdotq_s32` intrinsic is still
  nightly-gated `stdarch_neon_dotprod`; `sdot` is usable on stable through `asm!`
  + `target_feature(enable = "dotprod")`, guarded by `is_aarch64_feature_detected`).
  Activations pre-quantized to int8 per row (amortized across out_features in a
  real GEMM), so only the int8 dot is timed.

Runtime streaming code was not changed in R110.

## Evidence Inputs

Single-thread Ollama re-baseline (Apple A18 Pro, 6 core, 8GB, `llama3.2:1b`
Q8_0, `num_thread:1`, prompt `Answer yes or no: is fire cold?`):

- Ollama prefill ~`1200 tok/s` (1283/1213/1124), decode ~`51 tok/s` (52.88/51.31/51.08)
- RLLM R109 prefill `7.76s` / 55 tokens ≈ `7 tok/s`, decode `0.8 tok/s`, max RSS 1.6 GB
- Gap: ~`170x` prefill, ~`64x` decode (no RAM win in this config either)

Code audit: `grep vdotq|vmmla|i8mm|int8x16` across `crates/rllm-runtime/src` is
empty; the "NEON q8 kernel" converts int8→f32 (`vcvtq_f32_s32`+`vmulq_f32`) then
does f32 FMA.

## Setup

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench --json target/reedot-lab/r110.json --markdown target/reedot-lab/r110.md --batch 55 --in-features 2048 --iters 3000
```

## Results

Lab (batch 55, in_features 2048, iters 3000), speedup vs `baseline_i8_dot32_batch4`:

| variant | elapsed ns | speedup vs baseline | max abs diff |
|---|---:|---:|---:|
| `baseline_i8_dot32_batch4` (f32 ref) | 141026750 | 1.000x | 0.00000000 |
| `scaled_f32_dot32_batch4` | 60536000 | 2.330x | 0.00000000 |
| `reevec_neon_f32_dot32_batch4` | 27800625 | 5.073x | 0.00000000 |
| `reecast_neon_scale_batch4` | 28192375 | 5.002x | 0.00000000 |
| `reetail_neon_tail3_batch4` | 26403125 | 5.341x | 0.00000000 |
| `reewide_neon_f32_dot32_batch8` | 30513917 | 4.622x | 0.00000000 |
| **`reedot_i8_vdot` (int8 sdot)** | **7465542** | **18.890x** | **0.00749338** |

For context, the output2 bundling lab (own baseline) reached `6.566x`. The int8
`sdot` variant is ~`3.5x` faster than the best f32 NEON kernel (`reetail` 5.341x).

## Analysis

R110 passes the lab gate. Native int8 `sdot` is a multiplicative jump (`18.89x`)
that no f32 kernel reached across R78–R109 — confirming the prefill gap is the
arithmetic type, not loop polishing. The only nonzero `max_abs_diff` is the int8
variant's `0.0075`, which is the activation-quantization error (the f32 variants
stay bit-exact). The lab does not model chunk boundaries, output strides, lazy
loading, or profile overhead, and it amortizes activation quant. It does prove
the int8 direction is worth a runtime prototype, but the activation quant must be
parity-checked on the real model first (R111) before any runtime promotion.

## Decision

accepted lab

Reason: `reedot_i8_vdot` beat `baseline_i8_dot32_batch4` by `18.89x` and the best
f32 NEON kernel by ~`3.5x`, with `max_abs_diff=0.00749338` (activation-quant
error, expected for int8).

Paper value:

- positive lab evidence that native int8 `sdot` is the prefill/decode lever, not
  further f32 micro-optimization
- explains why R78–R109 plateaued (f32 NEON ceiling ~5x)
- does not claim runtime speedup yet; requires parity (R111) then promotion (R112)

## Next Experiment

R111 must validate that quantizing activations to int8 preserves output on the
real Llama 3.2 1B Q8 model (token + top-1/top-10 logit parity) before R112
promotes a `REEBORN-Q8-SDOT` kernel into `accumulate_q8_0_chunk`.
