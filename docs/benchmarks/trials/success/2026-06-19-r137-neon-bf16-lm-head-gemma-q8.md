# Trial: R137 NEON bf16 lm_head GEMV — decode 3.9 -> ~8 tok/s

Date: 2026-06-19
Owner: RLLM
Status: accepted (lm_head 165ms -> ~26ms/token; end-to-end decode ~2x)
Folder: success

## Hypothesis

After R136, the dominant decode cost was lm_head: ~165ms/token for the 262k-row
vocab GEMV over the bf16 tied embedding (1.34 GB read/token at only ~8 GB/s),
larger than the entire 34-layer q8 stack (~90ms). The inner kernel was fully
scalar (one bf16->f32 upcast + one f32 mul-add per element). bf16 is the high 16
bits of f32, so the upcast is the exact bit op `(bits as u32) << 16` — trivially
vectorizable. A NEON inner kernel should make lm_head bandwidth-bound (like the
q8 layers) and roughly double end-to-end decode, while staying lossless.

## Scope

- Mode: fast-lowram runtime (q8 layers + bf16 lm_head)
- REE kernel: bf16 lm_head GEMV (NEON, lane-parallel FMA) — name pending Erik
- Model/artifact: `models/gemma-3-4b-it-q8.rllm` (bf16 tied embedding, 262208 vocab x 2560 hidden)
- Architecture: Gemma 3 4B
- Target device/profile: Apple Silicon, 8 GB RAM, CPU only
- Expected bottleneck: scalar bf16 upcast + f32 FMA (compute), not bandwidth
- Bottleneck tag: CPU arithmetic (scalar -> NEON)

## Setup

```bash
cargo build --release -p rllm-cli
RLLM_Q8_KERNEL_PROFILE=1 ./target/release/gemma-test \
  --model models/gemma-3-4b-it-q8.rllm --prompt "The capital of France is" \
  --fast --max-new-tokens 32 --ctx 256
```

Runtime context: release; Apple Silicon ARM64 CPU-only; 8 GB; macOS; `--fast`.
Kernel: load 16 bf16, `vmovl_u16` + `vshlq_n_u32::<16>` -> f32 bits, 4 independent
`vfmaq_f32` chains, `vaddvq_f32` reduce; 4-wide and scalar tails; scalar fallback
for non-aarch64.

## Results

| metric | before (scalar) | after (NEON) |
|---|---:|---:|
| lm_head per token | ~165 ms | ~26 ms |
| lm_head bandwidth | ~8 GB/s | ~51 GB/s (~HW) |
| decode token (layers+lm_head) | ~255 ms (~3.9 tok/s) | ~120 ms (~8 tok/s) |
| end-to-end 32 tokens | 9.62 s | 5.53 s |

Output unchanged and coherent: "Paris is a global center for art, fashion,
gastronomy and culture. It is home to many famous landmarks, including the
Eiffel Tower, the Louvre". 284 runtime tests green.

Parity: the SIMD path differs from the scalar f32 path only in f32 ACCUMULATION
ORDER (lane reduction vs left-to-right) — max_abs_diff < 1e-4 on the unit test,
argmax preserved; the bf16->f32 upcast is exact (no quantization, lossless). The
former bit-for-bit lm_head test was updated to a tolerance+argmax assertion, and
`lm_head_bf16_simd_matches_scalar_reference_and_preserves_argmax` was added.

## Analysis

lm_head was compute-bound on the scalar loop (8 GB/s, far below the ~50 GB/s the
hardware sustains). Vectorizing the exact bf16 upcast + FMA made it
bandwidth-bound at ~51 GB/s, a 6.3x kernel speedup, halving the decode token.
The q8 layers (~95ms) are again the larger decode component. No lossless
compromise: only f32 accumulation order changed; weights are read exactly.

## Decision

accepted

Reason: 6.3x lm_head, ~2x end-to-end decode, lossless (exact upcast, argmax
preserved), output unchanged, tests green.

Paper value:

- use as positive evidence (the bf16 output head, not just the q8 body, must be
  SIMD to avoid becoming the decode bottleneck — and the exact bf16->f32 bit
  trick keeps it lossless).

## Next Experiment

Decode is now ~120ms/token (~8 tok/s) vs Ollama 12.4 (~1.5x). The q8 LAYERS
(~95ms) are the remaining decode cost — profile attention vs projections in the
resident regime. Prefill compute (~1.2s vs Ollama 0.5s) and the one-time ~2.7s
integrity prewarm are the other end-to-end gaps.
