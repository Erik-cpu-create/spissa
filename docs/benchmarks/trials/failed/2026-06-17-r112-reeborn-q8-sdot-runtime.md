# R112: REEBORN Q8 SDOT Runtime

Date: 2026-06-17
Owner: RLLM
Status: rejected
Folder: failed

## Hypothesis

R110 proved int8 `sdot` was ~18.9x faster than the f32 baseline in the lab and
R111 validated activation-q8 parity. R112 tests whether promoting `sdot` into the
runtime Q8 matmul (`accumulate_q8_0_chunk`, behind `RLLM_Q8_ACTIVATION`) — with
activation quantization hoisted/cached once per matmul and a batch4 NEON inner
loop — beats the tuned f32 path on real Llama 3.2 1B Q8 prefill.

## Scope

- Mode: exact-lowram runtime gate
- REE kernel lineage: `REEBORN-Q8-SDOT`
- Model/artifact: `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Threading: `RLLM_THREADS=1`; integrity `unchecked`; chat-template llama3
- Bottleneck tag: CPU arithmetic / Q8 integer-SIMD dot
- Implementation: per-32-block activation quant cached once per matmul
  (thread-local, fingerprint-guarded), int8×int8 `sdot` inner loop with a batch4
  group (weight block loaded once, 4 token rows per group) + 1-row tail; scalar
  fallback for non-dotprod CPUs.
- Coverage: only the matmuls that route through `accumulate_q8_0_chunk` — i.e.
  attention q/k/v/o and MLP gate/up (via `streaming_tile_linear_from_model`, since
  the fused `silu_gate_up` returns `None` for Q8). MLP down (`multiply_into`) and
  lm_head (`argmax`) use separate Q8 kernels and were NOT converted.

## Setup

```bash
cargo build --release -p rllm-cli --bin llama-test
# control (f32) vs candidate (int8 sdot)
printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | RLLM_THREADS=1 target/release/llama-test --model <artifact> --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked
printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | RLLM_THREADS=1 RLLM_Q8_ACTIVATION=1 target/release/llama-test --model <artifact> --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked
```

## Results

Output stayed `No` for both; R111 logit parity (top-1 / top-10) still held.

| run | prefill (best) | decode tok/s | MLP total | attention | lm_head |
|---|---:|---:|---:|---:|---:|
| control (f32) | 7.76s | 1.57 | 5664.94ms (gate 2299/up 1609/down 1747) | 993.29ms | 1103.50ms |
| REEBORN (int8 sdot batch4) | 8.06s | 0.83 | 6370.23ms (gate 2809/up 1509/down 2042) | 1061.10ms | 1226.78ms |

Single-thread runs were noisy (control prefill 7.76–8.57s; candidate 8.06–10.41s
across the naive→batch4 iterations). No candidate run beat the control best, and
component timings overlapped within run-to-run variance.

## Analysis

R112 is rejected. `sdot` does not beat the tuned f32 path in the runtime:

- **No prefill win:** candidate best (8.06s) ≥ control best (7.76s); covered
  components (gate/up/attention) did not drop beyond noise.
- **Decode regressed ~2x** (1.57 → 0.83 tok/s): at batch=1 there is no batch4
  benefit, so the per-token activation-quant + cache overhead is pure cost.
- **The lab number was misleading for promotion:** R110's `18.9x` was `sdot` vs
  the *scalar* `baseline_i8_dot32_batch4`. Versus the already-tuned f32 NEON
  kernel (`reevec`/`reebundle`, ~5x over that scalar baseline), real `sdot`
  headroom is only ~2-3x — and `sdot` is a 4-wide dot, comparable to the f32
  `fmla`×4 the runtime already uses, so the activation-quant overhead and missing
  output2 reuse erase the gain.
- **Partial coverage** (down/lm_head untouched) also caps any win.

## Decision

rejected (runtime code reverted; the R110 lab kernel and R111 parity tooling stay)

Reason: int8 `sdot` is within noise of the tuned f32 prefill path and regresses
decode ~2x; the lab speedup was relative to a scalar baseline, not the runtime
f32 NEON path.

Paper value:

- useful negative evidence that `sdot` (4-wide int8 dot) is NOT enough to beat a
  tuned f32 NEON Q8 matmul; the int8 advantage needs a wider GEMM instruction
- confirms a lab win measured against a scalar baseline must be re-checked against
  the real runtime kernel before promotion (cf. R85/R86/R92)

## Next Experiment

R113 should target **i8mm (`smmla`)** — ARMv8.6/ARMv9 int8 matrix-multiply that
computes an 8×8 int8 tile per instruction (what ggml/llama.cpp use for Q8 GEMM),
reusing the R112 activation-quant/cache design. This is the GEMM-level lever; the
4-wide `sdot` is not. Keep the f32 path as the default until i8mm clears the gate.
