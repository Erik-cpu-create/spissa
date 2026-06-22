# R112 REEBORN Q8 SDOT Runtime Promotion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the R110 lab-proven int8 `sdot` direction into the real Q8 matmul (`accumulate_q8_0_chunk`) as `REEBORN-Q8-SDOT`, carrying the batch4 / int32-accumulator-tile structure so the ~18.9x lab win transfers to runtime prefill, and accept it only if it beats a same-turn f32 control while keeping R111 output parity.

**Architecture:** Replace the f32 dequant-then-FMA inner loop, for full 32-element blocks in the batch>1 prefill path, with: quantize each prompt-token activation segment to int8 once, accumulate int8×int8 → int32 in a register tile over 4 (then 8) token rows via `sdot`, and apply `weight_scale * activation_scale` at the end. Keep the existing f32 path as the portable / non-dotprod fallback and for boundary/partial blocks. This is NOT the R111 correctness-only gated path (which quantizes per-segment with no batching) — R112 must batch and block.

**Tech Stack:** Rust, `rllm-runtime` `streaming/kernels.rs`, aarch64 `sdot` (inline asm, dotprod feature-gated + runtime-detected), `rllm-cli` `llama-test`, `rllm bench`, benchmark reports under `docs/benchmarks/trials/`.

## Evidence Inputs

- R110 lab: `reedot_i8_vdot` 18.890x over f32 baseline, ~3.5x over best f32 NEON kernel, max abs diff 0.00749338.
- R111 parity: int8 activations preserve token (3/3) and logit top-1 (3/3) / top-10 (10/10) on the real model.
- R109 runtime control: best prefill 7.76s, output `No`, peak 1,050,673,152 bytes.
- Ollama single-thread target: prefill ~1200 tok/s, decode ~51 tok/s.
- Caveat: the R111 gated path (`RLLM_Q8_ACTIVATION`) is correctness-only and ran ~23s prefill — proof that per-segment quant without batching loses; R112 must amortize activation quant and block the accumulator.

## Boundary

- Runtime owner: `crates/rllm-runtime/src/streaming/kernels.rs`
- Test owner: `crates/rllm-runtime/src/streaming/tests.rs`
- Benchmark docs owner: `docs/benchmarks/trials/`

Do not change the Q8 block format, model/container format, tokenizer, prompt template, memory-budget logic, Q8 argmax, or the batch1 decode fast path. Keep the exact f32 path as default until R112 passes; gate the new kernel or make it the full-block batch>1 path with the f32 fallback retained.

## Files

- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Quantize prompt-token activations to int8 once per matmul (per-row scale), reused across output features.
  - Add a `REEBORN-Q8-SDOT` batch4 int32-tile kernel (sdot inner, scalar fallback), used for full 32-blocks in the batch>1 path; keep f32 for tails/boundaries and non-dotprod CPUs.
- Modify: `crates/rllm-runtime/src/streaming/tests.rs`
  - Add a kernel test asserting the int8 batch4 tile matches the f32 path within the int8 activation tolerance.
- Create on success: `docs/benchmarks/trials/success/2026-06-17-r112-reeborn-q8-sdot-runtime.md`
- Create on failure: `docs/benchmarks/trials/failed/2026-06-17-r112-reeborn-q8-sdot-runtime.md`
- Modify: `docs/benchmarks/trials/index.md`

## Gates

Correctness:

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo test -p rllm-runtime streaming -- --nocapture
```

Build:

```bash
cargo fmt --check
cargo build --release -p rllm-cli --bin llama-test --bin rllm
git diff --check
```

Runtime (same prompt/flags as R109; add parity recheck):

```bash
# same-turn f32 control then REEBORN candidate
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked"
# parity recheck (R111 method)
# OFF vs candidate first-step logits with RLLM_FULL_LOGITS=1: top-1 must match, top-10 overlap 10/10
```

Acceptance:

- output remains exactly `No`; R111 parity still holds (top-1 match, top-10 10/10) on the 3 prompts
- best candidate prefill beats the same-turn f32 control by a clear margin (target: at least the lab-implied multiple, not a few %)
- RLLM peak transient stays at or below the f32 control
- decode tok/s does not regress

Rejection: output/parity changes, prefill regresses, peak transient grows, or the int8 tile is not materially faster than f32 in the real loop (R92-style lab-does-not-transfer outcome). If rejected, revert runtime code, keep the failed report + this plan.

## Tasks

## Task 1: Activation quantization + int8 batch4 tile kernel

- [ ] Add a per-matmul int8 activation quantizer (per-row absmax scale), computed once and reused across output blocks.
- [ ] Add `REEBORN-Q8-SDOT` batch4 kernel: load int8 weights + 4 int8 activation rows, `sdot` into 4 int32 accumulators, scale at the end; scalar fallback; dotprod runtime-detected.
- [ ] Wire it into the full-block batch>1 path of `accumulate_q8_0_chunk`; keep f32 for tails/boundaries and non-aarch64.

## Task 2: Correctness test

- [ ] Add a streaming test comparing the int8 batch4 tile to the f32 reference within the int8 activation tolerance (e.g. ~5% as in the lab gate), and a bit-exact test for the f32 fallback.

## Task 3: Runtime + parity gate

- [ ] Same-turn f32 control vs REEBORN candidate prefill/decode/peak.
- [ ] Re-run R111 logit parity (top-1 / top-10) on the candidate.
- [ ] `rllm bench` re-measure vs Ollama (single-thread) to quantify the closed gap.

## Task 4: Report, index, verification, commit

- [ ] Write success or failed R112 report with measured numbers.
- [ ] Update `docs/benchmarks/trials/index.md`.
- [ ] `cargo fmt --check`, tests, `git diff --check`.
- [ ] Commit: `bench(runtime): gate reeborn q8 sdot prefill` (or `reject ...` on failure).
