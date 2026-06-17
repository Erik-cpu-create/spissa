# R113 REEFUSE Q8 i8mm (smmla) Runtime Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Beat the tuned f32 Q8 prefill path using ARM **i8mm (`smmla`)** — an int8 8×8 matrix-multiply-accumulate per instruction — instead of the 4-wide `sdot` that R112 showed is not enough. Accept only if same-turn runtime prefill beats the f32 control and R111 parity holds.

**Architecture:** `smmla` multiplies a 2×8 int8 tile by an 8×2 int8 tile into a 2×2 int32 accumulator. Tile the Q8 matmul as (2 output features × 2 token rows) per `smmla`, walking the shared 8-wide int8 K dimension, so ONE instruction does what R112's batch4 did in 4 `sdot`s while also amortizing the activation load across 2 outputs (the output2 reuse R112 lacked). Reuse the R112 activation-quant design (per-32-block int8 + scales, cached once per matmul) — the lab/parity work from R110/R111 stands; only the inner kernel changes.

**Tech Stack:** Rust, `rllm-runtime` `streaming/kernels.rs`, aarch64 `smmla` (inline asm; `vmmlaq_s32` intrinsic is also nightly-gated, so use `asm!` + `target_feature(enable = "i8mm")`, runtime-detected via `is_aarch64_feature_detected!("i8mm")`), `rllm-cli` `llama-test`, `rllm bench`.

## Evidence Inputs

- R110 lab: sdot 18.9x vs scalar baseline (but only ~2-3x vs tuned f32 NEON).
- R112 rejected: sdot batch4 within noise of f32 prefill (best 8.06s vs 7.76s), decode regressed ~2x; the 4-wide dot ≈ f32 fmla×4.
- R112 profile: MLP dominates (~5.7s of 7.8s); gate/up + attention route through `accumulate_q8_0_chunk`; down (`multiply_into`) + lm_head (`argmax`) are separate kernels.
- A18 Pro is ARMv9.2 → has both `dotprod` and `i8mm`.

## Boundary

- Runtime owner: `crates/rllm-runtime/src/streaming/kernels.rs`
- Test owner: `crates/rllm-runtime/src/streaming/tests.rs`
- Benchmark docs: `docs/benchmarks/trials/`

Do not change Q8 block format, container/model format, tokenizer, prompt template, memory-budget logic, or the batch1 decode fast path. Keep the f32 path as default and the fallback for non-i8mm CPUs and batch1.

## Files

- Modify: `crates/rllm-runtime/src/streaming/kernels.rs`
  - Add a microbench lab variant first (`REEFUSE-Q8-I8MM-LAB`) in `q8_kernel_lab.rs` proving `smmla` beats `reedot_i8_vdot` (sdot) for the batch-55 shape.
  - Add an `smmla` (2×2 tile) runtime kernel behind `RLLM_Q8_ACTIVATION` + i8mm detection; reuse `with_quantized_activations`. Restrict to batch>1 (prefill); keep f32 for batch1 decode.
- Modify: `crates/rllm-runtime/src/streaming/tests.rs` — tile kernel matches f32 within int8 tolerance.
- Create: success/failed `docs/benchmarks/trials/.../2026-06-17-r113-reefuse-q8-i8mm-runtime.md`
- Modify: `docs/benchmarks/trials/index.md`

## Gates

Lab (first):

```bash
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench --json target/r113.json --markdown target/r113.md --batch 55 --in-features 2048 --iters 3000
# accept lab only if smmla variant clearly beats reedot_i8_vdot (sdot)
```

Runtime:

```bash
# same-turn f32 control vs i8mm candidate, prompt 'Answer yes or no: is fire cold?'
printf '%s\nquit\n' '...' | RLLM_THREADS=1 target/release/llama-test --model <artifact> --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked
printf '%s\nquit\n' '...' | RLLM_THREADS=1 RLLM_Q8_ACTIVATION=1 target/release/llama-test ... (same)
# parity recheck: RLLM_FULL_LOGITS=1 first-step logits top-1 match + top-10 10/10
```

Acceptance: output stays `No`; R111 parity holds; best candidate prefill beats the same-turn f32 control by a clear margin (not noise); decode does not regress (batch1 stays on f32); peak transient unchanged.

Rejection: no clear prefill win, decode regresses, parity breaks, or i8mm lab does not beat sdot. Revert runtime code, keep the failed report.

## Tasks

## Task 1: i8mm lab gate
- [ ] Add `REEFUSE-Q8-I8MM-LAB` (`smmla` 2×2 tile) to `q8_kernel_lab.rs`; prove it beats `reedot_i8_vdot` for batch 55 / in 2048. If it does not beat sdot, stop here and record the lab failure.

## Task 2: Runtime i8mm tile kernel
- [ ] Add the `smmla` 2×2 (2 outputs × 2 rows) kernel in `accumulate_q8_0_chunk` behind i8mm detection; reuse `with_quantized_activations`; restrict to batch>1; f32 fallback for batch1/non-i8mm.

## Task 3: Correctness + runtime + parity gate
- [ ] Streaming test: tile kernel matches f32 within int8 tolerance; bit-exact f32 fallback.
- [ ] Same-turn control vs candidate prefill/decode/peak; R111 logit parity recheck; `rllm bench` vs Ollama.

## Task 4: Report, index, verify, commit
- [ ] Write success/failed R113 report with measured numbers; update index; `cargo fmt --check` + tests + `git diff --check`.
- [ ] Commit `bench(runtime): gate reefuse q8 i8mm prefill` (or `reject ...`).

## Note

If i8mm also fails to beat tuned f32 at batch 55 single-thread, the remaining levers are: (a) multi-thread the prefill matmul (Ollama uses all 6 cores — the largest untapped factor), and (b) convert the down/lm_head kernels too. Reassess after R113.
