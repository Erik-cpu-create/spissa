# R133 REEBORN-Q8-SDOT Decode Redesign Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax. This is informed by a study of llama.cpp/ggml (see memory `ggml-cpu-q8-methodology`) — **reimplement ORIGINAL, do NOT port/copy ggml code** (hard-rule: no wrapping; MIT not copied).

**Goal:** Make **raw-codec q8 DECODE** (batch=1) fast by restructuring the orchestration to ggml's proven shape, while keeping RLLM's chunk machinery for the streaming/compressed/low-RAM/integrity cases. Target: close the bulk of the ~100× CPU decode gap vs llama.cpp (Gemma 4B q8: RLLM ~0.1 tok/s vs llama.cpp/Ollama-CPU ~12 tok/s). Accept only with measured decode speedup on a **cool** machine + near-exact parity (quant-only, same as llama.cpp).

**Root cause (measured + ggml study):** RLLM's gap is NOT hardware/RAM (same device, same RAM) — it's the **per-chunk streaming structure fighting the fast pieces**. (a) default decode kernel is scalar i8×f32 (`q8_0_dot_i8_f32`, ~0.57 GMAC/s; ggml NEVER does i8×f32 — always int8 sdot); (b) the int8-sdot path is single-threaded while the scalar path spawns a `std::thread::scope` PER CHUNK (~3366 chunks/token); (c) weight read is wrapped per-chunk (metadata+budget+integrity+closure ×3366/tok) vs ggml's one contiguous pointer+stride. RLLM already HAS the pieces (sdot R130 `batch1_x4_ilp`, act-cache R127, row-parallel R132) — they're wired in a structure that defeats them.

**Architecture (the fast-path):** Add a decode fast-path gated on `dtype==Q8_0 && codec==rtc-raw-v1 && batch==1`:
1. **Direct contiguous weight view** — read the whole weight tensor's q8 bytes as ONE contiguous `&[u8]` mmap slice (bypass the per-chunk `with_raw_chunk` loop on this path). Keeps zero-copy; drops per-chunk dispatch.
2. **Quantize the f32 activation row to int8 ONCE per matmul** (reuse `quantize_input_q8_blocks`, hoisted — not per chunk).
3. **int8×int8 sdot, row-major, ILP** (reuse `batch1_x4_ilp` / `i8_dot32`): per output row sum over 32-blocks `f32(vdot) * (d_w·d_a)`; this is the existing R130 kernel, now fed a contiguous tensor.
4. **Row-parallel via a PERSISTENT pool with dynamic work distribution** — NOT `thread::scope` per chunk. Split output rows across workers; each grabs a row-range. Reuse RLLM's existing rolling/REEWEAVE worker if persistent, else introduce one persistent pool for the decode matmul.

Keep the existing chunk path as the fallback for: compressed codecs, low-RAM budget, integrity-strict, non-aarch64, and batch>1 prefill (already paneled via smmla). The lossless/streaming differentiator stays intact.

**Tech Stack:** Rust, `rllm-runtime` `streaming/kernels.rs` + `streaming/linear.rs`, aarch64 `sdot` (existing inline asm), `LazyRllmModel` mmap, `rllm-cli` `gemma-test` (+ `llama-test`), `rllm bench`.

## Evidence Inputs
- Profile: 94% of Gemma q8 decode = `batch1_complete_linear` scalar path, 0.57 GMAC/s (~100× below int8 peak). lm_head only 6%.
- A/B (thermal-controlled): int8-activation prefill 2.16× FASTER, but decode ~1.4× SLOWER than scalar — because int8 path is single-thread + per-chunk overhead, scalar path is multi-thread (R132).
- Bug already fixed (commit 39dfef1): `r += 1` infinite loop in `accumulate_q8_0_chunk_int8_batch1_rowmajor` (non-÷4 chunk rows).
- ggml study (memory `ggml-cpu-q8-methodology`): always-int8-sdot, quantize-act-once, persistent pool + work-steal, contiguous mmap, GEMV chunk_size 64.
- Thermal: this session saturated (step-0 scalar drifted 22s→39s) → benchmark on a COOL machine, warm-iteration (paper "LLM Inference at the Edge", arXiv 2603.23640).

## Boundary
- Runtime owners: `crates/rllm-runtime/src/streaming/kernels.rs`, `streaming/linear.rs`.
- Test owner: `crates/rllm-runtime/src/streaming/tests.rs`.
- Benchmark docs: `docs/benchmarks/trials/`.
- Do NOT change: Q8 block format (34B), container/model format, tokenizer, the chunk machinery semantics for the non-fast-path, lossless/streaming defaults, or the compressed-codec path.

## Files
- `streaming/linear.rs` — add the decode fast-path dispatch (Q8_0 + rtc-raw-v1 + batch==1) that obtains a contiguous tensor view and calls the new orchestrator instead of the per-chunk loop.
- `LazyRllmModel` (lazy.rs) — confirm/add a `with_raw_tensor` (whole-tensor contiguous mmap slice) IF chunks for a tensor are contiguous in the file; else document why and process chunk-contiguous-runs.
- `streaming/kernels.rs` — the row-parallel persistent-pool orchestrator over the contiguous q8 tensor + the hoisted activation quant. Reuse `batch1_x4_ilp`.
- `streaming/tests.rs` — parity test: fast-path output == per-row int8 reference (bit-identical) for non-÷4 rows + multi-chunk-spanning tensor.
- `q8_kernel_lab.rs` — (optional) lab variant proving persistent-pool row-parallel sdot beats per-chunk thread::scope for a decode GEMV shape.
- Trial doc `docs/benchmarks/trials/<status>/2026-06-18-r133-reeborn-q8-sdot-decode-redesign.md` + `index.md` row.

## Design investigations (resolve before coding)
- [ ] **Contiguity:** are a tensor's rtc-raw-v1 q8 chunks contiguous in the mmap'd `.spsa`? If yes → one `&[u8]` slice for the whole tensor. If no → process maximal contiguous runs (still far fewer dispatches than per-chunk).
- [ ] **Persistent pool:** does RLLM have a reusable persistent worker pool (rolling/REEWEAVE)? If yes, route the row-split through it. If no, add ONE (avoid per-call spawn). Honor `RLLM_THREADS`.
- [ ] **Integrity:** the fast-path skips per-chunk VerifyOnce — verify the tensor once on first access (or under strict mode fall back to chunk path). Keep the lossless integrity guarantee available.

## Gates
Lab/parity first:
```bash
cargo build --release -p rllm-cli --bin gemma-test
cargo test -p rllm-runtime --lib    # new parity test green, full suite green
```
Runtime (COOL machine, warm-iteration, best-of-N):
```bash
cargo build --release -p rllm-cli
# decode tok/s, fast-path vs current, same prompt, greedy:
./target/release/gemma-test --model models/gemma-3-4b-it-q8.spsa --prompt "The capital of France is" --max-new-tokens 16 --ctx 256
```
Accept if: decode tok/s materially up vs current (record actual), "Paris…" preserved, parity diff = quant-only (report max_abs_diff + argmax match vs f32 scalar), full suite green.

## Steps
- [ ] Resolve the 3 design investigations above.
- [ ] Add contiguous-tensor view (or contiguous-run iteration) for Q8_0 rtc-raw-v1.
- [ ] Hoist activation quant to once-per-matmul on the fast-path.
- [ ] Wire row-parallel persistent-pool orchestrator over the contiguous q8 + `batch1_x4_ilp`.
- [ ] Gate the fast-path; keep chunk path as fallback (compressed/low-RAM/strict/non-aarch64/batch>1).
- [ ] Parity test (non-÷4 rows, multi-chunk tensor) + full suite.
- [ ] Benchmark on a COOL machine (warm-iteration); record honest before→after + Ollama-CPU gap as limitation.
- [ ] REE name (REEBORN lineage — Erik's call) in the trial Scope.
- [ ] File trial report per `docs/benchmarks/trials/README.md` (template → active → success/failed), update `index.md`.
- [ ] Update memory [[rllm-speed-thesis-streaming-vs-resident]] + [[ggml-cpu-q8-methodology]] with measured R133 numbers.

## Honest scope
This is a re-architecture of the q8 decode orchestration, not a one-liner. It attacks the per-chunk-structure tax + the scalar kernel together. It will NOT fully match llama.cpp (RLLM keeps a richer container/streaming layer), but should close the bulk of the CPU decode gap. The remaining delta + the prefill path are separate. No NPU (out of scope — RLLM is CPU/lossless by doctrine).
