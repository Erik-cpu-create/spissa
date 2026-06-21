# RLLM GPU backend — plan (decided 2026-06-21)

> **PHASE 1 GO/NO-GO RESULT (2026-06-21): NO-GO on the budget phone.** A coalesced GEMV
> (the decode primitive) on the Mali-G52 hit **1.4 GB/s effective → ~1.0 tok/s — identical
> to the CPU** (which also measured ~1.4 GB/s / ~1 tok/s). Two different straightforward
> kernels (tiled GEMM 4.4 GFLOP/s, reduction GEMV 1.4 GB/s) both land at CPU level → the
> **MT6768's LPDDR4 delivers only ~1.4 GB/s effective to ANY processor for this pattern**;
> decode is memory-bound, so CPU and GPU hit the same memory wall. **~1 tok/s for 1B is the
> ceiling on this phone regardless of backend.** A GPU backend would NOT beat the CPU here —
> the gate saved months. Caveat: kernels aren't expert-tuned (an optimized kernel might reach
> ~3-5 GB/s), but memory-bound + the phone's low effective bandwidth caps the upside.
> **Decision: do NOT build the GPU backend for budget phones; stay CPU-lossless. GPU stays a
> future option for FLAGSHIP phones / laptops (LPDDR5/dGPU = ~4-5× bandwidth) — the wgpu+Vulkan
> foundation + cross-build recipe are proven and saved.** The 20-tok/s app = flagship hardware.



Decision: add an auto-detecting GPU backend so RLLM is **plug-and-play** on phones — one
generic binary that picks the GPU when present and falls back to CPU otherwise. User copies
the binary + a repacked `.rllm` and runs; no per-device rebuild. (The CPU binary already
works this way: one generic aarch64 ELF + runtime feature detection.)

## Why this is achievable (proven in the prototype)

- `wgpu` cross-compiles to `aarch64-linux-android` and **runs headless on the Mali-G52 via
  Vulkan from a CLI process** (verified — `gpumm` ran, reported the adapter, executed a
  compute shader). So one generic binary + runtime adapter selection = plug-and-play.
- `wgpu` is **portable**: same code → Vulkan (Android/Linux), Metal (macOS), DX12/Vulkan
  (Windows). Aligns with the universal-device goal ([[universal-device-target]]).

## Plug-and-play architecture

- Backend trait: `CpuBackend` (current) and `GpuBackend` (new) behind one interface.
- Startup: try `GpuContext::new()` → wgpu instance → request a compute-capable adapter.
  - Found → GPU backend. Not found / init fails → **graceful CPU fallback** (today's path).
  - `RLLM_BACKEND=cpu|gpu|auto` override; `auto` (default) prefers GPU.
- Same `rllm chat <model.rllm>` UX; the binary decides CPU vs GPU at runtime.

## Honest constraints (from the prototype + physics)

- **Decode is memory-bound; the phone's RAM bandwidth caps it.** On the MT6768 (LPDDR4
  ~13 GB/s) the absolute ceiling is ~9 tok/s (1B q8) / ~18 (1B q4); GPU realistically
  ~4-7 tok/s (1B q8) vs CPU's ~1. GPU wins by *better bandwidth utilization* (MLP), not by
  adding bandwidth. The "20 tok/s 4B" app was a flagship phone (LPDDR5) — physically
  impossible on MT6768.
- **f16 (essential for GPU LLM perf) is NOT supported by wgpu/naga's WGSL yet** (naga issue
  #4384, confirmed on wgpu 24). → f16 kernels need **raw SPIR-V** (write GLSL → `glslc`
  from the NDK → `ShaderSource::SpirV` passthrough). f32 WGSL works now (for correctness
  bring-up); f16 SPIR-V comes in the perf phase.
- **Lossless on GPU is hard**: weights live in GPU buffers; rANS/bit-plane decode would run
  on CPU at load then upload (bf16 in GPU memory = lossless, ~2 GB for 1B — fits unified
  memory). Keeps the lossless angle (novel: most GPU runtimes are lossy q4).
- This is a **multi-month, multi-session build**, not a quick feature.

## Phased plan

**Phase 1 — foundation + GO/NO-GO (the critical gate).**
- `rllm-gpu` crate: wgpu instance, adapter auto-detect, CPU fallback, buffer upload/readback.
- ONE kernel: matmul / batch-1 GEMV (f32 first), validated **bit-approx vs the CPU kernel**.
- Benchmark the GEMV on the Mali-G52 **in the LLM shape** (e.g. 1152×6912). 
- **Gate:** does the GPU GEMV beat the CPU meaningfully on the phone? If yes → continue. If
  no (even after a tiled kernel) → reconsider before sinking months in.

**Phase 2 — full forward pass on GPU (Gemma first).**
- Kernels: RMSNorm, RoPE, attention + KV cache, SiLU-gate-up, softmax, embedding lookup,
  lm_head + argmax. Wire `gemma_forward_logits` to run on the GPU backend; parity vs CPU.

**Phase 3 — weights + f16.**
- Load path: decode `.rllm` (rANS/q8/bf16) → upload to GPU buffers (bf16 = lossless).
- f16 kernels via **SPIR-V** (glslc) for the perf the GPU is for.

**Phase 4 — optimize + measure.**
- Mali-tuned kernels (register blocking, vec4, larger tiles), KV cache resident on GPU,
  end-to-end tok/s on the phone. Target: ~4-7 tok/s (1B) — the bandwidth-bound realistic max.

**Phase 5 — universal + polish.**
- Validate Metal (macOS), Vulkan (Android), DX12/Vulkan (desktop). `RLLM_BACKEND` knob,
  docs, plug-and-play packaging (binary + repacked model).

## Prototype artifact

`~/Projects/wgpu-matmul-proto` (outside the repo) — proven: wgpu compute on Mali-G52 via
Vulkan, f32 naive 2.1 / tiled 4.4 GFLOP/s, f16 blocked by naga. Reuse its build recipe
(NDK linker env) for the rllm-gpu cross-build.
