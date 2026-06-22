# Trial: R138 row-parallel q8 prefill — ~2x (single-threaded short prompts fixed)

Date: 2026-06-19
Owner: RLLM
Status: accepted (prefill 1.25s -> 0.64s for 6 tokens; ~2x)
Folder: success

## Hypothesis

After R134-R137, decode was ~8 tok/s (~1.5x behind Ollama CPU) but prefill was
~7x behind (1.2s vs ~0.17s for 6 tokens). Profiling showed prefill ran
SINGLE-THREADED for short prompts while decode used all cores: the q8 prefill
matmul parallelized by-batch with a high threshold (batch>=8) and spawned threads
PER CHUNK. A whole-tensor, parallelize-once path (like R133 gave decode) should
fix it — but the split axis matters.

## Scope

- Mode: fast-lowram runtime (q8, codec rtc-raw-v1)
- REE kernel: q8 panel prefill (i8mm `smmla`), row-parallel wrapper — name pending Erik
- Model/artifact: `models/gemma-3-4b-it-q8.spsa`
- Architecture: Gemma 3 4B, Q8_0
- Target device/profile: Apple Silicon, 8 GB RAM, CPU only
- Expected bottleneck: prefill single-threaded (parallelization axis)
- Bottleneck tag: scheduler (threading) / CPU arithmetic

## Setup

```bash
cargo build --release -p rllm-cli
# prefill thread scaling (step 0 = prefill):
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 ./target/release/gemma-test --fast ...
RLLM_THREADS=8 RLLM_Q8_KERNEL_PROFILE=1 ./target/release/gemma-test --fast ...
```

## Results

Two dead ends found first (both recorded so they aren't retried):
- Lowering the by-batch threshold (MIN 4->2): REGRESSED — batch=6 went
  1249ms -> 2189ms at 8 threads, because the by-batch path spawns threads PER
  CHUNK (~238/token) and the per-chunk work is too small. Reverted.
- Splitting by BATCH at the whole-tensor level: also slower (1238 -> 1943ms),
  because each worker re-reads the WHOLE weight tensor for its few batch rows —
  tripling weight bandwidth and defeating the arithmetic-intensity of batching.

The fix — split by OUTPUT ROWS (each worker reads a disjoint weight-row slice
ONCE, computes all batch tokens, writes a local buffer, scatter-merged after):

| prompt | threads=1 | threads=8 | speedup |
|---|---:|---:|---:|
| 6 tokens (step 0) | 1253 ms | 639 ms | ~2.0x |
| 27 tokens (step 0) | 2287 ms | 1111 ms | ~2.1x |

End-to-end 58-token generation: 5.20 -> 6.23 tok/s. Per-layer prefill (layer 0):
attn 6.1->3.7ms, MLP 28.7->12.5ms. Output unchanged and coherent
("Photosynthesis is the process where plants, algae, and some bacteria use
sunlight ..."). Parity test asserts the row-split matches the single-threaded
kernel within f32 accumulation-order tolerance. 285 tests green.

## Analysis

The win is real but caps at ~2x rather than ~Ncore. The split axis was the key
correctness/perf lever: by-output-row keeps weights read once (bandwidth) and
batch whole (panel engaged), and is sound via per-worker local buffers + a
single-threaded scatter. The remaining gap to linear scaling is per-projection
overhead — each worker heap-allocates and zeroes a local `[batch, rows]` buffer
(allocator contention across 8 threads, ~816 allocations per prefill) and the
scatter-merge — plus partly-serial attention SDPA. A single shared transposed
`[out_features, batch]` scratch per projection (one alloc, workers write disjoint
contiguous slices) would remove the per-worker allocation; that needs a
transposed-output kernel variant and is deferred.

## Decision

accepted

Reason: ~2x prefill, correct (coherent output, parity test), tested. Closes the
prefill gap vs Ollama CPU from ~7x to ~3.8x (0.64s vs ~0.17s for 6 tokens).

Paper value:

- use as positive evidence (parallelization AXIS matters: by-output-row, not
  by-batch or per-chunk) with limitation (~2x, not linear; allocation overhead).

## Next Experiment

Shared transposed scratch to drop per-worker allocations (toward linear scaling);
profile attention SDPA parallelism for long prompts. Then prefill could approach
Ollama's ~0.17s.
