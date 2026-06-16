# R87: Q8 Prefill Bounded Tile Repack

## Status

Failed.

## Hypothesis

Q8 prefill is still compute-bound. A bounded row-tiled Q8 dequantization scratch for complete-row batch-1 paths should reduce per-block quantization overhead while keeping peak transient memory unchanged.

## Baseline

- R84 RLLM unchecked best prefill: 13.94s / 55 context tokens
- R85 best prefill: 12.68s
- R86 best prefill: 12.17s
- RLLM reference target: 11.45s (R83 unchecked best)
- Baseline peak transient reference: 1050673152 bytes

## Changes

- Added bounded Q8 tile constants in `crates/rllm-runtime/src/streaming/kernels.rs`
- Added `accumulate_q8_0_chunk_batch1_complete_rows_tiled`
- Added `accumulate_q8_0_chunk_multiply_into_batch1_complete_rows_tiled`
- Rewired q8 complete-row batch-1 calls to use tiled helpers
- Added tests:
  - `q8_0_batch1_row_tiled_complete_rows_match_reference`
  - `q8_0_tiled_repack_falls_back_for_non_aligned_inputs`
  - `q8_0_batch1_row_tiled_multiply_into_matches_reference`

## Tests

- `cargo test -p rllm-runtime q8_0 -- --nocapture` ✅
- `cargo test -p rllm-runtime streaming_tile_linear -- --nocapture` ✅

## Benchmark Commands

Executed exactly with release build + `RLLM_THREADS=1`:

```bash
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked --rama-trace /tmp/r87-rllm-trace-run1-rebased.json"
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked --rama-trace /tmp/r87-rllm-trace-run2-rebased.json"
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked --rama-trace /tmp/r87-rllm-trace-run3-rebased.json"
```

## Results

- output: `No` on all 3 runs
- prefill target reached? **No** (best 12.31s)

| Run | TTFT / prefill | MLP | gate | up | down | Decode | E2E | Peak transient | Max RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R87 run 1 | 12.34 s | 9,460.21 ms | 3,363.44 ms | 2,876.93 ms | 3,205.71 ms | 0.69 tok/s | 0.15 tok/s | 1,050,673,152 bytes | 1,956,298,752 bytes |
| R87 run 2 | 12.31 s | 10,031.21 ms | 3,513.77 ms | 3,111.01 ms | 3,391.61 ms | 1.05 tok/s | 0.15 tok/s | 1,050,673,152 bytes | 1,965,047,808 bytes |
| R87 run 3 | 15.18 s | 11,701.54 ms | 4,216.83 ms | 3,528.46 ms | 3,941.57 ms | 0.99 tok/s | 0.12 tok/s | 1,050,673,152 bytes | 1,684,488,192 bytes |

## Decision

Failed. Best prefill is slower than R83 (11.45s target), so the bounded repack tile path does **not** satisfy the gate. Runtime for this experiment is being reverted.
