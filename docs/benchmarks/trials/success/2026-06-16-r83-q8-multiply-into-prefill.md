# R83: Q8 Multiply-Into Prefill

## Status

Success.

## Hypothesis

R80/R81 optimized the regular Q8 linear path but not the `up_proj` multiply-into path. Applying the same scaled-block batch4 strategy to Q8 multiply-into should reduce exact prefill time, especially the `up_projection_ms` bucket.

## Artifact

- Model: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Mode: exact-lowram Q8 transformer keep-IO rowchunks
- Prompt: `Answer yes or no: is fire cold?`
- Chat template: `llama3`
- Integrity mode: `unchecked`

## Baselines

- R78 prefill: 26.75 s
- R81 best verify-once prefill: 21.41 s
- R82 best unchecked prefill: 16.38 s
- R82 best unchecked MLP total: 14,096.94 ms

## Change

The Q8 multiply-into path now mirrors the accepted R80/R81 scaled-block batch4 optimization:

- full Q8 blocks with `batch > 1` are scaled once into stack-local `[f32; 32]`
- four prompt-token accumulators are updated per helper call
- remainder batches use the existing scalar `f32_dot_32`
- batch-1 decode and partial-block behavior are unchanged

## Command

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked"
```

## Verification

```sh
cargo test -p rllm-runtime f32_dot_32_batch4_into_accumulates_existing_values
cargo test -p rllm-runtime q8_0
cargo test -p rllm-cli --bin llama-test
cargo build --release --bin llama-test
```

Results:

- `f32_dot_32_batch4_into_accumulates_existing_values`: passed
- `cargo test -p rllm-runtime q8_0`: passed 9 tests
- `cargo test -p rllm-cli --bin llama-test`: passed 22 tests
- release build passed

## Results

Both R83 runs kept the answer correct:

```text
No
```

| Run | Integrity | TTFT / prefill | MLP total | Gate | Up | Down | Decode | E2E | Peak transient | Max RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R78 baseline | verify-once | 26.75 s | 20,324.73 ms | 6,723.82 ms | 6,745.49 ms | 6,844.64 ms | 0.87 tok/s | 0.07 tok/s | 1,050,673,152 bytes | 2,233,090,048 bytes |
| R82 unchecked best | unchecked | 16.38 s | 14,096.94 ms | 3,732.81 ms | 6,895.01 ms | 3,455.24 ms | 1.45 tok/s | 0.12 tok/s | 1,050,673,152 bytes | 2,199,977,984 bytes |
| R83 run 1 | unchecked | 13.07 s | 9,914.73 ms | 3,411.15 ms | 3,093.32 ms | 3,397.15 ms | 1.02 tok/s | 0.14 tok/s | 1,050,673,152 bytes | 1,908,817,920 bytes |
| R83 run 2 | unchecked | 11.45 s | 9,367.19 ms | 3,211.08 ms | 2,909.82 ms | 3,232.99 ms | 0.97 tok/s | 0.16 tok/s | 1,050,673,152 bytes | 2,084,995,072 bytes |

R83 best versus R78:

- prefill improved by 15.30 s
- prefill improved by 57.19%
- MLP total improved by 10,957.54 ms
- MLP total improved by 53.91%
- peak transient memory unchanged
- output unchanged

R83 best versus R82 unchecked best:

- prefill improved by 4.93 s
- prefill improved by 30.10%
- MLP total improved by 4,729.75 ms
- MLP total improved by 33.55%
- up projection improved by 3,985.19 ms
- up projection improved by 57.80%
- peak transient memory unchanged
- output unchanged

## Bottleneck Tag

CPU arithmetic.

## Analysis

R83 confirms the R82/RLLM prefill bottleneck was not evenly distributed across MLP projections. `up_proj` was still using the older Q8 multiply-into path, so it did not receive the R80/R81 scaled-block batch4 optimization.

Applying the same pattern to multiply-into cuts `up_projection_ms` from 6,895.01 ms to 2,909.82 ms in the best run, and total prefill falls from 16.38 s to 11.45 s. This is a material improvement with unchanged output and unchanged internal peak transient memory.

This is still slower than Ollama-class native kernels, but the gap is now much smaller and the path remains pure Rust, low-RAM, CPU-only, and exact for the quantized weights.

## Decision

Accept. Keep the Q8 multiply-into scaled-block batch4 path.

## Next Experiment

R84 should benchmark RLLM versus Ollama on the same machine and same model with a controlled prompt:

- run RLLM unchecked with `--profile-phases`
- run Ollama with the closest local Llama 3.2 1B instruct artifact if available
- measure wall time, TTFT if available, and tokens/sec
- then decide whether next work is prefill attention/down/gate, decode token/s, or a larger Q8 kernel refactor
