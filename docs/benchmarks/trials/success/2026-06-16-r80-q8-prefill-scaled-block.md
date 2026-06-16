# R80: Q8 Prefill Scaled Block

## Status

Success.

## Hypothesis

For exact Q8 prefill with `batch > 1`, converting a Q8_0 block to scaled `f32` weights once per block and reusing it across prompt tokens will reduce CPU arithmetic versus repeatedly converting every `i8` weight inside every token dot product.

This targets the R79 finding that Q8 MLP compute dominates prefill time.

## Artifact

- Model: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Mode: exact-lowram Q8 transformer keep-IO rowchunks
- Prompt: `Answer yes or no: is fire cold?`
- Chat template: `llama3`

## Baseline

R78 non-traced baseline:

- Output: `No`
- TTFT/prefill: 26.75 s
- MLP total: 20,324.73 ms
- Decode: 0.87 tok/s
- Peak transient memory: 1,050,673,152 bytes
- Max RSS: 2,233,090,048 bytes

## Change

The Q8 streaming linear path now uses a stack-local scaled block only for full 32-element Q8 blocks when `config.batch > 1`.

The path keeps exact math and does not allocate heap scratch:

- convert 32 signed Q8 weights to scaled `f32` once per block
- reuse those 32 scaled weights across all prompt tokens in the batch
- leave batch-1 decode and partial-row behavior unchanged

## Command

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases"
```

## Verification

```sh
cargo test -p rllm-runtime q8_0_scaled_block_applies_scale_once
cargo test -p rllm-runtime q8_0
cargo test -p rllm-cli --bin llama-test
cargo build --release --bin llama-test
```

Results:

- `q8_0_scaled_block_applies_scale_once`: passed
- `cargo test -p rllm-runtime q8_0`: passed 9 tests
- `cargo test -p rllm-cli --bin llama-test`: passed 21 tests
- release build passed

## Results

Both benchmark runs kept the answer correct:

```text
No
```

| Run | TTFT / prefill | MLP total | Decode | E2E | Peak transient | Max RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R78 baseline | 26.75 s | 20,324.73 ms | 0.87 tok/s | 0.07 tok/s | 1,050,673,152 bytes | 2,233,090,048 bytes |
| R80 run 1 | 23.62 s | 18,741.00 ms | 1.39 tok/s | 0.08 tok/s | 1,050,673,152 bytes | 2,076,540,928 bytes |
| R80 run 2 | 22.06 s | 17,174.73 ms | 0.95 tok/s | 0.09 tok/s | 1,050,673,152 bytes | 1,852,915,712 bytes |

R80 run 2 versus R78:

- prefill improved by 4.69 s
- prefill improved by 17.53%
- MLP total improved by 3,150.00 ms
- MLP total improved by 15.50%
- peak transient memory unchanged
- output unchanged

R80 run 1 versus R78:

- prefill improved by 3.13 s
- prefill improved by 11.70%
- MLP total improved by 1,583.73 ms
- MLP total improved by 7.79%
- peak transient memory unchanged
- output unchanged

## Bottleneck Tag

CPU arithmetic.

## Analysis

R80 validates R79's attribution. Removing repeated Q8 signed-byte to `f32` conversion inside each prompt-token dot product reduces exact prefill time while preserving the same answer and memory profile.

The improvement applies to `batch > 1` prefill only. Decode remains batch-1 and intentionally keeps the existing row fast path. This avoids repeating the rejected R78 row-batch traversal change.

The up-projection timing remains noisier than gate/down in both R80 runs, but total MLP time still improves materially. The next step should continue targeting Q8 arithmetic, preferably with a measured helper-level micro path that improves all three MLP projections consistently.

## Decision

Accept. Keep the scaled-block Q8 prefill path.

## Next Experiment

R81 should optimize the exact Q8 `f32_dot_32` inner loop itself:

- keep stack-only memory behavior
- preserve exact output on the Llama 3.2 1B prompt
- benchmark non-traced prefill against R80 run 2 and R78 baseline
- consider multi-accumulator or target-feature-specific dot kernels only if correctness tests stay simple
