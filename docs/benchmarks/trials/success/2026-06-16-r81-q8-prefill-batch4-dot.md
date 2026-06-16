# R81: Q8 Prefill Batch4 Dot

## Status

Success.

## Hypothesis

The R80 scaled-block Q8 prefill path can be improved by processing four prompt-token rows per scaled block. This should reduce repeated scaled-weight loads and per-dot helper overhead while keeping exact output and stack-only memory behavior.

## Artifact

- Model: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Mode: exact-lowram Q8 transformer keep-IO rowchunks
- Prompt: `Answer yes or no: is fire cold?`
- Chat template: `llama3`

## Baselines

R78:

- Output: `No`
- TTFT/prefill: 26.75 s
- MLP total: 20,324.73 ms

R80 best:

- Output: `No`
- TTFT/prefill: 22.06 s
- MLP total: 17,174.73 ms
- Peak transient memory: 1,050,673,152 bytes

## Change

The R80 scaled-block branch now processes four prompt-token rows per scaled Q8 block:

- `q8_0_scaled_block` still creates one stack-local `[f32; 32]`.
- `accumulate_f32_dot_32_batch4` accumulates four output rows while reading each scaled weight once.
- Remainder batches still use the R80 scalar `f32_dot_32` helper.
- Batch-1 decode, partial blocks, multiply-into, and argmax paths are unchanged.

## Command

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases"
```

## Verification

```sh
cargo test -p rllm-runtime f32_dot_32_batch4_accumulates_four_outputs
cargo test -p rllm-runtime q8_0
cargo test -p rllm-cli --bin llama-test
cargo build --release --bin llama-test
```

Results:

- `f32_dot_32_batch4_accumulates_four_outputs`: passed
- `cargo test -p rllm-runtime q8_0`: passed 9 tests
- `cargo test -p rllm-cli --bin llama-test`: passed 21 tests
- release build passed

## Results

All R81 benchmark runs kept the answer correct:

```text
No
```

| Run | TTFT / prefill | MLP total | Decode | E2E | Peak transient | Max RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R78 baseline | 26.75 s | 20,324.73 ms | 0.87 tok/s | 0.07 tok/s | 1,050,673,152 bytes | 2,233,090,048 bytes |
| R80 best | 22.06 s | 17,174.73 ms | 0.95 tok/s | 0.09 tok/s | 1,050,673,152 bytes | 1,852,915,712 bytes |
| R81 run 1 | 22.73 s | 18,385.10 ms | 1.59 tok/s | 0.09 tok/s | 1,050,673,152 bytes | 2,187,902,976 bytes |
| R81 run 2 | 21.99 s | 17,081.33 ms | 0.43 tok/s | 0.08 tok/s | 1,050,673,152 bytes | 2,639,806,464 bytes |
| R81 run 3 | 21.41 s | 17,069.33 ms | 1.45 tok/s | 0.09 tok/s | 1,050,673,152 bytes | 2,536,652,800 bytes |

R81 best versus R78:

- prefill improved by 5.34 s
- prefill improved by 19.96%
- MLP total improved by 3,255.40 ms
- MLP total improved by 16.02%
- peak transient memory unchanged
- output unchanged

R81 best versus R80 best:

- prefill improved by 0.65 s
- prefill improved by 2.95%
- MLP total improved by 105.40 ms
- MLP total improved by 0.61%
- peak transient memory unchanged
- output unchanged

## Bottleneck Tag

CPU arithmetic.

## Analysis

R81 gives a smaller but real follow-up to R80. The best run improves prefill from R80's 22.06 s to 21.41 s and keeps the exact answer stable.

The MLP delta versus R80 is small, which means most of the easy repeated-conversion win was already captured by R80. R81's value is mainly reducing overhead around repeated dot helper calls and scaled-weight reloads for prompt batches.

`/usr/bin/time -l` max RSS varied upward during R81 runs, but RLLM's internal peak transient memory stayed unchanged at 1,050,673,152 bytes. Treat RSS as noisy for this tiny benchmark and keep watching it in larger multi-prompt runs.

## Decision

Accept. Keep the batch4 dot path.

## Next Experiment

R82 should stop micro-optimizing this exact helper blindly and re-profile after R81:

- run `--rama-trace` again on the R81 binary
- compare phase and tensor-bucket attribution to R79
- decide whether the remaining bottleneck is still Q8 MLP arithmetic or checksum/lm-head overhead
