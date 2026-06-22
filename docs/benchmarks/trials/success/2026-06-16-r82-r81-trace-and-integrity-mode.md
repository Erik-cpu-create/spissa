# R82: R81 Trace and Integrity Mode

## Status

Success with explicit trust tradeoff.

## Hypothesis

After R80/R81 reduce Q8 MLP arithmetic, checksum verification may be a material remaining cost. If so, an explicit trusted-artifact integrity mode can improve benchmark latency without changing math quality, while preserving the default verified mode.

## Artifact

- Model: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Mode: exact-lowram Q8 transformer keep-IO rowchunks
- Prompt: `Answer yes or no: is fire cold?`
- Chat template: `llama3`

## Baselines

- R78 prefill: 26.75 s
- R80 best prefill: 22.06 s
- R81 best prefill: 21.41 s

## Change

Added explicit `RamaIntegrityMode::Unchecked` and `llama-test --rama-integrity unchecked`.

Default `llama-test` behavior remains `verify-once`. The new mode is opt-in for trusted local artifacts and skips runtime SHA-256 verification while preserving the exact same Q8 math path.

## Trace Command

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-trace target/r82-r81-trace.json"
```

## Trace Result

The traced R81 run kept output correct:

```text
No
```

| Metric | Value |
| --- | ---: |
| TTFT / prefill | 21.50 s |
| MLP total | 16,649.23 ms |
| Peak transient memory | 1,050,673,152 bytes |
| Max RSS | 2,019,819,520 bytes |

Trace phase totals:

| Phase | Events | Total ms | Share of traced time |
| --- | ---: | ---: | ---: |
| chunk_compute_closure | 2,176 | 14,113.04 | 64.23% |
| chunk_compressed_checksum | 1,589 | 5,392.23 | 24.54% |
| chunk_original_checksum | 1,088 | 2,282.86 | 10.39% |
| chunk_decode | 2,176 | 182.41 | 0.83% |
| chunk_read | 3,178 | 3.38 | 0.02% |

Checksum verification total was 7,675.09 ms, or 34.93% of traced time. That justified adding an explicit trusted mode.

## Trusted Benchmark Command

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked"
```

## Verification

```sh
cargo test -p rllm-runtime unchecked_integrity_records_no_checksum_events
cargo test -p rllm-runtime integrity
cargo test -p rllm-cli --bin llama-test args_default_to_verify_once_integrity_and_accept_unchecked
cargo test -p rllm-cli --bin llama-test
cargo build --release --bin llama-test
```

Results:

- `unchecked_integrity_records_no_checksum_events`: passed
- `cargo test -p rllm-runtime integrity`: passed 2 tests
- `args_default_to_verify_once_integrity_and_accept_unchecked`: passed
- `cargo test -p rllm-cli --bin llama-test`: passed 22 tests
- release build passed

## Trusted Results

Both unchecked runs kept the answer correct:

```text
No
```

| Run | Integrity | TTFT / prefill | MLP total | Decode | E2E | Peak transient | Max RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R78 baseline | verify-once | 26.75 s | 20,324.73 ms | 0.87 tok/s | 0.07 tok/s | 1,050,673,152 bytes | 2,233,090,048 bytes |
| R81 best | verify-once | 21.41 s | 17,069.33 ms | 1.45 tok/s | 0.09 tok/s | 1,050,673,152 bytes | 2,536,652,800 bytes |
| R82 unchecked run 1 | unchecked | 17.23 s | 14,025.29 ms | 1.31 tok/s | 0.11 tok/s | 1,050,673,152 bytes | 2,114,551,808 bytes |
| R82 unchecked run 2 | unchecked | 16.38 s | 14,096.94 ms | 1.45 tok/s | 0.12 tok/s | 1,050,673,152 bytes | 2,199,977,984 bytes |

R82 unchecked best versus R78:

- prefill improved by 10.37 s
- prefill improved by 38.77%
- MLP total improved by 6,227.79 ms
- MLP total improved by 30.64%
- peak transient memory unchanged
- output unchanged

R82 unchecked best versus R81 best:

- prefill improved by 5.03 s
- prefill improved by 23.49%
- MLP total improved by 2,972.39 ms
- MLP total improved by 17.41%
- peak transient memory unchanged
- output unchanged

## Bottleneck Tag

CPU arithmetic and checksum verification.

## Analysis

R82 confirms that after R80/R81, checksum verification is a first-order latency cost for this one-turn low-RAM benchmark. Skipping it explicitly for trusted local artifacts gives the largest post-R78 improvement so far.

The mode is not a math shortcut: Q8 compute remains exact for the quantized weights. The tradeoff is artifact-integrity validation. Default behavior remains `verify-once`, so users who want corruption detection keep it by default.

For hackathon/funding claims, this should be described as two modes:

- `verify-once`: safer default, lower-RAM exact Q8, prefill best observed at 21.41 s
- `unchecked`: trusted local artifact fast mode, same exact math, prefill best observed at 16.38 s

## Decision

Accept. Keep explicit `unchecked` integrity mode and keep default `verify-once`.

## Next Experiment

R83 should benchmark a longer generation prompt to separate prefill improvement from decode token/s:

- use `--rama-integrity unchecked`
- generate at least 32 tokens
- record TTFT, decode tok/s, E2E tok/s, peak transient memory, and output sanity
- then decide whether decode token/s or remaining prefill compute is next
