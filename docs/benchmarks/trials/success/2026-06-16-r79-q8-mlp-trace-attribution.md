# R79: Q8 MLP Trace Attribution

## Status

Success with diagnostic scope.

## Hypothesis

The Llama 3.2 1B Q8 exact-lowram prefill slowdown can be attributed to a concrete event class in the existing RAMA trace: chunk compute closure, chunk decode, chunk read, or checksum verification. R78 showed that a naive row-batch compute rewrite did not help, so the next optimization must be chosen from measured attribution.

## Artifact

- Model: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Mode: exact-lowram Q8 transformer keep-IO rowchunks
- Prompt: `Answer yes or no: is fire cold?`
- Chat template: `llama3`
- Trace: `target/r79-q8-mlp-trace.json`

## Command

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-trace target/r79-q8-mlp-trace.json"
```

## Verification

```sh
cargo test -p rllm-cli --bin llama-test
cargo build --release --bin llama-test
```

Result: `cargo test -p rllm-cli --bin llama-test` passed 21 tests.

## Results

The answer remained correct:

```text
No
```

| Metric | Value |
| --- | ---: |
| TTFT / prefill | 32.48 s |
| Decode | 0.53 tok/s |
| End-to-end | 0.06 tok/s |
| Generated tokens | 2 |
| Context tokens | 55 |
| Peak transient memory | 1,050,673,152 bytes |
| Max RSS | 1,778,565,120 bytes |
| Wall time | 37.56 s |
| User CPU time | 34.05 s |
| Sys CPU time | 0.78 s |

`PrefillProfile`:

| Subphase | Value |
| --- | ---: |
| prefill_total | 32,479.92 ms |
| profiled | 32,479.91 ms |
| transformer | 30,172.93 ms |
| attention_total | 5,218.19 ms |
| mlp_total | 24,954.51 ms |
| final_norm | 0.12 ms |
| lm_head | 2,306.70 ms |
| q | 2,092.73 ms |
| k | 519.75 ms |
| v | 512.12 ms |
| attn | 30.39 ms |
| gate | 8,342.03 ms |
| up | 8,283.33 ms |
| down | 8,315.59 ms |

Trace totals by phase:

| Phase | Events | Total ms | Share of traced time |
| --- | ---: | ---: | ---: |
| chunk_compute_closure | 2,176 | 24,840.65 | 72.97% |
| chunk_compressed_checksum | 1,589 | 5,249.15 | 15.42% |
| chunk_original_checksum | 1,088 | 2,600.72 | 7.64% |
| chunk_decode | 2,176 | 1,349.88 | 3.97% |
| chunk_read | 3,178 | 3.46 | 0.01% |

Trace totals by compute tensor bucket:

| Bucket | Events | Total ms |
| --- | ---: | ---: |
| mlp.gate_proj | 576 | 6,887.88 |
| mlp.down_proj | 576 | 6,866.92 |
| mlp.up_proj | 576 | 6,799.01 |
| attention.q_proj | 160 | 1,732.60 |
| attention.o_proj | 160 | 1,696.89 |
| attention.k_proj | 64 | 431.14 |
| attention.v_proj | 64 | 426.22 |

MLP compute buckets total 20,553.81 ms. That is 60.37% of traced time and 82.74% of `chunk_compute_closure`.

## Analysis

R79 confirms the Llama 3.2 1B Q8 exact-lowram bottleneck is not storage IO and not Q8 decode/unpack. `chunk_read` is effectively zero in this run, and `chunk_decode` is only 3.97% of traced time.

The dominant cost is CPU arithmetic inside chunk compute closures, specifically the three MLP projections. This agrees with the R78 phase profile, where MLP prefill dominated the transformer phase and the attempted row-batch fast path did not improve the kernel.

Trace overhead is visible: R78 baseline prefill was 26.75 s, while this traced run measured 32.48 s. The trace should therefore be used for attribution, not as the speed baseline. The attribution is still actionable because phase and tensor-bucket shares are decisive.

Checksum verification is a secondary cost in the traced run: compressed plus original checksums total 7,849.87 ms. That is worth tracking, but it is not the primary prefill bottleneck and does not explain the MLP projection dominance.

## Bottleneck Tag

CPU arithmetic, with checksum verification as a secondary observed cost.

## Decision

Accept the diagnostic result. The next speed experiment should target Q8 MLP projection arithmetic/layout directly. Do not spend the next iteration on disk IO, chunk read, or Q8 decode/unpack.

## Next Experiment

R80 should implement the smallest measurable Q8 MLP arithmetic improvement:

- keep exact outputs unchanged
- focus on gate/up/down projection inner loops
- benchmark against the R78 non-traced baseline command
- use R79 trace only to validate attribution after any change
