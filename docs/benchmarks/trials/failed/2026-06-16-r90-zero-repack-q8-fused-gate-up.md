# R90: Zero-Repack Q8 Fused Gate/Up

## Status

Failed.

## Hypothesis

Fusing Q8_0 gate/up accumulation without repacking activations should reduce MLP prefill time by removing one projection pass boundary and one activation memory pass while preserving exact-lowram output.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Architecture: Q8_0 streaming MLP prefill
- Target device/profile: Mac CPU
- Expected bottleneck: CPU arithmetic / memory bandwidth
- Bottleneck tag: CPU arithmetic / cache locality

## Baseline

- R88 best unchecked prefill: `10.24s`
- R88 best MLP total: `8380ms`
- R89 failed best prefill: `12.23s`
- Required output: `No`
- Required peak transient ceiling: `1,050,673,152 bytes`

## Commands

```bash
cargo build --release -p rllm-cli --bin llama-test

for i in 1 2 3; do
  /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa \
    --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" \
    > /tmp/r90-run${i}.txt 2> /tmp/r90-run${i}.time
done
```

## Results

| run | output | context tokens | prefill | decode | MLP total | gate | up | down | peak transient | max RSS | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | No | 55 | 19.18s | 1.36 tok/s | 16,087.49ms | 12,805.94ms | 0.00ms | 3,279.79ms | 1,050,673,152 bytes | 1,637,941,248 bytes | 24.02s |
| 2 | No | 55 | 19.21s | 1.14 tok/s | 16,196.78ms | 13,125.34ms | 0.00ms | 3,069.58ms | 1,050,673,152 bytes | 1,642,053,632 bytes | 23.04s |
| 3 | No | 55 | 18.84s | 1.10 tok/s | 15,686.98ms | 12,515.00ms | 0.00ms | 3,170.13ms | 1,050,673,152 bytes | 1,767,571,456 bytes | 22.54s |

## Analysis

The fused Q8 gate/up path preserved output (`No`) and did not increase the internal peak transient memory, but it made prefill much slower. Best prefill regressed from the R88 baseline `10.24s` to `18.84s`, and best MLP total regressed from `8380ms` to `15686.98ms`.

The likely cause is that the zero-repack fused path moved the expensive work into the `gate` bucket and serialized gate/up accumulation through a less optimized paired loop. It removed the separate `up` bucket, but the combined gate bucket became much larger than the original gate plus optimized multiply-into path.

## Decision

Failed. Runtime changes were reverted because the gate failed:

- best prefill `18.84s` is not `< 10.24s`
- best MLP total `15686.98ms` is not `< 8380ms`
- output and peak memory were acceptable, but speed regressed too much

Paper value: useful negative evidence. Do not pursue this fused gate/up shape for Q8 prefill.

## Next Experiment

Stop gate/up fusion attempts for this artifact. The next useful stage should measure an isolated microbenchmark for the existing accepted R83 Q8 block dot helpers before touching full runtime again.
