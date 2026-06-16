# R89: Q8 MLP Shared Bucket Micro-Kernel Trial

## Status

Active.

## Hypothesis

By consolidating the Q8 MLP dot-product and repacking work into a shared bucket micro-kernel, we can measurably decrease the prefill CPU time while preserving stable exact-lowram output semantics.

## Scope

- Mode: exact-lowram
- Component: Q8 MLP layers (`gate`, `up`, `down` projections)
- Target device/profile: Mac (CPU)
- Constraint: strictly avoid broad architectural refactoring; isolate the change to a new micro-kernel implementation for Q8 MLP.

## Setup

Commands to run (using the benchmark harness established in R88):

```bash
cargo build --release -p rllm-cli --bin llama-test

for i in 1 2 3; do
  /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm \
    --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" \
    > /tmp/r89-run${i}.txt 2> /tmp/r89-run${i}.time
done
```

Baseline to beat (R88):
- Prefill: 10.24s (unchecked)
- Output: `No`

## Results

| run | output | context tokens | prefill | decode | MLP total | peak transient | max RSS | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | | | |
| 2 | | | | | | | | |
| 3 | | | | | | | | |

## Analysis

TBD

## Decision

TBD
