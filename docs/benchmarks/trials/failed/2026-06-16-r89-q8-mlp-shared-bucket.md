# R89: Q8 MLP Shared Bucket Micro-Kernel Trial

## Status

Failed.

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
    --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa \
    --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" \
    > /tmp/r89-run${i}.txt 2> /tmp/r89-run${i}.time
done
```

Baseline to beat (R88):
- Prefill: 10.24s (unchecked)
- Output: `No`

## Results

| run | output | context tokens | prefill | decode | MLP total | peak transient | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | No | 55 | 13.00s | 1.62 tok/s | 9918.72ms | 1,050,673,152 | 14.2s |
| 2 | No | 55 | 12.23s | 0.42 tok/s | 10097.59ms | 1,050,673,152 | 16.5s |
| 3 | No | 55 | 13.44s | 0.96 tok/s | 10207.50ms | 1,050,673,152 | 15.6s |

## Analysis

The shared bucket micro-kernel correctly achieved exact output (`No`) and kept peak transient memory flat at 1,050,673,152 bytes. However, the prefill latency (best `12.23s`) is slower than the R88 unchecked baseline (`10.24s`). 
The `mlp_total` in this trial ranges from `9918ms` to `10207ms`, whereas the R88 baseline was `8380ms`.
The additional overhead comes from the memory bandwidth required to repack the inputs (`mlp_input` and `gate_up_output`), allocating intermediate buffers (`up_output`), and performing the sequential element-wise multiplication (`gate_up_output * up_output`). The CPU cycles saved from contiguous L1 access during the dot product were outweighed by the extra memory passes over the ~1.8 MB activations.

## Decision

Failed. Best prefill is slower than the R88 baseline (12.23s > 10.24s). The shared bucket micro-kernel approach does not satisfy the gate. Runtime for this experiment should be reverted.
