# R99: RAMA Trace Projection Profile

Date: 2026-06-16
Owner: RLLM
Status: accepted diagnostic
Folder: success

## Hypothesis

After R96/R98, the next useful prefill optimization should be chosen from real
runtime attribution, not another blind Q8 micro-kernel. R99 tests whether the
existing RAMA trace can identify the remaining projection/layer bottleneck.

## Scope

- Mode: exact-lowram diagnostic
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Prompt: `Answer yes or no: is fire cold?`
- Threading: `RLLM_THREADS=1`
- Integrity: `--rama-integrity unchecked`
- Diagnostics: `--profile-phases`, `--rama-trace`, `RLLM_Q8_KERNEL_PROFILE=1`
- Bottleneck tag: Q8 prefill projection attribution

R99 made no runtime/kernel/model-format changes.

## Setup

```bash
cargo build --release -p rllm-cli --bin llama-test
RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked --rama-trace target/r99-rama-trace.json" > target/r99-trace.txt 2> target/r99-trace.time
RLLM_THREADS=1 RLLM_Q8_KERNEL_PROFILE=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked --rama-trace target/r99-rama-trace-profile.json" > target/r99-trace-profile.txt 2> target/r99-trace-profile.time
```

## Runtime Results

Both runs kept the answer correct:

```text
No
```

| run | output | context tokens | prefill | decode | MLP total | attention total | lm_head | peak transient | max RSS | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trace | No | 55 | 9.29s | 1.43 tok/s | 7040.58ms | 1404.54ms | 844.51ms | 1,050,673,152 | 1,887,027,200 | 14.17s |
| trace + Q8 profile | No | 55 | 12.45s | 0.59 tok/s | 9280.63ms | 1852.92ms | 1319.62ms | 1,050,673,152 | 1,649,164,288 | 16.65s |

The profiled run is slower because `RLLM_Q8_KERNEL_PROFILE=1` times the hot
path. Use the normal trace run for baseline timing and the profiled run for
branch attribution.

## Trace Summary

Normal trace bucket totals:

| bucket | events | total |
|---|---:|---:|
| `mlp.gate_proj` | 576 | 2384.54ms |
| `mlp.down_proj` | 576 | 1785.53ms |
| `mlp.up_proj` | 576 | 1357.44ms |
| `attention.q_proj` | 160 | 427.70ms |
| `attention.o_proj` | 160 | 425.17ms |
| `attention.v_proj` | 64 | 79.57ms |
| `attention.k_proj` | 64 | 78.50ms |

Normal trace phase totals:

| phase | events | total |
|---|---:|---:|
| `chunk_compute_closure` | 2176 | 6538.46ms |
| `chunk_decode` | 2176 | 2394.03ms |
| `chunk_read` | 3178 | 2.30ms |

Compute-only projection totals from the normal trace:

| bucket | events | compute time |
|---|---:|---:|
| `mlp.gate_proj` | 576 | 2384.54ms |
| `mlp.down_proj` | 576 | 1785.53ms |
| `mlp.up_proj` | 576 | 1357.44ms |
| `attention.q_proj` | 160 | 427.70ms |
| `attention.o_proj` | 160 | 425.17ms |
| `attention.v_proj` | 64 | 79.57ms |
| `attention.k_proj` | 64 | 78.50ms |

Top layer/projection pairs in `chunk_compute_closure`:

| pair | compute time |
|---|---:|
| layer 06 `mlp.gate_proj` | 153.28ms |
| layer 09 `mlp.gate_proj` | 153.27ms |
| layer 04 `mlp.gate_proj` | 151.33ms |
| layer 08 `mlp.gate_proj` | 151.20ms |
| layer 07 `mlp.gate_proj` | 151.19ms |
| layer 05 `mlp.gate_proj` | 149.52ms |
| layer 10 `mlp.gate_proj` | 149.50ms |
| layer 00 `mlp.gate_proj` | 149.32ms |
| layer 03 `mlp.gate_proj` | 149.30ms |
| layer 01 `mlp.gate_proj` | 149.15ms |

Top layers by total `chunk_compute_closure`:

| layer | compute time |
|---|---:|
| 09 | 420.00ms |
| 10 | 415.31ms |
| 12 | 415.07ms |
| 11 | 412.65ms |
| 07 | 411.49ms |
| 15 | 411.34ms |
| 13 | 411.10ms |
| 14 | 410.61ms |
| 06 | 410.51ms |
| 08 | 409.67ms |

## Q8 Branch Profile

Profiled trace run:

| path | calls | blocks | batch items | elapsed |
|---|---:|---:|---:|---:|
| `batch_gt1_scaled` | 30,408,704 | 30,408,704 | 1,611,661,312 | 5853.47ms |
| `batch1_complete_linear` | 800 | 22,020,096 | 800 | 251.98ms |
| `batch1_complete_multiply` | 288 | 8,388,608 | 288 | 96.20ms |

## Analysis

The remaining normal-run prefill cost is not disk I/O. Trace `chunk_read` is
only `2.30ms`, while `chunk_compute_closure` is `6538.46ms`.

The hottest bucket is not a single bad layer. The top layers are tightly
clustered around `410-420ms`, and the top ten layer/projection pairs are all
`mlp.gate_proj`. This means the next fix should attack repeated Q8 MLP work
across all layers, not a one-layer special case.

The branch profile agrees with the trace: `batch_gt1_scaled` remains the only
first-order Q8 kernel path. Batch-1 decode paths are too small to be the prefill
fix.

R99 also explains why previous gate/up fusion attempts were bad targets: the
normal trace has `mlp.gate_proj` and `mlp.down_proj` above `mlp.up_proj`, while
R90's gate/up fusion serialized hot work and regressed runtime.

## Decision

accepted diagnostic

Reason: R99 produced a precise prefill bottleneck map without changing runtime
behavior. Output stayed correct, peak transient memory stayed unchanged, and the
trace identifies the next target.

Paper value:

- useful post-NEON attribution evidence
- confirms that RLLM is now compute-bound in exact Q8 prefill, not I/O-bound
- supports a structured R100 target around MLP gate/down `batch_gt1_scaled`

## Next Experiment

R100 should target `batch_gt1_scaled` for MLP gate/down across all layers.

Recommended direction: add a lab-gated `REEDOWN/REEGATE` style kernel or bounded
layout experiment that reduces repeated per-block scale/dequant and dot work for
MLP gate/down without increasing resident RAM. Do not spend R100 on batch-1
decode, single-layer scheduling, or gate/up fusion.
