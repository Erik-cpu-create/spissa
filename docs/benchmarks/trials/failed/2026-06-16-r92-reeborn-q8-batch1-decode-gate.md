# R92: REEBORN-Q8 Batch1 Decode Gate

Date: 2026-06-16
Owner: RLLM
Status: rejected
Folder: failed

## Hypothesis

R91 showed that pre-scaling Q8 blocks can beat repeated signed-byte conversion
in a prefill-shaped lab. R92 tests whether the same idea works for batch1
decode-shaped complete-row Q8 paths before promoting a runtime kernel lineage.

## Scope

- Mode: exact-lowram lab plus conditional runtime trial
- REE kernel: `REEBORN-Q8-BATCH1-LAB`; runtime candidate `REEBORN-Q8-BATCH1` rejected
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`
- Architecture: Q8_0 complete-row batch1 decode path
- Target device/profile: CPU-only, single-thread benchmark
- Expected bottleneck: Q8 MLP decode arithmetic
- Bottleneck tag: CPU arithmetic / micro-kernel

## Setup

Lab gate:

```bash
cargo test -p rllm-runtime q8_kernel_lab -- --nocapture
cargo build --release -p rllm-runtime --bin q8-microbench
target/release/q8-microbench \
  --json target/r92-q8-batch1.json \
  --markdown target/r92-q8-batch1.md \
  --iters 2000 \
  --batch 1
```

Runtime trial:

```bash
cargo build --release -p rllm-cli --bin llama-test
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" > "target/r92-run${i}.txt" 2> "target/r92-run${i}.time"
done
```

Runtime context:

- build profile: release
- CPU: Apple A18 Pro
- RAM: 8589934592 bytes
- OS: Darwin 25.5.0 arm64
- relevant env/config: `RLLM_THREADS=1`, `--rama-integrity unchecked`, `--profile-phases`

## Lab Results

| variant | elapsed ns | speedup vs baseline | max abs diff | checksum |
|---|---:|---:|---:|---:|
| `baseline_i8_dot32_batch4` | 5270625 | 1.000x | 0.00000000 | 0.240234 |
| `scaled_f32_dot32_batch4` | 4106292 | 1.284x | 0.00000000 | 0.240234 |
| `unrolled_i8_dot32_batch4` | 1356875 | 3.884x | 0.00000000 | 0.240234 |
| `baseline_i8_dot32_batch1_row` | 2572917 | 1.000x | 0.00000000 | 0.240234 |
| `scaled_f32_dot32_batch1_row` | 1604875 | 1.603x | 0.00000000 | 0.240234 |

Lab gate result:

- batch1 candidate diff: `0`
- batch1 candidate speedup: `1.603x`
- lab gate: passed

## Runtime Results

All R92 runtime trial runs kept the answer correct:

```text
No
```

R88 single-thread baseline from `2026-06-16-r88-rllm-vs-ollama-token-alignment.md`:

| run | output | context tokens | prefill | decode | MLP total | peak transient | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|
| R88 ST 1 | No | 55 | 12.59s | 1.53 tok/s | 10.31s | 1,050,673,152 | 16.25s |
| R88 ST 2 | No | 55 | 11.34s | 1.48 tok/s | 9.24s | 1,050,673,152 | 14.68s |
| R88 ST 3 | No | 55 | 11.30s | 1.53 tok/s | 9.18s | 1,050,673,152 | 14.62s |

R92 runtime candidate:

| run | output | context tokens | prefill | decode | MLP total | gate | up | down | peak transient | max RSS | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R92 1 | No | 55 | 13.95s | 0.41 tok/s | 10454.12ms | 3613.24ms | 3283.84ms | 3544.35ms | 1,050,673,152 | 1,625,817,088 | 20.59s |
| R92 2 | No | 55 | 13.46s | 0.58 tok/s | 10161.56ms | 3602.05ms | 3122.90ms | 3422.91ms | 1,050,673,152 | 1,671,479,296 | 18.09s |
| R92 3 | No | 55 | 12.59s | 1.17 tok/s | 9452.36ms | 3361.54ms | 2872.84ms | 3205.17ms | 1,050,673,152 | 1,645,084,672 | 16.71s |

## Analysis

The lab result was positive, but the runtime trial failed. The best R92 decode
row reached only `1.17 tok/s`, below the R88 single-thread baseline range of
`1.48-1.53 tok/s`. Prefill also did not improve over the R88 best single-thread
baseline and stayed at `12.59-13.95s`.

The internal peak transient memory stayed unchanged at `1,050,673,152` bytes and
the output stayed correct, so the failure is speed-only. The likely cause is
that the isolated batch1 row loop does not represent the runtime's full decode
cost well enough; constructing scaled blocks inside the complete-row path did
not translate into a measured end-to-end decode win.

The runtime code was reverted after this failed gate. Only the lab benchmark
extension and this negative report remain.

## Decision

rejected

Reason: Runtime candidate `REEBORN-Q8-BATCH1` failed the decode speed gate even
though `REEBORN-Q8-BATCH1-LAB` passed the microbench gate.

Paper value:

- use as negative evidence for batch1 scaled-block runtime promotion
- use as process evidence that REE kernels require both lab and runtime gates

## Next Experiment

Do not promote `REEBORN-Q8-BATCH1`.

The next kernel plan should target the broader MLP prefill path or a profiling
slice that separates prompt prefill from decode more cleanly. A future runtime
candidate should not rely on a batch1-only complete-row scaled-block change.
