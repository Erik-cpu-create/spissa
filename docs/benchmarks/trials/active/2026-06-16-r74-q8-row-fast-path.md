# Trial: R74 Q8_0 Batch-1 Row Fast Path

Date: 2026-06-16
Owner: RLLM
Status: success with remaining Q8 multiply/argmax opportunity
Folder: active

## Hypothesis

R73 removed the Q8_0 F32 dequant scratch but still accumulated one 32-value
block at a time, loading and storing the output row once per block. SmolLM2 Q8
projection chunks are row-aligned and have `in_features` divisible by 32, so a
batch-1 complete-row fast path should accumulate a full row in a local register
and write the output once per row.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/SmolLM2-135M-Instruct-q8_transformer_keepio.spsa`
- Architecture: SmolLM2/Llama-compatible decoder
- Target device/profile: local macOS CPU, release binary
- Bottleneck tag: Q8 direct dot | row locality | decode CPU

## Change

- Added `accumulate_q8_0_chunk_batch1_complete_rows`.
- It activates only when:
  - `batch == 1`
  - `in_features` is divisible by 32
  - chunk starts on a row boundary
  - chunk contains complete rows
- Otherwise it returns `false` and the existing Q8 block path remains the
  fallback.

## Verification

Red test before implementation:

```text
error[E0425]: cannot find function `accumulate_q8_0_chunk_batch1_complete_rows` in this scope
```

After implementation:

```bash
cargo test -p rllm-runtime q8_0_batch1_row_fast_path -- --nocapture
cargo test -p rllm-runtime
cargo build --release -p rllm-cli --bins
```

Results:

- targeted row-fast tests: 2 passed
- full `rllm-runtime`: 237 passed
- release build: passed

## Q8 ChatML Matrix

Command:

```bash
set -u
out=/tmp/rllm-smollm-q8-rowfast-matrix-20260616.txt
: > "$out"
model="models/SmolLM2-135M-Instruct-q8_transformer_keepio.spsa"
prompts=(
  "Answer in one short sentence: what is 2 plus 2?"
  "Answer in one short sentence: what color is the sky on a clear day?"
  "Translate to Indonesian: I am learning Rust."
  "List exactly three fruits separated by commas."
  "Answer yes or no: is fire cold?"
)
for prompt in "${prompts[@]}"; do
  printf '%s\nquit\n' "$prompt" \
    | /usr/bin/time -l target/release/llama-test \
        --model "$model" \
        --ctx 512 \
        --max-new-tokens 32 \
        --chat-template chatml
done
```

| prompt | tokens | TTFT | decode tok/s | E2E tok/s | RSS bytes | peak transient bytes | output |
|---|---:|---:|---:|---:|---:|---:|---|
| `2 plus 2` | 12 | 2.59s | 18.42 | 3.77 | 379158528 | 113246208 | `I'm ready to help you with your math questions.` |
| `sky` | 14 | 2.57s | 18.03 | 4.25 | 379387904 | 113246208 | `The sky on a clear day is a deep, rich blue.` |
| `translate` | 32 | 2.29s | 16.70 | 7.72 | 380436480 | 113246208 | `Kalau mengatakan Rust, jika juga menggunakan komputer, jika juga menggunak` |
| `fruits` | 13 | 2.23s | 17.91 | 4.48 | 379469824 | 113246208 | `1. Banana` / `2. Apple` / `3. Orange` |
| `fire cold` | 7 | 2.24s | 17.74 | 2.71 | 378765312 | 113246208 | `Yes. Fire is cold.` |

## Comparison

R73 Q8 direct-dot:

- decode: 15.17-17.88 tok/s
- RSS: about 379-380 MB
- peak transient: 113246208 bytes

R74 Q8 row fast path:

- decode: 16.70-18.42 tok/s
- RSS: about 379-380 MB
- peak transient: 113246208 bytes
- output text unchanged from R73/R72 Q8 for this matrix

## Decision

Accept the row-fast path for `streaming_tile_linear_from_model`.

It is a scoped decode/prefill speed win on the row-aligned SmolLM2 Q8 artifact
without changing output text or memory budget behavior. The next useful slice is
the same complete-row fast path for Q8 `multiply_into` and argmax, then SIMD dot
inside `q8_0_dot_i8_f32`.
