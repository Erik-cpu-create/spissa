# Trial: R75 Q8_0 Row Fast Path For Multiply And Argmax

Date: 2026-06-16
Owner: RLLM
Status: success
Folder: active

## Hypothesis

R74 added a batch-1 complete-row fast path for Q8_0 `streaming_tile_linear`.
The remaining Q8_0 decode hot paths still used block-level output updates in
`multiply_into` and argmax. Applying the same complete-row fast path there
should improve decode speed while keeping memory and output unchanged.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/SmolLM2-135M-Instruct-q8_transformer_keepio.rllm`
- Architecture: SmolLM2/Llama-compatible decoder
- Target device/profile: local macOS CPU, release binary
- Bottleneck tag: Q8 row locality | MLP down projection | argmax row scan

## Change

- Added complete-row Q8_0 fast paths for:
  - `accumulate_q8_0_chunk_multiply_into`
  - `accumulate_q8_0_chunk_argmax`
- Refactored complete-row detection into `q8_0_complete_row_span`.
- The fast path activates only for batch-1, row-aligned, complete-row Q8 chunks
  with `in_features` divisible by 32. Other layouts still use the existing Q8
  block fallback.

## Verification

Red tests before implementation:

```text
error[E0425]: cannot find function `accumulate_q8_0_chunk_multiply_into_batch1_complete_rows` in this scope
error[E0425]: cannot find function `accumulate_q8_0_chunk_argmax_batch1_complete_rows` in this scope
```

After implementation:

```bash
cargo test -p rllm-runtime q8_0_batch1_ -- --nocapture
cargo test -p rllm-runtime
cargo build --release -p rllm-cli --bins
```

Results:

- targeted Q8 row-fast tests: 4 passed
- full `rllm-runtime`: 239 passed
- release build: passed

## Q8 ChatML Matrix

Command:

```bash
set -u
out=/tmp/rllm-smollm-q8-rowfast-all-matrix-20260616.txt
: > "$out"
model="models/SmolLM2-135M-Instruct-q8_transformer_keepio.rllm"
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
| `2 plus 2` | 12 | 2.37s | 19.63 | 4.09 | 379076608 | 113246208 | `I'm ready to help you with your math questions.` |
| `sky` | 14 | 2.35s | 19.26 | 4.63 | 378208256 | 113246208 | `The sky on a clear day is a deep, rich blue.` |
| `translate` | 32 | 2.15s | 18.90 | 8.45 | 380256256 | 113246208 | `Kalau mengatakan Rust, jika juga menggunakan komputer, jika juga menggunak` |
| `fruits` | 13 | 2.10s | 18.73 | 4.74 | 379338752 | 113246208 | `1. Banana` / `2. Apple` / `3. Orange` |
| `fire cold` | 7 | 2.14s | 18.60 | 2.84 | 378650624 | 113246208 | `Yes. Fire is cold.` |

## Comparison

R74 Q8 row-fast linear only:

- decode: 16.70-18.42 tok/s
- RSS: about 379-380 MB
- peak transient: 113246208 bytes

R75 Q8 row-fast linear, multiply, and argmax:

- decode: 18.60-19.63 tok/s
- RSS: about 378-380 MB
- peak transient: 113246208 bytes
- output text unchanged from R72/R73/R74 Q8 for this matrix

## Decision

Accept the Q8_0 complete-row fast path for multiply and argmax.

The next practical speed target is SIMD/vectorized `q8_0_dot_i8_f32` or a
larger fused gate/up/down MLP path. Q4 MLP remains rejected for this model
because it caused quality drift.
