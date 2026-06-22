# Trial: R76 Q8_0 Fused Gate/Up Rejection

Date: 2026-06-16
Owner: RLLM
Status: failed
Folder: active

## Hypothesis

R75 improved exact-lowram Q8 decode by adding complete-row fast paths for
linear, multiply, and argmax. The next candidate was fusing Q8_0 gate/up MLP
input projections so `silu(gate_proj(x)) * up_proj(x)` could be computed in one
combined pass rather than two projection calls plus a multiply.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/SmolLM2-135M-Instruct-q8_transformer_keepio.spsa`
- Architecture: SmolLM2/Llama-compatible decoder
- Target device/profile: local macOS CPU, release binary
- Bottleneck tag: Q8 MLP gate/up fusion

## Experiment

Implemented and tested a row-aligned Q8_0 fused gate/up path locally:

- `streaming_silu_gate_up_from_model` accepted Q8_0 gate/up tensors.
- A Q8_0 fused gate/up kernel computed gate and up row dots together.
- A targeted test verified materialized Q8 output correctness.

The code was reverted before commit because real model performance regressed.

## Verification Before Revert

```bash
cargo test -p rllm-runtime streaming_silu_gate_up_matches_materialized_q8_0_batch1 -- --nocapture
cargo test -p rllm-runtime
cargo build --release -p rllm-cli --bins
```

Results before revert:

- targeted Q8 fused gate/up test: 1 passed
- full `rllm-runtime`: 240 passed
- release build: passed

## Q8 ChatML Matrix

Command:

```bash
set -u
out=/tmp/rllm-smollm-q8-fused-gateup-matrix-20260616.txt
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
| `2 plus 2` | 12 | 4.22s | 14.12 | 2.40 | 269090816 | 113246208 | `I'm ready to help you with your math questions.` |
| `sky` | 14 | 2.65s | 17.48 | 4.12 | 379273216 | 113246208 | `The sky on a clear day is a deep, rich blue.` |
| `translate` | 32 | 2.34s | 17.84 | 7.84 | 380305408 | 113246208 | `Kalau mengatakan Rust, jika juga menggunakan komputer, jika juga menggunak` |
| `fruits` | 13 | 2.24s | 18.19 | 4.48 | 379355136 | 113246208 | `1. Banana` / `2. Apple` / `3. Orange` |
| `fire cold` | 7 | 2.34s | 18.42 | 2.63 | 379650048 | 113246208 | `Yes. Fire is cold.` |

## Comparison

R75 Q8 row-fast linear/multiply/argmax:

- decode: 18.60-19.63 tok/s
- peak transient: 113246208 bytes
- output unchanged

R76 fused Q8 gate/up experiment:

- decode: 14.12-18.42 tok/s
- peak transient unchanged
- output unchanged

## Decision

Reject Q8_0 fused gate/up in this form.

The combined pass was correct but slower on the measured matrix, likely because
the existing separate row-fast gate and up projections have better local row
accumulation and simpler state flow. The next useful speed target remains
`q8_0_dot_i8_f32` SIMD/vectorization, not Q8 gate/up fusion.
