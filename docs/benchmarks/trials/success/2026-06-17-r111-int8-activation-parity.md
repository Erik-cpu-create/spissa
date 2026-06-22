# R111: Int8-Activation Parity Validation

Date: 2026-06-17
Owner: RLLM
Status: accepted
Folder: success

## Hypothesis

R110 proved native int8 `sdot` is ~18.9x faster than the f32 baseline in the lab,
but the int8 path requires quantizing activations to int8 (the weights are
already Q8; the f32 path keeps activations in f32). Quantizing activations adds a
small extra error, which conflicts with RLLM's "no quality loss" doctrine unless
validated. R111 tests whether int8 activation quantization preserves the model's
output on the real Llama 3.2 1B Q8 model, at both the token and logit level,
before any runtime kernel is promoted.

## Scope

- Mode: exact-lowram runtime gate (validation only)
- REE kernel lineage: none (validation harness, not a promoted kernel)
- Model/artifact: `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Architecture: Llama 3.2 1B Instruct, chat-template llama3
- Target device/profile: Apple A18 Pro, single-thread (`RLLM_THREADS=1`)
- Bottleneck tag: runtime bug / quality validation

Mechanism added (default off; exact f32 path stays default):

- `RLLM_Q8_ACTIVATION=1` selects an int8-activation path in `accumulate_q8_0_chunk`
  (per-32-segment activation quant, int8×int8 dot via `sdot`, scalar fallback).
  Correctness-only; not yet optimized (no batching/register-block).
- `RLLM_FULL_LOGITS=1` recomputes the full lm_head logits (the fused argmax path
  does not materialize them), so parity can be checked at the logit level. Routed
  through the same kernel, so it reflects the int8 gate end-to-end.
- `llama-test --logits-out <path>` dumps the prefill→first-token logits as JSON.

## Setup

```bash
cargo build --release -p rllm-cli --bin llama-test
# token parity: OFF (f32) vs ON (int8 activations), 3 diverse prompts
for P in "Answer yes or no: is fire cold?" "Answer yes or no: is the sky blue?" "Answer yes or no: is ice hot?"; do
  printf '%s\nquit\n' "$P" | RLLM_THREADS=1 target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 6 --rama-integrity unchecked
  printf '%s\nquit\n' "$P" | RLLM_THREADS=1 RLLM_Q8_ACTIVATION=1 target/release/llama-test --model ... (same)
done
# logit parity: dump first-step logits OFF vs ON with RLLM_FULL_LOGITS=1, compare top-1 / top-10 / max abs diff
```

Runtime context:

- build profile: release
- CPU: Apple A18 Pro (6 core: 2 perf + 4 eff)
- RAM: 8 GB
- OS: macOS (darwin)
- relevant env/config: `RLLM_THREADS=1`, `--rama-integrity unchecked`,
  `--chat-template llama3`, `RLLM_Q8_ACTIVATION`, `RLLM_FULL_LOGITS`

## Results

Token parity (OFF f32 vs ON int8 activations):

| prompt | OFF answer | ON answer | match |
|---|---|---|---|
| is fire cold? | No | No | yes |
| is the sky blue? | Yes | Yes | yes |
| is ice hot? | No | No | yes |

Logit parity (prefill→first-token full-vocab logits, vocab 128256):

| prompt | top-1 off | top-1 on | top-1 match | top-10 overlap | max abs diff (% of peak) |
|---|---:|---:|---|---:|---|
| is fire cold? | 2822 | 2822 | yes | 10/10 | 0.2997 (1.06%) |
| is the sky blue? | 9642 | 9642 | yes | 10/10 | 0.3139 (1.13%) |
| capital of France? | 791 | 791 | yes | 10/10 | 0.5079 (2.04%) |

## Analysis

R111 passes. Int8 activation quantization preserves the model's output not just
at the final token (3/3 prompts) but at the logit distribution: top-1 argmax
matches on all tested prompts, the top-10 token set is identical (10/10), and the
maximum logit deviation is ~1–2% of the peak logit magnitude — consistent with
the phase77 HF top-1/top-10 acceptance criterion used elsewhere in the project.

This validates the int8 direction against the "no quality loss" doctrine for
these prompts. It is not exhaustive (3 prompts, greedy, single first-step
comparison), so it is a green light, not a proof for all inputs.

Important limitation: the gated int8 path is correctness-only and was measured
SLOWER than the tuned f32 path (e.g. prefill ~23s vs ~7s), because it quantizes
per-segment on the fly with no batching or register blocking. The ~18.9x lab win
only transfers if R112 carries the batch4 / int32-accumulator-tile structure into
the runtime, not just the dot.

## Decision

accepted

Reason: int8-activation output matched the f32 path on all tested prompts —
token (3/3), top-1 logit (3/3), top-10 overlap (10/10 each), max logit diff
~1–2% of peak.

Paper value:

- positive evidence that Q8 activation quantization is output-preserving for this
  model and prompt set
- unblocks `REEBORN-Q8-SDOT` runtime promotion against the quality doctrine
- limitation: small prompt set; gated path is correctness-only and currently slow

## Next Experiment

R112 should promote a runtime-gated `REEBORN-Q8-SDOT` kernel into
`accumulate_q8_0_chunk`: batch over prompt-token rows, accumulate int32 in a
register tile, apply weight/activation scales at the end, and `sdot` (then
`smmla`/i8mm) for the dot. Gate it on a same-turn f32 control and the R111 parity
check, and re-measure prefill/decode against Ollama with `rllm bench`.
