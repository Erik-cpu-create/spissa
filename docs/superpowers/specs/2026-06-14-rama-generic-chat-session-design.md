# R1 Design: RAMA Generic Chat Session

Date: 2026-06-14
Status: proposed
Owner: RLLM/RAMA

## Objective

Build a generic RAMA chat-session runtime that keeps per-layer KV-cache state
alive across chat turns. The first target is the local LLaMA-family
`SmolLM2-135M-raw.spsa` artifact, but the session orchestration must be generic
enough to support GPT-NeoX/Pythia first and later Gemma/Qwen without rewriting
the chat loop.

R1 is a latency architecture phase. It does not claim to solve 7B 30-40 tok/s
yet. It creates the correct long-lived session boundary so later kernel,
layout, and fast-lowram experiments measure the right workload.

## Problem

The current experimental `llama-test` binary stores conversation text and
re-encodes/re-processes the whole conversation on every user turn. This makes
turn 2 and later pay full-history prefill cost again, even though the KV-cache
for previous tokens could be retained.

The current GPT-NeoX/Pythia generation path already has internal prefill/decode
state during one generation call, but the state is returned as a completed
generation result rather than used as a reusable chat-session primitive.

## Non-Goals

- Do not copy source code, file formats, or implementation structure from other
  runtimes.
- Do not implement quantization, sparsity, speculative decoding, or lossy
  quality tradeoffs in R1.
- Do not optimize SIMD/GEMM kernels in R1.
- Do not build a production chat-template system in R1.
- Do not promise 7B 30-40 tok/s from this phase.

## Design

R1 introduces a generic session orchestration layer with model-family adapters.

```text
RamaChatSession
├── tokenizer
├── model adapter
├── persistent adapter state
├── token history
├── append_user_tokens(...)
├── generate_assistant_tokens(...)
└── per-turn metrics
```

The session loop owns conversation state and metrics. Model-family adapters own
architecture-specific tensor names, layer parameters, rotary behavior, GQA/MQA
shape rules, and KV-cache layout.

## Core Types

### `RamaChatSession`

Long-lived runtime object for one model and one conversation.

Responsibilities:

- encode user text or accept fixed token IDs
- append only new user tokens to existing context
- generate assistant tokens one at a time
- keep the adapter KV-cache state alive across turns
- collect per-turn timing and memory metrics
- expose token history for verification and benchmark reports

### `RamaSessionAdapter`

Trait-like boundary for model-family-specific behavior.

Required operations:

```text
context_len() -> usize
max_seq_len() -> usize
context_memory_bytes() -> usize
append_prefill(tokens, position_offset, emit_logits, budget) -> Option<Step>
decode_one(token, position_offset, budget) -> Step
```

`append_prefill` updates KV-cache with new user tokens. It may emit logits only
for the final appended token when a next-token decision is needed. Most chat
turns should append user tokens and then start decode from the last user token
without replaying older conversation tokens.

### `RamaSessionStep`

Common result for a model step:

```text
token_id
logits optional
phase timings
context_len_after
```

### `RamaSessionTurnMetrics`

Per-turn evidence collected for benchmark reports:

```text
input_tokens
generated_tokens
new_prefill_tokens
replayed_tokens
ttft_ms
prefill_ms
decode_ms
decode_tok_s
end_to_end_tok_s
context_memory_bytes
peak_transient_bytes
rss_bytes optional
```

`replayed_tokens` should be zero after the first turn in the intended R1 path.

## Adapter Plan

### First Adapter: LLaMA/SmolLM2

The first implementation target is `LayerDecodedLlamaRamaTransformer`.

Required changes:

- create a persistent LLaMA session state containing one `KvCache` per layer
- split the current one-shot generation function into reusable operations:
  - prepare pinned embedding/layernorm/lm-head resources once
  - append new token spans into existing caches
  - decode one token against existing caches
- keep output deterministic with argmax sampling

### Second Adapter: GPT-NeoX/Pythia

After LLaMA session evidence is collected, map `LayerDecodedGptNeoxRamaTransformer`
onto the same session interface. Existing `RamaContextState` and
`step_from_model_inner` already point in this direction, but the public surface
needs a reusable session wrapper instead of one-shot generation ownership.

### Future Adapters: Gemma/Qwen

Gemma and Qwen should plug into the same session interface. Their adapters will
be allowed to differ internally for tensor names, RMSNorm, GQA/MQA, rotary
variants, tied embeddings, and tokenizer/chat-template handling.

## Data Flow

Turn 1:

```text
encode user tokens
append_prefill(all user tokens, position 0)
sample/decode assistant token 1
decode assistant tokens 2..N
persist KV-cache and token history
record metrics
```

Turn 2+:

```text
encode only new user tokens
append_prefill(new user tokens, current context len)
sample/decode assistant token 1
decode assistant tokens 2..N
persist KV-cache and token history
record metrics
```

No previous token span should be replayed unless a future cache eviction policy
requires deterministic replay from a checkpoint. R1 does not implement eviction.

## CLI / Harness

Add an experimental benchmark binary or command that can run scripted two-turn
sessions:

```text
rllm-chat-session --model models/SmolLM2-135M-raw.spsa \
  --turn "Hello" \
  --turn "Continue" \
  --max-new-tokens 64 \
  --out docs/benchmarks/trials/<date>-r1-session-smollm2.md
```

The exact command name can change during implementation, but R1 must provide a
repeatable non-interactive harness. Interactive chat is useful manually, but the
benchmark must not depend on terminal typing.

## Benchmark Plan

Baseline:

- current `llama-test` behavior
- two turns with full conversation replay
- record TTFT/prefill, decode tok/s, wall time, RSS

R1 trial:

- same model, prompts, max token count, sampling mode
- persistent session with no replay after turn 1
- record identical metrics

Success criteria:

- turn 2 `replayed_tokens == 0`
- turn 2 TTFT is materially lower than baseline full replay
- decode tok/s does not regress materially
- context memory bytes increase predictably with token count
- all results documented in `docs/benchmarks/trials/`

## Error Handling

- reject empty session prompts unless fixed token IDs are provided
- reject context overflow before mutating cache state
- reject adapters whose metadata lacks required model fields
- preserve cache state if a decode step fails before append
- report unsupported architecture explicitly

## Testing

Unit tests:

- session appends only new tokens on turn 2
- session rejects context overflow
- deterministic argmax sequence is stable for a tiny test adapter
- LLaMA adapter returns errors instead of panicking on missing metadata

Integration/benchmark tests:

- scripted two-turn SmolLM2 raw run
- benchmark report is written with baseline and R1 rows
- metrics include TTFT, decode tok/s, context bytes, and replayed token count

## Documentation

Every R1 trial must create a report in `docs/benchmarks/trials/`. Failed results
remain in the folder because they become paper evidence and prevent repeated
dead ends.

## Risks

- Persistent KV-cache improves turn latency but increases long-session memory.
- Current LLaMA path pins large weights in memory; R1 must measure RSS honestly.
- Tokenizer/chat-template behavior is still primitive, so early tests should use
  fixed simple prompts and argmax.
- Generic session interfaces can become too abstract. R1 should keep the trait
  narrow and prove it with only LLaMA first before adding more adapters.

## Decision Gate

Proceed to implementation only after this spec is reviewed and accepted.
