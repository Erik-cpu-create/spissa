# Spec: Gemma interactive chat (persistent-KV REPL)

Date: 2026-06-19
Status: approved (design)

## Problem

`gemma-test` is single-shot: every message reloads the 4.5 GB model, mlocks it,
and runs the ~2.7 s integrity prewarm before prefilling. To chat you re-type the
whole command each time and pay that startup repeatedly. Users want a REPL: load
once, then keep typing, with the model remembering the conversation.

## Performance constraint (why persistent-KV)

Marginal prefill cost is ~26 ms/token (measured: 6 tok 201 ms, 27 tok 747 ms).
Re-prefilling the full conversation every turn compounds (256 tok ≈ 6.7 s, 512 ≈
13 s) and is unusable for multi-turn. So keep the KV cache resident across turns
and prefill only each new user message (~0.8 s regardless of history length).

## Design

### 1. `GemmaChatSession` (runtime: `crates/rllm-runtime/src/models/gemma/`)

Owns the per-conversation state set up ONCE:
- `caches: Vec<KvCache>` (one per layer, persistent), `total_tokens: usize`.
- Embedding context: `embed_id`, `vocab_size`, `embedding_f32: Option<Vec<f32>>`
  (bf16-direct or f32 fallback — same setup the generate fn does today).

Method:
`feed_and_decode(model, new_tokens, max_new, stop_ids, on_token) -> Result<Vec<usize>>`
- Prefill ONLY `new_tokens` in one forward at `position_offset = total_tokens`,
  appending to the existing caches; advance `total_tokens`.
- Decode loop: sample, stream via `on_token`, append each token to caches and
  `total_tokens`, stop on a `stop_ids` token (the stop token is NOT fed to the
  cache — the next turn re-supplies the turn-closing `<end_of_turn>`).
- Returns the generated tokens (excluding the stop token).
- `reset()` recreates the caches and zeroes `total_tokens` (new conversation).

### 2. Forward-step extraction (no behavior change)

Extract the per-step forward currently inlined in `gemma_generate_from_model`
(embed → 34 blocks → final norm → lm_head) into
`gemma_forward_logits(model, prepared, embed_ctx, tokens, position_offset, caches) -> Vec<f32>`.
Both `gemma_generate_from_model` and `GemmaChatSession` call it. The existing
generate path must stay bit-for-bit identical (covered by current tests +
output).

### 3. REPL in `gemma-test` (`--interactive` / `-i`)

Load model + tokenizer + prewarm once (honors `--fast`). Then loop:
- Read a line from stdin with a `you> ` prompt.
- Commands: `/exit` `/quit` (leave), `/reset` (new conversation), `/help`.
- Build the user turn: `<start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n`,
  with BOS only on turn 1; for later turns prepend the turn-closing `<end_of_turn>\n`.
- `session.feed_and_decode(... stop = [eos, <end_of_turn>] ...)`, stream the
  response (reuse the existing token streamer; hide stop tokens).
- Loop.

### 4. Context cap (2048)

Set `max_seq_len = 2048` so each `KvCache` allocates 2048 (~0.6 GB KV total). If
`total_tokens + new_tokens` would exceed it, print "context full — /reset to
start over" and skip the turn (no sliding-window eviction in v1).

### 5. `try-gemma.sh chat`

Add a `chat` subcommand / `-c` flag that runs `gemma-test --interactive --fast`.

## Out of scope (v1, YAGNI)

System prompt, multi-line input, temperature/sampling (stays greedy/argmax),
sliding-window KV eviction.

## Testing

- Session unit test: two turns through `GemmaChatSession` (persistent-KV) produce
  the same generated tokens as two turns via full re-prefill of the running
  sequence — proves KV continuity is correct (within int8 accumulation tolerance,
  argmax preserved).
- Existing gemma/runtime tests stay green (forward-step extraction is behavior-
  preserving).
- Smoke: `gemma-test -i` two-turn conversation stays coherent and on-topic.

## Components / isolation

- `GemmaChatSession`: what — holds conversation KV + drives a turn; how — `new`,
  `feed_and_decode`, `reset`; depends on — `LazyRllmModel`, `PreparedGemmaTransformer`,
  `gemma_forward_logits`.
- `gemma_forward_logits`: one forward → logits; pure given (tokens, position,
  caches); shared by generate + session.
- REPL: terminal I/O + chat-template token bookkeeping; depends on session +
  tokenizer.
