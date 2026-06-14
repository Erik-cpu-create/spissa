# R2 Design: RAMA Token-Native Session Benchmark

Date: 2026-06-14
Status: proposed
Owner: RLLM/RAMA

## Objective

Build a small R2 benchmark harness that compares full-replay generation against
persistent `RamaChatSession` generation using the exact same token stream.

R2 is a measurement phase, not a performance optimization phase. Its job is to
produce trustworthy evidence about whether persistent KV-cache helps once text
tokenization ambiguity is removed.

## Problem

The R1 text-transcript SmolLM2 benchmark is inconclusive. The first run showed
an apparent turn 2 speedup, but strict validation proved the decoded text
transcript did not re-encode to the same token history. That means text replay
and session append were not guaranteed to measure the same workload.

R2 removes that ambiguity by using token IDs as the benchmark input and source
of truth.

## Non-Goals

- Do not optimize matmul, projection kernels, KV layout, or chunk recall in R2.
- Do not add a production chat-template system.
- Do not add new model-family adapters.
- Do not claim 30-40 tok/s from this phase.
- Do not compare text prompts unless full transcript token equivalence is
  proven by the harness.

## Design

R2 introduces a token-native benchmark path in `rllm-cli`.

The benchmark accepts a sequence of user turns as comma-separated token IDs.
Each turn is generated twice:

1. **Full-replay baseline** builds the complete visible token history for the
   turn and runs one-shot LLaMA generation from that full history.
2. **Persistent session** appends only the new turn tokens into
   `RamaChatSession`, keeping KV-cache state alive across turns.

The harness compares token histories after every turn. A run is valid only when
baseline and session generated token IDs match for every turn.

## Data Flow

Turn 1:

```text
baseline_input = user_turn_1
session_input = user_turn_1
generate N tokens in both paths
assert baseline_generated == session_generated
```

Turn 2+:

```text
baseline_input = full_visible_history + user_turn_N
session_input = user_turn_N
generate N tokens in both paths
assert baseline_generated == session_generated
assert baseline_visible_history == session.token_history()
```

The baseline intentionally replays the whole visible token history. The session
must report `replayed_tokens=0` after turn 1.

## CLI Scope

Add a narrow scripted benchmark command or mode that supports:

```bash
rllm chat-session-token <model.rllm> \
  --turn-ids 1,2,3 \
  --turn-ids 4,5 \
  --max-new-tokens 64 \
  --ctx 2048 \
  --out docs/benchmarks/trials/active/<trial>.md
```

The output path guard from R1 still applies: the command may write only active
reports. Reviewed folders are updated manually after analysis.

## Metrics

Record per turn:

- baseline input token count
- session input token count
- generated token count
- baseline TTFT / decode tok/s / end-to-end tok/s
- session TTFT / decode tok/s / end-to-end tok/s
- session `replayed_tokens`
- session `flushed_pending_tokens`
- session context memory bytes
- transient peak bytes
- token match status

## Benchmark Classification

R2 reports start in `docs/benchmarks/trials/active/`.

A report can move to:

- `success` when token histories match and session shows useful measured
  improvement.
- `failed` when token histories match but session is slower or too memory-heavy.
- `inconclusive` when token histories do not match, the run aborts, or evidence
  is incomplete.

## Acceptance Criteria

- The harness accepts at least two token turns.
- Empty turns and invalid token ID lists fail before model open.
- Baseline and session generated tokens are compared per turn.
- Mismatch aborts the run and writes/records the reason as inconclusive evidence
  rather than positive evidence.
- A successful run writes a Markdown report in `docs/benchmarks/trials/active/`.
- Focused CLI tests cover token parsing, active-folder guard reuse, and mismatch
  report classification helpers.
- Verification passes:
  - `cargo fmt --check`
  - `cargo check -p rllm-cli`
  - `cargo test -p rllm-cli chat_session -- --nocapture`
  - `cargo clippy -p rllm-cli --all-targets -- -D warnings`

## Risks

- A one-shot full-replay baseline can be slower and may make the command take
  longer on larger models. R2 keeps the first trial small.
- If baseline and session diverge, the result is still valuable, but it is not
  speed evidence.
- If runtime generation has nondeterminism, use argmax only for R2.

## Next Step

Write the implementation plan for the mini R2 scope, then implement it with TDD
in small commits.
