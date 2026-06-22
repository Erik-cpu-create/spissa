# R84 Fair Ollama CPU Benchmark and RLLM Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Produce a fair, reproducible CPU-only benchmark between RLLM R83 and Ollama CPU-only, then re-profile RLLM to choose the next prefill optimization from evidence.

**Architecture:** R84 is a measurement and attribution stage, not a kernel rewrite. It creates a benchmark report, measures Ollama with `num_gpu:0`, measures RLLM with `--rama-integrity unchecked`, records RAM and output sanity, then runs a fresh RLLM trace to identify the post-R83 bottleneck.

**Tech Stack:** Rust `llama-test`, Ollama local API, shell `/usr/bin/time -l`, `jq`, RLLM benchmark docs.

---

## Scope

R84 must answer three questions:

1. What is the current fair gap between RLLM R83 CPU-only and Ollama CPU-only on the same prompt?
2. Where is RLLM's post-R83 prefill time going now?
3. What is the smallest justified R85 optimization target?

R84 must not change runtime kernels. If an obvious code fix appears during R84, write it as the R85 plan instead of patching it inside R84.

## Files

- Create `docs/benchmarks/trials/active/2026-06-16-r84-fair-ollama-cpu-benchmark-and-rllm-trace.md`
  - Holds all benchmark commands, raw metrics, analysis, and final decision.

- Modify `docs/benchmarks/trials/index.md`
  - Add one R84 row after measurement.

- Optional create `target/r84-rllm-trace.json`
  - Generated artifact only. Do not commit it.

No Rust source files should be modified in R84.

## Baselines

Use these already measured anchors:

- R78 RLLM baseline: prefill `26.75s`
- R82 RLLM unchecked best: prefill `16.38s`
- R83 RLLM unchecked best: prefill `11.45s`
- Ollama CPU-only first prompt observed manually: prompt eval `0.23534s / 34 tokens`

R84 must re-measure the relevant current numbers instead of relying on the manual notes.

## Task 1: Create Active Benchmark Report

**Files:**
- Create `docs/benchmarks/trials/active/2026-06-16-r84-fair-ollama-cpu-benchmark-and-rllm-trace.md`

- [x] Add this report skeleton:

```markdown
# R84: Fair Ollama CPU Benchmark and RLLM Trace

## Status

Active.

## Hypothesis

RLLM R83 is still much slower than Ollama CPU-only because its prefill kernels lack llama.cpp-class CPU repack/vectorized execution. R84 will measure the current gap and identify the next RLLM bottleneck before any new kernel work.

## Artifact

- RLLM model: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Ollama model: `llama3.2:1b`
- RLLM mode: exact-lowram Q8, `--rama-integrity unchecked`
- Ollama mode: CPU-only, `num_gpu:0`
- Prompt: `Answer yes or no: is fire cold?`
- Max new tokens: 4

## Commands

Pending execution.

## Results

Pending measurement.

## Trace Attribution

Pending measurement.

## Decision

Pending measurement.
```

- [x] Run:

```sh
test -f docs/benchmarks/trials/active/2026-06-16-r84-fair-ollama-cpu-benchmark-and-rllm-trace.md
```

Expected: exit code `0`.

## Task 2: Verify Ollama Environment

**Files:**
- Modify `docs/benchmarks/trials/active/2026-06-16-r84-fair-ollama-cpu-benchmark-and-rllm-trace.md`

- [x] Start Ollama only if it is not already running:

```sh
ollama serve
```

If this starts a foreground server, keep the session id and stop it before final response.

- [x] List local models:

```sh
ollama list
```

Expected model:

```text
llama3.2:1b
```

- [x] Record model details:

```sh
curl -s http://127.0.0.1:11434/api/tags | jq '.models[] | select(.name=="llama3.2:1b")'
```

Expected details include:

- `family: llama`
- `parameter_size: 1.2B`
- `quantization_level: Q8_0`

## Task 3: Measure Ollama CPU-Only First Prompt

**Files:**
- Modify `docs/benchmarks/trials/active/2026-06-16-r84-fair-ollama-cpu-benchmark-and-rllm-trace.md`

- [x] Ensure no prompt cache contaminates the first prompt by stopping loaded model:

```sh
ollama stop llama3.2:1b || true
```

- [x] Run one CPU-only first-prompt benchmark:

```sh
curl -s http://127.0.0.1:11434/api/chat -d '{"model":"llama3.2:1b","messages":[{"role":"user","content":"Answer yes or no: is fire cold?"}],"stream":false,"options":{"num_predict":4,"temperature":0,"num_ctx":2048,"num_gpu":0}}' | jq '{message:.message.content,total_duration,load_duration,prompt_eval_count,prompt_eval_duration,eval_count,eval_duration,prompt_eval_s:(.prompt_eval_duration/1000000000),eval_tok_s:(.eval_count/(.eval_duration/1000000000))}'
```

Expected:

- `message` contains `No`
- `prompt_eval_count` is near the full prompt length, not `1`
- `prompt_eval_s` is the Ollama prefill number to compare

- [x] Capture proof that Ollama used CPU-only:

Check Ollama server logs for:

```text
-ngl 0
offloaded 0/17 layers to GPU
```

If these lines are not visible, repeat with `num_gpu:0` and do not call the number CPU-only until verified.

## Task 4: Measure RLLM R83 Unchecked

**Files:**
- Modify `docs/benchmarks/trials/active/2026-06-16-r84-fair-ollama-cpu-benchmark-and-rllm-trace.md`

- [x] Build current RLLM binary:

```sh
cargo build --release --bin llama-test
```

Expected:

```text
Finished `release` profile
```

- [x] Run RLLM benchmark:

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked"
```

Expected:

- generated text includes `No`
- report captures `TTFT/Prefill`
- report captures `PrefillProfile`
- report captures `maximum resident set size`
- internal `Peak` remains close to `1,050,673,152 bytes`

## Task 5: Trace RLLM Post-R83 Bottleneck

**Files:**
- Modify `docs/benchmarks/trials/active/2026-06-16-r84-fair-ollama-cpu-benchmark-and-rllm-trace.md`
- Generate `target/r84-rllm-trace.json`

- [x] Run RLLM trace:

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked --rama-trace target/r84-rllm-trace.json"
```

Expected:

- generated text includes `No`
- trace JSON exists at `target/r84-rllm-trace.json`

- [x] Summarize phase totals:

```sh
jq '[.summary.duration_by_phase[] | {phase,event_count,total_ms}]' target/r84-rllm-trace.json
```

- [x] Summarize tensor bucket totals:

```sh
jq '[.summary.duration_by_tensor_bucket[] | {bucket,event_count,total_ms}] | sort_by(.total_ms) | reverse' target/r84-rllm-trace.json
```

Expected analysis buckets:

- `mlp.gate_proj`
- `mlp.up_proj`
- `mlp.down_proj`
- `attention.q_proj`
- `attention.o_proj`
- `attention.k_proj`
- `attention.v_proj`

## Task 6: Decide R85 Target

**Files:**
- Modify `docs/benchmarks/trials/active/2026-06-16-r84-fair-ollama-cpu-benchmark-and-rllm-trace.md`
- Modify `docs/benchmarks/trials/index.md`

- [x] If one MLP bucket still dominates, recommend R85 as a targeted Q8 kernel/layout optimization for that projection.
- [x] If gate/up/down are balanced and MLP still dominates, recommend R85 as shared Q8 dot/repack work.
- [x] If attention dominates, recommend R85 attention Q/O or KV path optimization.
- [x] If lm_head dominates in phase profile, recommend R85 LM-head argmax or output projection optimization.
- [x] Move report to one of:

```text
docs/benchmarks/trials/success/2026-06-16-r84-fair-ollama-cpu-benchmark-and-rllm-trace.md
docs/benchmarks/trials/inconclusive/2026-06-16-r84-fair-ollama-cpu-benchmark-and-rllm-trace.md
```

Use `success` if R84 produces usable benchmark + trace evidence. Use `inconclusive` only if Ollama CPU-only cannot be verified or RLLM trace fails.

- [x] Add index row:

```markdown
| 2026-06-16 | 2026-06-16-r84-fair-ollama-cpu-benchmark-and-rllm-trace.md | success | Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa vs llama3.2:1b | exact-lowram vs Ollama CPU-only | benchmark parity | R83 RLLM unchecked best prefill 11.45s | R84 measured RLLM and Ollama CPU-only plus post-R83 trace attribution | success | next R85 target chosen from trace |
```

Replace the measured values in the row with actual R84 numbers before committing.

## Task 7: Verification and Commit

**Files:**
- Benchmark report
- `docs/benchmarks/trials/index.md`
- This plan

- [x] Run:

```sh
git diff --check
```

Expected: no output.

- [x] Confirm generated trace is not staged:

```sh
git status --short target/r84-rllm-trace.json
```

Expected: no staged file.

- [x] Commit only docs:

```sh
git add docs/superpowers/plans/2026-06-16-r84-fair-ollama-cpu-benchmark-and-rllm-trace.md docs/benchmarks/trials/index.md docs/benchmarks/trials/*/2026-06-16-r84-fair-ollama-cpu-benchmark-and-rllm-trace.md
git commit -m "docs(bench): add r84 ollama cpu comparison plan"
```

## Self-Review

- Spec coverage: The plan measures RLLM vs Ollama CPU-only, verifies no GPU offload for Ollama, captures RLLM trace attribution, and defines how to pick R85.
- Placeholder scan: No `TBD`, `TODO`, or vague implementation-only instructions remain.
- Type consistency: Paths, commands, model names, and report names match the current RLLM workflow.
