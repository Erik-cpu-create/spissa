# RLLM Benchmark and Analysis Log

This folder is the canonical home for benchmark evidence, trial-and-error notes,
and performance analysis for RLLM/RAMA.

RLLM research claims must be backed by reproducible measurements. Every serious
runtime experiment should leave a short report under `trials/`, even when the
result is negative or the idea is rejected.

## Structure

- `trials/active/` - planned or currently running trials.
- `trials/success/` - accepted trials with useful measured improvements.
- `trials/failed/` - rejected trials, regressions, and negative evidence.
- `trials/inconclusive/` - mixed results or evidence that is not strong enough yet.
- `trials/index.md` - paper-oriented summary table across all trials.
- `templates/` - reusable report templates.

Existing older benchmark documents may remain in `docs/phase*.md`; new trial
writeups should be created here and can link back to older phase documents when
needed.

## Trial Naming

Use this pattern:

```text
YYYY-MM-DD-short-topic.md
```

Examples:

```text
2026-06-14-smollm2-raw-chat-speed.md
2026-06-14-compressed-vs-raw-llama.md
2026-06-14-kv-cache-chat-session.md
```

## Minimum Evidence

Each report should include:

- hypothesis being tested
- artifact/model path and model shape
- exact command or benchmark harness
- machine/runtime context
- metrics: TTFT/prefill, decode tok/s, end-to-end tok/s, RSS, peak transient memory
- bottleneck tag: CPU arithmetic, memory bandwidth, cache locality, allocation, IO/decode, tokenizer, scheduler, or model architecture
- result table
- analysis of bottleneck
- decision: accept, reject, inconclusive, or needs follow-up
- next experiment

## Status Routing

Use the folder as the current state of the evidence:

- `active/` - use while the trial is planned, running, or still missing numbers.
- `success/` - use when the hypothesis is accepted by measurement.
- `failed/` - use when the hypothesis is rejected, slower, too memory-heavy, unstable, or not worth pursuing.
- `inconclusive/` - use when the signal is mixed or the benchmark setup is not trustworthy yet.

Move a report when the decision changes. Do not delete failed reports; they are
important for bottleneck analysis and paper evidence.

## Rules

- Do not claim speedups without before/after numbers.
- Keep failed trials. Negative evidence prevents repeated dead ends.
- Separate exact/lossless results from fast/lossy or experimental results.
- Cite external papers as prior art when they inform a hypothesis.
- Do not copy source code, formats, or implementation structure from other projects.
