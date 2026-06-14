# Trial: <short title>

Date: YYYY-MM-DD
Owner: RLLM
Status: planned | running | accepted | rejected | inconclusive
Folder: active | success | failed | inconclusive

## Hypothesis

State the specific idea being tested and why it should improve RLLM.

## Scope

- Mode: exact-lowram | fast-lowram | experimental
- Model/artifact:
- Architecture:
- Target device/profile:
- Expected bottleneck:
- Bottleneck tag: CPU arithmetic | memory bandwidth | cache locality | allocation | IO/decode | tokenizer | scheduler | model architecture | runtime bug

## Setup

Commands:

```bash
# command(s) used for the trial
```

Runtime context:

- build profile:
- CPU:
- RAM:
- OS:
- relevant env/config:

## Results

| run | prompt/input tokens | generated tokens | TTFT/prefill | decode tok/s | end-to-end tok/s | RSS | peak transient | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | | | | | | | | |
| trial | | | | | | | | |

## Analysis

Explain what changed, what dominated runtime, and whether the result supports
the hypothesis.

## Decision

accepted | rejected | inconclusive | needs follow-up

Reason:

Paper value:

- use as positive evidence | use as negative evidence | use as limitation | not paper-worthy yet

## Next Experiment

Name the next concrete trial or state why this path stops here.
