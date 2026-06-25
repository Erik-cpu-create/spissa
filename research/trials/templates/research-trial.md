# Research Trial: <short title>

Date: YYYY-MM-DD
Owner: REEBORN
Status: planned | running | accepted | rejected | inconclusive
Folder: active | success | failed | inconclusive

## Hypothesis

The specific idea being tested and the information-theoretic reasoning for why it
should hold. Cite prior art if any.

## Scope

- Experiment type: entropy-floor | quant-rate | scale-stream | outlier | codec-validation
- REE codec: none | REEBORN-* | REEFORM-*
- Lossless reference: vs fp32/bf16 | vs q4/q3 checkpoint | behavioural | n/a
- Model/artifact + weight source:
- Dtype / param count:
- Finding tag: information-theoretic limit | redundancy source | quantization tradeoff | null result | codec validation

## Method

Script:

```bash
/tmp/reeborn-venv/bin/python research/<line>/<script>.py
```

Brief description of the measurement (quant scheme, block size, what is histogrammed).

Runtime context (reproducibility, NOT a perf claim):

- python / numpy:
- host CPU / RAM / OS:
- weight source path:

## Results

| config | bits/weight | entropy (+ components) | SQNR dB | compression ratio | notes |
|---|---:|---|---:|---:|---|
| baseline | | | | | |
| trial | | | | | |

## Analysis

What the numbers mean and where the entropy/redundancy lives. Distinguish the
information-theoretic floor from the achievable rate. State whether the hypothesis holds.

## Decision

accepted | rejected | inconclusive | needs follow-up

Reason:

Paper value: positive evidence | negative/null evidence | limitation | not paper-worthy yet

## Next Experiment

The next concrete experiment, or why this path stops here.
