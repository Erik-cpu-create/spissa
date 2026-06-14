# Benchmark Trial Index

This index is the paper-oriented map of RLLM/RAMA benchmark evidence. Keep one
row per trial and update the row when a report moves between status folders.

| date | trial | folder | model | mode | bottleneck tag | baseline | result | decision | paper value |
|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | example-trial.md | active | model/artifact | exact-lowram | memory bandwidth | baseline metric | trial metric | planned | not paper-worthy yet |
| 2026-06-14 | 2026-06-14-r1-session-smollm2.md | inconclusive | SmolLM2-135M-raw.rllm | exact-lowram | tokenizer | turn 2 TTFT 1990.00 ms, decode 7.16 tok/s | strict text transcript validation failed before turn 2 | inconclusive | use as limitation |
| 2026-06-14 | 2026-06-14-r2-token-native-smollm2.md | active | SmolLM2-135M-raw.rllm | exact-lowram | cache locality | turn 2 TTFT 871.27 ms, decode 10.22 tok/s | turn 2 TTFT 179.35 ms, decode 10.08 tok/s, token match=true | needs follow-up | use as positive evidence after review |
| 2026-06-14 | 2026-06-14-r5-fused-up-multiply-smollm2.md | failed | SmolLM2-135M-raw.rllm | exact-lowram | transformer MLP projection | R4 turn 1 10.75 tok/s, turn 2 9.14 tok/s | R5 turn 1 9.94 tok/s, turn 2 9.88 tok/s, token match=true | failed | useful negative result |

## Folder Mapping

- `active` means planned, running, or incomplete evidence.
- `success` means accepted by measurement.
- `failed` means rejected, slower, unstable, too memory-heavy, or not worth pursuing.
- `inconclusive` means the benchmark signal is mixed or not trustworthy yet.
