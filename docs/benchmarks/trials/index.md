# Benchmark Trial Index

This index is the paper-oriented map of RLLM/RAMA benchmark evidence. Keep one
row per trial and update the row when a report moves between status folders.

| date | trial | folder | model | mode | bottleneck tag | baseline | result | decision | paper value |
|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | example-trial.md | active | model/artifact | exact-lowram | memory bandwidth | baseline metric | trial metric | planned | not paper-worthy yet |
| 2026-06-14 | 2026-06-14-r1-session-smollm2.md | inconclusive | SmolLM2-135M-raw.rllm | exact-lowram | tokenizer | turn 2 TTFT 1990.00 ms, decode 7.16 tok/s | strict text transcript validation failed before turn 2 | inconclusive | use as limitation |
| 2026-06-14 | 2026-06-14-r2-token-native-smollm2.md | active | SmolLM2-135M-raw.rllm | exact-lowram | cache locality | turn 2 TTFT 871.27 ms, decode 10.22 tok/s | turn 2 TTFT 179.35 ms, decode 10.08 tok/s, token match=true | needs follow-up | use as positive evidence after review |
| 2026-06-14 | 2026-06-14-r5-fused-up-multiply-smollm2.md | failed | SmolLM2-135M-raw.rllm | exact-lowram | transformer MLP projection | R4 turn 1 10.75 tok/s, turn 2 9.14 tok/s | R5 turn 1 9.94 tok/s, turn 2 9.88 tok/s, token match=true | failed | useful negative result |
| 2026-06-14 | 2026-06-14-r6-rowblock-fp16-projection-smollm2.md | failed | SmolLM2-135M-raw.rllm | exact-lowram | projection row blocking | R5 turn 1 9.94 tok/s, turn 2 9.88 tok/s | R6 primary 10.08/9.99 tok/s, repeat 10.02/9.87 tok/s, token match=true | failed | useful negative result |
| 2026-06-14 | 2026-06-14-r7-fused-gate-up-smollm2.md | success | SmolLM2-135M-raw.rllm | exact-lowram | fused gate/up projection | R6 primary 10.08/9.99 tok/s, repeat 10.02/9.87 tok/s | R7 primary 10.81/10.81 tok/s, repeat 10.98/10.85 tok/s, token match=true | success | useful positive evidence |
| 2026-06-14 | 2026-06-14-r7-llama32-1b-model-comparison.md | failed | Llama-3.2-1B-Instruct-raw.rllm | exact-lowram | model shape | R7 SmolLM2 repeat 10.98/10.85 tok/s | Llama 1B turn 1 1.23 tok/s, turn 2 1.21 tok/s, token match=true | failed | useful negative result |
| 2026-06-14 | 2026-06-14-r8-bf16-direct-projection-smollm2.md | success | SmolLM2-135M-raw.rllm | exact-lowram | raw BF16 projection | R7 repeat 10.98/10.85 tok/s | R8 primary 17.81/17.64 tok/s, repeat 17.34/17.76 tok/s, token match=true | success | useful positive evidence |
| 2026-06-14 | 2026-06-14-r8-llama32-1b-bf16-direct-projection.md | success | Llama-3.2-1B-Instruct-raw.rllm | exact-lowram | raw BF16 projection | R7 Llama 1B 1.23/1.21 tok/s | R8 Llama 1B 1.68/1.46 tok/s, token match=true | success | useful positive evidence with limitation |

## Folder Mapping

- `active` means planned, running, or incomplete evidence.
- `success` means accepted by measurement.
- `failed` means rejected, slower, unstable, too memory-heavy, or not worth pursuing.
- `inconclusive` means the benchmark signal is mixed or not trustworthy yet.
