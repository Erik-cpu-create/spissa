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
| 2026-06-14 | 2026-06-14-r9-lm-head-argmax-smollm2.md | success | SmolLM2-135M-raw.rllm | exact-lowram | LM-head argmax memory | R8 17.81/17.64 tok/s, session peak 226492416 bytes | R9 17.15/16.86 tok/s, repeat 16.74/17.07 tok/s, session peak 113246208 bytes, token match=true | success | useful positive memory evidence with speed limitation |
| 2026-06-14 | 2026-06-14-r9-lm-head-argmax-llama32-1b.md | success | Llama-3.2-1B-Instruct-raw.rllm | exact-lowram | LM-head argmax memory | R8 1.68/1.46 tok/s, session peak 2101346304 bytes | R9 1.65/1.71 tok/s, session peak 1050673152 bytes, token match=true | success | useful positive memory evidence with speed limitation |
| 2026-06-14 | 2026-06-14-r10-rowblock-lm-head-argmax-smollm2.md | success | SmolLM2-135M-raw.rllm | exact-lowram | row-blocked LM-head argmax | R9 17.15/16.86 tok/s, repeat 16.74/17.07 tok/s, peak 113246208 bytes | R10 19.01/19.17 tok/s, repeat turn 2 19.03 tok/s, peak 113246208 bytes, token match=true | success | useful positive speed and memory evidence |
| 2026-06-14 | 2026-06-14-r10-rowblock-lm-head-argmax-llama32-1b.md | success | Llama-3.2-1B-Instruct-raw.rllm | exact-lowram | row-blocked LM-head argmax | R9 1.65/1.71 tok/s, peak 1050673152 bytes | R10 1.51/1.93 tok/s, peak 1050673152 bytes, token match=true | success | useful positive speed and memory evidence with limitation |
| 2026-06-14 | 2026-06-14-r11-llama-test-persistent-session-smollm2.md | success | SmolLM2-135M-raw.rllm | exact-lowram | full-history replay | llama-test user run reached turn 10 prefill 13.80s, decode 17.89 tok/s | R11 scripted smoke TTFT 1.47s, 0.17s, 0.20s; decode 18.70, 20.06, 19.58 tok/s | success | useful CLI correctness and bottleneck attribution evidence |
| 2026-06-14 | 2026-06-14-r12-llama-test-context-flag-memory-probe.md | success | SmolLM2-135M-raw.rllm, Llama-3.2-1B-Instruct-raw.rllm | exact-lowram | context capacity | llama-test had fixed ctx 2048 and max_new_tokens 64 | added --ctx and --max-new-tokens; short prompt 2K/4K/8K footprint stayed about 189MB SmolLM2 and 1.62GB Llama 1B | success | useful tooling evidence and context-memory caveat |
| 2026-06-14 | 2026-06-14-r13-cpu-aware-argmax-parallelism.md | success | SmolLM2-135M-raw.rllm, Llama-3.2-1B-Instruct-raw.rllm | exact-lowram | CPU row parallelism | threads=1 SmolLM2 19.25 tok/s, Llama 1B 0.59 tok/s | auto SmolLM2 20.61 tok/s, Llama 1B 0.70 tok/s, RLLM peak unchanged | success with limitation | useful CPU-only evidence and all-core scaling caveat |

## Folder Mapping

- `active` means planned, running, or incomplete evidence.
- `success` means accepted by measurement.
- `failed` means rejected, slower, unstable, too memory-heavy, or not worth pursuing.
- `inconclusive` means the benchmark signal is mixed or not trustworthy yet.
