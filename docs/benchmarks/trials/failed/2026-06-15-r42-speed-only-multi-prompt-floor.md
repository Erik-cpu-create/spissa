# Alternating Benchmark Harness

## Setup

- Model: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm`
- Runner: `/Users/deansanbhnanwr/Projects/rllm/target/release/llama-test`
- Prompts: 5
  - 1: `good morning`
  - 2: `halo`
  - 3: `who are you?`
  - 4: `explain artificial intelligence simply`
  - 5: `write a short helpful answer`
- Runs: 2 alternating control/candidate pairs per prompt
- Target decode band: 30.00-40.00 tok/s
- Profile phases: false

## Summary

| variant | runs | floor accepted | band accepted | min decode tok/s | max decode tok/s | avg decode tok/s | avg unique tokens | avg repetition ratio |
|---|---:|---|---|---:|---:|---:|---:|---:|
| R39-retention-100 | 10 | false | false | 24.61 | 40.46 | 32.08 | 16.00 | 0.17 |
| R25-speed-only-topk4 | 10 | false | false | 13.29 | 49.34 | 34.64 | 13.20 | 0.62 |

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R39-retention-100 | 1 | 1 | 12.45 | 39.57 | 4.56 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R25-speed-only-topk4 | 1 | 1 | 13.19 | 45.51 | 4.39 | 64 | 66 | 1050689536 | 0.62 | 18 | 10/64 |
| R39-retention-100 | 1 | 2 | 13.40 | 29.52 | 4.12 | 64 | 66 | 1050689536 | 0.19 | 2 | 18/64 |
| R25-speed-only-topk4 | 1 | 2 | 13.54 | 47.88 | 4.31 | 64 | 66 | 1050689536 | 0.59 | 10 | 11/64 |
| R39-retention-100 | 1 | 3 | 13.84 | 40.46 | 4.16 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R25-speed-only-topk4 | 1 | 3 | 13.51 | 49.34 | 4.33 | 64 | 68 | 1050689536 | 0.97 | 62 | 3/64 |
| R39-retention-100 | 1 | 4 | 14.34 | 30.04 | 3.89 | 64 | 68 | 1050689536 | 0.19 | 2 | 15/64 |
| R25-speed-only-topk4 | 1 | 4 | 13.78 | 27.55 | 3.98 | 64 | 68 | 1050689536 | 0.48 | 10 | 22/64 |
| R39-retention-100 | 1 | 5 | 14.81 | 28.92 | 3.77 | 64 | 69 | 1050689536 | 0.10 | 2 | 22/64 |
| R25-speed-only-topk4 | 1 | 5 | 14.19 | 30.23 | 3.93 | 64 | 69 | 1050689536 | 0.43 | 11 | 20/64 |
| R39-retention-100 | 2 | 1 | 14.03 | 35.34 | 4.05 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R25-speed-only-topk4 | 2 | 1 | 16.75 | 34.41 | 3.44 | 64 | 66 | 1050689536 | 0.62 | 18 | 10/64 |
| R39-retention-100 | 2 | 2 | 13.86 | 31.65 | 4.04 | 64 | 66 | 1050689536 | 0.19 | 2 | 18/64 |
| R25-speed-only-topk4 | 2 | 2 | 12.97 | 13.29 | 3.61 | 64 | 66 | 1050689536 | 0.59 | 10 | 11/64 |
| R39-retention-100 | 2 | 3 | 14.33 | 35.06 | 3.97 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R25-speed-only-topk4 | 2 | 3 | 13.85 | 42.66 | 4.18 | 64 | 68 | 1050689536 | 0.97 | 62 | 3/64 |
| R39-retention-100 | 2 | 4 | 15.18 | 25.64 | 3.63 | 64 | 68 | 1050689536 | 0.19 | 2 | 15/64 |
| R25-speed-only-topk4 | 2 | 4 | 14.06 | 23.32 | 3.82 | 64 | 68 | 1050689536 | 0.48 | 10 | 22/64 |
| R39-retention-100 | 2 | 5 | 16.16 | 24.61 | 3.42 | 64 | 69 | 1050689536 | 0.10 | 2 | 22/64 |
| R25-speed-only-topk4 | 2 | 5 | 13.73 | 32.23 | 4.08 | 64 | 69 | 1050689536 | 0.43 | 11 | 20/64 |

## Interpretation

R42 rejects speed-only top-k 4 as a multi-prompt floor solution. It improved
some high-side decode rows, but did not stabilize the floor: the candidate
dropped to 13.29 tok/s on prompt 2 run 2 and stayed below 30 tok/s on prompt 4
in both runs. Quality also collapsed badly, with average repetition ratio 0.62
and only 13.20 unique tokens per 64-token response.

Decision: failed. The result separates raw sparse speed from usable chat
behavior and shows that simply disabling quality controls is not enough.

## Raw Output

### R39-retention-100 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 12.45s | Decode: 39.57 tok/s | E2E: 4.56 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R25-speed-only-topk4 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth miryth mir mirolanyth mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir.swing disrede disilers.swing.swing.swing.swing.swing.swing.swing disfinerede mir.swing.swing.swing.swingfinefinerede mir.swing.swing.swing.swing.swing.swing.swing.swing.swing.swing.swing.swing

[TTFT/Prefill: 13.19s | Decode: 45.51 tok/s | E2E: 4.39 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 | Repetition: ratio=0.62 max_run=18 unique=10/64]
>
```

### R39-retention-100 run 1 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir.swing mir.swing.swing miryth mirihn mir mir.swing mir mir.swing.swing mirosteilersstitutions mir mir.swing dis.swing mir mir.swingÃ¤ÃŁ.swing fÃŃselage.swing.swing mir mirrede mir.swing mir mir.swing mirrawn.swingFinder.swing mir mir.swing mir mir.swing mir.swing mir mir.swingThreadPool carnhaulelage.swing

[TTFT/Prefill: 13.40s | Decode: 29.52 tok/s | E2E: 4.12 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=12/23 max_gap_milli=123 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=10 max_gap_milli=242 retentions=3 | Repetition: ratio=0.19 max_run=2 unique=18/64]
>
```

### R25-speed-only-topk4 run 1 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir mir mir miryth miryth miryth mir mir mir mir mir mir mir mir '// mir mir mir/host mirelage.swing.swing.swing.swing.swing.swing.swing mir mir mir mir '// mir '// mirilers mir mir mir.swing mal dis carn carn carn carn carn carn.swing carn carn carn carn carn carn carn carn carn carn

[TTFT/Prefill: 13.54s | Decode: 47.88 tok/s | E2E: 4.31 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 | Repetition: ratio=0.59 max_run=10 unique=11/64]
>
```

### R39-retention-100 run 1 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir dis/etcinnamon dis-INF mir mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing mir mir.swing.swing mir.swing mir mir.swing.swing mir.swing mir.swing%dfinefineipers mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing mir

[TTFT/Prefill: 13.84s | Decode: 40.46 tok/s | E2E: 4.16 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R25-speed-only-topk4 run 1 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir

[TTFT/Prefill: 13.51s | Decode: 49.34 tok/s | E2E: 4.33 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 | Repetition: ratio=0.97 max_run=62 unique=3/64]
>
```

### R39-retention-100 run 1 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  osengx mir disngx mir mir.swing mir mir.swingose mir mir.swing Ledger Ledger%d.io.swing mir.swing mir.swing.CopyTo disstitutions mir.swing mir mir.swing mir.swing disorexissororexorexfineelage mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing

[TTFT/Prefill: 14.34s | Decode: 30.04 tok/s | E2E: 3.89 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/26 max_gap_milli=489 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=2/64 max_ngram=2 gap_skips=19 max_gap_milli=210 retentions=6 | Repetition: ratio=0.19 max_run=2 unique=15/64]
>
```

### R25-speed-only-topk4 run 1 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  osengx mir disngx mir mir mir mir mir mir mir mir mir.swing wind Ledgerancell Wright Wrightamet carn carnboa fÃŃs mir mir mir mir mir mir mir mir mir mir reminessrede mir mir.swingfx fÃŃselage mir.swing.swing.swing.swing Mundelage mir mir mir mir.swing.swing.swing.swingFinderclub.swing.swing

[TTFT/Prefill: 13.78s | Decode: 27.55 tok/s | E2E: 3.98 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 | Repetition: ratio=0.48 max_run=10 unique=22/64]
>
```

### R39-retention-100 run 1 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir disngx Ledger Ledger fÃŃs fÃŃs Mund Mund">%cash rem diff Mundcash fÃŃs Mund Mund Canterrede rem fÃŃs fÃŃs%degend">%rede mir.swing mir mir.swing miregend fÃŃselage mir.swing mir fÃŃselage.swing fÃŃselage mir.swing fÃŃselage.swingStrip.swingstitute carnelage.swing fÃŃsclub carnhaulelage.swing fÃŃs

[TTFT/Prefill: 14.81s | Decode: 28.92 tok/s | E2E: 3.77 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=4/9 max_gap_milli=644 phrase_novelty=2/64 max_ngram=2 gap_skips=3 max_gap_milli=326 | Repetition: ratio=0.10 max_run=2 unique=22/64]
>
```

### R25-speed-only-topk4 run 1 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir mir Mund Mund Ledger Ledger Ledger Ledgerngx dis Mund">% Mund fÃŃs fÃŃs fÃŃs fÃŃsclub carnrede mir mir mir mir mir mir mir.swing.swing Mundboaboaboaboaboaboaboaboaboaboaboahaulelage.swingstitute Mundelage.swing Mundegendrede.swing.swing.swingStrip mir.swing fÃŃsstitutions vern mir.swing

[TTFT/Prefill: 14.19s | Decode: 30.23 tok/s | E2E: 3.93 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 | Repetition: ratio=0.43 max_run=11 unique=20/64]
>
```

### R39-retention-100 run 2 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 14.03s | Decode: 35.34 tok/s | E2E: 4.05 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R25-speed-only-topk4 run 2 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth miryth mir mirolanyth mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir.swing disrede disilers.swing.swing.swing.swing.swing.swing.swing disfinerede mir.swing.swing.swing.swingfinefinerede mir.swing.swing.swing.swing.swing.swing.swing.swing.swing.swing.swing.swing

[TTFT/Prefill: 16.75s | Decode: 34.41 tok/s | E2E: 3.44 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 | Repetition: ratio=0.62 max_run=18 unique=10/64]
>
```

### R39-retention-100 run 2 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir.swing mir.swing.swing miryth mirihn mir mir.swing mir mir.swing.swing mirosteilersstitutions mir mir.swing dis.swing mir mir.swingÃ¤ÃŁ.swing fÃŃselage.swing.swing mir mirrede mir.swing mir mir.swing mirrawn.swingFinder.swing mir mir.swing mir mir.swing mir.swing mir mir.swingThreadPool carnhaulelage.swing

[TTFT/Prefill: 13.86s | Decode: 31.65 tok/s | E2E: 4.04 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=12/23 max_gap_milli=123 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=10 max_gap_milli=242 retentions=3 | Repetition: ratio=0.19 max_run=2 unique=18/64]
>
```

### R25-speed-only-topk4 run 2 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir mir mir miryth miryth miryth mir mir mir mir mir mir mir mir '// mir mir mir/host mirelage.swing.swing.swing.swing.swing.swing.swing mir mir mir mir '// mir '// mirilers mir mir mir.swing mal dis carn carn carn carn carn carn.swing carn carn carn carn carn carn carn carn carn carn

[TTFT/Prefill: 12.97s | Decode: 13.29 tok/s | E2E: 3.61 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 | Repetition: ratio=0.59 max_run=10 unique=11/64]
>
```

### R39-retention-100 run 2 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir dis/etcinnamon dis-INF mir mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing mir mir.swing.swing mir.swing mir mir.swing.swing mir.swing mir.swing%dfinefineipers mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing mir

[TTFT/Prefill: 14.33s | Decode: 35.06 tok/s | E2E: 3.97 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R25-speed-only-topk4 run 2 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir

[TTFT/Prefill: 13.85s | Decode: 42.66 tok/s | E2E: 4.18 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 | Repetition: ratio=0.97 max_run=62 unique=3/64]
>
```

### R39-retention-100 run 2 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  osengx mir disngx mir mir.swing mir mir.swingose mir mir.swing Ledger Ledger%d.io.swing mir.swing mir.swing.CopyTo disstitutions mir.swing mir mir.swing mir.swing disorexissororexorexfineelage mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing

[TTFT/Prefill: 15.18s | Decode: 25.64 tok/s | E2E: 3.63 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/26 max_gap_milli=489 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=2/64 max_ngram=2 gap_skips=19 max_gap_milli=210 retentions=6 | Repetition: ratio=0.19 max_run=2 unique=15/64]
>
```

### R25-speed-only-topk4 run 2 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  osengx mir disngx mir mir mir mir mir mir mir mir mir.swing wind Ledgerancell Wright Wrightamet carn carnboa fÃŃs mir mir mir mir mir mir mir mir mir mir reminessrede mir mir.swingfx fÃŃselage mir.swing.swing.swing.swing Mundelage mir mir mir mir.swing.swing.swing.swingFinderclub.swing.swing

[TTFT/Prefill: 14.06s | Decode: 23.32 tok/s | E2E: 3.82 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 | Repetition: ratio=0.48 max_run=10 unique=22/64]
>
```

### R39-retention-100 run 2 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir disngx Ledger Ledger fÃŃs fÃŃs Mund Mund">%cash rem diff Mundcash fÃŃs Mund Mund Canterrede rem fÃŃs fÃŃs%degend">%rede mir.swing mir mir.swing miregend fÃŃselage mir.swing mir fÃŃselage.swing fÃŃselage mir.swing fÃŃselage.swingStrip.swingstitute carnelage.swing fÃŃsclub carnhaulelage.swing fÃŃs

[TTFT/Prefill: 16.16s | Decode: 24.61 tok/s | E2E: 3.42 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=4/9 max_gap_milli=644 phrase_novelty=2/64 max_ngram=2 gap_skips=3 max_gap_milli=326 | Repetition: ratio=0.10 max_run=2 unique=22/64]
>
```

### R25-speed-only-topk4 run 2 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir mir Mund Mund Ledger Ledger Ledger Ledgerngx dis Mund">% Mund fÃŃs fÃŃs fÃŃs fÃŃsclub carnrede mir mir mir mir mir mir mir.swing.swing Mundboaboaboaboaboaboaboaboaboaboaboahaulelage.swingstitute Mundelage.swing Mundegendrede.swing.swing.swingStrip mir.swing fÃŃsstitutions vern mir.swing

[TTFT/Prefill: 13.73s | Decode: 32.23 tok/s | E2E: 4.08 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 | Repetition: ratio=0.43 max_run=11 unique=20/64]
>
```
