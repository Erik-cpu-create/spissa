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
| R37-control | 10 | false | false | 24.73 | 37.35 | 31.16 | 16.60 | 0.14 |
| R39-retention-100 | 10 | false | false | 23.07 | 52.40 | 40.19 | 16.00 | 0.17 |

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R37-control | 1 | 1 | 12.55 | 31.66 | 4.40 | 64 | 66 | 1050689536 | 0.11 | 2 | 17/64 |
| R39-retention-100 | 1 | 1 | 13.12 | 41.61 | 4.37 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R37-control | 1 | 2 | 12.89 | 37.35 | 4.39 | 64 | 66 | 1050689536 | 0.14 | 2 | 16/64 |
| R39-retention-100 | 1 | 2 | 13.43 | 28.72 | 4.10 | 64 | 66 | 1050689536 | 0.19 | 2 | 18/64 |
| R37-control | 1 | 3 | 13.72 | 36.72 | 4.15 | 64 | 68 | 1050689536 | 0.19 | 2 | 12/64 |
| R39-retention-100 | 1 | 3 | 13.36 | 41.95 | 4.31 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R37-control | 1 | 4 | 14.33 | 29.80 | 3.89 | 64 | 68 | 1050689536 | 0.14 | 2 | 16/64 |
| R39-retention-100 | 1 | 4 | 13.82 | 23.07 | 3.87 | 64 | 68 | 1050689536 | 0.19 | 2 | 15/64 |
| R37-control | 1 | 5 | 15.78 | 24.73 | 3.49 | 64 | 69 | 1050689536 | 0.10 | 2 | 22/64 |
| R39-retention-100 | 1 | 5 | 14.71 | 50.45 | 4.01 | 64 | 69 | 1050689536 | 0.10 | 2 | 22/64 |
| R37-control | 2 | 1 | 13.77 | 30.01 | 4.03 | 64 | 66 | 1050689536 | 0.11 | 2 | 17/64 |
| R39-retention-100 | 2 | 1 | 13.85 | 52.40 | 4.25 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R37-control | 2 | 2 | 13.53 | 27.45 | 4.04 | 64 | 66 | 1050689536 | 0.14 | 2 | 16/64 |
| R39-retention-100 | 2 | 2 | 13.18 | 39.32 | 4.33 | 64 | 66 | 1050689536 | 0.19 | 2 | 18/64 |
| R37-control | 2 | 3 | 14.05 | 36.63 | 4.06 | 64 | 68 | 1050689536 | 0.19 | 2 | 12/64 |
| R39-retention-100 | 2 | 3 | 13.77 | 42.27 | 4.19 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R37-control | 2 | 4 | 14.10 | 29.64 | 3.94 | 64 | 68 | 1050689536 | 0.14 | 2 | 16/64 |
| R39-retention-100 | 2 | 4 | 13.70 | 36.26 | 4.15 | 64 | 68 | 1050689536 | 0.19 | 2 | 15/64 |
| R37-control | 2 | 5 | 14.29 | 27.59 | 3.86 | 64 | 69 | 1050689536 | 0.10 | 2 | 22/64 |
| R39-retention-100 | 2 | 5 | 14.50 | 45.83 | 4.03 | 64 | 69 | 1050689536 | 0.10 | 2 | 22/64 |

## Interpretation

R41 rejects the R39 retention preset as a multi-prompt speed-floor candidate before
the small top-k selector optimization. The candidate averaged 40.19 tok/s, but
two candidate runs fell below the 30 tok/s floor: prompt 2 run 1 at 28.72 tok/s
and prompt 4 run 1 at 23.07 tok/s. The strict 30-40 tok/s band also failed due
high-side variance.

Decision: failed. Keep this as pre-optimization evidence that a single-prompt
floor pass was not enough.

## Raw Output

### R37-control run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mirlaughter projectile

[TTFT/Prefill: 12.55s | Decode: 31.66 tok/s | E2E: 4.40 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 | Repetition: ratio=0.11 max_run=2 unique=17/64]
>
```

### R39-retention-100 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 13.12s | Decode: 41.61 tok/s | E2E: 4.37 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R37-control run 1 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir.swing mir.swing.swing miryth mirihn mir mir.swing mirlaughter inset mir mirrede mir mir.swing mirlaughter dis carnrede mir.swing mir mir.swing mirlaughter.swingFinder.swing mir fÃŃs fÃŃselage mir mir.swing mirlaughterelage mir mir.swing mirlaughter comlaughteræķ·.swing mir mir.swing mirlaughter com.swing mal

[TTFT/Prefill: 12.89s | Decode: 37.35 tok/s | E2E: 4.39 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=13/26 max_gap_milli=126 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=9/64 max_ngram=2 gap_skips=6 max_gap_milli=242 | Repetition: ratio=0.14 max_run=2 unique=16/64]
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

[TTFT/Prefill: 13.43s | Decode: 28.72 tok/s | E2E: 4.10 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=12/23 max_gap_milli=123 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=10 max_gap_milli=242 retentions=3 | Repetition: ratio=0.19 max_run=2 unique=18/64]
>
```

### R37-control run 1 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir dis/etcinnamon dis-INF mir mir.swing mir.swing.swing mir.swinglaughter mir.swing mir mir.swing.swing mir.swinglaughter mir mir.swing mirlaughter.swing mir.swing mir mir.swing mir.swing mir mir.swing mir.swing mir.swing.swing mir mir.swing mirlaughteræķ·ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ibandÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ibandÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ipers mir.swing

[TTFT/Prefill: 13.72s | Decode: 36.72 tok/s | E2E: 4.15 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/27 max_gap_milli=111 adaptive_throttles=5 min_margin_milli=18 phrase_novelty=8/64 max_ngram=2 gap_skips=22 max_gap_milli=223 | Repetition: ratio=0.19 max_run=2 unique=12/64]
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

[TTFT/Prefill: 13.36s | Decode: 41.95 tok/s | E2E: 4.31 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R37-control run 1 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  osengx mir disngx mir mir.swing mirlaughterNaN mir mir.swinginnamon mir mir.swinginnamon mir dis Mund fÃŃs Mund fÃŃs mir.swing material mir mir.swing mirlaughter mir.swing mir mir.swing mir.swing disrede mir.swing mir mir.swing mirlaughterervoervo financeselage mir.swing mir mir.swing mirlaughterervoervo finances

[TTFT/Prefill: 14.33s | Decode: 29.80 tok/s | E2E: 3.89 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=11/18 max_gap_milli=717 phrase_novelty=9/64 max_ngram=2 gap_skips=9 max_gap_milli=281 | Repetition: ratio=0.14 max_run=2 unique=16/64]
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

[TTFT/Prefill: 13.82s | Decode: 23.07 tok/s | E2E: 3.87 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/26 max_gap_milli=489 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=2/64 max_ngram=2 gap_skips=19 max_gap_milli=210 retentions=6 | Repetition: ratio=0.19 max_run=2 unique=15/64]
>
```

### R37-control run 1 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir disngx Ledger Ledger fÃŃs fÃŃs Mund Mund">%cash rem diff Mundcash fÃŃs Mund Mund Canterrede rem fÃŃs fÃŃs%degend">%rede mir.swing mir mir.swing miregend fÃŃselage mir.swing mir fÃŃselage.swing fÃŃselage mir.swing fÃŃselage.swingStrip.swingstitute carnelage.swing fÃŃsclub carnhaulelage.swing fÃŃs

[TTFT/Prefill: 15.78s | Decode: 24.73 tok/s | E2E: 3.49 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=4/9 max_gap_milli=644 phrase_novelty=2/64 max_ngram=2 gap_skips=3 max_gap_milli=326 | Repetition: ratio=0.10 max_run=2 unique=22/64]
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

[TTFT/Prefill: 14.71s | Decode: 50.45 tok/s | E2E: 4.01 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=4/9 max_gap_milli=644 phrase_novelty=2/64 max_ngram=2 gap_skips=3 max_gap_milli=326 | Repetition: ratio=0.10 max_run=2 unique=22/64]
>
```

### R37-control run 2 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mirlaughter projectile

[TTFT/Prefill: 13.77s | Decode: 30.01 tok/s | E2E: 4.03 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 | Repetition: ratio=0.11 max_run=2 unique=17/64]
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

[TTFT/Prefill: 13.85s | Decode: 52.40 tok/s | E2E: 4.25 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R37-control run 2 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir.swing mir.swing.swing miryth mirihn mir mir.swing mirlaughter inset mir mirrede mir mir.swing mirlaughter dis carnrede mir.swing mir mir.swing mirlaughter.swingFinder.swing mir fÃŃs fÃŃselage mir mir.swing mirlaughterelage mir mir.swing mirlaughter comlaughteræķ·.swing mir mir.swing mirlaughter com.swing mal

[TTFT/Prefill: 13.53s | Decode: 27.45 tok/s | E2E: 4.04 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=13/26 max_gap_milli=126 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=9/64 max_ngram=2 gap_skips=6 max_gap_milli=242 | Repetition: ratio=0.14 max_run=2 unique=16/64]
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

[TTFT/Prefill: 13.18s | Decode: 39.32 tok/s | E2E: 4.33 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=12/23 max_gap_milli=123 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=10 max_gap_milli=242 retentions=3 | Repetition: ratio=0.19 max_run=2 unique=18/64]
>
```

### R37-control run 2 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir dis/etcinnamon dis-INF mir mir.swing mir.swing.swing mir.swinglaughter mir.swing mir mir.swing.swing mir.swinglaughter mir mir.swing mirlaughter.swing mir.swing mir mir.swing mir.swing mir mir.swing mir.swing mir.swing.swing mir mir.swing mirlaughteræķ·ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ibandÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ibandÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ipers mir.swing

[TTFT/Prefill: 14.05s | Decode: 36.63 tok/s | E2E: 4.06 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/27 max_gap_milli=111 adaptive_throttles=5 min_margin_milli=18 phrase_novelty=8/64 max_ngram=2 gap_skips=22 max_gap_milli=223 | Repetition: ratio=0.19 max_run=2 unique=12/64]
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

[TTFT/Prefill: 13.77s | Decode: 42.27 tok/s | E2E: 4.19 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R37-control run 2 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  osengx mir disngx mir mir.swing mirlaughterNaN mir mir.swinginnamon mir mir.swinginnamon mir dis Mund fÃŃs Mund fÃŃs mir.swing material mir mir.swing mirlaughter mir.swing mir mir.swing mir.swing disrede mir.swing mir mir.swing mirlaughterervoervo financeselage mir.swing mir mir.swing mirlaughterervoervo finances

[TTFT/Prefill: 14.10s | Decode: 29.64 tok/s | E2E: 3.94 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=11/18 max_gap_milli=717 phrase_novelty=9/64 max_ngram=2 gap_skips=9 max_gap_milli=281 | Repetition: ratio=0.14 max_run=2 unique=16/64]
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

[TTFT/Prefill: 13.70s | Decode: 36.26 tok/s | E2E: 4.15 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/26 max_gap_milli=489 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=2/64 max_ngram=2 gap_skips=19 max_gap_milli=210 retentions=6 | Repetition: ratio=0.19 max_run=2 unique=15/64]
>
```

### R37-control run 2 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir disngx Ledger Ledger fÃŃs fÃŃs Mund Mund">%cash rem diff Mundcash fÃŃs Mund Mund Canterrede rem fÃŃs fÃŃs%degend">%rede mir.swing mir mir.swing miregend fÃŃselage mir.swing mir fÃŃselage.swing fÃŃselage mir.swing fÃŃselage.swingStrip.swingstitute carnelage.swing fÃŃsclub carnhaulelage.swing fÃŃs

[TTFT/Prefill: 14.29s | Decode: 27.59 tok/s | E2E: 3.86 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=4/9 max_gap_milli=644 phrase_novelty=2/64 max_ngram=2 gap_skips=3 max_gap_milli=326 | Repetition: ratio=0.10 max_run=2 unique=22/64]
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

[TTFT/Prefill: 14.50s | Decode: 45.83 tok/s | E2E: 4.03 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=4/9 max_gap_milli=644 phrase_novelty=2/64 max_ngram=2 gap_skips=3 max_gap_milli=326 | Repetition: ratio=0.10 max_run=2 unique=22/64]
>
```
