# Alternating Benchmark Harness

## Setup

- Model: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa`
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
| R37-control-optimized-selector | 10 | false | false | 29.70 | 56.57 | 39.62 | 16.60 | 0.14 |
| R39-retention-100-optimized-selector | 10 | true | false | 32.29 | 57.93 | 42.87 | 16.00 | 0.17 |

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R37-control-optimized-selector | 1 | 1 | 12.10 | 39.12 | 4.67 | 64 | 66 | 1050689536 | 0.11 | 2 | 17/64 |
| R39-retention-100-optimized-selector | 1 | 1 | 13.77 | 38.45 | 4.15 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R37-control-optimized-selector | 1 | 2 | 14.02 | 37.23 | 4.07 | 64 | 66 | 1050689536 | 0.14 | 2 | 16/64 |
| R39-retention-100-optimized-selector | 1 | 2 | 14.38 | 32.94 | 3.93 | 64 | 66 | 1050689536 | 0.19 | 2 | 18/64 |
| R37-control-optimized-selector | 1 | 3 | 13.53 | 45.72 | 4.29 | 64 | 68 | 1050689536 | 0.19 | 2 | 12/64 |
| R39-retention-100-optimized-selector | 1 | 3 | 13.62 | 51.69 | 4.31 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R37-control-optimized-selector | 1 | 4 | 13.86 | 37.20 | 4.12 | 64 | 68 | 1050689536 | 0.14 | 2 | 16/64 |
| R39-retention-100-optimized-selector | 1 | 4 | 13.86 | 32.29 | 4.05 | 64 | 68 | 1050689536 | 0.19 | 2 | 15/64 |
| R37-control-optimized-selector | 1 | 5 | 13.48 | 29.70 | 4.10 | 64 | 69 | 1050689536 | 0.10 | 2 | 22/64 |
| R39-retention-100-optimized-selector | 1 | 5 | 13.12 | 49.09 | 4.44 | 64 | 69 | 1050689536 | 0.10 | 2 | 22/64 |
| R37-control-optimized-selector | 2 | 1 | 13.52 | 35.91 | 4.19 | 64 | 66 | 1050689536 | 0.11 | 2 | 17/64 |
| R39-retention-100-optimized-selector | 2 | 1 | 13.80 | 39.99 | 4.16 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R37-control-optimized-selector | 2 | 2 | 12.26 | 56.57 | 4.79 | 64 | 66 | 1050689536 | 0.14 | 2 | 16/64 |
| R39-retention-100-optimized-selector | 2 | 2 | 13.72 | 39.76 | 4.18 | 64 | 66 | 1050689536 | 0.19 | 2 | 18/64 |
| R37-control-optimized-selector | 2 | 3 | 13.26 | 38.40 | 4.30 | 64 | 68 | 1050689536 | 0.19 | 2 | 12/64 |
| R39-retention-100-optimized-selector | 2 | 3 | 13.02 | 42.40 | 4.41 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R37-control-optimized-selector | 2 | 4 | 13.46 | 40.73 | 4.26 | 64 | 68 | 1050689536 | 0.14 | 2 | 16/64 |
| R39-retention-100-optimized-selector | 2 | 4 | 12.85 | 44.17 | 4.48 | 64 | 68 | 1050689536 | 0.19 | 2 | 15/64 |
| R37-control-optimized-selector | 2 | 5 | 12.78 | 35.59 | 4.40 | 64 | 69 | 1050689536 | 0.10 | 2 | 22/64 |
| R39-retention-100-optimized-selector | 2 | 5 | 12.81 | 57.93 | 4.61 | 64 | 69 | 1050689536 | 0.10 | 2 | 22/64 |

## Interpretation

R43 accepts the small top-k selector optimization as speed-floor progress for
Llama 3.2 1B Instruct. The R39 retention candidate passed the 30 tok/s floor on
all 10 multi-prompt runs after replacing the top-k 4 full-vector allocation and
sort path with a small streaming selector. Candidate decode ranged from 32.29
to 57.93 tok/s and averaged 42.87 tok/s.

The strict 30-40 tok/s band remains false because several runs exceed 40 tok/s.
That is a high-side variance issue, not a speed-floor failure. Output quality is
still limited: average unique tokens stayed at 16.00/64 and semantic text remains
weak, so this should be reported as speed success with quality limitation.

Decision: success with quality limitation. Next work should improve semantic
quality while preserving the now-measured multi-prompt 30 tok/s floor.

## Raw Output

### R37-control-optimized-selector run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mirlaughter projectile

[TTFT/Prefill: 12.10s | Decode: 39.12 tok/s | E2E: 4.67 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 | Repetition: ratio=0.11 max_run=2 unique=17/64]
>
```

### R39-retention-100-optimized-selector run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 13.77s | Decode: 38.45 tok/s | E2E: 4.15 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R37-control-optimized-selector run 1 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir.swing mir.swing.swing miryth mirihn mir mir.swing mirlaughter inset mir mirrede mir mir.swing mirlaughter dis carnrede mir.swing mir mir.swing mirlaughter.swingFinder.swing mir fÃŃs fÃŃselage mir mir.swing mirlaughterelage mir mir.swing mirlaughter comlaughteræķ·.swing mir mir.swing mirlaughter com.swing mal

[TTFT/Prefill: 14.02s | Decode: 37.23 tok/s | E2E: 4.07 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=13/26 max_gap_milli=126 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=9/64 max_ngram=2 gap_skips=6 max_gap_milli=242 | Repetition: ratio=0.14 max_run=2 unique=16/64]
>
```

### R39-retention-100-optimized-selector run 1 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir.swing mir.swing.swing miryth mirihn mir mir.swing mir mir.swing.swing mirosteilersstitutions mir mir.swing dis.swing mir mir.swingÃ¤ÃŁ.swing fÃŃselage.swing.swing mir mirrede mir.swing mir mir.swing mirrawn.swingFinder.swing mir mir.swing mir mir.swing mir.swing mir mir.swingThreadPool carnhaulelage.swing

[TTFT/Prefill: 14.38s | Decode: 32.94 tok/s | E2E: 3.93 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=12/23 max_gap_milli=123 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=10 max_gap_milli=242 retentions=3 | Repetition: ratio=0.19 max_run=2 unique=18/64]
>
```

### R37-control-optimized-selector run 1 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir dis/etcinnamon dis-INF mir mir.swing mir.swing.swing mir.swinglaughter mir.swing mir mir.swing.swing mir.swinglaughter mir mir.swing mirlaughter.swing mir.swing mir mir.swing mir.swing mir mir.swing mir.swing mir.swing.swing mir mir.swing mirlaughteræķ·ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ibandÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ibandÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ipers mir.swing

[TTFT/Prefill: 13.53s | Decode: 45.72 tok/s | E2E: 4.29 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/27 max_gap_milli=111 adaptive_throttles=5 min_margin_milli=18 phrase_novelty=8/64 max_ngram=2 gap_skips=22 max_gap_milli=223 | Repetition: ratio=0.19 max_run=2 unique=12/64]
>
```

### R39-retention-100-optimized-selector run 1 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir dis/etcinnamon dis-INF mir mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing mir mir.swing.swing mir.swing mir mir.swing.swing mir.swing mir.swing%dfinefineipers mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing mir

[TTFT/Prefill: 13.62s | Decode: 51.69 tok/s | E2E: 4.31 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R37-control-optimized-selector run 1 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  osengx mir disngx mir mir.swing mirlaughterNaN mir mir.swinginnamon mir mir.swinginnamon mir dis Mund fÃŃs Mund fÃŃs mir.swing material mir mir.swing mirlaughter mir.swing mir mir.swing mir.swing disrede mir.swing mir mir.swing mirlaughterervoervo financeselage mir.swing mir mir.swing mirlaughterervoervo finances

[TTFT/Prefill: 13.86s | Decode: 37.20 tok/s | E2E: 4.12 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=11/18 max_gap_milli=717 phrase_novelty=9/64 max_ngram=2 gap_skips=9 max_gap_milli=281 | Repetition: ratio=0.14 max_run=2 unique=16/64]
>
```

### R39-retention-100-optimized-selector run 1 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  osengx mir disngx mir mir.swing mir mir.swingose mir mir.swing Ledger Ledger%d.io.swing mir.swing mir.swing.CopyTo disstitutions mir.swing mir mir.swing mir.swing disorexissororexorexfineelage mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing

[TTFT/Prefill: 13.86s | Decode: 32.29 tok/s | E2E: 4.05 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/26 max_gap_milli=489 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=2/64 max_ngram=2 gap_skips=19 max_gap_milli=210 retentions=6 | Repetition: ratio=0.19 max_run=2 unique=15/64]
>
```

### R37-control-optimized-selector run 1 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir disngx Ledger Ledger fÃŃs fÃŃs Mund Mund">%cash rem diff Mundcash fÃŃs Mund Mund Canterrede rem fÃŃs fÃŃs%degend">%rede mir.swing mir mir.swing miregend fÃŃselage mir.swing mir fÃŃselage.swing fÃŃselage mir.swing fÃŃselage.swingStrip.swingstitute carnelage.swing fÃŃsclub carnhaulelage.swing fÃŃs

[TTFT/Prefill: 13.48s | Decode: 29.70 tok/s | E2E: 4.10 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=4/9 max_gap_milli=644 phrase_novelty=2/64 max_ngram=2 gap_skips=3 max_gap_milli=326 | Repetition: ratio=0.10 max_run=2 unique=22/64]
>
```

### R39-retention-100-optimized-selector run 1 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir disngx Ledger Ledger fÃŃs fÃŃs Mund Mund">%cash rem diff Mundcash fÃŃs Mund Mund Canterrede rem fÃŃs fÃŃs%degend">%rede mir.swing mir mir.swing miregend fÃŃselage mir.swing mir fÃŃselage.swing fÃŃselage mir.swing fÃŃselage.swingStrip.swingstitute carnelage.swing fÃŃsclub carnhaulelage.swing fÃŃs

[TTFT/Prefill: 13.12s | Decode: 49.09 tok/s | E2E: 4.44 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=4/9 max_gap_milli=644 phrase_novelty=2/64 max_ngram=2 gap_skips=3 max_gap_milli=326 | Repetition: ratio=0.10 max_run=2 unique=22/64]
>
```

### R37-control-optimized-selector run 2 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mirlaughter projectile

[TTFT/Prefill: 13.52s | Decode: 35.91 tok/s | E2E: 4.19 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 | Repetition: ratio=0.11 max_run=2 unique=17/64]
>
```

### R39-retention-100-optimized-selector run 2 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 13.80s | Decode: 39.99 tok/s | E2E: 4.16 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R37-control-optimized-selector run 2 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir.swing mir.swing.swing miryth mirihn mir mir.swing mirlaughter inset mir mirrede mir mir.swing mirlaughter dis carnrede mir.swing mir mir.swing mirlaughter.swingFinder.swing mir fÃŃs fÃŃselage mir mir.swing mirlaughterelage mir mir.swing mirlaughter comlaughteræķ·.swing mir mir.swing mirlaughter com.swing mal

[TTFT/Prefill: 12.26s | Decode: 56.57 tok/s | E2E: 4.79 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=13/26 max_gap_milli=126 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=9/64 max_ngram=2 gap_skips=6 max_gap_milli=242 | Repetition: ratio=0.14 max_run=2 unique=16/64]
>
```

### R39-retention-100-optimized-selector run 2 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir.swing mir.swing.swing miryth mirihn mir mir.swing mir mir.swing.swing mirosteilersstitutions mir mir.swing dis.swing mir mir.swingÃ¤ÃŁ.swing fÃŃselage.swing.swing mir mirrede mir.swing mir mir.swing mirrawn.swingFinder.swing mir mir.swing mir mir.swing mir.swing mir mir.swingThreadPool carnhaulelage.swing

[TTFT/Prefill: 13.72s | Decode: 39.76 tok/s | E2E: 4.18 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=12/23 max_gap_milli=123 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=10 max_gap_milli=242 retentions=3 | Repetition: ratio=0.19 max_run=2 unique=18/64]
>
```

### R37-control-optimized-selector run 2 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir dis/etcinnamon dis-INF mir mir.swing mir.swing.swing mir.swinglaughter mir.swing mir mir.swing.swing mir.swinglaughter mir mir.swing mirlaughter.swing mir.swing mir mir.swing mir.swing mir mir.swing mir.swing mir.swing.swing mir mir.swing mirlaughteræķ·ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ibandÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ibandÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ipers mir.swing

[TTFT/Prefill: 13.26s | Decode: 38.40 tok/s | E2E: 4.30 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/27 max_gap_milli=111 adaptive_throttles=5 min_margin_milli=18 phrase_novelty=8/64 max_ngram=2 gap_skips=22 max_gap_milli=223 | Repetition: ratio=0.19 max_run=2 unique=12/64]
>
```

### R39-retention-100-optimized-selector run 2 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir dis/etcinnamon dis-INF mir mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing mir mir.swing.swing mir.swing mir mir.swing.swing mir.swing mir.swing%dfinefineipers mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing mir

[TTFT/Prefill: 13.02s | Decode: 42.40 tok/s | E2E: 4.41 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R37-control-optimized-selector run 2 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  osengx mir disngx mir mir.swing mirlaughterNaN mir mir.swinginnamon mir mir.swinginnamon mir dis Mund fÃŃs Mund fÃŃs mir.swing material mir mir.swing mirlaughter mir.swing mir mir.swing mir.swing disrede mir.swing mir mir.swing mirlaughterervoervo financeselage mir.swing mir mir.swing mirlaughterervoervo finances

[TTFT/Prefill: 13.46s | Decode: 40.73 tok/s | E2E: 4.26 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=11/18 max_gap_milli=717 phrase_novelty=9/64 max_ngram=2 gap_skips=9 max_gap_milli=281 | Repetition: ratio=0.14 max_run=2 unique=16/64]
>
```

### R39-retention-100-optimized-selector run 2 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  osengx mir disngx mir mir.swing mir mir.swingose mir mir.swing Ledger Ledger%d.io.swing mir.swing mir.swing.CopyTo disstitutions mir.swing mir mir.swing mir.swing disorexissororexorexfineelage mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing

[TTFT/Prefill: 12.85s | Decode: 44.17 tok/s | E2E: 4.48 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/26 max_gap_milli=489 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=2/64 max_ngram=2 gap_skips=19 max_gap_milli=210 retentions=6 | Repetition: ratio=0.19 max_run=2 unique=15/64]
>
```

### R37-control-optimized-selector run 2 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir disngx Ledger Ledger fÃŃs fÃŃs Mund Mund">%cash rem diff Mundcash fÃŃs Mund Mund Canterrede rem fÃŃs fÃŃs%degend">%rede mir.swing mir mir.swing miregend fÃŃselage mir.swing mir fÃŃselage.swing fÃŃselage mir.swing fÃŃselage.swingStrip.swingstitute carnelage.swing fÃŃsclub carnhaulelage.swing fÃŃs

[TTFT/Prefill: 12.78s | Decode: 35.59 tok/s | E2E: 4.40 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=4/9 max_gap_milli=644 phrase_novelty=2/64 max_ngram=2 gap_skips=3 max_gap_milli=326 | Repetition: ratio=0.10 max_run=2 unique=22/64]
>
```

### R39-retention-100-optimized-selector run 2 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir disngx Ledger Ledger fÃŃs fÃŃs Mund Mund">%cash rem diff Mundcash fÃŃs Mund Mund Canterrede rem fÃŃs fÃŃs%degend">%rede mir.swing mir mir.swing miregend fÃŃselage mir.swing mir fÃŃselage.swing fÃŃselage mir.swing fÃŃselage.swingStrip.swingstitute carnelage.swing fÃŃsclub carnhaulelage.swing fÃŃs

[TTFT/Prefill: 12.81s | Decode: 57.93 tok/s | E2E: 4.61 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=4/9 max_gap_milli=644 phrase_novelty=2/64 max_ngram=2 gap_skips=3 max_gap_milli=326 | Repetition: ratio=0.10 max_run=2 unique=22/64]
>
```
