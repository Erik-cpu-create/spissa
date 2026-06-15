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
| R43-retention100-topk4 | 10 | false | false | 28.38 | 48.05 | 38.73 | 16.00 | 0.17 |
| R44-retention100-lmhead-topk8 | 10 | false | false | 26.16 | 51.26 | 36.76 | 19.80 | 0.25 |

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R43-retention100-topk4 | 1 | 1 | 12.04 | 40.88 | 4.71 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R44-retention100-lmhead-topk8 | 1 | 1 | 12.30 | 41.72 | 4.63 | 64 | 66 | 1050689536 | 0.27 | 2 | 21/64 |
| R43-retention100-topk4 | 1 | 2 | 12.01 | 38.43 | 4.69 | 64 | 66 | 1050689536 | 0.19 | 2 | 18/64 |
| R44-retention100-lmhead-topk8 | 1 | 2 | 12.35 | 51.00 | 4.71 | 64 | 66 | 1050689536 | 0.24 | 2 | 14/64 |
| R43-retention100-topk4 | 1 | 3 | 12.66 | 47.76 | 4.58 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R44-retention100-lmhead-topk8 | 1 | 3 | 13.15 | 30.19 | 4.20 | 64 | 68 | 1050689536 | 0.24 | 2 | 22/64 |
| R43-retention100-topk4 | 1 | 4 | 12.32 | 38.91 | 4.59 | 64 | 68 | 1050689536 | 0.19 | 2 | 15/64 |
| R44-retention100-lmhead-topk8 | 1 | 4 | 12.44 | 36.11 | 4.51 | 64 | 68 | 1050689536 | 0.27 | 2 | 21/64 |
| R43-retention100-topk4 | 1 | 5 | 12.90 | 34.39 | 4.34 | 64 | 69 | 1050689536 | 0.10 | 2 | 22/64 |
| R44-retention100-lmhead-topk8 | 1 | 5 | 12.88 | 34.54 | 4.35 | 64 | 69 | 1050689536 | 0.24 | 2 | 21/64 |
| R43-retention100-topk4 | 2 | 1 | 12.16 | 40.59 | 4.67 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R44-retention100-lmhead-topk8 | 2 | 1 | 12.19 | 37.08 | 4.61 | 64 | 66 | 1050689536 | 0.27 | 2 | 21/64 |
| R43-retention100-topk4 | 2 | 2 | 13.45 | 36.42 | 4.22 | 64 | 66 | 1050689536 | 0.19 | 2 | 18/64 |
| R44-retention100-lmhead-topk8 | 2 | 2 | 12.27 | 51.26 | 4.74 | 64 | 66 | 1050689536 | 0.24 | 2 | 14/64 |
| R43-retention100-topk4 | 2 | 3 | 12.82 | 48.05 | 4.53 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R44-retention100-lmhead-topk8 | 2 | 3 | 13.12 | 32.60 | 4.25 | 64 | 68 | 1050689536 | 0.24 | 2 | 22/64 |
| R43-retention100-topk4 | 2 | 4 | 12.79 | 33.53 | 4.36 | 64 | 68 | 1050689536 | 0.19 | 2 | 15/64 |
| R44-retention100-lmhead-topk8 | 2 | 4 | 13.88 | 26.93 | 3.95 | 64 | 68 | 1050689536 | 0.27 | 2 | 21/64 |
| R43-retention100-topk4 | 2 | 5 | 13.97 | 28.38 | 3.95 | 64 | 69 | 1050689536 | 0.10 | 2 | 22/64 |
| R44-retention100-lmhead-topk8 | 2 | 5 | 13.79 | 26.16 | 3.95 | 64 | 69 | 1050689536 | 0.24 | 2 | 21/64 |

## Interpretation

R44 rejects LM-head top-k 8 as the quality-preserving band preset. It improved
average unique tokens from 16.00/64 to 19.80/64, but it failed the required
30 tok/s floor on prompt 4 run 2 and prompt 5 run 2. The strict 30-40 tok/s
band also failed on the high side.

Decision: failed. LM-head widening improves diversity signal, but the extra
LM-head range reads are not stable enough for the target speed floor.

## Raw Output

### R43-retention100-topk4 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 12.04s | Decode: 40.88 tok/s | E2E: 4.71 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R44-retention100-lmhead-topk8 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   mal mirolan.swing mir mir com.swing mir.swing rem mir mir.swing.swing mir mir.swing mir mir.swing.swing org×Ļ× mir mir.swing.swing rem rem###.swing mir.swing.swing mir mir.swing AngelesÃ¤ÃŁvestvest lipvestvestelage mir mir.swing.swing infleldonipers mir mirlaughtereldoneldonissor×Ļ×dim×Ļ×vest

[TTFT/Prefill: 12.30s | Decode: 41.72 tok/s | E2E: 4.63 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77954088960 scratch=64 bytes input_tile_reads=28736 input_tile_bytes=321257472 lm_head_repeat_margin=3/19 max_gap_milli=595 phrase_novelty=1/64 max_ngram=2 gap_skips=5 max_gap_milli=343 | Repetition: ratio=0.27 max_run=2 unique=21/64]
>
```

### R43-retention100-topk4 run 1 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir.swing mir.swing.swing miryth mirihn mir mir.swing mir mir.swing.swing mirosteilersstitutions mir mir.swing dis.swing mir mir.swingÃ¤ÃŁ.swing fÃŃselage.swing.swing mir mirrede mir.swing mir mir.swing mirrawn.swingFinder.swing mir mir.swing mir mir.swing mir.swing mir mir.swingThreadPool carnhaulelage.swing

[TTFT/Prefill: 12.01s | Decode: 38.43 tok/s | E2E: 4.69 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=12/23 max_gap_milli=123 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=10 max_gap_milli=242 retentions=3 | Repetition: ratio=0.19 max_run=2 unique=18/64]
>
```

### R44-retention100-lmhead-topk8 run 1 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> +</ mir.swinglaughter.swing miryth mirolanyth.swing mir.swing mirlaughter mir.swing mir carnclubolandolandclubolandolandilers mir mir.swing.swing Angeleselage.swing mir.swing.swinglaughter.swing.swing mir.swing.swing mir mirlaughterilersilersSlashilersilersplanesplanesilersilersSlasholandolandilersilersSlashilersilersSlasholand

[TTFT/Prefill: 12.35s | Decode: 51.00 tok/s | E2E: 4.71 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77954088960 scratch=64 bytes input_tile_reads=28736 input_tile_bytes=321257472 lm_head_repeat_margin=3/18 max_gap_milli=556 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=3/64 max_ngram=2 gap_skips=9 max_gap_milli=548 | Repetition: ratio=0.24 max_run=2 unique=14/64]
>
```

### R43-retention100-topk4 run 1 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir dis/etcinnamon dis-INF mir mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing mir mir.swing.swing mir.swing mir mir.swing.swing mir.swing mir.swing%dfinefineipers mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing mir

[TTFT/Prefill: 12.66s | Decode: 47.76 tok/s | E2E: 4.58 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R44-retention100-lmhead-topk8 run 1 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  inn.swinglaughter mir.swing infl infl supposed supposed mir.swing Angeles stress rad radRad rad rad Rad infl inflflate mir.swing.swing Angeles Angelesgow mir.swing.swing Angeleselage.swing mir.swing.swing Angeleselage mir.swing.swing com Finefinefine FinefinefineFinefinefine Finefinelaughterlaughterensitivity rem mir.swing.swing AngelesComposition

[TTFT/Prefill: 13.15s | Decode: 30.19 tok/s | E2E: 4.20 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77954088960 scratch=64 bytes input_tile_reads=28736 input_tile_bytes=321257472 lm_head_repeat_margin=6/21 max_gap_milli=1580 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=3/64 max_ngram=2 gap_skips=4 max_gap_milli=1046 | Repetition: ratio=0.24 max_run=2 unique=22/64]
>
```

### R43-retention100-topk4 run 1 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  osengx mir disngx mir mir.swing mir mir.swingose mir mir.swing Ledger Ledger%d.io.swing mir.swing mir.swing.CopyTo disstitutions mir.swing mir mir.swing mir.swing disorexissororexorexfineelage mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing

[TTFT/Prefill: 12.32s | Decode: 38.91 tok/s | E2E: 4.59 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/26 max_gap_milli=489 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=2/64 max_ngram=2 gap_skips=19 max_gap_milli=210 retentions=6 | Repetition: ratio=0.19 max_run=2 unique=15/64]
>
```

### R44-retention100-lmhead-topk8 run 1 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem mir.swing mir mir disinnamon mir mir.swing.swinginnamon.swingalus mir mir.swing .swing mir mirlaughter reminessiness lip y lipfinefine FinefinefineFinefinefine Fineolandolandrede mir mir.swing.swing mir Angelesatrixatrixelage.swingFinder mir mir.swing.swing mal mir.swing.swinglaughterolandolandelage

[TTFT/Prefill: 12.44s | Decode: 36.11 tok/s | E2E: 4.51 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77954088960 scratch=64 bytes input_tile_reads=28736 input_tile_bytes=321257472 lm_head_repeat_margin=1/18 max_gap_milli=1227 phrase_novelty=2/64 max_ngram=2 gap_skips=2 max_gap_milli=933 | Repetition: ratio=0.27 max_run=2 unique=21/64]
>
```

### R43-retention100-topk4 run 1 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir disngx Ledger Ledger fÃŃs fÃŃs Mund Mund">%cash rem diff Mundcash fÃŃs Mund Mund Canterrede rem fÃŃs fÃŃs%degend">%rede mir.swing mir mir.swing miregend fÃŃselage mir.swing mir fÃŃselage.swing fÃŃselage mir.swing fÃŃselage.swingStrip.swingstitute carnelage.swing fÃŃsclub carnhaulelage.swing fÃŃs

[TTFT/Prefill: 12.90s | Decode: 34.39 tok/s | E2E: 4.34 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=4/9 max_gap_milli=644 phrase_novelty=2/64 max_ngram=2 gap_skips=3 max_gap_milli=326 | Repetition: ratio=0.10 max_run=2 unique=22/64]
>
```

### R44-retention100-lmhead-topk8 run 1 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  both.swing diff diff fix fix diff diff lip lip Mund Mund diff diffDiff Mund Mundclubrede rem lip Mundrede rem Mundclubrede mir.swing.swing-rede mir.swing.swing mir disrede mirMainWindow mir.swing.swing lipolandoland_patch MundilersilersÃ¤ÃŁolandolandilersilersÃ¤ÃŁolandolandelage.swing marg orgÃ¤ÃŁoland

[TTFT/Prefill: 12.88s | Decode: 34.54 tok/s | E2E: 4.35 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77954088960 scratch=64 bytes input_tile_reads=28736 input_tile_bytes=321257472 lm_head_repeat_margin=2/17 max_gap_milli=1092 phrase_novelty=0/64 max_ngram=0 | Repetition: ratio=0.24 max_run=2 unique=21/64]
>
```

### R43-retention100-topk4 run 2 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 12.16s | Decode: 40.59 tok/s | E2E: 4.67 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R44-retention100-lmhead-topk8 run 2 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   mal mirolan.swing mir mir com.swing mir.swing rem mir mir.swing.swing mir mir.swing mir mir.swing.swing org×Ļ× mir mir.swing.swing rem rem###.swing mir.swing.swing mir mir.swing AngelesÃ¤ÃŁvestvest lipvestvestelage mir mir.swing.swing infleldonipers mir mirlaughtereldoneldonissor×Ļ×dim×Ļ×vest

[TTFT/Prefill: 12.19s | Decode: 37.08 tok/s | E2E: 4.61 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77954088960 scratch=64 bytes input_tile_reads=28736 input_tile_bytes=321257472 lm_head_repeat_margin=3/19 max_gap_milli=595 phrase_novelty=1/64 max_ngram=2 gap_skips=5 max_gap_milli=343 | Repetition: ratio=0.27 max_run=2 unique=21/64]
>
```

### R43-retention100-topk4 run 2 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir.swing mir.swing.swing miryth mirihn mir mir.swing mir mir.swing.swing mirosteilersstitutions mir mir.swing dis.swing mir mir.swingÃ¤ÃŁ.swing fÃŃselage.swing.swing mir mirrede mir.swing mir mir.swing mirrawn.swingFinder.swing mir mir.swing mir mir.swing mir.swing mir mir.swingThreadPool carnhaulelage.swing

[TTFT/Prefill: 13.45s | Decode: 36.42 tok/s | E2E: 4.22 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=12/23 max_gap_milli=123 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=10 max_gap_milli=242 retentions=3 | Repetition: ratio=0.19 max_run=2 unique=18/64]
>
```

### R44-retention100-lmhead-topk8 run 2 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> +</ mir.swinglaughter.swing miryth mirolanyth.swing mir.swing mirlaughter mir.swing mir carnclubolandolandclubolandolandilers mir mir.swing.swing Angeleselage.swing mir.swing.swinglaughter.swing.swing mir.swing.swing mir mirlaughterilersilersSlashilersilersplanesplanesilersilersSlasholandolandilersilersSlashilersilersSlasholand

[TTFT/Prefill: 12.27s | Decode: 51.26 tok/s | E2E: 4.74 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77954088960 scratch=64 bytes input_tile_reads=28736 input_tile_bytes=321257472 lm_head_repeat_margin=3/18 max_gap_milli=556 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=3/64 max_ngram=2 gap_skips=9 max_gap_milli=548 | Repetition: ratio=0.24 max_run=2 unique=14/64]
>
```

### R43-retention100-topk4 run 2 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir dis/etcinnamon dis-INF mir mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing mir mir.swing.swing mir.swing mir mir.swing.swing mir.swing mir.swing%dfinefineipers mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing mir

[TTFT/Prefill: 12.82s | Decode: 48.05 tok/s | E2E: 4.53 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R44-retention100-lmhead-topk8 run 2 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  inn.swinglaughter mir.swing infl infl supposed supposed mir.swing Angeles stress rad radRad rad rad Rad infl inflflate mir.swing.swing Angeles Angelesgow mir.swing.swing Angeleselage.swing mir.swing.swing Angeleselage mir.swing.swing com Finefinefine FinefinefineFinefinefine Finefinelaughterlaughterensitivity rem mir.swing.swing AngelesComposition

[TTFT/Prefill: 13.12s | Decode: 32.60 tok/s | E2E: 4.25 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77954088960 scratch=64 bytes input_tile_reads=28736 input_tile_bytes=321257472 lm_head_repeat_margin=6/21 max_gap_milli=1580 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=3/64 max_ngram=2 gap_skips=4 max_gap_milli=1046 | Repetition: ratio=0.24 max_run=2 unique=22/64]
>
```

### R43-retention100-topk4 run 2 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  osengx mir disngx mir mir.swing mir mir.swingose mir mir.swing Ledger Ledger%d.io.swing mir.swing mir.swing.CopyTo disstitutions mir.swing mir mir.swing mir.swing disorexissororexorexfineelage mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing

[TTFT/Prefill: 12.79s | Decode: 33.53 tok/s | E2E: 4.36 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/26 max_gap_milli=489 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=2/64 max_ngram=2 gap_skips=19 max_gap_milli=210 retentions=6 | Repetition: ratio=0.19 max_run=2 unique=15/64]
>
```

### R44-retention100-lmhead-topk8 run 2 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem mir.swing mir mir disinnamon mir mir.swing.swinginnamon.swingalus mir mir.swing .swing mir mirlaughter reminessiness lip y lipfinefine FinefinefineFinefinefine Fineolandolandrede mir mir.swing.swing mir Angelesatrixatrixelage.swingFinder mir mir.swing.swing mal mir.swing.swinglaughterolandolandelage

[TTFT/Prefill: 13.88s | Decode: 26.93 tok/s | E2E: 3.95 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77954088960 scratch=64 bytes input_tile_reads=28736 input_tile_bytes=321257472 lm_head_repeat_margin=1/18 max_gap_milli=1227 phrase_novelty=2/64 max_ngram=2 gap_skips=2 max_gap_milli=933 | Repetition: ratio=0.27 max_run=2 unique=21/64]
>
```

### R43-retention100-topk4 run 2 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir disngx Ledger Ledger fÃŃs fÃŃs Mund Mund">%cash rem diff Mundcash fÃŃs Mund Mund Canterrede rem fÃŃs fÃŃs%degend">%rede mir.swing mir mir.swing miregend fÃŃselage mir.swing mir fÃŃselage.swing fÃŃselage mir.swing fÃŃselage.swingStrip.swingstitute carnelage.swing fÃŃsclub carnhaulelage.swing fÃŃs

[TTFT/Prefill: 13.97s | Decode: 28.38 tok/s | E2E: 3.95 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=4/9 max_gap_milli=644 phrase_novelty=2/64 max_ngram=2 gap_skips=3 max_gap_milli=326 | Repetition: ratio=0.10 max_run=2 unique=22/64]
>
```

### R44-retention100-lmhead-topk8 run 2 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  both.swing diff diff fix fix diff diff lip lip Mund Mund diff diffDiff Mund Mundclubrede rem lip Mundrede rem Mundclubrede mir.swing.swing-rede mir.swing.swing mir disrede mirMainWindow mir.swing.swing lipolandoland_patch MundilersilersÃ¤ÃŁolandolandilersilersÃ¤ÃŁolandolandelage.swing marg orgÃ¤ÃŁoland

[TTFT/Prefill: 13.79s | Decode: 26.16 tok/s | E2E: 3.95 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77954088960 scratch=64 bytes input_tile_reads=28736 input_tile_bytes=321257472 lm_head_repeat_margin=2/17 max_gap_milli=1092 phrase_novelty=0/64 max_ngram=0 | Repetition: ratio=0.24 max_run=2 unique=21/64]
>
```
