# Alternating Benchmark Harness

## Setup

- Model: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa`
- Runner: `/Users/deansanbhnanwr/Projects/rllm/target/release/llama-test`
- Prompts: 5
  - 1: `good morning`
  - 2: `who are you?`
  - 3: `explain artificial intelligence in one sentence`
  - 4: `write a short friendly reply`
  - 5: `what is two plus two?`
- Runs: 1 alternating control/candidate pairs per prompt
- Target decode band: 30.00-40.00 tok/s
- Profile phases: false

## Summary

| variant | runs | floor accepted | band accepted | min decode tok/s | max decode tok/s | avg decode tok/s | avg unique tokens | avg repetition ratio |
|---|---:|---|---|---:|---:|---:|---:|---:|
| R43-retention100-topk4 | 5 | false | false | 23.71 | 47.98 | 37.54 | 17.60 | 0.18 |
| R56-exact-edge-mlpdown1 | 5 | false | false | 20.24 | 39.71 | 29.97 | 14.80 | 0.20 |

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R43-retention100-topk4 | 1 | 1 | 11.89 | 40.70 | 4.76 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R56-exact-edge-mlpdown1 | 1 | 1 | 11.78 | 39.71 | 4.79 | 64 | 66 | 1050689536 | 0.17 | 2 | 13/64 |
| R43-retention100-topk4 | 1 | 2 | 12.36 | 47.98 | 4.68 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R56-exact-edge-mlpdown1 | 1 | 2 | 11.99 | 36.40 | 4.66 | 64 | 68 | 1050689536 | 0.27 | 2 | 9/64 |
| R43-retention100-topk4 | 1 | 3 | 13.01 | 32.27 | 4.28 | 64 | 70 | 1050689536 | 0.16 | 2 | 22/64 |
| R56-exact-edge-mlpdown1 | 1 | 3 | 13.53 | 29.47 | 4.08 | 64 | 70 | 1050689536 | 0.22 | 2 | 17/64 |
| R43-retention100-topk4 | 1 | 4 | 13.26 | 43.03 | 4.35 | 64 | 69 | 1050689536 | 0.16 | 2 | 20/64 |
| R56-exact-edge-mlpdown1 | 1 | 4 | 19.78 | 24.02 | 2.86 | 64 | 69 | 1050689536 | 0.17 | 2 | 17/64 |
| R43-retention100-topk4 | 1 | 5 | 13.86 | 23.71 | 3.88 | 64 | 70 | 1050689536 | 0.22 | 2 | 21/64 |
| R56-exact-edge-mlpdown1 | 1 | 5 | 13.41 | 20.24 | 3.87 | 64 | 70 | 1050689536 | 0.16 | 2 | 18/64 |

## Raw Output

### R43-retention100-topk4 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 11.89s | Decode: 40.70 tok/s | E2E: 4.76 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R56-exact-edge-mlpdown1 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   dis miryth mirihnervo mercashkeleton mir dis mir.swing mir.swing mir mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing mir mir.swing861 mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swingfinefineipers mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing

[TTFT/Prefill: 11.78s | Decode: 39.71 tok/s | E2E: 4.79 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5986 fallbacks=30 max_topk=4 skipped_madds=75874025472 scratch=32 bytes input_tile_reads=27976 input_tile_bytes=253526016 lm_head_repeat_margin=20/28 max_gap_milli=412 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=4/64 max_ngram=2 gap_skips=29 max_gap_milli=258 retentions=3 | Repetition: ratio=0.17 max_run=2 unique=13/64]
>
```

### R43-retention100-topk4 run 1 prompt 2

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir dis/etcinnamon dis-INF mir mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing mir mir.swing.swing mir.swing mir mir.swing.swing mir.swing mir.swing%dfinefineipers mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing mir

[TTFT/Prefill: 12.36s | Decode: 47.98 tok/s | E2E: 4.68 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R56-exact-edge-mlpdown1 run 1 prompt 2

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir.swing mir mir.swing mir mir.swing.swing mir mir.swing.swing mir.swing mir.swing.swing mir.swing disstitutionsstitutionsragmentstitutionsæķ·æķ·çĵ¶.swing mir.swing.swing mir mir.swing mir.swing mir mir.swing mir.swing mir mir.swing mir.swing mir mir.swing mir.swing mir mir.swing.swing mir mir.swing mir.swing

[TTFT/Prefill: 11.99s | Decode: 36.40 tok/s | E2E: 4.66 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5986 fallbacks=30 max_topk=4 skipped_madds=75874025472 scratch=32 bytes input_tile_reads=27976 input_tile_bytes=253526016 lm_head_repeat_margin=16/24 max_gap_milli=403 phrase_novelty=9/64 max_ngram=2 gap_skips=27 max_gap_milli=245 retentions=2 | Repetition: ratio=0.27 max_run=2 unique=9/64]
>
```

### R43-retention100-topk4 run 1 prompt 3

Prompt: `explain artificial intelligence in one sentence`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  ngxinnamoninnamon quotationsinnamonngx mir mir.swingngx mir mir.swing gl mir mir.swing mir.swing dis Mund fÃŃs fÃŃs Mundelage diff diffcash mir.swing mir.swing Canter fÃŃs fÃŃs Mund fÃŃs fÃŃsegend Hubbard.swing hoop mir.swing fÃŃs carnhaulelage mir.swingStrip mir mir.swingFinderclub.swing mir.swing mir mir.swing mir

[TTFT/Prefill: 13.01s | Decode: 32.27 tok/s | E2E: 4.28 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=9/17 max_gap_milli=338 phrase_novelty=3/64 max_ngram=2 gap_skips=6 max_gap_milli=226 | Repetition: ratio=0.16 max_run=2 unique=22/64]
>
```

### R56-exact-edge-mlpdown1 run 1 prompt 3

Prompt: `explain artificial intelligence in one sentence`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  ngxngxinnamoninnamonngx mir mir.swing disinnamoninnamon Fininnamoninnamon rem rem vern mir mir.swing mir mir.swing dis eu.swing mir.swing fÃŃs key.swing mir mir.swing%d disĻ fÃŃs fÃŃs Mundelage mir.swing mir mir.swing lever mir mir.swing mir.swing mir mir.swing mir.swing mir mir.swing mir.swing mir

[TTFT/Prefill: 13.53s | Decode: 29.47 tok/s | E2E: 4.08 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5986 fallbacks=30 max_topk=4 skipped_madds=75874025472 scratch=32 bytes input_tile_reads=27976 input_tile_bytes=253526016 lm_head_repeat_margin=10/20 max_gap_milli=549 phrase_novelty=4/64 max_ngram=2 gap_skips=10 max_gap_milli=549 retentions=4 | Repetition: ratio=0.22 max_run=2 unique=17/64]
>
```

### R43-retention100-topk4 run 1 prompt 4

Prompt: `write a short friendly reply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Harmose Mund mir.swing mir mir Ledger Ledger fÃŃs Mund fÃŃs fÃŃs diff Mund Mund fÃŃs fÃŃsclub Mund Mundrede mir mir.swing mir.swing fÃŃsclub carnhaulclubhaulboaboa mÅ©iclub carnhaulelage mir diffegendrede mir.swing hoop remegendrede mir.swing mir mir.swing mir.swingFinderboaboaÑĪÑĤ mir.swing fÃŃs

[TTFT/Prefill: 13.26s | Decode: 43.03 tok/s | E2E: 4.35 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=6/16 max_gap_milli=480 phrase_novelty=1/64 max_ngram=2 gap_skips=2 max_gap_milli=202 retentions=2 | Repetition: ratio=0.16 max_run=2 unique=20/64]
>
```

### R56-exact-edge-mlpdown1 run 1 prompt 4

Prompt: `write a short friendly reply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Harmose mir Ledgerinnamoninnamon mir Ledger Ledger fÃŃs Mund Mund mir Ledger Ledger fÃŃscash">% fÃŃs fÃŃs Mundrede mir mir.swing dis mir mir.swing fÃŃs fÃŃs Mund.swing fÃŃs fÃŃs Mundelage.swing mir.swing fÃŃs fÃŃs Mund carnhaulelage.swing hoop mir.swing fÃŃs fÃŃs Mundboa carnboahaulboa carnhaulboahaulboa carn

[TTFT/Prefill: 19.78s | Decode: 24.02 tok/s | E2E: 2.86 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5986 fallbacks=30 max_topk=4 skipped_madds=75874025472 scratch=32 bytes input_tile_reads=27976 input_tile_bytes=253526016 lm_head_repeat_margin=4/14 max_gap_milli=500 phrase_novelty=3/64 max_ngram=2 gap_skips=2 max_gap_milli=238 | Repetition: ratio=0.17 max_run=2 unique=17/64]
>
```

### R43-retention100-topk4 run 1 prompt 5

Prompt: `what is two plus two?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Fininnamoninnamon quotations Sheainnamon remynamodb dis mir mir.swing Fin rem remãĥ©ãĤ¤ãĥ³ diff mir mir.swing Fin FinFin mir mir.swing Neal Nealcash mir mir.swing mir.swing mir mir.swing mir.swing mir mir.swingelage.swingorexacons carnhaul carnØ³ØªÙħelage.swing mir mir.swing mir mir.swing mir mir.swing mir mir

[TTFT/Prefill: 13.86s | Decode: 23.71 tok/s | E2E: 3.88 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=13/27 max_gap_milli=929 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=1/64 max_ngram=2 gap_skips=12 max_gap_milli=201 retentions=3 | Repetition: ratio=0.22 max_run=2 unique=21/64]
>
```

### R56-exact-edge-mlpdown1 run 1 prompt 5

Prompt: `what is two plus two?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  innamoninnamon Enc mir dis mir Cobb%dinnamon mir mir dis vsinnamon remngx disungal.swing dis disose mir mir.swing mir mir.swing fÃŃs fÃŃselage mir.swing mir.swing.swingWindowState.swing fÃŃselage mir.swing fÃŃs carnelage.swing habit mir mir.swing mir.swing mir mir.swing mir.swing mir mir.swing fÃŃselage.swing

[TTFT/Prefill: 13.41s | Decode: 20.24 tok/s | E2E: 3.87 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5986 fallbacks=30 max_topk=4 skipped_madds=75874025472 scratch=32 bytes input_tile_reads=27976 input_tile_bytes=253526016 lm_head_repeat_margin=11/20 max_gap_milli=602 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=1/64 max_ngram=2 gap_skips=9 max_gap_milli=207 retentions=1 | Repetition: ratio=0.16 max_run=2 unique=18/64]
>
```

## Interpretation

R56 rejects exact edge-layer `mlp-down` calibration as a speed or quality
preset. It averaged 29.97 tok/s but failed the 30 tok/s floor with a 20.24 tok/s
minimum. It also reduced average unique tokens from 17.60/64 to 14.80/64 and
raised repetition from 0.18 to 0.20.

Decision: failed. Exacting only edge `mlp-down` spends speed budget without
recovering the R55 diversity signal, so the R55 quality gain is not primarily
coming from this projection.
