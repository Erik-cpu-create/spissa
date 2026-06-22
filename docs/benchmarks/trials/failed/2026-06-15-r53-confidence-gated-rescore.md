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
| R43-retention100-topk4 | 5 | false | false | 28.41 | 44.91 | 34.85 | 17.60 | 0.18 |
| R53-rescore2-gap250 | 5 | true | false | 42.24 | 69.53 | 52.41 | 8.60 | 0.78 |

## Interpretation

R53 rejects raw confidence-gated exact candidate rescore. The sparse confidence
gate did not protect quality because the exact-rescore token bypassed the
existing repeat-margin and phrase-novelty controllers. Rescore was used on
48-64 of 64 generated tokens depending on prompt, decode speed stayed above
the 30 tok/s floor, but repetition collapsed badly: average repetition rose
from 0.18 to 0.78 and average unique tokens fell from 17.60/64 to 8.60/64.

Decision: failed. The useful signal is architectural: exact candidate rescore
must be controller-preserving before it can be considered as a chat-readiness
path.

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R43-retention100-topk4 | 1 | 1 | 12.20 | 40.38 | 4.65 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R53-rescore2-gap250 | 1 | 1 | 11.82 | 51.21 | 4.90 | 64 | 66 | 1050689536 | 0.97 | 62 | 3/64 |
| R43-retention100-topk4 | 1 | 2 | 12.70 | 44.91 | 4.54 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R53-rescore2-gap250 | 1 | 2 | 12.61 | 69.53 | 4.73 | 64 | 68 | 1050689536 | 0.97 | 62 | 3/64 |
| R43-retention100-topk4 | 1 | 3 | 12.80 | 30.41 | 4.30 | 64 | 70 | 1050689536 | 0.16 | 2 | 22/64 |
| R53-rescore2-gap250 | 1 | 3 | 13.08 | 50.55 | 4.47 | 64 | 70 | 1050689536 | 0.76 | 34 | 8/64 |
| R43-retention100-topk4 | 1 | 4 | 12.52 | 30.15 | 4.38 | 64 | 69 | 1050689536 | 0.16 | 2 | 20/64 |
| R53-rescore2-gap250 | 1 | 4 | 13.16 | 42.24 | 4.37 | 64 | 69 | 1050689536 | 0.35 | 4 | 19/64 |
| R43-retention100-topk4 | 1 | 5 | 12.62 | 28.41 | 4.31 | 64 | 70 | 1050689536 | 0.22 | 2 | 21/64 |
| R53-rescore2-gap250 | 1 | 5 | 12.93 | 48.50 | 4.50 | 64 | 70 | 1050689536 | 0.84 | 37 | 10/64 |

## Raw Output

### R43-retention100-topk4 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 12.20s | Decode: 40.38 tok/s | E2E: 4.65 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R53-rescore2-gap250 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   dis mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir

[TTFT/Prefill: 11.82s | Decode: 51.21 tok/s | E2E: 4.90 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes lm_head_rescore=63/64 gap_skips=1 max_gap_milli=312 input_tile_reads=28480 input_tile_bytes=255590400 phrase_novelty=0/1 max_ngram=0 | Repetition: ratio=0.97 max_run=62 unique=3/64]
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

[TTFT/Prefill: 12.70s | Decode: 44.91 tok/s | E2E: 4.54 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R53-rescore2-gap250 run 1 prompt 2

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> - gl mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir

[TTFT/Prefill: 12.61s | Decode: 69.53 tok/s | E2E: 4.73 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes lm_head_rescore=64/64 gap_skips=0 max_gap_milli=101 input_tile_reads=28480 input_tile_bytes=255590400 | Repetition: ratio=0.97 max_run=62 unique=3/64]
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

[TTFT/Prefill: 12.80s | Decode: 30.41 tok/s | E2E: 4.30 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=9/17 max_gap_milli=338 phrase_novelty=3/64 max_ngram=2 gap_skips=6 max_gap_milli=226 | Repetition: ratio=0.16 max_run=2 unique=22/64]
>
```

### R53-rescore2-gap250 run 1 prompt 3

Prompt: `explain artificial intelligence in one sentence`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  ngxngxngxngx disngx.swingngx.swing.swingngx mir mir mir mir mir mir mir mir mir mir mir mir.swing%d.swingocr vern dis mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir

[TTFT/Prefill: 13.08s | Decode: 50.55 tok/s | E2E: 4.47 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes lm_head_rescore=62/64 gap_skips=2 max_gap_milli=377 input_tile_reads=28480 input_tile_bytes=255590400 phrase_novelty=0/2 max_ngram=0 | Repetition: ratio=0.76 max_run=34 unique=8/64]
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

[TTFT/Prefill: 12.52s | Decode: 30.15 tok/s | E2E: 4.38 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=6/16 max_gap_milli=480 phrase_novelty=1/64 max_ngram=2 gap_skips=2 max_gap_milli=202 retentions=2 | Repetition: ratio=0.16 max_run=2 unique=20/64]
>
```

### R53-rescore2-gap250 run 1 prompt 4

Prompt: `write a short friendly reply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Harmose Mund mir mir mir fÃŃs fÃŃs fÃŃs accidents Mundelage Mund Mundcash mir diff fÃŃs fÃŃs fÃŃs fÃŃs Hubbard.swing.swing fÃŃs fÃŃs Hubbard.swingStrip.swing fÃŃsclub carnrede.swing fÃŃs fÃŃs fÃŃs fÃŃs Mund Mund mÅ©irede.swingStriprede.swing fÃŃs fÃŃsçĸ carn carn carn carnelage.swing.swing.swing fÃŃs mÅ©iametametametÃ¤ÃŁ

[TTFT/Prefill: 13.16s | Decode: 42.24 tok/s | E2E: 4.37 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes lm_head_rescore=48/64 gap_skips=16 max_gap_milli=527 input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=0/2 max_gap_milli=378 phrase_novelty=0/16 max_ngram=0 | Repetition: ratio=0.35 max_run=4 unique=19/64]
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

[TTFT/Prefill: 12.62s | Decode: 28.41 tok/s | E2E: 4.31 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=13/27 max_gap_milli=929 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=1/64 max_ngram=2 gap_skips=12 max_gap_milli=201 retentions=3 | Repetition: ratio=0.22 max_run=2 unique=21/64]
>
```

### R53-rescore2-gap250 run 1 prompt 5

Prompt: `what is two plus two?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Fininnamoninnamoninnamon%d stress stress mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir mir fÃŃsÃ¤ÃŁÃ¤ÃŁÃ¤ÃŁÃ¤ÃŁÃ¤ÃŁÃ¤ÃŁÃ¤ÃŁÃ¤ÃŁogradhaulÃ¤ÃŁÃ¤ÃŁÃ¤ÃŁÃ¤ÃŁÃ¤ÃŁÃ¤ÃŁÃ¤ÃŁÃ¤ÃŁ

[TTFT/Prefill: 12.93s | Decode: 48.50 tok/s | E2E: 4.50 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes lm_head_rescore=61/64 gap_skips=3 max_gap_milli=929 input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=0/1 max_gap_milli=929 phrase_novelty=0/3 max_ngram=0 | Repetition: ratio=0.84 max_run=37 unique=10/64]
>
```
