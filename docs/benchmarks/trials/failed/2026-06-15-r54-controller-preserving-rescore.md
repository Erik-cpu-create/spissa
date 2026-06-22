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
| R43-retention100-topk4 | 5 | true | false | 35.24 | 43.30 | 37.88 | 17.60 | 0.18 |
| R54-rescore2-gap250-controllers | 5 | true | false | 31.37 | 58.59 | 43.82 | 14.60 | 0.21 |

## Interpretation

R54 rejects controller-preserving confidence-gated rescore as the next
chat-ready preset. It fixed the R53 controller bypass: max repeated run stayed
at 2 across all prompts, and the 30 tok/s floor passed. However it still missed
the strict 30-40 tok/s band on the high side and moved quality metrics backward:
average unique tokens fell from 17.60/64 to 14.60/64 and average repetition
rose from 0.18 to 0.21.

Decision: failed as a preset, useful as positive implementation evidence. Exact
candidate row rescoring can be made controller-safe, but it does not improve
semantic output enough to justify promotion over R43.

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R43-retention100-topk4 | 1 | 1 | 12.14 | 38.95 | 4.65 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R54-rescore2-gap250-controllers | 1 | 1 | 12.29 | 58.59 | 4.79 | 64 | 66 | 1050689536 | 0.22 | 2 | 8/64 |
| R43-retention100-topk4 | 1 | 2 | 12.52 | 43.30 | 4.58 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R54-rescore2-gap250-controllers | 1 | 2 | 12.73 | 49.05 | 4.57 | 64 | 68 | 1050689536 | 0.30 | 2 | 9/64 |
| R43-retention100-topk4 | 1 | 3 | 12.90 | 35.24 | 4.36 | 64 | 70 | 1050689536 | 0.16 | 2 | 22/64 |
| R54-rescore2-gap250-controllers | 1 | 3 | 13.04 | 31.37 | 4.25 | 64 | 70 | 1050689536 | 0.11 | 2 | 22/64 |
| R43-retention100-topk4 | 1 | 4 | 12.72 | 35.86 | 4.42 | 64 | 69 | 1050689536 | 0.16 | 2 | 20/64 |
| R54-rescore2-gap250-controllers | 1 | 4 | 12.89 | 46.76 | 4.49 | 64 | 69 | 1050689536 | 0.22 | 2 | 16/64 |
| R43-retention100-topk4 | 1 | 5 | 12.89 | 36.04 | 4.37 | 64 | 70 | 1050689536 | 0.22 | 2 | 21/64 |
| R54-rescore2-gap250-controllers | 1 | 5 | 12.90 | 33.34 | 4.33 | 64 | 70 | 1050689536 | 0.21 | 2 | 18/64 |

## Raw Output

### R43-retention100-topk4 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 12.14s | Decode: 38.95 tok/s | E2E: 4.65 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R54-rescore2-gap250-controllers run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   dis mir mir.swing mir.swing.swing mir dis mir mir.swing mir.swing.swinglaughter.swing ÑģÐ¾ÑģÑĤÐ°Ð².swing mir.swing disheck.swing mir mir.swing.swing mir.swing mir.swing.swingancestor.swing mir mir.swing.swing mir.swing dis mir mir.swing mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir.swing mir mir.swing

[TTFT/Prefill: 12.29s | Decode: 58.59 tok/s | E2E: 4.79 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes lm_head_rescore=63/64 gap_skips=1 max_gap_milli=312 input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=17/23 max_gap_milli=129 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=12/64 max_ngram=2 gap_skips=16 max_gap_milli=262 retentions=5 | Repetition: ratio=0.22 max_run=2 unique=8/64]
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

[TTFT/Prefill: 12.52s | Decode: 43.30 tok/s | E2E: 4.58 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R54-rescore2-gap250-controllers run 1 prompt 2

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> - gl mir.swing mir.swing.swing mir mir.swing.swing mir.swing mir.swing.swing mir mir.swing mir dis disibandÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ascaladj.swing mir.swing.swing mir mir.swing mir.swing mir mir.swing mir.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing mir.swing.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 12.73s | Decode: 49.05 tok/s | E2E: 4.57 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes lm_head_rescore=64/64 gap_skips=0 max_gap_milli=228 input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=15/22 max_gap_milli=228 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=13/64 max_ngram=2 gap_skips=18 max_gap_milli=242 retentions=3 | Repetition: ratio=0.30 max_run=2 unique=9/64]
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

[TTFT/Prefill: 12.90s | Decode: 35.24 tok/s | E2E: 4.36 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=9/17 max_gap_milli=338 phrase_novelty=3/64 max_ngram=2 gap_skips=6 max_gap_milli=226 | Repetition: ratio=0.16 max_run=2 unique=22/64]
>
```

### R54-rescore2-gap250-controllers run 1 prompt 3

Prompt: `explain artificial intelligence in one sentence`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  ngxinnamoninnamon quotationsinnamonngx disngxoseinnamoninnamon dis rem rem Fin rem dep dis conf mir diff diff mir Wed vern mir.swing dis rem mir mir.swing mir.swing mir.swing fÃŃs fÃŃs Mundrede.swing fÃŃs.swing mir.swing fÃŃselage.swing mir.swing fÃŃs carnØ³ØªÙħelage.swing fÃŃselage mir.swingstrip mir mir.swing

[TTFT/Prefill: 13.04s | Decode: 31.37 tok/s | E2E: 4.25 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes lm_head_rescore=52/64 gap_skips=12 max_gap_milli=652 input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=7/13 max_gap_milli=284 phrase_novelty=2/64 max_ngram=2 gap_skips=3 max_gap_milli=207 | Repetition: ratio=0.11 max_run=2 unique=22/64]
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

[TTFT/Prefill: 12.72s | Decode: 35.86 tok/s | E2E: 4.42 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=6/16 max_gap_milli=480 phrase_novelty=1/64 max_ngram=2 gap_skips=2 max_gap_milli=202 retentions=2 | Repetition: ratio=0.16 max_run=2 unique=20/64]
>
```

### R54-rescore2-gap250-controllers run 1 prompt 4

Prompt: `write a short friendly reply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Harmose Mund mir.swing mir mir Ledger Ledger fÃŃs Mund fÃŃs fÃŃs diff Mund Mund fÃŃs fÃŃsclub Mund Mundrede.swing fÃŃs fÃŃs Hubbard.swingStrip fÃŃs fÃŃsclub carnrede.swing fÃŃs fÃŃs Mund mÅ©i Mundboaboa Mund Mundrede.swing fÃŃs fÃŃs Hubbard.swing fÃŃs fÃŃselage.swing mir.swing mir mir.swingStrip.swing mir.swing mir.swing

[TTFT/Prefill: 12.89s | Decode: 46.76 tok/s | E2E: 4.49 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes lm_head_rescore=44/64 gap_skips=20 max_gap_milli=513 input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=7/19 max_gap_milli=480 phrase_novelty=2/64 max_ngram=2 gap_skips=4 max_gap_milli=205 | Repetition: ratio=0.22 max_run=2 unique=16/64]
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

[TTFT/Prefill: 12.89s | Decode: 36.04 tok/s | E2E: 4.37 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=13/27 max_gap_milli=929 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=1/64 max_ngram=2 gap_skips=12 max_gap_milli=201 retentions=3 | Repetition: ratio=0.22 max_run=2 unique=21/64]
>
```

### R54-rescore2-gap250-controllers run 1 prompt 5

Prompt: `what is two plus two?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Fininnamoninnamon quotationsinnamoninnamon lever lever stressinnamon.swing dis delta mir mir.swing mir.swing mir.swinginnamoninnamon.swing mir.swingæī¬egendegend pot mir.swing fÃŃselage.swing mir mir.swing mir.swing mir mir.swing mir mir.swingstrip.swing mir mir.swing pronto mir mir.swing mir mir.swing mir mir.swingFinder mir.swing

[TTFT/Prefill: 12.90s | Decode: 33.34 tok/s | E2E: 4.33 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes lm_head_rescore=57/64 gap_skips=7 max_gap_milli=929 input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=14/27 max_gap_milli=929 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=0/64 max_ngram=2 gap_skips=14 max_gap_milli=253 retentions=3 | Repetition: ratio=0.21 max_run=2 unique=18/64]
>
```
