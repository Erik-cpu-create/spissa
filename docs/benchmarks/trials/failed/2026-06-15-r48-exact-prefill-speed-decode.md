# Alternating Benchmark Harness

## Setup

- Model: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm`
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
| R43-retention100-topk4 | 5 | false | false | 27.85 | 62.02 | 38.15 | 17.60 | 0.18 |
| R48-exact-prefill | 5 | true | false | 41.52 | 62.54 | 52.41 | 14.40 | 0.21 |

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R43-retention100-topk4 | 1 | 1 | 12.72 | 37.76 | 4.45 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R48-exact-prefill | 1 | 1 | 15.30 | 62.54 | 3.92 | 64 | 66 | 1050689536 | 0.24 | 2 | 15/64 |
| R43-retention100-topk4 | 1 | 2 | 13.40 | 62.02 | 4.44 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R48-exact-prefill | 1 | 2 | 15.76 | 56.77 | 3.79 | 64 | 68 | 1050689536 | 0.27 | 2 | 4/64 |
| R43-retention100-topk4 | 1 | 3 | 13.86 | 30.39 | 4.02 | 64 | 70 | 1050689536 | 0.16 | 2 | 22/64 |
| R48-exact-prefill | 1 | 3 | 15.78 | 49.98 | 3.76 | 64 | 70 | 1050689536 | 0.21 | 2 | 13/64 |
| R43-retention100-topk4 | 1 | 4 | 14.01 | 32.73 | 4.02 | 64 | 69 | 1050689536 | 0.16 | 2 | 20/64 |
| R48-exact-prefill | 1 | 4 | 15.53 | 51.22 | 3.82 | 64 | 69 | 1050689536 | 0.19 | 2 | 20/64 |
| R43-retention100-topk4 | 1 | 5 | 14.00 | 27.85 | 3.93 | 64 | 70 | 1050689536 | 0.22 | 2 | 21/64 |
| R48-exact-prefill | 1 | 5 | 15.86 | 41.52 | 3.68 | 64 | 70 | 1050689536 | 0.16 | 2 | 20/64 |

## Interpretation

R48 rejects exact prompt prefill as the next accepted speed/quality preset. It
proved the flag works and kept decode AIP active, but it did not fix semantic
collapse: output remained fragmentary and the average unique-token count fell
from 17.60 to 14.40. Decode throughput cleared the 30 tok/s floor, but every
candidate run exceeded the strict 40 tok/s upper band. Prefill also rose from
roughly 12.72-14.00s to 15.30-15.86s.

This is useful negative evidence: corrupt prompt prefill is not the main
quality bottleneck. The remaining quality issue is dominated by approximate
decode/LM-head selection after the first token.

## Raw Output

### R43-retention100-topk4 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 12.72s | Decode: 37.76 tok/s | E2E: 4.45 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R48-exact-prefill run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> . miryth mirolan mir mir.swing mir Doug mir miryth mir.swing mir mir.swing.swing mir mir.swing mir.swing.swing mir mir.swing mir.swing mir mir.swing mir.swing disrede disilersSlash carnelage mir mir.swing mir.swing.swingfinefineipers mir.swing mir.swing.swing mir mir.swing.swing mir mir.swingÃ¤ÃŁ

[TTFT/Prefill: 15.30s | Decode: 62.54 tok/s | E2E: 3.92 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6111 fallbacks=0 max_topk=4 skipped_madds=77724767232 scratch=32 bytes input_tile_reads=28476 input_tile_bytes=254564352 lm_head_repeat_margin=12/25 max_gap_milli=260 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=5/63 max_ngram=2 gap_skips=12 max_gap_milli=246 retentions=1 | Repetition: ratio=0.24 max_run=2 unique=15/64]
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

[TTFT/Prefill: 13.40s | Decode: 62.02 tok/s | E2E: 4.44 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R48-exact-prefill run 1 prompt 2

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  I rem mir mir.swing mir.swing mir mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir mir.swing.swing mir.swing mir mir.swing.swing mir.swing mir.swing.swing mir mir.swing mir.swing.swing mir.swing.swing mir mir.swing mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir

[TTFT/Prefill: 15.76s | Decode: 56.77 tok/s | E2E: 3.79 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6111 fallbacks=0 max_topk=4 skipped_madds=77724767232 scratch=32 bytes input_tile_reads=28476 input_tile_bytes=254564352 lm_head_repeat_margin=22/29 max_gap_milli=195 phrase_novelty=10/63 max_ngram=2 gap_skips=34 max_gap_milli=233 retentions=5 | Repetition: ratio=0.27 max_run=2 unique=4/64]
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

[TTFT/Prefill: 13.86s | Decode: 30.39 tok/s | E2E: 4.02 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=9/17 max_gap_milli=338 phrase_novelty=3/64 max_ngram=2 gap_skips=6 max_gap_milli=226 | Repetition: ratio=0.16 max_run=2 unique=22/64]
>
```

### R48-exact-prefill run 1 prompt 3

Prompt: `explain artificial intelligence in one sentence`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> :

 mir disngx mir mir.swing mir disinnamon mir mir dis mir.swing Mundelage mir mir.swing mir mir.swing mir mir fÃŃs fÃŃs Mundrede mir.swing mir mir.swing dis fÃŃs disstitution Mund fÃŃs fÃŃs '// mir.swing mir mir.swing mir.swing mir mir.swingFinderFinder fÃŃs fÃŃs '//.swing mir.swing mir mir.swing mir

[TTFT/Prefill: 15.78s | Decode: 49.98 tok/s | E2E: 3.76 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6111 fallbacks=0 max_topk=4 skipped_madds=77724767232 scratch=32 bytes input_tile_reads=28476 input_tile_bytes=254564352 lm_head_repeat_margin=11/23 max_gap_milli=328 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=3/63 max_ngram=2 gap_skips=10 max_gap_milli=217 retentions=3 | Repetition: ratio=0.21 max_run=2 unique=13/64]
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

[TTFT/Prefill: 14.01s | Decode: 32.73 tok/s | E2E: 4.02 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=6/16 max_gap_milli=480 phrase_novelty=1/64 max_ngram=2 gap_skips=2 max_gap_milli=202 retentions=2 | Repetition: ratio=0.16 max_run=2 unique=20/64]
>
```

### R48-exact-prefill run 1 prompt 4

Prompt: `write a short friendly reply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  toose diff fÃŃs fÃŃs Ledger Ledger fÃŃs Ledger Ledger fÃŃs fÃŃs Mund fÃŃs Mund Mund fÃŃs">%.swing mir mir.swing mir mir.swing disegendrede mir.swing mir mir.swing fÃŃs fÃŃs Mund carnhaul fÃŃs fÃŃs Hubbard.swingFinder fÃŃs fÃŃs Mundrede mir.swingStrip mir Mundclub carnhaulelage.swing fÃŃs fÃŃs '//.swingFinder fÃŃselage

[TTFT/Prefill: 15.53s | Decode: 51.22 tok/s | E2E: 3.82 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6111 fallbacks=0 max_topk=4 skipped_madds=77724767232 scratch=32 bytes input_tile_reads=28476 input_tile_bytes=254564352 lm_head_repeat_margin=6/17 max_gap_milli=517 phrase_novelty=1/63 max_ngram=2 gap_skips=8 max_gap_milli=275 retentions=1 | Repetition: ratio=0.19 max_run=2 unique=20/64]
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

[TTFT/Prefill: 14.00s | Decode: 27.85 tok/s | E2E: 3.93 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=13/27 max_gap_milli=929 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=1/64 max_ngram=2 gap_skips=12 max_gap_milli=201 retentions=3 | Repetition: ratio=0.22 max_run=2 unique=21/64]
>
```

### R48-exact-prefill run 1 prompt 5

Prompt: `what is two plus two?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  The disagli.swing táº¥n mir.swing glirsch mir.swing mir mir.swing mir.swing rem remãĥ©ãĤ¤ãĥ³ mir.swing resp mir mir.swing diff mir mir.swing Mund Mund.swingStrip mir.swing hoop mir.swing pronto mir.swing.swing mir mir.swing mir mir.swing mir mir.swing Mund carnhaul carnØ³ØªÙħelage.swing Mund carnhaulhaul carnØ³ØªÙħ

[TTFT/Prefill: 15.86s | Decode: 41.52 tok/s | E2E: 3.68 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6111 fallbacks=0 max_topk=4 skipped_madds=77724767232 scratch=32 bytes input_tile_reads=28476 input_tile_bytes=254564352 lm_head_repeat_margin=14/24 max_gap_milli=394 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=3/63 max_ngram=2 gap_skips=9 max_gap_milli=199 retentions=2 | Repetition: ratio=0.16 max_run=2 unique=20/64]
>
```
