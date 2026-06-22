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
| R48-exact-prefill-topk4 | 5 | true | false | 32.53 | 54.67 | 40.58 | 14.40 | 0.21 |
| R49-exact-prefill-lmhead-topk5 | 5 | false | false | 22.11 | 76.91 | 44.29 | 16.60 | 0.21 |

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R48-exact-prefill-topk4 | 1 | 1 | 14.67 | 39.28 | 3.93 | 64 | 66 | 1050689536 | 0.24 | 2 | 15/64 |
| R49-exact-prefill-lmhead-topk5 | 1 | 1 | 15.40 | 54.46 | 3.86 | 64 | 66 | 1050689536 | 0.25 | 2 | 12/64 |
| R48-exact-prefill-topk4 | 1 | 2 | 16.08 | 54.67 | 3.71 | 64 | 68 | 1050689536 | 0.27 | 2 | 4/64 |
| R49-exact-prefill-lmhead-topk5 | 1 | 2 | 16.21 | 76.91 | 3.76 | 64 | 68 | 1050689536 | 0.29 | 2 | 4/64 |
| R48-exact-prefill-topk4 | 1 | 3 | 18.58 | 32.53 | 3.12 | 64 | 70 | 1050689536 | 0.21 | 2 | 13/64 |
| R49-exact-prefill-lmhead-topk5 | 1 | 3 | 16.35 | 22.11 | 3.33 | 64 | 70 | 1050689536 | 0.19 | 2 | 20/64 |
| R48-exact-prefill-topk4 | 1 | 4 | 17.62 | 41.32 | 3.34 | 64 | 69 | 1050689536 | 0.19 | 2 | 20/64 |
| R49-exact-prefill-lmhead-topk5 | 1 | 4 | 15.90 | 35.24 | 3.62 | 64 | 69 | 1050689536 | 0.14 | 2 | 24/64 |
| R48-exact-prefill-topk4 | 1 | 5 | 16.18 | 35.10 | 3.56 | 64 | 70 | 1050689536 | 0.16 | 2 | 20/64 |
| R49-exact-prefill-lmhead-topk5 | 1 | 5 | 16.16 | 32.71 | 3.54 | 64 | 70 | 1050689536 | 0.17 | 2 | 23/64 |

## Interpretation

R49 rejects exact prefill plus LM-head top-k 5 as a strict-band or quality fix.
The candidate improved unique-token count on prompts 3-5, but the sweep was
unstable: decode ranged from 22.11 to 76.91 tok/s and failed both the 30 tok/s
floor and the 30-40 tok/s band. Prompt 2 still collapsed to 4 unique tokens.

This confirms that simple LM-head widening on top of exact prefill is not a
stable path. The next useful stage should target decode correction directly,
for example a periodic exact/near-exact decode checkpoint or a confidence-gated
fallback, rather than only changing static top-k width.

## Raw Output

### R48-exact-prefill-topk4 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> . miryth mirolan mir mir.swing mir Doug mir miryth mir.swing mir mir.swing.swing mir mir.swing mir.swing.swing mir mir.swing mir.swing mir mir.swing mir.swing disrede disilersSlash carnelage mir mir.swing mir.swing.swingfinefineipers mir.swing mir.swing.swing mir mir.swing.swing mir mir.swingÃ¤ÃŁ

[TTFT/Prefill: 14.67s | Decode: 39.28 tok/s | E2E: 3.93 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6111 fallbacks=0 max_topk=4 skipped_madds=77724767232 scratch=32 bytes input_tile_reads=28476 input_tile_bytes=254564352 lm_head_repeat_margin=12/25 max_gap_milli=260 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=5/63 max_ngram=2 gap_skips=12 max_gap_milli=246 retentions=1 | Repetition: ratio=0.24 max_run=2 unique=15/64]
>
```

### R49-exact-prefill-lmhead-topk5 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> . miryth miryth.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir mir.swing mir mir.swing.swing inflendl mÅ©ielage mir mir.swing.swing mir.swing mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing.swingfinerede mir.swing.swing mir mir.swing mir.swing.swinglaughterclubrede mir mir

[TTFT/Prefill: 15.40s | Decode: 54.46 tok/s | E2E: 3.86 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6111 fallbacks=0 max_topk=5 skipped_madds=77716687104 scratch=40 bytes input_tile_reads=28539 input_tile_bytes=270724608 lm_head_repeat_margin=9/21 max_gap_milli=233 phrase_novelty=6/63 max_ngram=2 gap_skips=23 max_gap_milli=303 retentions=5 | Repetition: ratio=0.25 max_run=2 unique=12/64]
>
```

### R48-exact-prefill-topk4 run 1 prompt 2

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  I rem mir mir.swing mir.swing mir mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir mir.swing.swing mir.swing mir mir.swing.swing mir.swing mir.swing.swing mir mir.swing mir.swing.swing mir.swing.swing mir mir.swing mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir

[TTFT/Prefill: 16.08s | Decode: 54.67 tok/s | E2E: 3.71 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6111 fallbacks=0 max_topk=4 skipped_madds=77724767232 scratch=32 bytes input_tile_reads=28476 input_tile_bytes=254564352 lm_head_repeat_margin=22/29 max_gap_milli=195 phrase_novelty=10/63 max_ngram=2 gap_skips=34 max_gap_milli=233 retentions=5 | Repetition: ratio=0.27 max_run=2 unique=4/64]
>
```

### R49-exact-prefill-lmhead-topk5 run 1 prompt 2

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  I rem mir mir.swing mir mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing mir mir.swing mir.swing.swing mir.swing mir.swing mir mir.swing mir.swing mir mir.swing mir.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing mir.swing mir mir.swing.swing mir mir.swing.swing

[TTFT/Prefill: 16.21s | Decode: 76.91 tok/s | E2E: 3.76 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6111 fallbacks=0 max_topk=5 skipped_madds=77716687104 scratch=40 bytes input_tile_reads=28539 input_tile_bytes=270724608 lm_head_repeat_margin=20/31 max_gap_milli=249 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=7/63 max_ngram=2 gap_skips=30 max_gap_milli=295 retentions=5 | Repetition: ratio=0.29 max_run=2 unique=4/64]
>
```

### R48-exact-prefill-topk4 run 1 prompt 3

Prompt: `explain artificial intelligence in one sentence`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> :

 mir disngx mir mir.swing mir disinnamon mir mir dis mir.swing Mundelage mir mir.swing mir mir.swing mir mir fÃŃs fÃŃs Mundrede mir.swing mir mir.swing dis fÃŃs disstitution Mund fÃŃs fÃŃs '// mir.swing mir mir.swing mir.swing mir mir.swingFinderFinder fÃŃs fÃŃs '//.swing mir.swing mir mir.swing mir

[TTFT/Prefill: 18.58s | Decode: 32.53 tok/s | E2E: 3.12 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6111 fallbacks=0 max_topk=4 skipped_madds=77724767232 scratch=32 bytes input_tile_reads=28476 input_tile_bytes=254564352 lm_head_repeat_margin=11/23 max_gap_milli=328 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=3/63 max_ngram=2 gap_skips=10 max_gap_milli=217 retentions=3 | Repetition: ratio=0.21 max_run=2 unique=13/64]
>
```

### R49-exact-prefill-lmhead-topk5 run 1 prompt 3

Prompt: `explain artificial intelligence in one sentence`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> :

 mir.swing.swing disngx disinnamon mir mir.swing mir.swing dis diffDiffiness rem vern mir mir.swing mir mir.swing.swing mir mir.swing mir.swing.swing mir.swing mir.swing.swing mir mir.swing mir.swing mir mir.swing mir.swing mir mir.swing stickersboa skinclub Mercmandaæķ·æķ·iliz discreativecommons mir.swing mir

[TTFT/Prefill: 16.35s | Decode: 22.11 tok/s | E2E: 3.33 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6111 fallbacks=0 max_topk=5 skipped_madds=77716687104 scratch=40 bytes input_tile_reads=28539 input_tile_bytes=270724608 lm_head_repeat_margin=9/15 max_gap_milli=736 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=7/63 max_ngram=2 gap_skips=16 max_gap_milli=254 retentions=2 | Repetition: ratio=0.19 max_run=2 unique=20/64]
>
```

### R48-exact-prefill-topk4 run 1 prompt 4

Prompt: `write a short friendly reply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  toose diff fÃŃs fÃŃs Ledger Ledger fÃŃs Ledger Ledger fÃŃs fÃŃs Mund fÃŃs Mund Mund fÃŃs">%.swing mir mir.swing mir mir.swing disegendrede mir.swing mir mir.swing fÃŃs fÃŃs Mund carnhaul fÃŃs fÃŃs Hubbard.swingFinder fÃŃs fÃŃs Mundrede mir.swingStrip mir Mundclub carnhaulelage.swing fÃŃs fÃŃs '//.swingFinder fÃŃselage

[TTFT/Prefill: 17.62s | Decode: 41.32 tok/s | E2E: 3.34 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6111 fallbacks=0 max_topk=4 skipped_madds=77724767232 scratch=32 bytes input_tile_reads=28476 input_tile_bytes=254564352 lm_head_repeat_margin=6/17 max_gap_milli=517 phrase_novelty=1/63 max_ngram=2 gap_skips=8 max_gap_milli=275 retentions=1 | Repetition: ratio=0.19 max_run=2 unique=20/64]
>
```

### R49-exact-prefill-lmhead-topk5 run 1 prompt 4

Prompt: `write a short friendly reply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  to.swing fÃŃs Ledgerinnamon mir.swing mir discab diff Mundelage.swing fÃŃs fÃŃsclubrede remfineiness Mund Mundboaboa Mund Mundboa mÅ©iclub Mercfine Mund Mundboaboa Mund fÃŃs fÃŃs Mundrede.swing MundilersvestCab.swing  mir.swing fÃŃs fÃŃsilers Agenciesrede mir.swing fÃŃsrede.swing mir.swing.swing mir

[TTFT/Prefill: 15.90s | Decode: 35.24 tok/s | E2E: 3.62 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6111 fallbacks=0 max_topk=5 skipped_madds=77716687104 scratch=40 bytes input_tile_reads=28539 input_tile_bytes=270724608 lm_head_repeat_margin=3/11 max_gap_milli=635 phrase_novelty=2/63 max_ngram=2 gap_skips=1 max_gap_milli=156 | Repetition: ratio=0.14 max_run=2 unique=24/64]
>
```

### R48-exact-prefill-topk4 run 1 prompt 5

Prompt: `what is two plus two?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  The disagli.swing táº¥n mir.swing glirsch mir.swing mir mir.swing mir.swing rem remãĥ©ãĤ¤ãĥ³ mir.swing resp mir mir.swing diff mir mir.swing Mund Mund.swingStrip mir.swing hoop mir.swing pronto mir.swing.swing mir mir.swing mir mir.swing mir mir.swing Mund carnhaul carnØ³ØªÙħelage.swing Mund carnhaulhaul carnØ³ØªÙħ

[TTFT/Prefill: 16.18s | Decode: 35.10 tok/s | E2E: 3.56 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6111 fallbacks=0 max_topk=4 skipped_madds=77724767232 scratch=32 bytes input_tile_reads=28476 input_tile_bytes=254564352 lm_head_repeat_margin=14/24 max_gap_milli=394 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=3/63 max_ngram=2 gap_skips=9 max_gap_milli=199 retentions=2 | Repetition: ratio=0.16 max_run=2 unique=20/64]
>
```

### R49-exact-prefill-lmhead-topk5 run 1 prompt 5

Prompt: `what is two plus two?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  The rem remizens rem remose.swing gl dis dis rem remçİĩ.swing diff diff Diff mir mir.swing diff MundÃ¤ÃŁ.swing mir mir.swing Mundrede mir.swing Mund Mundhaul Mundrede mir mir.swingWindowState mir mir.swingstrip.swing Mundclub Merc Mundhaulilers Agenciesrede.swing mir.swing mir mir.swing  mir.swing fÃŃs

[TTFT/Prefill: 16.16s | Decode: 32.71 tok/s | E2E: 3.54 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6111 fallbacks=0 max_topk=5 skipped_madds=77716687104 scratch=40 bytes input_tile_reads=28539 input_tile_bytes=270724608 lm_head_repeat_margin=5/15 max_gap_milli=798 phrase_novelty=1/63 max_ngram=2 gap_skips=4 max_gap_milli=238 | Repetition: ratio=0.17 max_run=2 unique=23/64]
>
```
