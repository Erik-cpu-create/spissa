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
| R43-retention100-topk4 | 5 | true | false | 32.59 | 44.94 | 38.04 | 17.60 | 0.18 |
| R51-attention-topk16 | 5 | false | false | 18.26 | 26.94 | 23.79 | 14.80 | 0.31 |

## Interpretation

R51 rejects attention top-k 16 as a static quality fix. Increasing attention
candidate width increased tile reads and scratch use, but it did not improve
chat output. Candidate decode stayed below the 30 tok/s floor on every prompt,
average unique tokens fell from 17.60/64 to 14.80/64, and repetition rose from
0.18 to 0.31.

Decision: failed. Static attention widening spends the speed budget in the
wrong place; the bottleneck is not solved by giving attention a larger sparse
set globally.

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R43-retention100-topk4 | 1 | 1 | 12.12 | 39.37 | 4.67 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R51-attention-topk16 | 1 | 1 | 13.03 | 26.36 | 4.15 | 64 | 66 | 1050689536 | 0.30 | 2 | 13/64 |
| R43-retention100-topk4 | 1 | 2 | 13.19 | 44.94 | 4.39 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R51-attention-topk16 | 1 | 2 | 13.51 | 26.94 | 4.04 | 64 | 68 | 1050689536 | 0.30 | 2 | 9/64 |
| R43-retention100-topk4 | 1 | 3 | 13.91 | 36.45 | 4.09 | 64 | 70 | 1050689536 | 0.16 | 2 | 22/64 |
| R51-attention-topk16 | 1 | 3 | 13.84 | 24.47 | 3.90 | 64 | 70 | 1050689536 | 0.37 | 2 | 10/64 |
| R43-retention100-topk4 | 1 | 4 | 13.39 | 36.83 | 4.24 | 64 | 69 | 1050689536 | 0.16 | 2 | 20/64 |
| R51-attention-topk16 | 1 | 4 | 13.79 | 22.90 | 3.87 | 64 | 69 | 1050689536 | 0.33 | 2 | 17/64 |
| R43-retention100-topk4 | 1 | 5 | 13.87 | 32.59 | 4.05 | 64 | 70 | 1050689536 | 0.22 | 2 | 21/64 |
| R51-attention-topk16 | 1 | 5 | 14.01 | 18.26 | 3.67 | 64 | 70 | 1050689536 | 0.27 | 2 | 25/64 |

## Raw Output

### R43-retention100-topk4 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 12.12s | Decode: 39.37 tok/s | E2E: 4.67 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R51-attention-topk16 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   mirihn mir mir.swing mir mir.swing.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing.swing mir mir.swing mir mir.swing mir mir.swinghir.Bundle.BundleujÄħÄħdÅºÃ³ÅĤÃ¤ÃŁ.Bundle.BundleÃ¤ÃŁzÄħ mir mir.swing mir mir.swing mir mir.swingzÄħ mir mir.swingÃ¤ÃŁÃ¤ÃŁusan mir

[TTFT/Prefill: 13.03s | Decode: 26.36 tok/s | E2E: 4.15 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=16 skipped_madds=77924990976 scratch=128 bytes input_tile_reads=76864 input_tile_bytes=379453440 lm_head_repeat_margin=1/18 max_gap_milli=1201 phrase_novelty=2/64 max_ngram=2 gap_skips=24 max_gap_milli=144 retentions=1 | Repetition: ratio=0.30 max_run=2 unique=13/64]
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

[TTFT/Prefill: 13.19s | Decode: 44.94 tok/s | E2E: 4.39 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R51-attention-topk16 run 1 prompt 2

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing mirÃ¤ÃŁ.Bundlelaughter mir mir.swing mir mir.swing mir mir.swing.swing mirÂŃiÃ¤ÃŁreak mir mir.swing mir mir.swingreak mir mirÃ¤ÃŁÃ¤ÃŁ.Bundle.swing mir mir.swingÃ¤ÃŁ.swing

[TTFT/Prefill: 13.51s | Decode: 26.94 tok/s | E2E: 4.04 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=16 skipped_madds=77924990976 scratch=128 bytes input_tile_reads=76864 input_tile_bytes=379453440 lm_head_repeat_margin=0/16 max_gap_milli=537 phrase_novelty=3/64 max_ngram=2 gap_skips=21 max_gap_milli=144 retentions=1 | Repetition: ratio=0.30 max_run=2 unique=9/64]
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

[TTFT/Prefill: 13.91s | Decode: 36.45 tok/s | E2E: 4.09 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=9/17 max_gap_milli=338 phrase_novelty=3/64 max_ngram=2 gap_skips=6 max_gap_milli=226 | Repetition: ratio=0.16 max_run=2 unique=22/64]
>
```

### R51-attention-topk16 run 1 prompt 3

Prompt: `explain artificial intelligence in one sentence`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  innamon mir mir.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing mane.swing mir mir.swing span mir mir.swing mir mir.swing ÑģÐ¾ÑģÑĤÐ°Ð² mir mir.swingstride mir mir.swing mir mir.swingujÄħ.Bundle

[TTFT/Prefill: 13.84s | Decode: 24.47 tok/s | E2E: 3.90 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=16 skipped_madds=77924990976 scratch=128 bytes input_tile_reads=76864 input_tile_bytes=379453440 lm_head_repeat_margin=0/16 max_gap_milli=130 phrase_novelty=7/64 max_ngram=2 gap_skips=10 max_gap_milli=172 retentions=2 | Repetition: ratio=0.37 max_run=2 unique=10/64]
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

[TTFT/Prefill: 13.39s | Decode: 36.83 tok/s | E2E: 4.24 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=6/16 max_gap_milli=480 phrase_novelty=1/64 max_ngram=2 gap_skips=2 max_gap_milli=202 retentions=2 | Repetition: ratio=0.16 max_run=2 unique=20/64]
>
```

### R51-attention-topk16 run 1 prompt 4

Prompt: `write a short friendly reply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Harm strainngx mir mir.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing mir mir.swingÃ§uk mir mir.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swingirthday.Bundle.Bundlenel scar mir mir.swingingersÂŃiÃ¤ÃŁ Croat.swing mir mir.swing projectile mirizzle mir mir.swing...

[TTFT/Prefill: 13.79s | Decode: 22.90 tok/s | E2E: 3.87 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=16 skipped_madds=77924990976 scratch=128 bytes input_tile_reads=76864 input_tile_bytes=379453440 lm_head_repeat_margin=0/15 max_gap_milli=209 phrase_novelty=6/64 max_ngram=2 gap_skips=3 max_gap_milli=164 retentions=1 | Repetition: ratio=0.33 max_run=2 unique=17/64]
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

[TTFT/Prefill: 13.87s | Decode: 32.59 tok/s | E2E: 4.05 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=13/27 max_gap_milli=929 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=1/64 max_ngram=2 gap_skips=12 max_gap_milli=201 retentions=3 | Repetition: ratio=0.22 max_run=2 unique=21/64]
>
```

### R51-attention-topk16 run 1 prompt 5

Prompt: `what is two plus two?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   stocks mir mir.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing mir mir vern mir miressonosenÃ¥rÂŃifiresjÄħjÄħfÃ¼hrt mir mir.swingilos mir erf vern miræľºÃ¤ÃŁeled mir mir.swingbounce mir mir.swingä¿Ä¾Ä¾zÄħ mir mir.swingØ¹Ø¯ mir mir.swingettiÂŃiÂŃiogradhea

[TTFT/Prefill: 14.01s | Decode: 18.26 tok/s | E2E: 3.67 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=16 skipped_madds=77924990976 scratch=128 bytes input_tile_reads=76864 input_tile_bytes=379453440 lm_head_repeat_margin=0/14 max_gap_milli=412 phrase_novelty=3/64 max_ngram=2 gap_skips=2 max_gap_milli=266 retentions=1 | Repetition: ratio=0.27 max_run=2 unique=25/64]
>
```
