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
| R43-retention100-topk4 | 5 | true | false | 30.90 | 44.82 | 37.32 | 17.60 | 0.18 |
| R52-attention-topk8 | 5 | false | false | 26.20 | 43.35 | 33.57 | 15.80 | 0.29 |

## Interpretation

R52 rejects the lighter attention top-k 8 compromise. It recovered average
throughput compared with R51, but it still failed the 30 tok/s floor on three
of five prompts and exceeded the strict 40 tok/s upper band on two prompts.
Quality metrics also moved backward: average unique tokens dropped from
17.60/64 to 15.80/64 and repetition rose from 0.18 to 0.29.

Decision: failed. Attention top-k 8 is closer on speed than top-k 16, but it
does not produce a stable floor or better output. Static attention widening
should not be the next chat-readiness path.

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R43-retention100-topk4 | 1 | 1 | 12.35 | 37.91 | 4.57 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R52-attention-topk8 | 1 | 1 | 12.89 | 43.35 | 4.46 | 64 | 66 | 1050689536 | 0.35 | 2 | 9/64 |
| R43-retention100-topk4 | 1 | 2 | 13.47 | 44.82 | 4.30 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R52-attention-topk8 | 1 | 2 | 13.54 | 42.93 | 4.26 | 64 | 68 | 1050689536 | 0.44 | 2 | 7/64 |
| R43-retention100-topk4 | 1 | 3 | 13.88 | 30.90 | 4.02 | 64 | 70 | 1050689536 | 0.16 | 2 | 22/64 |
| R52-attention-topk8 | 1 | 3 | 14.13 | 28.55 | 3.92 | 64 | 70 | 1050689536 | 0.25 | 2 | 20/64 |
| R43-retention100-topk4 | 1 | 4 | 13.56 | 40.14 | 4.23 | 64 | 69 | 1050689536 | 0.16 | 2 | 20/64 |
| R52-attention-topk8 | 1 | 4 | 13.56 | 26.81 | 4.02 | 64 | 69 | 1050689536 | 0.16 | 2 | 20/64 |
| R43-retention100-topk4 | 1 | 5 | 14.21 | 32.82 | 3.97 | 64 | 70 | 1050689536 | 0.22 | 2 | 21/64 |
| R52-attention-topk8 | 1 | 5 | 13.94 | 26.20 | 3.92 | 64 | 70 | 1050689536 | 0.25 | 2 | 23/64 |

## Raw Output

### R43-retention100-topk4 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 12.35s | Decode: 37.91 tok/s | E2E: 4.57 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R52-attention-topk8 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   mir.swing Douglas mir mir.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing mir.swing mir mir.swing.swing mir mir.swing mir mir.swing mir mir.swing.swing mir mir.swing mir mir.swing mir mir.swing mir mirlaughterÂŃi.Bundle.BundleÂŃiÃ¤ÃŁ.BundleÃ¤ÃŁzÄħ mir mir.swing mir mir.swing.swing mir mir

[TTFT/Prefill: 12.89s | Decode: 43.35 tok/s | E2E: 4.46 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77966278656 scratch=64 bytes input_tile_reads=44608 input_tile_bytes=296878080 lm_head_repeat_margin=2/18 max_gap_milli=866 phrase_novelty=7/64 max_ngram=2 gap_skips=20 max_gap_milli=159 retentions=1 | Repetition: ratio=0.35 max_run=2 unique=9/64]
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

[TTFT/Prefill: 13.47s | Decode: 44.82 tok/s | E2E: 4.30 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R52-attention-topk8 run 1 prompt 2

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> - mir mir Winston mir.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing mir mir.swing.swing mir mir.swingÃ¤ÃŁÃ¤ÃŁ.Bundle.BundleYK mir mir.swing

[TTFT/Prefill: 13.54s | Decode: 42.93 tok/s | E2E: 4.26 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77966278656 scratch=64 bytes input_tile_reads=44608 input_tile_bytes=296878080 lm_head_repeat_margin=0/18 max_gap_milli=1169 phrase_novelty=12/64 max_ngram=2 gap_skips=4 max_gap_milli=203 | Repetition: ratio=0.44 max_run=2 unique=7/64]
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

[TTFT/Prefill: 13.88s | Decode: 30.90 tok/s | E2E: 4.02 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=9/17 max_gap_milli=338 phrase_novelty=3/64 max_ngram=2 gap_skips=6 max_gap_milli=226 | Repetition: ratio=0.16 max_run=2 unique=22/64]
>
```

### R52-attention-topk8 run 1 prompt 3

Prompt: `explain artificial intelligence in one sentence`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  ngx mir mir.swing mir mir disngx mir.swing mir mir disROP mir mir.swing rem rem Omaha mir mir.swing mir mir.swing fÃŃsï¼ģï¼ģ

ipel mir mir.swing.swing mir.swing mir.swing.swing disilerslaughter mir mir.swing mir mir.swing.swing mir mir.swing.swingTERN.Bundle.BundleYK miriorlaughterilersettieledÄħd

[TTFT/Prefill: 14.13s | Decode: 28.55 tok/s | E2E: 3.92 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77966278656 scratch=64 bytes input_tile_reads=44608 input_tile_bytes=296878080 lm_head_repeat_margin=3/16 max_gap_milli=1034 phrase_novelty=6/64 max_ngram=2 gap_skips=4 max_gap_milli=158 retentions=4 | Repetition: ratio=0.25 max_run=2 unique=20/64]
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

[TTFT/Prefill: 13.56s | Decode: 40.14 tok/s | E2E: 4.23 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=6/16 max_gap_milli=480 phrase_novelty=1/64 max_ngram=2 gap_skips=2 max_gap_milli=202 retentions=2 | Repetition: ratio=0.16 max_run=2 unique=20/64]
>
```

### R52-attention-topk8 run 1 prompt 4

Prompt: `write a short friendly reply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Harm Mund Ledger gl mir mir.swing mir mir dis fÃŃs Mund.swing dis accident mir.swing rib fÃŃs Mund sob sob mir dis membranes mir mir.swing mir mir.swingstitute mir.swingstitute.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir mirlaughterLaughæķ·ulairecreativecommons mir.swing mir mir.swing Mundlaughterilersilers entrustedlaughter

[TTFT/Prefill: 13.56s | Decode: 26.81 tok/s | E2E: 4.02 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77966278656 scratch=64 bytes input_tile_reads=44608 input_tile_bytes=296878080 lm_head_repeat_margin=10/19 max_gap_milli=452 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=4/64 max_ngram=2 gap_skips=10 max_gap_milli=223 retentions=3 | Repetition: ratio=0.16 max_run=2 unique=20/64]
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

[TTFT/Prefill: 14.21s | Decode: 32.82 tok/s | E2E: 3.97 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=13/27 max_gap_milli=929 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=1/64 max_ngram=2 gap_skips=12 max_gap_milli=201 retentions=3 | Repetition: ratio=0.22 max_run=2 unique=21/64]
>
```

### R52-attention-topk8 run 1 prompt 5

Prompt: `what is two plus two?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  ose mir mir.swing mir.swing.swing mir mir.swing mir mir.swing mir mir.swingfinefineulaireÃ¤ÃŁCallable mir mir.swing mir inset mirclublaughter mir mir.swinguretteMich.swing Finæķ·æķ·laughterlaughtermeld inset mir mir discreativecommons.Bundlefinefine nye.Bundle.BundleÂŃiÂŃiettiæķ·.Bundle.BundleYK.swing mir.swing.swing

[TTFT/Prefill: 13.94s | Decode: 26.20 tok/s | E2E: 3.92 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77966278656 scratch=64 bytes input_tile_reads=44608 input_tile_bytes=296878080 lm_head_repeat_margin=2/16 max_gap_milli=874 phrase_novelty=2/64 max_ngram=2 gap_skips=8 max_gap_milli=248 | Repetition: ratio=0.25 max_run=2 unique=23/64]
>
```
