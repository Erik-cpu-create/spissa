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
- Runs: 1 alternating control/candidate pairs per prompt
- Target decode band: 30.00-40.00 tok/s
- Profile phases: false

## Summary

| variant | runs | floor accepted | band accepted | min decode tok/s | max decode tok/s | avg decode tok/s | avg unique tokens | avg repetition ratio |
|---|---:|---|---|---:|---:|---:|---:|---:|
| R47-global-topk5 | 5 | false | false | 24.51 | 38.37 | 31.62 | 12.60 | 0.25 |
| R47-global-topk6 | 5 | false | false | 18.13 | 25.15 | 21.41 | 17.20 | 0.24 |

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R47-global-topk5 | 1 | 1 | 12.85 | 29.93 | 4.28 | 64 | 66 | 1050689536 | 0.27 | 2 | 12/64 |
| R47-global-topk6 | 1 | 1 | 13.49 | 25.15 | 4.00 | 64 | 66 | 1050689536 | 0.30 | 2 | 13/64 |
| R47-global-topk5 | 1 | 2 | 13.03 | 38.37 | 4.36 | 64 | 66 | 1050689536 | 0.25 | 2 | 16/64 |
| R47-global-topk6 | 1 | 2 | 13.57 | 18.13 | 3.76 | 64 | 66 | 1050689536 | 0.16 | 2 | 22/64 |
| R47-global-topk5 | 1 | 3 | 15.21 | 30.87 | 3.71 | 64 | 68 | 1050689536 | 0.27 | 2 | 8/64 |
| R47-global-topk6 | 1 | 3 | 14.40 | 24.20 | 3.76 | 64 | 68 | 1050689536 | 0.27 | 2 | 13/64 |
| R47-global-topk5 | 1 | 4 | 14.46 | 34.43 | 3.93 | 64 | 68 | 1050689536 | 0.29 | 2 | 6/64 |
| R47-global-topk6 | 1 | 4 | 13.74 | 19.80 | 3.78 | 64 | 68 | 1050689536 | 0.29 | 2 | 17/64 |
| R47-global-topk5 | 1 | 5 | 13.45 | 24.51 | 4.00 | 64 | 69 | 1050689536 | 0.19 | 2 | 21/64 |
| R47-global-topk6 | 1 | 5 | 14.01 | 19.79 | 3.72 | 64 | 69 | 1050689536 | 0.16 | 2 | 21/64 |

## Interpretation

R47 rejects global top-k widening as a strict-band preset. Top-k 5 brought the
high side under 40 tok/s in this one-run sweep, but it failed the 30 tok/s floor
on prompt 1 and prompt 5. Top-k 6 was too slow on every prompt class.

Decision: failed. Global widening can reduce high-side throughput, but it does
so by losing the low-end floor rather than by improving useful chat quality.

## Raw Output

### R47-global-topk5 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   mirihn.swing mir mir.swing Mund mir mir.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing mir.swing mir.swing mir mir.swing.swing mir mir.swing.swing mir.swing mir mir.swing.swing mir.swing mir.swing.swing mir.swingÃ¤ÃŁ.Bundle.Bundleato com mir.swing mir.swing.swing mir.swinglaughterusaniat mir

[TTFT/Prefill: 12.85s | Decode: 29.93 tok/s | E2E: 4.28 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=5 skipped_madds=77954973696 scratch=40 bytes input_tile_reads=35600 input_tile_bytes=319488000 lm_head_repeat_margin=11/20 max_gap_milli=744 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=8/64 max_ngram=2 gap_skips=14 max_gap_milli=282 retentions=4 | Repetition: ratio=0.27 max_run=2 unique=12/64]
>
```

### R47-global-topk6 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   mir.swing mir mir rem mir mir rem.swing mir mir.swing mir mir.swing mir.swing mir mirlaughter mir mir  mir mir.swing mir mirlaughter mir mir.swing mir mirlaughter.Bundle.BundleÂŃiÂŃiÃ¤ÃŁÃ¤ÃŁæķ·Ã¤ÃŁÃ¤ÃŁ.Bundle.BundleÂŃiÂŃizÄħ mir.swinglaughterÂŃiÂŃiÄħÅ¼Ã¤ÃŁÃ¤ÃŁÅĽÄĩÃ¤ÃŁÂŃiÂŃiÃ¤ÃŁaggio

[TTFT/Prefill: 13.49s | Decode: 25.15 tok/s | E2E: 4.00 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=6 skipped_madds=77923024896 scratch=48 bytes input_tile_reads=42720 input_tile_bytes=383385600 lm_head_repeat_margin=4/23 max_gap_milli=1331 phrase_novelty=5/64 max_ngram=2 gap_skips=13 max_gap_milli=801 retentions=1 | Repetition: ratio=0.30 max_run=2 unique=13/64]
>
```

### R47-global-topk5 run 1 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -Cal mir miryth mirmith mir mirounsilersellig mirihn mir mir.swing mir mir.swing.swing mir mir.swing mir mir.swing mir mir.swing.swing mir mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing.swing mir mir.swingascalulaire.swing mir.swingascalboaascalascalstitutionsrede mir.swingfinefineipers mir

[TTFT/Prefill: 13.03s | Decode: 38.37 tok/s | E2E: 4.36 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=5 skipped_madds=77954973696 scratch=40 bytes input_tile_reads=35600 input_tile_bytes=319488000 lm_head_repeat_margin=8/19 max_gap_milli=416 phrase_novelty=6/64 max_ngram=2 gap_skips=11 max_gap_milli=275 retentions=5 | Repetition: ratio=0.25 max_run=2 unique=16/64]
>
```

### R47-global-topk6 run 1 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -Cal mir.swing mirouns.swing mir miryth.swing mir.swing mir.swing mir inset mir.swinglaughterlaughterdoctype mir mir.swing  mir okogradusan mir mirki mirathed vern mirathed mirÙĩÙħÃ¤ÃŁÃ¤ÃŁæķ·Ã¤ÃŁÃ¤ÃŁ.swing mirycz mir mir.swingÂŃiÃ¤ÃŁÃ¤ÃŁæķ·laughterberra.swing mirÃ³ÅĤ mir mir.swing mir mir

[TTFT/Prefill: 13.57s | Decode: 18.13 tok/s | E2E: 3.76 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=6 skipped_madds=77923024896 scratch=48 bytes input_tile_reads=42720 input_tile_bytes=383385600 lm_head_repeat_margin=6/16 max_gap_milli=643 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=0/64 max_ngram=2 gap_skips=7 max_gap_milli=429 | Repetition: ratio=0.16 max_run=2 unique=22/64]
>
```

### R47-global-topk5 run 1 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   mir.swing mir.swing.swing mir mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir mir.swing mir mir.swing.swing mir mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swingippersegasusascalascalacidad mir mir.swing mir.swingrede mir mir.swing.swing mir mir

[TTFT/Prefill: 15.21s | Decode: 30.87 tok/s | E2E: 3.71 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=5 skipped_madds=77954973696 scratch=40 bytes input_tile_reads=35600 input_tile_bytes=319488000 lm_head_repeat_margin=15/24 max_gap_milli=525 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=8/64 max_ngram=2 gap_skips=23 max_gap_milli=300 retentions=5 | Repetition: ratio=0.27 max_run=2 unique=8/64]
>
```

### R47-global-topk6 run 1 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   mir mir.swing mir mirlaughter mir mir.swing mir mir.swing mir mir.swing.swing mir mir.swing mir mirlaughter mir mir.swing mir mirlaughter.swing mir Legend dil mir mir.swing vive mir mir.swing mir mirlaughterior.Bundle.swing mir.swingnessnesselage mir mir.swing mir mir.swinglaughterusan.swing mir mir.swingLOY

[TTFT/Prefill: 14.40s | Decode: 24.20 tok/s | E2E: 3.76 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=6 skipped_madds=77923024896 scratch=48 bytes input_tile_reads=42720 input_tile_bytes=383385600 lm_head_repeat_margin=5/21 max_gap_milli=288 phrase_novelty=6/64 max_ngram=2 gap_skips=14 max_gap_milli=288 | Repetition: ratio=0.27 max_run=2 unique=13/64]
>
```

### R47-global-topk5 run 1 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  ngx mir mir.swing mir mir.swing.swing mir.swing vs rem.swing mir.swing.swing mir mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir mir.swing.swing mir.swing mir mir.swing.swing mir mir.swing.swing mir.swing mir.swing mir mir.swing.swing mir.swing mir mir.swing.swing mir

[TTFT/Prefill: 14.46s | Decode: 34.43 tok/s | E2E: 3.93 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=5 skipped_madds=77954973696 scratch=40 bytes input_tile_reads=35600 input_tile_bytes=319488000 lm_head_repeat_margin=14/22 max_gap_milli=207 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=10/64 max_ngram=2 gap_skips=25 max_gap_milli=323 retentions=4 | Repetition: ratio=0.29 max_run=2 unique=6/64]
>
```

### R47-global-topk6 run 1 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   mir.swing.swing dis quotesinnamon rem/host.swing mir mir.swing mir mir.swing disc .swing mir mir.swing mir.swing mir mir.swing mir mirlaughter mir mir.swing.swing mir.swing.swing mir mir.swing.swing mir mirzamet projectile mir.swing mir mirlaughterlaughter Friendly  mir mir.swingÃ¤ÃŁÃ¤ÃŁanasÃ¤ÃŁÃ¤ÃŁnÄĽnÄĽ

[TTFT/Prefill: 13.74s | Decode: 19.80 tok/s | E2E: 3.78 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=6 skipped_madds=77923024896 scratch=48 bytes input_tile_reads=42720 input_tile_bytes=383385600 lm_head_repeat_margin=4/21 max_gap_milli=635 phrase_novelty=3/64 max_ngram=2 gap_skips=12 max_gap_milli=550 retentions=2 | Repetition: ratio=0.29 max_run=2 unique=17/64]
>
```

### R47-global-topk5 run 1 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  naturallyose mir mir diff diff fÃŃs fÃŃs diff.swing mir.swing.swing interactions Mund fÃŃs Mund Mundiness disfinefineclubrede mir.swing mir.swing mir.swing_orders mir mir.swingfineelage mir.swing.swing mir mir.swing.swing mÅ©iæķ·æķ·rede mir.swingfineilers mir.swingilerskeleton mir.swingolandclub Mercelage mir mir.swing

[TTFT/Prefill: 13.45s | Decode: 24.51 tok/s | E2E: 4.00 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=5 skipped_madds=77954973696 scratch=40 bytes input_tile_reads=35600 input_tile_bytes=319488000 lm_head_repeat_margin=9/19 max_gap_milli=490 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=2/64 max_ngram=2 gap_skips=3 max_gap_milli=233 | Repetition: ratio=0.19 max_run=2 unique=21/64]
>
```

### R47-global-topk6 run 1 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  below mir rem Fin diff diff Mund diff diff Mund Mund dis Mercasonry Mund Mundrede mir mir.swing mir.swing mir mirlaughterocr Merc comfinefineilersclub Merc mirrede mir lipolandOverriderede mir mir comilers.swing.swingfinerede mir.swing mirlaughterfinerede mir mir.swinglaughter helicopters mirlaughterilers.swing mir

[TTFT/Prefill: 14.01s | Decode: 19.79 tok/s | E2E: 3.72 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=6 skipped_madds=77923024896 scratch=48 bytes input_tile_reads=42720 input_tile_bytes=383385600 lm_head_repeat_margin=3/13 max_gap_milli=647 phrase_novelty=2/64 max_ngram=2 gap_skips=4 max_gap_milli=452 | Repetition: ratio=0.16 max_run=2 unique=21/64]
>
```
