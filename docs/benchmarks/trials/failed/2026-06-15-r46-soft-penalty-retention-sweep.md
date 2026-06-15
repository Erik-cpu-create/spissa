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
| R43-retention100 | 5 | false | false | 28.22 | 52.59 | 37.02 | 16.00 | 0.17 |
| R46-retention100-soft-penalty300 | 5 | true | false | 32.63 | 67.87 | 45.22 | 16.00 | 0.17 |

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R43-retention100 | 1 | 1 | 12.31 | 38.67 | 4.59 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R46-retention100-soft-penalty300 | 1 | 1 | 13.73 | 67.87 | 4.37 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R43-retention100 | 1 | 2 | 14.80 | 34.94 | 3.85 | 64 | 66 | 1050689536 | 0.19 | 2 | 18/64 |
| R46-retention100-soft-penalty300 | 1 | 2 | 13.69 | 37.72 | 4.17 | 64 | 66 | 1050689536 | 0.19 | 2 | 18/64 |
| R43-retention100 | 1 | 3 | 14.29 | 52.59 | 4.13 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R46-retention100-soft-penalty300 | 1 | 3 | 14.17 | 41.14 | 4.08 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R43-retention100 | 1 | 4 | 14.93 | 30.67 | 3.77 | 64 | 68 | 1050689536 | 0.19 | 2 | 15/64 |
| R46-retention100-soft-penalty300 | 1 | 4 | 14.37 | 32.63 | 3.93 | 64 | 68 | 1050689536 | 0.19 | 2 | 15/64 |
| R43-retention100 | 1 | 5 | 14.59 | 28.22 | 3.80 | 64 | 69 | 1050689536 | 0.10 | 2 | 22/64 |
| R46-retention100-soft-penalty300 | 1 | 5 | 14.41 | 46.73 | 4.06 | 64 | 69 | 1050689536 | 0.10 | 2 | 22/64 |

## Interpretation

R46 rejects soft novelty penalty 300 as a quality/band improvement. The
candidate passed the 30 tok/s floor in this one-run sweep, but it did not
improve average unique tokens or repetition ratio versus the R43 retention
preset. It also increased high-side variance, reaching 67.87 tok/s.

Decision: failed. The controller is cheap enough to preserve the floor in this
sweep, but it does not move output quality or the strict 30-40 tok/s band in the
right direction.

## Raw Output

### R43-retention100 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 12.31s | Decode: 38.67 tok/s | E2E: 4.59 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R46-retention100-soft-penalty300 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 13.73s | Decode: 67.87 tok/s | E2E: 4.37 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R43-retention100 run 1 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir.swing mir.swing.swing miryth mirihn mir mir.swing mir mir.swing.swing mirosteilersstitutions mir mir.swing dis.swing mir mir.swingÃ¤ÃŁ.swing fÃŃselage.swing.swing mir mirrede mir.swing mir mir.swing mirrawn.swingFinder.swing mir mir.swing mir mir.swing mir.swing mir mir.swingThreadPool carnhaulelage.swing

[TTFT/Prefill: 14.80s | Decode: 34.94 tok/s | E2E: 3.85 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=12/23 max_gap_milli=123 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=10 max_gap_milli=242 retentions=3 | Repetition: ratio=0.19 max_run=2 unique=18/64]
>
```

### R46-retention100-soft-penalty300 run 1 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir.swing mir.swing.swing miryth mirihn mir mir.swing mir mir.swing.swing mirosteilersstitutions mir mir.swing dis.swing mir mir.swingÃ¤ÃŁ.swing fÃŃselage.swing.swing mir mirrede mir.swing mir mir.swing mirrawn.swingFinder.swing mir mir.swing mir mir.swing mir.swing mir mir.swingThreadPool carnhaulelage.swing

[TTFT/Prefill: 13.69s | Decode: 37.72 tok/s | E2E: 4.17 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=12/23 max_gap_milli=123 adaptive_throttles=1 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=10 max_gap_milli=242 retentions=3 | Repetition: ratio=0.19 max_run=2 unique=18/64]
>
```

### R43-retention100 run 1 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir dis/etcinnamon dis-INF mir mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing mir mir.swing.swing mir.swing mir mir.swing.swing mir.swing mir.swing%dfinefineipers mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing mir

[TTFT/Prefill: 14.29s | Decode: 52.59 tok/s | E2E: 4.13 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R46-retention100-soft-penalty300 run 1 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir dis/etcinnamon dis-INF mir mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing mir mir.swing.swing mir.swing mir mir.swing.swing mir.swing mir.swing%dfinefineipers mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing mir

[TTFT/Prefill: 14.17s | Decode: 41.14 tok/s | E2E: 4.08 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R43-retention100 run 1 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  osengx mir disngx mir mir.swing mir mir.swingose mir mir.swing Ledger Ledger%d.io.swing mir.swing mir.swing.CopyTo disstitutions mir.swing mir mir.swing mir.swing disorexissororexorexfineelage mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing

[TTFT/Prefill: 14.93s | Decode: 30.67 tok/s | E2E: 3.77 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/26 max_gap_milli=489 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=2/64 max_ngram=2 gap_skips=19 max_gap_milli=210 retentions=6 | Repetition: ratio=0.19 max_run=2 unique=15/64]
>
```

### R46-retention100-soft-penalty300 run 1 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  osengx mir disngx mir mir.swing mir mir.swingose mir mir.swing Ledger Ledger%d.io.swing mir.swing mir.swing.CopyTo disstitutions mir.swing mir mir.swing mir.swing disorexissororexorexfineelage mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing mir mir.swing

[TTFT/Prefill: 14.37s | Decode: 32.63 tok/s | E2E: 3.93 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/26 max_gap_milli=489 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=2/64 max_ngram=2 gap_skips=19 max_gap_milli=210 retentions=6 | Repetition: ratio=0.19 max_run=2 unique=15/64]
>
```

### R43-retention100 run 1 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir disngx Ledger Ledger fÃŃs fÃŃs Mund Mund">%cash rem diff Mundcash fÃŃs Mund Mund Canterrede rem fÃŃs fÃŃs%degend">%rede mir.swing mir mir.swing miregend fÃŃselage mir.swing mir fÃŃselage.swing fÃŃselage mir.swing fÃŃselage.swingStrip.swingstitute carnelage.swing fÃŃsclub carnhaulelage.swing fÃŃs

[TTFT/Prefill: 14.59s | Decode: 28.22 tok/s | E2E: 3.80 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=4/9 max_gap_milli=644 phrase_novelty=2/64 max_ngram=2 gap_skips=3 max_gap_milli=326 | Repetition: ratio=0.10 max_run=2 unique=22/64]
>
```

### R46-retention100-soft-penalty300 run 1 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir disngx Ledger Ledger fÃŃs fÃŃs Mund Mund">%cash rem diff Mundcash fÃŃs Mund Mund Canterrede rem fÃŃs fÃŃs%degend">%rede mir.swing mir mir.swing miregend fÃŃselage mir.swing mir fÃŃselage.swing fÃŃselage mir.swing fÃŃselage.swingStrip.swingstitute carnelage.swing fÃŃsclub carnhaulelage.swing fÃŃs

[TTFT/Prefill: 14.41s | Decode: 46.73 tok/s | E2E: 4.06 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=4/9 max_gap_milli=644 phrase_novelty=2/64 max_ngram=2 gap_skips=3 max_gap_milli=326 | Repetition: ratio=0.10 max_run=2 unique=22/64]
>
```
