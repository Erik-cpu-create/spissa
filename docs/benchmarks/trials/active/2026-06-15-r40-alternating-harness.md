# R40 Alternating Benchmark Harness

## Setup

- Model: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.rllm`
- Runner: `/Users/deansanbhnanwr/Projects/rllm/target/release/llama-test`
- Prompt: `good morning`
- Runs: 2 alternating control/candidate pairs
- Target decode band: 30.00-40.00 tok/s
- Profile phases: false

## Summary

| variant | runs | accepted | min decode tok/s | max decode tok/s | avg decode tok/s | avg unique tokens | avg repetition ratio |
|---|---:|---|---:|---:|---:|---:|---:|
| R37-control | 2 | false | 29.86 | 45.43 | 37.64 | 17.00 | 0.11 |
| R39-retention-100 | 2 | true | 34.82 | 35.15 | 34.98 | 15.00 | 0.13 |

## Runs

| variant | run | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R37-control | 1 | 12.41 | 29.86 | 4.41 | 64 | 66 | 1050689536 | 0.11 | 2 | 17/64 |
| R39-retention-100 | 1 | 13.06 | 34.82 | 4.31 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R37-control | 2 | 13.07 | 45.43 | 4.43 | 64 | 66 | 1050689536 | 0.11 | 2 | 17/64 |
| R39-retention-100 | 2 | 13.38 | 35.15 | 4.22 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |

## Raw Output

### R37-control run 1

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mirlaughter projectile

[TTFT/Prefill: 12.41s | Decode: 29.86 tok/s | E2E: 4.41 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 | Repetition: ratio=0.11 max_run=2 unique=17/64]
> 
```

### R39-retention-100 run 1

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 13.06s | Decode: 34.82 tok/s | E2E: 4.31 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
> 
```

### R37-control run 2

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mirlaughter projectile

[TTFT/Prefill: 13.07s | Decode: 45.43 tok/s | E2E: 4.43 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 | Repetition: ratio=0.11 max_run=2 unique=17/64]
> 
```

### R39-retention-100 run 2

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 13.38s | Decode: 35.15 tok/s | E2E: 4.22 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
> 
```

