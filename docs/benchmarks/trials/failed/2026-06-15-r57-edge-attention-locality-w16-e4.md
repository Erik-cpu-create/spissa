# Alternating Benchmark Harness

## Setup

- Model: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa`
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
| R43-retention100-topk4 | 5 | true | false | 36.93 | 60.08 | 43.27 | 14.80 | 0.28 |
| R57-attn-locality-w16-e4 | 5 | true | false | 41.82 | 51.88 | 48.26 | 12.80 | 0.38 |

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R43-retention100-topk4 | 1 | 1 | 11.96 | 39.75 | 4.73 | 64 | 66 | 1050689536 | 0.25 | 7 | 14/64 |
| R57-attn-locality-w16-e4 | 1 | 1 | 12.37 | 48.46 | 4.68 | 64 | 66 | 1050689536 | 0.38 | 10 | 15/64 |
| R43-retention100-topk4 | 1 | 2 | 12.22 | 36.93 | 4.60 | 64 | 66 | 1050689536 | 0.30 | 8 | 20/64 |
| R57-attn-locality-w16-e4 | 1 | 2 | 12.29 | 41.82 | 4.64 | 64 | 66 | 1050689536 | 0.37 | 6 | 13/64 |
| R43-retention100-topk4 | 1 | 3 | 12.14 | 60.08 | 4.85 | 64 | 68 | 1050689536 | 0.19 | 5 | 8/64 |
| R57-attn-locality-w16-e4 | 1 | 3 | 12.60 | 51.16 | 4.63 | 64 | 68 | 1050689536 | 0.35 | 6 | 8/64 |
| R43-retention100-topk4 | 1 | 4 | 12.16 | 40.20 | 4.66 | 64 | 68 | 1050689536 | 0.35 | 4 | 16/64 |
| R57-attn-locality-w16-e4 | 1 | 4 | 12.35 | 47.98 | 4.68 | 64 | 68 | 1050689536 | 0.37 | 6 | 13/64 |
| R43-retention100-topk4 | 1 | 5 | 12.51 | 39.39 | 4.54 | 64 | 69 | 1050689536 | 0.29 | 8 | 16/64 |
| R57-attn-locality-w16-e4 | 1 | 5 | 12.45 | 51.88 | 4.68 | 64 | 69 | 1050689536 | 0.41 | 13 | 15/64 |

## Interpretation

R57 w16/e4 validates that the new edge attention locality path is active:
candidate rows report `attention_locality` usage, `max_topk=8`, and the
30 tok/s floor is preserved. However, this wider reuse window is rejected
because quality moved in the wrong direction on the same five-prompt matrix.
Average unique tokens fell from 14.80/64 to 12.80/64, and average repetition
rose from 0.28 to 0.38.

Decision: failed. The wider attention-locality cache adds too many stale
attention input features for this sparse path. Keep as negative evidence that
edge attention locality must be conservative.

## Raw Output

### R43-retention100-topk4 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth.swing dis mir.swing.swing.swing mir disilersorexelage mir.swing mir.swing.swing mir.swing disfine inset mir mir.swing disipers mir mir mir.swing dis.swing mir mir mir.swing mir mir mir.swing mir mir mir mir mir mir mir.swing mir.swing mir.swing mir

[TTFT/Prefill: 11.96s | Decode: 39.75 tok/s | E2E: 4.73 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/32 max_gap_milli=127 adaptive_throttles=5 min_margin_milli=18 phrase_novelty=6/64 max_ngram=4 gap_skips=13 max_gap_milli=202 retentions=23 | Repetition: ratio=0.25 max_run=7 unique=14/64]
> 
```

### R57-attn-locality-w16-e4 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem Doug mirouns Emin mir.swing mirolanyth mir diselage mir disrede mir.swing.swing.swing mir mir.swing disilers mir.swing mir mirfinefinefinefinefinefinefinefinefinefinerede mir mir mirrede.swing.swing mir mir mir mir.swing mir.swing mir mirervoervoervoervoervorede mir.swing

[TTFT/Prefill: 12.37s | Decode: 48.46 tok/s | E2E: 4.68 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77985404928 scratch=64 bytes input_tile_reads=29962 input_tile_bytes=258625536 attention_locality=494/126 max_selected=8 lm_head_repeat_margin=11/33 max_gap_milli=715 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=6/64 max_ngram=4 gap_skips=13 max_gap_milli=715 retentions=15 | Repetition: ratio=0.38 max_run=10 unique=15/64]
> 
```

### R43-retention100-topk4 run 1 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir.swing mir.swing.swing.swing mirythcash mir mir mir mir mir mir mir mirrede mir mir mir mir Horton.swing mir mir mir Hubbard mirior.swing pe.swing mir.swing dis.swing.swing.swing remrede.swing disolandclub.swingÃ¤ÃŁlaughter comæķ·oland.Bundlemute.swing mir.swing mir mir mir mir.swing mir.swing

[TTFT/Prefill: 12.22s | Decode: 36.93 tok/s | E2E: 4.60 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=10/25 max_gap_milli=135 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=10/64 max_ngram=4 gap_skips=5 max_gap_milli=205 retentions=17 | Repetition: ratio=0.30 max_run=8 unique=20/64]
> 
```

### R57-attn-locality-w16-e4 run 1 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir.swing miryth mirihncash mirabor.swing.swing.swing mir mir mir mir mircash.swing mir mir mir mir mir mir.swing mir mir mirrede dis carnhaulelage mir mirrede mirelage.swing mir mir mirrede.swing mir.swing.swing.swingÃ¤ÃŁ.swing.swing.swing.swing mirrede mir mir.swing.swing mir.swing mir

[TTFT/Prefill: 12.29s | Decode: 41.82 tok/s | E2E: 4.64 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77985411072 scratch=64 bytes input_tile_reads=29956 input_tile_bytes=258613248 attention_locality=492/126 max_selected=8 lm_head_repeat_margin=11/34 max_gap_milli=142 phrase_novelty=7/64 max_ngram=4 gap_skips=2 max_gap_milli=191 retentions=30 | Repetition: ratio=0.37 max_run=6 unique=13/64]
> 
```

### R43-retention100-topk4 run 1 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir dis/etcinnamon dis-INF mir.swing mir.swing.swing rem mir.swing mir.swing mir mir.swing mir mir mir mir mir.swing mir.swing mir mir.swing mir mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing mir.swing mir.swing mir mir.swing mir.swing mir.swing mir.swing mir.swing mir

[TTFT/Prefill: 12.14s | Decode: 60.08 tok/s | E2E: 4.85 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=25/37 max_gap_milli=92 adaptive_throttles=8 min_margin_milli=18 phrase_novelty=2/64 max_ngram=4 gap_skips=12 max_gap_milli=210 retentions=36 | Repetition: ratio=0.19 max_run=5 unique=8/64]
> 
```

### R57-attn-locality-w16-e4 run 1 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir dis rem mir.swing mir mir mir.swing.swing.swing mir.swing mir.swing mir mir mir mir mir.swing mir mir.swing mir.swing.swing mir.swing mir mir mir mir mir mir.swing mir.swing.swing mirlaughter.swing mir.swing.swing mir.swing disĻ.swing mir mir mir mir.swing mir.swing.swing mir.swing mir

[TTFT/Prefill: 12.60s | Decode: 51.16 tok/s | E2E: 4.63 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77985401856 scratch=64 bytes input_tile_reads=29965 input_tile_bytes=258631680 attention_locality=495/126 max_selected=8 lm_head_repeat_margin=26/47 max_gap_milli=171 adaptive_throttles=6 min_margin_milli=18 phrase_novelty=4/64 max_ngram=4 gap_skips=11 max_gap_milli=203 retentions=37 | Repetition: ratio=0.35 max_run=6 unique=8/64]
> 
```

### R43-retention100-topk4 run 1 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  osengx mir disngx mir mir mir mir.swing adm dis dis rem rem rem remRed mir mir mir.swing mir.swing.CopyTo mir.swing.swing%d Ledger Ledger Ledger Wright.swing constructive fÃŃs fÃŃs fÃŃs.swing mir.swing gu mir.swing mir mir mir.swing mir.swing.swing mir mir mir.swing mir.swing mir.swing mir mir mir mir

[TTFT/Prefill: 12.16s | Decode: 40.20 tok/s | E2E: 4.66 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=8/29 max_gap_milli=839 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=4/64 max_ngram=4 gap_skips=7 max_gap_milli=204 retentions=21 | Repetition: ratio=0.35 max_run=4 unique=16/64]
> 
```

### R57-attn-locality-w16-e4 run 1 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  ose Mir mir disinnamon rem mir mir mir mir.swing Fin conf mir.swing mir.swing.swing mir.swing.swing disstitutionsrede mir.swing mir.swing.swing mir mir mir mir mir.swing mir.swing mir mir.swing mir mir mir.swing mir mir mir mir.swing mir.swing.swing mir.swing mir mir.swingfinefinefinefinefinefine

[TTFT/Prefill: 12.35s | Decode: 47.98 tok/s | E2E: 4.68 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77985420288 scratch=64 bytes input_tile_reads=29947 input_tile_bytes=258594816 attention_locality=489/126 max_selected=8 lm_head_repeat_margin=21/44 max_gap_milli=425 adaptive_throttles=5 min_margin_milli=18 phrase_novelty=2/64 max_ngram=4 gap_skips=10 max_gap_milli=425 retentions=31 | Repetition: ratio=0.37 max_run=6 unique=13/64]
> 
```

### R43-retention100-topk4 run 1 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir disngx Ledger Ledger Ledger Ledger fÃŃs Mundcash Mund fÃŃs">% Mund">%boa Mund Mund Mund carn carncash mir.swing mir mir mir mir.swing mir.swing fÃŃs fÃŃs fÃŃs fÃŃs fÃŃs fÃŃs fÃŃs fÃŃselage.swingStrip mir.swingStriprede mir.swingboa carnhaulelage mir mir mir.swing dis carnrede.swingStrip fÃŃs

[TTFT/Prefill: 12.51s | Decode: 39.39 tok/s | E2E: 4.54 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=8/24 max_gap_milli=490 phrase_novelty=6/64 max_ngram=4 gap_skips=8 max_gap_milli=490 retentions=7 | Repetition: ratio=0.29 max_run=8 unique=16/64]
> 
```

### R57-attn-locality-w16-e4 run 1 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir dis mir mir mir.swing Mund Mund Mund fÃŃs Mund Mund Mund fÃŃs fÃŃs fÃŃs fÃŃs Mund fÃŃs fÃŃs fÃŃs fÃŃs fÃŃs fÃŃs fÃŃs fÃŃs fÃŃs fÃŃs fÃŃs fÃŃs fÃŃs Mundboa Mundelage.swing mir.swingstituterede mir mir mir.swing fÃŃsrede.swingStrip mir diffrede mir mir mir mir.swing fÃŃs carnhaulelage.swing fÃŃs

[TTFT/Prefill: 12.45s | Decode: 51.88 tok/s | E2E: 4.68 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=8 skipped_madds=77985408000 scratch=64 bytes input_tile_reads=29959 input_tile_bytes=258619392 attention_locality=493/126 max_selected=8 lm_head_repeat_margin=5/29 max_gap_milli=540 phrase_novelty=7/64 max_ngram=4 gap_skips=19 max_gap_milli=540 retentions=8 | Repetition: ratio=0.41 max_run=13 unique=15/64]
> 
```
