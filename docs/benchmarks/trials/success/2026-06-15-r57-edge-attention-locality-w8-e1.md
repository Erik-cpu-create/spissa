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
| R43-retention100-topk4 | 5 | true | false | 32.20 | 48.22 | 39.94 | 14.80 | 0.28 |
| R57-attn-locality-w8-e1 | 5 | true | false | 32.29 | 62.47 | 50.43 | 15.40 | 0.26 |

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R43-retention100-topk4 | 1 | 1 | 11.67 | 43.01 | 4.87 | 64 | 66 | 1050689536 | 0.25 | 7 | 14/64 |
| R57-attn-locality-w8-e1 | 1 | 1 | 12.72 | 62.47 | 4.66 | 64 | 66 | 1050689536 | 0.19 | 4 | 13/64 |
| R43-retention100-topk4 | 1 | 2 | 11.95 | 32.44 | 4.61 | 64 | 66 | 1050689536 | 0.30 | 8 | 20/64 |
| R57-attn-locality-w8-e1 | 1 | 2 | 12.52 | 58.33 | 4.71 | 64 | 66 | 1050689536 | 0.29 | 8 | 18/64 |
| R43-retention100-topk4 | 1 | 3 | 12.74 | 48.22 | 4.56 | 64 | 68 | 1050689536 | 0.19 | 5 | 8/64 |
| R57-attn-locality-w8-e1 | 1 | 3 | 13.19 | 42.82 | 4.36 | 64 | 68 | 1050689536 | 0.37 | 6 | 10/64 |
| R43-retention100-topk4 | 1 | 4 | 12.29 | 43.83 | 4.66 | 64 | 68 | 1050689536 | 0.35 | 4 | 16/64 |
| R57-attn-locality-w8-e1 | 1 | 4 | 13.30 | 32.29 | 4.20 | 64 | 68 | 1050689536 | 0.25 | 5 | 18/64 |
| R43-retention100-topk4 | 1 | 5 | 13.22 | 32.20 | 4.22 | 64 | 69 | 1050689536 | 0.29 | 8 | 16/64 |
| R57-attn-locality-w8-e1 | 1 | 5 | 12.49 | 56.25 | 4.70 | 64 | 69 | 1050689536 | 0.22 | 5 | 18/64 |

## Interpretation

R57 w8/e1 is accepted as a conservative experimental-speed improvement with
quality limitation. It keeps the 30 tok/s floor across the five-prompt matrix,
raises average decode throughput from 39.94 to 50.43 tok/s, and improves the
cheap quality counters relative to the alternating control: average unique
tokens moved from 14.80/64 to 15.40/64, and average repetition fell from 0.28
to 0.26.

The strict 30-40 tok/s band remains false because several runs exceed 40 tok/s.
Semantic output is still fragmentary, so this is not chat-ready. Paper value:
positive evidence that a tiny per-layer recent-index attention cache can add a
small quality/stability gain without increasing model size or resident memory.

Decision: success with quality limitation. Use `RLLM_AIP_ATTENTION_LOCALITY_WINDOW=8`
and `RLLM_AIP_ATTENTION_LOCALITY_EXTRA=1` as the retained R57 preset.

## Raw Output

### R43-retention100-topk4 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth.swing dis mir.swing.swing.swing mir disilersorexelage mir.swing mir.swing.swing mir.swing disfine inset mir mir.swing disipers mir mir mir.swing dis.swing mir mir mir.swing mir mir mir.swing mir mir mir mir mir mir mir.swing mir.swing mir.swing mir

[TTFT/Prefill: 11.67s | Decode: 43.01 tok/s | E2E: 4.87 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/32 max_gap_milli=127 adaptive_throttles=5 min_margin_milli=18 phrase_novelty=6/64 max_ngram=4 gap_skips=13 max_gap_milli=202 retentions=23 | Repetition: ratio=0.25 max_run=7 unique=14/64]
> 
```

### R57-attn-locality-w8-e1 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth.swing.swing dis mir.swing mir disrede mir.swing.swing mir.swing mir.swing mir mir.swing remadj.swing mir.swing disfinerede dis.swing mir.swing mir mir.swing mir.swing.swing mir.swingfine carnelage mir.swing.swing mir mir mir mir.swing mir mir mir mir

[TTFT/Prefill: 12.72s | Decode: 62.47 tok/s | E2E: 4.66 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=5 skipped_madds=77986541568 scratch=40 bytes input_tile_reads=28852 input_tile_bytes=256352256 attention_locality=124/126 max_selected=5 lm_head_repeat_margin=17/27 max_gap_milli=86 adaptive_throttles=6 min_margin_milli=18 phrase_novelty=6/64 max_ngram=4 gap_skips=9 max_gap_milli=216 retentions=24 | Repetition: ratio=0.19 max_run=4 unique=13/64]
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

[TTFT/Prefill: 11.95s | Decode: 32.44 tok/s | E2E: 4.61 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=10/25 max_gap_milli=135 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=10/64 max_ngram=4 gap_skips=5 max_gap_milli=205 retentions=17 | Repetition: ratio=0.30 max_run=8 unique=20/64]
> 
```

### R57-attn-locality-w8-e1 run 1 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Hy mir.swing mir.swing.swing.swing mirythcash mir mir mir mir mir mir mir mirrede mir mir mir mir Horton.swing mir mir mir '// dis.swing mirolanrede disÃ¤ÃŁ.swing mal dis.swing.swing remÃ¢ycabolandrede mir.swing mir mir.swing mir.swing mir.swing mir mir.swing Mundelage.swing mir mir.swing

[TTFT/Prefill: 12.52s | Decode: 58.33 tok/s | E2E: 4.71 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=5 skipped_madds=77986541568 scratch=40 bytes input_tile_reads=28852 input_tile_bytes=256352256 attention_locality=124/126 max_selected=5 lm_head_repeat_margin=13/28 max_gap_milli=138 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=5/64 max_ngram=4 gap_skips=4 max_gap_milli=205 retentions=24 | Repetition: ratio=0.29 max_run=8 unique=18/64]
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

[TTFT/Prefill: 12.74s | Decode: 48.22 tok/s | E2E: 4.56 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=25/37 max_gap_milli=92 adaptive_throttles=8 min_margin_milli=18 phrase_novelty=2/64 max_ngram=4 gap_skips=12 max_gap_milli=210 retentions=36 | Repetition: ratio=0.19 max_run=5 unique=8/64]
> 
```

### R57-attn-locality-w8-e1 run 1 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir mir.swing mir mir mir mir mir.swing.swing.swing mir.swing mir.swing mir mir.swing mir.swing mir mir mir.swing mir.swing mir mir mir.swing mir.swing mir mir.swing.swing.swing mir.swing.swing mir.swing mir mir.swing mir.swing mir.swingÃ¤ÃŁlaughter comÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ibandÐĲÑĢÑħÑĸÐ²Ð¾Ð²Ð°Ð½Ð¾ascal

[TTFT/Prefill: 13.19s | Decode: 42.82 tok/s | E2E: 4.36 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=5 skipped_madds=77986541568 scratch=40 bytes input_tile_reads=28852 input_tile_bytes=256352256 attention_locality=124/126 max_selected=5 lm_head_repeat_margin=21/42 max_gap_milli=125 adaptive_throttles=5 min_margin_milli=18 phrase_novelty=6/64 max_ngram=4 gap_skips=7 max_gap_milli=202 retentions=37 | Repetition: ratio=0.37 max_run=6 unique=10/64]
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

[TTFT/Prefill: 12.29s | Decode: 43.83 tok/s | E2E: 4.66 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=8/29 max_gap_milli=839 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=4/64 max_ngram=4 gap_skips=7 max_gap_milli=204 retentions=21 | Repetition: ratio=0.35 max_run=4 unique=16/64]
> 
```

### R57-attn-locality-w8-e1 run 1 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  ose windngx mir mir disngx dis conf mir.swing Fin Fin rem rem remé mir mir mir mir mir.swing mir.swing mir.swing%d Ledgerfine Ledgerèĵelage rem vern mir.swing.swing mir.swing mal dis.swing mir.swing.swing mir mir mir.swing mir.swing mir mir.swing mir.swing mir mir mir mir.swing mir

[TTFT/Prefill: 13.30s | Decode: 32.29 tok/s | E2E: 4.20 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=5 skipped_madds=77986541568 scratch=40 bytes input_tile_reads=28852 input_tile_bytes=256352256 attention_locality=124/126 max_selected=5 lm_head_repeat_margin=12/27 max_gap_milli=468 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=3/64 max_ngram=4 gap_skips=8 max_gap_milli=203 retentions=21 | Repetition: ratio=0.25 max_run=5 unique=18/64]
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

[TTFT/Prefill: 13.22s | Decode: 32.20 tok/s | E2E: 4.22 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=8/24 max_gap_milli=490 phrase_novelty=6/64 max_ngram=4 gap_skips=8 max_gap_milli=490 retentions=7 | Repetition: ratio=0.29 max_run=8 unique=16/64]
> 
```

### R57-attn-locality-w8-e1 run 1 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Ledger mir dis Ledger Ledger Ledger Ledger Ledger Mund">% fÃŃs">% mir Mundrede mir fÃŃs fÃŃs fÃŃs fÃŃs fÃŃs Ledger">%rede disegend fÃŃselage fÃŃs fÃŃsclub carn carnhaul Ledgerclubhaul fÃŃs Mund.swing mir fÃŃsStripStripboaboaclub mir.swing disegendrede mir mir mir.swing mir.swing fÃŃsjamin mir.swingstitute

[TTFT/Prefill: 12.49s | Decode: 56.25 tok/s | E2E: 4.70 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=5 skipped_madds=77986541568 scratch=40 bytes input_tile_reads=28852 input_tile_bytes=256352256 attention_locality=124/126 max_selected=5 lm_head_repeat_margin=8/22 max_gap_milli=482 phrase_novelty=4/64 max_ngram=2 gap_skips=7 max_gap_milli=482 retentions=2 | Repetition: ratio=0.22 max_run=5 unique=18/64]
> 
```
