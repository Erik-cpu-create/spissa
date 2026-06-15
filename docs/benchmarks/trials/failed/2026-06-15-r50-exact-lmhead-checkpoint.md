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
| R43-retention100-topk4 | 5 | true | false | 30.19 | 46.19 | 35.84 | 17.60 | 0.18 |
| R50-exact-lmhead-every8 | 5 | false | false | 13.46 | 16.08 | 14.74 | 16.00 | 0.27 |

## Interpretation

R50 rejects periodic exact LM-head checkpoints as the chat-ready correction
path. The probe did switch to the exact LM-head result on most checkpointed
tokens, which proves the instrumentation works, but every candidate prompt fell
well below the 30 tok/s floor. Average decode dropped from 35.84 tok/s to
14.74 tok/s, while average unique tokens also fell from 17.60/64 to 16.00/64
and repetition rose from 0.18 to 0.27.

Decision: failed. Full-vocabulary exact LM-head checks are too expensive for the
target path and do not fix semantic collapse enough to justify the cost. Keep
the knob only as a diagnostic probe, not as a speed preset.

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R43-retention100-topk4 | 1 | 1 | 12.77 | 38.48 | 4.44 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R50-exact-lmhead-every8 | 1 | 1 | 13.15 | 16.08 | 3.75 | 64 | 66 | 1050689536 | 0.27 | 3 | 12/64 |
| R43-retention100-topk4 | 1 | 2 | 13.38 | 46.19 | 4.34 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R50-exact-lmhead-every8 | 1 | 2 | 13.37 | 15.31 | 3.66 | 64 | 68 | 1050689536 | 0.30 | 3 | 8/64 |
| R43-retention100-topk4 | 1 | 3 | 13.71 | 31.48 | 4.07 | 64 | 70 | 1050689536 | 0.16 | 2 | 22/64 |
| R50-exact-lmhead-every8 | 1 | 3 | 14.39 | 14.54 | 3.42 | 64 | 70 | 1050689536 | 0.25 | 2 | 18/64 |
| R43-retention100-topk4 | 1 | 4 | 13.63 | 32.87 | 4.12 | 64 | 69 | 1050689536 | 0.16 | 2 | 20/64 |
| R50-exact-lmhead-every8 | 1 | 4 | 13.61 | 14.33 | 3.55 | 64 | 69 | 1050689536 | 0.22 | 3 | 20/64 |
| R43-retention100-topk4 | 1 | 5 | 14.16 | 30.19 | 3.94 | 64 | 70 | 1050689536 | 0.22 | 2 | 21/64 |
| R50-exact-lmhead-every8 | 1 | 5 | 13.57 | 13.46 | 3.51 | 64 | 70 | 1050689536 | 0.30 | 3 | 22/64 |

## Raw Output

### R43-retention100-topk4 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 12.77s | Decode: 38.48 tok/s | E2E: 4.44 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R50-exact-lmhead-every8 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir mir Doug mir.swing disrede mir.swing mir diselage mir mir.swing mir.swing.swing disfinefinerede mir.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing mir mir.swing mir.swing.swing.swingfinefineipers mir.swing mir mir mir.swing mir.swing mir mir.swing mir mir

[TTFT/Prefill: 13.15s | Decode: 16.08 tok/s | E2E: 3.75 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_exact=6/8 lm_head_repeat_margin=12/18 max_gap_milli=237 phrase_novelty=8/64 max_ngram=2 gap_skips=12 max_gap_milli=202 retentions=3 | Repetition: ratio=0.27 max_run=3 unique=12/64]
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

[TTFT/Prefill: 13.38s | Decode: 46.19 tok/s | E2E: 4.34 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R50-exact-lmhead-every8 run 1 prompt 2

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -innamon mir mir dis/etcinnamoninnamon mir.swing mir.swing.swing mir.swing.swing mir.swing mir.swing mir mir dis disĻascalascal.swing mir mir.swing.swing mir mir.swing mir.swing mir mir mir.swing mir.swing mir mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 13.37s | Decode: 15.31 tok/s | E2E: 3.66 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_exact=7/8 lm_head_repeat_margin=18/25 max_gap_milli=462 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=6/64 max_ngram=2 gap_skips=23 max_gap_milli=204 retentions=6 | Repetition: ratio=0.30 max_run=3 unique=8/64]
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

[TTFT/Prefill: 13.71s | Decode: 31.48 tok/s | E2E: 4.07 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=9/17 max_gap_milli=338 phrase_novelty=3/64 max_ngram=2 gap_skips=6 max_gap_milli=226 | Repetition: ratio=0.16 max_run=2 unique=22/64]
>
```

### R50-exact-lmhead-every8 run 1 prompt 3

Prompt: `explain artificial intelligence in one sentence`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  ngxinnamoninnamon quotationsinnamonngxngxinnamoninnamon dis reminnamoninnamon rem remigham mir mir.swing diff Mund.swing.swing%d%d dá»ĭch dá»ĭchirsch mir.swing.swing wind.swing mir.swing fÃŃs.io.swing.swing mir.swingelage.swing mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing mir.swing mir mir.swing mir mir

[TTFT/Prefill: 14.39s | Decode: 14.54 tok/s | E2E: 3.42 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_exact=6/8 lm_head_repeat_margin=13/20 max_gap_milli=763 phrase_novelty=3/64 max_ngram=2 gap_skips=11 max_gap_milli=216 retentions=4 | Repetition: ratio=0.25 max_run=2 unique=18/64]
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

[TTFT/Prefill: 13.63s | Decode: 32.87 tok/s | E2E: 4.12 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=6/16 max_gap_milli=480 phrase_novelty=1/64 max_ngram=2 gap_skips=2 max_gap_milli=202 retentions=2 | Repetition: ratio=0.16 max_run=2 unique=20/64]
>
```

### R50-exact-lmhead-every8 run 1 prompt 4

Prompt: `write a short friendly reply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Harmose Mund mir.swing mir mir mir fÃŃs fÃŃs Mund mir fÃŃs fÃŃs Mund Mund">% Mund fÃŃs fÃŃs Hubbard mir mir mir.swing fÃŃs Mundboaboa mÅ©iclubclub carnrede mir diffrede.swing mir mir Mund carnhaulelage.swing hoop rem remegendrede mir.swingFinder fÃŃselageelage.swing fÃŃsclub carnhaulelage.swing.swing

[TTFT/Prefill: 13.61s | Decode: 14.33 tok/s | E2E: 3.55 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_exact=8/8 lm_head_repeat_margin=5/11 max_gap_milli=443 phrase_novelty=1/64 max_ngram=2 | Repetition: ratio=0.22 max_run=3 unique=20/64]
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

[TTFT/Prefill: 14.16s | Decode: 30.19 tok/s | E2E: 3.94 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=13/27 max_gap_milli=929 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=1/64 max_ngram=2 gap_skips=12 max_gap_milli=201 retentions=3 | Repetition: ratio=0.22 max_run=2 unique=21/64]
>
```

### R50-exact-lmhead-every8 run 1 prompt 5

Prompt: `what is two plus two?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Fininnamoninnamon quotations Sheainnamoninnamon quotationsinnamoninnamon quotations quotations lever mir mir.swingemensemens binæľºæľº dis dis mir.swing mir mir.swing.swing mir mir.swing mir.swing.swing cadena carnhaulhaulose mir.swing fÃŃshaul carnhaulhaulosteroneosterone Canterrede.swing mir mir mir.swing mir mir.swing gu disaconsacons

[TTFT/Prefill: 13.57s | Decode: 13.46 tok/s | E2E: 3.51 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_exact=6/8 lm_head_repeat_margin=10/21 max_gap_milli=929 phrase_novelty=2/64 max_ngram=2 gap_skips=6 max_gap_milli=452 retentions=1 | Repetition: ratio=0.30 max_run=3 unique=22/64]
>
```
