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
| R43-retention100-topk4 | 5 | false | false | 29.81 | 49.76 | 38.58 | 17.60 | 0.18 |
| R55-exact-edge1 | 5 | false | false | 10.87 | 13.05 | 12.31 | 40.40 | 0.03 |

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R43-retention100-topk4 | 1 | 1 | 12.31 | 41.31 | 4.63 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R55-exact-edge1 | 1 | 1 | 12.32 | 12.94 | 3.72 | 64 | 66 | 1050689536 | 0.00 | 1 | 40/64 |
| R43-retention100-topk4 | 1 | 2 | 12.02 | 49.76 | 4.82 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R55-exact-edge1 | 1 | 2 | 12.86 | 13.05 | 3.62 | 64 | 68 | 1050689536 | 0.13 | 2 | 23/64 |
| R43-retention100-topk4 | 1 | 3 | 12.60 | 32.98 | 4.41 | 64 | 70 | 1050689536 | 0.16 | 2 | 22/64 |
| R55-exact-edge1 | 1 | 3 | 12.81 | 11.82 | 3.53 | 64 | 70 | 1050689536 | 0.02 | 2 | 43/64 |
| R43-retention100-topk4 | 1 | 4 | 12.34 | 39.06 | 4.59 | 64 | 69 | 1050689536 | 0.16 | 2 | 20/64 |
| R55-exact-edge1 | 1 | 4 | 12.72 | 12.85 | 3.63 | 64 | 69 | 1050689536 | 0.02 | 2 | 44/64 |
| R43-retention100-topk4 | 1 | 5 | 12.85 | 29.81 | 4.28 | 64 | 70 | 1050689536 | 0.22 | 2 | 21/64 |
| R55-exact-edge1 | 1 | 5 | 13.09 | 10.87 | 3.39 | 64 | 70 | 1050689536 | 0.00 | 1 | 52/64 |

## Raw Output

### R43-retention100-topk4 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 12.31s | Decode: 41.31 tok/s | E2E: 4.63 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R55-exact-edge1 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   ()č
 Infantitzer mirmuteÃ¤ÃŁThin mirpez mir sob vern mirlsa crossedimelaughter-CN mir dressingà¸¸à¸ļstellrousbing vern hypoth vern mir.swingstellà¸ĵ gez.swing vic Bentheel sob mir dressingarea.swingofficeignet vicIDI mirujièĦovenairs coherence mir sob vern mireldon mireldon vernÃ¤ÃŁfÃ¶rrawn mir

[TTFT/Prefill: 12.32s | Decode: 12.94 tok/s | E2E: 3.72 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5356 fallbacks=28 max_topk=4 skipped_madds=70335799296 scratch=32 bytes input_tile_reads=24952 input_tile_bytes=231849984 lm_head_repeat_margin=2/2 max_gap_milli=31 phrase_novelty=2/64 max_ngram=2 gap_skips=1 max_gap_milli=163 | Repetition: ratio=0.00 max_run=1 unique=40/64]
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

[TTFT/Prefill: 12.02s | Decode: 49.76 tok/s | E2E: 4.82 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R55-exact-edge1 run 1 prompt 2

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -IGNungen unpack specialised.swingpgsql.swingmobx tslint.swingcharg mirigin mir.swingcharg mir.swingÅ¯st colourful.swingcharg mir mir.swing mir mirlaughter mir mir.swing mir mirlaughterrawn.swingually mirlaughter mir.swing.swingetti mir mir.swing mirlaughter mir.swing.swingorexsign mir mir.swing mir.swingÑıÐ¶.swingREP sob mir

[TTFT/Prefill: 12.86s | Decode: 13.05 tok/s | E2E: 3.62 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5356 fallbacks=28 max_topk=4 skipped_madds=70335799296 scratch=32 bytes input_tile_reads=24952 input_tile_bytes=231849984 lm_head_repeat_margin=3/11 max_gap_milli=329 phrase_novelty=4/64 max_ngram=2 gap_skips=5 max_gap_milli=251 | Repetition: ratio=0.13 max_run=2 unique=23/64]
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

[TTFT/Prefill: 12.60s | Decode: 32.98 tok/s | E2E: 4.41 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=9/17 max_gap_milli=338 phrase_novelty=3/64 max_ngram=2 gap_skips=6 max_gap_milli=226 | Repetition: ratio=0.16 max_run=2 unique=22/64]
>
```

### R55-exact-edge1 run 1 prompt 3

Prompt: `explain artificial intelligence in one sentence`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Resisperhton globnipIGNresses loèĦ miristrteryiple Pantressesãĥ³ãĤ° impactinguredergoystarmeikaÃ¤ngonet/Foundation/change vernonetuliaetedtownureysteted Named eronetuliaiveryusetulia vernnota impactingpiessett raisieszulia miryst er miretonsettetedetediesz vic impacting mirev

[TTFT/Prefill: 12.81s | Decode: 11.82 tok/s | E2E: 3.53 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5356 fallbacks=28 max_topk=4 skipped_madds=70335799296 scratch=32 bytes input_tile_reads=24952 input_tile_bytes=231849984 lm_head_repeat_margin=1/2 max_gap_milli=149 phrase_novelty=0/64 max_ngram=0 | Repetition: ratio=0.02 max_run=2 unique=43/64]
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

[TTFT/Prefill: 12.34s | Decode: 39.06 tok/s | E2E: 4.59 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=6/16 max_gap_milli=480 phrase_novelty=1/64 max_ngram=2 gap_skips=2 max_gap_milli=202 retentions=2 | Repetition: ratio=0.16 max_run=2 unique=20/64]
>
```

### R55-exact-edge1 run 1 prompt 4

Prompt: `write a short friendly reply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  HarmubuenuityIELDolid mirangakerutationsrawnaveltutionsleradeournundersiltaventer mirestrure mirwing vern speedingegin mir/Foundationæķ·ÅĤasign heartelt mir Occ ancestors dressingsignity favourika mirovenaight mir analy1 mirovenrawn mir mir.swing dib mir.swing dimensions miræķ· solution mir.swing

[TTFT/Prefill: 12.72s | Decode: 12.85 tok/s | E2E: 3.63 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5356 fallbacks=28 max_topk=4 skipped_madds=70335799296 scratch=32 bytes input_tile_reads=24952 input_tile_bytes=231849984 lm_head_repeat_margin=2/3 max_gap_milli=100 phrase_novelty=0/64 max_ngram=2 gap_skips=1 max_gap_milli=171 | Repetition: ratio=0.02 max_run=2 unique=44/64]
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

[TTFT/Prefill: 12.85s | Decode: 29.81 tok/s | E2E: 4.28 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=13/27 max_gap_milli=929 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=1/64 max_ngram=2 gap_skips=12 max_gap_milli=201 retentions=3 | Repetition: ratio=0.22 max_run=2 unique=21/64]
>
```

### R55-exact-edge1 run 1 prompt 5

Prompt: `what is two plus two?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  ResrushugalFormattediciarpondesignature Grosåĥıæĺ¯apultazzrencetedovicipss Mathematical foreolorirsureackolorign miruneuseearly Namedindent stiruse kill prof impactingusingagain mireth vernolorrigÃ¢yymbÃ¹ngeted Sayseteduy quoting permanentlyonet mirypy stir impactingoniceted parad stirlook medicallyolor

[TTFT/Prefill: 13.09s | Decode: 10.87 tok/s | E2E: 3.39 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5356 fallbacks=28 max_topk=4 skipped_madds=70335799296 scratch=32 bytes input_tile_reads=24952 input_tile_bytes=231849984 phrase_novelty=0/64 max_ngram=0 | Repetition: ratio=0.00 max_run=1 unique=52/64]
>
```

## Interpretation

R55 rejects exact edge-layer calibration as a speed preset. Forcing the first
and last transformer layers to exact projection improved diversity and reduced
repetition sharply: average unique tokens rose from 17.60/64 to 40.40/64 and
average repetition fell from 0.18 to 0.03. Decode speed collapsed to
10.87-13.05 tok/s, far below the 30 tok/s floor.

Decision: failed as a preset, but useful as diagnostic evidence. The quality
improvement says the remaining semantic issue is likely in transformer
hidden-state approximation, not final LM-head selection alone. The next
direction should approximate this signal more cheaply through projection or
layer micro-calibration instead of full exact edge layers.
