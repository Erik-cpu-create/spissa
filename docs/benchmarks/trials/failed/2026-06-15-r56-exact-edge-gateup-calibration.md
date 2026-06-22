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
| R43-retention100-topk4 | 5 | false | false | 26.14 | 43.42 | 33.08 | 17.60 | 0.18 |
| R56-exact-edge-gateup1 | 5 | false | false | 13.49 | 15.05 | 14.32 | 27.60 | 0.03 |

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R43-retention100-topk4 | 1 | 1 | 11.71 | 39.87 | 4.81 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R56-exact-edge-gateup1 | 1 | 1 | 13.11 | 13.49 | 3.60 | 64 | 66 | 1050689536 | 0.00 | 1 | 28/64 |
| R43-retention100-topk4 | 1 | 2 | 13.42 | 43.42 | 4.30 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R56-exact-edge-gateup1 | 1 | 2 | 12.74 | 15.05 | 3.78 | 64 | 68 | 1050689536 | 0.05 | 2 | 21/64 |
| R43-retention100-topk4 | 1 | 3 | 13.35 | 26.14 | 4.06 | 64 | 70 | 1050689536 | 0.16 | 2 | 22/64 |
| R56-exact-edge-gateup1 | 1 | 3 | 12.92 | 14.06 | 3.68 | 64 | 70 | 1050689536 | 0.05 | 2 | 28/64 |
| R43-retention100-topk4 | 1 | 4 | 13.45 | 29.54 | 4.11 | 64 | 69 | 1050689536 | 0.16 | 2 | 20/64 |
| R56-exact-edge-gateup1 | 1 | 4 | 13.09 | 14.09 | 3.64 | 64 | 69 | 1050689536 | 0.05 | 2 | 30/64 |
| R43-retention100-topk4 | 1 | 5 | 12.81 | 26.43 | 4.21 | 64 | 70 | 1050689536 | 0.22 | 2 | 21/64 |
| R56-exact-edge-gateup1 | 1 | 5 | 12.80 | 14.90 | 3.76 | 64 | 70 | 1050689536 | 0.02 | 2 | 31/64 |

## Raw Output

### R43-retention100-topk4 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 11.71s | Decode: 39.87 tok/s | E2E: 4.81 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R56-exact-edge-gateup1 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  sign miryth island mir.swingyth mir.swingpackage mir.swing mir.swinggether mir.swingervo clubs mir.swing gezurrets mir.swingÐºÐ¾Ð²Ð¾Ð´ mir.swingenalApproved mir.swing ground formieseenasocol mir.swingarticles mir.swing vern mir.swingatomyecess mir.swinglaughter.swingÃ¤ÃŁ_Lean mir.swingstringLiteral.swing_Lean mir.swingstringLiteral.swing_Lean

[TTFT/Prefill: 13.11s | Decode: 13.49 tok/s | E2E: 3.60 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5986 fallbacks=30 max_topk=4 skipped_madds=73767321600 scratch=32 bytes input_tile_reads=27472 input_tile_bytes=239075328 lm_head_repeat_margin=15/15 max_gap_milli=23 adaptive_throttles=12 min_margin_milli=18 phrase_novelty=1/64 max_ngram=2 gap_skips=7 max_gap_milli=191 | Repetition: ratio=0.00 max_run=1 unique=28/64]
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

[TTFT/Prefill: 13.42s | Decode: 43.42 tok/s | E2E: 4.30 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R56-exact-edge-gateup1 run 1 prompt 2

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -.swing vern mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing represented mir.swingitol mir.swing189VIDEO mir.swing somewhereè¯¢ okrophe mir.swing di mir.swing amountoisesselomoragus mir.swing di mir.swing amountchet mir.swing di mir.swing amountoisavesæĬķ mir.swing

[TTFT/Prefill: 12.74s | Decode: 15.05 tok/s | E2E: 3.78 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5986 fallbacks=30 max_topk=4 skipped_madds=73767321600 scratch=32 bytes input_tile_reads=27472 input_tile_bytes=239075328 lm_head_repeat_margin=18/18 max_gap_milli=13 adaptive_throttles=9 min_margin_milli=18 phrase_novelty=6/64 max_ngram=2 gap_skips=15 max_gap_milli=205 retentions=3 | Repetition: ratio=0.05 max_run=2 unique=21/64]
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

[TTFT/Prefill: 13.35s | Decode: 26.14 tok/s | E2E: 4.06 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=9/17 max_gap_milli=338 phrase_novelty=3/64 max_ngram=2 gap_skips=6 max_gap_milli=226 | Repetition: ratio=0.16 max_run=2 unique=22/64]
>
```

### R56-exact-edge-gateup1 run 1 prompt 3

Prompt: `explain artificial intelligence in one sentence`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   mir mir.swing mir.swing mir.swingounsassellaughter laughter pity mir.swing táº¥n mir.swing mir.swing.swing mir.swing mir.swingariat mir.swing.swingkin mir.swingolandnell vern mir.swingonym Ð´ÑĥÐ¼ÐºÑĥ vern mir.swing moreoverpackage mir.swingarticles graduationsign mir.swingÐºÐ¾Ð²Ð¾Ð´ mir.swing RencontreTagsheten mir.swingotopeeterogenicocr

[TTFT/Prefill: 12.92s | Decode: 14.06 tok/s | E2E: 3.68 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5986 fallbacks=30 max_topk=4 skipped_madds=73767321600 scratch=32 bytes input_tile_reads=27472 input_tile_bytes=239075328 lm_head_repeat_margin=16/18 max_gap_milli=118 adaptive_throttles=8 min_margin_milli=18 phrase_novelty=2/64 max_ngram=2 gap_skips=10 max_gap_milli=198 retentions=1 | Repetition: ratio=0.05 max_run=2 unique=28/64]
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

[TTFT/Prefill: 13.45s | Decode: 29.54 tok/s | E2E: 4.11 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=6/16 max_gap_milli=480 phrase_novelty=1/64 max_ngram=2 gap_skips=2 max_gap_milli=202 retentions=2 | Repetition: ratio=0.16 max_run=2 unique=20/64]
>
```

### R56-exact-edge-gateup1 run 1 prompt 4

Prompt: `write a short friendly reply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Harmngx mir Douglas Winn mir.swing mir.swing.swing mir.swing mir.swing.swing nativeowitz mir.swing.swing membranes mir.swing dimensions mir.swingolanaysisinvenge mir.swing dimensions mir.swingixonÃ¤ÃŁouns inset mir.swingotherwise mir.swingteesØ¬Ø¯ionsUEDewis mir.swing)[-dnaç»´ mir.swingÐºÐ¾Ð²Ð¾Ð´ mir.swingbove mir.swingteesursor

[TTFT/Prefill: 13.09s | Decode: 14.09 tok/s | E2E: 3.64 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5986 fallbacks=30 max_topk=4 skipped_madds=73767321600 scratch=32 bytes input_tile_reads=27472 input_tile_bytes=239075328 lm_head_repeat_margin=15/17 max_gap_milli=235 adaptive_throttles=7 min_margin_milli=18 phrase_novelty=1/64 max_ngram=2 gap_skips=9 max_gap_milli=202 retentions=1 | Repetition: ratio=0.05 max_run=2 unique=30/64]
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

[TTFT/Prefill: 12.81s | Decode: 26.43 tok/s | E2E: 4.21 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=13/27 max_gap_milli=929 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=1/64 max_ngram=2 gap_skips=12 max_gap_milli=201 retentions=3 | Repetition: ratio=0.22 max_run=2 unique=21/64]
>
```

### R56-exact-edge-gateup1 run 1 prompt 5

Prompt: `what is two plus two?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  sign mir dri mir dimensions mir.swingpackage mir.swinglaughterildersign mir.swingalm mir.swing medically opportunitylite mir.swingomm rejo mir.swingtram Wid mir.swinghaar mir.swing láº¡c mir.swingotopefstlaughteralezaÃ¹ngmostatounsstrap mir.swingcroftsign mir.swingèŃľ mir.swingÐºÐ¾Ð²Ð¾Ð´ mir.swingpedo mir mir.swing favor mir

[TTFT/Prefill: 12.80s | Decode: 14.90 tok/s | E2E: 3.76 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5986 fallbacks=30 max_topk=4 skipped_madds=73767321600 scratch=32 bytes input_tile_reads=27472 input_tile_bytes=239075328 lm_head_repeat_margin=13/14 max_gap_milli=45 adaptive_throttles=11 min_margin_milli=18 phrase_novelty=0/64 max_ngram=2 gap_skips=6 max_gap_milli=188 | Repetition: ratio=0.02 max_run=2 unique=31/64]
>
```

## Interpretation

R56 rejects exact edge-layer `mlp-gate-up` calibration as a speed preset. It
reduced repetition strongly from 0.18 to 0.03 and raised average unique tokens
to 27.60/64, but decode collapsed to 13.49-15.05 tok/s.

Decision: failed. Exacting gate/up on edge layers is too expensive for the
target and less quality-positive than exact edge attention, so it should not be
the next optimization surface.
