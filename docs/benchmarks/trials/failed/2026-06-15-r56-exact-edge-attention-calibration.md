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
| R43-retention100-topk4 | 5 | false | false | 25.41 | 43.85 | 33.44 | 17.60 | 0.18 |
| R56-exact-edge-attention1 | 5 | false | false | 18.19 | 23.51 | 21.45 | 35.40 | 0.10 |

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R43-retention100-topk4 | 1 | 1 | 11.75 | 40.33 | 4.81 | 64 | 66 | 1050689536 | 0.13 | 2 | 15/64 |
| R56-exact-edge-attention1 | 1 | 1 | 12.67 | 23.00 | 4.15 | 64 | 66 | 1050689536 | 0.11 | 2 | 34/64 |
| R43-retention100-topk4 | 1 | 2 | 12.78 | 43.85 | 4.50 | 64 | 68 | 1050689536 | 0.22 | 2 | 10/64 |
| R56-exact-edge-attention1 | 1 | 2 | 15.60 | 18.19 | 3.36 | 64 | 68 | 1050689536 | 0.06 | 2 | 41/64 |
| R43-retention100-topk4 | 1 | 3 | 13.20 | 29.06 | 4.16 | 64 | 70 | 1050689536 | 0.16 | 2 | 22/64 |
| R56-exact-edge-attention1 | 1 | 3 | 12.90 | 23.51 | 4.11 | 64 | 70 | 1050689536 | 0.13 | 2 | 34/64 |
| R43-retention100-topk4 | 1 | 4 | 15.69 | 28.53 | 3.58 | 64 | 69 | 1050689536 | 0.16 | 2 | 20/64 |
| R56-exact-edge-attention1 | 1 | 4 | 13.04 | 21.10 | 3.99 | 64 | 69 | 1050689536 | 0.13 | 2 | 31/64 |
| R43-retention100-topk4 | 1 | 5 | 13.62 | 25.41 | 3.97 | 64 | 70 | 1050689536 | 0.22 | 2 | 21/64 |
| R56-exact-edge-attention1 | 1 | 5 | 13.33 | 21.45 | 3.93 | 64 | 70 | 1050689536 | 0.06 | 2 | 37/64 |

## Raw Output

### R43-retention100-topk4 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   rem miryth mirihn mir.swing mirolanyth mir.swing dis mir.swing mir mir disilersorexelage mir.swing mir.swing mir mir.swing disfinerede disipers mir.swing mir mir.swing.swing mir.swing mir mir.swing mir.swingfinerede disolandelage.swing mir mir.swing mir.swing mir mir.swing mir.swing.swing

[TTFT/Prefill: 11.75s | Decode: 40.33 tok/s | E2E: 4.81 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=16/22 max_gap_milli=84 adaptive_throttles=4 min_margin_milli=18 phrase_novelty=5/64 max_ngram=2 gap_skips=15 max_gap_milli=201 retentions=1 | Repetition: ratio=0.13 max_run=2 unique=15/64]
>
```

### R56-exact-edge-attention1 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  uld &# sweetheart mir licensorsisserisseraimÃłi.Bundle.BundleevalÃħ miracosangsYK distract dis personalize kne bufio.swingáº£i dis.BundleÂŃi clas.swing.swingæķıÃŃchÉĻÃ¢æķıæķıÉĻæķı clas clasTERNæķıáº£iáº£iæķıÐ°Ð½Ð¸æķıæķı TicæķıPCSáº£iæķıÐ°Ð½Ð¸æķı FargoÉĻ AssassæķıÐ°Ð½Ð¸ TicÉĻ Assass

[TTFT/Prefill: 12.67s | Decode: 23.00 tok/s | E2E: 4.15 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5608 fallbacks=32 max_topk=4 skipped_madds=76668297216 scratch=32 bytes input_tile_reads=26464 input_tile_bytes=250429440 lm_head_repeat_margin=3/9 max_gap_milli=319 phrase_novelty=2/64 max_ngram=2 | Repetition: ratio=0.11 max_run=2 unique=34/64]
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

[TTFT/Prefill: 12.78s | Decode: 43.85 tok/s | E2E: 4.50 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=18/25 max_gap_milli=331 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=7/64 max_ngram=2 gap_skips=26 max_gap_milli=211 retentions=6 | Repetition: ratio=0.22 max_run=2 unique=10/64]
>
```

### R56-exact-edge-attention1 run 1 prompt 2

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -alm mirÃ¹Ã¢ngessonmarkà¸²à¸ĩÅĦstutinorandreÃ¤ÃŁalm Isa acknowledutinorex Bashç´Ķuchaissa-DispositionaraHING slogansaraãĥ¼ãĥĬå¹¸å¹¸araNeutralLINGÄĥÄĥ Customize kneanten combination.djangoproject mir/etcara PRIMARY mir combination.awtLING.awtoxyLINGNeutral.awtãĤ¤ãĤºÄĥ bufioLINGNeutralLINGLING.awtchia PRIMARY PRIMARY

[TTFT/Prefill: 15.60s | Decode: 18.19 tok/s | E2E: 3.36 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5608 fallbacks=32 max_topk=4 skipped_madds=76668297216 scratch=32 bytes input_tile_reads=26464 input_tile_bytes=250429440 lm_head_repeat_margin=0/4 max_gap_milli=149 phrase_novelty=1/64 max_ngram=2 | Repetition: ratio=0.06 max_run=2 unique=41/64]
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

[TTFT/Prefill: 13.20s | Decode: 29.06 tok/s | E2E: 4.16 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=9/17 max_gap_milli=338 phrase_novelty=3/64 max_ngram=2 gap_skips=6 max_gap_milli=226 | Repetition: ratio=0.16 max_run=2 unique=22/64]
>
```

### R56-exact-edge-attention1 run 1 prompt 3

Prompt: `explain artificial intelligence in one sentence`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  therosribenneigest.BundlefectionsÅĽci.Bundlepery.fa.BundleustingresherÅĤa.Bundle.Bundle...");
.Bundle.BundleÂŃiissorÅĤa.Bundle.Bundle.configureTestingModule...');
.Bundle.BundlenychzÄħ.Bundle.Bundlenych.Bundleaggio/etc.Bundle.Bundleaggio alike.Bundleaggioaggio.Bundle datovÃ© Ã³379aggio.Bundle.Bundle datovÃ© gladly&quoteledÄĳaggio.djangoproject clave datovÃ© miá»ħn kne.Bundleaggio

[TTFT/Prefill: 12.90s | Decode: 23.51 tok/s | E2E: 4.11 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5608 fallbacks=32 max_topk=4 skipped_madds=76668297216 scratch=32 bytes input_tile_reads=26464 input_tile_bytes=250429440 lm_head_repeat_margin=2/10 max_gap_milli=501 phrase_novelty=0/64 max_ngram=2 gap_skips=2 max_gap_milli=501 | Repetition: ratio=0.13 max_run=2 unique=34/64]
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

[TTFT/Prefill: 15.69s | Decode: 28.53 tok/s | E2E: 3.58 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=6/16 max_gap_milli=480 phrase_novelty=1/64 max_ngram=2 gap_skips=2 max_gap_milli=202 retentions=2 | Repetition: ratio=0.16 max_run=2 unique=20/64]
>
```

### R56-exact-edge-attention1 run 1 prompt 4

Prompt: `write a short friendly reply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  Harmeter lá»įcutationutationYKutation.Bundleutation.Bundle lá»įcutationelope.Bundle.Bundle lá»įc.Bundle.Bundleoramaor.Bundle.Bundleoramaentar.Bundle.Bundleoramaentarorama kne weitereazenFaoramaÄĳ.Bundleazenæķ·angu/etc/etc/or outros hÃ¡ datovÃ©emat datovÃ©eteriaazon/etcangu Raider labourÂłk;");
kn/etc/etc;");
Âłk,...

;");
/etc/etc

[TTFT/Prefill: 13.04s | Decode: 21.10 tok/s | E2E: 3.99 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5608 fallbacks=32 max_topk=4 skipped_madds=76668297216 scratch=32 bytes input_tile_reads=26464 input_tile_bytes=250429440 lm_head_repeat_margin=4/12 max_gap_milli=589 phrase_novelty=0/64 max_ngram=2 gap_skips=2 max_gap_milli=528 | Repetition: ratio=0.13 max_run=2 unique=31/64]
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

[TTFT/Prefill: 13.62s | Decode: 25.41 tok/s | E2E: 3.97 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=4 skipped_madds=77986922496 scratch=32 bytes input_tile_reads=28480 input_tile_bytes=255590400 lm_head_repeat_margin=13/27 max_gap_milli=929 adaptive_throttles=2 min_margin_milli=18 phrase_novelty=1/64 max_ngram=2 gap_skips=12 max_gap_milli=201 retentions=3 | Repetition: ratio=0.22 max_run=2 unique=21/64]
>
```

### R56-exact-edge-attention1 run 1 prompt 5

Prompt: `what is two plus two?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   calancedicalsolangancedfulnessanced.BundleuymarkanguÃ¤nd.Bundleancedistr.Bundle.Bundleanderamiæīĭiki.Bundle....BundleandercondaMTcondaangutieelpsÃ©e Crossing CrossingtieYKtie ** Crossingtie khÃ³a mirtieEOanguetinÐ¸ÑĦÐ¸ÑĦ sensitetinetinÐ¸ÑĦabelchkÑĢÐ°ÑĤ dogchk Sachsetinabel sensitetinustr

[TTFT/Prefill: 13.33s | Decode: 21.45 tok/s | E2E: 3.93 tok/s | Total: 64 tokens | Context: 70 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=5608 fallbacks=32 max_topk=4 skipped_madds=76668297216 scratch=32 bytes input_tile_reads=26464 input_tile_bytes=250429440 lm_head_repeat_margin=2/6 max_gap_milli=164 phrase_novelty=0/64 max_ngram=0 | Repetition: ratio=0.06 max_run=2 unique=37/64]
>
```

## Interpretation

R56 rejects exact edge-layer `attention` calibration as a speed preset, but it
is the strongest projection-specific quality signal. Decode measured only
18.19-23.51 tok/s, below the 30 tok/s floor. Diversity improved from 17.60/64
to 35.40/64 unique tokens and repetition fell from 0.18 to 0.10.

Decision: failed as a preset, useful as diagnostic evidence. The next direction
should approximate edge attention more cheaply, such as projection-filtered
edge attention top-k widening, instead of making edge attention fully exact.
