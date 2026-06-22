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
| R45-retention100-lmhead-topk5 | 5 | false | false | 26.76 | 62.73 | 40.35 | 14.60 | 0.20 |
| R45-retention100-lmhead-topk6 | 5 | false | false | 26.77 | 52.21 | 36.89 | 19.60 | 0.28 |

## Runs

| variant | run | prompt | prefill s | decode tok/s | e2e tok/s | generated | context | peak bytes | repetition | max run | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R45-retention100-lmhead-topk5 | 1 | 1 | 11.61 | 40.60 | 4.86 | 64 | 66 | 1050689536 | 0.19 | 2 | 14/64 |
| R45-retention100-lmhead-topk6 | 1 | 1 | 12.01 | 52.21 | 4.84 | 64 | 66 | 1050689536 | 0.32 | 2 | 17/64 |
| R45-retention100-lmhead-topk5 | 1 | 2 | 12.38 | 35.14 | 4.52 | 64 | 66 | 1050689536 | 0.17 | 2 | 19/64 |
| R45-retention100-lmhead-topk6 | 1 | 2 | 12.77 | 39.43 | 4.45 | 64 | 66 | 1050689536 | 0.19 | 2 | 19/64 |
| R45-retention100-lmhead-topk5 | 1 | 3 | 13.25 | 62.73 | 4.49 | 64 | 68 | 1050689536 | 0.32 | 2 | 5/64 |
| R45-retention100-lmhead-topk6 | 1 | 3 | 13.55 | 38.86 | 4.22 | 64 | 68 | 1050689536 | 0.35 | 2 | 18/64 |
| R45-retention100-lmhead-topk5 | 1 | 4 | 13.86 | 36.52 | 4.11 | 64 | 68 | 1050689536 | 0.19 | 2 | 12/64 |
| R45-retention100-lmhead-topk6 | 1 | 4 | 14.59 | 26.77 | 3.78 | 64 | 68 | 1050689536 | 0.30 | 2 | 22/64 |
| R45-retention100-lmhead-topk5 | 1 | 5 | 15.61 | 26.76 | 3.56 | 64 | 69 | 1050689536 | 0.14 | 2 | 23/64 |
| R45-retention100-lmhead-topk6 | 1 | 5 | 14.30 | 27.18 | 3.85 | 64 | 69 | 1050689536 | 0.22 | 2 | 22/64 |

## Interpretation

R45 rejects smaller LM-head widening steps. Top-k 5 and top-k 6 both failed
the 30 tok/s floor in this one-run multi-prompt sweep. Top-k 6 improved average
unique tokens to 19.60/64, but prompt 4 and prompt 5 fell below the floor.

Decision: failed. The top-k 5/6 sweep confirms that LM-head widening remains
too unstable for the speed target even after the small top-k selector
optimization.

## Raw Output

### R45-retention100-lmhead-topk5 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   mal miryth miryth.swing mirolanyth mir.swing mir mir.swing.swing mir.swing mir mir.swing.swing mir mir.swing mir.swing.swing mir mir.swing mir.swing mir.swing mir mir.swing mir.swing mir.swing disilersvestelage mir.swingfinerede mir.swing.swing mir mir.swing mir.swing.swing mir.swing.swinglaughterclub

[TTFT/Prefill: 11.61s | Decode: 40.60 tok/s | E2E: 4.86 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=5 skipped_madds=77978714112 scratch=40 bytes input_tile_reads=28544 input_tile_bytes=272007168 lm_head_repeat_margin=15/26 max_gap_milli=229 adaptive_throttles=5 min_margin_milli=18 phrase_novelty=4/64 max_ngram=2 gap_skips=22 max_gap_milli=240 retentions=1 | Repetition: ratio=0.19 max_run=2 unique=14/64]
>
```

### R45-retention100-lmhead-topk6 run 1 prompt 1

Prompt: `good morning`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   dis mir.swing mir mir.swinglaughterĳľ.swing mir rem mir mir.swing mir mir infleldoneldonskinLaugh scar rem rem mir mir.swing mir.swing.swing mir mir.swing.swing rem mir mir.swing.swing mir mir.swing.swing mir mir.swing.swingfineunefine mÅ©iogradogradelage mir mir.swing.swing mir mir.swing.swing mir

[TTFT/Prefill: 12.01s | Decode: 52.21 tok/s | E2E: 4.84 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=6 skipped_madds=77970505728 scratch=48 bytes input_tile_reads=28608 input_tile_bytes=288423936 lm_head_repeat_margin=5/23 max_gap_milli=637 phrase_novelty=3/64 max_ngram=2 gap_skips=3 max_gap_milli=353 retentions=1 | Repetition: ratio=0.32 max_run=2 unique=17/64]
>
```

### R45-retention100-lmhead-topk5 run 1 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -Cal mir.swing miryth miryth.swing Hubbard mirmith mir.swing mir.swing.swing mir.swing.swing remimpl mirlaughterfinefineipers mirlaughter.swing mir mirlaughterfinefineipers mirlaughter.swing.swing disilers mir mir.swing.swing disilersstitutionsilersSlashrede mir.swing mir mir.swing.swing mir mir.swing stickers mÅ©ielage.swing

[TTFT/Prefill: 12.38s | Decode: 35.14 tok/s | E2E: 4.52 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=5 skipped_madds=77978714112 scratch=40 bytes input_tile_reads=28544 input_tile_bytes=272007168 lm_head_repeat_margin=5/13 max_gap_milli=567 phrase_novelty=5/64 max_ngram=2 gap_skips=5 max_gap_milli=303 retentions=1 | Repetition: ratio=0.17 max_run=2 unique=19/64]
>
```

### R45-retention100-lmhead-topk6 run 1 prompt 2

Prompt: `halo`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
> -Cal.swing mir.swing miryth miryth miryth.swing mir.swing mirlaughter.swing mir.swing.swingMainrede mirlaughterilers.swing.swing mir mirlaughter.swingAdvelage.swing Gladvestvestilers.swing mal yolandolandrede mir.swing.swing skinfinefine mÅ©iogradrede mir mir.swing.swing mir mir.swing.swing mir mir.swingAdv

[TTFT/Prefill: 12.77s | Decode: 39.43 tok/s | E2E: 4.45 tok/s | Total: 64 tokens | Context: 66 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=6 skipped_madds=77970505728 scratch=48 bytes input_tile_reads=28608 input_tile_bytes=288423936 lm_head_repeat_margin=1/13 max_gap_milli=670 phrase_novelty=2/64 max_ngram=2 gap_skips=6 max_gap_milli=300 | Repetition: ratio=0.19 max_run=2 unique=19/64]
>
```

### R45-retention100-lmhead-topk5 run 1 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  iltyelyn mir mir.swing mir mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing.swing mir mir.swing.swing mir.swing mir.swing.swing mir mir.swing.swing mir.swing mir.swing.swing mir.swing mir.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing mir.swing mir mir.swing.swing mir mir.swing.swing mir

[TTFT/Prefill: 13.25s | Decode: 62.73 tok/s | E2E: 4.49 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=5 skipped_madds=77978714112 scratch=40 bytes input_tile_reads=28544 input_tile_bytes=272007168 lm_head_repeat_margin=15/27 max_gap_milli=240 adaptive_throttles=3 min_margin_milli=18 phrase_novelty=8/64 max_ngram=2 gap_skips=23 max_gap_milli=274 retentions=4 | Repetition: ratio=0.32 max_run=2 unique=5/64]
>
```

### R45-retention100-lmhead-topk6 run 1 prompt 3

Prompt: `who are you?`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  iltyelyn mir mir.swing disngx mir mir.swing mir mirlaughter mir mir.swing mir mir inflograd mÅ©i mÅ©ivestveststitutionsstitutionselage.swing.swing mir mir.swing.swing mir mir.swing.swing inflolandilers.swing.swing gu mir.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing.swing disiband

[TTFT/Prefill: 13.55s | Decode: 38.86 tok/s | E2E: 4.22 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=6 skipped_madds=77970505728 scratch=48 bytes input_tile_reads=28608 input_tile_bytes=288423936 lm_head_repeat_margin=3/22 max_gap_milli=594 phrase_novelty=5/64 max_ngram=2 gap_skips=4 max_gap_milli=392 | Repetition: ratio=0.35 max_run=2 unique=18/64]
>
```

### R45-retention100-lmhead-topk5 run 1 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  osengx disngx mir mir.swing glinnamon dis vern mir.swing vern mir mir.swing gu mir.swing mir mir.swing mir.swing mir mir.swing mir.swing mir.swing mir.swing%dascal.swing mir.swing mir mir.swing mir.swing mir mir.swing mir.swing mir mir.swing mir mir.swing.swing mir mir.swing.swing mir mir.swing

[TTFT/Prefill: 13.86s | Decode: 36.52 tok/s | E2E: 4.11 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=5 skipped_madds=77978714112 scratch=40 bytes input_tile_reads=28544 input_tile_bytes=272007168 lm_head_repeat_margin=12/18 max_gap_milli=134 phrase_novelty=7/64 max_ngram=2 gap_skips=21 max_gap_milli=259 retentions=5 | Repetition: ratio=0.19 max_run=2 unique=12/64]
>
```

### R45-retention100-lmhead-topk6 run 1 prompt 4

Prompt: `explain artificial intelligence simply`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>   Fin.swing mir mir.swing mir mirlaughter rem rem Fin Fin.swingalus.swingickey Colum vern mir mir infl mir mir.swing restfinefine lipfinefineHint Colum mir.swing mal rem âĢĭâĢĭascalascalolandolandelage.swing.swing mir.swing.swing mir mir.swing.swing mir mir.swing.swinglaughterlaughterilersilersBlocks.swing.swing mir

[TTFT/Prefill: 14.59s | Decode: 26.77 tok/s | E2E: 3.78 tok/s | Total: 64 tokens | Context: 68 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=6 skipped_madds=77970505728 scratch=48 bytes input_tile_reads=28608 input_tile_bytes=288423936 lm_head_repeat_margin=1/18 max_gap_milli=1193 phrase_novelty=3/64 max_ngram=2 gap_skips=6 max_gap_milli=440 | Repetition: ratio=0.30 max_run=2 unique=22/64]
>
```

### R45-retention100-lmhead-topk5 run 1 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  naturally Fin fÃŃs accidents diff fÃŃs fÃŃs accidents helicopters mir diff diff fÃŃs Mund fÃŃs Ledger LedgerAdvcabiness.swing.swing%d Mund Mund diffinessCab mir.swing mir mir.swing fÃŃs '// mir fÃŃs fÃŃsclub Mund Hubbard mir Mundclubrede mir.swing fÃŃselage.swing stickerscabrede mir.swing mir mir.swing fÃŃs fÃŃsilersvestilers.swing

[TTFT/Prefill: 15.61s | Decode: 26.76 tok/s | E2E: 3.56 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=5 skipped_madds=77978714112 scratch=40 bytes input_tile_reads=28544 input_tile_bytes=272007168 lm_head_repeat_margin=8/15 max_gap_milli=454 phrase_novelty=4/64 max_ngram=2 gap_skips=2 max_gap_milli=268 | Repetition: ratio=0.14 max_run=2 unique=23/64]
>
```

### R45-retention100-lmhead-topk6 run 1 prompt 5

Prompt: `write a short helpful answer`

```text
===================================================
RLLM Interactive Chat (Llama Architecture, token-native session)
Type 'quit' or 'exit' to end.
===================================================
>  below mir Montefinefine lip diff fÃŃs accidents Mund Mund diff diff Mund Mundrede.swing mir lip Mund Mundrede mir.swing mir mir.swing.swing mir mir.swing fÃŃs Mund Mundelage.swing.swing mir.swing.swingordovafineÌ£elage.swingNumrede mir.swingåĲ¾ilersilers wardrobe Canterilersilers wardrobeclubrede mir.swing.swing dis mÅ©i

[TTFT/Prefill: 14.30s | Decode: 27.18 tok/s | E2E: 3.85 tok/s | Total: 64 tokens | Context: 69 tokens | Peak: 1050689536 bytes | AIP: policy=speed calls=6112 fallbacks=32 max_topk=6 skipped_madds=77970505728 scratch=48 bytes input_tile_reads=28608 input_tile_bytes=288423936 lm_head_repeat_margin=0/12 max_gap_milli=699 phrase_novelty=2/64 max_ngram=2 gap_skips=1 max_gap_milli=257 retentions=1 | Repetition: ratio=0.22 max_run=2 unique=22/64]
>
```
