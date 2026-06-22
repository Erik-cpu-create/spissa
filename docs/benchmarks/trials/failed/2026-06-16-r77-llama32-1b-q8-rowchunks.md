# Trial: Llama 3.2 1B Q8 Row-Aligned Chunks

Date: 2026-06-16
Owner: RLLM
Status: rejected
Folder: failed

## Hypothesis

Packing Q8_0 2D tensors on whole-row chunk boundaries should let the Q8 row fast
path trigger more consistently on Llama 3.2 1B Instruct. The expected result was
better exact-lowram decode speed without changing output quality.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Architecture: Llama 3.2 1B Instruct
- Target device/profile: local CPU, release build
- Expected bottleneck: Q8 transformer projection chunk locality
- Bottleneck tag: cache locality

## Setup

Commands:

```bash
target/release/rllm pack models/downloads/llama-3.2-1b-instruct-unsloth/model.safetensors --out models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --codec raw --quantize q8_transformer_keep_io

printf '%s\nquit\n' 'Answer in one short sentence: what is 2 plus 2?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 8
printf '%s\nquit\n' 'Translate to Indonesian: I am learning Rust.' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 8
printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4
```

Runtime context:

- build profile: release
- OS: macOS
- relevant config: `--chat-template llama3`
- artifact sizes: raw 2.3G, Q8 rowchunks 1.5G
- pack output: 1622 chunks total, Q8 transformer weights with raw embed/lm_head

## Results

| run | prompt/input tokens | generated tokens | TTFT/prefill | decode tok/s | end-to-end tok/s | RSS | peak transient | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| old Q8 | 66 | 8 | 30.75s | 1.69 | 0.23 | 1820524544 | 1050673152 | `2 + 2 = 4` |
| rowchunks | 66 | 8 | 29.52s | 1.59 | 0.24 | 2001174528 | 1050673152 | `2 + 2 = 4` |
| old Q8 | 60 | 7 | 29.53s | 1.23 | 0.20 | 1644855296 | 1050673152 | `Saya belajar Rust.` |
| rowchunks | 60 | 7 | 27.79s | 1.43 | 0.22 | 3172663296 | 1050673152 | `Saya belajar Rust.` |
| old Q8 | 55 | 2 | 29.36s | 0.49 | 0.06 | 1809907712 | 1050673152 | `No` |
| rowchunks | 55 | 2 | 27.94s | 0.60 | 0.07 | 2754134016 | 1050673152 | `No` |

## Analysis

The model choice matters more than the rowchunk layout for short chat quality:
Llama 3.2 1B Instruct answered the sanity prompts correctly under both raw and
Q8 keep-IO artifacts.

The row-aligned packer change is still the right container layout for Q8 row
fast paths. For shape `[8192, 2048]`, one Q8 row is `(2048 / 32) * 34 = 2176`
bytes, and the packer now chooses `1046656`, a whole-row multiple. The previous
generic Q8 block alignment only guaranteed 34-byte block boundaries, so it could
split a logical matrix row.

However, the measured runtime improvement was mixed and too small to accept as a
speed win. Internal peak transient memory stayed at `1050673152` bytes, while
`/usr/bin/time` RSS was noisy and sometimes higher. The dominant bottleneck is
still exact Llama 1B prefill/projection cost, not just Q8 chunk row alignment.

## Decision

rejected

Reason: keep the packer layout fix, but reject row-aligned Q8 chunking as a
standalone Llama 1B speed improvement.

Paper value:

- use as negative evidence

## Next Experiment

Do not keep tuning chunk boundaries. The next useful slice is profiler-driven
Llama 1B exact-Q8 prefill attribution, then a targeted runtime change for the
dominant projection path.
