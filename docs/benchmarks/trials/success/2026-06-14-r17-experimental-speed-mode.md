# Trial: R17 Experimental Speed Mode

Date: 2026-06-14
Owner: RLLM
Status: success
Folder: success

## Hypothesis

Turbo Sparse Decode can improve Llama 3.2 1B Instruct CPU-only decode speed by
reducing raw BF16 MLP projection work without changing model weights or default
exact-lowram behavior.

## Scope

- Mode: experimental-speed
- Models/artifacts: `models/SmolLM2-135M-raw.rllm`, `models/Llama-3.2-1B-Instruct-raw.rllm`
- Architecture: llama
- Target device/profile: CPU-only, low RAM
- Bottleneck tag: sparse MLP projection
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- Top-k sweep: `RLLM_TURBO_TOPK=128`, `256`, `512`

## Setup

Commands:

```bash
cargo build --release -p rllm-cli --bin llama-test

printf 'good morning\nexit\n' | \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_TURBO_TOPK=256 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_TURBO_TOPK=128 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16

printf 'good morning\nexit\n' | \
  RLLM_EXPERIMENTAL_SPEED=1 RLLM_TURBO_TOPK=512 \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-raw.rllm \
    --ctx 2048 \
    --max-new-tokens 16
```

## Results

| model | variant | generated | TTFT/prefill | decode tok/s | end-to-end tok/s | sparse calls | fallbacks | max top-k | skipped madds | max RSS | peak footprint | RLLM peak transient |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | exact baseline | 16 | 13.61 s | 0.40 | 0.31 | 0 | 0 | 0 | 0 | 2475032576 bytes | 1620643104 bytes | 1050689536 bytes |
| Llama-3.2-1B-Instruct | experimental top-k 256 | 16 | 12.94 s | 0.67 | 0.45 | 480 | 32 | 256 | 10947133440 | 2477981696 bytes | 1620561184 bytes | 1050689536 bytes |
| Llama-3.2-1B-Instruct | experimental top-k 128 | 16 | 12.40 s | 1.61 | 0.74 | 480 | 32 | 128 | 11513364480 | 2493988864 bytes | 1620512032 bytes | 1050689536 bytes |
| Llama-3.2-1B-Instruct | experimental top-k 512 | 16 | 11.50 s | 1.35 | 0.71 | 480 | 32 | 512 | 9814671360 | 2494267392 bytes | 1620479264 bytes | 1050689536 bytes |
| SmolLM2-135M | exact baseline | 16 | 0.95 s | 22.50 | 9.89 | 0 | 0 | 0 | 0 | 458915840 bytes | 188891928 bytes | 113262592 bytes |
| SmolLM2-135M | experimental top-k 128 | 16 | 1.34 s | 32.94 | 8.93 | 900 | 60 | 128 | 984268800 | 459849728 bytes | 189809408 bytes | 113262592 bytes |

## Analysis

R17 validates the first experimental-speed path. The top-k 128 Llama 1B run
improved decode throughput from `0.40 tok/s` to `1.61 tok/s`, about `4.0x`,
while RLLM tracked peak transient memory stayed flat at `1050689536` bytes.
The sparse path was active: the best Llama run recorded `480` sparse projection
calls, `32` exact fallbacks, and `11513364480` estimated skipped multiply-adds.

The result is approximate, as intended. Output quality degraded in the sparse
runs: top-k 128 repeated `morning`, top-k 256 repeated `tone`, and top-k 512
mostly produced punctuation/space-like output. This does not qualify for
default inference, but it is useful research evidence that activation-guided
sparse MLP compute can move CPU-only speed without changing model weights or
raising RLLM's tracked transient memory.

Top-k 128 was the best Llama setting in this first sweep. Top-k 256 improved
only to `0.67 tok/s`, and top-k 512 reached `1.35 tok/s`. This suggests that
the speed tradeoff is not monotonic and should be swept per model or guided by
quality gates in a later stage.

SmolLM2 also produced a useful positive control: exact baseline measured
`22.50 tok/s`, while experimental top-k 128 reached `32.94 tok/s`. That reaches
the project's 30-40 tok/s band for the small model, but the text is also
approximate and degenerated into repeated punctuation/short tokens.

## Decision

success

Reason: Llama 3.2 1B Instruct improved by more than 2x in experimental-speed
mode while RLLM tracked peak transient memory stayed flat. The path is not a
quality-complete chat mode yet, but it proves a strong speed signal.

Paper value:

- useful positive evidence: RLLM's original activation-guided sparse MLP path
  can move dense Llama 1B CPU decode speed by about 4x without model
  compression or higher tracked transient memory
- useful limitation: naive sparse MLP alone hurts output quality and still
  remains below the 30-40 tok/s Llama 1B target

## Next Experiment

R18 should keep `RLLM_EXPERIMENTAL_SPEED=1` opt-in and target quality-aware
sparse routing. The next candidates are sparse LM-head shortlist projection,
adaptive top-k by layer/projection, or a packed sparse-friendly access layout.
The speed target should be pushed toward `5-10 tok/s` on Llama 1B before
claiming a credible path to `30-40 tok/s`.
