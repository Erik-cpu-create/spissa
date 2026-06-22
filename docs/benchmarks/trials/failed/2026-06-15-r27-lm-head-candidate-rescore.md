# Trial: R27 LM-Head Candidate Rescore

Date: 2026-06-15
Owner: RLLM
Status: failed
Folder: failed

## Hypothesis

R26 removed adjacent repeat collapse while staying at the 30 tok/s gate, but
semantic quality stayed poor. R27 tested a sparse-to-exact LM-head bridge:

1. sparse input-tile LM-head computes approximate full-vocab logits
2. top sparse candidate token IDs are selected
3. the original row-major LM-head scores only those candidate rows exactly
4. argmax is chosen from that small exact candidate set

The goal was to improve token choice while avoiding full-vocab exact LM-head
scan.

## Scope

- Mode: experimental-speed
- Model/artifact: `models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa`
- Architecture: llama
- Runtime gate: `RLLM_EXPERIMENTAL_SPEED=1`
- Input-tile gate: `RLLM_AIP_INPUT_TILES=1`
- AIP policy: `RLLM_AIP_POLICY=speed`
- Base transformer top-k: `RLLM_AIP_TOPK=4`
- Repeat guard: `RLLM_AIP_NO_REPEAT_LAST=1`
- Candidate rescore: `RLLM_AIP_LM_HEAD_RESCORE=<n>`

## Setup

Build:

```bash
cargo build --release -p rllm-cli --bin rllm --bin llama-test
```

Current selective-rescore benchmark:

```bash
printf 'good morning\nexit\n' | \
  RLLM_AIP_INPUT_TILES=1 RLLM_EXPERIMENTAL_SPEED=1 RLLM_AIP_POLICY=speed \
  RLLM_AIP_TOPK=4 RLLM_AIP_NO_REPEAT_LAST=1 RLLM_AIP_LM_HEAD_RESCORE=<n> \
  /usr/bin/time -l target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-r25-inputtiles-all-lmhead.spsa \
    --ctx 2048 \
    --max-new-tokens 64 \
    --profile-phases
```

## Results

| variant | generated | TTFT/prefill | decode tok/s | E2E tok/s | max top-k | input-tile reads | input-tile bytes | repetition ratio | max run | unique tokens | RLLM peak transient | max RSS | peak footprint | LM-head time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R26 no-repeat top-k 4 baseline | 64 | 12.93s | 30.06 | 4.26 | 4 | 28480 | 255590400 | 0.00 | 1 | 12/64 | 1050689536 | 2012823552 | 2157563072 | 75.07ms |
| R27 always-rescore 4 prototype | 64 | 15.17s | 22.60 | 3.56 | 4 | 28480 | 255590400 | 0.00 | 1 | 17/64 | 1050689536 | 1962704896 | 2157333648 | 494.44ms |
| R27 always-rescore 2 prototype | 64 | 14.56s | 26.83 | 3.78 | 4 | 28480 | 255590400 | 0.00 | 1 | 7/64 | 1050689536 | 1345290240 | 2157153496 | 459.08ms |
| R27 selective-rescore 4 | 64 | 13.08s | 14.01 | 3.64 | 4 | 28480 | 255590400 | 0.00 | 1 | 19/64 | 1050689536 | 1811660800 | 2157612224 | 2285.56ms |
| R27 selective-rescore 2 | 64 | 12.69s | 14.81 | 3.78 | 4 | 28480 | 255590400 | 0.00 | 1 | 12/64 | 1050689536 | 1374175232 | 2156989536 | 2398.16ms |

## Analysis

The hypothesis failed. Exact candidate rescoring reads original row-major
LM-head chunks. Even with only two to four candidates, candidate token rows are
distributed across large chunks, so the runtime reads much more row-major data
than the R26 sparse-only path.

The selective rescore implementation only triggers when sparse top-1 would
repeat the previous decode token, but this prompt hits that condition often.
LM-head time grew from 75.07ms in R26 to more than 2.2s in selective R27 runs,
dropping decode speed to 14-15 tok/s.

Always-rescore prototypes were also below the target. They improved unique
token count in the candidate-4 run, but stayed under 30 tok/s.

## Decision

failed

Reason: R27 does not preserve the required 30-40 tok/s decode speed for Llama
3.2 1B Instruct. It is useful negative evidence, but not a runtime mode to
recommend.

Paper value:

- negative evidence that row-major exact candidate rescoring is too expensive
  without row-range sidecars or row-block metadata
- confirms that R26 remains the current best speed-gated mode
- suggests future quality work needs either exact candidate rows in a dedicated
  small-row sidecar or a better sparse approximation before LM-head

## Next Experiment

R28 should avoid row-major chunk reads:

- add a candidate-row sidecar for LM-head rows if exact candidate rescore remains
  worth testing
- or measure exact-reference token agreement first, then tune sparse projection
  policy by layer group before adding more storage
- keep R26 as the benchmark floor: any new quality mode must beat or match
  30 tok/s over the same 64-token prompt
