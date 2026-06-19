# Trial: R135 residency × int8-sdot — decode reaches ~11 tok/s (corrects R133)

Date: 2026-06-19
Owner: RLLM
Status: accepted (int8 sdot layers ~10x vs scalar-resident) — but see CORRECTION: the "~11 tok/s" was a layers-only mis-read; true decode ~3.9 tok/s
Folder: success

## CORRECTION (2026-06-19, same day)

The "~11 tok/s decode, matches Ollama" headline below was WRONG: the
`[gemma-profile] ... 34 layers Xms` line times ONLY the transformer-layer loop,
not the whole decode token. lm_head (the 262k-row vocab GEMV over the bf16 tied
embedding) runs after the timed region and is the LARGER cost. Corrected
per-token decode breakdown (--fast, steady-state): layers ~90ms + lm_head
~165ms = ~255ms/token = **~3.9 tok/s**, not 11. The int8 sdot result for the
LAYERS is real (~88ms layers, ~10x over scalar-resident); the error was reading
layers-only time as the whole token. vs Ollama 12.4 tok/s, decode is ~3.2x
behind (not matched). lm_head bf16 is now the dominant decode lever. The tables
below are kept as-recorded with this correction noted; R136 carries the honest
end-to-end numbers.

## Hypothesis

R133 concluded "decode is memory-bound, the int8 sdot kernel does not help" —
but that A/B was run WITHOUT residency (the model was thrashing from disk, so
page faults masked the kernel). With R134's mlock making the weights resident,
re-test the int8-activation sdot/i8mm path (`RLLM_Q8_ACTIVATION=1`). The
prediction: residency and the fast kernel are MULTIPLICATIVE, not redundant —
residency removes the page-fault stall so the int8 kernel can finally run at
SIMD speed; the scalar kernel even when resident is compute-bound at a low
rate. Combined, decode should approach the hardware/kernel ceiling.

## Scope

- Mode: fast-lowram runtime (q8, near-exact int8 activation — quant-only diff, same family as llama.cpp q8 inference)
- REE kernel: REEBORN-Q8-SDOT (int8-activation sdot/i8mm, R110/R127/R130/R132) — wired, not new
- Model/artifact: `models/gemma-3-4b-it-q8.rllm` (q8_transformer_keep_io, codec rtc-raw-v1, ~4.5 GB)
- Architecture: Gemma 3 4B, Q8_0
- Target device/profile: Apple Silicon, 8 GB RAM, CPU only
- Expected bottleneck: was mis-attributed (R133 said memory-bound); actually scalar-kernel compute once resident
- Bottleneck tag: CPU arithmetic (scalar → int8 sdot, in the resident regime)

## Setup

Commands (per-step timing via `RLLM_Q8_KERNEL_PROFILE`, steady-state = steps ≥2):

```bash
cargo build --release -p rllm-cli

# the four-cell matrix (mlock = residency, sdot = RLLM_Q8_ACTIVATION=1)
RLLM_MLOCK=1                       RLLM_Q8_KERNEL_PROFILE=1 ./target/release/gemma-test ... # mlock + scalar
RLLM_MLOCK=1 RLLM_Q8_ACTIVATION=1  RLLM_Q8_KERNEL_PROFILE=1 ./target/release/gemma-test ... # mlock + sdot

# shipped as one flag:
./target/release/gemma-test --model models/gemma-3-4b-it-q8.rllm \
  --prompt "The capital of France is" --fast --max-new-tokens 16 --ctx 256
```

Runtime context:

- build profile: release
- CPU: Apple Silicon (ARM64), CPU-only, RLLM_THREADS=8
- RAM: 8 GB
- OS: macOS (Darwin 25.5.0)
- relevant env/config: `--fast` sets RLLM_MLOCK=1 + RLLM_Q8_ACTIVATION=1

## Results

Steady-state decode per token (profile, seq_len=1, steps ≥2), Gemma 3 4B q8:

| config | decode steady-state | decode tok/s | note |
|---|---:|---:|---|
| scalar, no mlock (R133) | ~5700 ms | ~0.17 | thrashing (page faults) |
| sdot, no mlock | ~5700 ms | ~0.17 | thrashing masks the kernel |
| mlock + scalar | ~810 ms | ~1.23 | resident, scalar compute-bound |
| **mlock + sdot (`--fast`)** | **~88 ms** | **~11.3** | resident + int8 SIMD |

Confirmation run (`--fast`, 16 tokens) — decode steps 8–15: 89, 92, 92, 102,
89, 88, 89, 89 ms (stable ~11 tok/s). Output coherent and consistent:
"Paris is a global center for art, fashion, gastronomy and culture".

Whole-run phases (`--fast`, profile):

| phase | time | note |
|---|---:|---|
| step 0 prefill (6 tok) | ~7.1 s | 2.2x faster than scalar (15.7 s); still ~13x behind Ollama |
| step 1 decode (warmup) | ~6.2 s | ONE-TIME (int8 activation panel cache cold) |
| steps ≥2 decode | ~88 ms | steady-state ~11 tok/s |

vs Ollama CPU (same machine, Gemma 3 4B q8): decode 12.4 tok/s, prefill 11.3 tok/s.

## Analysis

R133's "memory-bound, sdot doesn't help" was an artifact of testing without
residency: with the model thrashing from disk, every config measured the same
~5.7 s/token because the bottleneck was the page fault, not the kernel. Once
mlock makes the weights resident, the real picture appears — the two levers are
multiplicative:

- residency alone (mlock + scalar): 810 ms — resident but the scalar
  `q8_0_dot_i8_f32` (i8→f32 cast + scalar FMA) is compute-bound at a low rate.
- kernel alone (sdot, no mlock): 5700 ms — fast kernel starved by faults.
- both (`--fast`): 88 ms — ~10x over scalar-resident, ~11 tok/s, matching Ollama.

So in the RESIDENT regime, batch=1 decode is CPU-arithmetic-bound on the scalar
kernel, and the int8 sdot/i8mm path unlocks the ~10x. This corrects R133 and
the memory-bound framing for the resident case (R133 remains correct for the
thrashing case).

The end-to-end tok/s (0.64–0.82 for 12–16 tokens) is now dragged by two
NON-decode costs: prefill (~7.1 s for 6 tokens) and a one-time step-1 decode
warmup (~6.2 s, the cold int8 activation panel cache). Decode itself is solved.

## Decision

accepted

Reason: ~10x steady-state decode over scalar-resident, ~11 tok/s matching
Ollama CPU, coherent/consistent output. Shipped as a first-class `--fast` flag
(bundles mlock + int8 activation, which only pay off together). Uses near-exact
int8 activation (quant-only diff vs the exact scalar path).

Limitation: end-to-end still dominated by prefill (~7.1 s, ~13x behind Ollama)
and a one-time step-1 warmup (~6.2 s). Those are the next levers, not decode.

Paper value:

- use as positive evidence (residency × kernel are multiplicative; isolating
  one lever can falsely conclude the other is useless — R133 cautionary tale).

## Next Experiment

Prefill: ~7.1 s for 6 tokens is ~13x behind Ollama and is now the dominant
end-to-end cost. Profile the prefill (batch=6, compute-bound) panel i8mm path
and the one-time step-1 decode warmup (~6.2 s, cold activation panel cache) —
either pre-warm the panel cache or make the prefill kernel feed it.
