# R158a — compressed-resident RAM win PROVEN: 0.40 GB vs bf16 0.61 GB (−33%, lossless) (GO)

- Date: 2026-06-20
- Model: Gemma 3 1B IT lm-head ([262144×1152] bf16)
- Verdict: **GO** — holding the rANS-compressed weights resident and decoding the GEMV
  **block-by-block** (never materializing the full bf16) uses **33% less peak RAM**
  (0.40 GB vs bf16's 0.61 GB) for the same tensor, lossless. The RAM advantage is real,
  not on paper — the earlier 3.17 GB was the *naive whole-tensor-decode* path.

## Why this matters

R157c measured a rANS-packed model running inference at **3.17 GB peak RSS — WORSE than
bf16's 2.34 GB**, because the container decode path materializes the *whole* tensor to
bf16 per access (1.8 GB transient). That made rANS look pointless for inference. This
trial isolates whether the *block-decode* design (R153–R157b) — which never materializes
the full bf16 — actually delivers the RAM win.

## Method

`r158_lmhead_resident_ram` (#[ignore], modes via `RLLM_BENCH_MODE`), measured with
`/usr/bin/time -l` (max RSS). To remove model-open noise (`LazyRllmModel::open`
SHA-verifies the whole 1.9 GB), the measured modes never open the model — they read the
pre-dumped lm-head two ways:

- **bf16:** read the 604 MB bf16 lm-head into heap + GEMV.
- **rans:** read the 381 MB rANS sidecar into heap + `gemv_from_resident_rans` — parse the
  RLMR header, decode each block (interleaved rANS) into a reused exp buffer, reconstruct
  one bf16 row at a time into a reused scratch, dot. The full bf16 tensor is **never
  materialized**; the only heap beyond the compressed bytes is one block of exponents
  (~0.5 MB) + one bf16 scratch row + the output.

## Results — GO

```
bf16  (hold 604 MB bf16 + GEMV)                 max RSS = 607,633,408 B  (0.608 GB)
rANS  (hold 381 MB compressed + block-decode)   max RSS = 402,882,560 B  (0.403 GB)
=> rANS uses 33% less RAM, same tensor, lossless.
```

(For contrast, the naive whole-tensor-decode path was 2.55 GB in the same harness when it
*also* opened the model; isolating the per-tensor footprint is what makes the win visible.)

## Analysis

- **The win is the compression ratio realized as RAM:** 381/604 ≈ 0.63, and the RSS tracks
  it (0.40/0.61 ≈ 0.66). Block-decode adds only kilobytes of reused buffers, so the
  resident footprint ≈ the compressed size, not the bf16 size.
- **The 3.17 GB (R157c) was an implementation artifact**, not a property of rANS: the
  generic container path materializes whole tensors. The streaming/block-decode kernel
  (R153–R157b) does not, and this trial proves it pays off in RAM.
- **Scales to the whole model:** ~1.324 GB compressed vs ~2.0 GB bf16 → a ~0.7 GB (−34%)
  resident win if the forward pass uses block-decode. This is the > RAM enabler: a model
  whose bf16 doesn't fit can fit losslessly as rANS.

## Decision

**GO** — the compressed-resident RAM win is measured and real (−33% per tensor, lossless).
The remaining work (R158b) is wiring block-decode into the engine forward pass so a full
model run hits ~1.3 GB instead of the naive path's 3.17 GB.

## Next (R158b)

- Replace the forward pass's whole-tensor `decode_tensor` for rANS weights with a
  block-decode GEMV (compressed weights resident, decode per block, reuse buffers).
- Re-measure full-model peak RSS — target < bf16's 2.34 GB.
- Then the > RAM demo: a model whose bf16 exceeds RAM but whose rANS fits.

## Verification status

- [x] Compressed-resident block-decode GEMV: 0.40 GB vs bf16 0.61 GB (−33%), lossless.
- [x] Confirms the 3.17 GB (R157c) was the naive whole-tensor-decode path, not rANS itself.
- [x] 0 warnings; lossless round-trips green.
