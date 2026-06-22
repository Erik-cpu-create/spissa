# Trial: R148 — REESTREAM pipelined streaming bit-plane GEMV (the capacity-bound runtime kernel)

Date: 2026-06-19
Owner: RLLM
Status: accepted (GO)
Folder: success

## Hypothesis

R147 measured the capacity-bound win (model > RAM, cold SSD) at 1.13× but
un-pipelined (read a whole block, *then* decode → a small additive residual,
1.50 vs 1.62 GB/s effective). Hypothesis: a **double-buffer pipeline** that reads
the next block from disk while decoding+dotting the current one hides the decode
fully under the cold SSD, pushing the win toward the ~1.23× pure-byte ratio.

## Scope

- Mode: experimental (capacity-bound streaming GEMV, pipelined — the runtime kernel)
- REE kernel: REESTREAM (working name; Erik's final call)
- Model/artifact: `Llama-3.2-1B-Instruct-raw.spsa` bf16 embedding, block-framed (256 rows/block), replicated > RAM
- Target device/profile: Apple A18 Pro, macOS; release; cold SSD (F_NOCACHE, files > RAM)
- Bottleneck tag: storage bandwidth

## Setup

```bash
# bit-identity (pipelined == single-thread decode+dot)
cargo test -p rllm-runtime --lib streaming_gemv_matches_reference -- --nocapture
# >RAM cold-stream bench: pipelined vs raw bf16 pure read
cargo test -p rllm-runtime --release streaming_gemv_capacity_bound_bench -- --ignored --nocapture
```

Runtime context: release; Apple A18 Pro; files replicated to 6.3 GB (raw) / 5.1 GB
(bit-plane) > ~3 GB free RAM → genuinely cold SSD reads (F_NOCACHE). The raw
baseline is a **pure cold read** (no dot) — the strongest, fairest baseline (a real
pipelined raw path would hide its dot under the read too).

## Results

| stream | size | wall-clock | effective | logits |
|---|---:|---:|---:|---|
| raw bf16 (pure cold read) | 6.3 GB | 3865 ms | 1.63 GB/s | baseline |
| R148 pipelined (read + decode + dot) | 5.1 GB | 2918 ms | 1.76 GB/s | bit-identical ✓ |

- **SPEEDUP vs raw: 1.32× — GO.** (R147 un-pipelined scout was 1.13×.)
- **Lossless parity: bit-identical** — the pipelined streaming logits equal a
  single-thread `decode_neon_w5 + bf16_row_dot_f32` reference byte-for-byte
  (test green). 292 rllm-runtime lib tests pass.

## Analysis

The pipeline works, and it beats both the un-pipelined scout (1.13×) and the
pure-byte ratio (1.235×):

- **Decode fully hidden.** The R147 residual (decode additive, effective 1.50 GB/s)
  is gone — the consumer decodes block N while the reader streams block N+1, so the
  decode (~28× faster than the SSD) never shows up in the wall-clock.
- **Better I/O pattern, too.** The pipelined effective rate (1.76 GB/s) *exceeds*
  the raw single-read rate (1.63 GB/s): the double-buffered 852 KB block reads keep
  the SSD queue fuller than one large `read_exact`, so the compressed stream both
  reads fewer bytes AND reads them slightly faster. That is why 1.32× > 1.235×.

Where this lands in the arc: R148 is the **production capacity-bound kernel**. The
R140–R145 in-RAM NO-GOs were the wrong regime; R147 identified the right one
(model > RAM, slow storage); R148 builds the kernel that realizes it — **1.32×
faster, lossless, 19% less I/O/RAM**, streaming a model that exceeds RAM from cold
SSD. This is the mission operating point: cheap, low-RAM devices running models
bigger than their RAM. (And if compression makes the model *fit* in RAM instead of
streaming, the win is far larger — that is a separate, bigger lever.)

## Decision

accepted (GO) — pipelined streaming bit-plane GEMV is 1.32× faster than raw bf16
streaming from cold SSD, lossless (bit-identical), 19% less I/O/RAM, in the
capacity-bound regime. The decode is fully hidden under the SSD read.

Paper value:

- use as positive evidence: the working CPU/ARM capacity-bound runtime kernel —
  block-framed lossless bit-plane planes streamed from SSD, decode pipelined with
  the read, 1.32× over raw bf16. Completes the R147 finding from "the regime wins"
  to "the production kernel realizes it." The CPU/edge analog of GPU
  decompress-in-kernel (Cloudflare/DFloat), at the model-exceeds-RAM operating point.

## Next Experiment

R149+ — full model wiring: a `pack` mode that writes a model's bf16 tensors in the
block-framed bit-plane layout, register `rtc-bitplane-v1` in `codec_for_id`, and a
forward pass that streams the compressed weights from disk through
`streaming_bitplane_gemv` (and the analogous per-projection kernels), so a model
that genuinely exceeds device RAM runs faster + lighter than raw bf16, lossless.
Measure generation tok/s on a model > RAM (e.g. a 7B+ bf16 model on an 8 GB device).
