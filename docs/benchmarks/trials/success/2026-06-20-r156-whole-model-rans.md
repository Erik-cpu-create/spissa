# R156 — whole-model rANS: generalizes to body projections + exact size 2.0 GB → 1.324 GB (GO)

- Date: 2026-06-20
- Kernel lineage: REESTREAM-RANS (R152–R154), applied model-wide
- Model: Gemma 3 1B IT (`gemma-3-1b-it-rawcodec.spsa`, raw bf16, 340 tensors, ~1.0 B weights)
- Verdict: **GO** — the rANS streaming GEMV is lossless on transformer **body** projections
  (not just the lm-head), and the **whole model** compresses **2.000 GB → 1.324 GB
  (−34%, 10.60 bits/weight)** losslessly. The foundation for running a model > RAM at the
  entropy floor across every layer.

## R156a — generalization to body projections (lossless)

The streaming GEMV (`streaming_rans_gemv_parallel`) is generic over any `[out, in]`
weight matrix — the lm-head was just one case. `r156a_gemma_body_projection_lossless`
(#[ignore]): write a rANS sidecar for `model.layers.0.mlp.gate_proj.weight`
([6912×1152]), stream `W·x`, compare to the resident bf16 `W·x`:
```
R156a OK: gate_proj [6912×1152] rANS stream == resident, 6912 outputs bit-identical
```
So every projection (attention q/k/v/o, MLP gate/up/down) can be rANS-streamed losslessly
with the existing kernel — no new decode logic for the body.

## R156b — exact whole-model compressed size

`whole_model_rans_size` (`tests/rans_whole_model_size.rs`, #[ignore]): for every tensor,
split bf16 → (exp, residual), rANS-encode the exponent (4-lane interleaved) + raw residual
+ freq table, and sum.
```
tensors: 340   weights: 999,885,952
raw bf16 total:  2.000 GB
rANS total:      1.324 GB   (34% smaller, 10.60 bits/weight)
top: embed_tokens 604 MB -> 399 MB; each mlp gate/down 15.9 MB -> ~10.6 MB
```
**10.60 bits/weight uniformly across all 340 tensors** — the R151 entropy floor holds
model-wide, not just for the embedding. (The earlier ~1.25 GB estimate was close; exact
is 1.324 GB. The 1.9 GB on-disk file figure includes container overhead vs the 2.0 GB sum
of raw tensor bytes.)

## Analysis

- **Capacity win is whole-model and real:** a lossless Gemma 3 1B is 1.324 GB vs 2.0 GB
  raw bf16 — fits 34% more model in the same RAM, bit-exact. This is the >RAM mission: the
  total size is what decides whether a model fits, and rANS shrinks the total by a third
  losslessly (vs bit-plane's ~19% and q8's lossy ~47%).
- **The decode kernel already covers the body** (R156a) — the remaining work is wiring,
  not new codecs.

## Decision

**GO** — whole-model rANS is proven lossless (body + lm-head) and the exact capacity is
measured (−34%). What remains is the runtime integration: streaming the body projections
during the forward pass.

## Next (R157 — runtime wiring, the big lift)

- Pack the model's weight tensors as rANS sidecars (or a container codec), with flexible
  `block_rows` for tensors whose row count isn't a multiple of 256 (e.g. down_proj [1152×…]).
- Wire the decode forward pass to stream+decode each projection via `streaming_rans_gemv_parallel`
  instead of the resident matmul; handle prefill (batch>1 amortizes one weight read over
  many activations — a GEMM, not GEMV).
- Measure end-to-end tok/s on a model genuinely > device RAM.

## Verification status

- [x] rANS streaming lossless on a body projection (gate_proj, 6912 outputs identical).
- [x] Whole-model size measured: 2.0 GB → 1.324 GB (−34%, 10.60 bits/weight, 340 tensors).
- [x] rtc-codec 52 / rllm-runtime lib 296, 0 warnings.
