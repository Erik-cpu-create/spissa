# Trial: R149a — REESTREAM streaming lm-head on Gemma 3 1B (correctness gate BLOCKED at sidecar)

Date: 2026-06-20
Owner: RLLM
Status: blocked (no-go for this model) — useful negative
Folder: failed

## Hypothesis

The R148 REESTREAM pipelined bit-plane GEMV (proven lossless and capacity-bound
on a synthetic > RAM artifact) should drop into a *real* model's lm-head as an
opt-in path (`RLLM_STREAM_LMHEAD=<sidecar>`) and produce a **byte-for-byte
identical** generated token sequence vs the resident path on Gemma 3 1B. The
gate is correctness/lossless first; speed is deferred (Gemma 1B fits RAM).

## Scope

- Mode: experimental, lm-head streaming, opt-in (`RLLM_STREAM_LMHEAD`)
- REE kernel: REESTREAM (reused from R148; `streaming_bitplane_gemv` + `stream_lmhead_from_sidecar`)
- Model/artifact: Gemma 3 1B (`models/gemma-3-1b-it-rawcodec.spsa`, repacked `--quantize raw --codec raw`)
- Tied lm-head tensor: `model.embed_tokens.weight`, shape **[262144, 1152]** (vocab 262144 × hidden 1152), bf16
- Target device/profile: Apple aarch64, macOS; release
- Bottleneck tag: codec width selection (decode-kernel constraint)

## Setup / commands run

```bash
# Step 1 — repack Gemma 1B with raw codec (embedding readable by with_raw_tensor)
./target/release/rllm pack /tmp/gemma1b --out models/gemma-3-1b-it-rawcodec.spsa \
  --quantize raw --codec raw --config /tmp/gemma1b/config.json --tokenizer /tmp/gemma1b/tokenizer.json
# -> "Found 340 tensors (340 packed for architecture 'gemma3')"; raw, ratio 100.0%, written.

# Step 2 — drive the sidecar writer over the real Gemma embedding
cargo test -p rllm-runtime --release write_gemma_lmhead_sidecar -- --ignored --nocapture
```

## Results — BLOCKED

The sidecar writer **refused** to produce `/tmp/gemma1b-lmhead.sidecar`:

```
called `Result::unwrap()` on an `Err` value:
InvalidTensorData("lm-head bit-plane width 6 != 5; decode16 kernel needs w=5")
```

Root cause (verified directly against `/tmp/gemma1b/model.safetensors`):

- The bit-plane palette is the set of **distinct bf16 exponent bytes** in the
  tensor. The streaming decode kernel (`decode_neon_w5` / `decode16`) is a
  fixed-width w=5 tbl-gather, i.e. it only supports a palette of **≤ 32**
  distinct exponents.
- Gemma 3 1B's `model.embed_tokens.weight` contains **34 distinct bf16
  exponents** → `index_width = ceil(log2(34)) = 6`. The writer's
  `enc.data[15] != 5` guard fires and errors out (by design).
- This is **intrinsic to the real model data**, not a pack/config mistake.
  Repacking cannot change it: the palette width is determined by the weights'
  exponent diversity. The R143/R148 synthetic fixture forced w=5 by construction
  (`exp = 96 + (k % 32)` = exactly 32 exponents, and hidden=2048); the real
  Gemma embedding is wider (34 exponents) and narrower (hidden=1152).

Consequence: **no valid sidecar exists for this model**, so the Step-3
correctness gate (resident vs `RLLM_STREAM_LMHEAD` token-id comparison) **could
not be run**. There are therefore no two token-id lines to compare. Per the
honest-metrics rule this is recorded as a BLOCKED/failed trial, not fudged.

## Analysis

- The R148 streaming kernel and the sidecar plumbing (writer, reader, opt-in
  Gemma branch) are all wired and compile; the blocker is purely the **w=5-only
  decode kernel** meeting a w=6 real tensor.
- Correctness-first was the right call: the gate caught the gap *before* any
  speed claim. Speed was already out of scope here (Gemma 1B fits RAM, so there
  is no capacity-bound win to demonstrate on this model regardless).
- The reproducer test (`write_gemma_lmhead_sidecar`, `#[ignore]`) is committed so
  the blocker is one command away from re-verification.

## Decision

**No-go for R149a as scoped.** The lossless-identical-tokens gate on Gemma 3 1B
cannot be reached until the streaming decode supports w=6 (palette 33–64). This
is a genuine negative result, not a regression in already-shipped code.

## Next (R149b)

1. Generalize the streaming decode to **variable index width** (at least add a
   w=6 path; `decode16`/`decode_neon_w5` currently hard-wired to w=5), then
   re-run this exact correctness gate on Gemma 3 1B.
2. Only after identical-tokens is green: move to the capacity-bound speed demo on
   a **> RAM** model (or cold-read), per the original R149b scope.
3. Stretch: stream the transformer projections, not just the lm-head.

## Verification status

- [x] Step 1 repack: OK (340 tensors, raw, written).
- [x] Step 2 sidecar writer driven: errored with w=6 (BLOCKED), as the guard intends.
- [ ] Step 3 identical-tokens gate: **not reachable** (no sidecar). Blocked.
