# R149b — REEPLANE-W6 w=6 bit-plane decode unblocks Gemma streaming lm-head (GO)

- Date: 2026-06-20
- Kernel lineage: **REEPLANE-W6** (new w=6 decode; extends R143 REEPLANE / R148 REESTREAM)
- Model: Gemma 3 1B IT (`gemma-3-1b-it-rawcodec.spsa`, raw codec; tied lm-head
  `model.embed_tokens.weight` [vocab=262144, hidden=1152] bf16; **34 exponents → w=6**)
- Verdict: **GO** — lossless streaming lm-head on Gemma proven bit-identical to
  resident, and the full generation token sequence is identical end-to-end.

## Scope

Close the R149a-Gemma blocker. R149a proved the streaming lm-head lossless on
real **w=5** Llama weights but was BLOCKED on Gemma: its embedding is **w=6** (34
distinct bf16 exponents) and the streaming decode was hard-wired to w=5
(`decode16_w5_into`, `vqtbl2q` 32-entry palette). R149b adds a w=6 SIMD decode and
threads the index width `w` through the sidecar, then re-runs the exact gate
R149a-Gemma could not reach.

## Hypothesis

A w=6 NEON decode that gathers exponents from a 64-entry palette via `vqtbl4q_u8`
reproduces the scalar bit-plane decode bit-for-bit, and once `w` is recorded in the
sidecar, Gemma 3 1B generates the **identical token sequence** with the streaming
lm-head as with the resident path.

## Method / changes

- **`decode16_w6_into`** (`rtc-codec/src/bitplane.rs`): 16-wide w=6 decode. w=6
  bit-packing period is 4 indices = 24 bits = 3 bytes ⇒ 16 indices = 12-byte group
  stride. Index-extraction tables derived for the MSB-first stream
  (`bidx_hi=[0,0,1,2,3,3,4,5,6,6,7,8,9,9,10,11]`, shift `10−(6j%8)`, mask `0x3f`).
  Exponents gathered from a **64-entry** palette with `vqtbl4q_u8` (a single TBL
  over 4 registers — `vqtbl2q`'s 32-entry table cannot reach indices ≥ 32). bf16
  reconstruction identical to w=5. Byte-aligned scalar tail.
- **`decode_bitplane_row_into`** (codec dispatcher): single width-dispatch entry
  point — w=5 → REEPLANE, w=6 → REEPLANE-W6, else scalar. The runtime stays
  width-agnostic (kernel selection lives in the codec crate).
- **Sidecar width propagation** (`bitplane_stream.rs`): `write_lmhead_sidecar`
  rejects raw-fallback (palette > 64), accepts `w ∈ 1..=6`, requires
  `hidden·w % 8 == 0`, and writes header **v2** (`…, palette_len, w, palette`);
  `stream_lmhead_from_sidecar` reads `w` (v1 ⇒ w=5 back-compat); `streaming_bitplane_gemv`
  takes `w`, `row_idx = hidden·w/8`, decodes via the dispatcher.
- **No change to `gemma/api.rs`** — the `RLLM_STREAM_LMHEAD` branch already calls
  `stream_lmhead_from_sidecar`.

## Results — GO

**Unit (bit-exact, no model):**
- `decode16_w6_matches_scalar_bit_for_bit` — w=6 SIMD == scalar `BitplaneCodec::decode`
  across n ∈ {34,35,47,48,64,80,96,1000,4096,4099,65536} (SIMD/scalar boundary + tails).
- `decode_bitplane_row_dispatch_matches_scalar` — dispatcher lossless for w=5 and w=6.
- `w6_sidecar_streams_equal_to_reference` — v2 sidecar (hidden=1152, w=6) streams ==
  scalar decode+dot reference, bit-for-bit.
- Suites: **rtc-codec 48 passed**, **rllm-runtime lib 294 passed**, 0 failed; clean
  build, 0 warnings.

**Real Gemma 3 1B, targeted lossless (`r149b_gemma_streaming_lmhead_lossless`, #[ignore]):**
```
R149b OK: Gemma (w=6) streaming lm-head == resident, 262144 logits bit-identical
```
All **262144** logits bit-identical (exact `assert_eq!`) — REEPLANE-W6 streaming
lm-head == resident bf16 GEMV on the real Gemma embedding.

**Real Gemma 3 1B, full-generation identical-token gate (the R149a-Gemma blocker):**
`gemma-test`, prompt "The capital of France is", 16 greedy tokens, resident vs
`RLLM_STREAM_LMHEAD=/tmp/gemma1b-lmhead.sidecar`:
```
resident : [9079, 236761, 108, 818, 7488, 3207, 528, 7001, 563, 9079, 236761, 108, 818, 1346, 4913, 3207]
streaming: [9079, 236761, 108, 818, 7488, 3207, 528, 7001, 563, 9079, 236761, 108, 818, 1346, 4913, 3207]
RESULT: IDENTICAL
```
16/16 tokens identical through the actual wired path. Sidecar = 504 MB (idx 864 B/row
+ residual 1152 B/row over 262144 rows).

## Analysis

The blocker was purely the w=5-only decode meeting a w=6 real tensor; the kernel,
sidecar plumbing, and Gemma branch were already correct (R149a). The w=6 decode is
not a hack — w=6 is *cleaner* than w=5 (period 4 vs 8), and `vqtbl4q` is the natural
64-entry sibling of `vqtbl2q`. The lossless-logits proof (262144/262144) logically
entails identical argmax → identical tokens; the end-to-end generation gate confirms
it through the real wiring.

Deferred, on purpose:
- **Speed**: Gemma 1B fits RAM, so streaming the lm-head is *not* faster here (it
  re-reads per token). The capacity-bound win (model > RAM, cold device) is a later
  phase, per R147/R148. **No speed claim is made.**
- Streaming the transformer projections (lm-head only).

## Decision

**GO** — REEPLANE-W6 makes the streaming lm-head lossless and width-general (w∈{5,6}
SIMD, any w≤6 scalar). Gemma 3 1B's identical-token generation gate is green. The
R149a-Gemma blocker is closed.

## Next

- Capacity-bound speed demo for Gemma in the > RAM / cold regime (where REESTREAM is
  designed to win); stream the transformer projections, not just the lm-head.

## Verification status

- [x] w=6 SIMD == scalar, bit-exact (unit, all sizes).
- [x] Dispatcher lossless for w=5 and w=6.
- [x] v2 sidecar round-trip (w=6 synthetic) bit-exact.
- [x] Real Gemma w=6 streaming lm-head == resident, 262144 logits bit-identical.
- [x] Real Gemma full-generation token sequence identical (resident vs streaming).
- [x] rtc-codec 48 / rllm-runtime lib 294 green; 0 warnings.
