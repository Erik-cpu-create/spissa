# R158b — rANS embedding to bf16-resident: RAM 3.17 → 2.26 GB, now BEATS bf16 (GO)

- Date: 2026-06-20
- Model: Gemma 3 1B IT, packed `--codec rans` vs bf16 rawcodec; `gemma-test`, peak RSS via `/usr/bin/time -l`
- Verdict: **GO** — routing the rANS (non-raw bf16) embedding to a resident **bf16** table
  (604 MB) instead of the **f32** fallback (1.2 GB) drops full-model peak RSS from
  **3.17 GB → 2.26 GB**, now **below bf16's 2.34 GB** — lossless (token-identical), no
  regression on the bf16/q8 paths.

## Diagnosis (R158a → here)

The 3.17 GB (R157c, worse than bf16) was NOT the body (already chunk-decoded via
`with_decoded_chunk`) — it was the tied **embedding** forced to **f32** (262144×1152×4 =
1.2 GB) in `resolve_gemma_embedding`, because `with_raw_tensor` only fires for raw codecs
so rANS fell to the f32 fallback. Plus the whole-tensor decode's 1.8 GB transient.

## Change (additive, low-risk)

- `LazyRllmModel::decode_tensor_raw_bytes` — decode a tensor to its bf16 bytes without the
  f32 conversion `decode_tensor` does.
- `resolve_gemma_embedding`: raw bf16 → zero-copy (None/None); **non-raw bf16 (rANS) →
  decode ONCE to resident bf16 bytes (604 MB)**; non-bf16 (q8) → keep f32. (4-tuple return.)
- `GemmaEmbedCtx`/`GemmaChatSession` carry `embedding_bf16`; `gemma_embed_input` and
  `gemma_lm_head` try the resident bf16 (via `gemma_embed_lookup_bf16` /
  `lm_head_logits_parallel_bf16`) before the f32 / `with_raw_tensor` paths. f32 and q8
  paths untouched.

## Results — GO

```
                 peak RSS     tokens                              lossless
bf16 rawcodec    2.34 GB      [9079,236761,108,818,7488,...]       -
rANS (R158b)     2.26 GB  ✅  [9079,236761,108,818,7488,...]       yes (identical)
rANS (R157c)     3.17 GB      (same tokens)                        yes
```
peak transient 1.81 GB → 16 KB (no whole-tensor f32 materialize). bf16 rawcodec
unchanged (tokens identical, 1.71 tok/s). rllm-runtime lib 296 green, 0 warnings.

## Analysis (honest)

- **The thesis flipped from negative to positive:** rANS went from WORSE than bf16 (3.17)
  to BETTER (2.26 < 2.34), lossless. The −900 MB came from killing the f32 table (1.2 GB)
  and the whole-tensor transient (1.8 GB).
- **Win magnitude on 1B is modest (80 MB below bf16)** because the embedding (604 MB,
  the biggest single tensor) is now held **uncompressed bf16-resident** — not block-decoded.
  The body is compressed/chunk-decoded. To get the full ~700 MB win, the embedding must
  also be block-decoded (never materialized) — R158c.
- Speed unchanged (~0.27 tok/s, decode-per-token) — R158 is the RAM axis; the >RAM regime
  is where this matters (a model whose bf16 doesn't fit but rANS does).

## Next (R158c)

- Block-decode the embedding/lm-head from the container (compressed-resident, never
  materializing the 604 MB bf16) → drop another ~200 MB toward ~2.0 GB. Then the >RAM demo.

## Verification status

- [x] rANS peak RSS 3.17 → 2.26 GB, below bf16 2.34 GB.
- [x] Lossless (token-identical to bf16); bf16/q8/raw paths unchanged (bf16 rawcodec verified).
- [x] rllm-runtime lib 296 green, 0 warnings.
