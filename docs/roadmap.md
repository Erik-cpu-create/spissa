# RLLM Roadmap

## Phase 0 — Project Skeleton ✅

**Status:** Complete

**Deliverables:**
- [x] Cargo workspace with 3 crates
- [x] CLI skeleton with clap (all commands stubbed)
- [x] Basic structs: `RllmHeader`, `TensorMeta`, `ChunkMeta`, `DType`
- [x] `rtc-raw-v1` codec with round-trip verification
- [x] Unit test framework (12 tests passing)
- [x] README and documentation

**Acceptance Criteria:**
- [x] Project builds without errors
- [x] All tests pass
- [x] CLI help works for all commands

---

## Phase 1 — RLLM Container v1

**Status:** 🔜 Next

**Deliverables:**
- [ ] Write .rllm header to file
- [ ] Write global metadata (JSON)
- [ ] Write tensor directory
- [ ] Write chunk directory
- [ ] Write raw chunks (using rtc-raw-v1)
- [ ] Read .rllm file back
- [ ] `rllm inspect` command (read and display metadata)

**Acceptance Criteria:**
- [ ] Can create .rllm file from sample tensors
- [ ] Can inspect file and list tensors/chunks
- [ ] Can decode raw chunks back to original bytes
- [ ] Round-trip: write → read → verify checksums match

---

## Phase 2 — RTC Codec v1

**Status:** ✅ Complete

**Deliverables:**
- [x] Implement `rtc-rle-v1` codec
- [x] Implement `rtc-huff-v1` in-house byte-level Huffman codec
- [x] Codec selection logic (try multiple, pick best)
- [x] Per-chunk verification during packing
- [x] Benchmark compression ratios on sample data and Pythia-70M

**Acceptance Criteria:**
- [x] `decode(encode(tensor)) == tensor` for all test tensors
- [x] RLE compresses zero-filled tensors effectively
- [x] Huffman improves Pythia-70M compression versus raw+RLE
- [x] Codec selection picks smallest valid output
- [x] Fallback to raw when compression makes data larger

---

## Phase 3 — Pack/Unpack/Verify

**Status:** 🔜 Future

**Deliverables:**
- [ ] `rllm pack` with real tensor input and codec selection
- [ ] `rllm unpack` to restore original tensors
- [ ] `rllm verify` with bit-identical check
- [ ] Progress reporting during pack/unpack

**Acceptance Criteria:**
- [ ] Original files and unpacked files are byte-identical
- [ ] SHA-256 checksums match
- [ ] Compression ratio reported honestly

---

## Phase 4 — Safetensors Import

**Status:** 🔜 Future

**Deliverables:**
- [ ] Read safetensors metadata (JSON header)
- [ ] Read tensor bytes from safetensors
- [ ] Pack safetensors into .rllm
- [ ] Unpack back to safetensors-compatible layout
- [ ] Verify hashes match

**Acceptance Criteria:**
- [ ] Small safetensors model can be packed/unpacked losslessly
- [ ] Tensor names, shapes, and dtypes preserved
- [ ] Checksums match original safetensors

---

## Phase 5 — Memory-First Runtime Foundation

**Status:** ✅ Phase 5A–5E runtime foundations complete through tokenizer-backed RAMA text smoke path

**Deliverables:**
- [x] Full-decode `.rllm` runtime loader
- [x] Runtime tensor conversion to f32 (fp16, bf16, fp32, integers)
- [x] Minimal tensor operations (matmul, add, etc.)
- [x] Embedding lookup
- [x] Linear layer
- [x] RMSNorm or LayerNorm
- [x] GELU and softmax primitives
- [x] Simple attention mechanism
- [x] MLP (feed-forward network)
- [x] Deterministic argmax sampling primitive
- [x] Sampling (temperature, top-p)
- [x] Metadata-only `.rllm` open path for low-RAM modes
- [x] Memory budget accounting with over-budget detection
- [x] Layer-stream/tile-stream dry-run memory planner
- [x] CLI flags: `--memory-budget`, `--ctx`, `--mode full-decode|layer-stream|tile-stream`, `--dry-run`
- [x] Chunk-scoped streaming linear kernel that matches full-decode linear output
- [x] Transient decode/scratch budget verification for streaming linear chunks
- [x] Streaming MLP sub-block that matches full-decode MLP output
- [x] Budgeted intermediate activation for streaming MLP
- [x] Streaming attention/QKV projection that matches full-decode QKV + attention + output projection baseline
- [x] Budgeted fused QKV, split Q/K/V, and attention output activations for streaming attention
- [x] Streaming pre-norm transformer block skeleton that matches a full-decode block baseline
- [x] Budgeted layernorm, attention output, and MLP output activations inside the streaming block
- [x] Tiny one-block next-token smoke path (`embedding → block → final LN → lm_head → sample`)
- [x] Streaming embedding lookup and LM head projection over chunked `.rllm` tensors
- [x] GPT-NeoX/Pythia-style rotary embeddings (`rotary_pct`, `rotary_emb_base`, position offset)
- [x] KV-cache primitive with append/capacity validation and cached causal attention
- [x] Streaming attention runtime options for rotary + KV-cache reuse
- [x] RAMA Architecture spec for the original brain-inspired memory-first direction
- [x] First executable cached generation loop: tiny multi-step token-ID generation
- [x] Explicit prefill and decode-step APIs with rotary position offsets
- [x] `ContextEchoState` per-layer KV-cache container for generation state
- [x] Incremental cached tiny generation matches full-context recomputation for each generated token
- [x] Multi-layer token-ID stack with per-layer ContextEcho caches
- [x] Prefill/decode/generate APIs over all configured streaming blocks
- [x] Multi-layer cached generation matches full-context recomputation for each generated token
- [x] GPT-NeoX/Pythia adapter that infers standard tensor names/shapes from `.rllm` metadata
- [x] Owned prepared generation stack with decoded norm/bias vectors and generated-token execution helper
- [x] Optional original `config.json` field persistence in `.rllm` global metadata
- [x] GPT-NeoX/Pythia auto-build path that derives `num_heads`, rotary config, layer norm eps, and context length from persisted metadata
- [x] Optional tokenizer vocabulary/config metadata persistence in `.rllm` global metadata
- [x] Tokenizer-backed GPT-NeoX/Pythia text boundary over the prepared token-ID stack
- [x] CLI-facing text smoke path via `rllm run --prompt ... --max-new-tokens ...`
- [x] Phase 6 RAMA layer-decode GPT-NeoX/Pythia path that stores layer names only, decodes per-layer norm/bias params just-in-time, budgets active layer params, and matches the prepared stack baseline

**Acceptance Criteria:**
- [x] `rllm run <model.rllm>` can full-decode tensors into runtime memory
- [x] `rllm run <model.rllm> --memory-budget 100mb --ctx 1024 --mode tile-stream` reports planned peak RAM without decoding all tensors
- [x] Streaming linear can decode a weight chunk, accumulate matmul, and release transient buffers while matching the full-decode baseline
- [x] Streaming MLP can run `dense_h_to_4h → GELU → dense_4h_to_h` with chunked weights and release intermediate budget
- [x] Streaming attention can run `query_key_value → split Q/K/V → scaled dot-product attention → dense` with chunked weights and release intermediate budget
- [x] Streaming block can compose `LN → attention → residual → LN → MLP → residual` and match the full-decode baseline
- [x] Same tiny prompt + same sampling config produces the same next token
- [x] Tiny next-token output is deterministic for argmax sampling
- [x] Can run a tiny custom one-block transformer smoke model
- [x] Rotary Q/K values match an independent GPT-NeoX-style baseline and preserve non-rotary tail dims
- [x] Cached next-token attention matches the last row of full causal attention
- [x] Streaming attention can apply rotary, attend over past KV cache, append current K/V, and match full-decode last-token baseline
- [x] `docs/rllm-rama-architecture.md` defines originality rules, RAMA/ERIK naming, and the memory-first runtime boundary; the previous ECHO/EMBER doc is superseded
- [x] Tiny prefill path fills ContextEcho KV-cache and matches full-context rotary baseline
- [x] Tiny decode-step path consumes one generated token, advances rotary position, appends K/V, and matches full-context recompute
- [x] Tiny multi-step generation returns deterministic generated token IDs, step logits, and resident ContextEcho bytes
- [x] Multi-layer generation stack keeps one KV-cache per configured layer and validates every layer cache advances together
- [x] Multi-layer prefill/decode/generate logits match full-context recompute across all layers
- [x] GPT-NeoX adapter detects contiguous layers, validates weight shapes, decodes small params, and can generate through the prepared stack
- [x] Existing `.rllm` metadata stays backward-compatible when `model_config` is absent
- [x] Pack path can persist sibling or explicit HuggingFace `config.json` fields for runtime adapters
- [x] Existing `.rllm` metadata stays backward-compatible when tokenizer metadata is absent
- [x] Pack path can persist sibling or explicit HuggingFace `tokenizer.json` vocabulary metadata
- [x] Prepared GPT-NeoX text generation encodes prompt text, executes token-ID generation, and decodes generated/full text through the persisted tokenizer metadata
- [x] Layer-decode GPT-NeoX/Pythia generation matches prepared-stack generated token IDs, full token sequence, logits, and context memory bytes

**Next implementation slice:**
- Begin Phase 7 tile-decode / fused decode+matmul under RAMA: reduce large matrix weight memory below the current streaming chunk window while matching the Phase 6 layer-decode baseline.

**Measured local Pythia-70M planning examples:**
- 32MiB chunks (`pythia-70m-huff-32mb.rllm`): 120.60 MiB compressed, planned peak 139.46 MiB → over 100MiB budget
- 16MiB chunks (`pythia-70m-huff-16mb.rllm`): 120.60 MiB compressed, planned peak 76.76 MiB → within 100MiB budget
- 1MiB chunks (`pythia-70m-huff.rllm`): 120.82 MiB compressed, planned peak 19.08 MiB → within 100MiB budget

---

## Phase 6 — Layer Decode Runtime

**Status:** ✅ Partial — GPT-NeoX/Pythia RAMA layer-param decode path complete

**Deliverables:**
- [x] Runtime reads compressed .rllm through metadata-only `LazyRllmModel`
- [x] Decode only needed per-layer norm/bias params for the active layer
- [x] Release per-layer params after layer completes
- [x] Track active layer-param bytes in `MemoryBudget` while executing each layer
- [x] CLI `--prompt` path uses the Phase 6 RAMA layer-decode GPT-NeoX/Pythia runner
- [ ] Full large-weight layer materialization mode for architectures that do not use chunked streaming kernels

**Acceptance Criteria:**
- [x] Resident parameter memory lower than the Phase 5 prepared stack because per-layer params are no longer stored for every layer
- [x] Generated token IDs, token sequence, logits, and context memory match the prepared stack baseline in tests
- [x] Transient budget returns to zero after generation; active per-layer param bytes are reserved/released during each layer
- [ ] Compare against a full production reference implementation for real-model logits

---

## Phase 7 — Tile Decode Runtime

**Status:** 🔜 Future

**Deliverables:**
- [ ] Matrix weight chunks aligned to matmul tiles
- [ ] Decode tile → multiply → accumulate → release
- [ ] `--mode tile-decode` option
- [ ] Fused decode + matmul (optional)

**Acceptance Criteria:**
- [ ] Output numerically identical to full-decode mode
- [ ] Peak memory lower than layer-decode mode
- [ ] Performance is acceptable (document trade-offs)

---

## Phase 8 — Real Model Support

**Status:** 🔜 Future

**Deliverables:**
- [ ] Support one small real architecture (e.g., TinyLlama-class)
- [ ] Load tokenizer (SentencePiece or BPE)
- [ ] Load model config
- [ ] Run prompt and stream tokens
- [ ] Compare logits with reference implementation

**Acceptance Criteria:**
- [ ] Can run a real small model from .rllm
- [ ] Can verify decoded weights
- [ ] Logits match reference within tolerance
- [ ] Token generation is deterministic

---

## Future Enhancements

- [ ] SIMD-optimized codecs (AVX2, NEON)
- [ ] Memory-mapped compressed loading
- [ ] GPU support for inference
- [ ] Additional codecs (delta, bitplane, entropy)
- [ ] Multi-GPU sharding
- [ ] Quantization-aware compression (lossy-optimize mode)
- [ ] Model hub integration (download .rllm files)

---

## Success Criteria

### v1 (MVP)
1. Packs model tensors into .rllm
2. Unpacks them exactly (bit-identical)
3. Verifies lossless correctness
4. Shows honest compression metrics
5. Runs a tiny model from .rllm
6. Supports full-decode runtime

### v2 (Stronger)
1. Supports safetensors import/export
2. Supports layer-wise decode
3. Reduces peak RAM compared to full-decode
4. Can run a small real transformer model

### v3 (Unique)
1. Supports tile-wise compressed inference
2. Avoids full decompression of large tensors
3. Has fused decode + matmul kernels
4. Can run useful local LLMs with lower peak RAM
