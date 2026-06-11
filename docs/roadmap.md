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

## Phase 5 — Toy Inference Runtime

**Status:** 🔜 Future

**Deliverables:**
- [ ] Minimal tensor operations (matmul, add, etc.)
- [ ] Embedding lookup
- [ ] Linear layer
- [ ] RMSNorm or LayerNorm
- [ ] Simple attention mechanism
- [ ] MLP (feed-forward network)
- [ ] Sampling (temperature, top-p)
- [ ] Streaming token output

**Acceptance Criteria:**
- [ ] Same prompt + same seed produces same tokens
- [ ] Output is deterministic
- [ ] Can run a tiny custom transformer model

---

## Phase 6 — Layer Decode Runtime

**Status:** 🔜 Future

**Deliverables:**
- [ ] Runtime reads compressed .rllm
- [ ] Decode only needed layer weights
- [ ] Release weights after layer completes
- [ ] Track peak memory usage
- [ ] `--mode layer-decode` option

**Acceptance Criteria:**
- [ ] Peak memory lower than full-decode mode
- [ ] Output identical to full-decode mode
- [ ] Memory tracking is accurate

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
