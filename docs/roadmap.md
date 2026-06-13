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
- [x] Phase 7 RAMA tiled runtime routing: MLP, attention, transformer blocks, tiny/RAMA/GPT-NeoX final projection heads use the fused tile-linear path while preserving baselines
- [x] Phase 7.6 actual local Pythia-70M token generation with persisted config/tokenizer metadata and macOS max RSS measurement
- [x] Phase 7.7 fixed-token HF/PyTorch logits comparison against local Pythia-70M; top-1/top-10 match after GPT-NeoX parallel residual metadata + per-head QKV split fixes

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
- [x] Fused tile-linear matches full-decode linear output and succeeds under a transient budget where full f32 chunk scratch fails
- [x] Tiled MLP/attention routing succeeds under transient budgets where full f32 chunk scratch fails, while block/tiny/GPT-NeoX logits/token tests still pass
- [x] Repacked local Pythia-70M with explicit `config.json` + `tokenizer.json`, generated one token from `Hello`, and measured process RSS via `/usr/bin/time -l`
- [x] Fixed-token RLLM logits match HuggingFace/PyTorch reference top-1/top-10 on tested Pythia-70M prompts; see [`docs/phase77-hf-logits-comparison.md`](phase77-hf-logits-comparison.md)

**Next implementation slice:**
- Continue Phase 7 from the Phase 7.12B projection-reuse foundation: generic eight-row tiled-linear accumulation improves the measured Pythia-160M projection bottleneck without model-specific code. Next either pursue another measured dense-projection slice or start Phase 8 LLaMA-family adapter work if architecture breadth becomes the priority.

**Measured local Pythia-70M planning examples after Phase 7 fused tile scratch cap:**
- 32MiB chunks (`pythia-70m-huff-32mb.rllm`): 120.60 MiB compressed, planned peak 75.48 MiB → within 100MiB budget
- 16MiB chunks (`pythia-70m-huff-16mb.rllm`): 120.60 MiB compressed, planned peak 44.77 MiB → within 100MiB budget
- 1MiB chunks (`pythia-70m-huff.rllm`): 120.82 MiB compressed, planned peak 16.04 MiB → within 100MiB budget

**Measured local Pythia-70M Phase 7.6 release benchmark matrix:**
- Repacked artifact (`pythia-70m-phase76-16mb.rllm`, ignored local file): 120.46 MiB compressed, 16MiB chunks, persisted `gpt_neox` config + `hf-bpe` tokenizer metadata
- Command: `python3 scripts/phase76_release_rss_benchmark.py --tokens 1,4,8,16 --ctx 128,512,1024 --memory-budget 100mb`
- Matrix: 12/12 runs succeeded for `ctx=128/512/1024` and `max-new-tokens=1/4/8/16`
- Output behavior after Phase 7.7 HF-fidelity fixes: prompt token `[12092]`, first generated token `[13]`, generated text starts `Hello,`; 16-token run produces `Hello, I'm trying to get the name of the phone number in the phone number`
- Runtime range: 4.47–83.41s total, ~4.47–5.29s/token in release
- Memory range: 88.62–94.62 MiB max RSS, 87.23–93.23 MiB peak memory footprint, 48.00 MiB peak tracked transient budget
- Planner comparison: 16MiB tile-stream planned peak is 44.77 MiB; measured release RSS peak is ~2.11× planner because RSS includes process/runtime overhead outside `MemoryBudget`
- Full table: [`docs/phase76-release-rss-benchmark.md`](phase76-release-rss-benchmark.md)

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
- [x] Compare against a full production reference implementation for fixed-token real-model logits

---

## Phase 7 — Tile Decode Runtime

**Status:** ✅ Partial — tiled MLP/attention/RAMA generation routing, tile-stream planner scratch cap, release RSS benchmark matrices, fixed-token HF logits comparison, Phase 7.8 tile-block real-artifact benchmark, Phase 7.9A RAMA trace profiler, Phase 7.9B embedding row recall, Phase 7.9C low-ram-fast raw/tile-block profile, Phase 7.9D real long-prompt benchmark, Phase 7.9E RAMA chunked prefill optimization, Phase 7.10A row-span linear accumulation, Phase 7.10B RAMA prefill homeostasis, Phase 7.11 Pythia-160M scale/window validation, Phase 7.12A generic shape/budget-aware prefill policy, and Phase 7.12B generic eight-row projection reuse complete

**Deliverables:**
- [x] Bounded f32 tile scratch for linear matmul accumulation
- [x] Decode chunk → convert tile → multiply/accumulate → release tile scratch
- [x] `--mode tile-decode` / `tile-stream` planner estimates fused tile scratch instead of full f32 chunk scratch
- [x] Route MLP/attention/transformer block/tiny/RAMA/GPT-NeoX generation projections through tiled linear kernels
- [x] Run actual local Pythia-70M token generation with measured max RSS
- [x] Compare fixed-token local Pythia-70M logits against HuggingFace/PyTorch reference
- [x] Add lossless range-decode API foundation: `DecodeRange`, codec `decode_range`, native raw range decode, runtime `with_decoded_chunk_range`, and container compressed byte-range reads
- [x] Add per-range integrity metadata foundation: optional `ChunkRangeMeta` records original/compressed byte spans and SHA-256s; writer/runtime tests verify range metadata and checksum mismatch detection
- [x] Add opt-in `rllm pack --range-checksum-size` path for identity-mapped raw chunks; default artifacts remain unchanged
- [x] Add opt-in `rllm pack --tile-block-elements` path that packs each tensor into dtype-sized tile/block chunks while preserving existing chunk-level checksums
- [x] Benchmark real local Pythia-70M tile-block artifact (`--tile-block-elements 65536`): 12/12 RSS matrix succeeded at 18.39–22.64 MiB RSS and 386 KiB tracked transient peak
- [x] Add opt-in `rllm run --rama-trace <path>` profiler for generation: records chunk read/checksum/decode/compute-closure timing JSON and summary by phase
- [x] Add RAMA embedding row recall: `streaming_embedding_lookup_from_model` recalls only touched token row chunks/ranges instead of scanning the full embedding table
- [x] Add explicit pack codec policy (`--codec auto|raw|rle|huff`) and raw/tile-block low-ram-fast benchmark harness for compute-ready artifacts
- [x] Add opt-in runtime integrity policy (`--rama-integrity strict|verify-once`) so pre-verified artifacts can verify each chunk once per process instead of every recall
- [x] Benchmark local Pythia-70M Phase 7.9C raw/tile-block profile: strict mode reached 0.58 average seconds/token; verify-once reached 0.35 average seconds/token / 3.26 average tok/s / 4.35 best tok/s with RSS still 19.17–23.36 MiB
- [x] Benchmark actual long prompts with deterministic `--token-ids` input lengths 1/128/512/1024 under `ctx=2048`: short prompt remains 4.30 tok/s at 20.67 MiB RSS, but 512-token and 1024-token prompts expose prefill/context bottlenecks at 0.30 tok/s / 44.98 MiB RSS and 0.15 tok/s / 70.84 MiB RSS for 16 generated tokens
- [x] Run long-prompt HF/PyTorch logits parity on 128-token and 512-token fixed prompts; both preserve top-1/top-5/top-10 parity with max abs diff 0.01708984 and 0.02746582 respectively
- [x] Add low-overhead aggregate RAMA timing split for prefill vs decode-step vs lm-head under real long prompts
- [x] Add opt-in RAMA chunked prefill (`--rama-prefill-chunk-tokens`) that bounds real-prompt activation windows and skips intermediate lm-head projection while preserving full-vocab logits
- [x] Optimize tiled-linear accumulation by processing contiguous row spans instead of per-weight division/modulo in the hot loop while preserving row-major accumulation order and exact tested logits
- [x] Run broader post-7.10A matrix for `input_tokens=1,128,512,1024` × `new_tokens=1,4,16`
- [x] Sweep RAMA prefill chunk windows for 512/1024-token prompts and make the measured 32-token window the CLI default
- [x] Add deeper RAMA prefill timing inside embedding recall, layer-param recall, attention, MLP, norms, and residuals; Phase 7.10C shows MLP dominates prefill (57.7–61.4%) and attention is secondary (33.8–39.8%)
- [x] Optimize measured RAMA prefill MLP bottleneck with four-prompt-row accumulation reuse while preserving bounded active memory and exact tested logits
- [x] Split attention timing into QKV projection, rotary/KV append, score/context, and output projection; Phase 7.10E shows score/context dominates 1024-token attention before optimization
- [x] Optimize measured attention score/context bottleneck with K/V row-slice reuse while preserving bounded active memory and tested logits semantics
- [x] Scale-validate RAMA on Pythia-160M before LLaMA-family architecture expansion; Phase 7.11A proves the existing GPT-NeoX/Pythia path packs, verifies, runs, matches HF top-k, and completes the timing/RSS matrix without model-specific code
- [x] Sweep Pythia-160M prefill chunk/window and memory-budget behavior before adding more kernel complexity; Phase 7.11B recommends chunk=64 as the low-RAM-safe 160M override and chunk=128 as the speed-biased setting
- [x] Implement a generic shape/budget-aware prefill policy with low-ram/speed modes, preserving Pythia-70M-like 32-token low-RAM defaults while selecting 64/128-token windows for larger Pythia-160M-like shapes when budget allows
- [x] Optimize measured Pythia-160M MLP/QKV projection bottlenecks with generic eight-row tiled-linear accumulation reuse while preserving the existing 4-row/scalar tails, bounded transient memory, and tested output semantics
- [ ] Optional next dense-projection slice if fresh timing identifies a safe generic candidate
- [ ] Optional low-RAM parallel row-span accumulation if short-prompt decode/lm-head remains the priority
- [ ] Codec-level range decode + multiply in one step

**Acceptance Criteria:**
- [x] Fused tile-linear output matches full-decode linear baseline
- [x] Fused tile-linear succeeds under a memory budget where full f32 chunk scratch fails
- [x] Tiled MLP and attention succeed under budgets where full f32 chunk scratch fails
- [x] Too-small tile scratch budget fails without leaking budget state
- [x] Tiny/RAMA/GPT-NeoX generation tests still match their full-context/prepared-stack baselines after tiled routing
- [x] Performance trade-off documented for release benchmark matrix: low RSS (88.62–94.62 MiB max RSS under `100mb` internal budget) but slow CPU runtime (~4.47–5.29s/token)
- [x] HF/PyTorch logits comparison passes for tested fixed-token prompts: top-1/top-10 match, max abs diff ≤ 0.00769043
- [x] Range-decode foundation tests pass for codec bounds, raw native range, runtime raw range budget, RLE full-decode fallback, and container chunk byte ranges
- [x] Range checksum tests pass for legacy chunk metadata, writer-generated identity range checksums, out-of-bounds range rejection, runtime original/compressed range verification, and corruption detection
- [x] Tiny raw safetensors fixture packs with `--range-checksum-size`, `inspect` reports 9 range checksums, and `verify` reports `LOSSLESS VERIFIED`
- [x] Tiny raw safetensors fixture packs with `--tile-block-elements 64`, producing 5 chunk-aligned blocks; `inspect` reports `Chunks: 5` and `Range checksums: 9`; `verify` reports `LOSSLESS VERIFIED`
- [x] Local Pythia-70M tile-block artifact (`1,520` chunks, 127,330,065 compressed bytes) passes the `ctx=128,512,1024` × `tokens=1,4,8,16` RSS matrix under `100mb`; max RSS 22.64 MiB; tested HF logits parity preserved (`top1_match=True`, max abs diff 0.00769043 for `[12092,13]`)
- [x] RAMA trace profiler smoke on the same artifact generates `Hello,`, writes JSON, and shows `chunk_decode` as the dominant recorded phase (1,074 events / ~3716 ms), with disk read only ~32 ms
- [x] Embedding row recall preserves `Hello,` and HF parity while reducing `gpt_neox.embed_in.weight` trace events from 1,965 to 5; the 12-row tile-block matrix improved from 5.07 to 2.93 average seconds/token (~1.73× average speedup) with max RSS 22.28 MiB and transient peak 291.68 KiB
- [x] Low-ram-fast raw/tile-block artifact (`rtc-raw-v1`, `--tile-block-elements 65536`, 160.60 MiB local file) verifies losslessly, preserves tested HF top-1 parity, and reduces the 12-row matrix to 0.35 average seconds/token in `verify-once` mode while keeping max RSS 23.36 MiB
- [x] Phase 7.9C trace confirms `chunk_decode` is reduced to ~24 ms across a 16-token run; repeated checksum events are reduced to one per unique chunk in `verify-once`, leaving `chunk_compute_closure` / `embed_out.weight` as the dominant remaining bottleneck
- [x] Phase 7.9D real long-prompt matrix succeeds for 12/12 rows; 1-token prompt + 16 generated tokens reaches 4.301 tok/s at 20.67 MiB RSS, but 512-token prompt + 16 generated tokens drops to 0.300 tok/s at 44.98 MiB RSS and 1024-token prompt + 16 generated tokens drops to 0.148 tok/s at 70.84 MiB RSS
- [x] Phase 7.9D long-prompt logits parity passes for 128-token and 512-token deterministic prompts (`top1_match=True`, top-10 overlap 10/10)
- [x] Phase 7.9E timing split confirms prefill dominates long prompts: 512-token full prefill records 52.08s prefill vs 3.83s decode and 2.58s lm-head; 1024-token full prefill records 106.46s prefill vs 3.82s decode and 2.55s lm-head
- [x] Phase 7.9E chunked prefill with `--rama-prefill-chunk-tokens 64` improves 512-token + 16 generated from 56.43s / 0.284 tok/s / 46.22 MiB RSS to 35.20s / 0.455 tok/s / 34.05 MiB RSS, and 1024-token + 16 generated from 110.29s / 0.145 tok/s / 70.55 MiB RSS to 63.84s / 0.251 tok/s / 44.98 MiB RSS
- [x] Phase 7.9E chunked 512-token logits parity preserves tested HF/PyTorch top-1/top-10 (`max_abs_diff=0.02746582`)
- [x] Phase 7.10A row-span linear accumulation improves the local short prompt + 16 generated row from 4.39s / 3.647 tok/s / 20.61 MiB RSS to 2.25s / 7.101 tok/s / 20.66 MiB RSS; 512-token chunked prefill improves from 35.20s / 0.455 tok/s to 7.08s / 2.259 tok/s; 1024-token chunked prefill improves from 63.84s / 0.251 tok/s to 12.76s / 1.254 tok/s
- [x] Phase 7.10A 512-token chunked logits parity remains `top1_match=True`, top-10 overlap 10/10, max abs diff 0.02746582
- [x] Phase 7.10B broader post-rowspan matrix succeeds for 12/12 rows; short prompt + 16 generated reaches 9.756 tok/s at 20.33 MiB RSS, 512-token + 16 reaches 2.219 tok/s at 33.92 MiB RSS with chunk=64, and 1024-token + 16 reaches 1.117 tok/s at 45.14 MiB RSS with chunk=64
- [x] Phase 7.10B prefill chunk sweep selects 32 tokens as the measured default: 512-token + 16 improves to 2.3495 tok/s / 32.77 MiB RSS and 1024-token + 16 improves to 1.1653 tok/s / 44.91 MiB RSS while reducing peak transient to 794 KiB
- [x] Phase 7.10B default-32 logits are identical to the prior HF-validated 512-token chunk-64 RLLM logits (`max_abs_diff=0.0`, top-10 overlap 10/10); direct HF rerun was blocked because `uv`/`torch` were unavailable in the shell
- [x] Phase 7.10C deep prefill timing succeeds for 512/1024-token prompts with default chunk=32. 512-token + 16 reaches 2.2346 tok/s / 33.22 MiB RSS with 5.38s prefill; 1024-token + 16 reaches 1.2003 tok/s / 44.92 MiB RSS with 11.89s prefill. MLP accounts for 61.4% / 57.7% of prefill, attention for 33.8% / 39.8%, and layer-param recall only ~0.2%.
- [x] Phase 7.10D MLP split timing shows MLP output projection dominates the MLP bucket while GELU is negligible. Larger projection tiles and single-row dot unroll were measured and rejected as regressions. Accepted four-prompt-row accumulation reuse reduces 512-token + 16 from 7.56s / 2.116 tok/s to 5.25s / 3.048 tok/s, and 1024-token + 16 from 13.95s / 1.147 tok/s to 9.12s / 1.754 tok/s; optimized 512-token logits are identical to prior saved default-32 logits (`max_abs_diff=0.0`, top-10 overlap 10/10).
- [x] Phase 7.10E attention split timing shows score/context dominates 1024-token attention (1982.70 ms / 55.6% of attention), QKV projection is second, and rotary/KV append/QKV split are tiny. In-place softmax was measured and rejected as a regression. Accepted K/V row-slice score/context optimization reduces 512-token + 16 from 4.95s / 3.232 tok/s to 4.56s / 3.509 tok/s, and 1024-token + 16 from 8.63s / 1.854 tok/s to 7.12s / 2.247 tok/s while keeping RSS effectively bounded (~32.9/45.0 MiB).
- [x] Phase 7.11A Pythia-160M scale validation packs a raw/tile-block artifact (`184` tensors / `3366` chunks / `367 MiB`), verifies `374,977,752` bytes losslessly, runs `Hello!` from token `[12092]`, and passes HF/PyTorch fixed-token top-k parity (`top1_match=true`, top-10 overlap `10/10`, max abs diff `0.02246094`). The conservative matrix succeeds for input tokens `1,128,512,1024` × new tokens `1,4,16`; 1024 + 16 reaches `31.08s` / `0.515 tok/s` / `99.47 MiB RSS` with tracked transient `1.04 MiB`. Existing RAMA optimizations transfer without model-specific 160M code.
- [x] Phase 7.11B Pythia-160M chunk/window sweep (`512/1024` prompt tokens, `16` generated, chunks `8/16/32/64/128/256`) shows larger chunks improve 160M throughput until RSS/transient trade-offs dominate. For 1024 + 16, chunk=32 gives `31.23s` / `0.512 tok/s` / `93.75 MiB RSS`; chunk=64 gives `28.22s` / `0.567 tok/s` / `99.02 MiB RSS`; chunk=128 gives `26.65s` / `0.600 tok/s` / `100.06 MiB RSS`; chunk=256 gives only `0.609 tok/s` but jumps to `107.20 MiB RSS`. Memory-budget threshold at chunk=128 requires just under 4 MiB tracked transient (`3840kb` fails; `3968kb` passes), proving `--memory-budget` is a transient cap, not an RSS cap. Recommendation: use chunk=64 for Pythia-160M low-RAM-safe runs, chunk=128 for speed-biased runs with ~100-102 MiB RSS tolerance.
- [x] Phase 7.12A generic shape/budget-aware prefill policy moves the previous measured defaults into runtime logic: low-ram policy chooses 32 for Pythia-70M-like shapes and 64 for Pythia-160M-like shapes, speed policy chooses 128 for 160M-like shapes when budget allows, explicit `--rama-prefill-chunk-tokens` still wins, and `--no-rama-prefill-chunking` still reproduces full-prompt prefill. Unit coverage verifies policy selection, prompt-length clamping, budget downshift, CLI parser aliases, fixed override, and disabled mode.
- [x] Phase 7.12B generic eight-row projection reuse extends the shared tiled-linear accumulation hot loop from 4 prompt-token rows to 8 rows with 4-row/scalar fallbacks. Same-session Pythia-160M 512-token speed-policy timing improves wall time `13.21s -> 12.34s`, prefill `9174.13ms -> 8268.24ms`, MLP `5601.07ms -> 4939.77ms`, and QKV projection `2071.98ms -> 1845.34ms` while tracked transient peak stays `3.79 MiB`. A 1024-token speed-policy confirmation completes at `20.13s` / `0.795 tok/s` / `98.92 MiB RSS` with no model-specific branches.

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
