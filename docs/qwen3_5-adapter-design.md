# Qwen3.5-2B Adapter — Design Document (text-only)

Status: **design / not yet implemented**. Author target: a new `models/qwen/` family in
`rllm-runtime`, packed text-only (vision + MTP dropped). This is the first RLLM family with
**heterogeneous per-layer operators** (linear-attention vs softmax-attention) and the first with
a **persistent recurrent state** in place of a growing K/V cache.

Everything below is grounded in (a) the real safetensors tensor names/shapes of
`Qwen/Qwen3.5-2B`, and (b) the HF reference `modeling_qwen3_next.py` recurrence. Numbers are not
guesses; where a detail still needs a numerical parity check it is flagged **[VERIFY]**.

---

## 1. What this model is

`model_type: qwen3_5`, class `Qwen3_5ForConditionalGeneration`. Practically a **Qwen3-Next**:
hybrid **Gated DeltaNet (linear attention)** + **Gated full attention**, dense FFN (the 2B variant
has *no* MoE — `mlp_only_layers: []`, single `intermediate_size 6144`). Multimodal (vision tower)
and Multi-Token-Prediction (MTP) head exist in the checkpoint but are **out of scope** here.

Text decoder layout (24 layers), from `config.text_config`:

```
6 × ( 3 × GatedDeltaNet→FFN  +  1 × GatedAttention→FFN )      # full attn at layer idx % 4 == 3
```

Key dims: `hidden 2048`, `intermediate 6144` (SwiGLU/silu), `vocab 248320`, `tie_word_embeddings`,
`rms_norm_eps 1e-6`, `max_position_embeddings 262144`, `rope_theta 1e7`,
`partial_rotary_factor 0.25`, `mrope interleaved, section [11,11,10]`.

Linear-attn dims: `linear_num_{key,value}_heads 16`, `linear_{key,value}_head_dim 128`,
`linear_conv_kernel_dim 4`, `mamba_ssm_dtype float32`.
Full-attn dims: `num_attention_heads 8`, `num_key_value_heads 2`, `head_dim 256`,
`attn_output_gate true`.

### Parameter budget (text-only)

~1.88 B params (matches "2B"): embeddings 248320×2048 = **508 M (27%)**, 18 linear-attn blocks
≈ 1.06 B, 6 full-attn blocks ≈ 314 M. bf16 ≈ **3.76 GB**; q8 ≈ 2.0 GB; lossless rANS ≈ 2.4 GB.
(The full 4.55 GB file includes vision + MTP, which we drop.)

### Why this architecture matters for RLLM's edge thesis

The 18 linear-attn layers keep a **constant-size recurrent state** (`[16,128,128] f32 ≈ 1 MB`/layer,
**independent of context length**). Only the 6 full-attn layers grow a K/V cache. At 4 k context the
total context memory is ≈ 100 MB K/V + 18 MB state — versus a same-size all-softmax model that would
grow K/V on all 24 layers. **Long context with small memory** is exactly the on-device win RLLM is
chasing. Per-token compute for the linear layers is also cheap (~14 M MACs/token total) — decode is
dominated by the FFNs and the 248 k-row `lm_head`, not by the recurrence.

---

## 2. Exact math (the new kernels)

### 2.1 Gated DeltaNet — recurrent single-step (decode), per head `h ∈ 0..16`

State per layer: `S ∈ R^{16 × 128 × 128}` (f32), conv state `C ∈ R^{6144 × 3}` (last 3 inputs).

Projections from hidden `x ∈ R^2048` (separate matrices in this checkpoint):

```
qkv = in_proj_qkv · x            # [6144] = q‖k‖v, each [16×128]
z   = in_proj_z   · x            # [2048] output gate, [16×128]
b   = in_proj_b   · x            # [16]   per-head
a   = in_proj_a   · x            # [16]   per-head
```

Short causal conv + activation, then split:

```
qkv = SiLU( depthwise_causal_conv1d(qkv, conv1d.weight[6144,1,4], state=C) )   # no bias
q,k,v = split(qkv, [2048,2048,2048])  → per head [128]
q = l2norm(q_h, eps=1e-6);  k = l2norm(k_h, eps=1e-6)   # per head; v unnormalized
```

Gates (per head, scalar):

```
β_h = sigmoid(b_h)
g_h = exp( -exp(A_log_h) · softplus(a_h + dt_bias_h) )      # decay ∈ (0,1)
```

Recurrence + readout (per head):

```
S_h     = S_h · g_h                       # decay  [128,128]
kvmem   = S_hᵀ · k_h                       # [128]  (sum over key dim)
Δ       = (v_h − kvmem) · β_h              # [128]  delta rule
S_h     = S_h + k_h ⊗ Δ                    # rank-1 update
o_h     = S_hᵀ · q_h                       # [128]  readout
```

Output (gated RMSNorm with the z gate, per head over the 128 value dim) then merge + project:

```
o_h = norm.weight ⊙ ( o_h · rsqrt(mean(o_h²)+1e-6) ) ⊙ SiLU(z_h)     # Qwen3NextRMSNormGated
out = out_proj · concat_h(o_h)                                        # [2048]
```

REE kernel name (provisional, per naming rule): **REEDELTA** for the gated-delta scan.

**Prefill:** the recurrence above is sequential O(T). Correct MVP = run it token-by-token for prefill
too. The fast path is the **chunked** GatedDeltaNet parallel scan (matmul-friendly) — deferred to a
perf phase; keep the recurrent form as the parity oracle.

### 2.2 Gated full attention (layer idx % 4 == 3)

```
qg = q_proj · x → split → query[8×256], gate[8×256]     # attn_output_gate: q_proj is 2× wide (4096)
k  = k_proj · x → [2×256];  v = v_proj · x → [2×256]    # GQA 8 q / 2 kv
query = q_norm(query); k = k_norm(k)                    # RMSNorm over head_dim 256, eps 1e-6
# partial RoPE: rotate first 64 of 256 dims (rotary_dim = 0.25·256), θ=1e7, rotate_half (half-split)
attn = softmax( (query·kᵀ)/√256 + causal_mask ) · v     # standard GQA, scale 256^-0.5
attn = attn ⊙ sigmoid(gate)                             # output gate, elementwise [8×256]
out  = o_proj · attn
```

REE kernel name (provisional): **REEGATE** (gated-attention wrapper around existing SDPA).

For text-only, **mRoPE collapses to ordinary RoPE** (all three position components equal the token
position) — implement as plain partial RoPE on 64 dims. **[VERIFY]** the interleaved section split
`[11,11,10]` reduces to identity for single-stream text.

### 2.3 FFN, norms, head

Standard pre-norm block: `RMSNorm → {attn|delta} → +residual → RMSNorm → SwiGLU(silu) → +residual`.
`lm_head` is **tied** to `embed_tokens` (248320×2048). Final `model.norm`.

---

## 3. Mapping onto RLLM (codebase seams, file:line)

RLLM has **no architecture trait**; each family is a free-standing module under
`crates/rllm-runtime/src/models/{...}` with its own config struct, builder, block fn, and session.
Dispatch is one `match` on a string in the chat CLI. So Qwen3.5 = a **new `models/qwen/` module**
plus a handful of extension points.

| Concern | Extend at | Change |
|---|---|---|
| Arch string normalize | `rllm-import/src/safetensors.rs:140` `normalize_architecture_type` | map `qwen3_5`/`qwen3` → `"qwen3"` |
| Chat dispatch | `rllm-cli/src/commands/chat.rs:81` `match architecture` | add `"qwen3" => qwen_chat(...)` |
| HF config struct | `rllm-import/src/safetensors.rs:41` `HuggingFaceModelConfig` | add `text_config` read of `layer_types`/`full_attention_interval`, `linear_{key,value}_head_dim`, `linear_num_{key,value}_heads`, `linear_conv_kernel_dim`, `partial_rotary_factor`, `attn_output_gate` |
| Persisted config | `rllm-container/src/metadata.rs:150` `ModelConfigMetadata` | mirror those (all `#[serde(default, skip_serializing_if)]` → back-compatible) + copy-through at `safetensors.rs:102` |
| New module | new `models/qwen/` + register `models/mod.rs:1` | `model.rs` (types), `api.rs` (builder + session), `generate.rs` (blocks), `state.rs` (recurrent state) |
| Per-layer dispatch | per-layer loop body `llama/session/mod.rs:303` (precedent: Gemma `is_global_layer` branch `gemma/generate.rs:194`) | branch on `layer_kind[i]` → REEGATE block vs REEDELTA block |
| Softmax attn | reuse `scaled_dot_product_attention_with_cache` `rotary.rs:331` (scale already `1/√head_dim` = `1/√256` ✓); QK-norm precedent `gemma/generate.rs:189-190`; partial rotary via `apply_gpt_neox_rotary_inplace` + `RotaryEmbeddingConfig.rotary_dim` `rotary.rs:8` set to 64 | add the q/gate split + `sigmoid(gate)` modulation |
| Recurrent state | new struct beside `KvCache` `rotary.rs:191`; per-layer `Vec<LayerCache>` allocated like `llama/session/mod.rs:141`, threaded like `caches[i]` `session/mod.rs:349` | `GatedDeltaNetState { s:[16*128*128] f32, conv:[6144*3] f32 }` + `resident_bytes` + checkpoint/restore |
| Decode interface | impl `RamaSessionAdapter` `session.rs:213` (like `LlamaRamaSessionAdapter` `session/mod.rs:664`) to reuse `RamaChatSession` `session.rs:287` | rollback must checkpoint **state**, not just truncate K/V |
| Tokenizer | **none** — byte-level BPE from `tokenizer.json` already works (`runtime/src/tokenizer.rs:195`) | just supply Qwen's `tokenizer.json` |
| Pack weight names | `pack.rs:233` `map_tensor_names` (or hardcode in `qwen/api.rs`) | strip `model.language_model.` prefix; **drop** `model.visual.*` and `mtp.*`; keep Gated-DeltaNet names |

### 3.1 Per-layer cache — the structural change

Today every layer holds a `KvCache` (`rotary.rs:191`, `[len, num_heads, head_dim] f32`). Introduce:

```rust
enum LayerCache {
    Attn(KvCache),                 // full-attn layers (6)
    Delta(GatedDeltaNetState),     // linear-attn layers (18)
}
struct GatedDeltaNetState {        // models/qwen/state.rs
    heads: usize, k_dim: usize, v_dim: usize, conv_kernel: usize,
    s: Vec<f32>,    // [heads*k_dim*v_dim] f32, persistent, CONSTANT size
    conv: Vec<f32>, // [ (q+k+v channels) * (conv_kernel-1) ] f32 ring of recent inputs
}
```

Allocate one `Vec<LayerCache>` per session (mirror `llama/session/mod.rs:141`), thread `&mut cache[i]`
into the block (mirror `session/mod.rs:349`). `resident_bytes` for state = `s.len()+conv.len()` × 4,
folded into `context_memory_bytes` (`session/mod.rs:673`).

### 3.2 Rollback (the one genuinely new contract)

`RamaSessionAdapter::append_tokens` must be transactional (`session.rs:213-236`). K/V rolls back via
`KvCache::truncate` (`rotary.rs:260`). The recurrent `S`/`conv` are overwritten in place, so rollback
needs an explicit **checkpoint**: snapshot `s`/`conv` of the Delta layers before the step, restore on
error. State is only 18 MB total, so a full clone-on-write per step is acceptable for v1; optimize to
double-buffer later. **No precedent in the codebase — write it fresh.**

---

## 4. Packing plan (text-only `.spsa`)

- `normalize_architecture_type` → `"qwen3"`.
- `map_tensor_names`: strip `model.language_model.` → canonical `model.layers.{i}.*`,
  `model.embed_tokens.weight`, `model.norm.weight`. **Drop** all `model.visual.*` and `mtp.*`.
- Codec per tensor: big 2-D matrices (`in_proj_qkv`, `in_proj_z`, `out_proj`, `q/k/v/o_proj`, `mlp.*`,
  `embed_tokens`) → normal `auto`/`rans`/`q8` path. **Keep the small/sensitive linear-attn tensors raw
  (f32/bf16)**: `A_log` (f32), `dt_bias`, `conv1d.weight`, `norm.weight` (f32), `in_proj_a`,
  `in_proj_b` — they are tiny and feed the recurrence; lossy quant there is risky. Quantize the
  recurrence-feeding matrices only after parity holds.
- Expected text-only sizes: bf16 ≈ 3.76 GB, q8 ≈ 2.0 GB, rANS-lossless ≈ 2.4 GB.

---

## 5. Verification strategy (parity-first)

The recurrence is subtle; build a numerical oracle before trusting decode output.

1. **Reference dump:** run HF `transformers` on `Qwen/Qwen3.5-2B` for a fixed short prompt, dump
   per-layer hidden states + one full-attn and one Gated-DeltaNet layer's internals (q/k/v/β/g/S/o).
2. **Unit parity** (Rust test, like `bin/gemma-test.rs`): feed identical inputs to REEDELTA and REEGATE
   blocks, assert max-abs error < 1e-3 (bf16 tolerance) vs the dump. This catches the high-risk details:
   conv layout, l2norm placement, `g` formula, decay-before-readout ordering, gated-RMSNorm form,
   q/gate split, partial-rotary convention (`rotate_half` half-split), QK-norm.
3. **Whole-model logits parity:** greedy next-token over the prompt must match HF argmax; then
   free-running generation must be coherent.

---

## 6. Phased plan

- **P0 — pack plumbing.** Config fields + arch normalize + tensor-name map (drop vision/MTP). Produce a
  text-only `.spsa`; verify metadata/inspect. *No inference.*
- **P1 — REEGATE (full attn) + FFN + embed + tied head.** Reuses existing SDPA/QK-norm/partial-rotary.
  Stub the 18 linear layers as pass-through to get a compiling end-to-end forward (won't be coherent).
- **P2 — REEDELTA recurrent kernel + parity test.** The novel core; validate against the P5 dump.
- **P3 — heterogeneous per-layer dispatch + session.** Full text-only forward, greedy decode coherent,
  logits parity vs HF.
- **P4 — rollback-safe state checkpoint + `RamaSessionAdapter`.** Wire into `RamaChatSession`.
- **P5 — chunked GatedDeltaNet prefill** (perf) + codec/quant tuning + REEPOOL integration.

P0–P3 is the milestone: *coherent text from a real Qwen3.5-2B `.spsa` on the laptop.*

---

## 7. Risks / open questions

- **[VERIFY] mRoPE→RoPE collapse** for single-stream text (section `[11,11,10]`, interleaved).
- **[VERIFY] rotate_half convention** — HF uses half-split; map to `apply_gpt_neox_rotary_inplace`, not
  the llama "adjacent pairs" variant. Parity test decides.
- **[VERIFY] conv1d details** — depthwise over the concatenated `q‖k‖v` 6144 channels, kernel 4, **no
  bias**, SiLU *after* conv, channel order = `q,k,v`.
- **[VERIFY] gated-RMSNorm form** — `weight ⊙ rmsnorm(o) ⊙ SiLU(z)` (norm then ×SiLU(z)).
- **Gate shape** — `attn_output_gate` is per-(head,dim) `[8×256]`, elementwise `sigmoid(gate)`.
- **f32 recurrence** — keep `S`, `conv`, and the scan math in f32 (`mamba_ssm_dtype`); only the weights
  are bf16/quantized.
- **Prefill cost** — recurrent prefill is O(T) sequential per linear layer; acceptable for correctness,
  but long prompts will be slow until P5 chunking.
- **Decode bottleneck** — 248 k-row tied `lm_head` (~1 GB bf16 read/token) dominates, same class as
  Gemma's big head; existing streaming-head work applies.
