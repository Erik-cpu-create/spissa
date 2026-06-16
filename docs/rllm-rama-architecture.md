# RLLM RAMA Architecture

Status: accepted Phase 5D.5 architecture direction
Scope: original brain-inspired, memory-first RLLM runtime architecture
Audience: RLLM implementers and future contributors

## Objective

Define the official brain-inspired architecture identity for RLLM without changing what RLLM already means as a product and file/runtime system.

RLLM remains:

```text
RLLM = Runtime-compressed Local LLM
RTC  = RLLM Tensor Codec
```

The official runtime architecture inside RLLM is:

```text
RAMA = Rama Active Memory Architecture
```

Reserved future subsystem name:

```text
ERIK = Episodic Recall Inference Kernel
```

Required kernel lineage prefix:

```text
REE = Rama Erik Esprada kernel lineage
```

Short positioning:

```text
RLLM is powered by RAMA: compressed memory, selective recall, bounded active thought.
```

Formal positioning:

```text
RLLM is a runtime-compressed local LLM system powered by RAMA, a memory-first execution architecture where compressed model weights are treated as dormant long-term memory, bounded RAM is treated as active working memory, and the runtime recalls only the chunks, layers, tiles, and context traces required for the current inference step.
```

## Naming Decision

RLLM is the product and system name. It owns:

- the CLI identity (`rllm`)
- the `.rllm` container format
- the runtime-compressed local LLM value proposition
- the workspace/crate-level project identity

RAMA is the runtime architecture. It owns:

- memory-first inference rules
- active-memory limits
- recall/decode scheduling
- short-term context cache behavior
- future eviction, consolidation, and adaptive layout policies

ERIK is reserved for a later focused subsystem. It should not be used as a broad architecture name. Its intended scope is smarter episodic/context recall after RAMA has a stable runtime foundation.

REE is the required lineage prefix for original RLLM execution kernels. Any new CPU
kernel that becomes a serious benchmark candidate must receive a REE name before
it is reported, merged, or promoted into runtime use. This naming is not cosmetic:
it is part of RLLM's kernel versioning and research traceability contract.

Required REE naming rules:

- Use original names beginning with `REE`, for example `REEDOT-LAB`, `REEBORN-Q8`, `REETHINK-Q8`, `REEFUSE-Q8`, or `REELITE-Q8`.
- Use `*-LAB` for microbench-only kernels that are not wired into inference.
- Use `REEBORN-*` for the first measured kernel in a lineage that is promoted into runtime.
- Use `REETHINK-*` for a redesigned replacement after a failed kernel direction.
- Use `REEFUSE-*` for fused kernels such as gate/up or matmul+scale variants.
- Use `REELITE-*` for kernels explicitly optimized for low-end or IoT CPU profiles.
- Benchmark reports must include the REE kernel name, even when the trial fails.
- Do not merge anonymous "fast path", "candidate", or "optimized kernel" changes without assigning and documenting the REE name.

## Originality Doctrine

RLLM/RAMA must be genuine from-scratch engineering work.

Always:

- Design mechanisms from first principles around RLLM's constraints: low RAM, local-first execution, lossless storage, chunk/layer/tile streaming, deterministic verification, and correctness.
- Implement custom RLLM/RTC/RAMA mechanisms directly in this repository unless the user explicitly approves a dependency trade-off.
- Define every major concept with an executable RLLM-specific meaning before coding it.
- Credit external facts when used for compatibility, for example safetensors layout, GPT-NeoX tensor names, dtype definitions, tokenizer formats, or known model config fields.
- Treat brain terminology as analogy for memory/runtime structure, not as biological equivalence.

Never:

- Copy architecture, code, naming systems, diagrams, or prose from another project.
- Claim RLLM is biologically accurate, conscious, self-learning, or human-brain equivalent.
- Add generic compression dependencies such as zstd/lz4 unless explicitly approved.
- Use neuroscience words as decoration without a concrete runtime contract.
- Rebrand standard transformer behavior as novel without explaining what RLLM/RAMA specifically adds.

If a name or mechanism resembles existing work by coincidence, this spec must define the RLLM-specific meaning and avoid implying affiliation or derivation.

## What RAMA Is / Is Not

RAMA is:

- A memory-first runtime architecture for RLLM.
- A design language for deciding what to load, decode, cache, release, evict, and verify.
- A hierarchy for compressed storage, bounded working memory, recall, context cache, and streaming execution.
- A brain-inspired engineering analogy around memory pressure and selective activation.

RAMA is not:

- A new neural network architecture.
- A training algorithm.
- A biological brain simulator.
- A consciousness or cognition claim.
- A shortcut around transformer math correctness.
- A reason to weaken bit-exact/lossless verification.

## Core Principle

```text
A model should not be loaded all at once.
A model should be recalled into active memory only when needed.
```

RAMA treats inference as controlled memory activation instead of unconditional model loading.

## RAMA Laws

### 1. Active Memory Law

Only data required for the current inference step may enter active memory.

Practical meaning:

- Decode only the needed chunks/layers/tiles.
- Bound activations, temporary buffers, logits, and KV-cache growth.
- Reject plans that exceed `MemoryBudget` instead of silently over-allocating.

### 2. Recall Before Decode

Every tensor decode must be planned as recall.

Practical meaning:

- Know where the tensor/chunk lives.
- Know how much memory the decode will consume.
- Know when the decoded buffer can be released.
- Prefer metadata-only planning before touching compressed payloads.

### 3. Short-Term Context Law

Recent context should live in explicit context memory, not be recomputed blindly.

Practical meaning:

- KV-cache is RAMA's current short-term context primitive.
- Rotary position offsets must align with cached absolute positions.
- Cache append must be validated and capacity-bounded.

### 4. Forgetting Law

Anything no longer needed must be released, evicted, or compacted.

Practical meaning:

- Scratch buffers must be released after use.
- Cache growth requires explicit policy.
- Future cache compression or sliding-window eviction must be deterministic and visible in plans/logs.

### 5. Fidelity Law

Runtime compression must not change model weights unless explicitly labeled as lossy.

Practical meaning:

- `.rllm` storage stays lossless by default.
- RTC codec round-trips must verify bit-identical bytes.
- Runtime dtype conversion must be explicit and tested.

### 6. No Fake Brain Law

RAMA is brain-inspired, not brain-equivalent.

Practical meaning:

- Use terms like working memory, recall, and context only when there is a concrete runtime mechanism.
- Do not claim biological learning, self-awareness, or human-like cognition.

## Conceptual Model

RAMA models inference as movement across memory states:

```text
Dormant Memory       -> compressed .rllm tensor chunks on disk
Recall Path          -> chunk/layer/tile decode into temporary buffers
Active Working Memory-> bounded activations, logits, scratch, and planner-visible buffers
Context Memory       -> per-layer KV-cache / recent-token state
Focus Operator       -> attention and sampling mechanisms that select next-token state
Release/Forgetting   -> deterministic memory release, cache eviction, or future compaction
```

Central rule:

```text
Every active byte must justify why it exists right now.
```

## RAMA Runtime Layers

### 1. Dormant Memory

Purpose:

```text
Store model knowledge compactly while inactive.
```

Current implementation:

- `.rllm` container
- tensor metadata and chunk directory
- RTC codecs: raw, RLE, Huffman
- SHA-256 verification and lossless round-trip checks

Future direction:

- hot/cold chunk profiling
- layout-aware repacking
- codec selection informed by runtime access patterns

### 2. Recall Path

Purpose:

```text
Move only required dormant memory into active computation.
```

Current implementation:

- `LazyRllmModel`
- chunk-scoped decode
- metadata-only open path
- streaming linear/MLP/attention/block primitives

Future direction:

- layer streaming for real models
- tile streaming / fused decode+matmul
- prefetch hints driven by generation schedule

### 3. Active Working Memory

Purpose:

```text
Bound active bytes and reject plans that exceed budget.
```

Current implementation:

- `MemoryBudget`
- runtime planner
- `--memory-budget`, `--ctx`, `--mode full-decode|layer-stream|tile-stream`
- budgeted scratch buffers for streaming activations

Future direction:

- live budget scheduler
- adaptive chunk size selection
- separate budgets for weights, activations, cache, and output buffers

### 4. Context Memory

Purpose:

```text
Preserve recent computation traces so decoding does not recompute all history.
```

Current implementation:

- `KvCache`
- cached causal attention
- rotary position offset support
- existing generation state types may still use legacy `Echo` naming until the code is migrated

Future direction:

- per-layer KV-cache ownership under RAMA naming
- sliding-window cache
- cache compression
- cache eviction policy

### 5. Focus Operator

Purpose:

```text
Choose relevant context and transform it into next-token state.
```

Current implementation:

- scaled dot-product attention
- causal mask
- GPT-NeoX/Pythia rotary embeddings
- streaming QKV attention path
- argmax/top-p sampling primitives

Future direction:

- fused attention windows
- tile-local attention scheduling
- cache-aware attention memory accounting

### 6. Consolidation Loop

Purpose:

```text
Improve storage/runtime layout after observation, without changing model weights.
```

Status: future.

Potential implementation:

- collect chunk access telemetry
- identify hot chunks and cold chunks
- offline repack `.rllm` files for better streaming locality
- record recommended chunk size/layout profiles per model

Boundary:

```text
Consolidation changes storage layout, not learned model parameters.
```

### 7. Forgetting Loop

Purpose:

```text
Release, evict, or compact active traces when memory pressure rises.
```

Status: future.

Potential implementation:

- KV-cache capacity policy
- sliding context window
- cache compression tiers
- deterministic eviction decisions visible in logs/plans

## ERIK Future Subsystem

ERIK is reserved for a focused subsystem inside RAMA:

```text
ERIK = Episodic Recall Inference Kernel
```

Intended future responsibility:

- manage long-context recall policies
- decide what context survives under limited cache budget
- provide deterministic episodic summaries or compressed context traces when exact KV-cache retention is too expensive
- expose recall decisions to the planner and CLI

ERIK must not be introduced until the exact runtime contract is testable. It should not become a vague name for all generation logic.

## Current Implementation Mapping

Current RLLM pieces map into RAMA as follows:

| RAMA concept | Current RLLM implementation |
|---|---|
| Dormant Memory | `.rllm` container, chunk metadata, RTC codecs |
| Recall Path | `LazyRllmModel`, chunk decode, streaming primitives |
| Active Working Memory | `MemoryBudget`, runtime planner, budgeted activations |
| Context Memory | `KvCache`, cached attention, rotary offsets |
| Focus Operator | attention, causal mask, sampling |
| Fidelity Law | SHA-256 verification, bit-identical codec round-trips |

Some code may currently use the earlier ECHO/EMBER names. Those names are superseded as official architecture branding. Public RAMA aliases/wrappers should be preferred for new code while legacy ECHO names remain available for compatibility until a dedicated removal decision.

## Implementation Boundary

RAMA belongs primarily in `rllm-runtime` and runtime-facing CLI/status planning.

RAMA may define:

- runtime memory policy
- decode scheduling policy
- cache state policy
- budget reporting
- generation loop structure
- layer/tile streaming contracts

RAMA must not own:

- `.rllm` binary container semantics that belong in `rllm-container`
- codec internals that belong in `rtc-codec`
- safetensors parsing that belongs in `rllm-import`
- broad CLI formatting that belongs in `rllm-cli`
- external model architecture facts beyond compatibility metadata

## Success Criteria

RAMA is respected when:

- A runtime path can explain every active allocation and release.
- A low-RAM path avoids full model decode unless explicitly requested.
- KV/context memory is explicit, bounded, and testable.
- Storage compression remains lossless by default.
- Claims are backed by tests, planner output, or measured smoke runs.
- Brain-inspired terminology maps to a concrete runtime mechanism.

## Near-Term Roadmap Alignment

The next slices should follow this order:

1. Prefer RAMA public API names for new runtime and CLI code.
2. Keep existing ECHO/ContextEcho runtime code working as compatibility aliases until a dedicated deprecation/removal decision.
3. Phase 6: GPT-NeoX/Pythia layer-param decode path under RAMA terminology; keep only names + final params resident and decode/release active layer params per layer.
4. Phase 7: fused tile-linear first, tiled MLP/attention/RAMA/GPT-NeoX routing second, measured real Pythia release RSS benchmark matrix third, fixed-token HF/PyTorch logits comparison fourth, range-decode API/container foundation fifth, per-range checksum metadata sixth, opt-in raw/identity pack range metadata seventh, pack-time tile/block chunk alignment eighth, real Pythia tile-block RSS benchmark ninth, RAMA trace profiler tenth, embedding row recall eleventh, low-ram-fast raw/tile-block profile and verify-once integrity twelfth, then parallelize the remaining lm-head/`embed_out.weight` compute path based on measured trace evidence.
5. Future ERIK: explicit episodic recall/cache policy only after baseline RAMA layer/tile runtime is correct.

## Non-Goals

Do not use this spec to justify:

- lossy quantization by default
- unverified compression claims
- copying external runtime designs
- broad rewrites without tests
- biological brain claims
- adding generic compression dependencies without approval

## Decision Summary

Accepted naming:

```text
Product/system: RLLM = Runtime-compressed Local LLM
Codec layer:    RTC  = RLLM Tensor Codec
Architecture:   RAMA = Rama Active Memory Architecture
Future kernel:  ERIK = Episodic Recall Inference Kernel
```

RLLM remains the product. RAMA is the official runtime architecture inside it.
