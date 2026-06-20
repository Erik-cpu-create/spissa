# R149a — Streaming lm-head on Gemma 3 1B (correctness-first) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the R148 streaming bit-plane GEMV into Gemma's LM head (opt-in) and prove Gemma 3 1B generates the identical token sequence with the streaming lm-head as with the resident path — lossless integration, on a small model.

**Architecture:** A sidecar file holds the model's tied bf16 embedding as block-framed bit-plane planes (R148 layout) behind a small header. `streaming_bitplane_gemv` gains a `data_offset` to skip that header. An opt-in branch in `gemma_lm_head` (gated by `RLLM_STREAM_LMHEAD=<sidecar>`) computes the logits by streaming the sidecar instead of reading the resident embedding.

**Tech Stack:** Rust (stable), crates `rllm-runtime` + `rllm-cli`, `std::arch::aarch64`. No new dependencies. Pre-flight already confirmed: Gemma 3 1B embedding = 26 distinct exponents → w=5; shape [262144, 1152] (hidden 1152 satisfies `%32==0` & `*5%8==0`).

## Global Constraints

- Pure Rust, **no new dependencies**; `cargo build` only.
- **Lossless (hard rule):** Gemma 3 1B produces the **identical token sequence** with the streaming lm-head as resident (same exact weights, same `bf16_row_dot_f32`).
- **Reuse, don't duplicate:** reuse `streaming_bitplane_gemv` (R148), `BitplaneCodec` (R143), `LazyRllmModel`. No new decode/dot/stream logic beyond the `data_offset` add + the sidecar framing.
- **Constraints:** lm-head `w == 5`, `hidden % 32 == 0`, `hidden*5 % 8 == 0`, `vocab % block_rows == 0` (Gemma: w=5, hidden=1152, vocab=262144, block_rows=256 → 1024 blocks, no padding).
- **Sidecar format:** `[magic "RLMH"(4)][version u8=1][hidden u32 LE][vocab u32 LE][block_rows u32 LE][palette_len u8][palette P bytes]` then the R148 framed blocks. `header_len = 18 + P`.
- **Scope:** lm-head only, opt-in (env-gated), default path unchanged. NO transformer-projection streaming, NO container reformat, NO `codec_for_id`. Correctness only — no speed claim (Gemma 1B fits in RAM).
- **REE kernel:** REESTREAM (R148, reused). No new kernel.

## File Structure

- **Modify** `crates/rllm-runtime/src/streaming/bitplane_stream.rs` — add `data_offset` to `streaming_bitplane_gemv`; add `write_lmhead_sidecar` + `stream_lmhead_from_sidecar` + round-trip test.
- **Modify** `crates/rllm-runtime/src/models/gemma/api.rs` — opt-in streaming branch in `gemma_lm_head`.
- **Modify** `crates/rllm-runtime/src/lib.rs` (or wherever the streaming module re-exports) — export `write_lmhead_sidecar` for the CLI/tool if needed (pub).
- **Create** `docs/benchmarks/trials/<verdict-folder>/2026-06-20-r149a-stream-lmhead-gemma.md` — trial report.
- **Modify** `docs/benchmarks/trials/index.md` + memory.

---

### Task 1: `data_offset` on `streaming_bitplane_gemv`

**Files:** Modify `crates/rllm-runtime/src/streaming/bitplane_stream.rs`

**Interfaces:**
- Produces: `streaming_bitplane_gemv(path, palette, hidden, block_rows, num_blocks, last_hidden, out, nocache, data_offset: u64)` — the reader seeks to `data_offset` before streaming blocks (to skip a header).

- [ ] **Step 1: Update the signature + reader seek**

In `streaming_bitplane_gemv`, add `data_offset: u64` as the last parameter, and in the reader thread, after opening the file (and the optional `fcntl`), seek:

```rust
use std::io::{Read, Seek, SeekFrom};
// ... inside the reader spawn, after File::open + optional fcntl:
if data_offset > 0 {
    if f.seek(SeekFrom::Start(data_offset)).is_err() {
        return;
    }
}
```

(Add `Seek, SeekFrom` to the existing `use std::io::Read;` import in the function.)

- [ ] **Step 2: Update the two R148 callers to pass `0`**

In the same file, in `streaming_gemv_matches_reference_bit_for_bit` and `streaming_gemv_capacity_bound_bench`, add `, 0` as the final arg to each `streaming_bitplane_gemv(...)` call (no header → offset 0).

- [ ] **Step 3: Run the R148 tests (still green)**

Run: `cargo test -p rllm-runtime --lib streaming_gemv_matches_reference -- --nocapture`
Expected: PASS (bit-identical preserved; offset 0 = no behavior change).

- [ ] **Step 4: Commit**

```bash
git add crates/rllm-runtime/src/streaming/bitplane_stream.rs
git commit -m "feat(runtime): streaming_bitplane_gemv data_offset (skip sidecar header) (R149a)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Sidecar producer + reader + round-trip parity

**Files:** Modify `crates/rllm-runtime/src/streaming/bitplane_stream.rs`

**Interfaces:**
- Consumes: `streaming_bitplane_gemv` (Task 1), `LazyRllmModel`, `rtc_codec::{BitplaneCodec, EncodeMeta, TensorCodec, decode_neon_w5}`, `bf16_row_dot_f32`.
- Produces: `pub fn write_lmhead_sidecar(model_path: &str, tensor_name: &str, block_rows: usize, out_path: &str) -> Result<()>`; `pub(crate) fn stream_lmhead_from_sidecar(path: &str, last_hidden: &[f32]) -> Result<Vec<f32>>`.

- [ ] **Step 1: Write the failing round-trip test**

Add to `mod bitplane_stream_tests`:

```rust
#[test]
fn lmhead_sidecar_streams_equal_to_reference() {
    // Build a tiny synthetic "model" embedding directly as a sidecar and confirm
    // stream_lmhead_from_sidecar == single-thread decode+dot reference.
    let (vocab, hidden, b) = (128usize, 1152usize, 64usize);
    let bf16 = make_embedding(vocab, hidden);
    let enc = BitplaneCodec
        .encode(&bf16, &EncodeMeta { name: "e".into(), shape: vec![(vocab * hidden) as u64], dtype: "bf16".into() })
        .unwrap();
    assert_eq!(enc.data[15], 5);
    let p = enc.data[14] as usize;
    let palette = enc.data[16..16 + p].to_vec();
    let row_idx = hidden * 5 / 8;
    let idx_plane = enc.data[16 + p..16 + p + vocab * row_idx].to_vec();
    let residuals = enc.data[16 + p + vocab * row_idx..16 + p + vocab * row_idx + vocab * hidden].to_vec();

    // write a sidecar by hand (same format write_lmhead_sidecar produces)
    let mut sc = Vec::new();
    sc.extend_from_slice(b"RLMH");
    sc.push(1);
    sc.extend_from_slice(&(hidden as u32).to_le_bytes());
    sc.extend_from_slice(&(vocab as u32).to_le_bytes());
    sc.extend_from_slice(&(b as u32).to_le_bytes());
    sc.push(p as u8);
    sc.extend_from_slice(&palette);
    sc.extend_from_slice(&frame_blocks(&idx_plane, &residuals, hidden, vocab, b));
    let path = "/tmp/r149a_unit.sidecar";
    std::fs::write(path, &sc).unwrap();

    let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.019).sin() * 0.3).collect();
    let mut reference = vec![0f32; vocab];
    for (r, slot) in reference.iter_mut().enumerate() {
        let dec = rtc_codec::decode_neon_w5(&palette, &idx_plane[r * row_idx..], &residuals[r * hidden..], hidden);
        *slot = bf16_row_dot_f32(&act, &dec, hidden);
    }
    let streamed = stream_lmhead_from_sidecar(path, &act).unwrap();
    let _ = std::fs::remove_file(path);
    assert_eq!(streamed, reference, "sidecar stream must equal decode+dot reference bit-for-bit");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p rllm-runtime --lib lmhead_sidecar_streams_equal -- --nocapture`
Expected: FAIL — `cannot find function stream_lmhead_from_sidecar`.

- [ ] **Step 3: Implement the producer + reader**

Add ABOVE the test module in `bitplane_stream.rs`:

```rust
/// Read a model's tied bf16 embedding/LM-head tensor, bit-plane encode it, and
/// write a block-framed sidecar file the streaming lm-head path consumes.
/// SAFETY/constraints: the tensor must be raw-bf16 readable (pack with `--codec raw`),
/// w must be 5, and `vocab % block_rows == 0`.
#[cfg(target_arch = "aarch64")]
pub fn write_lmhead_sidecar(
    model_path: &str,
    tensor_name: &str,
    block_rows: usize,
    out_path: &str,
) -> crate::Result<()> {
    use rtc_codec::{BitplaneCodec, EncodeMeta, TensorCodec};
    let mut m = crate::LazyRllmModel::open(model_path)?;
    let meta = m.tensor(tensor_name)?.clone();
    let vocab = meta.shape[0] as usize;
    let hidden = meta.shape[1] as usize;
    let bf16 = m
        .with_raw_tensor(meta.tensor_id, |b| Ok(b.to_vec()))?
        .ok_or_else(|| crate::RuntimeError::InvalidTensorData(
            "lm-head must be raw bf16 (repack with --codec raw)".into(),
        ))?;
    let n = vocab * hidden;
    let enc = BitplaneCodec
        .encode(&bf16, &EncodeMeta { name: tensor_name.into(), shape: vec![n as u64], dtype: "bf16".into() })
        .map_err(|e| crate::RuntimeError::InvalidTensorData(format!("bitplane encode: {e}")))?;
    if enc.data[15] != 5 {
        return Err(crate::RuntimeError::InvalidTensorData(format!(
            "lm-head bit-plane width {} != 5; decode16 kernel needs w=5",
            enc.data[15]
        )));
    }
    if vocab % block_rows != 0 {
        return Err(crate::RuntimeError::InvalidTensorData(
            "vocab must be a multiple of block_rows".into(),
        ));
    }
    let p = enc.data[14] as usize;
    let row_idx = hidden * 5 / 8;
    let idx_plane = &enc.data[16 + p..16 + p + vocab * row_idx];
    let residuals = &enc.data[16 + p + vocab * row_idx..16 + p + vocab * row_idx + n];

    let mut sidecar = Vec::with_capacity(18 + p + vocab * (row_idx + hidden));
    sidecar.extend_from_slice(b"RLMH");
    sidecar.push(1);
    sidecar.extend_from_slice(&(hidden as u32).to_le_bytes());
    sidecar.extend_from_slice(&(vocab as u32).to_le_bytes());
    sidecar.extend_from_slice(&(block_rows as u32).to_le_bytes());
    sidecar.push(p as u8);
    sidecar.extend_from_slice(&enc.data[16..16 + p]); // palette
    for blk in 0..vocab / block_rows {
        for r in 0..block_rows {
            let row = blk * block_rows + r;
            sidecar.extend_from_slice(&idx_plane[row * row_idx..(row + 1) * row_idx]);
        }
        for r in 0..block_rows {
            let row = blk * block_rows + r;
            sidecar.extend_from_slice(&residuals[row * hidden..(row + 1) * hidden]);
        }
    }
    std::fs::write(out_path, &sidecar)
        .map_err(|e| crate::RuntimeError::InvalidTensorData(format!("write sidecar: {e}")))?;
    Ok(())
}

/// Compute lm-head logits by streaming the bit-plane sidecar (R148 kernel).
#[cfg(target_arch = "aarch64")]
pub(crate) fn stream_lmhead_from_sidecar(path: &str, last_hidden: &[f32]) -> crate::Result<Vec<f32>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)
        .map_err(|e| crate::RuntimeError::InvalidTensorData(format!("open sidecar: {e}")))?;
    let mut head = [0u8; 256];
    let got = f.read(&mut head)
        .map_err(|e| crate::RuntimeError::InvalidTensorData(format!("read sidecar header: {e}")))?;
    if got < 18 || &head[0..4] != b"RLMH" || head[4] != 1 {
        return Err(crate::RuntimeError::InvalidTensorData("bad sidecar header".into()));
    }
    let hidden = u32::from_le_bytes(head[5..9].try_into().unwrap()) as usize;
    let vocab = u32::from_le_bytes(head[9..13].try_into().unwrap()) as usize;
    let block_rows = u32::from_le_bytes(head[13..17].try_into().unwrap()) as usize;
    let p = head[17] as usize;
    if got < 18 + p {
        return Err(crate::RuntimeError::InvalidTensorData("sidecar palette truncated".into()));
    }
    let palette = head[18..18 + p].to_vec();
    let header_len = (18 + p) as u64;
    let num_blocks = vocab / block_rows;
    let mut logits = vec![0f32; vocab];
    streaming_bitplane_gemv(
        path, &palette, hidden, block_rows, num_blocks, last_hidden, &mut logits, false, header_len,
    );
    Ok(logits)
}
```

- [ ] **Step 4: Run the round-trip test**

Run: `cargo test -p rllm-runtime --lib lmhead_sidecar_streams_equal -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Run the runtime lib suite + commit**

Run: `cargo test -p rllm-runtime --lib` → PASS.

```bash
git add crates/rllm-runtime/src/streaming/bitplane_stream.rs
git commit -m "feat(runtime): lm-head bit-plane sidecar writer + streaming reader (R149a)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Opt-in Gemma lm-head streaming branch

**Files:** Modify `crates/rllm-runtime/src/models/gemma/api.rs`

**Interfaces:**
- Consumes: `stream_lmhead_from_sidecar` (Task 2). Make it reachable: it is `pub(crate)` in the streaming module; import it in `gemma/api.rs`.

- [ ] **Step 1: Add the opt-in branch at the top of `gemma_lm_head`**

In `crates/rllm-runtime/src/models/gemma/api.rs`, at the start of `fn gemma_lm_head(...)` (before the `if let Some(emb) = embedding_f32` block), add:

```rust
    // R149a: opt-in streaming lm-head from a bit-plane sidecar (capacity-bound mode).
    #[cfg(target_arch = "aarch64")]
    if let Ok(sidecar) = std::env::var("RLLM_STREAM_LMHEAD") {
        if !sidecar.is_empty() {
            let _ = (embedding_f32, embed_id); // resident inputs unused on this path
            return crate::streaming::stream_lmhead_from_sidecar(&sidecar, last_hidden);
        }
    }
```

(If `stream_lmhead_from_sidecar` is not reachable as `crate::streaming::stream_lmhead_from_sidecar`, adjust the path to wherever the streaming module is — it is `pub(crate)` and lives in the `streaming` module; confirm the module path with `grep -n "mod streaming" crates/rllm-runtime/src/lib.rs`.)

- [ ] **Step 2: Build to verify it compiles**

Run: `cargo build -p rllm-runtime 2>&1 | tail -3`
Expected: compiles (resolve the module path if the import fails).

- [ ] **Step 3: Run the runtime lib suite**

Run: `cargo test -p rllm-runtime --lib` → PASS (default path unchanged; the branch is env-gated and not hit in tests).

- [ ] **Step 4: Commit**

```bash
git add crates/rllm-runtime/src/models/gemma/api.rs
git commit -m "feat(runtime): opt-in streaming lm-head for Gemma via RLLM_STREAM_LMHEAD (R149a)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Produce the Gemma sidecar + identical-output correctness gate + docs

**Files:** Create trial report; modify index + memory. (No source change — a CLI/tool path drives `write_lmhead_sidecar`.)

**Interfaces:** Consumes `write_lmhead_sidecar` (Task 2) + the `gemma-test` binary + `RLLM_STREAM_LMHEAD`.

- [ ] **Step 1: Repack Gemma 1B with raw codec (so the embedding is raw bf16)**

```bash
./target/release/rllm pack /tmp/gemma1b --out models/gemma-3-1b-it-rawcodec.rllm \
  --quantize raw --codec raw --config /tmp/gemma1b/config.json --tokenizer /tmp/gemma1b/tokenizer.json 2>&1 | tail -2
```
Expected: "Found 340 tensors (... 'gemma3')", written.

- [ ] **Step 2: Add an `#[ignore]` tool test that writes the Gemma sidecar**

Add to `mod bitplane_stream_tests` in `bitplane_stream.rs`:

```rust
#[test]
#[ignore]
fn write_gemma_lmhead_sidecar() {
    write_lmhead_sidecar(
        "../../models/gemma-3-1b-it-rawcodec.rllm",
        "model.embed_tokens.weight",
        256,
        "/tmp/gemma1b-lmhead.sidecar",
    )
    .unwrap();
    eprintln!("wrote /tmp/gemma1b-lmhead.sidecar");
}
```

Run: `cargo test -p rllm-runtime --release write_gemma_lmhead_sidecar -- --ignored --nocapture`
Expected: "wrote /tmp/gemma1b-lmhead.sidecar" (asserts w=5 internally).

- [ ] **Step 3: Generate twice (resident vs streaming) and confirm identical tokens**

```bash
cargo build --release -p rllm-cli
P="Name three colors."
echo "=== resident ==="; ./target/release/gemma-test --model models/gemma-3-1b-it-rawcodec.rllm --prompt "$P" --max-new-tokens 16 --ctx 256 2>&1 | grep "Generated token ids"
echo "=== streaming ==="; RLLM_STREAM_LMHEAD=/tmp/gemma1b-lmhead.sidecar ./target/release/gemma-test --model models/gemma-3-1b-it-rawcodec.rllm --prompt "$P" --max-new-tokens 16 --ctx 256 2>&1 | grep "Generated token ids"
```
Expected: the two `Generated token ids: [...]` lines are **identical** → lossless streaming lm-head. **Record both lines.** (If they differ, the streaming path has a bug — diff the logits before declaring done.)

- [ ] **Step 4: Write the trial report**

Create `docs/benchmarks/trials/success/2026-06-20-r149a-stream-lmhead-gemma.md` (or `failed/` if tokens differ): Scope (REESTREAM reused; Gemma 3 1B; lm-head streaming, opt-in), Hypothesis (streaming lm-head is lossless in a real model), Results (the two identical token-id lines; w=5 confirmed; correctness gate green), Analysis (correctness-first; speed deferred to >RAM/cold; Gemma 1B fits RAM so no speed claim), Decision (GO if identical), Next (R149b: >RAM model / cold measurement + transformer-projection streaming).

- [ ] **Step 5: Index row + memory + commit**

Add the R149a row to `docs/benchmarks/trials/index.md`; append to memory `rllm-speed-thesis-streaming-vs-resident.md` (streaming lm-head works lossless on Gemma 3 1B, opt-in, identical tokens; speed is R149b).

```bash
git add docs/benchmarks/trials/ "/Users/deansanbhnanwr/.claude/projects/-Users-deansanbhnanwr-Projects-rllm/memory/rllm-speed-thesis-streaming-vs-resident.md"
git commit -m "docs(bench): R149a streaming lm-head on Gemma 3 1B (lossless, identical tokens) + index + memory

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

## Verification (end-to-end)

1. `cargo test -p rllm-runtime --lib` → green (R148 bit-identity + R149a sidecar round-trip).
2. `cargo build --release` → compiles, no new deps.
3. The two `Generated token ids` lines (resident vs `RLLM_STREAM_LMHEAD`) are **identical** on Gemma 3 1B.
4. Trial report filed; index + memory updated.

## Out of scope (R149b+ follow-on)

- Speed measurement on a > RAM model (or cold-read) — the capacity-bound win demo.
- Streaming the transformer projections (not just lm-head).
- Container reformat / `codec_for_id` registration / a first-class `pack` sidecar subcommand.
