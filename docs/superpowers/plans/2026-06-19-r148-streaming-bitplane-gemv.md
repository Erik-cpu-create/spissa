# R148 — Pipelined streaming bit-plane GEMV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A double-buffer pipelined streaming bit-plane GEMV that reads block-framed compressed planes sequentially from a file while decoding+dotting the previous block, hiding the decode under the cold-SSD read — measured to push the R147 capacity-bound win from 1.13× toward ~1.23×, lossless.

**Architecture:** A reader thread streams fixed-stride blocks (`B` rows each, `[B×index bytes ++ B×residual bytes]` contiguous) from a file into one of two buffers via a bounded channel, while the consumer decodes each row (`rtc_codec::decode16_w5_into`, R146) and dots it (`bf16_row_dot_f32`, mod.rs) from the other buffer. The channel double-buffers so block N+1's disk read overlaps block N's decode.

**Tech Stack:** Rust (stable), crate `rllm-runtime`, `std::thread` + `std::sync::mpsc::sync_channel`, `std::arch::aarch64`. No new dependencies.

## Global Constraints

- Pure Rust, **no new dependencies**; `cargo build` only. Threads + channels from `std`; `F_NOCACHE` via a self-declared `extern "C" fn fcntl` (no libc crate).
- **Lossless / bit-identical (hard rule):** the streaming GEMV's logits are bit-identical to a single-thread `decode_neon_w5 + bf16_row_dot_f32` reference (same weights, same dot, same per-row order). Parity test asserts exact f32 equality.
- **Reuse, don't duplicate (DRY):** reuse `rtc_codec::decode_neon_w5` / `decode16_w5_into` (R143/R146) and `bf16_row_dot_f32` (mod.rs). No new decode/dot math.
- **Constraints:** `w = 5`, `hidden % 32 == 0` and `hidden*5 % 8 == 0` (true for 2048 → row index = 1280 bytes, residual = 2048 bytes). Block stride = `B*(1280+2048)` bytes.
- **REE kernel working name: REESTREAM** (Erik's final call) — trial Scope line.
- **Scope:** the pipelined streaming kernel + a >RAM cold-stream benchmark only. NO full model wiring / `codec_for_id` / pack — R149+ follow-on. NO multi-threaded decode (one overlapped decoder suffices when disk-bound).
- **Gate:** on a >RAM cold stream, the pipelined kernel beats raw bf16 streaming and improves on R147's 1.13× (toward ~1.23×). GO if faster than raw + ≥ scout; report honestly otherwise.

## File Structure

- **Create** `crates/rllm-runtime/src/streaming/bitplane_stream.rs` — `streaming_bitplane_gemv` (pipelined kernel) + a `#[cfg(test)] mod` (bit-identity unit test + the >RAM bench).
- **Modify** `crates/rllm-runtime/src/streaming/mod.rs` — add `include!("bitplane_stream.rs");`.
- **Create** `docs/benchmarks/trials/<verdict-folder>/2026-06-19-r148-reestream-streaming-gemv.md` — trial report.
- **Modify** `docs/benchmarks/trials/index.md` — R148 row.
- **Modify** memory `rllm-speed-thesis-streaming-vs-resident.md` — measured R148 number.

---

### Task 1: Pipelined `streaming_bitplane_gemv` + bit-identity parity

**Files:**
- Create: `crates/rllm-runtime/src/streaming/bitplane_stream.rs`
- Modify: `crates/rllm-runtime/src/streaming/mod.rs`

**Interfaces:**
- Consumes: `rtc_codec::decode16_w5_into` (R146), `rtc_codec::decode_neon_w5` (R143, reference), `bf16_row_dot_f32` (mod.rs).
- Produces: `#[cfg(target_arch = "aarch64")] fn streaming_bitplane_gemv(path: &str, palette: &[u8], hidden: usize, block_rows: usize, num_blocks: usize, last_hidden: &[f32], out: &mut [f32], nocache: bool)`.

- [ ] **Step 1: Wire the new file into `mod.rs`**

In `crates/rllm-runtime/src/streaming/mod.rs`, add after `include!("bitplane_gemv.rs");` (and before `include!("tests.rs");`):

```rust
include!("bitplane_stream.rs");
```

- [ ] **Step 2: Write the failing bit-identity test**

Create `crates/rllm-runtime/src/streaming/bitplane_stream.rs` with the header + test module only:

```rust
// Pipelined streaming bit-plane GEMV (R148 REESTREAM, capacity-bound runtime kernel).
//
// Reads block-framed compressed planes sequentially from a file while decoding +
// dotting the previous block, so the cold-SSD read of block N+1 overlaps the
// decode of block N. Reuses rtc-codec decode (R143/R146) + bf16_row_dot_f32.

#[cfg(all(test, target_arch = "aarch64"))]
mod bitplane_stream_tests {
    use super::*;
    use rtc_codec::{BitplaneCodec, EncodeMeta, TensorCodec};

    fn make_embedding(vocab: usize, hidden: usize) -> Vec<u8> {
        let mut state = 0x0BAD_F00D_1234_5678u64;
        let mut out = Vec::with_capacity(vocab * hidden * 2);
        for k in 0..vocab * hidden {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let exp = (96 + (k % 32)) as u16 & 0xFF;
            let bits = (((state >> 31) & 1) as u16) << 15 | (exp << 7) | (state & 0x7F) as u16;
            out.extend_from_slice(&bits.to_le_bytes());
        }
        out
    }

    // Frame the flat planes into [B×index ++ B×residual] contiguous blocks.
    fn frame_blocks(idx_plane: &[u8], residuals: &[u8], hidden: usize, vocab: usize, b: usize) -> Vec<u8> {
        let row_idx = hidden * 5 / 8;
        let mut framed = Vec::new();
        for blk in 0..(vocab / b) {
            for r in 0..b {
                let row = blk * b + r;
                framed.extend_from_slice(&idx_plane[row * row_idx..(row + 1) * row_idx]);
            }
            for r in 0..b {
                let row = blk * b + r;
                framed.extend_from_slice(&residuals[row * hidden..(row + 1) * hidden]);
            }
        }
        framed
    }

    #[test]
    fn streaming_gemv_matches_reference_bit_for_bit() {
        let (vocab, hidden, b) = (128usize, 2048usize, 64usize);
        let bf16 = make_embedding(vocab, hidden);
        let enc = BitplaneCodec
            .encode(&bf16, &EncodeMeta { name: "e".into(), shape: vec![(vocab * hidden) as u64], dtype: "bf16".into() })
            .unwrap();
        let p = enc.data[14] as usize;
        assert_eq!(enc.data[15], 5);
        let mut off = 16;
        let palette = enc.data[off..off + p].to_vec();
        off += p;
        let row_idx = hidden * 5 / 8;
        let idx_bytes = vocab * row_idx;
        let idx_plane = &enc.data[off..off + idx_bytes];
        off += idx_bytes;
        let residuals = &enc.data[off..off + vocab * hidden];
        let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.021).sin() * 0.3).collect();

        // single-thread reference: decode each row + dot
        let mut reference = vec![0f32; vocab];
        for (r, slot) in reference.iter_mut().enumerate() {
            let dec = rtc_codec::decode_neon_w5(&palette, &idx_plane[r * row_idx..], &residuals[r * hidden..], hidden);
            *slot = bf16_row_dot_f32(&act, &dec, hidden);
        }

        // block-framed file + streaming kernel
        let framed = frame_blocks(idx_plane, residuals, hidden, vocab, b);
        let path = "/tmp/r148_unit.bin";
        std::fs::write(path, &framed).unwrap();
        let mut out = vec![0f32; vocab];
        streaming_bitplane_gemv(path, &palette, hidden, b, vocab / b, &act, &mut out, false);
        let _ = std::fs::remove_file(path);

        assert_eq!(out, reference, "streaming GEMV must equal single-thread decode+dot bit-for-bit");
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `cargo test -p rllm-runtime --lib streaming_gemv_matches_reference -- --nocapture`
Expected: FAIL — `cannot find function streaming_bitplane_gemv`.

- [ ] **Step 4: Implement the pipelined kernel**

Insert ABOVE the `#[cfg(all(test, target_arch = "aarch64"))] mod bitplane_stream_tests` block in `bitplane_stream.rs`:

```rust
/// Double-buffer pipelined streaming bit-plane GEMV. Reads `num_blocks` blocks of
/// `block_rows` rows each (`[B×index bytes ++ B×residual bytes]`, w=5) sequentially
/// from `path`; a reader thread streams the next block while this thread decodes +
/// dots the current one. Writes `num_blocks*block_rows` logits. Bit-identical to a
/// single-thread decode+dot. Not yet runtime-wired — the R148 capacity-bound kernel.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
fn streaming_bitplane_gemv(
    path: &str,
    palette: &[u8],
    hidden: usize,
    block_rows: usize,
    num_blocks: usize,
    last_hidden: &[f32],
    out: &mut [f32],
    nocache: bool,
) {
    use std::fs::File;
    use std::io::Read;
    use std::os::unix::io::AsRawFd;
    use std::sync::mpsc::sync_channel;
    extern "C" {
        fn fcntl(fd: i32, cmd: i32, arg: i32) -> i32;
    }
    const F_NOCACHE: i32 = 48;

    let row_idx = hidden * 5 / 8;
    let row_res = hidden;
    let block_bytes = block_rows * (row_idx + row_res);

    let (full_tx, full_rx) = sync_channel::<(usize, Vec<u8>)>(2);
    let (empty_tx, empty_rx) = sync_channel::<Vec<u8>>(2);
    empty_tx.send(vec![0u8; block_bytes]).unwrap();
    empty_tx.send(vec![0u8; block_bytes]).unwrap();

    std::thread::scope(|s| {
        // reader: fill the spare buffer with block N+1 while the consumer drains N.
        s.spawn(move || {
            let mut f = File::open(path).unwrap();
            if nocache {
                unsafe { fcntl(f.as_raw_fd(), F_NOCACHE, 1) };
            }
            for blk in 0..num_blocks {
                let mut buf = match empty_rx.recv() {
                    Ok(b) => b,
                    Err(_) => break,
                };
                if f.read_exact(&mut buf).is_err() {
                    break;
                }
                if full_tx.send((blk, buf)).is_err() {
                    break;
                }
            }
            // full_tx drops here -> consumer's recv() ends after draining.
        });

        // consumer: decode+dot each row of each received block.
        let mut scratch = vec![0u8; hidden * 2];
        while let Ok((blk, buf)) = full_rx.recv() {
            for r in 0..block_rows {
                let idx = &buf[r * row_idx..];
                let res = &buf[block_rows * row_idx + r * row_res..];
                unsafe { rtc_codec::decode16_w5_into(palette, idx, res, hidden, &mut scratch) };
                out[blk * block_rows + r] = bf16_row_dot_f32(last_hidden, &scratch, hidden);
            }
            let _ = empty_tx.send(buf);
        }
    });
}
```

- [ ] **Step 5: Run the parity test to verify it passes**

Run: `cargo test -p rllm-runtime --lib streaming_gemv_matches_reference -- --nocapture`
Expected: PASS — the pipelined streaming logits equal the single-thread reference bit-for-bit.

- [ ] **Step 6: Run the runtime lib suite**

Run: `cargo test -p rllm-runtime --lib`
Expected: PASS — all existing tests + the new parity test (292 total).

- [ ] **Step 7: Commit**

```bash
git add crates/rllm-runtime/src/streaming/bitplane_stream.rs crates/rllm-runtime/src/streaming/mod.rs
git commit -m "feat(runtime): pipelined streaming bit-plane GEMV, bit-identical (R148 REESTREAM)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: >RAM cold-stream benchmark + measurement

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/bitplane_stream.rs` (add the `#[ignore]` bench)

**Interfaces:**
- Consumes: `streaming_bitplane_gemv` (Task 1), `bf16_row_dot_f32`; `/tmp/rllm-bf16-sample.bin`.
- Produces: raw-bf16 vs R148-pipelined ms on a >RAM cold stream + the speedup + verdict.

- [ ] **Step 1: Add the `#[ignore]` bench**

Add to `mod bitplane_stream_tests` in `bitplane_stream.rs`:

```rust
#[test]
#[ignore]
fn streaming_gemv_capacity_bound_bench() {
    use std::io::{Read, Write};
    use std::os::unix::io::AsRawFd;
    extern "C" {
        fn fcntl(fd: i32, cmd: i32, arg: i32) -> i32;
    }
    const F_NOCACHE: i32 = 48;

    let bytes = std::fs::read("/tmp/rllm-bf16-sample.bin")
        .expect("run dump_bf16_embedding_sample first");
    let hidden = 2048usize;
    let n = bytes.len() / 2;
    let vocab = n / hidden;
    let b = 256usize; // block rows
    assert_eq!(vocab % b, 0, "vocab must be a multiple of block size");
    let enc = BitplaneCodec
        .encode(&bytes, &EncodeMeta { name: "e".into(), shape: vec![n as u64], dtype: "bf16".into() })
        .unwrap();
    let p = enc.data[14] as usize;
    let mut off = 16;
    let palette = enc.data[off..off + p].to_vec();
    off += p;
    let row_idx = hidden * 5 / 8;
    let idx_bytes = vocab * row_idx;
    let idx_plane = &enc.data[off..off + idx_bytes];
    off += idx_bytes;
    let residuals = &enc.data[off..off + n];
    let framed = frame_blocks(idx_plane, residuals, hidden, vocab, b);

    // Replicate both files > RAM (~3 GB free) so reads are genuinely cold.
    let k = 12usize;
    let raw_path = "/tmp/r148_raw.bin";
    let comp_path = "/tmp/r148_comp.bin";
    {
        let mut fr = std::fs::File::create(raw_path).unwrap();
        let mut fc = std::fs::File::create(comp_path).unwrap();
        for _ in 0..k {
            fr.write_all(&bytes).unwrap();
            fc.write_all(&framed).unwrap();
        }
    }
    let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();

    // raw bf16 stream cold: pure read of the bytes raw MUST move (the strongest,
    // fairest baseline -- in a real pipelined raw path the dot is hidden under the
    // read too, so we give raw the benefit of zero compute).
    let raw_ms = {
        let mut f = std::fs::File::open(raw_path).unwrap();
        unsafe { fcntl(f.as_raw_fd(), F_NOCACHE, 1) };
        let mut buf = vec![0u8; bytes.len()];
        let t = std::time::Instant::now();
        for _ in 0..k {
            f.read_exact(&mut buf).unwrap();
            std::hint::black_box(&buf);
        }
        t.elapsed().as_secs_f64() * 1000.0
    };

    // R148 pipelined stream cold: blocks across the whole replicated file.
    let comp_ms = {
        let total_blocks = (vocab / b) * k;
        let mut out = vec![0f32; total_blocks * b];
        let t = std::time::Instant::now();
        streaming_bitplane_gemv(comp_path, &palette, hidden, b, total_blocks, &act, &mut out, true);
        std::hint::black_box(&out);
        t.elapsed().as_secs_f64() * 1000.0
    };

    let _ = std::fs::remove_file(raw_path);
    let _ = std::fs::remove_file(comp_path);
    let raw_gb = (bytes.len() * k) as f64 / 1e9;
    let comp_gb = (framed.len() * k) as f64 / 1e9;
    eprintln!(
        "\n=== R148 REESTREAM pipelined capacity-bound BENCH (cold SSD, > RAM) ===\n\
         raw bf16   stream {raw_gb:.1} GB -> {raw_ms:.0} ms  ({:.2} GB/s)\n\
         pipelined  stream {comp_gb:.1} GB -> {comp_ms:.0} ms  ({:.2} GB/s, decode overlapped)\n\
         SPEEDUP vs raw: {:.2}x   (R147 un-pipelined scout was 1.13x)\n\
         VERDICT: {}\n",
        raw_gb / (raw_ms / 1e3),
        comp_gb / (comp_ms / 1e3),
        raw_ms / comp_ms,
        if comp_ms < raw_ms { "GO -- pipelined streaming bit-plane beats raw bf16 from SSD" } else { "NO-GO" }
    );
}
```

- [ ] **Step 2: Ensure the sample exists**

Run:

```bash
test -f /tmp/rllm-bf16-sample.bin || \
  cargo test -p rllm-runtime --release dump_bf16_embedding_sample -- --ignored --nocapture
```
Expected: file present (525,336,576 bytes).

- [ ] **Step 3: Run the bench (release) and capture the verdict**

Run:

```bash
cargo test -p rllm-runtime --release streaming_gemv_capacity_bound_bench -- --ignored --nocapture
```
Expected: the `=== R148 REESTREAM ... BENCH ===` block. **Record verbatim**: raw ms + GB/s, pipelined ms + GB/s, the speedup vs raw, and how it compares to R147's 1.13×. These feed Task 3.

- [ ] **Step 4: Commit**

```bash
git add crates/rllm-runtime/src/streaming/bitplane_stream.rs
git commit -m "test(runtime): R148 pipelined streaming capacity-bound bench + measurement

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Trial report + index + memory

**Files:**
- Create: `docs/benchmarks/trials/<verdict-folder>/2026-06-19-r148-reestream-streaming-gemv.md` (`success/` if it beats raw + improves on the scout, else `inconclusive/`)
- Modify: `docs/benchmarks/trials/index.md`
- Modify: `/Users/deansanbhnanwr/.claude/projects/-Users-deansanbhnanwr-Projects-rllm/memory/rllm-speed-thesis-streaming-vs-resident.md`

**Interfaces:**
- Consumes: the measured numbers from Task 2 Step 3; the template; the R146/R147 trial as a format reference.

- [ ] **Step 1: Read the template and the R146/R147 trial**

Run: `cat docs/benchmarks/templates/trial-report.md docs/benchmarks/trials/success/2026-06-19-r146-decode16-and-r147-capacity-bound.md`
Expected: the required section structure.

- [ ] **Step 2: Write the trial report**

Create the report in the verdict folder. Fill from Task 2's numbers:
- **Scope → REE kernel:** `REESTREAM (working name; Erik's final call)`. Mode: `experimental (capacity-bound streaming GEMV, pipelined)`. Artifact: `Llama-3.2-1B-Instruct-raw.rllm` embedding, block-framed, replicated > RAM. Device: Apple A18 Pro. Bottleneck tag: storage bandwidth.
- **Hypothesis:** double-buffer pipelining (overlap disk read with decode) hides the decode under the cold SSD, pushing R147's 1.13× toward the ~1.23× pure-byte ratio.
- **Results:** raw-bf16 vs R148-pipelined ms + GB/s on the >RAM cold stream; the speedup; comparison to R147's 1.13×; lossless parity bit-identical (test green).
- **Analysis:** did pipelining close the un-pipelined residual (1.50→toward 1.62 GB/s effective)? Place in the R140–R148 arc as the production capacity-bound kernel.
- **Decision:** GO if pipelined beats raw and ≥ R147 scout; otherwise inconclusive (state the residual honestly).
- **Next Experiment:** R149+ full model wiring (pack a model bit-plane, register `codec_for_id`, stream the forward pass, generation tok/s on a model > RAM).

- [ ] **Step 3: Add the index row**

In `docs/benchmarks/trials/index.md`, add an R148 row mirroring the R146/R147 row's columns. Baseline = R147 un-pipelined 1.13×; result = R148 pipelined `<speedup>`× vs raw + verdict.

- [ ] **Step 4: Update memory**

Append the measured R148 number to `rllm-speed-thesis-streaming-vs-resident.md`: pipelined streaming GEMV `<comp_ms>` vs raw bf16 `<raw_ms>` ms on the >RAM cold stream (`<speedup>`×, vs R147 un-pipelined 1.13×), lossless bit-identical → `<verdict>`. State whether pipelining hid the decode, and the next lever (R149 full model wiring).

- [ ] **Step 5: Commit**

```bash
git add docs/benchmarks/trials/ "/Users/deansanbhnanwr/.claude/projects/-Users-deansanbhnanwr-Projects-rllm/memory/rllm-speed-thesis-streaming-vs-resident.md"
git commit -m "docs(bench): R148 REESTREAM pipelined streaming GEMV trial (<verdict>) + index + memory

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Verification (end-to-end)

1. `cargo test -p rllm-runtime --lib` → green, including `streaming_gemv_matches_reference_bit_for_bit` (lossless).
2. `cargo build` → compiles, no new dependencies.
3. The `#[ignore]` bench printed the `=== R148 REESTREAM ... BENCH ===` block with a real verdict on the >RAM cold stream.
4. Trial report in the verdict folder; `index.md` has the R148 row; memory updated.
5. `git grep -n "streaming_bitplane_gemv" crates/rllm-runtime/src | grep -v bitplane_stream.rs` → **no hits** (no model wiring yet; kernel is the measured building block).

## Out of scope (R149+ follow-on — do NOT build here)

- Full model wiring: pack a model with `rtc-bitplane-v1`, register in `codec_for_id`, stream the real forward pass, generation tok/s on a model > RAM.
- Multi-threaded decode (one overlapped decoder suffices when disk-bound).
- General `(hidden, w)`; q8 layers; GPU; KV-cache.
