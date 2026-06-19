# R144 — Fused REEPLANE decode→bfdot GEMV (Phase C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fused lm-head GEMV that decodes each weight row from the resident `rtc-bitplane-v1` planes into an L1 scratch and `bfdot`s it against the activation (bf16 never hits DRAM), then benchmark it vs the plain bf16 GEMV on the real 525 MB Llama 1B embedding: time, resident bytes, bit-identical logit parity, verdict.

**Architecture:** A no-alloc `decode_neon_w5_into` wrapper (rtc-codec) decodes one row into a caller buffer. A fused kernel `lm_head_logits_rows_bitplane` (rllm-runtime, `include!`d into `streaming/mod.rs` so it shares scope with `Bf16DotActivation`/`bf16_row_dot_bf16`) loops rows: decode row → scratch, `bfdot`. A `#[ignore]` bench compares it to `lm_head_logits_rows_bf16`.

**Tech Stack:** Rust (stable), crates `rtc-codec` + `rllm-runtime`, `std::arch::aarch64` NEON. No new dependencies (rllm-runtime already depends on rtc-codec).

## Global Constraints

- Pure Rust, **no new dependencies**; `cargo build` only. NEON via `std::arch`.
- **Lossless / bit-identical (hard rule):** fused logits bit-identical to the plain-bf16 path (same exact weights, same `Bf16DotActivation` kernel). A unit test asserts exact f32 equality.
- **Reuse, don't duplicate (DRY):** reuse `decode_w5_neon_inner` (rtc-codec, already writes into `out: &mut [u8]`), `Bf16DotActivation` + `bf16_row_dot_bf16` (rllm-runtime streaming/mod.rs). No new decode or dot logic.
- **Row byte-alignment:** the kernel requires `hidden * 5 % 8 == 0` (true for hidden=2048 → 1280-byte rows). Pass **open-ended** plane slices `&idx_plane[r*1280..]`, `&residuals[r*hidden..]` (NOT row-exact) so NEON group loads stay in-bounds; the last row is covered by the kernel's `simd_groups` guard.
- **REE kernel working name: REEFUSE-PLANE-DOT** (Erik's final call) — trial Scope line.
- **Scope:** GEMV-level fused kernel + benchmark only. NO `--fast` wiring, NO `codec_for_id` registration, NO model packing — gated follow-up.
- **Threshold:** GO = fused faster AND smaller; MARGINAL = smaller, ≈same speed; NO-GO = slower. Report honestly.

## File Structure

- **Modify** `crates/rtc-codec/src/bitplane.rs` — add `pub fn decode_neon_w5_into`; refactor `decode_neon_w5` to call it (DRY); add a parity test.
- **Create** `crates/rllm-runtime/src/streaming/bitplane_gemv.rs` — `lm_head_logits_rows_bitplane` + a `#[cfg(test)] mod` (lossless parity unit test + `#[ignore]` bench).
- **Modify** `crates/rllm-runtime/src/streaming/mod.rs` — add `include!("bitplane_gemv.rs");`.
- **Create** `docs/benchmarks/trials/<verdict-folder>/2026-06-19-r144-reefuse-plane-dot-gemv.md` — trial report.
- **Modify** `docs/benchmarks/trials/index.md` — R144 row.
- **Modify** memory `rllm-speed-thesis-streaming-vs-resident.md` — measured R144 number.

---

### Task 1: `decode_neon_w5_into` no-alloc wrapper (rtc-codec)

**Files:**
- Modify: `crates/rtc-codec/src/bitplane.rs`

**Interfaces:**
- Consumes: existing `decode_w5_neon_inner(palette, idx_plane, residuals, n, out: &mut [u8])`.
- Produces: `#[cfg(target_arch = "aarch64")] pub fn decode_neon_w5_into(palette: &[u8], idx_plane: &[u8], residuals: &[u8], n: usize, out: &mut [u8])` — decodes `n` w=5 weights into `out` (len ≥ `n*2`), no allocation.

- [ ] **Step 1: Write the failing parity test**

Add to the `#[cfg(test)] mod tests` in `crates/rtc-codec/src/bitplane.rs`:

```rust
#[cfg(target_arch = "aarch64")]
#[test]
fn decode_neon_w5_into_matches_allocating_variant() {
    let codec = BitplaneCodec;
    for &n in &[32usize, 33, 100, 4096, 4099] {
        let bytes = make_bf16(32, n);
        let meta = EncodeMeta { name: "w".into(), shape: vec![n as u64], dtype: "bf16".into() };
        let enc = codec.encode(&bytes, &meta).unwrap();
        let p = enc.data[14] as usize;
        let mut off = 16;
        let palette = &enc.data[off..off + p];
        off += p;
        let idx_bytes = (n * 5 + 7) / 8;
        let idx_plane = &enc.data[off..off + idx_bytes];
        off += idx_bytes;
        let residuals = &enc.data[off..off + n];

        let alloc = decode_neon_w5(palette, idx_plane, residuals, n);
        let mut into = vec![0u8; n * 2];
        decode_neon_w5_into(palette, idx_plane, residuals, n, &mut into);
        assert_eq!(into, alloc, "n={n}: decode_neon_w5_into must match decode_neon_w5");
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p rtc-codec --lib decode_neon_w5_into -- --nocapture`
Expected: FAIL — `cannot find function decode_neon_w5_into`.

- [ ] **Step 3: Add the wrapper and refactor `decode_neon_w5` to use it**

In `crates/rtc-codec/src/bitplane.rs`, replace the existing `decode_neon_w5` function:

```rust
/// NEON `w=5` bit-plane decode to bf16 bytes. Bit-identical to scalar `decode`.
#[cfg(target_arch = "aarch64")]
pub fn decode_neon_w5(palette: &[u8], idx_plane: &[u8], residuals: &[u8], n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n * 2];
    decode_neon_w5_into(palette, idx_plane, residuals, n, &mut out);
    out
}

/// NEON `w=5` bit-plane decode into a caller-provided buffer (`out.len() >= n*2`),
/// no allocation. Used by the fused per-row GEMV so decode lands in an L1 scratch.
#[cfg(target_arch = "aarch64")]
pub fn decode_neon_w5_into(
    palette: &[u8],
    idx_plane: &[u8],
    residuals: &[u8],
    n: usize,
    out: &mut [u8],
) {
    assert!(out.len() >= n * 2, "decode_neon_w5_into: out too small");
    unsafe { decode_w5_neon_inner(palette, idx_plane, residuals, n, out) };
}
```

- [ ] **Step 4: Run the parity test to verify it passes**

Run: `cargo test -p rtc-codec --lib decode_neon_w5 -- --nocapture`
Expected: PASS — `decode_neon_w5_into_matches_allocating_variant` and the existing `decode_neon_w5_matches_scalar_bit_for_bit`.

- [ ] **Step 5: Run the full crate suite**

Run: `cargo test -p rtc-codec`
Expected: PASS — all green.

- [ ] **Step 6: Commit**

```bash
git add crates/rtc-codec/src/bitplane.rs
git commit -m "feat(rtc-codec): decode_neon_w5_into no-alloc wrapper (R144 fused GEMV prep)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Fused `lm_head_logits_rows_bitplane` + lossless parity (rllm-runtime)

**Files:**
- Create: `crates/rllm-runtime/src/streaming/bitplane_gemv.rs`
- Modify: `crates/rllm-runtime/src/streaming/mod.rs` (add `include!`)

**Interfaces:**
- Consumes: `rtc_codec::decode_neon_w5_into` (Task 1); `Bf16DotActivation`, `bf16_row_dot_bf16`/`row_dot` (mod.rs, same scope via `include!`); `rtc_codec::BitplaneCodec` (tests); `lm_head_logits_rows_bf16` (mod.rs, the plain baseline).
- Produces: `#[cfg(target_arch = "aarch64")] fn lm_head_logits_rows_bitplane(last_hidden: &[f32], palette: &[u8], idx_plane: &[u8], residuals: &[u8], hidden: usize, row_offset: usize, out: &mut [f32])`.

- [ ] **Step 1: Wire the new file into `mod.rs`**

In `crates/rllm-runtime/src/streaming/mod.rs`, add after `include!("kernels.rs");` (and before `include!("tests.rs");`):

```rust
include!("bitplane_gemv.rs");
```

- [ ] **Step 2: Write the failing lossless-parity test**

Create `crates/rllm-runtime/src/streaming/bitplane_gemv.rs` with the kernel doc + a test module. Start with the test only so it fails to compile:

```rust
//! Fused REEPLANE decode -> bfdot GEMV (R144, Phase C).
//!
//! Computes lm-head logits directly from resident rtc-bitplane-v1 planes: per
//! row, decode the row's bf16 weights into an L1 scratch (no DRAM
//! materialization) and bfdot against the once-converted activation. Reuses
//! rtc-codec's `decode_neon_w5_into` (R143) + this module's `Bf16DotActivation`
//! / `bf16_row_dot_bf16` (R141). Not yet wired into the runtime — the Phase C
//! building block measured by the R144 bench.

#[cfg(all(test, target_arch = "aarch64"))]
mod bitplane_gemv_tests {
    use super::*;
    use rtc_codec::{BitplaneCodec, EncodeMeta, TensorCodec};

    // vocab*hidden bf16 weights cycling 32 exponents (=> w=5), random mantissa.
    fn make_embedding(vocab: usize, hidden: usize) -> Vec<u8> {
        let mut state = 0xDEAD_BEEF_1234_5678u64;
        let mut out = Vec::with_capacity(vocab * hidden * 2);
        for k in 0..vocab * hidden {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let exp = (96 + (k % 32)) as u16 & 0xFF;
            let sign = ((state >> 31) & 1) as u16;
            let mant = (state & 0x7F) as u16;
            let bits = (sign << 15) | (exp << 7) | mant;
            out.extend_from_slice(&bits.to_le_bytes());
        }
        out
    }

    #[test]
    fn fused_bitplane_gemv_matches_plain_bf16_bit_for_bit() {
        let (vocab, hidden) = (64usize, 2048usize);
        let bf16 = make_embedding(vocab, hidden);
        let enc = BitplaneCodec
            .encode(
                &bf16,
                &EncodeMeta { name: "e".into(), shape: vec![(vocab * hidden) as u64], dtype: "bf16".into() },
            )
            .unwrap();
        let p = enc.data[14] as usize;
        assert_eq!(enc.data[15], 5, "expected w=5");
        let mut off = 16;
        let palette = &enc.data[off..off + p];
        off += p;
        let idx_bytes = (vocab * hidden * 5 + 7) / 8;
        let idx_plane = &enc.data[off..off + idx_bytes];
        off += idx_bytes;
        let residuals = &enc.data[off..off + vocab * hidden];

        // deterministic activation
        let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.01).sin() * 0.5).collect();

        let mut plain = vec![0f32; vocab];
        lm_head_logits_rows_bf16(&act, &bf16, hidden, 0, &mut plain);

        let mut fused = vec![0f32; vocab];
        lm_head_logits_rows_bitplane(&act, palette, idx_plane, residuals, hidden, 0, &mut fused);

        assert_eq!(fused, plain, "fused bit-plane GEMV must equal plain bf16 GEMV bit-for-bit");
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `cargo test -p rllm-runtime --lib fused_bitplane_gemv -- --nocapture`
Expected: FAIL — `cannot find function lm_head_logits_rows_bitplane`.

- [ ] **Step 4: Implement the fused kernel**

Insert ABOVE the `#[cfg(all(test, target_arch = "aarch64"))] mod bitplane_gemv_tests` block in `bitplane_gemv.rs`:

```rust
/// Fused bit-plane lm-head GEMV. `palette`/`idx_plane`/`residuals` are the
/// `rtc-bitplane-v1` planes (w=5) of a row-major weight matrix with `hidden`
/// weights per row. For each output row, decode the row's bf16 weights into a
/// reused L1 scratch and `bfdot` against the once-converted activation; bf16 is
/// never written to DRAM. Writes `out.len()` logits starting at `row_offset`.
/// Requires `hidden * 5 % 8 == 0` (rows byte-aligned). Not yet runtime-wired.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
fn lm_head_logits_rows_bitplane(
    last_hidden: &[f32],
    palette: &[u8],
    idx_plane: &[u8],
    residuals: &[u8],
    hidden: usize,
    row_offset: usize,
    out: &mut [f32],
) {
    debug_assert_eq!((hidden * 5) % 8, 0, "bit-plane row index plane must be byte-aligned");
    let row_idx_bytes = hidden * 5 / 8; // 1280 for hidden=2048
    let act = Bf16DotActivation::new(last_hidden);
    let mut scratch = vec![0u8; hidden * 2];
    for (r, logit) in out.iter_mut().enumerate() {
        let row = row_offset + r;
        // Open-ended slices: NEON group loads may read a few bytes past the row's
        // span into the next row (in-bounds); the last row is covered by the
        // decode kernel's simd_groups guard + scalar tail.
        let idx = &idx_plane[row * row_idx_bytes..];
        let res = &residuals[row * hidden..];
        rtc_codec::decode_neon_w5_into(palette, idx, res, hidden, &mut scratch);
        *logit = act.row_dot(&scratch, hidden);
    }
}
```

- [ ] **Step 5: Run the parity test to verify it passes**

Run: `cargo test -p rllm-runtime --lib fused_bitplane_gemv -- --nocapture`
Expected: PASS — `fused_bitplane_gemv_matches_plain_bf16_bit_for_bit` (fused logits exactly equal the plain bf16 path: lossless e2e at unit scale).

- [ ] **Step 6: Run the runtime lib suite**

Run: `cargo test -p rllm-runtime --lib`
Expected: PASS — all existing tests + the new parity test green.

- [ ] **Step 7: Commit**

```bash
git add crates/rllm-runtime/src/streaming/bitplane_gemv.rs crates/rllm-runtime/src/streaming/mod.rs
git commit -m "feat(runtime): fused bit-plane decode->bfdot lm-head GEMV, lossless parity (R144 REEFUSE-PLANE-DOT)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Benchmark + measurement

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/bitplane_gemv.rs` (add `#[ignore]` bench to the test module)

**Interfaces:**
- Consumes: `lm_head_logits_rows_bitplane` (Task 2), `lm_head_logits_rows_bf16` (mod.rs); `/tmp/rllm-bf16-sample.bin` (the 262.7M-weight embedding = vocab 128256 × hidden 2048).
- Produces: plain vs fused ms/GEMV, resident bytes, exact-parity confirmation, verdict (printed; transcribed into Task 4).

- [ ] **Step 1: Add the `#[ignore]` bench**

Add to the `bitplane_gemv_tests` module in `bitplane_gemv.rs`:

```rust
#[test]
#[ignore]
fn fused_bitplane_gemv_feasibility() {
    use rtc_codec::{BitplaneCodec, EncodeMeta, TensorCodec};
    // Both paths use bfdot for an apples-to-apples compute comparison.
    std::env::set_var("RLLM_Q8_ACTIVATION", "1");
    std::env::set_var("RLLM_BF16_DOT", "1");

    let bf16 = std::fs::read("/tmp/rllm-bf16-sample.bin")
        .expect("run dump_bf16_embedding_sample first");
    let hidden = 2048usize;
    let n_weights = bf16.len() / 2;
    let vocab = n_weights / hidden;
    assert_eq!(vocab * hidden, n_weights, "sample must be vocab*2048");

    let enc = BitplaneCodec
        .encode(
            &bf16,
            &EncodeMeta { name: "e".into(), shape: vec![n_weights as u64], dtype: "bf16".into() },
        )
        .unwrap();
    assert_eq!(enc.data[15], 5, "expected w=5");
    let p = enc.data[14] as usize;
    let mut off = 16;
    let palette = enc.data[off..off + p].to_vec();
    off += p;
    let idx_bytes = (n_weights * 5 + 7) / 8;
    let idx_plane = enc.data[off..off + idx_bytes].to_vec();
    off += idx_bytes;
    let residuals = enc.data[off..off + n_weights].to_vec();

    let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();

    // Correctness + parity on the real sample.
    let mut plain = vec![0f32; vocab];
    lm_head_logits_rows_bf16(&act, &bf16, hidden, 0, &mut plain);
    let mut fused = vec![0f32; vocab];
    lm_head_logits_rows_bitplane(&act, &palette, &idx_plane, &residuals, hidden, 0, &mut fused);
    assert_eq!(fused, plain, "fused must equal plain bf16 (lossless) on the real embedding");

    let iters = 5;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        lm_head_logits_rows_bf16(&act, &bf16, hidden, 0, &mut plain);
        std::hint::black_box(&plain);
    }
    let plain_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let t = std::time::Instant::now();
    for _ in 0..iters {
        lm_head_logits_rows_bitplane(&act, &palette, &idx_plane, &residuals, hidden, 0, &mut fused);
        std::hint::black_box(&fused);
    }
    let fused_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let bf16_mb = bf16.len() as f64 / 1e6;
    let plane_mb = (palette.len() + idx_plane.len() + residuals.len()) as f64 / 1e6;
    let speedup = plain_ms / fused_ms;
    let ram_save = (1.0 - plane_mb / bf16_mb) * 100.0;
    let verdict = if speedup >= 1.05 && plane_mb < bf16_mb {
        "GO (faster + smaller)"
    } else if plane_mb < bf16_mb && speedup >= 0.95 {
        "MARGINAL (smaller, ~same speed)"
    } else {
        "NO-GO (slower)"
    };
    eprintln!(
        "\n=== R144 REEFUSE-PLANE-DOT fused GEMV FEASIBILITY (single-core) ===\n\
         vocab={vocab} hidden={hidden}  (lossless parity: OK)\n\
         plain bf16 GEMV:   {plain_ms:.1} ms/token  (resident {bf16_mb:.0} MB)\n\
         fused bit-plane:   {fused_ms:.1} ms/token  (resident {plane_mb:.0} MB, {ram_save:.0}% less)\n\
         speedup: {speedup:.2}x\n\
         VERDICT: {verdict}\n",
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
cargo test -p rllm-runtime --release fused_bitplane_gemv_feasibility -- --ignored --nocapture
```
Expected: the `=== R144 REEFUSE-PLANE-DOT fused GEMV FEASIBILITY ===` block. **Record verbatim**: plain ms, fused ms, resident MB (both, ~525 vs ~427), the speedup, the `VERDICT`. These feed Task 4.

- [ ] **Step 4: Commit**

```bash
git add crates/rllm-runtime/src/streaming/bitplane_gemv.rs
git commit -m "test(runtime): R144 fused bit-plane GEMV feasibility bench + measurement

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Trial report + index + memory

**Files:**
- Create: `docs/benchmarks/trials/<verdict-folder>/2026-06-19-r144-reefuse-plane-dot-gemv.md` (`success/` if GO, `inconclusive/` if MARGINAL, `failed/` if NO-GO)
- Modify: `docs/benchmarks/trials/index.md`
- Modify: `/Users/deansanbhnanwr/.claude/projects/-Users-deansanbhnanwr-Projects-rllm/memory/rllm-speed-thesis-streaming-vs-resident.md`

**Interfaces:**
- Consumes: the measured numbers + verdict from Task 3 Step 3; the template `docs/benchmarks/templates/trial-report.md`; the R143 trial as a format reference.

- [ ] **Step 1: Read the template and R143 trial**

Run: `cat docs/benchmarks/templates/trial-report.md docs/benchmarks/trials/success/2026-06-19-r143-reeplane-bitplane-codec.md`
Expected: the required section structure.

- [ ] **Step 2: Write the trial report**

Create the report in the verdict-matching folder. Fill from Task 3's numbers:
- **Scope → REE kernel:** `REEFUSE-PLANE-DOT (working name; Erik's final call)`. Mode: `experimental (compressed-resident fused GEMV, Phase C)`. Artifact: `Llama-3.2-1B-Instruct-raw.rllm` embedding (vocab 128256 × hidden 2048). Device: Apple A18 Pro. Bottleneck tag: IO/decode + CPU arithmetic.
- **Hypothesis:** decode→bfdot from a resident 13-bit bit-plane buffer beats read-bf16→bfdot, lossless.
- **Results:** the table — plain vs fused ms/token, resident MB (525 vs ~427, 19% less), speedup, exact-parity OK.
- **Analysis:** is the 19% DRAM saving worth the per-row decode compute, single-core? Place vs R143 (decode-throughput GO) — this is the e2e-of-GEMV confirmation (or not). Note both paths used bfdot. Whatever the verdict: the lossless parity (fused == plain bit-for-bit) is the proof that compressed-resident is exact.
- **Decision:** the verdict. If GO/MARGINAL → next is `--fast` wiring + real generation (own spec). If NO-GO → the decode compute eats the bandwidth saving single-core; revisit with multi-thread or note the frontier.
- **Next Experiment:** `--fast` lm-head wiring + model packing if GO/MARGINAL.

- [ ] **Step 3: Add the index row**

In `docs/benchmarks/trials/index.md`, add an R144 row mirroring R143's columns. Baseline = plain bf16 GEMV `<plain_ms>` ms; result = fused `<fused_ms>` ms, `<speedup>`×, 19% less RAM, lossless parity + verdict.

- [ ] **Step 4: Update memory**

Append the measured R144 number to `rllm-speed-thesis-streaming-vs-resident.md`: fused bit-plane GEMV `<fused_ms>` vs plain bf16 `<plain_ms>` ms/token single-core (`<speedup>`×), resident 525→~427 MB (19%), lossless parity exact → `<verdict>`. State whether the lossless-compressed-resident e2e win is demonstrated at the GEMV level, and the next lever.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmarks/trials/ "/Users/deansanbhnanwr/.claude/projects/-Users-deansanbhnanwr-Projects-rllm/memory/rllm-speed-thesis-streaming-vs-resident.md"
git commit -m "docs(bench): R144 REEFUSE-PLANE-DOT fused GEMV trial (<verdict>) + index + memory

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Verification (end-to-end)

1. `cargo test -p rtc-codec` → green, including `decode_neon_w5_into_matches_allocating_variant`.
2. `cargo test -p rllm-runtime --lib` → green, including `fused_bitplane_gemv_matches_plain_bf16_bit_for_bit` (lossless parity).
3. `cargo build` → compiles, no new dependencies.
4. The `#[ignore]` bench printed the `=== R144 REEFUSE-PLANE-DOT ... ===` block with a real verdict on the 525 MB sample.
5. Trial report in the verdict folder with measured numbers; `index.md` has the R144 row; memory updated.
6. `git grep -n "lm_head_logits_rows_bitplane" crates/rllm-runtime/src | grep -v "bitplane_gemv.rs"` → **no hits** outside the new file (no runtime wiring; the kernel stays a measured building block).

## Out of scope (gated follow-up — do NOT build here)

- `--fast` lm-head wiring of the fused kernel; packing models with `rtc-bitplane-v1`; `codec_for_id` registration.
- General `(hidden, w)` shapes; multi-threaded fused GEMV (the parallel lm-head wrapper can call it later).
- Beating the ~1.2×/19% ceiling; q8/q4 layers; KV-cache; GPU.
