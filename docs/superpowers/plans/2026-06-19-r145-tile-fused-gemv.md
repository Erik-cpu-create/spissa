# R145 — Tile-fused decode⟷bfdot lm-head GEMV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tile-fused kernel that decodes each weight row's bit-plane straight into NEON registers and `bfdot`-accumulates it in the same pass (no L1 scratch round-trip), then benchmark it multi-core vs plain bf16 (9.6 ms floor) and report GO/MARGINAL/NO-GO.

**Architecture:** Per row, loop 8-weight groups: decode each group into a `uint16x8` bf16 vector (reusing R143's proven group logic) and feed it to a `bfdot` accumulator chain in R141's exact 4-chain order — so the fused result is **bit-identical** to R144's decode-then-bfdot, while removing the scratch store/load and letting the out-of-order engine overlap decode with the dot. Multi-thread by splitting vocab rows.

**Tech Stack:** Rust (stable), crate `rllm-runtime`, `std::arch::aarch64` NEON + inline-asm `bfdot`. No new dependencies.

## Global Constraints

- Pure Rust, **no new dependencies**; `cargo build` only. NEON/bf16 via `std::arch` / inline asm.
- **Lossless / bit-identical (hard rule):** the fused kernel's logits are **bit-identical** to R144's `lm_head_logits_rows_bitplane` (same exact weights, same R141 4-chain accumulation order, no scalar tail for `hidden=2048`). Parity test asserts exact f32 equality with both paths under bfdot.
- **Reuse, don't duplicate (DRY):** reuse R143's 8-wide group-decode sequence (the `vtbl1`/window/`vtbl4`/join steps from `decode_w5_neon_inner`), R141's `bfdot` asm shape + `convert_f32_to_bf16`. No new decode or dot math.
- **Constraints:** `hidden % 32 == 0` (true for 2048; debug_assert) and `w = 5`. Requires FEAT_BF16 (caller checks `bf16_dot_available()`). Open-ended plane slices so group loads stay in-bounds; per-group bounds-check covers the final row.
- **REE kernel working name: REEFUSE-PLANE-DOT v2** (Erik's final call) — trial Scope line.
- **Scope:** kernel + multi-core bench only. NO `--fast` wiring / `codec_for_id` / packing — gated follow-up. 16-wide `vqtbl2q` decode is the documented next lever **only if** 8-wide fusion misses the gate.
- **Gate (from the scout):** beat plain bf16 multi-core (~9.6 ms/token at 6 threads). GO ≤9.6, MARGINAL ~9.6–12, NO-GO >12.

## File Structure

- **Modify** `crates/rllm-runtime/src/streaming/bitplane_gemv.rs` — add `bitplane_row_dot_bfdot` (fused row kernel) + `lm_head_logits_bitplane_fused` (GEMV wrapper); extend the test module with the bit-identical parity test and the multi-core bench (the scout bench already lives here).
- **Create** `docs/benchmarks/trials/<verdict-folder>/2026-06-19-r145-reefuse-plane-dot-v2-gemv.md` — trial report.
- **Modify** `docs/benchmarks/trials/index.md` — R145 row.
- **Modify** memory `rllm-speed-thesis-streaming-vs-resident.md` — measured R145 number.

---

### Task 1: Tile-fused kernel `bitplane_row_dot_bfdot` + bit-identical parity

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/bitplane_gemv.rs`

**Interfaces:**
- Consumes: `convert_f32_to_bf16`, `bf16_dot_available`, `bf16_row_dot_bf16` (mod.rs, R141); `rtc_codec::decode_neon_w5` (R143). The parity oracle is `decode_neon_w5` + `bf16_row_dot_bf16` (env-free, always bfdot) — not the env-gated R144 wrapper.
- Produces: `#[cfg(target_arch = "aarch64")] fn lm_head_logits_bitplane_fused(last_hidden: &[f32], palette: &[u8], idx_plane: &[u8], residuals: &[u8], hidden: usize, row_offset: usize, out: &mut [f32])`; helper `unsafe fn bitplane_row_dot_bfdot(act_bf16: &[u16], pal: &[u8;32], idx_row: &[u8], res_row: &[u8], hidden: usize) -> f32`.

- [ ] **Step 1: Write the failing bit-identical parity test**

Add to `mod bitplane_gemv_tests` in `crates/rllm-runtime/src/streaming/bitplane_gemv.rs`:

The reference is built **without env mutation** (so it is safe under `cargo test`'s
parallel runner): decode each row with `rtc_codec::decode_neon_w5` and dot it with
`bf16_row_dot_bf16` (mod.rs, R141) — which always uses `bfdot`, independent of the
`RLLM_BF16_DOT` env gate. The fused kernel decodes the same lossless weights and
uses the same 4-chain bfdot order, so the two are bit-identical.

```rust
#[test]
fn fused_kernel_matches_reference_bit_for_bit() {
    if !bf16_dot_available() {
        return; // FEAT_BF16 required; no-op on non-bf16 hardware
    }
    let (vocab, hidden) = (96usize, 2048usize);
    let bf16 = make_embedding(vocab, hidden);
    let enc = BitplaneCodec
        .encode(&bf16, &EncodeMeta { name: "e".into(), shape: vec![(vocab * hidden) as u64], dtype: "bf16".into() })
        .unwrap();
    let p = enc.data[14] as usize;
    assert_eq!(enc.data[15], 5);
    let mut off = 16;
    let palette = &enc.data[off..off + p];
    off += p;
    let idx_bytes = (vocab * hidden * 5 + 7) / 8;
    let idx_plane = &enc.data[off..off + idx_bytes];
    off += idx_bytes;
    let residuals = &enc.data[off..off + vocab * hidden];
    let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.017).cos() * 0.4).collect();

    // Env-free bfdot reference: decode each row, then R141 bf16_row_dot_bf16.
    let act_bf16 = convert_f32_to_bf16(&act);
    let row_idx_bytes = hidden * 5 / 8;
    let mut reference = vec![0f32; vocab];
    for (r, slot) in reference.iter_mut().enumerate() {
        let decoded = rtc_codec::decode_neon_w5(
            palette,
            &idx_plane[r * row_idx_bytes..],
            &residuals[r * hidden..],
            hidden,
        );
        *slot = unsafe { bf16_row_dot_bf16(&act_bf16, &decoded, hidden) };
    }

    let mut fused = vec![0f32; vocab];
    lm_head_logits_bitplane_fused(&act, palette, idx_plane, residuals, hidden, 0, &mut fused);

    assert_eq!(fused, reference, "fused kernel must equal decode+bfdot reference bit-for-bit");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p rllm-runtime --lib fused_kernel_matches_reference -- --nocapture`
Expected: FAIL — `cannot find function lm_head_logits_bitplane_fused`.

- [ ] **Step 3: Implement the fused row kernel + wrapper**

Insert into `bitplane_gemv.rs` ABOVE the `#[cfg(all(test, target_arch = "aarch64"))] mod bitplane_gemv_tests` block:

```rust
/// One logit = dot(activation, decoded weight row), fused: each 8-weight group is
/// decoded straight into a `uint16x8` bf16 vector (R143 logic) and `bfdot`-ed into
/// one of 4 accumulator chains in R141's order — no L1 scratch, decode overlaps
/// the dot via out-of-order execution. Bit-identical to R144 decode-then-bfdot.
/// SAFETY: caller guarantees FEAT_BF16; `hidden % 32 == 0`; `act_bf16.len() >= hidden`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "bf16")]
unsafe fn bitplane_row_dot_bfdot(
    act_bf16: &[u16],
    pal: &[u8; 32],
    idx_row: &[u8],
    res_row: &[u8],
    hidden: usize,
) -> f32 {
    use std::arch::aarch64::*;
    debug_assert_eq!(hidden % 32, 0);
    let pal_tbl = uint8x8x4_t(
        vld1_u8(pal.as_ptr()),
        vld1_u8(pal.as_ptr().add(8)),
        vld1_u8(pal.as_ptr().add(16)),
        vld1_u8(pal.as_ptr().add(24)),
    );
    let bidx_hi: [u8; 8] = [0, 0, 1, 1, 2, 3, 3, 4];
    let bidx_lo: [u8; 8] = [1, 1, 2, 2, 3, 4, 4, 5];
    let neg_shift: [i16; 8] = [-11, -6, -9, -4, -7, -10, -5, -8];
    let vhi = vld1_u8(bidx_hi.as_ptr());
    let vlo = vld1_u8(bidx_lo.as_ptr());
    let vshift = vld1q_s16(neg_shift.as_ptr());
    let mask5 = vdupq_n_u16(0x1f);
    let mask80 = vdupq_n_u16(0x80);
    let mask7f = vdupq_n_u16(0x7f);
    let idx_len = idx_row.len();

    // Decode 8 weights of group g into a uint16x8 of bf16. Bounds-checked load so
    // the final row's last group (whose 8-byte load runs past the plane) is safe;
    // the branch is predictable (taken only at the very end).
    let decode_group = |g: usize| -> uint16x8_t {
        let off = g * 5;
        let grp = if off + 8 <= idx_len {
            vld1_u8(idx_row.as_ptr().add(off))
        } else {
            let mut buf = [0u8; 8];
            let avail = idx_len - off;
            core::ptr::copy_nonoverlapping(idx_row.as_ptr().add(off), buf.as_mut_ptr(), avail);
            vld1_u8(buf.as_ptr())
        };
        let hi = vtbl1_u8(grp, vhi);
        let lo = vtbl1_u8(grp, vlo);
        let window = vorrq_u16(vshlq_n_u16(vmovl_u8(hi), 8), vmovl_u8(lo));
        let idx16 = vandq_u16(vshlq_u16(window, vshift), mask5);
        let idx8 = vmovn_u16(idx16);
        let exp8 = vtbl4_u8(pal_tbl, idx8);
        let res8 = vld1_u8(res_row.as_ptr().add(g * 8));
        let res16 = vmovl_u8(res8);
        let exp16 = vmovl_u8(exp8);
        let sign = vshlq_n_u16(vandq_u16(res16, mask80), 8);
        let ep = vshlq_n_u16(exp16, 7);
        let mant = vandq_u16(res16, mask7f);
        vorrq_u16(vorrq_u16(sign, ep), mant)
    };

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let groups = hidden / 8; // multiple of 4 since hidden % 32 == 0
    let aptr = act_bf16.as_ptr();
    let mut g = 0usize;
    while g < groups {
        let w0 = decode_group(g);
        let w1 = decode_group(g + 1);
        let w2 = decode_group(g + 2);
        let w3 = decode_group(g + 3);
        let a0 = vld1q_u16(aptr.add(g * 8));
        let a1 = vld1q_u16(aptr.add((g + 1) * 8));
        let a2 = vld1q_u16(aptr.add((g + 2) * 8));
        let a3 = vld1q_u16(aptr.add((g + 3) * 8));
        core::arch::asm!(
            "bfdot {acc0:v}.4s, {w0:v}.8h, {a0:v}.8h",
            "bfdot {acc1:v}.4s, {w1:v}.8h, {a1:v}.8h",
            "bfdot {acc2:v}.4s, {w2:v}.8h, {a2:v}.8h",
            "bfdot {acc3:v}.4s, {w3:v}.8h, {a3:v}.8h",
            acc0 = inout(vreg) acc0,
            acc1 = inout(vreg) acc1,
            acc2 = inout(vreg) acc2,
            acc3 = inout(vreg) acc3,
            w0 = in(vreg) w0, w1 = in(vreg) w1, w2 = in(vreg) w2, w3 = in(vreg) w3,
            a0 = in(vreg) a0, a1 = in(vreg) a1, a2 = in(vreg) a2, a3 = in(vreg) a3,
            options(nomem, nostack),
        );
        g += 4;
    }
    vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)))
}

/// Fused bit-plane lm-head GEMV: convert the activation to bf16 once, preload the
/// palette, then dot every row directly from the resident planes via
/// `bitplane_row_dot_bfdot`. Bit-identical to R144; not yet runtime-wired.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
fn lm_head_logits_bitplane_fused(
    last_hidden: &[f32],
    palette: &[u8],
    idx_plane: &[u8],
    residuals: &[u8],
    hidden: usize,
    row_offset: usize,
    out: &mut [f32],
) {
    debug_assert_eq!((hidden * 5) % 8, 0);
    let row_idx_bytes = hidden * 5 / 8;
    let act_bf16 = convert_f32_to_bf16(last_hidden);
    let mut pal = [0u8; 32];
    pal[..palette.len()].copy_from_slice(palette);
    for (r, logit) in out.iter_mut().enumerate() {
        let row = row_offset + r;
        let idx = &idx_plane[row * row_idx_bytes..];
        let res = &residuals[row * hidden..];
        *logit = unsafe { bitplane_row_dot_bfdot(&act_bf16, &pal, idx, res, hidden) };
    }
}
```

- [ ] **Step 4: Run the parity test; fix until bit-identical**

Run: `cargo test -p rllm-runtime --lib fused_kernel_matches_reference -- --nocapture`
Expected: PASS. If a logit differs, diff `fused` vs `r144` at the first row: check the `decode_group` constants (must match `decode_w5_neon_inner`) and the 4-chain assignment (group g → acc(g%4), matching R141). Do not proceed until exactly equal.

- [ ] **Step 5: Run the runtime lib suite**

Run: `cargo test -p rllm-runtime --lib`
Expected: PASS — all existing tests + the new parity test (291 total).

- [ ] **Step 6: Commit**

```bash
git add crates/rllm-runtime/src/streaming/bitplane_gemv.rs
git commit -m "feat(runtime): tile-fused decode<->bfdot lm-head kernel, bit-identical (R145 REEFUSE-PLANE-DOT v2)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Multi-core benchmark + measurement

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/bitplane_gemv.rs` (extend the scout bench with the fused kernel)

**Interfaces:**
- Consumes: `lm_head_logits_bitplane_fused` (Task 1), `lm_head_logits_rows_bf16`, the scout's `time_par` helper; `/tmp/rllm-bf16-sample.bin`.
- Produces: a per-thread-count table — plain bf16 vs R144 naive-fused vs R145 tile-fused ms/token — and the verdict.

- [ ] **Step 1: Add the R145 comparison bench**

Add to `mod bitplane_gemv_tests` (the `time_par` helper from the scout already exists in this module):

```rust
#[test]
#[ignore]
fn fused_kernel_multicore_bench() {
    if !bf16_dot_available() {
        eprintln!("FEAT_BF16 not present; skipping");
        return;
    }
    // Set so the plain bf16 path also uses bfdot (apples-to-apples). This is an
    // #[ignore] bench run alone, so the global env mutation is safe here.
    std::env::set_var("RLLM_Q8_ACTIVATION", "1");
    std::env::set_var("RLLM_BF16_DOT", "1");
    let bf16 = std::fs::read("/tmp/rllm-bf16-sample.bin")
        .expect("run dump_bf16_embedding_sample first");
    let hidden = 2048usize;
    let n_weights = bf16.len() / 2;
    let vocab = n_weights / hidden;
    let enc = BitplaneCodec
        .encode(&bf16, &EncodeMeta { name: "e".into(), shape: vec![n_weights as u64], dtype: "bf16".into() })
        .unwrap();
    let p = enc.data[14] as usize;
    let mut off = 16;
    let palette = enc.data[off..off + p].to_vec();
    off += p;
    let idx_bytes = (n_weights * 5 + 7) / 8;
    let idx_plane = enc.data[off..off + idx_bytes].to_vec();
    off += idx_bytes;
    let residuals = enc.data[off..off + n_weights].to_vec();
    let act: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    let bf16_mb = bf16.len() as f64 / 1e6;
    let plane_mb = (palette.len() + idx_plane.len() + residuals.len()) as f64 / 1e6;

    eprintln!(
        "\n=== R145 tile-fused GEMV multi-core BENCH ===\n\
         resident: bf16 {bf16_mb:.0} MB vs bit-plane {plane_mb:.0} MB ({:.0}% less)\n\
         threads | plain bf16 | R145 fused | speedup",
        (1.0 - plane_mb / bf16_mb) * 100.0
    );
    let mut best = f64::INFINITY;
    let mut best_plain = f64::INFINITY;
    for &nt in &[1usize, 2, 4, 6, 8] {
        let plain = time_par(vocab, nt, |base, slice| {
            lm_head_logits_rows_bf16(&act, &bf16, hidden, base, slice)
        });
        let fused = time_par(vocab, nt, |base, slice| {
            lm_head_logits_bitplane_fused(&act, &palette, &idx_plane, &residuals, hidden, base, slice)
        });
        eprintln!(
            "   {nt:2}   |  {plain:6.1} ms |  {fused:6.1} ms | {:.2}x{}",
            plain / fused,
            if fused < plain { "  <-- WIN" } else { "" }
        );
        if fused < best { best = fused; best_plain = plain; }
    }
    let verdict = if best <= best_plain * 1.0 {
        "GO (faster + 19% less RAM, lossless)"
    } else if best <= best_plain * 1.25 {
        "MARGINAL (close; try 16-wide vqtbl2q / strategy B)"
    } else {
        "NO-GO (decode still loses)"
    };
    eprintln!("\nbest fused {best:.1} ms vs best plain {best_plain:.1} ms => VERDICT: {verdict}\n");
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
cargo test -p rllm-runtime --release fused_kernel_multicore_bench -- --ignored --nocapture
```
Expected: the `=== R145 tile-fused GEMV multi-core BENCH ===` table. **Record verbatim**: the per-thread plain vs fused ms, the best of each, the speedup, the resident MB, and the `VERDICT`. These feed Task 3.

- [ ] **Step 4: Commit**

```bash
git add crates/rllm-runtime/src/streaming/bitplane_gemv.rs
git commit -m "test(runtime): R145 tile-fused GEMV multi-core bench + measurement

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Trial report + index + memory

**Files:**
- Create: `docs/benchmarks/trials/<verdict-folder>/2026-06-19-r145-reefuse-plane-dot-v2-gemv.md` (`success/` if GO, `inconclusive/` if MARGINAL, `failed/` if NO-GO)
- Modify: `docs/benchmarks/trials/index.md`
- Modify: `/Users/deansanbhnanwr/.claude/projects/-Users-deansanbhnanwr-Projects-rllm/memory/rllm-speed-thesis-streaming-vs-resident.md`

**Interfaces:**
- Consumes: the measured table + verdict from Task 2 Step 3; the template `docs/benchmarks/templates/trial-report.md`; the R144 trial as a format reference.

- [ ] **Step 1: Read the template and the R144 trial**

Run: `cat docs/benchmarks/templates/trial-report.md docs/benchmarks/trials/failed/2026-06-19-r144-reefuse-plane-dot-gemv.md`
Expected: the required section structure.

- [ ] **Step 2: Write the trial report**

Create the report in the verdict-matching folder. Fill from Task 2's numbers:
- **Scope → REE kernel:** `REEFUSE-PLANE-DOT v2 (working name; Erik's final call)`. Mode: `experimental (compressed-resident fused GEMV, optimized)`. Artifact: `Llama-3.2-1B-Instruct-raw.rllm` embedding. Device: Apple A18 Pro. Bottleneck tag: CPU arithmetic (decode) vs memory bandwidth.
- **Hypothesis:** tile-fusion (no L1 scratch) + OOO overlap closes the R144 gap; multi-core lets decode scale past the bus-bound bf16 read.
- **Results:** the per-thread table (plain bf16 vs R144 naive-fused vs R145 tile-fused), best ms each, speedup, resident MB; lossless parity bit-identical (test green).
- **Analysis:** compare R145 best vs plain 9.6 ms and vs R144's 19.9 ms; how much did fusion+OOO recover, and did multi-core flip it. If GO: the lossless-compressed-resident speed win is demonstrated (the arc's first speed GO). If MARGINAL/NO-GO: state exactly the remaining gap and that 16-wide `vqtbl2q` decode + strategy-B pipelining are the next levers (R146).
- **Decision:** the verdict. If GO/MARGINAL → next is `--fast` wiring (own spec). 
- **Next Experiment:** `--fast` wiring if GO; 16-wide decode / explicit pipelining if MARGINAL/NO-GO.

- [ ] **Step 3: Add the index row**

In `docs/benchmarks/trials/index.md`, add an R145 row mirroring R144's columns. Baseline = plain bf16 9.6 ms + R144 naive-fused 19.9 ms; result = R145 tile-fused `<best>` ms, `<speedup>`, 19% less RAM, lossless + verdict.

- [ ] **Step 4: Update memory**

Append the measured R145 number to `rllm-speed-thesis-streaming-vs-resident.md`: tile-fused GEMV `<best>` ms vs plain bf16 `<best_plain>` ms multi-core (vs R144 naive 19.9 ms), 19% less RAM, lossless bit-identical → `<verdict>`. State whether the lossless-compressed-resident speed win is now demonstrated, and the next lever.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmarks/trials/ "/Users/deansanbhnanwr/.claude/projects/-Users-deansanbhnanwr-Projects-rllm/memory/rllm-speed-thesis-streaming-vs-resident.md"
git commit -m "docs(bench): R145 REEFUSE-PLANE-DOT v2 tile-fused GEMV trial (<verdict>) + index + memory

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Verification (end-to-end)

1. `cargo test -p rllm-runtime --lib` → green, including `fused_kernel_matches_reference_bit_for_bit` (bit-identical lossless).
2. `cargo build` → compiles, no new dependencies.
3. The `#[ignore]` bench printed the `=== R145 tile-fused GEMV multi-core BENCH ===` table with a real verdict on the 525 MB sample.
4. Trial report in the verdict folder; `index.md` has the R145 row; memory updated.
5. `git grep -n "bitplane_row_dot_bfdot\|lm_head_logits_bitplane_fused" crates/rllm-runtime/src | grep -v bitplane_gemv.rs` → **no hits** (no runtime wiring; kernel stays a measured building block).

## Out of scope (gated follow-up — do NOT build here)

- `--fast` wiring of the fused kernel; `codec_for_id` registration; model packing.
- 16-wide `vqtbl2q` decode + explicit software pipelining (strategy B) — only if the gate is missed (next round).
- General `(hidden, w)`; q8/q4 layers; KV-cache; GPU.
