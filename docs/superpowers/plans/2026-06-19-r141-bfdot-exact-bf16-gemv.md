# R141 — bfdot exact-bf16 GEMV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LM-head/embedding exact-bf16 GEMV (bf16→f32 upcast + f32 FMA) with an ARM `bfdot` kernel that keeps weights bit-exact and computes at native bf16 precision, ~2× faster, runtime-gated with the f32 path as fallback.

**Architecture:** A new inline-asm `bfdot` row-dot kernel plus a once-per-GEMV `f32→bf16` activation conversion, wrapped in a `Bf16DotActivation` holder that converts the shared activation once and dispatches per weight row to either bfdot (when `FEAT_BF16` is present and not disabled) or the existing f32 kernel. The holder replaces the three current `bf16_row_dot_f32` call sites. A feasibility gate (Task 2) measures the speedup before wiring (Task 3).

**Tech Stack:** Rust, `std::arch` aarch64 inline asm (`bfdot`, NEON), `is_aarch64_feature_detected!`. No new crates.

## Global Constraints

- **Lossless = exact WEIGHTS** (bit-for-bit bf16, never quantized). Weights are loaded directly as bf16; only activations are converted to bf16. This is the project doctrine for this work.
- **Native-bf16 compute is allowed** (Option A). bfdot results need NOT be bit-identical to full-f32; they must be within bf16-input precision and preserve argmax.
- **No external dependencies.** Inline asm via `std::arch` only; no bf16 crate, no nightly intrinsics. `cargo build` stays the only requirement (the reason asm was chosen over `vbfdotq_f32`).
- **Runtime-gated + fallback.** Gate on `is_aarch64_feature_detected!("bf16")`; the existing f32 path runs when bf16 is absent, on non-aarch64, or when `RLLM_BF16_DOT=0`. Nothing regresses on CPUs without bf16.
- **Honest metrics.** Report decode tok/s before→after and the bfdot-vs-f32 parity diff (max_abs_diff + argmax match). State the remaining gap vs q8 `--fast` as a limitation.
- **REE kernel name** (working): `REEFLOW-BF16-DOT` — Erik's final call before the trial report / merge.
- **Feasibility gate is a hard checkpoint.** If Task 2 shows bf16 absent or bfdot not faster, STOP — do not wire (Task 3+). Record the negative result.
- Reference spec: `docs/superpowers/specs/2026-06-19-r141-bfdot-exact-bf16-gemv-design.md`.

## File Structure

- `crates/rllm-runtime/src/streaming/mod.rs` — **all new kernel code lives here**, beside the existing `bf16_row_dot_f32` / `bf16_row_dot_f32_neon` (mod.rs:199-267). New: `bf16_dot_available()`, `bf16_dot_enabled()`, `f32_to_bf16_rne()`, `convert_f32_to_bf16()`, `bf16_row_dot_bf16()` (the bfdot asm kernel), `Bf16DotActivation<'a>`. New tests in the existing `#[cfg(test)] mod tests` of this file.
- `crates/rllm-runtime/src/streaming/argmax.rs` — modify caller `raw_16bit_argmax_rows_range` (argmax.rs:207-227).
- `crates/rllm-runtime/src/streaming/kernels.rs` — modify caller block at kernels.rs:3389-3395.
- `docs/benchmarks/trials/<status>/2026-06-19-r141-reeflow-bf16-dot.md` — trial report (Task 4).
- `docs/benchmarks/trials/index.md` — add the trial row (Task 4).

All three call sites currently call `bf16_row_dot_f32(input, wrow, n)` inside a per-row loop where the activation (`input` / `last_hidden`) is constant. Each becomes: build `Bf16DotActivation::new(input)` ONCE before the loop, then `act.row_dot(wrow, n)` per row.

---

### Task 1: bfdot kernel, gate, and activation conversion

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/mod.rs` (add code after `bf16_row_dot_f32_neon`, ~mod.rs:267)
- Test: same file, in the existing `#[cfg(test)] mod tests`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (used by Tasks 2 & 3):
  - `fn f32_to_bf16_rne(x: f32) -> u16`
  - `fn convert_f32_to_bf16(hid: &[f32]) -> Vec<u16>`
  - `#[cfg(target_arch = "aarch64")] fn bf16_dot_available() -> bool`
  - `#[cfg(target_arch = "aarch64")] #[target_feature(enable = "bf16")] unsafe fn bf16_row_dot_bf16(hid_bf16: &[u16], wrow: &[u8], hidden: usize) -> f32`

- [ ] **Step 1: Write the failing test**

Add to `mod tests` in `crates/rllm-runtime/src/streaming/mod.rs`:

```rust
#[test]
fn bf16_rne_matches_known_values() {
    // 1.0f32 = 0x3F800000 -> bf16 0x3F80; exact, no rounding.
    assert_eq!(super::f32_to_bf16_rne(1.0), 0x3F80);
    // 0.0 -> 0x0000
    assert_eq!(super::f32_to_bf16_rne(0.0), 0x0000);
    // round-to-nearest-even: 0x3F808000 ties -> even (0x3F80, not 0x3F81).
    assert_eq!(super::f32_to_bf16_rne(f32::from_bits(0x3F80_8000)), 0x3F80);
    // just above the tie rounds up to 0x3F81.
    assert_eq!(super::f32_to_bf16_rne(f32::from_bits(0x3F80_8001)), 0x3F81);
    // NaN stays NaN (exponent all ones, nonzero mantissa).
    let n = super::f32_to_bf16_rne(f32::NAN);
    assert_eq!(n & 0x7F80, 0x7F80);
    assert_ne!(n & 0x007F, 0);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn bfdot_row_matches_f32_reference_within_bf16_tol() {
    if !super::bf16_dot_available() {
        eprintln!("skip: FEAT_BF16 not present on this CPU");
        return;
    }
    // Deterministic pseudo-random hidden + bf16 weight row, length 70 (covers the
    // 32-wide main loop twice + a 6-element scalar tail).
    let hidden = 70usize;
    let mut hid = vec![0.0f32; hidden];
    let mut wrow = vec![0u8; hidden * 2];
    let mut s: u32 = 0x1234_5678;
    let mut next = || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); s };
    for i in 0..hidden {
        hid[i] = (next() as i32 as f32) / (i32::MAX as f32) * 3.0;
        // a valid finite bf16 weight in [-2, 2): build from a small f32 then truncate.
        let wf = (next() as i32 as f32) / (i32::MAX as f32) * 2.0;
        let wb = super::f32_to_bf16_rne(wf);
        wrow[i * 2] = (wb & 0xFF) as u8;
        wrow[i * 2 + 1] = (wb >> 8) as u8;
    }
    // Reference: dot of bf16(hid) against bf16(wrow), accumulated in f32.
    let mut reference = 0.0f64;
    for i in 0..hidden {
        let a = f32::from_bits((super::f32_to_bf16_rne(hid[i]) as u32) << 16);
        let wb = u16::from_le_bytes([wrow[i * 2], wrow[i * 2 + 1]]);
        let w = f32::from_bits((wb as u32) << 16);
        reference += (a as f64) * (w as f64);
    }
    let act = super::convert_f32_to_bf16(&hid);
    let got = unsafe { super::bf16_row_dot_bf16(&act, &wrow, hidden) } as f64;
    let denom = reference.abs().max(1e-3);
    assert!(
        (got - reference).abs() / denom < 1e-2,
        "bfdot {got} vs reference {reference} (rel err too large)"
    );
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p rllm-runtime --lib bf16_rne_matches_known_values bfdot_row_matches`
Expected: FAIL to COMPILE (`f32_to_bf16_rne`, `convert_f32_to_bf16`, `bf16_dot_available`, `bf16_row_dot_bf16` not found).

- [ ] **Step 3: Implement the kernel, gate, and conversion**

Add after `bf16_row_dot_f32_neon` (around mod.rs:267):

```rust
/// f32 -> bf16 with round-to-nearest, ties-to-even. Returns the bf16 bit pattern.
fn f32_to_bf16_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        // NaN: truncate to bf16 but force a nonzero mantissa so it stays NaN.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounding_bias = 0x0000_7FFFu32 + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// Convert an f32 activation row to a bf16 scratch buffer (done once per GEMV).
fn convert_f32_to_bf16(hid: &[f32]) -> Vec<u16> {
    hid.iter().map(|&x| f32_to_bf16_rne(x)).collect()
}

#[cfg(target_arch = "aarch64")]
fn bf16_dot_available() -> bool {
    static AVAIL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAIL.get_or_init(|| std::arch::is_aarch64_feature_detected!("bf16"))
}

/// Dot product of a bf16 activation row (native u16) with one bf16 weight row
/// (little-endian u16 bytes), accumulated in f32 via ARM `bfdot`. Weights are read
/// exactly as stored. 4 independent accumulator chains for ILP, 32 bf16 per
/// iteration; scalar tail upcasts both operands to f32 (exact).
///
/// SAFETY: caller guarantees `FEAT_BF16` (see `bf16_dot_available`), `hid_bf16` has
/// `hidden` u16, and `wrow` has `hidden * 2` bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "bf16")]
unsafe fn bf16_row_dot_bf16(hid_bf16: &[u16], wrow: &[u8], hidden: usize) -> f32 {
    use std::arch::aarch64::*;
    debug_assert!(hid_bf16.len() >= hidden && wrow.len() >= hidden * 2);
    let aptr = hid_bf16.as_ptr();
    let wptr = wrow.as_ptr();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 32 <= hidden {
        let a0 = aptr.add(i) as *const u8;
        let a1 = aptr.add(i + 8) as *const u8;
        let a2 = aptr.add(i + 16) as *const u8;
        let a3 = aptr.add(i + 24) as *const u8;
        let w0 = wptr.add(i * 2);
        let w1 = wptr.add((i + 8) * 2);
        let w2 = wptr.add((i + 16) * 2);
        let w3 = wptr.add((i + 24) * 2);
        std::arch::asm!(
            "ld1 {{v0.8h}}, [{a0}]",
            "ld1 {{v1.8h}}, [{a1}]",
            "ld1 {{v2.8h}}, [{a2}]",
            "ld1 {{v3.8h}}, [{a3}]",
            "ld1 {{v4.8h}}, [{w0}]",
            "ld1 {{v5.8h}}, [{w1}]",
            "ld1 {{v6.8h}}, [{w2}]",
            "ld1 {{v7.8h}}, [{w3}]",
            "bfdot {acc0:v}.4s, v4.8h, v0.8h",
            "bfdot {acc1:v}.4s, v5.8h, v1.8h",
            "bfdot {acc2:v}.4s, v6.8h, v2.8h",
            "bfdot {acc3:v}.4s, v7.8h, v3.8h",
            a0 = in(reg) a0, a1 = in(reg) a1, a2 = in(reg) a2, a3 = in(reg) a3,
            w0 = in(reg) w0, w1 = in(reg) w1, w2 = in(reg) w2, w3 = in(reg) w3,
            acc0 = inout(vreg) acc0,
            acc1 = inout(vreg) acc1,
            acc2 = inout(vreg) acc2,
            acc3 = inout(vreg) acc3,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            options(readonly, nostack),
        );
        i += 32;
    }
    let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    while i < hidden {
        let wb = u16::from_le_bytes([*wptr.add(i * 2), *wptr.add(i * 2 + 1)]);
        let w = f32::from_bits((wb as u32) << 16);
        let a = f32::from_bits((*aptr.add(i) as u32) << 16);
        sum += a * w;
        i += 1;
    }
    sum
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p rllm-runtime --lib bf16_rne_matches_known_values bfdot_row_matches`
Expected: PASS (2 passed). On a non-bf16 CPU the bfdot test prints "skip" and passes.

- [ ] **Step 5: Commit**

```bash
git add crates/rllm-runtime/src/streaming/mod.rs
git commit -m "feat(runtime): bfdot exact-bf16 row-dot kernel + FEAT_BF16 gate + f32->bf16 RNE"
```

---

### Task 2: Feasibility gate — measure bf16 presence and bfdot speedup (GO/NO-GO)

**Files:**
- Test: `crates/rllm-runtime/src/streaming/mod.rs` (in `mod tests`, an `#[ignore]` measurement)

**Interfaces:**
- Consumes: `bf16_dot_available`, `convert_f32_to_bf16`, `bf16_row_dot_bf16`, and the existing `bf16_row_dot_f32` (all in this module).
- Produces: a recorded measurement (ns/row bfdot vs f32, FEAT_BF16 yes/no). No code consumed downstream.

**This task is the hard checkpoint.** After running it, the controller/engineer reads the output and decides GO/NO-GO. GO only if `FEAT_BF16` is present AND bfdot is faster than the f32 path. If NO-GO: stop here, record the negative result in a trial doc (Task 4 format), and do not implement Task 3.

- [ ] **Step 1: Write the measurement (an `#[ignore]` test that always "passes")**

Add to `mod tests`:

```rust
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore = "feasibility measurement: cargo test -- --ignored --nocapture bfdot_feasibility"]
fn bfdot_feasibility() {
    let has_bf16 = super::bf16_dot_available();
    eprintln!("FEAT_BF16 detected: {has_bf16}");
    if !has_bf16 {
        eprintln!("NO-GO: bf16 absent; bfdot path will never engage on this CPU.");
        return;
    }
    let hidden = 2048usize; // LLaMA-1B hidden
    let rows = 2000usize;   // a vocab slice
    let mut hid = vec![0.0f32; hidden];
    let mut weights = vec![0u8; rows * hidden * 2];
    let mut s: u32 = 0xC0FF_EE11;
    let mut next = || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); s };
    for v in hid.iter_mut() { *v = (next() as i32 as f32) / (i32::MAX as f32); }
    for b in weights.iter_mut() { *b = (next() >> 16) as u8; }

    // f32 path: per-row upcast dot.
    let t0 = std::time::Instant::now();
    let mut sink = 0.0f32;
    for r in 0..rows {
        let wrow = &weights[r * hidden * 2..(r + 1) * hidden * 2];
        sink += super::bf16_row_dot_f32(&hid, wrow, hidden);
    }
    let f32_ns = t0.elapsed().as_nanos() as f64 / rows as f64;

    // bfdot path: convert activation ONCE, then per-row bfdot.
    let t1 = std::time::Instant::now();
    let act = super::convert_f32_to_bf16(&hid);
    let mut sink2 = 0.0f32;
    for r in 0..rows {
        let wrow = &weights[r * hidden * 2..(r + 1) * hidden * 2];
        sink2 += unsafe { super::bf16_row_dot_bf16(&act, wrow, hidden) };
    }
    let bf_ns = t1.elapsed().as_nanos() as f64 / rows as f64;

    eprintln!("f32-upcast: {f32_ns:.1} ns/row   bfdot: {bf_ns:.1} ns/row   speedup: {:.2}x   (sinks {sink} {sink2})", f32_ns / bf_ns);
    eprintln!("DECISION: {}", if bf_ns < f32_ns { "GO (bfdot faster)" } else { "NO-GO (bfdot not faster)" });
}
```

- [ ] **Step 2: Run the measurement**

Run: `cargo test -p rllm-runtime --lib -- --ignored --nocapture bfdot_feasibility`
Expected: prints `FEAT_BF16 detected: true/false`, `ns/row` for both paths, a speedup ratio, and a `DECISION` line.

- [ ] **Step 3: Record the result and decide**

Capture the printed numbers (FEAT_BF16, both ns/row, speedup) for the Task 4 trial report. **If DECISION is NO-GO, STOP** — do not proceed to Task 3; write the negative-result trial doc instead.

- [ ] **Step 4: Commit**

```bash
git add crates/rllm-runtime/src/streaming/mod.rs
git commit -m "test(runtime): bfdot feasibility gate (FEAT_BF16 + bfdot-vs-f32 ns/row)"
```

---

### Task 3: `Bf16DotActivation` holder, env override, and wire the three call sites

**Files:**
- Modify: `crates/rllm-runtime/src/streaming/mod.rs` (add holder + `bf16_dot_enabled`; change `lm_head_logits_rows_bf16` at mod.rs:179-191)
- Modify: `crates/rllm-runtime/src/streaming/argmax.rs` (mod.rs:207-227 region)
- Modify: `crates/rllm-runtime/src/streaming/kernels.rs` (kernels.rs:3389-3395 region)
- Test: `crates/rllm-runtime/src/streaming/mod.rs` (`mod tests`)

**Interfaces:**
- Consumes: `bf16_dot_available`, `convert_f32_to_bf16`, `bf16_row_dot_bf16`, `bf16_row_dot_f32`.
- Produces:
  - `fn bf16_dot_enabled() -> bool`
  - `struct Bf16DotActivation<'a>` with `fn new(hid: &'a [f32]) -> Self` and `fn row_dot(&self, wrow: &[u8], hidden: usize) -> f32`

- [ ] **Step 1: Write the failing test**

Add to `mod tests`:

```rust
#[test]
fn bf16_activation_holder_matches_f32_argmax() {
    // Holder over many rows selects the same argmax as the pure-f32 path.
    let hidden = 96usize;
    let rows = 40usize;
    let mut hid = vec![0.0f32; hidden];
    let mut weights = vec![0u8; rows * hidden * 2];
    let mut s: u32 = 0x55AA_1234;
    let mut next = || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); s };
    for v in hid.iter_mut() { *v = (next() as i32 as f32) / (i32::MAX as f32) * 2.0; }
    for r in 0..rows {
        for i in 0..hidden {
            let wf = (next() as i32 as f32) / (i32::MAX as f32) * 2.0;
            let wb = super::f32_to_bf16_rne(wf);
            weights[(r * hidden + i) * 2] = (wb & 0xFF) as u8;
            weights[(r * hidden + i) * 2 + 1] = (wb >> 8) as u8;
        }
    }
    let row = |r: usize| &weights[r * hidden * 2..(r + 1) * hidden * 2];

    // f32 reference argmax.
    let mut ref_best = (0usize, f32::MIN);
    for r in 0..rows {
        let v = super::bf16_row_dot_f32(&hid, row(r), hidden);
        if v > ref_best.1 { ref_best = (r, v); }
    }
    // Holder argmax (uses bfdot when available, else f32 — both must agree on argmax).
    let act = super::Bf16DotActivation::new(&hid);
    let mut got_best = (0usize, f32::MIN);
    for r in 0..rows {
        let v = act.row_dot(row(r), hidden);
        if v > got_best.1 { got_best = (r, v); }
    }
    assert_eq!(ref_best.0, got_best.0, "bfdot holder argmax must match f32 argmax");
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p rllm-runtime --lib bf16_activation_holder_matches_f32_argmax`
Expected: FAIL to COMPILE (`Bf16DotActivation` not found).

- [ ] **Step 3: Implement the holder and env gate**

Add to `crates/rllm-runtime/src/streaming/mod.rs` (after the Task 1 code):

```rust
/// bfdot is used when the CPU has FEAT_BF16 and it is not disabled via
/// `RLLM_BF16_DOT=0`. False on non-aarch64.
fn bf16_dot_enabled() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(std::env::var("RLLM_BF16_DOT").as_deref(), Ok("0")) {
            return false;
        }
        bf16_dot_available()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Holds a GEMV activation in the form the row-dot kernel needs, converting f32 ->
/// bf16 ONCE when bfdot is active so the per-row cost is pure bfdot. Falls back to
/// holding the f32 slice and using the f32 kernel otherwise.
enum Bf16DotActivation<'a> {
    Bf16(Vec<u16>),
    F32(&'a [f32]),
}

impl<'a> Bf16DotActivation<'a> {
    fn new(hid: &'a [f32]) -> Self {
        if bf16_dot_enabled() {
            Bf16DotActivation::Bf16(convert_f32_to_bf16(hid))
        } else {
            Bf16DotActivation::F32(hid)
        }
    }

    #[inline]
    fn row_dot(&self, wrow: &[u8], hidden: usize) -> f32 {
        match self {
            Bf16DotActivation::Bf16(a) => {
                #[cfg(target_arch = "aarch64")]
                {
                    // SAFETY: the Bf16 variant is only constructed when
                    // bf16_dot_enabled() -> bf16_dot_available() == FEAT_BF16.
                    unsafe { bf16_row_dot_bf16(a, wrow, hidden) }
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    let _ = a;
                    unreachable!("Bf16 activation variant cannot exist off aarch64")
                }
            }
            Bf16DotActivation::F32(h) => bf16_row_dot_f32(h, wrow, hidden),
        }
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p rllm-runtime --lib bf16_activation_holder_matches_f32_argmax`
Expected: PASS.

- [ ] **Step 5: Wire call site 1 — `lm_head_logits_rows_bf16` (mod.rs:179-191)**

Replace the body of `lm_head_logits_rows_bf16`:

```rust
fn lm_head_logits_rows_bf16(
    last_hidden: &[f32],
    weight_bf16: &[u8],
    hidden: usize,
    row_offset: usize,
    out: &mut [f32],
) {
    let act = Bf16DotActivation::new(last_hidden);
    for (r, logit) in out.iter_mut().enumerate() {
        let row_base = (row_offset + r) * hidden * 2;
        let wrow = &weight_bf16[row_base..row_base + hidden * 2];
        *logit = act.row_dot(wrow, hidden);
    }
}
```

- [ ] **Step 6: Wire call site 2 — `raw_16bit_argmax_rows_range` (argmax.rs)**

In `crates/rllm-runtime/src/streaming/argmax.rs`, build the holder once before the row loop and use it inside. The `Bf16DotActivation` type is private to the module tree; reference it via the module path used by the other `bf16_row_dot_f32` import in this file (same `super`/`crate::streaming` path already in scope at argmax.rs:218).

Replace lines 207-227 (from `let fast_bf16 = ...` through the closing `best`):

```rust
    let fast_bf16 = q8_activation_path_enabled() && matches!(dtype, rllm_container::DType::Bf16);
    let n = config.in_features;
    let bf16_act = fast_bf16.then(|| Bf16DotActivation::new(input));
    let mut best = ArgmaxCandidate::empty();
    for row_idx in 0..rows {
        let out_feature = out_feature_start + row_idx;
        let row_start = local_row_start + row_idx * config.in_features;
        let mut acc = bias
            .and_then(|values| values.get(out_feature))
            .copied()
            .unwrap_or(0.0);
        if let Some(act) = &bf16_act {
            acc += act.row_dot(&raw_bytes[row_start * 2..(row_start + n) * 2], n);
        } else {
            let mut input_idx = 0usize;
            while input_idx < n {
                acc += input[input_idx] * raw_16bit_weight_at(raw_bytes, row_start + input_idx, dtype);
                input_idx += 1;
            }
        }
        best.observe(out_feature, acc);
    }
    best
```

Add `Bf16DotActivation` to the existing `use super::...` import block at the top of `argmax.rs` (the one already importing `bf16_row_dot_f32`). If `bf16_row_dot_f32` is imported as `use super::bf16_row_dot_f32;`, change/add `use super::{bf16_row_dot_f32, Bf16DotActivation};` — but since `bf16_row_dot_f32` is no longer called directly here after this edit, you may replace it with `use super::Bf16DotActivation;`. Verify by compiling.

- [ ] **Step 7: Wire call site 3 — `kernels.rs:3389-3395`**

In `crates/rllm-runtime/src/streaming/kernels.rs`, the `fast_bf16` block is inside a `while` loop over rows; `input` is constant for the whole function. Build the holder once where `fast_bf16` is first known (just before the `while local_idx + ... ` loop that contains line 3378) and use it in the block:

```rust
        if fast_bf16 {
            let n = config.in_features;
            let dot = |rs: usize| bf16_act
                .as_ref()
                .expect("fast_bf16 implies bf16_act is Some")
                .row_dot(&raw_bytes[rs * 2..(rs + n) * 2], n);
            output[out_feature] += dot(row0_start);
            output[out_feature + 1] += dot(row1_start);
            output[out_feature + 2] += dot(row2_start);
            output[out_feature + 3] += dot(row3_start);
        } else {
```

Just before that `while` loop, add:

```rust
    let bf16_act = fast_bf16.then(|| Bf16DotActivation::new(input));
```

Import `Bf16DotActivation` into `kernels.rs` via the module path already used for `bf16_row_dot_f32` at kernels.rs:3391 (`use super::bf16_row_dot_f32;` → add `Bf16DotActivation`). Verify by compiling. Note: `Bf16DotActivation` and `bf16_dot_*` are currently private `fn`/`enum` in `mod.rs`; they are visible to child modules `argmax`/`kernels` without `pub` because child modules can access private items of ancestor modules. No visibility change needed.

- [ ] **Step 8: Run the full runtime suite + parity test**

Run: `cargo test -p rllm-runtime --lib`
Expected: PASS (all prior tests + the 3 new ones; 289+ passed, 0 failed). This proves the f32 path is unchanged (existing bit-identical tests still pass) and the holder wiring compiles and selects correct argmax.

- [ ] **Step 9: Commit**

```bash
git add crates/rllm-runtime/src/streaming/mod.rs crates/rllm-runtime/src/streaming/argmax.rs crates/rllm-runtime/src/streaming/kernels.rs
git commit -m "feat(runtime): wire bfdot via Bf16DotActivation (convert-once) into lm_head/argmax; RLLM_BF16_DOT override"
```

---

### Task 4: End-to-end measurement, REE name, trial report

**Files:**
- Create: `docs/benchmarks/trials/<status>/2026-06-19-r141-reeflow-bf16-dot.md`
- Modify: `docs/benchmarks/trials/index.md`

**Interfaces:** none (documentation + measurement).

- [ ] **Step 1: Build the release CLI**

Run: `cargo build --release -p rllm-cli`
Expected: builds clean.

- [ ] **Step 2: Measure decode tok/s — f32 path (baseline)**

Run: `RLLM_BF16_DOT=0 ./target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --fast --chat-template llama3 --max-new-tokens 64 --ctx 512 <<< $'What is the capital of Australia?\nquit'`
Capture: decode tok/s, output text (for argmax sanity — should say Canberra).

- [ ] **Step 3: Measure decode tok/s — bfdot path**

Run: `./target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm --fast --chat-template llama3 --max-new-tokens 64 --ctx 512 <<< $'What is the capital of Australia?\nquit'`
Capture: decode tok/s, output text. Compare output to Step 2 (should be the same / coherent — argmax parity in practice).

- [ ] **Step 4: Confirm the REE kernel name with Erik**

The working name is `REEFLOW-BF16-DOT`. Per [[ree-kernel-naming-rule]] the kernel needs a confirmed REE-lineage name before the trial report is finalized/merged. Use `REEFLOW-BF16-DOT` unless Erik chooses another; put the final name in the trial's `## Scope → REE kernel` line.

- [ ] **Step 5: Write the trial report from the template**

Start from `docs/benchmarks/templates/trial-report.md`. Create `docs/benchmarks/trials/success/2026-06-19-r141-reeflow-bf16-dot.md` (or `failed/`/`inconclusive/` per the Task 2 gate + Step 2/3 result). Required sections (mirror the R140/R133 reports): title `# Trial: R141 …`, the `Date`/`Owner: RLLM`/`Status`/`Folder` block, then `## Hypothesis`, `## Scope` (Mode: exact-weight bf16 runtime; REE kernel: REEFLOW-BF16-DOT; Model/artifact: `Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm`; Architecture: LLaMA 3.2 1B, tied bf16 embedding/LM-head; Target device: Apple A18 Pro; Bottleneck tag: CPU bf16 GEMV / f32-FMA → bfdot), `## Setup`, `## Results` (FEAT_BF16 yes/no; Task 2 ns/row + speedup; decode tok/s before→after from Steps 2-3; the bfdot-vs-f32 parity note; the remaining gap vs q8 `--fast` stated as a limitation), `## Analysis`, `## Decision`.

- [ ] **Step 6: Add the index row**

In `docs/benchmarks/trials/index.md`, add one row: date | R141 | folder | model | mode (exact-weight bf16) | bottleneck tag (bf16 GEMV) | baseline (f32-upcast ns/row + tok/s) | result (bfdot speedup + tok/s) | decision | paper value (DFloat11/native-bf16 CPU compute).

- [ ] **Step 7: Commit**

```bash
git add docs/benchmarks/trials/
git commit -m "docs(bench): R141 REEFLOW-BF16-DOT trial report + index row (bfdot exact-bf16 GEMV)"
```

---

## Self-Review

**1. Spec coverage:**
- Lossless = exact weights / native-bf16 compute (Option A) → Global Constraints + Task 1 kernel reads weights as bf16 directly. ✓
- Convert-activation-once insight → Task 3 `Bf16DotActivation::new` + holder built before each row loop. ✓
- `bf16_row_dot_bfdot` inline-asm kernel → Task 1 (`bf16_row_dot_bf16`). ✓
- Activation-convert helper (RNE) → Task 1 `f32_to_bf16_rne`/`convert_f32_to_bf16`. ✓
- Runtime gate `bf16_dot_available()` → Task 1. ✓
- Dispatcher + `RLLM_BF16_DOT=0` override → Task 3 `bf16_dot_enabled` + holder. ✓
- Wiring into LM-head/argmax → Task 3 Steps 5-7 (all three `bf16_row_dot_f32` call sites). ✓
- Feasibility gate (GO/NO-GO before wiring) → Task 2 (hard checkpoint). ✓
- Testing: correctness vs f32 ref, argmax parity, fallback bit-identical, honest metrics → Task 1/3 tests + Task 4 measurement. ✓
- REE name → Global Constraints + Task 4 Step 4. ✓
- Trial doc + index per docs/benchmarks rules → Task 4. ✓
- Phase 2 is design-level only (not in this plan) → correctly omitted from tasks. ✓

**2. Placeholder scan:** No TBD/TODO; every code step has complete code; commands have expected output. The trial-report folder (`success`/`failed`/`inconclusive`) is intentionally decided by the measured result, not a placeholder. ✓

**3. Type consistency:** `f32_to_bf16_rne(f32)->u16`, `convert_f32_to_bf16(&[f32])->Vec<u16>`, `bf16_row_dot_bf16(&[u16],&[u8],usize)->f32`, `Bf16DotActivation::new(&[f32])`, `.row_dot(&[u8],usize)->f32` — used consistently across Tasks 1-3 and all three call sites. The existing `bf16_row_dot_f32(&[f32],&[u8],usize)->f32` signature is unchanged. ✓

## Notes for the implementer

- The bfdot asm mirrors the existing `batch1_x4_ilp` (kernels.rs:1030) and `i8_dot32_sdot` (kernels.rs:292) idioms: explicit `v0`-`v7` clobbers, named `inout(vreg)` accumulators, `{reg:v}.4s` formatting. If the assembler rejects a `bfdot`/`ld1 .8h` mnemonic, confirm the toolchain targets ARMv8.6+; the `#[target_feature(enable = "bf16")]` attribute is what makes `bfdot` legal in the block.
- The correctness test (Task 1) is the safety net for any asm transcription slip — a wrong lane/stride blows the 1% tolerance immediately.
- `Bf16DotActivation` allocates one `Vec<u16>` per construction (once per worker/block, not per row) — acceptable; do not micro-optimize without a measurement (YAGNI).
