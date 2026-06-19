# R143 — `rtc-bitplane-v1` SIMD-decodable lossless bf16 codec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fixed-width palette-index bit-plane lossless bf16 codec (`rtc-bitplane-v1`) with a NEON `tbl`-gather decode kernel, then measure whether SIMD decode of the real 525 MB Llama 1B bf16 embedding clears the throughput bar (GO/MARGINAL/NO-GO).

**Architecture:** Encode splits each bf16 into (exponent, residual) — reusing the crate's `split_bf16` — builds a per-tensor exponent palette (≤64, else raw-fallback), packs palette indices at `w = ceil(log2(palette))` bits, and stores residuals as a byte plane. Scalar `decode` (the lossless oracle) reuses `BufferedBitReader`/`join_bf16`. A `w=5`-specialized NEON kernel (`decode_neon_w5`) unpacks indices with per-lane shifts, gathers exponents via `vtbl4_u8`, reconstructs bf16, and is parity-checked bit-for-bit against the scalar decode. An `#[ignore]` bench reports the verdict.

**Tech Stack:** Rust (stable), crate `rtc-codec`, `std::arch::aarch64` NEON intrinsics. No new dependencies.

## Global Constraints

- Pure Rust, **no new dependencies**; `cargo build` stays the only requirement. NEON via `std::arch::aarch64` (built in).
- **Lossless / bit-identical (hard rule):** `decode(encode(x)) == x` byte-for-byte; `decode_neon_w5` bit-identical to scalar `decode`. A single differing byte fails.
- **Reuse, don't duplicate (DRY):** reuse `split_bf16`, `join_bf16` (dfloat.rs), `BitWriter` (dfloat.rs), `BufferedBitReader` (bitreader_fast.rs). Do not re-implement Huffman or bit I/O.
- **Codec format `rtc-bitplane-v1`:** `[magic "RTCB"(4)][version u8=1][flags u8][num_weights u64][palette_len u8][index_width u8][palette P bytes][index_plane ceil(n*w/8)][residual_plane n bytes]`. `flags` bit0 = raw-fallback (body = raw bf16). Palette > 64 → raw-fallback.
- **REE kernel working name: REEPLANE** (Erik's final call) — use in the trial report Scope line.
- **Scope:** Phase A (codec) + Phase B (NEON decode + throughput gate) only. NO fused decode→bfdot, NO `codec_for_id` registration, NO runtime wiring — those are Phase C, gated behind the measured verdict.
- **Threshold (identical to R142, composable):** aggregate (single-core ×3.5) ≥12 Gweight/s = GO, ≥5 = MARGINAL, <5 = NO-GO.
- **Honest metrics:** report single-core Gweight/s, speedup vs scalar bitplane decode, and the verdict — MARGINAL/NO-GO stated plainly.

## File Structure

- **Create** `crates/rtc-codec/src/bitplane.rs` — `BitplaneCodec` (encode + scalar decode), `index_width` helper, `decode_neon_w5` NEON kernel, all tests + the `#[ignore]` bench. One file, one codec (mirrors `dfloat.rs`).
- **Modify** `crates/rtc-codec/src/lib.rs` — `mod bitplane;`, `pub use bitplane::*;`, `pub const CODEC_BITPLANE_V1`.
- **Create** `docs/benchmarks/trials/<verdict-folder>/2026-06-19-r143-reeplane-bitplane-codec.md` — trial report.
- **Modify** `docs/benchmarks/trials/index.md` — R143 row.
- **Modify** memory `rllm-speed-thesis-streaming-vs-resident.md` — measured R143 number.

---

### Task 1: `rtc-bitplane-v1` codec — encode + scalar decode (Phase A)

**Files:**
- Create: `crates/rtc-codec/src/bitplane.rs`
- Modify: `crates/rtc-codec/src/lib.rs`
- Test: in `crates/rtc-codec/src/bitplane.rs`

**Interfaces:**
- Consumes: `split_bf16`, `join_bf16`, `BitWriter` (crate, from `dfloat.rs`); `BufferedBitReader` (`crate::bitreader_fast`); `TensorCodec`, `EncodeMeta`, `DecodeMeta`, `EncodedChunk`, `CodecError`, `Result` (crate).
- Produces: `pub struct BitplaneCodec` with `pub const ID: &str = "rtc-bitplane-v1"`, impl `TensorCodec` (`encode`, `decode`); `fn index_width(p: usize) -> u8`.

- [ ] **Step 1: Add module + exports to `lib.rs`**

In `crates/rtc-codec/src/lib.rs`, make exactly three additions:
1. In the `mod` block, add `mod bitplane;` (e.g. right after `mod bitreader_fast;`).
2. In the `pub use` block, add `pub use bitplane::*;` (e.g. right after `pub use dfloat::*;`).
3. After the `CODEC_DFLOAT_V1` constant, add:

```rust
/// Codec ID for the SIMD-decodable bit-plane bf16 codec
pub const CODEC_BITPLANE_V1: &str = "rtc-bitplane-v1";
```

- [ ] **Step 2: Write the failing roundtrip tests**

Create `crates/rtc-codec/src/bitplane.rs` with the header + test module only:

```rust
//! rtc-bitplane-v1: SIMD-decodable lossless bf16 codec.
//!
//! Like rtc-dfloat-v1 it splits bf16 into (exponent, residual), but instead of
//! Huffman-coding the exponent it stores a fixed-width palette index per weight.
//! Fixed width => branchless, SIMD `tbl`-gather decode (no per-symbol serial
//! dependency). Trades ratio (~13 bits/weight vs Huffman's 10.6) for fast decode.

use crate::bitreader_fast::BufferedBitReader;
use crate::codec::{EncodeMeta, EncodedChunk, TensorCodec};
use crate::dfloat::{join_bf16, split_bf16, BitWriter};
use crate::error::{CodecError, Result};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DecodeMeta;

    fn dmeta() -> DecodeMeta {
        DecodeMeta { codec_id: "rtc-bitplane-v1".into(), uncompressed_size: 0 }
    }

    // Build bf16 bytes whose exponents cycle through `distinct` values, with
    // pseudo-random sign+mantissa, so the palette has exactly `distinct` entries.
    fn make_bf16(distinct: usize, n: usize) -> Vec<u8> {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            let exp = (96 + (i % distinct)) as u16 & 0xFF; // exponents in a tight band
            let sign = ((state >> 31) & 1) as u16;
            let mant = (state & 0x7F) as u16;
            let bits = (sign << 15) | (exp << 7) | mant;
            out.extend_from_slice(&bits.to_le_bytes());
        }
        out
    }

    #[test]
    fn bitplane_roundtrip_bit_exact_various_palettes() {
        let codec = BitplaneCodec;
        for &distinct in &[1usize, 2, 3, 17, 32, 64] {
            let bytes = make_bf16(distinct, 1000);
            let meta = EncodeMeta { name: "w".into(), shape: vec![1000], dtype: "bf16".into() };
            let enc = codec.encode(&bytes, &meta).unwrap();
            // not raw-fallback for palette <= 64 => must be smaller than bf16
            assert!(enc.data.len() < bytes.len(), "distinct={distinct}: must compress");
            let dec = codec.decode(&enc.data, &dmeta()).unwrap();
            assert_eq!(dec, bytes, "distinct={distinct}: must be bit-exact lossless");
        }
    }

    #[test]
    fn bitplane_roundtrip_raw_fallback_over_64_exponents() {
        let codec = BitplaneCodec;
        let bytes = make_bf16(120, 2000); // >64 distinct => raw fallback
        let meta = EncodeMeta { name: "w".into(), shape: vec![2000], dtype: "bf16".into() };
        let enc = codec.encode(&bytes, &meta).unwrap();
        let dec = codec.decode(&enc.data, &dmeta()).unwrap();
        assert_eq!(dec, bytes, "raw-fallback must be bit-exact lossless");
    }

    #[test]
    fn bitplane_roundtrip_tail_and_edge_sizes() {
        let codec = BitplaneCodec;
        for &n in &[0usize, 1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 100] {
            let bytes = make_bf16(32, n); // w=5
            let meta = EncodeMeta { name: "w".into(), shape: vec![n as u64], dtype: "bf16".into() };
            let enc = codec.encode(&bytes, &meta).unwrap();
            let dec = codec.decode(&enc.data, &dmeta()).unwrap();
            assert_eq!(dec, bytes, "n={n}: must be bit-exact");
        }
    }

    #[test]
    fn bitplane_index_width_is_ceil_log2() {
        assert_eq!(index_width(1), 0);
        assert_eq!(index_width(2), 1);
        assert_eq!(index_width(3), 2);
        assert_eq!(index_width(4), 2);
        assert_eq!(index_width(5), 3);
        assert_eq!(index_width(32), 5);
        assert_eq!(index_width(33), 6);
        assert_eq!(index_width(64), 6);
    }
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test -p rtc-codec --lib bitplane -- --nocapture`
Expected: FAIL — compile error, `cannot find ... BitplaneCodec / index_width`.

- [ ] **Step 4: Implement `index_width`, `BitplaneCodec`, encode, scalar decode**

Insert ABOVE the `#[cfg(test)]` module in `bitplane.rs`:

```rust
/// Minimum bits to index a palette of `p` entries. `p<=1` needs 0 bits
/// (every weight uses palette[0]); otherwise ceil(log2(p)).
pub fn index_width(p: usize) -> u8 {
    if p <= 1 { 0 } else { (usize::BITS - (p - 1).leading_zeros()) as u8 }
}

pub struct BitplaneCodec;

impl BitplaneCodec {
    pub const ID: &'static str = "rtc-bitplane-v1";
}

const HEADER_LEN: usize = 4 + 1 + 1 + 8 + 1 + 1; // magic+version+flags+n+palette_len+width = 16
const FLAG_RAW: u8 = 0x01;

impl TensorCodec for BitplaneCodec {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn encode(&self, input: &[u8], meta: &EncodeMeta) -> Result<EncodedChunk> {
        if meta.dtype != "bf16" {
            return Err(CodecError::InvalidData(format!(
                "rtc-bitplane-v1 only supports bf16, got {}",
                meta.dtype
            )));
        }
        if input.len() % 2 != 0 {
            return Err(CodecError::InvalidData("bf16 byte length must be even".into()));
        }
        let n = input.len() / 2;

        let mut present = [false; 256];
        let mut exps = Vec::with_capacity(n);
        let mut residuals = Vec::with_capacity(n);
        for w in input.chunks_exact(2) {
            let bits = u16::from_le_bytes([w[0], w[1]]);
            let (e, r) = split_bf16(bits);
            present[e as usize] = true;
            exps.push(e);
            residuals.push(r);
        }
        let palette: Vec<u8> =
            (0..256usize).filter(|&i| present[i]).map(|i| i as u8).collect();

        let mut data = Vec::with_capacity(HEADER_LEN + palette.len() + n * 2);
        data.extend_from_slice(b"RTCB");
        data.push(1); // version

        if palette.len() > 64 {
            // Raw fallback: not usefully palette-compressible.
            data.push(FLAG_RAW);
            data.extend_from_slice(&(n as u64).to_le_bytes());
            data.push(0); // palette_len
            data.push(0); // index_width
            data.extend_from_slice(input);
            return Ok(EncodedChunk {
                codec_id: Self::ID.to_string(),
                data,
                original_size: input.len() as u64,
            });
        }

        let w = index_width(palette.len());
        data.push(0); // flags
        data.extend_from_slice(&(n as u64).to_le_bytes());
        data.push(palette.len() as u8);
        data.push(w);
        data.extend_from_slice(&palette);

        if w > 0 {
            let mut map = [0u8; 256];
            for (i, &e) in palette.iter().enumerate() {
                map[e as usize] = i as u8;
            }
            let mut bw = BitWriter::new();
            for &e in &exps {
                bw.write(map[e as usize] as u32, w);
            }
            data.extend_from_slice(&bw.finish());
        }
        data.extend_from_slice(&residuals);

        Ok(EncodedChunk {
            codec_id: Self::ID.to_string(),
            data,
            original_size: input.len() as u64,
        })
    }

    fn decode(&self, encoded: &[u8], _meta: &DecodeMeta) -> Result<Vec<u8>> {
        let err = || CodecError::InvalidData("truncated rtc-bitplane-v1 chunk".to_string());
        if encoded.len() < HEADER_LEN || &encoded[0..4] != b"RTCB" || encoded[4] != 1 {
            return Err(err());
        }
        let flags = encoded[5];
        let n = u64::from_le_bytes(encoded[6..14].try_into().map_err(|_| err())?) as usize;
        let p = encoded[14] as usize;
        let w = encoded[15];
        let mut off = HEADER_LEN;

        if flags & FLAG_RAW != 0 {
            let body = encoded.get(off..).ok_or_else(err)?;
            if body.len() < n * 2 {
                return Err(err());
            }
            return Ok(body[..n * 2].to_vec());
        }

        let palette = encoded.get(off..off + p).ok_or_else(err)?;
        off += p;
        let idx_bytes = (n * w as usize + 7) / 8;
        let idx_plane = encoded.get(off..off + idx_bytes).ok_or_else(err)?;
        off += idx_bytes;
        let residuals = encoded.get(off..off + n).ok_or_else(err)?;

        let mut out = vec![0u8; n * 2];
        if n == 0 {
            return Ok(out);
        }
        if w == 0 {
            if p == 0 {
                return Err(err());
            }
            let exp = palette[0];
            for i in 0..n {
                let bits = join_bf16(exp, residuals[i]);
                out[2 * i] = bits as u8;
                out[2 * i + 1] = (bits >> 8) as u8;
            }
            return Ok(out);
        }

        let mut reader = BufferedBitReader::new(idx_plane);
        for i in 0..n {
            reader.refill();
            let idx = reader.peek(w) as usize;
            reader.consume(w);
            if idx >= p {
                return Err(CodecError::InvalidData(
                    "rtc-bitplane-v1: palette index out of range".into(),
                ));
            }
            let bits = join_bf16(palette[idx], residuals[i]);
            out[2 * i] = bits as u8;
            out[2 * i + 1] = (bits >> 8) as u8;
        }
        Ok(out)
    }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p rtc-codec --lib bitplane -- --nocapture`
Expected: PASS — all 4 tests.

- [ ] **Step 6: Run the full crate suite**

Run: `cargo test -p rtc-codec`
Expected: PASS — existing tests + the 4 new ones, 0 failures.

- [ ] **Step 7: Commit**

```bash
git add crates/rtc-codec/src/bitplane.rs crates/rtc-codec/src/lib.rs
git commit -m "feat(rtc-codec): rtc-bitplane-v1 lossless codec encode + scalar decode (R143 REEPLANE)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: NEON `decode_neon_w5` kernel + bit-identical parity (Phase B, part 1)

**Files:**
- Modify: `crates/rtc-codec/src/bitplane.rs`
- Test: in `crates/rtc-codec/src/bitplane.rs`

**Interfaces:**
- Consumes: the Task 1 format + scalar `decode` (parity oracle).
- Produces: `#[cfg(target_arch = "aarch64")] pub fn decode_neon_w5(palette: &[u8], idx_plane: &[u8], residuals: &[u8], n: usize) -> Vec<u8>` — decodes a `w=5` bit-plane (palette ≤ 32) to bf16 bytes, bit-identical to scalar `decode`.

**Note on scope:** the kernel is specialized to `w=5` because the real Llama 1B embedding has exactly 32 distinct exponents (`index_width(32)=5`) — this is the case the feasibility gate measures. Other widths use the scalar path. The parity test is the correctness gate: implement, run it, fix until bit-identical, then proceed.

- [ ] **Step 1: Write the failing parity test**

Add to the `#[cfg(test)] mod tests` in `bitplane.rs`:

```rust
#[cfg(target_arch = "aarch64")]
#[test]
fn decode_neon_w5_matches_scalar_bit_for_bit() {
    let codec = BitplaneCodec;
    // Various sizes incl. tail (non-multiple-of-8) and small, all 32-exponent (w=5).
    for &n in &[8usize, 16, 17, 31, 33, 64, 1000, 4096, 4099] {
        let bytes = make_bf16(32, n);
        let meta = EncodeMeta { name: "w".into(), shape: vec![n as u64], dtype: "bf16".into() };
        let enc = codec.encode(&bytes, &meta).unwrap();
        // Re-parse the chunk sections the NEON kernel needs.
        assert_eq!(&enc.data[0..4], b"RTCB");
        let p = enc.data[14] as usize;
        let w = enc.data[15];
        assert_eq!(w, 5, "n={n}: expected w=5 for 32 exponents");
        let mut off = 16;
        let palette = &enc.data[off..off + p]; off += p;
        let idx_bytes = (n * 5 + 7) / 8;
        let idx_plane = &enc.data[off..off + idx_bytes]; off += idx_bytes;
        let residuals = &enc.data[off..off + n];

        let scalar = codec.decode(&enc.data, &DecodeMeta {
            codec_id: "rtc-bitplane-v1".into(), uncompressed_size: 0 }).unwrap();
        let neon = decode_neon_w5(palette, idx_plane, residuals, n);
        assert_eq!(neon, scalar, "n={n}: NEON decode must equal scalar bit-for-bit");
        assert_eq!(neon, bytes, "n={n}: NEON decode must be lossless");
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p rtc-codec --lib decode_neon_w5 -- --nocapture`
Expected: FAIL — `cannot find function decode_neon_w5`.

- [ ] **Step 3: Implement `decode_neon_w5`**

Add to `bitplane.rs` (above the test module). The kernel processes 8 weights per iteration: unpack 8 five-bit indices from each 5-byte group via a 2-byte window + per-lane right shift, gather exponents with `vtbl4_u8`, reconstruct bf16, store. A scalar tail (reusing the scalar logic) covers the trailing weights and any group whose 8-byte load would read past the index plane.

```rust
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decode_w5_neon_inner(
    palette: &[u8],
    idx_plane: &[u8],
    residuals: &[u8],
    n: usize,
    out: &mut [u8],
) {
    use std::arch::aarch64::*;

    // Palette padded to 32 bytes for vtbl4 (indices < palette.len() <= 32).
    let mut pal = [0u8; 32];
    pal[..palette.len()].copy_from_slice(palette);
    let pal_tbl = uint8x8x4_t(
        vld1_u8(pal.as_ptr()),
        vld1_u8(pal.as_ptr().add(8)),
        vld1_u8(pal.as_ptr().add(16)),
        vld1_u8(pal.as_ptr().add(24)),
    );

    // For 8 indices (5 bits each) packed MSB-first in 5 bytes:
    // index j occupies bits [5j, 5j+5); read a 2-byte big-endian window at
    // byte 5j/8, then right-shift by (11 - 5j%8) and mask 0x1f.
    let bidx_hi: [u8; 8] = [0, 0, 1, 1, 2, 3, 3, 4];
    let bidx_lo: [u8; 8] = [1, 1, 2, 2, 3, 4, 4, 5];
    let neg_shift: [i16; 8] = [-11, -6, -9, -4, -7, -10, -5, -8];
    let vhi = vld1_u8(bidx_hi.as_ptr());
    let vlo = vld1_u8(bidx_lo.as_ptr());
    let vshift = vld1q_s16(neg_shift.as_ptr());
    let mask5 = vdupq_n_u16(0x1f);
    let mask80 = vdupq_n_u16(0x80);
    let mask7f = vdupq_n_u16(0x7f);

    // SIMD-safe groups: each iteration loads 8 bytes at offset g*5, so require
    // g*5 + 8 <= idx_plane.len(); also g*8 + 8 <= n.
    let groups_by_bytes = if idx_plane.len() >= 8 { (idx_plane.len() - 8) / 5 + 1 } else { 0 };
    let simd_groups = core::cmp::min(n / 8, groups_by_bytes);

    let out_u16 = out.as_mut_ptr() as *mut u16;
    for g in 0..simd_groups {
        let grp = vld1_u8(idx_plane.as_ptr().add(g * 5)); // 8 bytes (need up to byte 5)
        let hi = vtbl1_u8(grp, vhi);
        let lo = vtbl1_u8(grp, vlo);
        let window = vorrq_u16(vshlq_n_u16(vmovl_u8(hi), 8), vmovl_u8(lo));
        let idx16 = vandq_u16(vshlq_u16(window, vshift), mask5);
        let idx8 = vmovn_u16(idx16);
        let exp8 = vtbl4_u8(pal_tbl, idx8);
        let res8 = vld1_u8(residuals.as_ptr().add(g * 8));
        let res16 = vmovl_u8(res8);
        let exp16 = vmovl_u8(exp8);
        let sign = vshlq_n_u16(vandq_u16(res16, mask80), 8);
        let ep = vshlq_n_u16(exp16, 7);
        let mant = vandq_u16(res16, mask7f);
        let bf16 = vorrq_u16(vorrq_u16(sign, ep), mant);
        vst1q_u16(out_u16.add(g * 8), bf16);
    }

    // Scalar tail: weights [simd_groups*8 .. n]. simd_groups*8 is a multiple of 8,
    // so its bit offset (×5) is a multiple of 40 bits = 5 bytes => byte-aligned.
    let tail_start = simd_groups * 8;
    if tail_start < n {
        let byte_off = (tail_start / 8) * 5;
        let mut reader = BufferedBitReader::new(&idx_plane[byte_off..]);
        for i in tail_start..n {
            reader.refill();
            let idx = reader.peek(5) as usize;
            reader.consume(5);
            let bits = join_bf16(palette[idx], residuals[i]);
            out[2 * i] = bits as u8;
            out[2 * i + 1] = (bits >> 8) as u8;
        }
    }
}

/// NEON `w=5` bit-plane decode to bf16 bytes. Bit-identical to scalar `decode`.
#[cfg(target_arch = "aarch64")]
pub fn decode_neon_w5(palette: &[u8], idx_plane: &[u8], residuals: &[u8], n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n * 2];
    unsafe { decode_w5_neon_inner(palette, idx_plane, residuals, n, &mut out) };
    out
}
```

- [ ] **Step 4: Run the parity test; fix until bit-identical**

Run: `cargo test -p rtc-codec --lib decode_neon_w5 -- --nocapture`
Expected: PASS. If it fails on a specific `n`, the mismatch is in the unpack constants or the reconstruct — compare `neon` vs `scalar` at the first differing weight, check the `bidx_*`/`neg_shift` lane that produced it. Do not proceed until bit-identical.

- [ ] **Step 5: Run the full crate suite**

Run: `cargo test -p rtc-codec`
Expected: PASS — all green.

- [ ] **Step 6: Commit**

```bash
git add crates/rtc-codec/src/bitplane.rs
git commit -m "feat(rtc-codec): NEON w=5 bit-plane decode kernel, parity-exact (R143 REEPLANE)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Feasibility bench + measurement (Phase B, part 2 — the gate)

**Files:**
- Modify: `crates/rtc-codec/src/bitplane.rs`

**Interfaces:**
- Consumes: `decode_neon_w5` (Task 2); `/tmp/rllm-bf16-sample.bin` (the real 525 MB Llama 1B bf16 embedding, written by `dump_bf16_embedding_sample` in `crates/rllm-runtime/src/lazy.rs:1227`).
- Produces: single-core Gweight/s, speedup vs scalar bitplane decode, and the GO/MARGINAL/NO-GO verdict (printed; transcribed into Task 4).

- [ ] **Step 1: Add the `#[ignore]` feasibility bench**

Add to the `#[cfg(test)] mod tests` in `bitplane.rs`:

```rust
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore]
fn bitplane_neon_decode_feasibility() {
    let bytes = std::fs::read("/tmp/rllm-bf16-sample.bin")
        .expect("run dump_bf16_embedding_sample first (see plan Task 3, Step 2)");
    let n = bytes.len() / 2;
    let codec = BitplaneCodec;
    let meta = EncodeMeta { name: "embed".into(), shape: vec![n as u64], dtype: "bf16".into() };
    let enc = codec.encode(&bytes, &meta).unwrap();
    let bits_per_weight = (enc.data.len() as f64 * 8.0) / n as f64;
    let p = enc.data[14] as usize;
    let w = enc.data[15];
    assert_eq!(w, 5, "expected w=5 (32 exponents) for the real embedding");
    let mut off = 16;
    let palette = enc.data[off..off + p].to_vec(); off += p;
    let idx_bytes = (n * 5 + 7) / 8;
    let idx_plane = enc.data[off..off + idx_bytes].to_vec(); off += idx_bytes;
    let residuals = enc.data[off..off + n].to_vec();

    // Correctness on the real sample.
    let neon = decode_neon_w5(&palette, &idx_plane, &residuals, n);
    assert_eq!(neon, bytes, "NEON decode must be lossless on the real embedding");

    // Timed NEON decode (materializing; the fused kernel would skip the store,
    // so this is a conservative floor).
    let iters = 8;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        let d = decode_neon_w5(&palette, &idx_plane, &residuals, n);
        std::hint::black_box(&d);
    }
    let neon_s = t.elapsed().as_secs_f64() / iters as f64;
    let neon_gw = (n as f64 / 1e9) / neon_s;

    // Scalar bitplane decode for the speedup ratio (one pass).
    let t = std::time::Instant::now();
    let sc = codec.decode(&enc.data, &DecodeMeta {
        codec_id: "rtc-bitplane-v1".into(), uncompressed_size: 0 }).unwrap();
    std::hint::black_box(&sc);
    let scalar_s = t.elapsed().as_secs_f64();
    let scalar_gw = (n as f64 / 1e9) / scalar_s;

    let agg = neon_gw * 3.5;
    let verdict = if agg >= 12.0 { "GO" } else if agg >= 5.0 { "MARGINAL" } else { "NO-GO" };
    eprintln!(
        "\n=== R143 REEPLANE bit-plane NEON decode FEASIBILITY ===\n\
         weights={n}  bits/weight={bits_per_weight:.3}  palette={p} w={w}\n\
         NEON single-core: {neon_gw:.2} Gweight/s  ({:.1} ms/decode, materializing)\n\
         scalar bitplane: {scalar_gw:.3} Gweight/s  (NEON speedup {:.1}x)\n\
         aggregate (x3.5): {agg:.1} Gweight/s\n\
         threshold: GO>=12, MARGINAL 5-12, NO-GO<5 (Gweight/s aggregate)\n\
         VERDICT: {verdict}\n",
        neon_s * 1000.0, neon_gw / scalar_gw,
    );
}
```

- [ ] **Step 2: Ensure the real sample exists**

Run (skip if `/tmp/rllm-bf16-sample.bin` already present from R142):

```bash
test -f /tmp/rllm-bf16-sample.bin || \
  cargo test -p rllm-runtime --release dump_bf16_embedding_sample -- --ignored --nocapture
```
Expected: file present (525,336,576 bytes).

- [ ] **Step 3: Run the bench (release) and capture the verdict**

Run:

```bash
cargo test -p rtc-codec --release bitplane_neon_decode_feasibility -- --ignored --nocapture
```
Expected: the `=== R143 REEPLANE bit-plane NEON decode FEASIBILITY ===` block. **Record verbatim**: `bits/weight` (~13), NEON `Gweight/s` + `ms`, the `speedup` vs scalar, the `aggregate`, the `VERDICT`. These feed Task 4.

- [ ] **Step 4: Commit**

```bash
git add crates/rtc-codec/src/bitplane.rs
git commit -m "test(rtc-codec): R143 REEPLANE NEON bit-plane decode feasibility bench + measurement

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Trial report + index + memory

**Files:**
- Create: `docs/benchmarks/trials/<verdict-folder>/2026-06-19-r143-reeplane-bitplane-codec.md` (`success/` if GO, `inconclusive/` if MARGINAL, `failed/` if NO-GO)
- Modify: `docs/benchmarks/trials/index.md`
- Modify: `/Users/deansanbhnanwr/.claude/projects/-Users-deansanbhnanwr-Projects-rllm/memory/rllm-speed-thesis-streaming-vs-resident.md`

**Interfaces:**
- Consumes: the measured numbers + verdict from Task 3 Step 3; the template `docs/benchmarks/templates/trial-report.md`; the R142 trial as a format reference.

- [ ] **Step 1: Read the template and the R142 trial**

Run: `cat docs/benchmarks/templates/trial-report.md docs/benchmarks/trials/failed/2026-06-19-r142-reedrip-fast-dfloat-decode.md`
Expected: the required section structure.

- [ ] **Step 2: Write the trial report**

Create the report in the verdict-matching folder. Fill from Task 3's measured numbers:
- **Scope → REE kernel:** `REEPLANE (working name; Erik's final call)`. Mode: `experimental (compressed-resident codec, decode only)`. Artifact: `Llama-3.2-1B-Instruct-raw.rllm` embedding (262.7M bf16, palette 32, w=5). Device: Apple A18 Pro. Bottleneck tag: IO/decode.
- **Hypothesis:** a fixed-width palette index + NEON `tbl`-gather decode is fast enough (unlike Huffman) for compressed-resident, at ~13 bits/weight (19% RAM).
- **Results:** the table — scalar bitplane vs NEON Gweight/s + the NEON speedup; bits/weight (~13); aggregate; verdict; bit-identical parity confirmed.
- **Analysis:** compare aggregate to the 12 Gweight/s plain-bf16 rate; place the result in the R140-R142 frontier (Huffman 34%/0.18 Gw/s vs bit-plane 19%/<measured>). State the ratio-vs-decode-speed tradeoff conclusion. If GO/MARGINAL, note Phase C (fused decode→bfdot, no DRAM store) is the next gate; if NO-GO, the frontier is closed for lossless bf16 compressed-resident on this CPU.
- **Decision:** the verdict.
- **Next Experiment:** Phase C fused kernel if GO/MARGINAL; otherwise ship storage-compression / write up the frontier.

- [ ] **Step 3: Add the index row**

In `docs/benchmarks/trials/index.md`, add an R143 row mirroring the R142 row's columns (date | trial | folder | model | mode | bottleneck tag | baseline | result | decision | paper value). Baseline = R142 NO-GO (Huffman buffered 0.18 Gw/s); result = NEON bit-plane `<measured>` Gw/s + verdict.

- [ ] **Step 4: Update memory**

Append the measured R143 number to `rllm-speed-thesis-streaming-vs-resident.md`: bit-plane NEON decode `<measured>` single-core / `<agg>` aggregate at 13 bits/weight (19% RAM) → `<verdict>`; states the resolved ratio-vs-decode-speed tradeoff and whether lossless compressed-resident is viable on CPU.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmarks/trials/ "/Users/deansanbhnanwr/.claude/projects/-Users-deansanbhnanwr-Projects-rllm/memory/rllm-speed-thesis-streaming-vs-resident.md"
git commit -m "docs(bench): R143 REEPLANE bit-plane NEON decode trial (<verdict>) + index + memory

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Verification (end-to-end)

1. `cargo test -p rtc-codec` → all green, including bitplane roundtrip (palettes 1..64 + raw fallback + tail sizes) and the NEON-vs-scalar parity test.
2. `cargo build` (workspace) → compiles, no new dependencies.
3. The `#[ignore]` bench printed the `=== R143 REEPLANE ... FEASIBILITY ===` block with a real verdict on the 525 MB sample.
4. Trial report in the verdict folder with measured numbers; `index.md` has the R143 row; memory updated.
5. `git grep -n "BitplaneCodec\|decode_neon_w5\|bitplane" crates/rllm-runtime crates/rllm-cli` → **no hits** (no runtime wiring; gate stayed isolated to the codec crate).

## Out of scope (Phase C — gated behind a GO/MARGINAL, do NOT build here)

- Fused decode→bfdot resident GEMV (decode a tile into registers, feed R141's `bf16_row_dot_bf16`, no DRAM store).
- Registering `BitplaneCodec` in `codec_for_id` (`crates/rllm-runtime/src/loader.rs:121`); runtime/`--fast` wiring; packing models with the codec.
- General-width NEON decode (only `w=5` is needed for the gate); multi-threaded decode.
- Compressing the residual; KV-cache; GPU; sub-bf16 precision.
