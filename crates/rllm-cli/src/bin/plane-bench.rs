// Decisive test of the "compressed-resident + fused decode beats bf16-raw on slow
// memory" hypothesis. Two GEMVs over the SAME weights:
//   A) bf16-raw     : read 2 bytes/weight from RAM, dot.
//   B) bitplane-fused: keep bit-plane (~1.6 B/weight) resident, decode each row into an
//      L1 scratch (rtc-codec's real NEON kernel), dot. Reads ~19% fewer bytes from RAM.
// On fast memory (A18) B loses (decode additive, R144 NO-GO). Hypothesis: on slow phone
// memory B WINS (fewer bytes from the bottleneck, decode hidden). Lossless either way.

use rtc_codec::{BitplaneCodec, EncodeMeta, TensorCodec};
use std::time::Instant;

#[inline]
fn bf16_dot(w: &[u8], x: &[f32], hidden: usize) -> f32 {
    let mut acc = 0f32;
    for i in 0..hidden {
        let b = u16::from_le_bytes([w[2 * i], w[2 * i + 1]]);
        acc += f32::from_bits((b as u32) << 16) * x[i];
    }
    acc
}

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let vocab = *a.first().unwrap_or(&8192);
    let hidden = *a.get(1).unwrap_or(&2048);
    let iters = *a.get(2).unwrap_or(&20);
    let n = vocab * hidden;
    println!("GEMV {vocab}x{hidden} ({:.0} MB bf16), iters={iters}", (n * 2) as f64 / 1e6);

    // Realistic-ish bf16 weights: small values so exponents cluster (-> w in {5,6}).
    let mut bf16 = vec![0u8; n * 2];
    let mut s = 0x12345u32;
    for i in 0..n {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let u = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
        let b = (((u * 0.08).to_bits()) >> 16) as u16;
        bf16[2 * i] = (b & 0xff) as u8;
        bf16[2 * i + 1] = (b >> 8) as u8;
    }

    let enc = BitplaneCodec
        .encode(&bf16, &EncodeMeta { name: "e".into(), shape: vec![n as u64], dtype: "bf16".into() })
        .unwrap();
    let w = enc.data[15] as usize;
    let p = enc.data[14] as usize;
    assert!(w == 5 || w == 6, "expected bit-plane w in {{5,6}}, got w={w} (raw fallback?)");
    assert_eq!((hidden * w) % 8, 0, "need hidden*w byte-aligned for per-row decode");
    let mut off = 16;
    let palette = enc.data[off..off + p].to_vec();
    off += p;
    let idx_bytes = (n * w + 7) / 8;
    let idx = enc.data[off..off + idx_bytes].to_vec();
    off += idx_bytes;
    let residuals = enc.data[off..off + n].to_vec();
    let row_idx_bytes = hidden * w / 8;

    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    let mut ya = vec![0f32; vocab];
    let mut yb = vec![0f32; vocab];
    let mut scratch = vec![0u8; hidden * 2];

    let fused_row = |scratch: &mut [u8], r: usize| {
        #[cfg(target_arch = "aarch64")]
        rtc_codec::decode_bitplane_row_into(
            &palette,
            &idx[r * row_idx_bytes..],
            &residuals[r * hidden..],
            hidden,
            w as u8,
            scratch,
        );
        #[cfg(not(target_arch = "aarch64"))]
        let _ = (scratch, r);
    };

    // correctness
    for r in 0..vocab {
        ya[r] = bf16_dot(&bf16[r * hidden * 2..], &x, hidden);
    }
    for r in 0..vocab {
        fused_row(&mut scratch, r);
        yb[r] = bf16_dot(&scratch, &x, hidden);
    }
    let max_diff = ya.iter().zip(&yb).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);

    // bench A: bf16-raw
    let t = Instant::now();
    for _ in 0..iters {
        for r in 0..vocab {
            ya[r] = bf16_dot(&bf16[r * hidden * 2..], &x, hidden);
        }
        std::hint::black_box(&ya);
    }
    let a_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // bench B: bitplane-fused
    let t = Instant::now();
    for _ in 0..iters {
        for r in 0..vocab {
            fused_row(&mut scratch, r);
            yb[r] = bf16_dot(&scratch, &x, hidden);
        }
        std::hint::black_box(&yb);
    }
    let b_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let bf16_mb = (n * 2) as f64 / 1e6;
    let plane_mb = (p + idx_bytes + n) as f64 / 1e6;
    println!("lossless parity max_diff = {max_diff:.5}  (w={w})");
    println!("A) bf16-raw       : {a_ms:.1} ms  | reads {bf16_mb:.0} MB/token");
    println!("B) bitplane-fused : {b_ms:.1} ms  | reads {plane_mb:.0} MB/token ({:.0}% less)", (1.0 - plane_mb / bf16_mb) * 100.0);
    println!("speedup B vs A    : {:.2}x", a_ms / b_ms);
}
