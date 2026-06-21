// De-risk the lossless-fast-on-edge frontier: does a THREADED + NEON fused bit-plane
// GEMV (compressed-resident: read ~13 bits/weight, decode each row into L1, dot) beat
// a THREADED + NEON bf16 GEMV (read 16 bits/weight) on a weak phone? The single-thread
// scalar plane-bench said no (decode additive); this tests the real pooled+NEON regime.
// If fused wins here, the runtime bit-plane-fused path is worth building.

use rtc_codec::{BitplaneCodec, EncodeMeta, TensorCodec};
use std::time::Instant;

#[derive(Clone, Copy)]
struct P(*mut f32);
unsafe impl Send for P {}
unsafe impl Sync for P {}
impl P {
    #[inline]
    unsafe fn at(self, i: usize) -> *mut f32 {
        self.0.add(i)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_bf16_dot(w: &[u8], x: &[f32], h: usize) -> f32 {
    use std::arch::aarch64::*;
    let (mut a0, mut a1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let wp = w.as_ptr() as *const u16;
    let xp = x.as_ptr();
    let mut i = 0;
    while i + 8 <= h {
        let b = vld1q_u16(wp.add(i));
        a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(b))), vld1q_f32(xp.add(i)));
        a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_high_n_u16::<16>(b)), vld1q_f32(xp.add(i + 4)));
        i += 8;
    }
    let mut s = vaddvq_f32(vaddq_f32(a0, a1));
    while i < h {
        s += f32::from_bits((u16::from_le_bytes([w[2 * i], w[2 * i + 1]]) as u32) << 16) * x[i];
        i += 1;
    }
    s
}
#[cfg(not(target_arch = "aarch64"))]
unsafe fn neon_bf16_dot(w: &[u8], x: &[f32], h: usize) -> f32 {
    let mut s = 0f32;
    for i in 0..h {
        s += f32::from_bits((u16::from_le_bytes([w[2 * i], w[2 * i + 1]]) as u32) << 16) * x[i];
    }
    s
}

fn time_threaded<F: Fn(usize, P) + Sync>(vocab: usize, threads: usize, iters: usize, f: F) -> f64 {
    let t = Instant::now();
    for _ in 0..iters {
        let mut y = vec![0f32; vocab];
        let yp = P(y.as_mut_ptr());
        let rows_per = vocab.div_ceil(threads);
        let fr = &f;
        std::thread::scope(|s| {
            let mut base = 0usize;
            while base < vocab {
                let lo = base;
                let hi = (base + rows_per).min(vocab);
                s.spawn(move || {
                    for r in lo..hi {
                        fr(r, yp);
                    }
                });
                base = hi;
            }
        });
        std::hint::black_box(&y);
    }
    t.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let vocab = *a.first().unwrap_or(&8192);
    let hidden = *a.get(1).unwrap_or(&2048);
    let iters = *a.get(2).unwrap_or(&20);
    let n = vocab * hidden;
    println!("GEMV {vocab}x{hidden} ({} MB bf16), iters={iters}", (n * 2) / 1_000_000);

    let mut bf16 = vec![0u8; n * 2];
    let mut s = 0x12345u32;
    for i in 0..n {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let b = (((((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.08).to_bits()) >> 16) as u16;
        bf16[2 * i] = (b & 0xff) as u8;
        bf16[2 * i + 1] = (b >> 8) as u8;
    }
    let enc = BitplaneCodec
        .encode(&bf16, &EncodeMeta { name: "e".into(), shape: vec![n as u64], dtype: "bf16".into() })
        .unwrap();
    let w = enc.data[15] as usize;
    let p = enc.data[14] as usize;
    assert!(w == 5 || w == 6, "want w in {{5,6}}, got {w}");
    assert_eq!((hidden * w) % 8, 0);
    let mut off = 16;
    let palette = enc.data[off..off + p].to_vec();
    off += p;
    let idx_bytes = (n * w + 7) / 8;
    let idx = enc.data[off..off + idx_bytes].to_vec();
    off += idx_bytes;
    let residuals = enc.data[off..off + n].to_vec();
    let row_idx_bytes = hidden * w / 8;
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.013).sin()).collect();

    let bf16_mb = (n * 2) as f64 / 1e6;
    let plane_mb = (p + idx_bytes + n) as f64 / 1e6;
    println!("bf16 {bf16_mb:.0} MB  vs  bit-plane {plane_mb:.0} MB ({:.0}% less), w={w}\n", (1.0 - plane_mb / bf16_mb) * 100.0);

    for &t in &[1usize, 2, 4, 6] {
        let xa = &x;
        let bf = &bf16;
        let a_ms = time_threaded(vocab, t, iters, move |r, yp| {
            let d = unsafe { neon_bf16_dot(&bf[r * hidden * 2..], xa, hidden) };
            unsafe { *yp.at(r) = d };
        });
        let xb = &x;
        let (pal, ix, res) = (&palette, &idx, &residuals);
        let b_ms = time_threaded(vocab, t, iters, move |r, yp| {
            let mut scratch = [0u8; 8192 * 2]; // hidden<=8192
            #[cfg(target_arch = "aarch64")]
            rtc_codec::decode_bitplane_row_into(pal, &ix[r * row_idx_bytes..], &res[r * hidden..], hidden, w as u8, &mut scratch[..hidden * 2]);
            let d = unsafe { neon_bf16_dot(&scratch[..hidden * 2], xb, hidden) };
            unsafe { *yp.at(r) = d };
        });
        println!(
            "threads={t}: bf16 {a_ms:5.1}ms ({:4.1} GB/s)  |  bitplane-fused {b_ms:5.1}ms ({:4.1} GB/s)  |  speedup {:.2}x",
            bf16_mb / a_ms, plane_mb / b_ms, a_ms / b_ms
        );
    }
}
