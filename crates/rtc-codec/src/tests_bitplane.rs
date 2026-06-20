// Unit tests for rtc-bitplane-v1, split from bitplane.rs to keep the production
// module under the modular-code-guard line budget (test code is exempt).
// Included via #[path] as a child module of bitplane, so `super::*` reaches
// the crate's private SIMD internals (decode_scalar_w, decode_w5_neon_inner, …).
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
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
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

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn decode_neon_w5_matches_scalar_bit_for_bit() {
        let codec = BitplaneCodec;
        // Sizes >= 32 so make_bf16(32, n) yields all 32 distinct exponents (w=5),
        // covering tail cases (n%8 in {0,1,7,3}) and the SIMD/scalar boundary.
        for &n in &[32usize, 33, 39, 40, 47, 64, 1000, 4096, 4099] {
            let bytes = make_bf16(32, n);
            let meta = EncodeMeta { name: "w".into(), shape: vec![n as u64], dtype: "bf16".into() };
            let enc = codec.encode(&bytes, &meta).unwrap();
            assert_eq!(&enc.data[0..4], b"RTCB");
            let p = enc.data[14] as usize;
            let w = enc.data[15];
            assert_eq!(w, 5, "n={n}: expected w=5 for 32 exponents");
            let mut off = 16;
            let palette = &enc.data[off..off + p];
            off += p;
            let idx_bytes = (n * 5 + 7) / 8;
            let idx_plane = &enc.data[off..off + idx_bytes];
            off += idx_bytes;
            let residuals = &enc.data[off..off + n];

            let scalar = codec
                .decode(
                    &enc.data,
                    &DecodeMeta { codec_id: "rtc-bitplane-v1".into(), uncompressed_size: 0 },
                )
                .unwrap();
            let neon = decode_neon_w5(palette, idx_plane, residuals, n);
            assert_eq!(neon, scalar, "n={n}: NEON decode must equal scalar bit-for-bit");
            assert_eq!(neon, bytes, "n={n}: NEON decode must be lossless");
        }
    }

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
        let palette = enc.data[off..off + p].to_vec();
        off += p;
        let idx_bytes = (n * 5 + 7) / 8;
        let idx_plane = enc.data[off..off + idx_bytes].to_vec();
        off += idx_bytes;
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
        let sc = codec
            .decode(
                &enc.data,
                &DecodeMeta { codec_id: "rtc-bitplane-v1".into(), uncompressed_size: 0 },
            )
            .unwrap();
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
            neon_s * 1000.0,
            neon_gw / scalar_gw,
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn decode16_matches_8wide_bit_for_bit() {
        let codec = BitplaneCodec;
        // n >= 32 so make_bf16(32, n) yields all 32 distinct exponents (w=5).
        for &n in &[32usize, 48, 64, 1000, 4096, 4099, 65536] {
            let bytes = make_bf16(32, n);
            let meta = EncodeMeta { name: "w".into(), shape: vec![n as u64], dtype: "bf16".into() };
            let enc = codec.encode(&bytes, &meta).unwrap();
            assert_eq!(enc.data[15], 5, "n={n}: expected w=5");
            let p = enc.data[14] as usize;
            let mut off = 16;
            let palette = &enc.data[off..off + p];
            off += p;
            let idx_bytes = (n * 5 + 7) / 8;
            let idx_plane = &enc.data[off..off + idx_bytes];
            off += idx_bytes;
            let residuals = &enc.data[off..off + n];
            let eight = decode_neon_w5(palette, idx_plane, residuals, n);
            let mut sixteen = vec![0u8; n * 2];
            unsafe { decode16_w5_into(palette, idx_plane, residuals, n, &mut sixteen) };
            assert_eq!(sixteen, eight, "n={n}: 16-wide must equal 8-wide bit-for-bit");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn decode16_throughput_scout() {
        let bytes = std::fs::read("/tmp/rllm-bf16-sample.bin")
            .expect("run dump_bf16_embedding_sample first");
        let n = bytes.len() / 2;
        let codec = BitplaneCodec;
        let enc = codec
            .encode(&bytes, &EncodeMeta { name: "e".into(), shape: vec![n as u64], dtype: "bf16".into() })
            .unwrap();
        let p = enc.data[14] as usize;
        let mut off = 16;
        let palette = enc.data[off..off + p].to_vec();
        off += p;
        let idx_bytes = (n * 5 + 7) / 8;
        let idx_plane = enc.data[off..off + idx_bytes].to_vec();
        off += idx_bytes;
        let residuals = enc.data[off..off + n].to_vec();
        let mut out = vec![0u8; n * 2];

        // correctness
        unsafe { decode16_w5_into(&palette, &idx_plane, &residuals, n, &mut out) };
        assert_eq!(out, bytes, "16-wide lossless on real sample");

        let mut bench = |label: &str, f: &dyn Fn(&mut [u8])| -> f64 {
            f(&mut out); // warm
            let iters = 8;
            let t = std::time::Instant::now();
            for _ in 0..iters {
                f(&mut out);
                std::hint::black_box(&out);
            }
            let s = t.elapsed().as_secs_f64() / iters as f64;
            let gw = (n as f64 / 1e9) / s;
            eprintln!("  {label:10} {gw:.2} Gweight/s  ({:.1} ms)", s * 1000.0);
            gw
        };
        eprintln!("\n=== R146 16-wide decode THROUGHPUT SCOUT (single-core) ===");
        let g8 = bench("8-wide", &|o| decode_neon_w5_into(&palette, &idx_plane, &residuals, n, o));
        let g16 = bench("16-wide", &|o| unsafe {
            decode16_w5_into(&palette, &idx_plane, &residuals, n, o)
        });
        let agg = g16 * 3.5;
        eprintln!(
            "  16-wide vs 8-wide: {:.2}x   aggregate(x3.5): {:.1} Gweight/s   (need ~34 for the win)\n  VERDICT: {}\n",
            g16 / g8,
            agg,
            if agg >= 34.0 { "GO scout (build pipelined R146)" }
            else if g16 / g8 >= 1.3 { "PARTIAL (faster but short of 34 agg)" }
            else { "NO-GO (16-wide not enough)" }
        );
    }

    // R147: end-to-end capacity-bound proof. Streams >RAM raw-bf16 vs bit-plane
    // files COLD from SSD (F_NOCACHE), decoding the compressed one, and times both.
    // Proves the regime where lossless compression wins on CPU: model > RAM.
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn capacity_bound_stream_scout() {
        use std::io::{Read, Write};
        use std::os::unix::io::AsRawFd;
        extern "C" {
            fn fcntl(fd: i32, cmd: i32, arg: i32) -> i32;
        }
        const F_NOCACHE: i32 = 48;

        let bytes = std::fs::read("/tmp/rllm-bf16-sample.bin")
            .expect("run dump_bf16_embedding_sample first");
        let n = bytes.len() / 2;
        let codec = BitplaneCodec;
        let enc = codec
            .encode(&bytes, &EncodeMeta { name: "e".into(), shape: vec![n as u64], dtype: "bf16".into() })
            .unwrap();
        let p = enc.data[14] as usize;
        let mut off = 16;
        let palette = enc.data[off..off + p].to_vec();
        off += p;
        let idx_bytes = (n * 5 + 7) / 8;
        let idx_plane = enc.data[off..off + idx_bytes].to_vec();
        off += idx_bytes;
        let residuals = enc.data[off..off + n].to_vec();
        // one "copy" of compressed = idx_plane ++ residuals
        let comp_copy: Vec<u8> = idx_plane.iter().chain(residuals.iter()).copied().collect();

        // K copies so both files exceed RAM (~3 GB free) => true cold SSD reads.
        let k = 12usize;
        let raw_path = "/tmp/r147_raw.bin";
        let comp_path = "/tmp/r147_comp.bin";
        {
            let mut fr = std::fs::File::create(raw_path).unwrap();
            let mut fc = std::fs::File::create(comp_path).unwrap();
            for _ in 0..k {
                fr.write_all(&bytes).unwrap();
                fc.write_all(&comp_copy).unwrap();
            }
        }
        let raw_gb = (bytes.len() * k) as f64 / 1e9;
        let comp_gb = (comp_copy.len() * k) as f64 / 1e9;

        // stream RAW cold: read each 525MB copy, cheap dot (sum bf16 as f32).
        let raw_ms = {
            let mut f = std::fs::File::open(raw_path).unwrap();
            unsafe { fcntl(f.as_raw_fd(), F_NOCACHE, 1) };
            let mut buf = vec![0u8; bytes.len()];
            let t = std::time::Instant::now();
            for _ in 0..k {
                f.read_exact(&mut buf).unwrap();
                std::hint::black_box(&buf); // real dot (bfdot) is ~9ms/copy, negligible vs read
            }
            t.elapsed().as_secs_f64() * 1000.0
        };

        // stream COMPRESSED cold: read each copy, decode16 -> bf16, cheap dot.
        let comp_ms = {
            let mut f = std::fs::File::open(comp_path).unwrap();
            unsafe { fcntl(f.as_raw_fd(), F_NOCACHE, 1) };
            let mut buf = vec![0u8; comp_copy.len()];
            let mut decoded = vec![0u8; n * 2];
            let t = std::time::Instant::now();
            for _ in 0..k {
                f.read_exact(&mut buf).unwrap();
                unsafe {
                    decode16_w5_into(&palette, &buf[..idx_bytes], &buf[idx_bytes..], n, &mut decoded)
                };
                std::hint::black_box(&decoded);
            }
            t.elapsed().as_secs_f64() * 1000.0
        };

        let _ = std::fs::remove_file(raw_path);
        let _ = std::fs::remove_file(comp_path);

        eprintln!(
            "\n=== R147 CAPACITY-BOUND e2e stream SCOUT (cold SSD, files > RAM) ===\n\
             raw bf16   stream {raw_gb:.1} GB -> {raw_ms:.0} ms  ({:.2} GB/s)\n\
             bit-plane  stream {comp_gb:.1} GB -> {comp_ms:.0} ms  ({:.2} GB/s, incl. decode)\n\
             SPEEDUP: {:.2}x   (compressed reads 19% fewer bytes; decode hidden under SSD)\n\
             VERDICT: {}\n",
            raw_gb / (raw_ms / 1e3),
            comp_gb / (comp_ms / 1e3),
            raw_ms / comp_ms,
            if comp_ms < raw_ms { "GO -- lossless compression WINS when model streams from SSD" }
            else { "NO-GO" }
        );
    }

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

    // R149b: REEPLANE-W6. 34 distinct exponents => w=6 (the real Gemma 3 1B case).
    // The 16-wide vqtbl4q decode must equal the scalar BitplaneCodec::decode
    // bit-for-bit across the SIMD/scalar boundary and tail sizes.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn decode16_w6_matches_scalar_bit_for_bit() {
        let codec = BitplaneCodec;
        // n >= 34 so all 34 exponents appear (=> w=6); covers SIMD/scalar boundary + tails.
        for &n in &[34usize, 35, 47, 48, 64, 80, 96, 1000, 4096, 4099, 65536] {
            let bytes = make_bf16(34, n); // 34 exponents => w=6
            let meta = EncodeMeta { name: "w".into(), shape: vec![n as u64], dtype: "bf16".into() };
            let enc = codec.encode(&bytes, &meta).unwrap();
            assert_eq!(enc.data[15], 6, "n={n}: expected w=6 for 34 exponents");
            let p = enc.data[14] as usize;
            let mut off = 16;
            let palette = &enc.data[off..off + p];
            off += p;
            let idx_bytes = (n * 6 + 7) / 8;
            let idx_plane = &enc.data[off..off + idx_bytes];
            off += idx_bytes;
            let residuals = &enc.data[off..off + n];

            let scalar = codec
                .decode(&enc.data, &DecodeMeta { codec_id: "rtc-bitplane-v1".into(), uncompressed_size: 0 })
                .unwrap();
            let mut neon = vec![0u8; n * 2];
            unsafe { decode16_w6_into(palette, idx_plane, residuals, n, &mut neon) };
            assert_eq!(neon, scalar, "n={n}: w=6 16-wide must equal scalar bit-for-bit");
            assert_eq!(neon, bytes, "n={n}: w=6 decode must be lossless");
        }
    }

    // The width dispatcher must reproduce the scalar decode for both SIMD widths.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn decode_bitplane_row_dispatch_matches_scalar() {
        let codec = BitplaneCodec;
        for &(distinct, w) in &[(32usize, 5u8), (34usize, 6u8)] {
            for &n in &[64usize, 1000, 4099] {
                let bytes = make_bf16(distinct, n);
                let meta = EncodeMeta { name: "w".into(), shape: vec![n as u64], dtype: "bf16".into() };
                let enc = codec.encode(&bytes, &meta).unwrap();
                assert_eq!(enc.data[15], w, "distinct={distinct}: expected w={w}");
                let p = enc.data[14] as usize;
                let mut off = 16;
                let palette = &enc.data[off..off + p];
                off += p;
                let idx_bytes = (n * w as usize + 7) / 8;
                let idx_plane = &enc.data[off..off + idx_bytes];
                off += idx_bytes;
                let residuals = &enc.data[off..off + n];

                let mut dispatched = vec![0u8; n * 2];
                decode_bitplane_row_into(palette, idx_plane, residuals, n, w, &mut dispatched);
                assert_eq!(dispatched, bytes, "distinct={distinct} n={n}: dispatch must be lossless");
            }
        }
    }
