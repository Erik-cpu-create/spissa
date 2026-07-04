// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

// R152 SCOUT — rANS exponent codec on REAL bf16 weights.
//
// R151 showed the lossless floor is ~10.5 bits/weight and that bit-plane wastes
// ~3 bits paying a fixed-width index for the ~2.6-bit exponent. This scout encodes
// the REAL exponent plane with static rANS and answers two questions:
//   (1) RATIO  — does rANS reach the ~2.6-bit exponent entropy (=> ~10.5 total)?
//   (2) SPEED  — is scalar rANS decode fast enough to HIDE under the cold-read
//                bandwidth when parallelized (R150a)? Need ~1.3 Gweight/s aggregate
//                (cold read ~1.7 GB/s / ~1.31 B/weight), i.e. ~0.22 Gw/s/core on 6 cores.
//
// Run: cargo test -p spissa-runtime --release --test rans_exponent_scout -- --ignored --nocapture

use spissa_runtime::LazySpissaModel;

const MODEL: &str = "../../models/gemma-3-1b-it-rawcodec.spsa";

fn shannon_bits(hist: &[u32]) -> f64 {
    let total: u64 = hist.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return 0.0;
    }
    hist.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total as f64;
            -p * p.log2()
        })
        .sum()
}

fn ceil_log2(n: usize) -> u32 {
    if n <= 1 {
        0
    } else {
        usize::BITS - (n - 1).leading_zeros()
    }
}

// Extract the exponent byte of every bf16 weight.
fn exponents(bytes: &[u8]) -> Vec<u8> {
    let n = bytes.len() / 2;
    let mut exps = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
        exps.push(((bits >> 7) & 0xFF) as u8);
    }
    exps
}

fn scout_tensor(name: &str, bytes: &[u8]) {
    let exps = exponents(bytes);
    let n = exps.len();
    let counts = rtc_codec::count_symbols(&exps);
    let distinct = counts.iter().filter(|&&c| c > 0).count();
    let h_exp = shannon_bits(&counts);
    let w = ceil_log2(distinct);

    let freq = rtc_codec::normalize_freqs(&counts);
    let stream = rtc_codec::rans_encode(&exps, &freq);

    // Correctness: decode == original exponents.
    let decoded = rtc_codec::rans_decode(&stream, n, &freq);
    assert_eq!(decoded, exps, "{name}: rANS decode must be bit-exact");

    let rans_bits = stream.len() as f64 * 8.0 / n as f64;
    let table_bytes = distinct * 5; // ~ store (symbol u8 + freq u32) per used symbol
    let table_bits = table_bytes as f64 * 8.0 / n as f64;
    let rans_total = rans_bits + table_bits + 8.0; // exponent stream + table + residual byte
    let bitplane_total = w as f64 + 8.0;
    let floor = h_exp + 7.95; // R151 residual ~7.95

    // Decode throughput (scalar single-stream vs 4-lane interleaved), single-core.
    let iters = 5;
    let time_dec = |f: &dyn Fn()| {
        f(); // warm
        let t = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        t.elapsed().as_secs_f64() / iters as f64
    };
    let scalar_s = time_dec(&|| {
        std::hint::black_box(rtc_codec::rans_decode(&stream, n, &freq));
    });
    let streams4 = rtc_codec::rans_encode_interleaved4(&exps, &freq);
    let sl4 = [
        &streams4[0][..],
        &streams4[1][..],
        &streams4[2][..],
        &streams4[3][..],
    ];
    assert_eq!(
        rtc_codec::rans_decode_interleaved4(sl4, n, &freq),
        exps,
        "{name}: interleaved decode bit-exact"
    );
    let inter_s = time_dec(&|| {
        std::hint::black_box(rtc_codec::rans_decode_interleaved4(sl4, n, &freq));
    });
    let scalar_gw = (n as f64 / 1e9) / scalar_s;
    let gw_per_core = (n as f64 / 1e9) / inter_s; // interleaved is the real candidate

    // Hide-under-cold-read check: need aggregate decode >= read rate.
    let cores = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(6);
    let read_gbps = 1.7; // R150a cold-read aggregate
    let bytes_per_weight = rans_total / 8.0;
    let read_gw = read_gbps / bytes_per_weight; // Gweight/s the read delivers
    let agg_decode_gw = gw_per_core * cores as f64;
    let hides = agg_decode_gw >= read_gw;

    eprintln!("\n=== {name}  ({n} weights, exponent distinct={distinct}) ===");
    eprintln!("  RATIO:");
    eprintln!("    H(exponent)            = {h_exp:.3} bits   (floor for the exponent)");
    eprintln!("    rANS exponent stream   = {rans_bits:.3} bits  (+table {table_bits:.4})");
    eprintln!("    bit-plane index (w={w}) = {w}.000 bits");
    eprintln!("    => lossless TOTAL: rANS {rans_total:.3} vs bit-plane {bitplane_total:.3} vs floor {floor:.3}");
    eprintln!(
        "    => {:.2} bits/weight smaller than bit-plane ({:.0}%), {:+.3} vs floor",
        bitplane_total - rans_total,
        (1.0 - rans_total / bitplane_total) * 100.0,
        rans_total - floor
    );
    eprintln!("  SPEED (single-core):");
    eprintln!("    rANS scalar       = {scalar_gw:.3} Gweight/s/core");
    eprintln!(
        "    rANS interleaved4 = {gw_per_core:.3} Gweight/s/core  ({:.2}x ILP, {:.1} ms)",
        gw_per_core / scalar_gw,
        inter_s * 1000.0
    );
    eprintln!(
        "    cold read delivers ~{read_gw:.3} Gw/s (@{read_gbps} GB/s, {bytes_per_weight:.2} B/weight); aggregate decode {cores}-core = {agg_decode_gw:.2} Gw/s",
    );
    eprintln!(
        "    => {} (interleaved decode {} hide under cold read; margin {:.2}x)",
        if hides { "GO" } else { "RISK" },
        if hides { "DOES" } else { "does NOT" },
        agg_decode_gw / read_gw
    );
}

#[test]
#[ignore]
fn rans_exponent_scout_real_weights() {
    let mut m = LazySpissaModel::open(MODEL)
        .unwrap_or_else(|e| panic!("open {MODEL}: {e} (need the raw-codec model)"));
    let names: Vec<String> = m.tensor_names().iter().map(|s| s.to_string()).collect();
    let targets: Vec<String> = ["embed_tokens.weight", "layers.0.mlp.gate_proj.weight"]
        .iter()
        .filter_map(|t| names.iter().find(|n| n.contains(t)).cloned())
        .collect();
    assert!(!targets.is_empty(), "no target tensors found");

    eprintln!("\n########## R152 rANS exponent scout (model: {MODEL}) ##########");
    for name in &targets {
        let meta = m.tensor(name).unwrap().clone();
        m.with_raw_tensor(meta.tensor_id, |bytes| {
            scout_tensor(name, bytes);
            Ok(())
        })
        .unwrap();
    }
    eprintln!("\n########## end ##########\n");
}
