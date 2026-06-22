// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — see LICENSE. Unauthorized copying, use, or
// distribution of this file, via any medium, is strictly prohibited.

// R151 EXPERIMENT — Shannon-entropy floor of real bf16 weights.
//
// Before designing a lower-bit lossless codec we must know the THEORETICAL floor:
// how many bits/weight the real Gemma/Llama weights actually carry. This reads the
// raw-bf16 model and reports, per tensor:
//   - H(exponent)            order-0 entropy of the 8-bit exponent byte
//   - H(delta-exponent)      exponent delta-coded along each row (2D-structure test)
//   - H(residual byte)       order-0 entropy of sign+mantissa (what bit-plane stores RAW)
//   - per-bit entropy        sign + 7 mantissa bits (where is the entropy?)
// and compares the Shannon floor to: raw bf16 (16), current bit-plane (w+8), q8 (~8.5).
//
// Run: cargo test -p spissa-runtime --release --test weight_entropy_analysis -- --ignored --nocapture
//
// Uses only the public API (LazySpissaModel) on the raw-codec model (every tensor is
// raw bf16, so body projections are readable losslessly too, not just the embedding).

use spissa_runtime::LazySpissaModel;

const MODEL: &str = "../../models/gemma-3-1b-it-rawcodec.spsa";

fn shannon_bits(hist: &[u64]) -> f64 {
    let total: u64 = hist.iter().sum();
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

fn binary_entropy(ones: u64, total: u64) -> f64 {
    if ones == 0 || ones == total {
        return 0.0;
    }
    let p = ones as f64 / total as f64;
    -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
}

fn ceil_log2(n: usize) -> u32 {
    if n <= 1 {
        0
    } else {
        usize::BITS - (n - 1).leading_zeros()
    }
}

// Compute + print the entropy report for one tensor's bf16 bytes.
fn analyze(name: &str, bytes: &[u8], row_len: usize) {
    let n = bytes.len() / 2;
    let mut exp_hist = vec![0u64; 256];
    let mut res_hist = vec![0u64; 256];
    let mut delta_hist = vec![0u64; 256];
    let mut bit_ones = [0u64; 8]; // [0]=sign, [1..8]=mantissa bits 0..6

    let rows = n / row_len.max(1);
    for r in 0..rows {
        let mut prev_exp: Option<usize> = None;
        for c in 0..row_len {
            let i = r * row_len + c;
            let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
            let exp = ((bits >> 7) & 0xFF) as usize;
            let sign = ((bits >> 15) & 1) as u16;
            let mant = bits & 0x7F;
            let res = ((sign << 7) | mant) as usize;
            exp_hist[exp] += 1;
            res_hist[res] += 1;
            if sign == 1 {
                bit_ones[0] += 1;
            }
            for b in 0..7 {
                if (mant >> b) & 1 == 1 {
                    bit_ones[1 + b] += 1;
                }
            }
            if let Some(pe) = prev_exp {
                let d = (exp as i32 - pe as i32).rem_euclid(256) as usize;
                delta_hist[d] += 1;
            }
            prev_exp = Some(exp);
        }
    }

    let total = n as u64;
    let distinct_exp = exp_hist.iter().filter(|&&c| c > 0).count();
    let h_exp = shannon_bits(&exp_hist);
    let h_delta = shannon_bits(&delta_hist);
    let h_res = shannon_bits(&res_hist);
    let w = ceil_log2(distinct_exp);
    let bitplane_now = w as f64 + 8.0; // index width + raw residual byte
    let floor0 = h_exp + h_res; // order-0 Shannon floor (exp + residual byte)
    let floor_delta = h_delta + h_res; // with delta-coded exponent
    let h_sign = binary_entropy(bit_ones[0], total);
    let mant_bits: Vec<f64> = (1..8).map(|b| binary_entropy(bit_ones[b], total)).collect();
    let mant_sum: f64 = mant_bits.iter().sum();

    eprintln!("\n=== {name}  ({n} weights, row_len={row_len}) ===");
    eprintln!("  exponent: {distinct_exp} distinct  H(exp)={h_exp:.3} bits  (bit-plane fixed width w={w})");
    eprintln!("  H(delta-exp along row) = {h_delta:.3} bits   (2D-structure headroom)");
    eprintln!("  H(residual byte=sign+mantissa) = {h_res:.3} bits  (bit-plane stores this as 8 raw)");
    eprintln!(
        "  per-bit entropy: sign={h_sign:.3}  mantissa[m6..m0]=[{}]  (sum mantissa={mant_sum:.3})",
        mant_bits.iter().rev().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(", ")
    );
    eprintln!("  --- bits/weight comparison ---");
    eprintln!("  raw bf16:            16.000");
    eprintln!("  bit-plane (current): {bitplane_now:.3}");
    eprintln!("  Shannon floor (order-0, exp+resid):       {floor0:.3}");
    eprintln!("  Shannon floor (delta-exp + resid):        {floor_delta:.3}");
    eprintln!("  q8 (LOSSY reference, +fp16 scale/32):     ~8.50");
    eprintln!(
        "  => headroom below bit-plane: {:.3} bits ({:.0}% smaller); gap to q8: {:+.3} bits",
        bitplane_now - floor0,
        (1.0 - floor0 / bitplane_now) * 100.0,
        floor0 - 8.5
    );
}

// Joint entropy H(X,Y) over a 256x256 histogram (bits).
fn joint_entropy(joint: &[u64; 65536]) -> f64 {
    let total: u64 = joint.iter().sum();
    if total == 0 {
        return 0.0;
    }
    joint
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total as f64;
            -p * p.log2()
        })
        .sum()
}

// R155 question: is the ~10.5-bit floor lowerable by CONTEXT (cross-weight
// correlation)? Measure order-1 conditional entropy H(X | X_prev) and the mutual
// information between exponent and residual. If conditioning doesn't reduce entropy,
// 10.5 bits is the true lossless floor and the 1.51x streaming ceiling is hard.
fn conditional_analysis(name: &str, bytes: &[u8], row_len: usize) {
    let n = bytes.len() / 2;
    let mut exp_hist = vec![0u64; 256];
    let mut res_hist = vec![0u64; 256];
    let mut exp_joint = vec![0u64; 65536]; // (exp_prev, exp)
    let mut res_joint = vec![0u64; 65536]; // (res_prev, res)
    let mut er_joint = vec![0u64; 65536]; // (exp, res) same weight
    let rows = n / row_len.max(1);
    for r in 0..rows {
        let (mut pe, mut pr): (Option<usize>, Option<usize>) = (None, None);
        for c in 0..row_len {
            let i = r * row_len + c;
            let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
            let e = ((bits >> 7) & 0xFF) as usize;
            let sign = ((bits >> 15) & 1) as usize;
            let res = (sign << 7) | (bits & 0x7F) as usize;
            exp_hist[e] += 1;
            res_hist[res] += 1;
            er_joint[e * 256 + res] += 1;
            if let Some(p) = pe {
                exp_joint[p * 256 + e] += 1;
            }
            if let Some(p) = pr {
                res_joint[p * 256 + res] += 1;
            }
            pe = Some(e);
            pr = Some(res);
        }
    }
    let h_exp = shannon_bits(&exp_hist.iter().map(|&c| c).collect::<Vec<_>>());
    let h_res = shannon_bits(&res_hist.iter().map(|&c| c).collect::<Vec<_>>());
    let exp_joint: [u64; 65536] = exp_joint.try_into().unwrap();
    let res_joint: [u64; 65536] = res_joint.try_into().unwrap();
    let er_joint: [u64; 65536] = er_joint.try_into().unwrap();
    // H(X|Xprev) = H(Xprev,X) - H(Xprev) ~= joint - marginal (marginal ~= H(X) here).
    let h_exp_cond = joint_entropy(&exp_joint) - h_exp;
    let h_res_cond = joint_entropy(&res_joint) - h_res;
    // Mutual info I(exp;res) = H(exp)+H(res) - H(exp,res).
    let mi_er = h_exp + h_res - joint_entropy(&er_joint);

    eprintln!("\n=== {name}  context/conditional entropy ===");
    eprintln!("  H(exp)            = {h_exp:.3}   H(exp | exp_prev) = {h_exp_cond:.3}  (gain {:.3})", h_exp - h_exp_cond);
    eprintln!("  H(residual)       = {h_res:.3}   H(res | res_prev) = {h_res_cond:.3}  (gain {:.3})", h_res - h_res_cond);
    eprintln!("  I(exp ; residual) = {mi_er:.4} bits  (mutual info; >0 means joint-coding could save)");
    eprintln!(
        "  => order-1 floor ≈ {:.3} bits/weight (order-0 was {:.3}); ceiling {} move",
        h_exp_cond + h_res_cond,
        h_exp + h_res,
        if (h_exp - h_exp_cond) + (h_res - h_res_cond) + mi_er > 0.1 { "COULD" } else { "does NOT" }
    );
}

#[test]
#[ignore]
fn weight_context_entropy_real_weights() {
    let mut m = LazySpissaModel::open(MODEL).unwrap();
    let names: Vec<String> = m.tensor_names().iter().map(|s| s.to_string()).collect();
    let targets: Vec<String> = ["embed_tokens.weight", "layers.0.mlp.gate_proj.weight"]
        .iter()
        .filter_map(|t| names.iter().find(|n| n.contains(t)).cloned())
        .collect();
    eprintln!("\n########## R155 context-entropy (can the 10.5-bit floor be lowered?) ##########");
    for name in &targets {
        let meta = m.tensor(name).unwrap().clone();
        let row_len = *meta.shape.last().unwrap() as usize;
        m.with_raw_tensor(meta.tensor_id, |b| {
            conditional_analysis(name, b, row_len);
            Ok(())
        })
        .unwrap();
    }
    eprintln!("\n########## end ##########\n");
}

#[test]
#[ignore]
fn weight_entropy_floor_real_weights() {
    let mut m = LazySpissaModel::open(MODEL)
        .unwrap_or_else(|e| panic!("open {MODEL}: {e} (need the raw-codec model)"));

    // Pick representative tensors: the tied embedding/lm-head + a few 2D body
    // projections (attention + MLP), whichever names exist.
    let names: Vec<String> = m.tensor_names().iter().map(|s| s.to_string()).collect();
    let pick = |needle: &str| -> Option<String> {
        names.iter().find(|n| n.contains(needle)).cloned()
    };
    let targets: Vec<String> = [
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.down_proj.weight",
    ]
    .iter()
    .filter_map(|t| pick(t))
    .collect();
    assert!(!targets.is_empty(), "no target tensors found; names sample: {:?}", &names[..names.len().min(8)]);

    eprintln!("\n########## R151 weight-entropy floor (model: {MODEL}) ##########");
    for name in &targets {
        let meta = m.tensor(name).unwrap().clone();
        let row_len = *meta.shape.last().unwrap() as usize;
        let id = meta.tensor_id;
        let got = m
            .with_raw_tensor(id, |bytes| {
                analyze(name, bytes, row_len);
                Ok(())
            })
            .unwrap();
        if got.is_none() {
            eprintln!("  (skip {name}: not raw-bf16 readable)");
        }
    }
    eprintln!("\n########## end ##########\n");
}
