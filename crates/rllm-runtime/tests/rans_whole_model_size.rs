// R156b — exact WHOLE-MODEL rANS compressed size on real weights.
//
// R156a proved rANS streaming is lossless on body projections too, so it applies to
// every weight tensor. This measures the real total: for each tensor, split bf16 into
// (exponent, residual), rANS-encode the exponent (4-lane interleaved) + raw residual +
// freq table, and sum. Answers "how big is the whole model, losslessly?" exactly.
//
// Run: cargo test -p rllm-runtime --release --test rans_whole_model_size -- --ignored --nocapture

use rllm_runtime::LazyRllmModel;

const MODEL: &str = "../../models/gemma-3-1b-it-rawcodec.rllm";

// rANS-compressed size of one bf16 tensor: interleaved exponent stream + raw residual
// + freq table (512 B) + 4 lane-length words. Returns (raw_bytes, rans_bytes).
fn rans_size(bytes: &[u8]) -> (usize, usize) {
    let n = bytes.len() / 2;
    if n == 0 {
        return (bytes.len(), bytes.len());
    }
    let mut exp = vec![0u8; n];
    let mut res = vec![0u8; n];
    for i in 0..n {
        let (e, r) = rtc_codec::split_bf16(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]));
        exp[i] = e;
        res[i] = r;
    }
    let freq = rtc_codec::normalize_freqs(&rtc_codec::count_symbols(&exp));
    let lanes = rtc_codec::rans_encode_interleaved4(&exp, &freq);
    let exp_bytes: usize = lanes.iter().map(|l| l.len()).sum::<usize>() + 16;
    let rans = exp_bytes + res.len() + 512; // streams + lane lens + freq table
    (bytes.len(), rans)
}

#[test]
#[ignore]
fn whole_model_rans_size() {
    let mut m = LazyRllmModel::open(MODEL).unwrap_or_else(|e| panic!("open {MODEL}: {e}"));
    // Snapshot (id, name, even-length?) so we don't borrow m while calling with_raw_tensor.
    let ids: Vec<(u64, String)> =
        m.tensors().map(|t| (t.tensor_id, t.name.clone())).collect();

    let (mut raw_total, mut rans_total, mut n_tensors, mut n_weights) = (0usize, 0usize, 0usize, 0u64);
    let mut biggest: Vec<(String, usize, usize)> = Vec::new();
    for (id, name) in &ids {
        let got = m
            .with_raw_tensor(*id, |bytes| {
                if bytes.len() % 2 != 0 {
                    return Ok((bytes.len(), bytes.len(), 0u64)); // not bf16-shaped; keep raw
                }
                let (raw, rans) = rans_size(bytes);
                Ok((raw, rans, (bytes.len() / 2) as u64))
            })
            .unwrap();
        if let Some((raw, rans, w)) = got {
            raw_total += raw;
            rans_total += rans;
            n_weights += w;
            n_tensors += 1;
            biggest.push((name.clone(), raw, rans));
        }
    }
    biggest.sort_by_key(|t| std::cmp::Reverse(t.1));

    let raw_gb = raw_total as f64 / 1e9;
    let rans_gb = rans_total as f64 / 1e9;
    let bits_per_weight = if n_weights > 0 { rans_total as f64 * 8.0 / n_weights as f64 } else { 0.0 };
    eprintln!("\n=== R156b WHOLE-MODEL rANS size ({MODEL}) ===");
    eprintln!("  tensors: {n_tensors}   weights: {n_weights}");
    eprintln!("  raw bf16 total:  {raw_gb:.3} GB", );
    eprintln!("  rANS total:      {rans_gb:.3} GB   ({:.0}% smaller, {bits_per_weight:.2} bits/weight)",
        (1.0 - rans_total as f64 / raw_total as f64) * 100.0);
    eprintln!("  top tensors (raw -> rANS):");
    for (name, raw, rans) in biggest.iter().take(6) {
        eprintln!("    {:>9.1} MB -> {:>6.1} MB  {name}", *raw as f64 / 1e6, *rans as f64 / 1e6);
    }
    eprintln!();
}
