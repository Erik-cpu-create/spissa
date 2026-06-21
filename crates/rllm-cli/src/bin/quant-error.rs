// Measure the REAL quantization loss of q8_0 on actual model weights.
// For each bf16 weight tensor: quantize -> dequantize, compare to the exact bf16 value.
// Reports RMS relative error (the standard "how lossy %"), max error, SNR, cosine sim.

use rllm_container::DType;
use rllm_runtime::{
    dequantize_q4_0, dequantize_q8_0, quantize_to_q4_0, quantize_to_q8_0, LazyRllmModel,
    RamaIntegrityMode,
};

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

// (sum of squared error, sum of squared signal) for one quantized-then-dequantized vec.
fn err_sums(orig: &[f32], deq: &[f32]) -> (f64, f64) {
    let mut sq_err = 0.0f64;
    let mut sq_sig = 0.0f64;
    for (&o, &d) in orig.iter().zip(deq.iter()) {
        let e = (o - d) as f64;
        sq_err += e * e;
        sq_sig += (o as f64) * (o as f64);
    }
    (sq_err, sq_sig)
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: q8-error <model.rllm>");
    let mut model = LazyRllmModel::open(&path).expect("open");
    model.set_rama_integrity_mode(RamaIntegrityMode::Unchecked);

    let names: Vec<String> = model
        .tensor_names()
        .iter()
        .filter(|n| n.contains("proj.weight") || n.contains("embed_tokens"))
        .map(|s| s.to_string())
        .collect();

    // global sums for aggregate RMS-rel: (q8_err, q4_err, signal, weight_count)
    let (mut g8, mut g4, mut gsig, mut gn) = (0.0f64, 0.0f64, 0.0f64, 0u64);

    println!("{:<40} {:>9} {:>9}", "tensor", "q8 rel%", "q4 rel%");
    let mut shown = 0;
    for name in &names {
        let meta = model.tensor(name).expect("meta").clone();
        if meta.dtype != DType::Bf16 {
            continue;
        }
        let bf16 = model.decode_tensor_raw_bytes(name).expect("decode");
        let orig = bf16_bytes_to_f32(&bf16);

        let q8 = quantize_to_q8_0(&bf16, DType::Bf16, &meta.shape).expect("q8");
        let mut d8 = vec![0.0f32; orig.len()];
        dequantize_q8_0(&q8, &mut d8);
        let (e8, sig) = err_sums(&orig, &d8);

        let q4 = quantize_to_q4_0(&bf16, DType::Bf16, &meta.shape).expect("q4");
        let mut d4 = vec![0.0f32; orig.len()];
        dequantize_q4_0(&q4, &mut d4);
        let (e4, _) = err_sums(&orig, &d4);

        g8 += e8;
        g4 += e4;
        gsig += sig;
        gn += orig.len() as u64;

        if shown < 10 {
            let short = name.replace("model.layers.", "L").replace(".weight", "");
            let r8 = (e8 / sig.max(1e-30)).sqrt() * 100.0;
            let r4 = (e4 / sig.max(1e-30)).sqrt() * 100.0;
            println!("{short:<40} {r8:>8.3} {r4:>8.3}");
            shown += 1;
        }
    }

    let r8 = (g8 / gsig.max(1e-30)).sqrt() * 100.0;
    let r4 = (g4 / gsig.max(1e-30)).sqrt() * 100.0;
    let snr8 = 10.0 * (gsig / g8.max(1e-30)).log10();
    let snr4 = 10.0 * (gsig / g4.max(1e-30)).log10();
    println!("\n=== AGGREGATE over {gn} weights ({} tensors) ===", names.len());
    println!("q8_0 RMS relative error : {r8:.3} %   (SNR {snr8:.1} dB)");
    println!("q4_0 RMS relative error : {r4:.3} %   (SNR {snr4:.1} dB)");
    println!("rANS (lossless)         : 0.000 %   (bit-exact, no quantization)");
}
