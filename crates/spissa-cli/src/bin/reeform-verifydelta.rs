// Copyright (c) 2026 Rama Erik Esprada. All Rights Reserved.
// Proprietary and confidential — research/test instrument (REEFORM).
//
// Proves a delta-`.spsa` reconstructs BIT-EXACT through the real runtime loader (the same path
// `spissa chat` uses): open the delta `.spsa` (the loader opens its base + rebuilds W_ft = W_base
// + Δ on decode), decode every tensor, and compare bytes to the original fine-tune safetensors.
//
//   reeform-verifydelta <delta.spsa> <original-fine-tune.safetensors>

use anyhow::Result;
use spissa_import::SafetensorsReader;
use spissa_runtime::LazySpissaModel;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (delta_spsa, original) = (&a[1], &a[2]);
    let mut model = LazySpissaModel::open(delta_spsa)
        .map_err(|e| anyhow::anyhow!("open delta .spsa: {e}"))?;
    let mut orig = SafetensorsReader::open(original)?;
    let orig_names: std::collections::HashSet<String> =
        orig.list_tensors().iter().map(|s| s.to_string()).collect();

    let names: Vec<String> = model
        .tensor_names()
        .iter()
        .filter(|n| !n.starts_with("__"))
        .map(|s| s.to_string())
        .collect();
    let (mut checked, mut exact, mut missing, mut mismatch) = (0u64, 0u64, 0u64, 0u64);
    let mut bad: Vec<String> = Vec::new();
    for name in &names {
        if !orig_names.contains(name) {
            missing += 1;
            continue;
        }
        let got = model
            .decode_tensor_raw_bytes(name)
            .map_err(|e| anyhow::anyhow!("decode {name}: {e}"))?;
        let want = orig.read_tensor(name)?;
        checked += 1;
        if got == want {
            exact += 1;
        } else {
            mismatch += 1;
            if bad.len() < 5 {
                bad.push(name.clone());
            }
        }
    }
    println!("delta tensors (excl. __internal__): {}", names.len());
    println!("compared: {checked} | BIT-EXACT: {exact} | mismatch: {mismatch} | not-in-original: {missing}");
    if !bad.is_empty() {
        println!("first mismatches: {bad:?}");
    }
    println!(
        "{}",
        if mismatch == 0 && checked > 0 {
            "✅ DELTA .spsa RECONSTRUCTS BIT-EXACT THROUGH THE LOADER"
        } else {
            "❌ NOT bit-exact"
        }
    );
    Ok(())
}
