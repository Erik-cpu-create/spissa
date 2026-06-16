use serde::Serialize;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::time::Instant;

pub const REE_KERNEL_NAME: &str = "REEDOT-LAB";

#[derive(Debug, Clone, Copy)]
pub struct Q8KernelBenchConfig {
    pub batch: usize,
    pub in_features: usize,
    pub blocks_per_row: usize,
    pub out_features: usize,
    pub iters: usize,
}

impl Default for Q8KernelBenchConfig {
    fn default() -> Self {
        Self {
            batch: 55,
            in_features: 2048,
            blocks_per_row: 64,
            out_features: 8192,
            iters: 2000,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Q8KernelBenchResult {
    pub variant: String,
    pub elapsed_ns: u128,
    pub checksum: f32,
    pub max_abs_diff: f32,
    pub speedup_vs_baseline: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Q8KernelBenchReport {
    pub ree_kernel: String,
    pub batch: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub iters: usize,
    pub results: Vec<Q8KernelBenchResult>,
}

pub fn run_suite(config: Q8KernelBenchConfig) -> Q8KernelBenchReport {
    assert!(config.batch > 0);
    assert!(config.iters > 0);
    assert_eq!(config.in_features % 32, 0);
    assert_eq!(config.blocks_per_row, config.in_features / 32);

    let input = deterministic_input(config.batch, config.in_features);
    let q8 = deterministic_q8_blocks(config.blocks_per_row);
    let scale = 0.125f32;
    #[cfg(target_arch = "aarch64")]
    let prescaled_sidecar = prescaled_sidecar_blocks(&q8, scale);

    let (baseline_ns, baseline_output) = time_variant(config.iters, config.batch, || {
        baseline_i8_dot32_batch4(&q8, scale, &input, config.batch, config.in_features)
    });
    let baseline_checksum = checksum(&baseline_output);

    let mut results = Vec::new();
    results.push(Q8KernelBenchResult {
        variant: "baseline_i8_dot32_batch4".to_string(),
        elapsed_ns: baseline_ns,
        checksum: baseline_checksum,
        max_abs_diff: 0.0,
        speedup_vs_baseline: 1.0,
    });

    for (variant, elapsed_ns, output) in [
        {
            let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
                scaled_f32_dot32_batch4(&q8, scale, &input, config.batch, config.in_features)
            });
            ("scaled_f32_dot32_batch4", elapsed_ns, output)
        },
        {
            let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
                scaled_f32_dot32_batch4_runtime(
                    &q8,
                    scale,
                    &input,
                    config.batch,
                    config.in_features,
                )
            });
            ("scaled_f32_dot32_batch4_runtime", elapsed_ns, output)
        },
        {
            let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
                reelane_f32_dot32_batch4(&q8, scale, &input, config.batch, config.in_features)
            });
            ("reelane_f32_dot32_batch4", elapsed_ns, output)
        },
        {
            let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
                reflow_i8_scaled_batch4(&q8, scale, &input, config.batch, config.in_features)
            });
            ("reeflow_i8_scaled_batch4", elapsed_ns, output)
        },
        {
            let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
                unrolled_i8_dot32_batch4(&q8, scale, &input, config.batch, config.in_features)
            });
            ("unrolled_i8_dot32_batch4", elapsed_ns, output)
        },
    ] {
        results.push(Q8KernelBenchResult {
            variant: variant.to_string(),
            elapsed_ns,
            checksum: checksum(&output),
            max_abs_diff: max_abs_diff(&baseline_output, &output),
            speedup_vs_baseline: baseline_ns as f64 / elapsed_ns.max(1) as f64,
        });
    }

    #[cfg(target_arch = "aarch64")]
    {
        let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
            reevec_neon_f32_dot32_batch4(&q8, scale, &input, config.batch, config.in_features)
        });
        results.push(Q8KernelBenchResult {
            variant: "reevec_neon_f32_dot32_batch4".to_string(),
            elapsed_ns,
            checksum: checksum(&output),
            max_abs_diff: max_abs_diff(&baseline_output, &output),
            speedup_vs_baseline: baseline_ns as f64 / elapsed_ns.max(1) as f64,
        });

        let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
            reecast_neon_scale_batch4(&q8, scale, &input, config.batch, config.in_features)
        });
        results.push(Q8KernelBenchResult {
            variant: "reecast_neon_scale_batch4".to_string(),
            elapsed_ns,
            checksum: checksum(&output),
            max_abs_diff: max_abs_diff(&baseline_output, &output),
            speedup_vs_baseline: baseline_ns as f64 / elapsed_ns.max(1) as f64,
        });

        let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
            reewide_neon_f32_dot32_batch8(&q8, scale, &input, config.batch, config.in_features)
        });
        results.push(Q8KernelBenchResult {
            variant: "reewide_neon_f32_dot32_batch8".to_string(),
            elapsed_ns,
            checksum: checksum(&output),
            max_abs_diff: max_abs_diff(&baseline_output, &output),
            speedup_vs_baseline: baseline_ns as f64 / elapsed_ns.max(1) as f64,
        });

        let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
            reeduo_neon_block64_batch4(&q8, scale, &input, config.batch, config.in_features)
        });
        results.push(Q8KernelBenchResult {
            variant: "reeduo_neon_block64_batch4".to_string(),
            elapsed_ns,
            checksum: checksum(&output),
            max_abs_diff: max_abs_diff(&baseline_output, &output),
            speedup_vs_baseline: baseline_ns as f64 / elapsed_ns.max(1) as f64,
        });

        let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
            reeside_prescaled_f32_batch4(
                &prescaled_sidecar,
                &input,
                config.batch,
                config.in_features,
            )
        });
        results.push(Q8KernelBenchResult {
            variant: "reeside_prescaled_f32_batch4".to_string(),
            elapsed_ns,
            checksum: checksum(&output),
            max_abs_diff: max_abs_diff(&baseline_output, &output),
            speedup_vs_baseline: baseline_ns as f64 / elapsed_ns.max(1) as f64,
        });

        let (elapsed_ns, output) = time_variant(config.iters, config.batch, || {
            reetail_neon_tail3_batch4(&q8, scale, &input, config.batch, config.in_features)
        });
        results.push(Q8KernelBenchResult {
            variant: "reetail_neon_tail3_batch4".to_string(),
            elapsed_ns,
            checksum: checksum(&output),
            max_abs_diff: max_abs_diff(&baseline_output, &output),
            speedup_vs_baseline: baseline_ns as f64 / elapsed_ns.max(1) as f64,
        });
    }

    if config.batch == 1 {
        let (baseline_batch1_ns, baseline_batch1_output) = time_variant(config.iters, 1, || {
            baseline_i8_dot32_batch1_row(&q8, scale, &input, config.in_features)
        });

        results.push(Q8KernelBenchResult {
            variant: "baseline_i8_dot32_batch1_row".to_string(),
            elapsed_ns: baseline_batch1_ns,
            checksum: checksum(&baseline_batch1_output),
            max_abs_diff: 0.0,
            speedup_vs_baseline: 1.0,
        });

        let (scaled_batch1_ns, scaled_batch1_output) = time_variant(config.iters, 1, || {
            scaled_f32_dot32_batch1_row(&q8, scale, &input, config.in_features)
        });

        results.push(Q8KernelBenchResult {
            variant: "scaled_f32_dot32_batch1_row".to_string(),
            elapsed_ns: scaled_batch1_ns,
            checksum: checksum(&scaled_batch1_output),
            max_abs_diff: max_abs_diff(&baseline_batch1_output, &scaled_batch1_output),
            speedup_vs_baseline: baseline_batch1_ns as f64 / scaled_batch1_ns.max(1) as f64,
        });
    }

    Q8KernelBenchReport {
        ree_kernel: REE_KERNEL_NAME.to_string(),
        batch: config.batch,
        in_features: config.in_features,
        out_features: config.out_features,
        iters: config.iters,
        results,
    }
}

fn deterministic_input(batch: usize, in_features: usize) -> Vec<f32> {
    (0..batch * in_features)
        .map(|idx| (idx as f32 % 97.0) * 0.00390625 - 0.1875)
        .collect()
}

fn deterministic_q8_blocks(blocks_per_row: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(blocks_per_row * 34);
    for block in 0..blocks_per_row {
        bytes.extend_from_slice(&crate::tensor::f32_to_fp16(0.125).to_le_bytes());
        for idx in 0..32 {
            let q = (((block * 7 + idx * 3) as i16 % 17) - 8) as i8;
            bytes.push(q as u8);
        }
    }
    bytes
}

fn time_variant(
    iters: usize,
    output_len: usize,
    mut f: impl FnMut() -> Vec<f32>,
) -> (u128, Vec<f32>) {
    let warmup = f();
    assert_eq!(warmup.len(), output_len);
    let started = Instant::now();
    let mut output = warmup;
    for _ in 0..iters {
        output = f();
        std::hint::black_box(&output);
    }
    (started.elapsed().as_nanos(), output)
}

pub fn baseline_i8_dot32_batch4(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    for block in 0..blocks {
        let offset = block * 34;
        let qs = &q8[offset + 2..offset + 34];
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            for lane in 0..4 {
                output[batch_idx + lane] +=
                    scale * dot_i8_f32(qs, &input[(batch_idx + lane) * in_features + in_feature..]);
            }
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] +=
                scale * dot_i8_f32(qs, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}

pub fn scaled_f32_dot32_batch4(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    for block in 0..blocks {
        let offset = block * 34;
        let scaled = scaled_block(&q8[offset + 2..offset + 34], scale);
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            accumulate_scaled_batch4(
                &scaled,
                &input[batch_idx * in_features + in_feature..],
                in_features,
                &mut output,
                batch_idx,
            );
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] +=
                dot_f32_32(&scaled, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}

pub fn scaled_f32_dot32_batch4_runtime(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    for block in 0..blocks {
        let offset = block * 34;
        let scaled = scaled_block(&q8[offset + 2..offset + 34], scale);
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            accumulate_scaled_batch4_runtime(
                &scaled,
                &input[batch_idx * in_features + in_feature..],
                in_features,
                &mut output,
                batch_idx,
            );
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] +=
                dot_f32_32(&scaled, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}

pub fn reelane_f32_dot32_batch4(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    for block in 0..blocks {
        let offset = block * 34;
        let scaled = scaled_block(&q8[offset + 2..offset + 34], scale);
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            accumulate_reelane_scaled_batch4(
                &scaled,
                &input[batch_idx * in_features + in_feature..],
                in_features,
                &mut output,
                batch_idx,
            );
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] +=
                dot_f32_32(&scaled, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}

#[cfg(target_arch = "aarch64")]
pub fn reevec_neon_f32_dot32_batch4(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    for block in 0..blocks {
        let offset = block * 34;
        let scaled = scaled_block(&q8[offset + 2..offset + 34], scale);
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            unsafe {
                accumulate_neon_scaled_batch4(
                    &scaled,
                    &input[batch_idx * in_features + in_feature..],
                    in_features,
                    &mut output,
                    batch_idx,
                );
            }
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] +=
                dot_f32_32(&scaled, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}

#[cfg(target_arch = "aarch64")]
pub fn reecast_neon_scale_batch4(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    for block in 0..blocks {
        let offset = block * 34;
        let scaled = unsafe { scaled_block_neon(&q8[offset + 2..offset + 34], scale) };
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            unsafe {
                accumulate_neon_scaled_batch4(
                    &scaled,
                    &input[batch_idx * in_features + in_feature..],
                    in_features,
                    &mut output,
                    batch_idx,
                );
            }
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] +=
                dot_f32_32(&scaled, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}

#[cfg(target_arch = "aarch64")]
pub fn reetail_neon_tail3_batch4(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    for block in 0..blocks {
        let offset = block * 34;
        let scaled = unsafe { scaled_block_neon(&q8[offset + 2..offset + 34], scale) };
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            unsafe {
                accumulate_neon_scaled_batch4(
                    &scaled,
                    &input[batch_idx * in_features + in_feature..],
                    in_features,
                    &mut output,
                    batch_idx,
                );
            }
            batch_idx += 4;
        }
        if batch - batch_idx == 3 {
            unsafe {
                accumulate_neon_scaled_tail3(
                    &scaled,
                    &input[batch_idx * in_features + in_feature..],
                    in_features,
                    &mut output,
                    batch_idx,
                );
            }
        } else {
            while batch_idx < batch {
                output[batch_idx] +=
                    dot_f32_32(&scaled, &input[batch_idx * in_features + in_feature..]);
                batch_idx += 1;
            }
        }
    }
    output
}

#[cfg(target_arch = "aarch64")]
pub fn reewide_neon_f32_dot32_batch8(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    for block in 0..blocks {
        let offset = block * 34;
        let scaled = unsafe { scaled_block_neon(&q8[offset + 2..offset + 34], scale) };
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 8 <= batch {
            unsafe {
                accumulate_neon_scaled_batch8(
                    &scaled,
                    &input[batch_idx * in_features + in_feature..],
                    in_features,
                    &mut output,
                    batch_idx,
                );
            }
            batch_idx += 8;
        }
        while batch_idx + 4 <= batch {
            unsafe {
                accumulate_neon_scaled_batch4(
                    &scaled,
                    &input[batch_idx * in_features + in_feature..],
                    in_features,
                    &mut output,
                    batch_idx,
                );
            }
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] +=
                dot_f32_32(&scaled, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}

#[cfg(target_arch = "aarch64")]
pub fn reeduo_neon_block64_batch4(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    let mut block = 0usize;
    while block + 1 < blocks {
        let first_offset = block * 34;
        let second_offset = first_offset + 34;
        let scaled = unsafe {
            scaled_pair_block_neon(
                &q8[first_offset + 2..first_offset + 34],
                &q8[second_offset + 2..second_offset + 34],
                scale,
            )
        };
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            unsafe {
                accumulate_neon_scaled64_batch4(
                    &scaled,
                    &input[batch_idx * in_features + in_feature..],
                    in_features,
                    &mut output,
                    batch_idx,
                );
            }
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] += dot_f32(&scaled, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
        block += 2;
    }
    while block < blocks {
        let offset = block * 34;
        let scaled = unsafe { scaled_block_neon(&q8[offset + 2..offset + 34], scale) };
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            unsafe {
                accumulate_neon_scaled_batch4(
                    &scaled,
                    &input[batch_idx * in_features + in_feature..],
                    in_features,
                    &mut output,
                    batch_idx,
                );
            }
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] +=
                dot_f32_32(&scaled, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
        block += 1;
    }
    output
}

#[cfg(target_arch = "aarch64")]
pub fn reeside_prescaled_f32_batch4(
    sidecar: &[[f32; 32]],
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    for (block, scaled) in sidecar.iter().enumerate() {
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            unsafe {
                accumulate_neon_scaled_batch4(
                    scaled,
                    &input[batch_idx * in_features + in_feature..],
                    in_features,
                    &mut output,
                    batch_idx,
                );
            }
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] += dot_f32_32(scaled, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}

pub fn unrolled_i8_dot32_batch4(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    for block in 0..blocks {
        let offset = block * 34;
        let qs = &q8[offset + 2..offset + 34];
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            for lane in 0..4 {
                output[batch_idx + lane] += scale
                    * dot_i8_f32_unrolled(
                        qs,
                        &input[(batch_idx + lane) * in_features + in_feature..],
                    );
            }
            batch_idx += 4;
        }
        while batch_idx < batch {
            output[batch_idx] +=
                scale * dot_i8_f32_unrolled(qs, &input[batch_idx * in_features + in_feature..]);
            batch_idx += 1;
        }
    }
    output
}

pub fn reflow_i8_scaled_batch4(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    batch: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch];
    let blocks = q8.len() / 34;
    for block in 0..blocks {
        let offset = block * 34;
        let qs = &q8[offset + 2..offset + 34];
        let in_feature = block * 32;
        let mut batch_idx = 0usize;
        while batch_idx + 4 <= batch {
            accumulate_reeflow_i8_scaled_batch4(
                qs,
                scale,
                &input[batch_idx * in_features + in_feature..],
                in_features,
                &mut output,
                batch_idx,
            );
            batch_idx += 4;
        }
        if batch_idx < batch {
            let scaled = scaled_block(qs, scale);
            while batch_idx < batch {
                output[batch_idx] +=
                    dot_f32_32(&scaled, &input[batch_idx * in_features + in_feature..]);
                batch_idx += 1;
            }
        }
    }
    output
}

pub fn baseline_i8_dot32_batch1_row(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    in_features: usize,
) -> Vec<f32> {
    let blocks = q8.len() / 34;
    let mut output = vec![0.0f32; 1];
    for block in 0..blocks {
        let offset = block * 34;
        let in_feature = block * 32;
        output[0] += scale * dot_i8_f32(&q8[offset + 2..offset + 34], &input[in_feature..]);
    }
    assert_eq!(blocks * 32, in_features);
    output
}

pub fn scaled_f32_dot32_batch1_row(
    q8: &[u8],
    scale: f32,
    input: &[f32],
    in_features: usize,
) -> Vec<f32> {
    let blocks = q8.len() / 34;
    let mut output = vec![0.0f32; 1];
    for block in 0..blocks {
        let offset = block * 34;
        let in_feature = block * 32;
        let scaled = scaled_block(&q8[offset + 2..offset + 34], scale);
        output[0] += dot_f32_32(&scaled, &input[in_feature..]);
    }
    assert_eq!(blocks * 32, in_features);
    output
}

fn dot_i8_f32(qs: &[u8], input: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for idx in 0..32 {
        acc += (qs[idx] as i8 as f32) * input[idx];
    }
    acc
}

fn dot_i8_f32_unrolled(qs: &[u8], input: &[f32]) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut idx = 0usize;
    while idx < 32 {
        acc0 += (qs[idx] as i8 as f32) * input[idx];
        acc1 += (qs[idx + 1] as i8 as f32) * input[idx + 1];
        acc2 += (qs[idx + 2] as i8 as f32) * input[idx + 2];
        acc3 += (qs[idx + 3] as i8 as f32) * input[idx + 3];
        idx += 4;
    }
    (acc0 + acc1) + (acc2 + acc3)
}

fn scaled_block(qs: &[u8], scale: f32) -> [f32; 32] {
    let mut scaled = [0.0f32; 32];
    for idx in 0..32 {
        scaled[idx] = (qs[idx] as i8 as f32) * scale;
    }
    scaled
}

#[cfg(target_arch = "aarch64")]
unsafe fn scaled_block_neon(qs: &[u8], scale: f32) -> [f32; 32] {
    let mut out = [0.0f32; 32];
    let scale_vec = vdupq_n_f32(scale);
    let mut offset = 0usize;
    while offset < 32 {
        let q_i8 = vld1q_s8(qs.as_ptr().add(offset) as *const i8);
        let low_i16 = vmovl_s8(vget_low_s8(q_i8));
        let high_i16 = vmovl_s8(vget_high_s8(q_i8));

        let low_low_i32 = vmovl_s16(vget_low_s16(low_i16));
        let low_high_i32 = vmovl_s16(vget_high_s16(low_i16));
        let high_low_i32 = vmovl_s16(vget_low_s16(high_i16));
        let high_high_i32 = vmovl_s16(vget_high_s16(high_i16));

        vst1q_f32(
            out.as_mut_ptr().add(offset),
            vmulq_f32(vcvtq_f32_s32(low_low_i32), scale_vec),
        );
        vst1q_f32(
            out.as_mut_ptr().add(offset + 4),
            vmulq_f32(vcvtq_f32_s32(low_high_i32), scale_vec),
        );
        vst1q_f32(
            out.as_mut_ptr().add(offset + 8),
            vmulq_f32(vcvtq_f32_s32(high_low_i32), scale_vec),
        );
        vst1q_f32(
            out.as_mut_ptr().add(offset + 12),
            vmulq_f32(vcvtq_f32_s32(high_high_i32), scale_vec),
        );
        offset += 16;
    }
    out
}

#[cfg(target_arch = "aarch64")]
unsafe fn scaled_pair_block_neon(first: &[u8], second: &[u8], scale: f32) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    let first_scaled = scaled_block_neon(first, scale);
    let second_scaled = scaled_block_neon(second, scale);
    out[..32].copy_from_slice(&first_scaled);
    out[32..].copy_from_slice(&second_scaled);
    out
}

#[cfg(target_arch = "aarch64")]
fn prescaled_sidecar_blocks(q8: &[u8], scale: f32) -> Vec<[f32; 32]> {
    let blocks = q8.len() / 34;
    let mut sidecar = Vec::with_capacity(blocks);
    for block in 0..blocks {
        let offset = block * 34;
        sidecar.push(unsafe { scaled_block_neon(&q8[offset + 2..offset + 34], scale) });
    }
    sidecar
}

fn accumulate_scaled_batch4(
    scaled: &[f32; 32],
    input: &[f32],
    stride: usize,
    output: &mut [f32],
    batch_idx: usize,
) {
    output[batch_idx] += dot_f32_32(scaled, input);
    output[batch_idx + 1] += dot_f32_32(scaled, &input[stride..]);
    output[batch_idx + 2] += dot_f32_32(scaled, &input[stride * 2..]);
    output[batch_idx + 3] += dot_f32_32(scaled, &input[stride * 3..]);
}

fn accumulate_scaled_batch4_runtime(
    scaled: &[f32; 32],
    input: &[f32],
    stride: usize,
    output: &mut [f32],
    batch_idx: usize,
) {
    let mut acc0 = output[batch_idx];
    let mut acc1 = output[batch_idx + 1];
    let mut acc2 = output[batch_idx + 2];
    let mut acc3 = output[batch_idx + 3];
    let mut idx = 0usize;
    while idx < 32 {
        let weight = scaled[idx];
        acc0 += weight * input[idx];
        acc1 += weight * input[stride + idx];
        acc2 += weight * input[stride * 2 + idx];
        acc3 += weight * input[stride * 3 + idx];
        idx += 1;
    }
    output[batch_idx] = acc0;
    output[batch_idx + 1] = acc1;
    output[batch_idx + 2] = acc2;
    output[batch_idx + 3] = acc3;
}

fn accumulate_reelane_scaled_batch4(
    scaled: &[f32; 32],
    input: &[f32],
    stride: usize,
    output: &mut [f32],
    batch_idx: usize,
) {
    let mut acc0 = output[batch_idx];
    let mut acc1 = output[batch_idx + 1];
    let mut acc2 = output[batch_idx + 2];
    let mut acc3 = output[batch_idx + 3];
    let mut idx = 0usize;
    while idx < 32 {
        let weight0 = scaled[idx];
        let weight1 = scaled[idx + 1];
        let weight2 = scaled[idx + 2];
        let weight3 = scaled[idx + 3];

        acc0 += weight0 * input[idx];
        acc1 += weight0 * input[stride + idx];
        acc2 += weight0 * input[stride * 2 + idx];
        acc3 += weight0 * input[stride * 3 + idx];

        acc0 += weight1 * input[idx + 1];
        acc1 += weight1 * input[stride + idx + 1];
        acc2 += weight1 * input[stride * 2 + idx + 1];
        acc3 += weight1 * input[stride * 3 + idx + 1];

        acc0 += weight2 * input[idx + 2];
        acc1 += weight2 * input[stride + idx + 2];
        acc2 += weight2 * input[stride * 2 + idx + 2];
        acc3 += weight2 * input[stride * 3 + idx + 2];

        acc0 += weight3 * input[idx + 3];
        acc1 += weight3 * input[stride + idx + 3];
        acc2 += weight3 * input[stride * 2 + idx + 3];
        acc3 += weight3 * input[stride * 3 + idx + 3];

        idx += 4;
    }
    output[batch_idx] = acc0;
    output[batch_idx + 1] = acc1;
    output[batch_idx + 2] = acc2;
    output[batch_idx + 3] = acc3;
}

#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_neon_scaled_batch4(
    scaled: &[f32; 32],
    input: &[f32],
    stride: usize,
    output: &mut [f32],
    batch_idx: usize,
) {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut idx = 0usize;
    while idx < 32 {
        let weights = vld1q_f32(scaled.as_ptr().add(idx));
        let x0 = vld1q_f32(input.as_ptr().add(idx));
        let x1 = vld1q_f32(input.as_ptr().add(stride + idx));
        let x2 = vld1q_f32(input.as_ptr().add(stride * 2 + idx));
        let x3 = vld1q_f32(input.as_ptr().add(stride * 3 + idx));
        acc0 = vfmaq_f32(acc0, weights, x0);
        acc1 = vfmaq_f32(acc1, weights, x1);
        acc2 = vfmaq_f32(acc2, weights, x2);
        acc3 = vfmaq_f32(acc3, weights, x3);
        idx += 4;
    }
    output[batch_idx] += vaddvq_f32(acc0);
    output[batch_idx + 1] += vaddvq_f32(acc1);
    output[batch_idx + 2] += vaddvq_f32(acc2);
    output[batch_idx + 3] += vaddvq_f32(acc3);
}

#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_neon_scaled_tail3(
    scaled: &[f32; 32],
    input: &[f32],
    stride: usize,
    output: &mut [f32],
    batch_idx: usize,
) {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut idx = 0usize;
    while idx < 32 {
        let weights = vld1q_f32(scaled.as_ptr().add(idx));
        acc0 = vfmaq_f32(acc0, weights, vld1q_f32(input.as_ptr().add(idx)));
        acc1 = vfmaq_f32(acc1, weights, vld1q_f32(input.as_ptr().add(stride + idx)));
        acc2 = vfmaq_f32(
            acc2,
            weights,
            vld1q_f32(input.as_ptr().add(stride * 2 + idx)),
        );
        idx += 4;
    }
    output[batch_idx] += vaddvq_f32(acc0);
    output[batch_idx + 1] += vaddvq_f32(acc1);
    output[batch_idx + 2] += vaddvq_f32(acc2);
}

#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_neon_scaled_batch8(
    scaled: &[f32; 32],
    input: &[f32],
    stride: usize,
    output: &mut [f32],
    batch_idx: usize,
) {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut acc4 = vdupq_n_f32(0.0);
    let mut acc5 = vdupq_n_f32(0.0);
    let mut acc6 = vdupq_n_f32(0.0);
    let mut acc7 = vdupq_n_f32(0.0);
    let mut idx = 0usize;
    while idx < 32 {
        let weights = vld1q_f32(scaled.as_ptr().add(idx));
        acc0 = vfmaq_f32(acc0, weights, vld1q_f32(input.as_ptr().add(idx)));
        acc1 = vfmaq_f32(acc1, weights, vld1q_f32(input.as_ptr().add(stride + idx)));
        acc2 = vfmaq_f32(
            acc2,
            weights,
            vld1q_f32(input.as_ptr().add(stride * 2 + idx)),
        );
        acc3 = vfmaq_f32(
            acc3,
            weights,
            vld1q_f32(input.as_ptr().add(stride * 3 + idx)),
        );
        acc4 = vfmaq_f32(
            acc4,
            weights,
            vld1q_f32(input.as_ptr().add(stride * 4 + idx)),
        );
        acc5 = vfmaq_f32(
            acc5,
            weights,
            vld1q_f32(input.as_ptr().add(stride * 5 + idx)),
        );
        acc6 = vfmaq_f32(
            acc6,
            weights,
            vld1q_f32(input.as_ptr().add(stride * 6 + idx)),
        );
        acc7 = vfmaq_f32(
            acc7,
            weights,
            vld1q_f32(input.as_ptr().add(stride * 7 + idx)),
        );
        idx += 4;
    }
    output[batch_idx] += vaddvq_f32(acc0);
    output[batch_idx + 1] += vaddvq_f32(acc1);
    output[batch_idx + 2] += vaddvq_f32(acc2);
    output[batch_idx + 3] += vaddvq_f32(acc3);
    output[batch_idx + 4] += vaddvq_f32(acc4);
    output[batch_idx + 5] += vaddvq_f32(acc5);
    output[batch_idx + 6] += vaddvq_f32(acc6);
    output[batch_idx + 7] += vaddvq_f32(acc7);
}

#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_neon_scaled64_batch4(
    scaled: &[f32; 64],
    input: &[f32],
    stride: usize,
    output: &mut [f32],
    batch_idx: usize,
) {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut idx = 0usize;
    while idx < 64 {
        let weights = vld1q_f32(scaled.as_ptr().add(idx));
        acc0 = vfmaq_f32(acc0, weights, vld1q_f32(input.as_ptr().add(idx)));
        acc1 = vfmaq_f32(acc1, weights, vld1q_f32(input.as_ptr().add(stride + idx)));
        acc2 = vfmaq_f32(
            acc2,
            weights,
            vld1q_f32(input.as_ptr().add(stride * 2 + idx)),
        );
        acc3 = vfmaq_f32(
            acc3,
            weights,
            vld1q_f32(input.as_ptr().add(stride * 3 + idx)),
        );
        idx += 4;
    }
    output[batch_idx] += vaddvq_f32(acc0);
    output[batch_idx + 1] += vaddvq_f32(acc1);
    output[batch_idx + 2] += vaddvq_f32(acc2);
    output[batch_idx + 3] += vaddvq_f32(acc3);
}

fn accumulate_reeflow_i8_scaled_batch4(
    qs: &[u8],
    scale: f32,
    input: &[f32],
    stride: usize,
    output: &mut [f32],
    batch_idx: usize,
) {
    let mut acc0 = output[batch_idx];
    let mut acc1 = output[batch_idx + 1];
    let mut acc2 = output[batch_idx + 2];
    let mut acc3 = output[batch_idx + 3];
    let mut idx = 0usize;
    while idx < 32 {
        let weight = scale * (qs[idx] as i8) as f32;
        acc0 += weight * input[idx];
        acc1 += weight * input[stride + idx];
        acc2 += weight * input[stride * 2 + idx];
        acc3 += weight * input[stride * 3 + idx];
        idx += 1;
    }
    output[batch_idx] = acc0;
    output[batch_idx + 1] = acc1;
    output[batch_idx + 2] = acc2;
    output[batch_idx + 3] = acc3;
}

fn dot_f32_32(weights: &[f32; 32], input: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for idx in 0..32 {
        acc += weights[idx] * input[idx];
    }
    acc
}

fn dot_f32(weights: &[f32], input: &[f32]) -> f32 {
    weights
        .iter()
        .zip(input.iter())
        .map(|(weight, value)| weight * value)
        .sum()
}

fn checksum(values: &[f32]) -> f32 {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| value * ((idx % 13) as f32 + 1.0))
        .sum()
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_kernel_lab_reports_required_ree_variants() {
        let report = run_suite(Q8KernelBenchConfig {
            batch: 5,
            in_features: 64,
            blocks_per_row: 2,
            out_features: 1,
            iters: 2,
        });

        assert_eq!(report.ree_kernel, "REEDOT-LAB");

        let variants = report
            .results
            .iter()
            .map(|result| result.variant.as_str())
            .collect::<Vec<_>>();
        let portable_variants = [
            "baseline_i8_dot32_batch4",
            "scaled_f32_dot32_batch4",
            "scaled_f32_dot32_batch4_runtime",
            "reelane_f32_dot32_batch4",
            "reeflow_i8_scaled_batch4",
            "unrolled_i8_dot32_batch4",
        ];
        assert_eq!(&variants[..portable_variants.len()], portable_variants);
        #[cfg(target_arch = "aarch64")]
        assert!(variants.contains(&"reevec_neon_f32_dot32_batch4"));
        #[cfg(target_arch = "aarch64")]
        assert!(variants.contains(&"reecast_neon_scale_batch4"));
        #[cfg(target_arch = "aarch64")]
        assert!(variants.contains(&"reewide_neon_f32_dot32_batch8"));
        #[cfg(target_arch = "aarch64")]
        assert!(variants.contains(&"reeduo_neon_block64_batch4"));
        #[cfg(target_arch = "aarch64")]
        assert!(variants.contains(&"reeside_prescaled_f32_batch4"));
        #[cfg(target_arch = "aarch64")]
        assert!(variants.contains(&"reetail_neon_tail3_batch4"));

        for result in &report.results {
            assert!(
                result.elapsed_ns > 0,
                "{} should report elapsed time",
                result.variant
            );
            assert!(
                result.max_abs_diff <= 0.0001,
                "{} diff {} exceeded tolerance",
                result.variant,
                result.max_abs_diff
            );
        }
    }

    #[test]
    fn q8_kernel_lab_reports_batch1_decode_gate_variants() {
        let report = run_suite(Q8KernelBenchConfig {
            batch: 1,
            in_features: 2048,
            blocks_per_row: 64,
            out_features: 8192,
            iters: 2,
        });

        let variants = report
            .results
            .iter()
            .map(|result| result.variant.as_str())
            .collect::<Vec<_>>();

        assert!(variants.contains(&"baseline_i8_dot32_batch1_row"));
        assert!(variants.contains(&"scaled_f32_dot32_batch1_row"));

        for result in report
            .results
            .iter()
            .filter(|result| result.variant.ends_with("_batch1_row"))
        {
            assert!(
                result.max_abs_diff <= 0.0001,
                "{} diff {} exceeded tolerance",
                result.variant,
                result.max_abs_diff
            );
        }
    }
}
