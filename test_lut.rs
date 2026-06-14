const fn fp16_to_f32_const(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = (bits & 0x03ff) as u32;

    let f32_bits = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut frac_norm = frac;
                let mut exp_norm = -14i32;
                while (frac_norm & 0x0400) == 0 {
                    frac_norm <<= 1;
                    exp_norm -= 1;
                }
                frac_norm &= 0x03ff;
                sign | (((exp_norm + 127) as u32) << 23) | (frac_norm << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => {
            let exp32 = (exp as u32) + (127 - 15);
            sign | (exp32 << 23) | (frac << 13)
        }
    };

    f32::from_bits(f32_bits)
}

const fn generate_fp16_lut() -> [f32; 65536] {
    let mut lut = [0.0; 65536];
    let mut i = 0;
    while i < 65536 {
        lut[i] = fp16_to_f32_const(i as u16);
        i += 1;
    }
    lut
}

pub static FP16_TO_F32_LUT: [f32; 65536] = generate_fp16_lut();

fn main() {
    println!("{}", FP16_TO_F32_LUT[0x3c00]); // should be 1.0
}
