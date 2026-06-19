//! rtc-dfloat-v1: lossless bf16 codec.
//!
//! bf16 = [sign:1][exponent:8][mantissa:7]. The exponent has low entropy for LLM
//! weights, so we entropy-code it (canonical Huffman) and store sign+mantissa raw.
//! Original implementation (technique from DFloat11, arXiv 2504.11651); no code
//! was copied and no external dependency is used.

/// Split a bf16 bit pattern into (exponent, residual=sign|mantissa).
/// exponent = bits 14..=7 ; residual = (sign << 7) | mantissa(bits 6..=0).
#[inline]
pub fn split_bf16(bits: u16) -> (u8, u8) {
    let exponent = ((bits >> 7) & 0xFF) as u8;
    let sign = ((bits >> 15) & 0x1) as u8;
    let mantissa = (bits & 0x7F) as u8;
    let residual = (sign << 7) | mantissa;
    (exponent, residual)
}

/// Inverse of `split_bf16`.
#[inline]
pub fn join_bf16(exponent: u8, residual: u8) -> u16 {
    let sign = ((residual >> 7) & 0x1) as u16;
    let mantissa = (residual & 0x7F) as u16;
    (sign << 15) | ((exponent as u16) << 7) | mantissa
}

pub struct BitWriter {
    bytes: Vec<u8>,
    cur: u8,
    nbits: u8, // bits currently filled in `cur` (0..8)
}

impl BitWriter {
    pub fn new() -> Self {
        Self { bytes: Vec::new(), cur: 0, nbits: 0 }
    }

    /// Write the low `len` bits of `code`, most-significant bit first.
    pub fn write(&mut self, code: u32, len: u8) {
        let mut i = len;
        while i > 0 {
            i -= 1;
            let bit = ((code >> i) & 1) as u8;
            self.cur = (self.cur << 1) | bit;
            self.nbits += 1;
            if self.nbits == 8 {
                self.bytes.push(self.cur);
                self.cur = 0;
                self.nbits = 0;
            }
        }
    }

    /// Flush, zero-padding the final partial byte.
    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.cur <<= 8 - self.nbits;
            self.bytes.push(self.cur);
        }
        self.bytes
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

pub struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize, // absolute bit offset from the start
}

impl<'a> BitReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    /// Peek the next `n` bits (n <= 24) as an integer, MSB-first, zero-padded
    /// past the end of the buffer.
    pub fn peek(&self, n: u8) -> u32 {
        let mut out = 0u32;
        for k in 0..n {
            let abs = self.bit_pos + k as usize;
            let byte = abs / 8;
            let bit_in_byte = 7 - (abs % 8);
            let bit = if byte < self.bytes.len() {
                ((self.bytes[byte] >> bit_in_byte) & 1) as u32
            } else {
                0
            };
            out = (out << 1) | bit;
        }
        out
    }

    pub fn advance(&mut self, n: u8) {
        self.bit_pos += n as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_join_roundtrips_every_bf16_value() {
        for bits in 0u32..=0xFFFF {
            let bits = bits as u16;
            let (e, r) = split_bf16(bits);
            assert_eq!(join_bf16(e, r), bits, "roundtrip failed for {bits:#06x}");
        }
    }

    #[test]
    fn bit_writer_reader_roundtrips_variable_codes() {
        // (code, len) pairs, MSB-first.
        let symbols = [(0b1u32, 1u8), (0b01, 2), (0b001, 3), (0b0, 1), (0b101, 3)];
        let mut w = BitWriter::new();
        for &(c, l) in &symbols {
            w.write(c, l);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for &(c, l) in &symbols {
            assert_eq!(r.peek(l), c, "peek mismatch for code {c:#b}/{l}");
            r.advance(l);
        }
    }
}
