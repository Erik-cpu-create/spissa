//! Buffered bit-reader for fast canonical-Huffman decode.
//!
//! The original `BitReader` (in `dfloat.rs`) computes each bit with a byte
//! index + bit index — a div and a mod per bit, up to `MAX_CODE_LEN` bits per
//! symbol. For 262M symbols that is tens of seconds (it dominated R140a's
//! 0.02 GB/s decode). This reader keeps a 64-bit MSB-aligned window, so a peek
//! is one shift and a symbol lookup costs one shift + one mask.

/// A bit-reader over a byte slice that buffers up to 64 bits in a register-wide
/// window. Bits are consumed most-significant-first, matching `BitWriter`'s
/// MSB-first output and the canonical-Huffman code layout. Reads past the end of
/// the input yield zero bits (the same zero-padding contract as the original
/// `BitReader`), so the final symbols decode identically.
pub struct BufferedBitReader<'a> {
    bytes: &'a [u8],
    pos: usize,   // index of the next byte to pull into the window
    bitbuf: u64,  // buffered bits, next bit to read at the MSB (bit 63)
    bitcnt: u32,  // number of valid bits currently in `bitbuf` (0..=64)
}

impl<'a> BufferedBitReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        let mut r = Self { bytes, pos: 0, bitbuf: 0, bitcnt: 0 };
        r.refill();
        r
    }

    /// Pull bytes into the window until it holds more than 56 bits, i.e. at least
    /// 57 valid bits — enough for any code up to 32 bits. Past end-of-input no
    /// real bytes remain, so zero bytes are folded in: the low bits stay zero,
    /// reproducing the original reader's zero-padding. `bitcnt` never exceeds 64.
    #[inline]
    pub fn refill(&mut self) {
        while self.bitcnt <= 56 {
            let byte = if self.pos < self.bytes.len() {
                let b = self.bytes[self.pos];
                self.pos += 1;
                b
            } else {
                0
            };
            // Place the new byte just below the bits already buffered.
            self.bitbuf |= (byte as u64) << (56 - self.bitcnt);
            self.bitcnt += 8;
        }
    }

    /// Peek the next `n` bits (1..=32) as an integer, MSB-first. Caller must have
    /// called `refill()` so at least `n` bits are buffered.
    #[inline]
    pub fn peek(&self, n: u8) -> u32 {
        debug_assert!((1..=32).contains(&n));
        (self.bitbuf >> (64 - n as u32)) as u32
    }

    /// Advance past `n` consumed bits (`n <= bitcnt`, `n <= 32`).
    #[inline]
    pub fn consume(&mut self, n: u8) {
        debug_assert!(n as u32 <= self.bitcnt);
        self.bitbuf <<= n as u32;
        self.bitcnt -= n as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dfloat::BitWriter;

    #[test]
    fn buffered_reader_matches_bitwriter_stream() {
        // Same (code, len) pairs the original BitReader test uses, MSB-first.
        let symbols = [(0b1u32, 1u8), (0b01, 2), (0b001, 3), (0b0, 1), (0b101, 3)];
        let mut w = BitWriter::new();
        for &(c, l) in &symbols {
            w.write(c, l);
        }
        let bytes = w.finish();
        let mut r = BufferedBitReader::new(&bytes);
        for &(c, l) in &symbols {
            r.refill();
            assert_eq!(r.peek(l), c, "peek mismatch for {c:#b}/{l}");
            r.consume(l);
        }
    }

    #[test]
    fn buffered_reader_zero_pads_past_end() {
        // One byte 0b1010_0000; after consuming the 3 real-ish bits the reader
        // must keep returning zeros (no panic, no garbage).
        let bytes = [0b1010_0000u8];
        let mut r = BufferedBitReader::new(&bytes);
        r.refill();
        assert_eq!(r.peek(4), 0b1010);
        r.consume(8);
        r.refill();
        assert_eq!(r.peek(8), 0, "past end-of-input must read as zeros");
    }
}
