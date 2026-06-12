# RTC — RLLM Tensor Codec

## Overview

RTC is a library of lossless compression codecs designed for LLM tensor data. Every codec must satisfy the fundamental contract:

```
decode(encode(input)) == input  (bit-identical)
```

## Codec Trait

```rust
pub trait TensorCodec: Send + Sync {
    fn id(&self) -> &'static str;
    fn encode(&self, input: &[u8], meta: &EncodeMeta) -> Result<EncodedChunk>;
    fn decode(&self, encoded: &[u8], meta: &DecodeMeta) -> Result<Vec<u8>>;
    fn verify_roundtrip(&self, input: &[u8], meta: &EncodeMeta) -> Result<bool>;
}
```

## Implemented Codecs

### rtc-raw-v1 (Identity)

**Status:** ✅ Implemented

Passes data through unchanged. Serves as:
- Fallback when other codecs make data larger
- Baseline for testing the codec framework
- Reference implementation

### rtc-rle-v1 (Run-Length Encoding)

**Status:** ✅ Implemented

Encodes runs of repeated bytes. Effective for:
- Zero-filled tensors
- Tensors with large constant regions
- Sparse data

### rtc-huff-v1 (Byte-Level Huffman)

**Status:** ✅ Implemented

Builds a static byte-frequency Huffman tree per chunk and stores:
- 256 × u32 little-endian frequency table
- u64 little-endian encoded bit length
- MSB-first Huffman bitstream

Effective for:
- FP16/BF16 tensor chunks with non-uniform byte distributions
- Attention masks/bias tensors with skewed byte frequencies
- Any tensor bytes where entropy coding beats raw/RLE output

This codec is implemented in-house as part of RTC. RLLM should not add generic
external compression dependencies by default; new compression stages should be
custom RTC codecs unless explicitly approved.

### rtc-delta-v1 (Delta Encoding)

**Status:** 🔜 Future

Stores differences between consecutive values. Effective for:
- Monotonically increasing/decreasing data
- Smooth gradients
- Sorted indices

### rtc-bitplane-v1 (Bitplane Packing)

**Status:** 🔜 Future

Separates and compresses individual bit planes. Effective for:
- Floating-point tensors (exponent bits are compressible)
- Quantized integers
- Data with non-uniform bit distributions

### rtc-entropy-v1 (Entropy Coding)

**Status:** 🔜 Future

More advanced entropy coding beyond `rtc-huff-v1` (for example canonical tables,
range coding, or transform+entropy pipelines). Effective for:
- Any data with non-uniform byte distribution
- Final compression stage after other transforms

## Codec Selection

The packer tries multiple codecs on each chunk and selects the one that:
1. Produces the smallest output
2. Passes round-trip verification

```rust
for each chunk:
    candidates = []
    candidates.append(encode_raw(chunk))
    candidates.append(encode_rle(chunk))
    candidates.append(encode_huff(chunk))
    candidates.append(encode_delta(chunk))
    candidates.append(encode_bitplane(chunk))
    choose smallest candidate that decodes exactly
```

If all codecs make the data larger, use `rtc-raw-v1`.

## Dtype-Aware Compression

Different data types benefit from different strategies:

### fp16 / bf16
- Split into sign/exponent/mantissa
- Compress exponent stream (often repetitive)
- Try bitplane packing
- Fallback to raw if mantissa is random

### Quantized integers (int4, int8)
- Respect original packed representation
- Never change values
- Try bit-level repacking only if reversible
- Compress scales and zero-points separately

### Metadata / scale tensors
- Try delta coding
- Try RLE
- Try entropy coding

## Lossless Verification

Every encode operation includes a self-test:

```rust
let encoded = codec.encode(chunk, meta)?;
let decoded = codec.decode(&encoded.data, &decode_meta)?;
assert_eq!(decoded, chunk, "Round-trip failed");
```

If verification fails, the codec candidate is rejected.

## Testing Requirements

Every codec must pass:

- Empty input
- Small input (< 1KB)
- Large input (> 1MB)
- Random input
- Repeated input (all zeros, all ones)
- Structured tensor input
- Corrupted input handling
- Wrong checksum handling

## Performance Metrics

Track for each codec:

- Compression ratio (compressed / original)
- Encode speed (MB/s)
- Decode speed (MB/s)
- Memory usage during encode/decode

## Design Principles

1. **Correctness first** — lossless verification is non-negotiable
2. **Honest metrics** — report actual compression, never overclaim
3. **Fallback to raw** — if compression makes data larger, use raw
4. **Dtype awareness** — adapt strategy to data type
5. **Incremental** — add codecs one at a time, test thoroughly
