# Phase 7.8 Range Decode Foundation

Phase 7.8 adds the correctness foundation for verified sub-chunk/tile reads without yet routing production matmul through partial compressed reads.

## Status

```text
Phase 7.8A: codec/runtime range-decode API foundation       ✅ done
Phase 7.8B: container compressed byte-range read primitive  ✅ done
Phase 7.8C: per-range checksum metadata foundation          ✅ done
Phase 7.8D0: opt-in raw/identity pack range metadata        ✅ done
Phase 7.8D: pack-time tile/block chunk alignment            ✅ done
Phase 7.8D+: compressed intra-chunk tile blocks             ⏳ future
Phase 7.8E0: real Pythia tile-block RSS benchmark           ✅ done
Phase 7.8E+: intra-chunk compressed range decode/routing    ⏳ future
```

## Implemented primitives

### Codec layer

- `DecodeRange { offset, len }`
- `TensorCodec::decode_range(...)`
- `TensorCodec::supports_native_range_decode()`
- Native range decode for `rtc-raw-v1`
- Full-decode + slice fallback for codecs without native range support

### Container layer

- `RllmReader::read_chunk_range(chunk_id, byte_offset, byte_len)`
- `ChunkRangeMeta` persisted in each `ChunkMeta` as optional `range_checksums`
- Legacy chunk JSON remains readable because `range_checksums` defaults to empty
- `RllmWriter::write_chunk_with_range_specs(...)`
- `RllmWriter::write_chunk_with_identity_range_checksums(...)`

Each `ChunkRangeMeta` records:

```text
range_id
original_offset/original_size
compressed_offset/compressed_size
sha256_original
sha256_compressed
```

Offsets are relative to the parent chunk payload, not the full tensor/file.

### Runtime layer

- `LazyRllmModel::with_decoded_chunk_range(...)`
- `chunk_range_for_original_bytes(...)`
- `verify_original_chunk_range_checksum(...)`
- `verify_compressed_chunk_range_checksum(...)`

Runtime memory accounting is intentionally conservative:

```text
native range codec:
  account requested decoded range

non-native range codec:
  fall back to full chunk decode
  account full decoded chunk
```

### CLI pack path

`rllm pack` now accepts:

```bash
--range-checksum-size 32kb
--tile-block-elements 65536
```

`--tile-block-elements` overrides `--chunk-size` per tensor by multiplying the
element count by that tensor's dtype size. This creates chunk-level tile/block
alignment while preserving existing full-chunk SHA-256 verification. It works for
compressed chunks because each tile/block is still an independent `.spsa` chunk.

`--range-checksum-size` is also opt-in and currently emits per-range checksums
only for identity-mapped raw chunks (`rtc-raw-v1`) where compressed and original
byte spans are the same. For non-identity compressed chunks the packer preserves
the chunk-verified path and reports skipped range checksum emission.

## Integrity guard

Production tiled-linear routing still uses the chunk-verified path. The runtime should not verify a full chunk SHA-256 from partial compressed bytes.

Partial compressed reads become production-safe only when the requested tile/range has independent checksum metadata and the packer emits chunks/blocks whose compressed byte spans map cleanly to decoded tile spans.

## Current verification

Targeted tests cover:

- `DecodeRange` bounds and slicing
- raw native range decode
- runtime raw range budget behavior
- RLE full-decode fallback budget behavior
- container compressed chunk byte-range reads
- chunk metadata legacy deserialization without `range_checksums`
- writer-generated identity range checksums
- out-of-bounds range spec rejection
- runtime original/compressed range checksum verification and corruption detection
- `rllm pack --range-checksum-size` on a tiny raw safetensors fixture, followed by
  `inspect` showing range checksum counts and `verify` proving the packed file is
  still lossless
- `rllm pack --tile-block-elements 64 --range-checksum-size 32b` on the same
  fixture produced 5 chunk-aligned blocks, 9 range checksums, `inspect` reported
  both counts, and `verify` still reported `LOSSLESS VERIFIED`

## Next step

Preliminary Phase 7.8E benchmarking is now complete for a real local Pythia-70M tile-block artifact; see [`phase78-tileblock-rss-benchmark.md`](phase78-tileblock-rss-benchmark.md). The smaller independently verified chunks reduce measured RSS enough to validate the tile-block direction. True intra-chunk partial compressed reads remain future work until compressed range mapping is implemented for non-identity codecs.
