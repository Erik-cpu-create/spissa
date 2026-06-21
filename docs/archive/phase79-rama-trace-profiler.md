# Phase 7.9A RAMA Trace Profiler

Phase 7.9A adds a diagnostic RAMA trace profiler for `rllm run` generation paths.

The profiler is intentionally opt-in and has runtime/RSS overhead because it keeps detailed events in memory and writes JSON at the end. It is for bottleneck diagnosis, not normal inference.

## Command

```bash
target/release/rllm run models/pythia-70m-phase78d-tileblocks.rllm \
  --token-ids 12092 \
  --max-new-tokens 1 \
  --ctx 128 \
  --memory-budget 100mb \
  --rama-trace target/phase79a/rama_trace_timed.json
```

`--rama-trace <path>` requires `--prompt` or `--token-ids` because it instruments actual generation, not dry-run planning.

## JSON shape

The output is pretty JSON:

```json
{
  "trace": {
    "schema_version": 1,
    "model_name": "model",
    "architecture": "gpt_neox",
    "started_at_unix_ms": 1781260000000,
    "events": [
      {
        "event_index": 0,
        "phase": "chunk_read",
        "label": "read chunk 123",
        "tensor_name": "gpt_neox.embed_in.weight",
        "tensor_id": 0,
        "chunk_id": 123,
        "codec_id": "rtc-huff-v1",
        "compressed_bytes": 121433,
        "decoded_bytes": 131072,
        "start_ns": 12345,
        "duration_ns": 67890,
        "budget_current_bytes": 121433,
        "budget_peak_bytes": 395264
      }
    ]
  },
  "summary": {
    "event_count": 5370,
    "total_recorded_ns": 4504744000,
    "total_recorded_ms": 4504.744,
    "duration_by_phase": []
  }
}
```

## Instrumented phases

`LazyRllmModel::with_decoded_chunk` records these phases for every full chunk recall:

| Phase | Meaning |
|---|---|
| `chunk_read` | Read compressed chunk payload from `.rllm` container |
| `chunk_compressed_checksum` | SHA-256 verify compressed payload |
| `chunk_decode` | RTC codec decode into original tensor bytes |
| `chunk_original_checksum` | SHA-256 verify decoded/original bytes |
| `chunk_compute_closure` | Caller work while decoded bytes are active; in tiled linear this includes f32 conversion + matmul accumulation |

The first slice deliberately starts at chunk recall granularity. It does not yet split `chunk_compute_closure` into separate tile f32-conversion and matmul sub-events.

## Measured local Pythia-70M trace smoke

Artifact:

```text
models/pythia-70m-phase78d-tileblocks.rllm
```

Command:

```bash
/usr/bin/time -l target/release/rllm run models/pythia-70m-phase78d-tileblocks.rllm \
  --token-ids 12092 \
  --max-new-tokens 1 \
  --ctx 128 \
  --memory-budget 100mb \
  --rama-trace target/phase79a/rama_trace_timed.json
```

Result:

```text
Generated token IDs: [13]
Full text: Hello,
Peak transient budget: 386.00 KiB
RAMA trace JSON: target/phase79a/rama_trace_timed.json
4.48 real / 4.38 user / 0.05 sys
maximum resident set size: 41,631,744 bytes
```

The RSS is higher than non-traced Phase 7.8E runs because trace mode stores 5,370 detailed events and serializes a ~2.5 MiB JSON file. Treat trace-mode RSS as diagnostic overhead, not normal inference RSS.

## Bottleneck summary

Parsed from `target/phase79a/rama_trace.json`:

```text
event_count: 5,370
trace JSON size: 2,551,338 bytes
```

Recorded duration by phase:

| Phase | Events | Total |
|---|---:|---:|
| `chunk_decode` | 1,074 | 3716.017 ms |
| `chunk_original_checksum` | 1,074 | 265.100 ms |
| `chunk_compute_closure` | 1,074 | 254.648 ms |
| `chunk_compressed_checksum` | 1,074 | 237.288 ms |
| `chunk_read` | 1,074 | 31.691 ms |

Interpretation:

```text
The current one-token speed bottleneck is RTC chunk decode, especially Huffman decode, not disk reads.
```

Top tensors by recorded time:

| Tensor | Events | Total |
|---|---:|---:|
| `gpt_neox.embed_in.weight` | 1,965 | 1676.590 ms |
| `embed_out.weight` | 1,965 | 1552.302 ms |
| layer MLP/attention weights | ~80 each | ~70 ms each |

Important RAMA implication:

```text
The runtime is repeatedly recalling large embedding/lm-head memories through the compressed chunk path. RAM is excellent, but speed is dominated by repeated codec decode. Next optimization should reduce repeated decode/recall cost while preserving bounded active memory.
```

## Next RAMA-native optimization candidates

Safe/original candidates, distinct from PowerInfer-style neuron prediction:

1. **Embedding row recall** ✅ implemented in Phase 7.9B: `gpt_neox.embed_in.weight` trace events dropped from 1,965 to 5, and the 12-row matrix improved from 5.07 to 2.93 average seconds/token.
2. **Low-ram-fast layout** ✅ implemented in Phase 7.9C: raw/tile-block artifact plus `--rama-integrity verify-once` moved the 12-row matrix to 0.35 average seconds/token / 3.26 average tok/s while RSS stayed 19.17–23.36 MiB.
3. **LM-head compute strategy**: keep exact full-vocab logits path as correctness baseline, then parallelize/tile `embed_out.weight` compute. Phase 7.9C trace shows `chunk_compute_closure`, not Huffman decode, is now dominant.
4. **Trace sub-events**: split `chunk_compute_closure` into dtype conversion and matmul if needed before implementing the next compute optimization.

## Status

[PRODUCTION-READY]

- `--rama-trace` is opt-in and does not affect default inference behavior.
- Unit test verifies trace events on a raw chunk path.
- Real Pythia tile-block trace smoke produces valid generation and JSON.

[EXPERIMENTAL]

- Trace granularity is currently chunk-level.
- Trace mode increases RSS because events are buffered until process exit.

[NOT DOING]

- No hot/cold neuron predictor.
- No activation locality engine.
- No PowerInfer-style GPU/CPU neuron partitioning.
