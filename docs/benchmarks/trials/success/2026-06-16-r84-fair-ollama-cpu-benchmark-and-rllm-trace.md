# R84: Fair Ollama CPU Benchmark and RLLM Trace

## Status

Success.

## Hypothesis

RLLM R83 is still much slower than Ollama CPU-only because its prefill kernels
lack llama.cpp-class CPU repack/vectorized execution. R84 measures the current
gap and identifies the next RLLM bottleneck before any new kernel work.

## Artifact

- RLLM model: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa`
- Ollama model: `llama3.2:1b`
- RLLM mode: exact-lowram Q8, `--rama-integrity unchecked`
- Ollama mode: CPU-only, `num_gpu:0`
- Prompt: `Answer yes or no: is fire cold?`
- Max new tokens: 4

Ollama model details:

- `digest`: `baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878`
- `size`: `1321098329`
- `format`: `gguf`
- `family`: `llama`
- `parameter_size`: `1.2B`
- `quantization_level`: `Q8_0`
- `context_length`: `131072`
- `embedding_length`: `2048`

## Commands

Ollama model check:

```sh
ollama list
curl -s http://127.0.0.1:11434/api/tags | jq '.models[] | select(.name=="llama3.2:1b")'
```

Ollama CPU-only first prompt:

```sh
ollama stop llama3.2:1b || true
curl -s http://127.0.0.1:11434/api/chat -d '{"model":"llama3.2:1b","messages":[{"role":"user","content":"Answer yes or no: is fire cold?"}],"stream":false,"options":{"num_predict":4,"temperature":0,"num_ctx":2048,"num_gpu":0}}' | tee /tmp/r84-ollama-cpu-first.json | jq '{message:.message.content,total_duration,load_duration,prompt_eval_count,prompt_eval_duration,eval_count,eval_duration,prompt_eval_s:(.prompt_eval_duration/1000000000),eval_tok_s:(.eval_count/(.eval_duration/1000000000))}'
```

RLLM build:

```sh
cargo build --release --bin llama-test
```

RLLM R83 unchecked benchmark:

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked"
```

RLLM trace:

```sh
/usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked --rama-trace target/r84-rllm-trace.json"
jq '[.summary.duration_by_phase[] | {phase,event_count,total_ms}]' target/r84-rllm-trace.json
jq '[.summary.duration_by_tensor_bucket[] | {bucket,event_count,total_ms}] | sort_by(.total_ms) | reverse' target/r84-rllm-trace.json
```

## Results

Ollama CPU-only first prompt:

```json
{
  "message": "No.",
  "total_duration": 3061740333,
  "load_duration": 2712573792,
  "prompt_eval_count": 34,
  "prompt_eval_duration": 285813000,
  "eval_count": 3,
  "eval_duration": 59531000,
  "prompt_eval_s": 0.285813,
  "eval_tok_s": 50.39391241538022
}
```

Ollama CPU-only proof from server log:

- runner args included `-ngl 0`
- `offloaded 0/17 layers to GPU`
- `CPU model buffer size = 266.41 MiB`
- `CPU_REPACK model buffer size = 1252.16 MiB`
- `CPU KV buffer size = 64.00 MiB`
- prompt eval log: `285.81 ms / 34 tokens`, `118.96 tokens per second`
- server reported `n_threads = 2`, `n_threads_batch = 2`

RLLM R83 unchecked benchmark:

- output: `No`
- TTFT/prefill: `13.94s`
- decode: `1.31 tok/s`
- end-to-end: `0.14 tok/s`
- total generated tokens: `2`
- context: `55 tokens`
- internal peak transient: `1050673152 bytes`
- `/usr/bin/time -l` max RSS: `1655635968 bytes`
- `/usr/bin/time -l` real: `19.09s`

RLLM prefill profile:

- prefill total: `13944.54ms`
- transformer: `13041.61ms`
- attention total: `2337.53ms`
- MLP total: `10703.88ms`
- lm_head: `902.54ms`
- q: `906.07ms`
- k: `238.45ms`
- v: `238.62ms`
- attn: `35.50ms`
- gate: `3623.39ms`
- up: `3388.24ms`
- down: `3678.19ms`

RLLM traced run:

- output: `No`
- TTFT/prefill: `13.14s`
- decode: `0.66 tok/s`
- end-to-end: `0.14 tok/s`
- total generated tokens: `2`
- context: `55 tokens`
- internal peak transient: `1050673152 bytes`
- `/usr/bin/time -l` max RSS: `1659977728 bytes`
- `/usr/bin/time -l` real: `17.55s`

## Gap

Raw prefill comparison:

- Ollama CPU-only: `0.285813s / 34 prompt tokens`
- RLLM unchecked: `13.94s / 55 context tokens`
- RLLM/Ollama total prefill gap: about `48.8x`

Per-prompt-token comparison:

- Ollama CPU-only: `8.41ms/token`
- RLLM unchecked: about `253.54ms/token`
- RLLM/Ollama per-token gap: about `30.1x`

The token counts differ because the runtimes expose prompt/context accounting
differently, so the per-token gap is the fairer directional number and the raw
wall-clock gap is the user-visible first-response gap.

## Trace Attribution

Phase totals:

```json
[
  {
    "phase": "chunk_compute_closure",
    "event_count": 2176,
    "total_ms": 11836.073283
  },
  {
    "phase": "chunk_decode",
    "event_count": 2176,
    "total_ms": 1544.814886
  },
  {
    "phase": "chunk_read",
    "event_count": 3178,
    "total_ms": 3.122858
  }
]
```

Tensor bucket totals:

```json
[
  {
    "bucket": "mlp.down_proj",
    "event_count": 576,
    "total_ms": 3354.262448
  },
  {
    "bucket": "mlp.gate_proj",
    "event_count": 576,
    "total_ms": 3337.438512
  },
  {
    "bucket": "mlp.up_proj",
    "event_count": 576,
    "total_ms": 3102.476702
  },
  {
    "bucket": "attention.q_proj",
    "event_count": 160,
    "total_ms": 832.902828
  },
  {
    "bucket": "attention.o_proj",
    "event_count": 160,
    "total_ms": 810.18104
  },
  {
    "bucket": "attention.v_proj",
    "event_count": 64,
    "total_ms": 199.589377
  },
  {
    "bucket": "attention.k_proj",
    "event_count": 64,
    "total_ms": 199.222376
  }
]
```

Attribution:

- `chunk_read` is effectively eliminated at `3.12ms`.
- `chunk_decode` is not dominant at `1544.81ms`.
- `chunk_compute_closure` dominates at `11836.07ms`.
- MLP buckets are balanced and dominate together: down `3354.26ms`, gate
  `3337.44ms`, up `3102.48ms`.
- Attention Q/O are secondary: q `832.90ms`, o `810.18ms`.

## Decision

R84 is successful as a measurement stage. The fair CPU-only comparison confirms
RLLM is still far behind Ollama on prefill even after R83, while keeping the
expected low-RAM exact-Q8 behavior and correct answer on the sanity prompt.

R85 should target shared Q8 dot/repack work for MLP projections, not file IO,
checksum, or a single projection-specific patch. The trace says the next
meaningful fix must reduce the common compute path used by gate/up/down, ideally
moving closer to llama.cpp-style packed CPU execution while preserving RLLM's
exact low-RAM invariant.
