# R88: Token-Aligned RLLM vs Ollama Prefill Comparison (Llama 3.2 1B)

## Status

Active.

## Hypothesis

The reported RLLM vs Ollama prefill gap is partly a measurement artifact from different prompt framing and thread settings.  
We need an apples-to-apples, same prompt-token count comparison before deciding the next kernel direction.

## Scope

- Mode: exact-lowram
- Model/artifact: `models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa` vs `llama3.2:1b`
- Architecture: Q8 streaming prefill, CPU-only paths
- Target device/profile: Mac (CPU)
- Expected bottleneck: CPU arithmetic
- Bottleneck tag: CPU arithmetic

## Setup

Commands used (same prompt, same `--max-new-tokens 4`, repeatable):

```bash
# Build benchmark binary
cargo build --release -p rllm-cli --bin llama-test

# RLLM: default threading
for i in 1 2 3; do
  /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa \
    --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" \
    > /tmp/r84-rllm-run${i}.txt 2> /tmp/r84-rllm-run${i}.time
done

# RLLM: single-thread mode (explicitly RLLM_THREADS=1)
for i in 1 2 3; do
  RLLM_THREADS=1 /usr/bin/time -l sh -c "printf '%s\nquit\n' 'Answer yes or no: is fire cold?' | target/release/llama-test \
    --model models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.spsa \
    --chat-template llama3 --max-new-tokens 4 --profile-phases --rama-integrity unchecked" \
    > /tmp/r88-rllm-st1-run${i}.txt 2> /tmp/r88-rllm-st1-run${i}.time
done

# Ollama: CPU-only, 2 threads in server logs, chat API
# (server was started with `ollama serve` and stopped after collection)
for i in 1 2 3; do
  curl -sS -H 'Content-Type: application/json' -d \
    '{"model":"llama3.2:1b","messages":[{"role":"user","content":"Answer yes or no: is fire cold?"}],"stream":false,"options":{"num_predict":4,"temperature":0,"num_ctx":2048,"num_gpu":0,"num_thread":2}}' \
    http://127.0.0.1:11434/api/chat | tee /tmp/r84-ollama-${i}.json | jq '{message:.message.content,total_duration,prompt_eval_count,prompt_eval_duration,eval_count,eval_duration}'
done
```

## Runtime Context

- RLLM binary: `target/release/llama-test`
- RLLM runtime integrity: `--rama-integrity unchecked` for timing signal
- Ollama command used: `/api/chat` with `num_gpu:0`, `num_thread:2` from request options
- Ollama server CLI showed `-np 1 --threads 2 -t 2 --chat-template chatml --no-mmap --ngl 0`

## Results

### RLLM (llama3 template, default threads)

| run | output | context tokens | prefill | decode | MLP total | peak transient | max RSS | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | No | 55 | 11.05s | 1.77 tok/s | 8.97s | 1,050,673,152 | 3,159,195,648 bytes | 14.17s |
| 2 | No | 55 | 10.24s | 1.75 tok/s | 8.38s | 1,050,673,152 | 3,281,715,200 bytes | 13.25s |
| 3 | No | 55 | 11.04s | 1.61 tok/s | 9.06s | 1,050,673,152 | 3,281,371,136 bytes | 14.13s |

### RLLM (llama3 template, RLLM_THREADS=1)

| run | output | context tokens | prefill | decode | MLP total | peak transient | elapsed |
|---|---|---:|---:|---:|---:|---:|
| 1 | No | 55 | 12.59s | 1.53 tok/s | 10.31s | 1,050,673,152 | 16.25s |
| 2 | No | 55 | 11.34s | 1.48 tok/s | 9.24s | 1,050,673,152 | 14.68s |
| 3 | No | 55 | 11.30s | 1.53 tok/s | 9.18s | 1,050,673,152 | 14.62s |

### Ollama CPU-only

| run | output | prompt_eval_count | prompt_eval_s | eval_count | eval_s | total_s |
|---|---|---:|---:|---:|---:|---:|
| 1 | No. | 34 | 0.302572 | 3 | 56.8 /s | 3.086432 |
| 2 | No. | 34 | 0.03366 | 3 | 43.1 /s | 0.307723 |
| 3 | No. | 34 | 0.02981 | 3 | 50.7 /s | 0.263493 |

## Analysis

- Output semantics: both runtimes consistently return `No`/`No.`.
- There is a large remaining delta. Even excluding Ollama warm-up, prompt prefill is still about `~23x` slower on RLLM when comparing `/api/chat` responses at 34 prompt tokens (Ollama `~8ms/token` vs RLLM `~190ms/token` on 55 tokens).
- `RLLM_THREADS=1` does not close the gap; it is slower than default-thread mode in these runs.
- Both runs were benchmarked in CPU-only intent, but Ollama logs show explicit CPU thread configuration and llama-server usage (`-t 2`, `-np 1`, `n_threads=2`).

## Decision

needs follow-up

## Next Experiment

Add a controlled parity harness in-repo that keeps the same prompt formatting path on both runtimes and stores full prompt/token-count traces, then focus next kernel work strictly on the high-cost shared `Q8_0` `MLP` bucket in `attention_total`-excluded mode.
