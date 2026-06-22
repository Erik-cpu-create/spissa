# Trial: R72 Q4_0 ChatML Smoke

Date: 2026-06-16
Owner: RLLM
Status: running
Folder: active

## Hypothesis

Q4_0 block quantization should reduce resident memory and artifact size while
keeping the runtime on the exact FP32 execution path after JIT dequantization.
If tokenizer and chat-template boundaries are correct, Q4_0 should no longer
show immediate stop-token collapse or malformed ChatML prompting.

## Scope

- Mode: exact-lowram
- Model/artifact:
  - `models/SmolLM2-135M-Instruct-raw.spsa`
  - `models/SmolLM2-135M-Instruct-q4_0.spsa`
  - `models/SmolLM2-135M-Instruct-q4_0_keep_io.spsa`
  - `models/SmolLM2-135M-Instruct-q4_0_mlp_only.spsa`
  - `models/SmolLM2-135M-Instruct-q4_0_attention_only.spsa`
  - `models/SmolLM2-135M-Instruct-q8_transformer_keepio.spsa`
- Architecture: SmolLM2/Llama-compatible decoder
- Target device/profile: local macOS CPU, release binary
- Expected bottleneck: memory footprint vs dequantization CPU cost
- Bottleneck tag: memory bandwidth | tokenizer | runtime bug

## Setup

Commands:

```bash
printf 'Answer in one short sentence: what is 2 plus 2?\nquit\n' \
  | /usr/bin/time -l target/release/llama-test \
      --model models/SmolLM2-135M-Instruct-q4_0.spsa \
      --ctx 512 \
      --max-new-tokens 32 \
      --chat-template chatml

set -u
out=/tmp/rllm-smollm-chatml-matrix-20260616.txt
: > "$out"
models=("models/SmolLM2-135M-Instruct-raw.spsa" "models/SmolLM2-135M-Instruct-q4_0.spsa")
prompts=(
  "Answer in one short sentence: what is 2 plus 2?"
  "Answer in one short sentence: what color is the sky on a clear day?"
  "Translate to Indonesian: I am learning Rust."
  "List exactly three fruits separated by commas."
  "Answer yes or no: is fire cold?"
)
for model in "${models[@]}"; do
  echo "=== MODEL: $model ===" >> "$out"
  for prompt in "${prompts[@]}"; do
    {
      echo "--- PROMPT: $prompt ---"
      printf '%s\nquit\n' "$prompt" \
        | /usr/bin/time -l target/release/llama-test \
            --model "$model" \
            --ctx 512 \
            --max-new-tokens 32 \
            --chat-template chatml 2>&1
      echo
    } >> "$out"
  done
done
```

External greedy baseline:

```bash
uv run --with torch --with transformers --with accelerate python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_dir = 'models/downloads/smollm2-135m-instruct'
tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, local_files_only=True)
model.eval()
# Generate with tokenizer.apply_chat_template(..., add_generation_prompt=True),
# max_new_tokens=32, do_sample=False.
PY
```

Keep-IO Q4_0 repack:

```bash
target/release/rllm pack \
  models/downloads/smollm2-135m-instruct/model.safetensors \
  --out models/SmolLM2-135M-Instruct-q4_0_keep_io.spsa \
  --codec raw \
  --quantize q4_0_keep_io
```

Projection-family Q4_0 controls:

```bash
target/release/rllm pack \
  models/downloads/smollm2-135m-instruct/model.safetensors \
  --out models/SmolLM2-135M-Instruct-q4_0_mlp_only.spsa \
  --codec raw \
  --quantize q4_0_mlp_only

target/release/rllm pack \
  models/downloads/smollm2-135m-instruct/model.safetensors \
  --out models/SmolLM2-135M-Instruct-q4_0_attention_only.spsa \
  --codec raw \
  --quantize q4_0_attention_only

target/release/rllm pack \
  models/downloads/smollm2-135m-instruct/model.safetensors \
  --out models/SmolLM2-135M-Instruct-q8_transformer_keepio.spsa \
  --codec raw \
  --quantize q8_transformer_keep_io
```

Runtime context:

- build profile: release
- CPU: local macOS host
- RAM: local macOS host
- OS: macOS
- relevant env/config: `--chat-template chatml`, `--ctx 512`, `--max-new-tokens 32`

## Results

Artifact sizes:

| artifact | size |
|---|---:|
| `SmolLM2-135M-Instruct-raw.spsa` | 260M |
| `SmolLM2-135M-Instruct-q4_0.spsa` | 76M |
| `SmolLM2-135M-Instruct-q4_0_keep_io.spsa` | 115M |
| `SmolLM2-135M-Instruct-q4_0_mlp_only.spsa` | 151M |
| `SmolLM2-135M-Instruct-q4_0_attention_only.spsa` | 224M |
| `SmolLM2-135M-Instruct-q8_transformer_keepio.spsa` | 165M |

Prompt matrix:

| run | prompt/input | generated tokens | TTFT/prefill | decode tok/s | end-to-end tok/s | RSS | peak transient | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| raw | `2 plus 2` | 12 | 1.56s | 24.66 | 5.99 | 477577216 | 113262592 | `I'm ready to help you with your math questions.` |
| raw | sky color | 16 | 1.63s | 23.76 | 7.06 | 478773248 | 113262592 | says clear sky is indigo |
| raw | translate to Indonesian | 32 | 1.59s | 22.67 | 10.83 | 478756864 | 113262592 | poor translation, truncated at max token cap |
| raw | three fruits | 13 | 1.60s | 23.04 | 6.14 | 477691904 | 113262592 | lists numbered fruits instead of commas |
| raw | is fire cold | 7 | 1.60s | 23.19 | 3.77 | 477659136 | 113262592 | wrong answer: `Yes. Fire is cold.` |
| Q4_0 | `2 plus 2` | 8 | 1.12s | 12.15 | 4.71 | 255852544 | 120702720 | correct: `2 plus 2 is 4` |
| Q4_0 | sky color | 14 | 1.14s | 12.12 | 6.34 | 255393792 | 120702720 | acceptable: deep blue |
| Q4_0 | translate to Indonesian | 32 | 1.05s | 12.06 | 8.83 | 257589248 | 120702720 | repetitive: `Ini, ini, ...` |
| Q4_0 | three fruits | 13 | 1.03s | 11.93 | 6.38 | 256393216 | 120702720 | lists numbered fruits instead of commas |
| Q4_0 | is fire cold | 8 | 1.03s | 12.07 | 4.96 | 255311872 | 120702720 | wrong answer: `Yes, fire is indeed cold.` |
| Q4_0 keep-IO | `2 plus 2` | 8 | 1.32s | 12.96 | 4.30 | 329285632 | 116785152 | correct: `2 plus 2 is 4` |
| Q4_0 keep-IO | sky color | 14 | 1.22s | 13.07 | 6.34 | 330842112 | 116785152 | acceptable: deep blue |
| Q4_0 keep-IO | translate to Indonesian | 32 | 1.09s | 11.84 | 8.62 | 331792384 | 116785152 | still poor: `Ngaliknya, ini-iya, ...` |
| Q4_0 keep-IO | three fruits | 13 | 1.11s | 13.09 | 6.42 | 331120640 | 116785152 | lists numbered fruits instead of commas |
| Q4_0 keep-IO | is fire cold | 8 | 1.11s | 12.96 | 4.86 | 329859072 | 116785152 | wrong answer: `Yes, fire is indeed cold.` |
| Q4_0 MLP-only | `2 plus 2` | 19 | 1.54s | 14.38 | 6.80 | 367476736 | 116785152 | answers 4 but verbose |
| Q4_0 MLP-only | sky color | 14 | 1.36s | 14.90 | 6.26 | 366821376 | 116785152 | acceptable: deep blue |
| Q4_0 MLP-only | translate to Indonesian | 32 | 1.26s | 14.81 | 9.53 | 368459776 | 116785152 | still poor/repetitive: `Ia makalah ini, ...` |
| Q4_0 MLP-only | three fruits | 9 | 1.22s | 14.49 | 5.07 | 367394816 | 116785152 | closer to raw first token, still numbered |
| Q4_0 MLP-only | is fire cold | 32 | 1.24s | 14.62 | 9.52 | 368328704 | 116785152 | worse than raw, verbose wrong answer |
| Q4_0 attention-only | `2 plus 2` | 8 | 1.95s | 18.33 | 3.42 | 439877632 | 114573312 | correct: `Two plus two is 4.` |
| Q4_0 attention-only | sky color | 16 | 1.68s | 18.23 | 6.40 | 441434112 | 114573312 | matches raw text |
| Q4_0 attention-only | translate to Indonesian | 32 | 1.54s | 17.92 | 9.78 | 442269696 | 114573312 | closer than full Q4, still poor |
| Q4_0 attention-only | three fruits | 14 | 1.53s | 17.87 | 6.19 | 441384960 | 114573312 | numbered fruits |
| Q4_0 attention-only | is fire cold | 8 | 1.55s | 17.75 | 4.12 | 441122816 | 114573312 | wrong answer like Q4 variants |
| Q8 transformer keep-IO | `2 plus 2` | 12 | 1.58s | 14.44 | 5.13 | 382189568 | 116785152 | matches raw text |
| Q8 transformer keep-IO | sky color | 14 | 1.37s | 14.68 | 6.21 | 382222336 | 116785152 | semantically acceptable: deep rich blue |
| Q8 transformer keep-IO | translate to Indonesian | 32 | 1.27s | 14.57 | 9.41 | 383320064 | 116785152 | matches raw text |
| Q8 transformer keep-IO | three fruits | 13 | 1.25s | 14.56 | 6.26 | 382304256 | 116785152 | matches raw text |
| Q8 transformer keep-IO | is fire cold | 7 | 1.29s | 14.69 | 4.12 | 382287872 | 116785152 | matches raw text |

External Hugging Face greedy generation matched the RLLM raw outputs for the
five prompts above, including the weak `fire cold` answer and the poor
Indonesian translation. That isolates those raw-quality failures to the
model/greedy baseline rather than RLLM's tokenizer or runtime.

Verification already run for the implementation commits:

```bash
cargo test -p rllm-runtime
cargo test -p rllm-import
cargo test -p rllm-cli
git diff --check
```

## Analysis

The Q4_0 path now clears the basic runtime requirements:

- The packed Q4_0 artifact is about 29% of the raw artifact size.
- Process RSS drops from about 478-479 MB raw to about 255-258 MB Q4_0 in this
  smoke matrix.
- ChatML prompting no longer immediately stops after the first assistant token.
- Q4_0 can produce a correct short factual answer on the arithmetic prompt.

The quality result is still mixed. The same small SmolLM2-Instruct artifact gives
bad answers in raw, Hugging Face greedy, and Q4_0 for the `fire cold` prompt, so
that case is not specific evidence against quantization. However, Q4_0 shows a
clear repetitive translation failure (`Ini, ini, ...`) that is not present in the
raw/Hugging Face baseline.

The `q4_0_keep_io` variant preserves `model.embed_tokens.weight` and
`lm_head.weight`/tied embedding-output weights as raw BF16. This increased the
artifact from 76M to 115M and the process RSS from about 255-258 MB to about
329-332 MB. It made the translation output less degenerate than full Q4_0, but
it still did not recover raw/Hugging Face behavior. That points to transformer
weight quantization drift, not only embedding/output-head quantization.

The projection-family controls refine the attribution:

- `q4_0_mlp_only` keeps attention and IO raw, but still produces poor translation
  and a worse verbose answer for `fire cold`. This makes Q4 MLP weights the main
  suspect for quality drift.
- `q4_0_attention_only` is closer to raw/Hugging Face on the sky prompt and has a
  less degenerate translation than full Q4, but its memory savings are small
  because MLP and IO remain raw.

In short: full Q4_0 is the best memory result, but Q4_0 MLP quantization is too
lossy for this small SmolLM2 chat-quality matrix.

The Q8 transformer keep-IO variant is the first practical fix in this sweep. It
keeps embeddings/output head raw BF16 and quantizes attention plus MLP projection
weights to Q8_0. It preserves raw text exactly on four of five prompts, with the
remaining sky-color prompt staying semantically acceptable. It reduces artifact
size from 260M to 165M and process RSS from about 478-479 MB to about 382-383
MB. Decode speed falls from about 22-24 tok/s raw to about 14-15 tok/s because
Q8_0 is currently JIT-dequantized into FP32 scratch before matmul.

This trial therefore rejects Q4_0 for chat-quality parity on SmolLM2-Instruct
and accepts Q8 transformer keep-IO as the current practical low-RAM chat
artifact.

## Decision

accepted for Q8 keep-IO; rejected for Q4_0 chat parity

Reason: Raw RLLM matches Hugging Face greedy on this matrix. Full Q4_0,
keep-IO Q4_0, MLP-only Q4_0, and attention-only Q4_0 all diverge on at least
one prompt, so Q4_0 chat-quality parity is rejected. Q8 transformer keep-IO
matches raw text on four of five prompts, keeps the fifth semantically
acceptable, and cuts RSS by about 96 MB versus raw.

Paper value:

- use as positive evidence for Q8 transformer low-RAM exact execution feasibility
- use as negative evidence for Q4_0 MLP chat-quality parity

## Next Experiment

Make Q8 transformer keep-IO the default recommended low-RAM chat artifact for
SmolLM2-Instruct. Next optimization should target Q8_0 direct dot kernels or
SIMD dequantized dot products to recover decode speed without returning to Q4
MLP drift.
