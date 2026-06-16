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
  - `models/SmolLM2-135M-Instruct-raw.rllm`
  - `models/SmolLM2-135M-Instruct-q4_0.rllm`
  - `models/SmolLM2-135M-Instruct-q4_0_keep_io.rllm`
- Architecture: SmolLM2/Llama-compatible decoder
- Target device/profile: local macOS CPU, release binary
- Expected bottleneck: memory footprint vs dequantization CPU cost
- Bottleneck tag: memory bandwidth | tokenizer | runtime bug

## Setup

Commands:

```bash
printf 'Answer in one short sentence: what is 2 plus 2?\nquit\n' \
  | /usr/bin/time -l target/release/llama-test \
      --model models/SmolLM2-135M-Instruct-q4_0.rllm \
      --ctx 512 \
      --max-new-tokens 32 \
      --chat-template chatml

set -u
out=/tmp/rllm-smollm-chatml-matrix-20260616.txt
: > "$out"
models=("models/SmolLM2-135M-Instruct-raw.rllm" "models/SmolLM2-135M-Instruct-q4_0.rllm")
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
  --out models/SmolLM2-135M-Instruct-q4_0_keep_io.rllm \
  --codec raw \
  --quantize q4_0_keep_io
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
| `SmolLM2-135M-Instruct-raw.rllm` | 260M |
| `SmolLM2-135M-Instruct-q4_0.rllm` | 76M |
| `SmolLM2-135M-Instruct-q4_0_keep_io.rllm` | 115M |

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

This trial therefore supports the Q4_0 pivot for memory/runtime feasibility, but
does not yet close the chat-quality question.

## Decision

needs follow-up

Reason: Q4_0 passes the low-RAM runtime smoke and avoids the previous immediate
ChatML/tokenizer collapse. Raw RLLM matches Hugging Face greedy on this matrix.
Full Q4_0 and keep-IO Q4_0 both diverge on at least one prompt, so the
multi-prompt quality matrix is not strong enough to call Q4_0 chat-quality
parity accepted.

Paper value:

- use as positive evidence for low-RAM exact execution feasibility
- use as limitation for quality validation still needing an external baseline

## Next Experiment

Run a quantization-policy sweep that preserves progressively more sensitive
subsystems:

- Q4 transformer with raw embeddings/output already tested as `q4_0_keep_io`.
- Next: raw attention + Q4 MLP.
- Next: raw first/last N layers + Q4 middle layers.
- Compare each against the Hugging Face/raw greedy token sequence, not only text.
