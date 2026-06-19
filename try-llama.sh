#!/usr/bin/env bash
# Quick Llama 3.2 (q8, lossless) chat runner for RLLM.
#
# Usage:
#   ./try-llama.sh chat                            # interactive multi-turn chat (REPL)
#   ./try-llama.sh -m 3b chat                      # pick the 3B model (default: 1b)
#   ./try-llama.sh "your prompt here"              # one-shot, 80 tokens
#   ./try-llama.sh "your prompt here" 200          # one-shot, 200 tokens
#   ./try-llama.sh -m 3b -v "your prompt here"     # 3B + show prefill/decode timing
#   ./try-llama.sh -r chat                         # -r: raw template (no chat formatting)
#
# Models: -m 1b (fast, default) | -m 3b (better quality, ~2x slower).
# The model loads once and remembers the conversation (resident KV cache, only the
# new message is prefilled each turn). In chat, type 'quit' or 'exit' to leave.
# --fast is always on (mlock residency + int8 sdot kernels). Default threads.
set -euo pipefail

cd "$(dirname "$0")"

BIN="target/release/llama-test"
MODEL_KEY="1b"
VERBOSE=0
TEMPLATE="llama3"
while [[ "${1:-}" == -* ]]; do
  case "$1" in
    -v) VERBOSE=1; shift ;;
    -r) TEMPLATE="raw"; shift ;;
    -m) shift; MODEL_KEY="${1:?-m needs a model: 1b or 3b}"; shift ;;
    -m=*|--model=*) MODEL_KEY="${1#*=}"; shift ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

case "$MODEL_KEY" in
  1b) MODEL="models/Llama-3.2-1B-Instruct-q8_transformer_keepio-rowchunks.rllm" ;;
  3b) MODEL="models/Llama-3.2-3B-Instruct-q8.rllm" ;;
  *) echo "unknown model '$MODEL_KEY' (use 1b or 3b)" >&2; exit 1 ;;
esac

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Pack the $MODEL_KEY model first, or point MODEL= at your .rllm file." >&2
  exit 1
fi

build_if_stale() {
  if [[ ! -x "$BIN" ]] || [[ -n "$(find crates -name '*.rs' -newer "$BIN" -print -quit 2>/dev/null)" ]]; then
    echo "[try-llama] building release binary..." >&2
    cargo build --release -p rllm-cli >&2
  fi
}

# Interactive chat mode: `./try-llama.sh chat`
if [[ "${1:-}" == "chat" ]]; then
  build_if_stale
  exec "$BIN" --model "$MODEL" --fast --chat-template "$TEMPLATE" \
    --ctx "${2:-1024}" --max-new-tokens 512
fi

# One-shot: llama-test is always a REPL, so pipe one prompt + quit.
PROMPT="${1:-What is the capital of Australia?}"
TOKENS="${2:-80}"
build_if_stale

PROFILE=()
if [[ "$VERBOSE" == 1 ]]; then
  PROFILE=(--profile-phases)
fi

echo "[try-llama] model: $MODEL_KEY  prompt: $PROMPT" >&2
echo "[try-llama] tokens: $TOKENS  template: $TEMPLATE  (--fast)" >&2
echo >&2

printf '%s\nquit\n' "$PROMPT" | "$BIN" \
  --model "$MODEL" \
  --fast --chat-template "$TEMPLATE" \
  --max-new-tokens "$TOKENS" \
  --ctx 512 ${PROFILE[@]+"${PROFILE[@]}"}
