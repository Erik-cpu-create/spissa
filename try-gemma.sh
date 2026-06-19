#!/usr/bin/env bash
# Quick Gemma 3 4B (q8, lossless) chat runner for RLLM.
#
# Usage:
#   ./try-gemma.sh chat                            # interactive multi-turn chat (REPL)
#   ./try-gemma.sh "your prompt here"              # one-shot, 80 tokens
#   ./try-gemma.sh "your prompt here" 200          # one-shot, 200 tokens
#   ./try-gemma.sh -v "your prompt here"           # -v: show prefill/decode timing
#   ./try-gemma.sh -r "continue this text"         # -r: raw (no chat template)
#
# In chat mode the model loads once and remembers the conversation (resident KV
# cache, only the new message is prefilled each turn). Commands: /reset, /exit.
# --fast is always on (mlock residency + int8 sdot kernels). Default threads.
set -euo pipefail

cd "$(dirname "$0")"

MODEL="models/gemma-3-4b-it-q8.rllm"
BIN="target/release/gemma-test"

VERBOSE=0
CHAT="--chat"
while [[ "${1:-}" == -* ]]; do
  case "$1" in
    -v) VERBOSE=1; shift ;;
    -r) CHAT=""; shift ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

# Interactive chat mode: `./try-gemma.sh chat`
if [[ "${1:-}" == "chat" ]]; then
  if [[ ! -x "$BIN" ]] || [[ -n "$(find crates -name '*.rs' -newer "$BIN" -print -quit 2>/dev/null)" ]]; then
    echo "[try-gemma] building release binary..." >&2
    cargo build --release -p rllm-cli >&2
  fi
  exec "$BIN" --model "$MODEL" --fast --interactive --ctx "${2:-1024}"
fi

PROMPT="${1:-What is the capital of Australia?}"
TOKENS="${2:-80}"

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Pack it first, or point MODEL= at your .rllm file." >&2
  exit 1
fi

# Build the release binary if it's missing or older than the sources.
if [[ ! -x "$BIN" ]] || [[ -n "$(find crates -name '*.rs' -newer "$BIN" -print -quit 2>/dev/null)" ]]; then
  echo "[try-gemma] building release binary..." >&2
  cargo build --release -p rllm-cli >&2
fi

ENV=()
if [[ "$VERBOSE" == 1 ]]; then
  ENV=(RLLM_Q8_KERNEL_PROFILE=1)
fi

echo "[try-gemma] prompt: $PROMPT" >&2
echo "[try-gemma] tokens: $TOKENS  mode: ${CHAT:-raw}  (--fast)" >&2
echo >&2

env ${ENV[@]+"${ENV[@]}"} "$BIN" \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  $CHAT --fast \
  --max-new-tokens "$TOKENS" \
  --ctx 512
