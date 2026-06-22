#!/usr/bin/env python
"""HF reference forward for Qwen3.5-2B text decoder — ground-truth to diff the RLLM
adapter against. Prints top-5 next-token logits and per-layer last-token hidden norms
for the prompt, so we can localize where the Rust forward diverges."""
import sys, torch
from transformers import AutoTokenizer, AutoModelForImageTextToText

SRC = "models/qwen3.5-2b-src"
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"

tok = AutoTokenizer.from_pretrained(SRC)
print("loading model (cpu, fp32)...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(SRC, torch_dtype=torch.float32)
model.eval()

# Text decoder + tied head.
lm = model.model.language_model  # Qwen3_5 text model
lm_head = model.lm_head if hasattr(model, "lm_head") else None

ids = tok(PROMPT, return_tensors="pt").input_ids
print("prompt:", repr(PROMPT), "ids:", ids.tolist()[0], flush=True)

with torch.no_grad():
    out = lm(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple: embeddings + each layer
    last = out.last_hidden_state[0, -1]  # [hidden]
    # tied lm_head = embed_tokens
    emb = lm.embed_tokens.weight  # [vocab, hidden]
    logits = last @ emb.t() if lm_head is None else lm_head(last)

print("\n=== per-layer last-token hidden norm (HF reference) ===", flush=True)
dump = []
for i, h in enumerate(hs):
    v = h[0, -1]
    n = v.norm().item()
    tag = "embed" if i == 0 else f"L{i-1:02}"
    print(f"  {tag}: |h_last|={n:.3f}")
    dump.append((tag, v.float().tolist()))
with open("qwen_ref_dump.txt", "w") as f:
    for tag, vec in dump:
        f.write(tag + " " + " ".join(f"{x:.6f}" for x in vec) + "\n")
print("wrote qwen_ref_dump.txt", flush=True)

top = torch.topk(logits, 8)
print("\n=== top-8 next-token (HF reference) ===", flush=True)
for v, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  id={i:6} logit={v:.3f} tok={tok.decode([i])!r}")
