#!/usr/bin/env python
"""Capture layer-0 Gated-DeltaNet internals from HF (in_proj_qkv output and the final
mixer/out_proj output) to diff against the RLLM deltanet and localize the bug."""
import sys, torch
from transformers import AutoTokenizer, AutoModelForImageTextToText

SRC = "models/qwen3.5-2b-src"
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"

tok = AutoTokenizer.from_pretrained(SRC)
model = AutoModelForImageTextToText.from_pretrained(SRC, dtype=torch.float32).eval()
lm = model.model.language_model
la = lm.layers[0].linear_attn
print("linear_attn submodules:", [n for n, _ in la.named_children()], flush=True)

caps = {}
def mk(name):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        caps[name] = o.detach()
    return hook

lm.layers[0].input_layernorm.register_forward_hook(mk("normx"))
la.in_proj_qkv.register_forward_hook(mk("qkv"))
la.conv1d.register_forward_hook(mk("conv"))  # [1, 6144, seq] pre-silu
def norm_in_hook(mod, inp, out):
    caps["o"] = inp[0].detach()  # core_attn_out (o) pre gated-norm
la.norm.register_forward_hook(norm_in_hook)
la.register_forward_hook(mk("mixout"))

ids = tok(PROMPT, return_tensors="pt").input_ids
seq = ids.shape[1]
with torch.no_grad():
    lm(input_ids=ids, use_cache=False)

import torch.nn.functional as F
with open("qwen_l0_dump.txt", "w") as f:
    for tag, key in [("hf_normx", "normx"), ("hf_qkv", "qkv"), ("hf_mixout", "mixout")]:
        v = caps[key]
        v = v.reshape(v.shape[0], v.shape[1], -1)[0, -1].float().tolist()
        f.write(tag + " " + " ".join(f"{x:.6f}" for x in v) + "\n")
        print(f"{tag}: shape_lasttok={len(v)}", flush=True)
    # conv: [1, channels, seq] pre-silu -> last valid token (seq-1), apply silu
    cv = caps["conv"][0, :, seq - 1]
    cv = F.silu(cv.float()).tolist()
    f.write("hf_conv " + " ".join(f"{x:.6f}" for x in cv) + "\n")
    print(f"hf_conv: shape_lasttok={len(cv)}", flush=True)
    ot = caps["o"]
    print("hf_o raw shape:", tuple(ot.shape), flush=True)
    # Bring to [seq, features]; last-token = features for the final position.
    if ot.dim() == 4:      # [b, s, h, d] or [b, h, s, d]
        ot = ot[0]
        # heuristic: seq dim is the one matching `seq`
        if ot.shape[0] == seq:      # [s, h, d]
            ov = ot[-1].reshape(-1)
        else:                        # [h, s, d]
            ov = ot[:, -1, :].reshape(-1)
    elif ot.dim() == 3:    # [b, s, D]
        ov = ot[0, -1]
    elif ot.dim() == 2:    # [seq*heads, vd] flattened
        N, D = ot.shape
        heads_n = N // seq
        ov = ot.reshape(seq, heads_n, D)[-1].reshape(-1)  # last token, all heads
    else:                  # [s, D]
        ov = ot[-1]
    ov = ov.float().tolist()
    f.write("hf_o " + " ".join(f"{x:.6f}" for x in ov) + "\n")
    print(f"hf_o: shape_lasttok={len(ov)}", flush=True)
print("wrote qwen_l0_dump.txt", flush=True)
