#!/usr/bin/env python3
# Research v2: EXTREME compression = block-scaled quant (adaptive, like q8/q4) THEN
# rANS-code the indices (entropy-code their non-uniform/peaked distribution). The first
# sweep showed naive mantissa-truncation is dominated by block-quant; the block SCALE is
# what buys precision-per-bit. Here we measure the real frontier: for each q-bit level,
# raw-quant bits vs rANS-coded-index bits, plus RMS rel error. gemma bf16 (4B source,
# representative of the 1B family).
import json, struct
import numpy as np

PATHS = [
    "models/gemma-3-4b-it/model-00001-of-00002.safetensors",
    "models/gemma-3-4b-it/model-00002-of-00002.safetensors",
]
BLOCK = 32

def load():
    out = []
    for path in PATHS:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n)); base = 8 + n
            cand = sorted(k for k, v in h.items()
                          if isinstance(v, dict) and v.get("dtype") == "BF16"
                          and len(v.get("shape", [])) == 2 and "proj" in k and "vision" not in k)
            for k in cand[:: max(1, len(cand)//4)][:4]:
                s, e = h[k]["data_offsets"]; f.seek(base + s)
                out.append(np.frombuffer(f.read(e - s), dtype=np.uint16).copy())
    return np.concatenate(out)

def H(sym):
    _, c = np.unique(sym, return_counts=True); p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

w16 = load()
orig = (w16.astype(np.uint32) << 16).view(np.float32)
n = (orig.size // BLOCK) * BLOCK
wb = orig[:n].reshape(-1, BLOCK)
rms = np.sqrt(np.mean(orig[orig != 0] ** 2))
print(f"sampled {orig.size:,} gemma bf16 weights, block={BLOCK}\n")
print(f"{'scheme':>22} | {'raw bits':>8} | {'rANS bits':>9} | {'RMS err':>8} | {'ratio':>6}")
print("-" * 68)

amax = np.abs(wb).max(axis=1, keepdims=True)
for q in [8, 6, 5, 4, 3, 2]:
    levels = (1 << (q - 1)) - 1
    scale = amax / levels
    scale[scale == 0] = 1
    idx = np.round(wb / scale).clip(-levels, levels).astype(np.int32)
    deq = idx * scale
    err = np.sqrt(np.mean((deq - wb) ** 2)) / rms
    scale_overhead = 16.0 / BLOCK  # one bf16 scale per block
    raw_bits = q + scale_overhead
    rans_bits = H((idx + levels).ravel()) + scale_overhead  # entropy-code the indices
    print(f"{'q'+str(q)+' block-scaled':>22} | {raw_bits:8.2f} | {rans_bits:9.2f} | {err*100:6.3f}% | {16.0/rans_bits:5.2f}x")

print(f"\n{'lossless rANS (ref)':>22} | {'':>8} | {10.5:9.2f} | {0.0:6.3f}% | {16.0/10.5:5.2f}x")
print("(extreme = smallest rANS bits with tolerable err; rANS-bits < raw-bits = entropy win)")
