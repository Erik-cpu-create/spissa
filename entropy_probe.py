#!/usr/bin/env python3
# Does the lossless "10.5 bit floor" hold, or do gemma weights have cross-weight
# structure a transform/predictive codec can exploit to go lower (losslessly)?
# rANS hits the MARGINAL per-weight entropy. If conditional/delta entropy is lower,
# the floor is the codec's, not physics -> a smarter lossless codec wins.
import json, struct
import numpy as np

PATH = "models/gemma-3-4b-it/model-00001-of-00002.safetensors"

with open(PATH, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    data_start = 8 + n
    cand = [k for k, v in header.items()
            if isinstance(v, dict) and v.get("dtype") == "BF16"
            and len(v.get("shape", [])) == 2 and "proj" in k
            and "vision" not in k]  # LANGUAGE model weights only
    if not cand:
        cand = [k for k, v in header.items()
                if isinstance(v, dict) and v.get("dtype") == "BF16"
                and len(v.get("shape", [])) == 2 and "proj" in k]
    name = sorted(cand)[len(cand) // 2]
    meta = header[name]
    s, e = meta["data_offsets"]
    f.seek(data_start + s)
    raw = f.read(e - s)

w = np.frombuffer(raw, dtype=np.uint16)
shape = meta["shape"]
print(f"tensor: {name}  shape={shape}  n={w.size:,}")

def H(sym):
    _, c = np.unique(sym, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

sign = (w >> 15) & 1
exp  = (w >> 7) & 0xFF
man  = w & 0x7F
marg = H(sign) + H(exp) + H(man)
print(f"\n-- MARGINAL (what rANS ~achieves) --")
print(f"  sign {H(sign):.2f} + exp {H(exp):.2f} + mantissa {H(man):.2f} = {marg:.2f} bits/weight")
print(f"  whole-uint16 order-0: {H(w):.2f} bits/weight")

W = w.reshape(shape).astype(np.int64)
E = exp.reshape(shape).astype(np.int64)
M = man.reshape(shape).astype(np.int64)

def cond_H(field, mod):
    left = np.roll(field, 1, axis=1)[:, 1:].ravel()
    cur  = field[:, 1:].ravel()
    return H(left * mod + cur) - H(left)  # H(cur | left)

print(f"\n-- ORDER-1 (predict from left neighbor in the row) --")
print(f"  exp:      H0 {H(exp):.2f} -> H(exp|left)      {cond_H(E,256):.2f} bits")
print(f"  mantissa: H0 {H(man):.2f} -> H(man|left)      {cond_H(M,128):.2f} bits")

print(f"\n-- ROW-DELTA of uint16 bit-pattern (smoothness?) --")
d = (W - np.roll(W, 1, axis=1))[:, 1:].ravel()
print(f"  delta order-0: {H(d):.2f} bits/weight")

# col-delta too (cross-row structure)
dc = (W - np.roll(W, 1, axis=0))[1:, :].ravel()
print(f"  col-delta order-0: {H(dc):.2f} bits/weight")

print(f"\n-- VERDICT --")
best_struct = min(cond_H(M,128) + H(exp) + H(sign), H(d), marg)
print(f"  marginal floor (rANS): {marg:.2f} bits  |  best structured: {best_struct:.2f} bits")
print(f"  q8 reference: ~8.5 bits (LOSSY)")
if best_struct < marg - 0.3:
    print(f"  => STRUCTURE EXISTS: a transform/predictive lossless codec can beat rANS by ~{marg-best_struct:.1f} bits. Floor is the codec's, NOT physics.")
else:
    print(f"  => weights look ~marginal; structure gain small here. Try other transforms.")
