#!/usr/bin/env python3
"""REEBORN E9 - validate the FOR exponent codec on REAL Llama-3.2-1B exponents.

The FOR decode speed is distribution-independent (always `width` bits, deterministic refill),
so E7/edge-bench already proved the 6x decode win. What's NOT yet known is the real RATIO:
what fixed `width` do actual per-tensor exponent ranges need, and what bits/weight does
REEBORN-FOR (raw 8-bit significand + per-tensor fixed-width exponent) land at on a real model?

Per 2-D bf16 tensor: width = ceil(log2(exp_max - exp_min + 1)) covers the full range losslessly,
no escape needed. Reports the per-tensor width distribution, the weighted REEBORN-FOR b/w, and a
global-width+escape alternative, vs DFloat/rANS (~10.6) and bf16 (16).

Run: /tmp/reeborn-venv/bin/python research/reeborn/exp9_for_width_realmodel.py
"""
import json
import struct
import mmap
import math
import numpy as np

PATH = "models/downloads/llama-3.2-1b-instruct-unsloth/model.safetensors"
HEADER_BITS = 12  # per-tensor: base(8) + width(4)


def main():
    with open(PATH, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    ds = 8 + hlen
    fh = open(PATH, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    width_hist = {}
    tot_w = tot_n = 0
    g_exp = np.zeros(256, np.int64)
    per_tensor = []
    for name, info in header.items():
        if info["dtype"] != "BF16" or len(info["shape"]) == 1:
            continue
        s, e = info["data_offsets"]
        buf = np.frombuffer(mm, dtype=np.uint16, count=(e - s) // 2, offset=ds + s)
        exp = ((buf >> 7) & 0xFF)
        n = exp.size
        emin, emax = int(exp.min()), int(exp.max())
        width = max(1, math.ceil(math.log2(emax - emin + 1)))
        width_hist[width] = width_hist.get(width, 0) + n
        tot_w += width * n
        tot_n += n
        g_exp += np.bincount(exp, minlength=256)
        per_tensor.append((width, n, emin, emax))

    avg_w = tot_w / tot_n
    for_bw = 8 + avg_w + HEADER_BITS * len(per_tensor) / tot_n
    # global fixed width + escape: pick the smallest W covering >=99.5% of mass around the
    # global mode; rare out-of-window exps cost W + 8 (escape).
    cdf = np.cumsum(g_exp) / g_exp.sum()
    mode = int(np.argmax(g_exp))
    best = None
    for W in range(3, 9):
        span = (1 << W) - 1  # one code reserved as escape
        base = max(0, mode - span // 2)
        covered = int(g_exp[base:base + span].sum())
        esc = tot_n - covered
        bits = covered * W + esc * (W + 8)
        bw = 8 + bits / tot_n + HEADER_BITS / tot_n
        if best is None or bw < best[0]:
            best = (bw, W, esc / tot_n)

    print(f"\nLlama-3.2-1B  REEBORN-FOR exponent-width validation ({tot_n/1e6:.0f}M weights, "
          f"{len(per_tensor)} tensors)\n")
    print(f"  per-tensor exponent width distribution (bits -> % of weights):")
    for w in sorted(width_hist):
        print(f"    width {w}: {100*width_hist[w]/tot_n:5.1f}%")
    print(f"\n  weighted avg exponent width        = {avg_w:.3f} bits")
    print(f"  REEBORN-FOR b/w (significand 8 + per-tensor width + hdr) = {for_bw:.3f} b/w "
          f"({16/for_bw:.2f}x vs bf16)")
    print(f"  global width+escape best: W={best[1]}, escape {best[2]*100:.2f}% -> {best[0]:.3f} b/w\n")
    print(f"  reference: DFloat/rANS ~10.6 b/w (1.51x), bf16 16 (1.00x)")
    print(f"  -> REEBORN-FOR trades ~{for_bw-10.6:.1f} b/w ratio for table-free, 6x-faster, "
          f"NEON-able decode (edge win in >RAM streaming).")


if __name__ == "__main__":
    main()
