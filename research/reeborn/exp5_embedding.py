#!/usr/bin/env python3
"""REEBORN E5 - embedding row sub-populations (filament #3).

The token embedding (128256 x 2048 = 21% of params) is plausibly a MIXTURE: rare/reserved
tokens stay near initialisation (~random), frequent tokens are heavily trained (different
value distribution). If we cluster rows by a trained-ness signal (row L2-norm, free from the
weights) and code each cluster with its own table, we may beat the embedding's single-table
order-0. Cluster ids cost 128256*log2(K) bits (shipped) -> tiny over 262M weights.

Also checks the reserved/special token rows (ids >= 128000) as a distinct sub-population.

Run: /tmp/reeborn-venv/bin/python research/reeborn/exp5_embedding.py
"""
import json
import struct
import mmap
import math
import numpy as np

PATH = "models/downloads/llama-3.2-1b-instruct-unsloth/model.safetensors"
LN2 = math.log(2.0)
K = 8                # row-norm clusters
TABLE_BITS = 16
BPE_VOCAB = 128000   # ids >= this are reserved/special in Llama-3.x


def h_plugin(h):
    h = np.asarray(h, np.float64); N = h.sum()
    if N <= 0:
        return 0.0
    nz = h > 0
    return -(h[nz] * np.log2(h[nz] / N)).sum() / N


def entropy_mm(h):
    N = float(np.sum(h)); m = int(np.count_nonzero(h))
    Hp = h_plugin(h)
    return Hp + (m - 1) / (2 * N * LN2)


def cond_mm(group_hists, id_cost_bits=0.0):
    N = sum(float(h.sum()) for h in group_hists)
    Hp = sum((h.sum() / N) * h_plugin(h) for h in group_hists if h.sum() > 0)
    cells = sum(int(np.count_nonzero(h)) for h in group_hists)
    Kc = sum(1 for h in group_hists if h.sum() > 0)
    mm = Hp + (cells - Kc) / (2 * N * LN2)
    table_cost = cells * TABLE_BITS / N + id_cost_bits
    return mm, table_cost


def klass(values, k):
    qs = np.quantile(values, [(i + 1) / k for i in range(k - 1)])
    return np.searchsorted(qs, values).astype(np.int64)


def group_hists(symbols, row_cls, V, Hdim, width):
    """Per-cluster histograms of `symbols` (1-D, row-major V*Hdim), clusters per row."""
    cls = np.broadcast_to(row_cls[:, None], (V, Hdim)).ravel()
    out = []
    for c in range(int(row_cls.max()) + 1):
        out.append(np.bincount(symbols[cls == c], minlength=width))
    return out


def main():
    with open(PATH, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    fh = open(PATH, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    info = header["model.embed_tokens.weight"]
    V, Hd = info["shape"]
    s, e = info["data_offsets"]
    buf = np.frombuffer(mm, dtype=np.uint16, count=(e - s) // 2, offset=8 + hlen + s)
    exp = ((buf >> 7) & 0xFF)
    f32 = (buf.astype(np.uint32) << 16).view(np.float32).reshape(V, Hd)
    norm = np.sqrt((f32.astype(np.float64) ** 2).sum(axis=1))

    print(f"\nembedding {V} x {Hd} = {V*Hd/1e6:.1f}M weights ({V*Hd/1235814400*100:.1f}% of model)")
    qs = np.quantile(norm, [0.0, 0.01, 0.5, 0.99, 1.0])
    print(f"row-norm: min {qs[0]:.3f}  p1 {qs[1]:.3f}  median {qs[2]:.3f}  "
          f"p99 {qs[3]:.3f}  max {qs[4]:.3f}  (max/median {qs[4]/qs[2]:.1f}x)")

    Hexp0 = entropy_mm(np.bincount(exp, minlength=256))
    Hu0 = entropy_mm(np.bincount(buf, minlength=65536))
    print(f"\nembedding single-table floor: H(exp) {Hexp0:.4f}  H(u16) {Hu0:.4f} b/w")

    cls = klass(norm, K)
    id_cost = V * math.log2(K) / (V * Hd)
    exp_mm, exp_cost = cond_mm(group_hists(exp, cls, V, Hd, 256), id_cost)
    u_mm, u_cost = cond_mm(group_hists(buf, cls, V, Hd, 65536), id_cost)
    frac = V * Hd / 1235814400
    print(f"\nrow-norm {K}-cluster:")
    for name, mmv, cost, base in [("exp", exp_mm, exp_cost, Hexp0), ("u16", u_mm, u_cost, Hu0)]:
        net = (base - mmv) - cost
        print(f"  {name}: H_MM {mmv:.4f}  gain {base-mmv:.4f}  cost {cost:.5f}  "
              f"NET {net:+.4f} b/w  (overall {net*frac:+.4f})  "
              f"{'*** WIN ***' if net > 0.005 else 'null'}")

    # reserved/special token rows
    if V > BPE_VOCAB:
        sp = slice(BPE_VOCAB * Hd, V * Hd)
        bpe = slice(0, BPE_VOCAB * Hd)
        print(f"\nreserved rows (id>={BPE_VOCAB}, {V-BPE_VOCAB} tokens, "
              f"{(V-BPE_VOCAB)/V*100:.2f}% of vocab):")
        print(f"  H(exp) reserved {entropy_mm(np.bincount(exp[sp], minlength=256)):.4f}  "
              f"vs BPE {entropy_mm(np.bincount(exp[bpe], minlength=256)):.4f}  "
              f"| norm median reserved {np.median(norm[BPE_VOCAB:]):.3f} vs BPE {np.median(norm[:BPE_VOCAB]):.3f}")


if __name__ == "__main__":
    main()
