#!/usr/bin/env python3
"""REEBORN E6 - permutation / BWT-style reorder (filament #2).

Order-0 entropy is permutation-INVARIANT, so a reorder only helps if paired with a context
coder AND it manufactures local coherence the natural order lacks. We test the fairest
affordable permutation: sort rows by mean-exponent (so vertical neighbours become similar)
and columns by mean-exponent, and measure whether H(exp|neighbour) drops by MORE than the cost
of storing the permutation (log2(R!) bits, amortised). Also reports the fine-grain bound
(sorting all weights) to show why element-level permutation is hopeless.

Run: /tmp/reeborn-venv/bin/python research/reeborn/exp6_permutation.py
"""
import json
import struct
import mmap
import math
import numpy as np

PATH = "models/downloads/llama-3.2-1b-instruct-unsloth/model.safetensors"
LN2 = math.log(2.0)


def H(counts):
    c = np.asarray(counts, np.float64); N = c.sum()
    if N <= 0:
        return 0.0
    p = c[c > 0] / N
    return float(-(p * np.log2(p)).sum())


def cond_up(joint):
    j = joint.reshape(256, 256)
    return H(j) - H(j.sum(axis=1))


def iter_tensors(header, mm, ds):
    for info in header.values():
        if info["dtype"] != "BF16" or len(info["shape"]) == 1:
            continue
        s, e = info["data_offsets"]
        buf = np.frombuffer(mm, dtype=np.uint16, count=(e - s) // 2, offset=ds + s)
        yield ((buf >> 7) & 0xFF).astype(np.int64), info["shape"]


def log2fact(n):
    return math.lgamma(n + 1) / LN2


def main():
    with open(PATH, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    fh = open(PATH, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    ds = 8 + hlen

    marg = np.zeros(256, np.int64)
    up_nat = np.zeros(256 * 256, np.int64)     # H(exp|up) natural order
    up_sorted = np.zeros(256 * 256, np.int64)  # after sorting rows by mean-exp
    left_nat = np.zeros(256 * 256, np.int64)
    left_sorted = np.zeros(256 * 256, np.int64)  # after sorting cols by mean-exp
    perm_row_bits = perm_col_bits = 0.0
    N = 0

    for e2, (R, C) in iter_tensors(header, mm, ds):
        e2 = e2.reshape(R, C)
        marg += np.bincount(e2.ravel(), minlength=256)
        N += R * C
        # natural neighbours
        up_nat += np.bincount(e2[:-1, :].ravel() * 256 + e2[1:, :].ravel(), minlength=65536)
        left_nat += np.bincount(e2[:, :-1].ravel() * 256 + e2[:, 1:].ravel(), minlength=65536)
        # sort rows by mean exponent -> vertical neighbours become similar
        rs = e2[np.argsort(e2.mean(axis=1))]
        up_sorted += np.bincount(rs[:-1, :].ravel() * 256 + rs[1:, :].ravel(), minlength=65536)
        perm_row_bits += log2fact(R)
        # sort columns by mean exponent -> horizontal neighbours become similar
        cs = e2[:, np.argsort(e2.mean(axis=0))]
        left_sorted += np.bincount(cs[:, :-1].ravel() * 256 + cs[:, 1:].ravel(), minlength=65536)
        perm_col_bits += log2fact(C)

    Hexp = H(marg)
    rows_cost = perm_row_bits / N
    cols_cost = perm_col_bits / N
    print(f"\nLlama-3.2-1B. H(exp) = {Hexp:.4f} b/w   (order-0, permutation-invariant)\n")
    print(f"{'scheme':<26} {'H(exp|nbr)':>11} {'gain':>8} {'perm_cost':>10} {'NET':>9}  verdict")
    print("-" * 74)
    rows = [
        ("natural up", cond_up(up_nat), 0.0),
        ("row-sorted up", cond_up(up_sorted), rows_cost),
        ("natural left", cond_up(left_nat), 0.0),
        ("col-sorted left", cond_up(left_sorted), cols_cost),
    ]
    for name, hc, cost in rows:
        gain = Hexp - hc
        net = gain - cost
        # honest bar: must beat E3's free per-layer-type clustering (~0.026 b/w, no permutation)
        v = "" if cost == 0 else ("beats free clustering" if net > 0.026
                                  else "dominated by E3 (free 0.026)")
        print(f"{name:<26} {hc:>11.4f} {gain:>8.4f} {cost:>10.4f} {net:>9.4f}  {v}")

    fine = log2fact(N) / N
    print(f"\nfine-grain bound: sorting ALL {N/1e6:.0f}M weights costs log2(N!)/N = {fine:.2f} b/w")
    print(f"  -> dwarfs the {Hexp:.2f} b/w exponent budget. Element-level permutation is hopeless.")
    print(f"compare: E3 per-layer-type clustering already gets ~0.026 b/w FREE (no permutation).")


if __name__ == "__main__":
    main()
