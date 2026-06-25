#!/usr/bin/env python3
"""REEBORN E3 - mixture / clustered-table coding (filament #1).

Can PER-CONTEXT frequency tables beat the GLOBAL-table order-0 floor (~10.54 b/w) that
REEFORM / R152 quoted? Those numbers used ONE global frequency table. But different layers
have different weight-scale distributions, so coding each cluster (tensor / layer-type /
depth) with its OWN exponent table can go BELOW the global floor. The gain = I(context;
exponent), genuinely sub-floor, and an exponent table is ~free (few active values per
context). The project's global-table measurements never captured this.

Reports H(exp | context) for context in {tensor, layer-type, depth}, Miller-Madow corrected,
MINUS the per-context table storage cost. A surviving NET gain = a real win below the floor.
Also H(u16 | tensor) for the full symbol (table cost is larger there, reported honestly).

Run: /tmp/reeborn-venv/bin/python research/reeborn/exp3_mixture.py
"""
import json
import struct
import mmap
import math
import re
import numpy as np
from collections import defaultdict

PATH = "models/downloads/llama-3.2-1b-instruct-unsloth/model.safetensors"
LN2 = math.log(2.0)
TABLE_BITS = 16  # bits to store one frequency-table entry (upper bound on model cost)


def h_plugin(h):
    h = np.asarray(h, np.float64)
    N = h.sum()
    if N <= 0:
        return 0.0
    nz = h > 0
    return -(h[nz] * np.log2(h[nz] / N)).sum() / N


def entropy_mm(h):
    """(plugin, Miller-Madow) for one histogram, bits."""
    N = float(np.sum(h))
    m = int(np.count_nonzero(h))
    Hp = h_plugin(h)
    return Hp, Hp + (m - 1) / (2 * N * LN2)


def cond_mm(group_hists):
    """H(X|group) over a list of per-group histograms: (plugin, MM, table_cost_bits/weight)."""
    N = sum(float(h.sum()) for h in group_hists)
    Hp = sum((h.sum() / N) * h_plugin(h) for h in group_hists if h.sum() > 0)
    cells = sum(int(np.count_nonzero(h)) for h in group_hists)
    K = sum(1 for h in group_hists if h.sum() > 0)
    mm = Hp + (cells - K) / (2 * N * LN2)
    table_cost = cells * TABLE_BITS / N
    return Hp, mm, table_cost


def tensor_type(name):
    for key in ("embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"):
        if key in name:
            return key
    return "other"


def tensor_depth(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def accumulate(header, mm, data_start):
    per_exp, per_u16, meta = {}, {}, {}
    for name, info in header.items():
        if info["dtype"] != "BF16" or len(info["shape"]) == 1:
            continue
        s, e = info["data_offsets"]
        buf = np.frombuffer(mm, dtype=np.uint16, count=(e - s) // 2, offset=data_start + s)
        per_exp[name] = np.bincount(((buf >> 7) & 0xFF), minlength=256)
        per_u16[name] = np.bincount(buf, minlength=65536)
        meta[name] = (tensor_type(name), tensor_depth(name))
    return per_exp, per_u16, meta


def group_by(per_hist, meta, keyfn, width):
    groups = defaultdict(lambda: np.zeros(width, np.int64))
    for name, h in per_hist.items():
        groups[keyfn(meta[name])] += h
    return list(groups.values())


def report(per_exp, per_u16, meta):
    g_exp = sum(per_exp.values())
    _, Hexp = entropy_mm(g_exp)
    _, Hu16 = entropy_mm(sum(per_u16.values()))
    print(f"\nLlama-3.2-1B. GLOBAL-table floor (what REEFORM/R152 quoted):")
    print(f"  H(exp) = {Hexp:.4f} b/w   H(u16) = {Hu16:.4f} b/w\n")

    print("EXPONENT plane — does a per-context table beat the global H(exp)?")
    print(f"{'context':<14} {'#ctx':>5} {'H_MM':>8} {'gain':>8} {'tbl_cost':>9} {'NET gain':>9}  verdict")
    print("-" * 70)
    contexts = [
        ("tensor", list(per_exp.values())),
        ("layer-type", group_by(per_exp, meta, lambda m: m[0], 256)),
        ("depth", group_by(per_exp, meta, lambda m: m[1], 256)),
    ]
    for label, gh in contexts:
        _, mmv, cost = cond_mm(gh)
        gain = Hexp - mmv
        net = gain - cost
        verdict = "*** REAL WIN ***" if net > 0.01 else "null"
        print(f"{label:<14} {len(gh):>5} {mmv:>8.4f} {gain:>8.4f} {cost:>9.5f} "
              f"{net:>9.4f}  {verdict}")

    # full-symbol per-tensor (table cost is real here)
    _, mm_u16, cost_u16 = cond_mm(list(per_u16.values()))
    gain_u16 = Hu16 - mm_u16
    net_u16 = gain_u16 - cost_u16
    print(f"\nFULL u16 symbol | tensor: H_MM {mm_u16:.4f}  gain {gain_u16:.4f}  "
          f"tbl_cost {cost_u16:.4f}  NET {net_u16:.4f}  "
          f"({'win' if net_u16 > 0.01 else 'table cost eats it'})")

    best_net_exp = max(Hexp - cond_mm(gh)[1] - cond_mm(gh)[2] for _, gh in contexts)
    print(f"\nbest NET exponent win = {best_net_exp:.4f} b/w  ->  "
          f"new floor ~ {Hu16 - max(best_net_exp, 0):.4f} vs global {Hu16:.4f} b/w")


def main():
    with open(PATH, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    fh = open(PATH, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    report(*accumulate(header, mm, 8 + hlen))


if __name__ == "__main__":
    main()
