#!/usr/bin/env python3
"""REEBORN E4 - structured-sparsity / near-zero sub-population (filament #5).

Order-0 (H(u16)=10.548) ALREADY codes near-zero weights efficiently via their marginal
frequency. A near-zero sub-population only beats order-0 if it is STRUCTURED -- spatially
clustered (runs) or whole dead rows/cols -- so a sparse / run-length code can describe the
mask cheaply. If near-zero weights are scattered i.i.d., order-0 is already optimal and the
mixture idea is null.

E4 tests exactly that: at a near-zero exponent threshold (auto-picked at the ~3rd percentile),
measure (a) near-zero + exact-zero fractions, (b) spatial clustering ratio
P(nz | left nz) / P(nz)  [1.0 = i.i.d., >>1 = clustered = exploitable], (c) dead rows / cols.

Run: /tmp/reeborn-venv/bin/python research/reeborn/exp4_sparsity.py
"""
import json
import struct
import mmap
import numpy as np

PATH = "models/downloads/llama-3.2-1b-instruct-unsloth/model.safetensors"
PCTL = 0.03          # near-zero = the lowest ~3% of weights by magnitude (exponent)
DEAD = 0.90          # a row/col is "dead" if >=90% of it is near-zero


def iter_tensors(header, mm, data_start):
    for info in header.values():
        if info["dtype"] != "BF16" or len(info["shape"]) == 1:
            continue
        s, e = info["data_offsets"]
        buf = np.frombuffer(mm, dtype=np.uint16, count=(e - s) // 2, offset=data_start + s)
        yield buf, info["shape"]


def global_exp_hist(header, mm, data_start):
    h = np.zeros(256, np.int64)
    for buf, _ in iter_tensors(header, mm, data_start):
        h += np.bincount((buf >> 7) & 0xFF, minlength=256)
    return h


def pick_threshold(hexp, pctl):
    cdf = np.cumsum(hexp) / hexp.sum()
    return int(np.searchsorted(cdf, pctl))  # smallest exp with cdf >= pctl


def sparsity_stats(header, mm, data_start, T):
    tot = nz = exact0 = left_nz = pair = 0
    dead_rows = tot_rows = dead_cols = tot_cols = 0
    runs = 0  # number of near-zero runs (for mean run length)
    for buf, (R, C) in iter_tensors(header, mm, data_start):
        exp = (buf >> 7) & 0xFF
        m = (exp < T)
        tot += m.size
        nz += int(m.sum())
        exact0 += int(((buf & 0x7FFF) == 0).sum())
        m2 = m.reshape(R, C)
        l, c = m2[:, :-1], m2[:, 1:]
        left_nz += int(l.sum())
        pair += int((l & c).sum())
        # count near-zero runs per row: a run starts where m is True and left is False
        starts = m2[:, 1:] & ~m2[:, :-1]
        runs += int(starts.sum()) + int(m2[:, 0].sum())
        rowfrac, colfrac = m2.mean(1), m2.mean(0)
        dead_rows += int((rowfrac >= DEAD).sum()); tot_rows += R
        dead_cols += int((colfrac >= DEAD).sum()); tot_cols += C
    return dict(tot=tot, nz=nz, exact0=exact0, left_nz=left_nz, pair=pair, runs=runs,
                dead_rows=dead_rows, tot_rows=tot_rows, dead_cols=dead_cols, tot_cols=tot_cols)


def report(T, s):
    p = s["nz"] / s["tot"]
    p_cond = s["pair"] / s["left_nz"] if s["left_nz"] else 0.0
    cluster = p_cond / p if p else 0.0
    mean_run = s["nz"] / s["runs"] if s["runs"] else 0.0
    iid_run = 1 / (1 - p) if p < 1 else float("inf")  # geometric mean run if i.i.d.
    print(f"\nLlama-3.2-1B. near-zero = exponent < {T} (lowest ~{100*PCTL:.0f}% by magnitude)\n")
    print(f"  near-zero fraction p      = {p*100:.3f}%")
    print(f"  exact +/-0 fraction       = {s['exact0']/s['tot']*100:.4f}%")
    print(f"  P(nz | left nz)           = {p_cond*100:.3f}%   (marginal P(nz) = {p*100:.3f}%)")
    print(f"  CLUSTERING ratio          = {cluster:.3f}   (1.0 = i.i.d. scattered, >>1 = clustered)")
    print(f"  mean near-zero run length = {mean_run:.3f}   (i.i.d. would give {iid_run:.3f})")
    print(f"  dead rows (>={int(DEAD*100)}% nz)      = {s['dead_rows']}/{s['tot_rows']}")
    print(f"  dead cols (>={int(DEAD*100)}% nz)      = {s['dead_cols']}/{s['tot_cols']}")
    clustered = cluster > 1.2 or s["dead_rows"] > 0 or s["dead_cols"] > 0
    print(f"\nverdict: {'STRUCTURED sparsity -> sparse/run-length code can beat order-0 (quantify next)' if clustered else 'near-zero is i.i.d. scattered -> order-0 already optimal -> mixture NULL'}")


def main():
    with open(PATH, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    fh = open(PATH, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    ds = 8 + hlen
    T = pick_threshold(global_exp_hist(header, mm, ds), PCTL)
    report(T, sparsity_stats(header, mm, ds, T))


if __name__ == "__main__":
    main()
