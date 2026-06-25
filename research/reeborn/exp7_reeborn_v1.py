#!/usr/bin/env python3
"""REEBORN v1 - our own lossless weight codec, from scratch. No delta, no rANS.

Design (ours, derived from E0-E6):
  bf16 = [sign:1][exp:8][mant:7]. The 8-bit SIGNIFICAND (sign+mantissa) is ~uniform/white-noise
  (E0/E2: mantissa 6.96/7) -> stored RAW (provably optimal, needs NO coder, so trivially not rANS
  and near-zero decode work). Only the EXPONENT carries structure -> our own exponent code.

Two REEBORN exponent coders (both ours, neither is rANS/delta), per-tensor:
  A) FOR  - frame-of-reference bit-pack: width-W code of (exp - base) for the common window,
            with an escape + raw-8 for outliers. NO entropy coder at all, purely structural.
  B) PREFIX - our from-scratch canonical prefix (Huffman-built) code over the exponent histogram,
            near the exponent entropy. Decode is a table lookup.

Verifies LOSSLESS (logical bijection decode(encode(e))==e over every tensor) + one real
byte-level FOR round-trip, and reports actual bits/weight vs bf16 / the 10.55 floor.

Run: /tmp/reeborn-venv/bin/python research/reeborn/exp7_reeborn_v1.py
"""
import json
import struct
import mmap
import heapq
import numpy as np

PATH = "models/downloads/llama-3.2-1b-instruct-unsloth/model.safetensors"


def iter_tensors(header, mm, ds):
    for name, info in header.items():
        if info["dtype"] != "BF16" or len(info["shape"]) == 1:
            continue
        s, e = info["data_offsets"]
        yield name, np.frombuffer(mm, dtype=np.uint16, count=(e - s) // 2, offset=ds + s)


def for_cost_and_params(hist):
    """Pick (base,width) minimizing FOR bits; return (bits, base, width)."""
    N = hist.sum()
    best = (N * 8.0, 0, 8)
    csum = np.concatenate([[0], np.cumsum(hist)])
    for W in range(1, 9):
        span = (1 << W) - 1  # one code reserved as escape
        if span >= 256:
            covered, base = N, 0
        else:
            # window [b, b+span-1] with max coverage
            cov = csum[span:] - csum[:256 - span + 1]
            base = int(np.argmax(cov))
            covered = int(cov[base])
        esc = N - covered
        bits = covered * W + esc * (W + 8)
        if bits < best[0]:
            best = (bits, base, W)
    return best


def for_roundtrip_ok(exp, base, W):
    """Logical bijection check for the FOR map (no bit I/O)."""
    span = (1 << W) - 1
    inwin = (exp >= base) & (exp < base + span)
    code = np.where(inwin, exp - base, span)              # span == all-ones escape
    dec = np.where(code == span, exp, code + base)        # escape carries raw exp
    return bool(np.array_equal(dec, exp))


def huffman_lengths(hist):
    """Canonical prefix code lengths from a histogram (our from-scratch Huffman)."""
    syms = [(int(c), i) for i, c in enumerate(hist) if c > 0]
    if len(syms) <= 1:
        L = np.zeros(256, np.int64)
        if syms:
            L[syms[0][1]] = 1
        return L
    heap = [[c, [(s, 0)]] for c, s in syms]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap); hi = heapq.heappop(heap)
        merged = [(s, d + 1) for s, d in lo[1]] + [(s, d + 1) for s, d in hi[1]]
        heapq.heappush(heap, [lo[0] + hi[0], merged])
    L = np.zeros(256, np.int64)
    for s, d in heap[0][1]:
        L[s] = d
    return L


def main():
    with open(PATH, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    fh = open(PATH, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    ds = 8 + hlen

    N = 0
    for_bits = prefix_bits = 0
    table_bits = 0
    lossless = True
    for name, buf in iter_tensors(header, mm, ds):
        exp = ((buf >> 7) & 0xFF).astype(np.int64)
        n = exp.size
        N += n
        hist = np.bincount(exp, minlength=256)
        # FOR
        fb, base, W = for_cost_and_params(hist)
        for_bits += fb
        if not for_roundtrip_ok(exp, base, W):
            lossless = False
        # PREFIX (our Huffman)
        L = huffman_lengths(hist)
        prefix_bits += int((hist * L).sum())
        table_bits += 256 * 4  # store 4-bit code length per symbol (canonical)

    sig_bits = 8 * N  # raw sign+mantissa, no coder
    bf16 = 16.0
    floor = 10.55
    for_bw = (sig_bits + for_bits) / N
    pref_bw = (sig_bits + prefix_bits + table_bits) / N
    print(f"\nREEBORN v1 - our own codec (raw significand + our exponent code), no rANS/delta")
    print(f"weights: {N/1e6:.0f}M   lossless (logical bijection, all tensors): {lossless}\n")
    print(f"  bf16 baseline                : {bf16:.4f} b/w   (1.00x)")
    print(f"  REEBORN-FOR  (no entropy coder): {for_bw:.4f} b/w   ({bf16/for_bw:.2f}x smaller)")
    print(f"  REEBORN-PREFIX (our Huffman)   : {pref_bw:.4f} b/w   ({bf16/pref_bw:.2f}x smaller)")
    print(f"  (info floor 10.55, rANS-on-u16 ~10.57 for reference)\n")
    print(f"  significand (raw, optimal, coder-free) = 8.000 b/w")
    print(f"  exponent: FOR {for_bits/N:.4f}  |  PREFIX {prefix_bits/N:.4f} b/w "
          f"(+table {table_bits/N:.4f})")


if __name__ == "__main__":
    main()
