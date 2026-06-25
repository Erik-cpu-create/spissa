#!/usr/bin/env python3
"""REEBORN Experiment 0 - entropy floors of Llama-3.2-1B weights.

Measures, on the ORIGINAL bf16 weights, the three numbers that decide REEBORN's
whole direction:

  (a) order-0 entropy of the 16-bit bf16 symbols   -> TRUE-lossless floor (vs fp)
      decomposed into sign / exponent / mantissa    -> shows WHERE the entropy lives.
      If this is >> 4 bits, then "<4 bit, lossless vs fp32/bf16" is below the
      Shannon limit => physically impossible, not a tuning problem.

  (b) entropy of q4 / q3 codes (per-32 symmetric absmax) -> QUANTIZED-lossless floor.
      Reported as code-only AND code + fp16-scale effective rate. This is the
      bit-exact-to-the-quantized-model target; <4 bit lives here or nowhere.

  (c) conditional entropy H(code | left) and H(code | up) -> upside of spatial
      modelling (REEBORN lever #2). If ~= marginal, cross-block modelling is a
      dead end (a valid null result that redirects the codec design).

Run: /tmp/reeborn-venv/bin/python research/reeborn/exp0_entropy.py
"""
import json
import struct
import mmap
import numpy as np

PATH = "models/downloads/llama-3.2-1b-instruct-unsloth/model.safetensors"
BLOCK = 32            # quant block size (matches spissa q8_0 block)
SCALE_BITS = 16       # fp16 scale per block -> SCALE_BITS/BLOCK bits/weight overhead


def H(counts):
    """Shannon entropy (bits) of a histogram."""
    c = np.asarray(counts, dtype=np.float64)
    t = c.sum()
    if t <= 0:
        return 0.0
    p = c[c > 0] / t
    return float(-(p * np.log2(p)).sum())


def bf16_to_f32(u16):
    """Exact bf16 -> f32 widening (bf16 is the top 16 bits of f32)."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def quantize(f32, levels):
    """Per-block symmetric absmax quant. Returns int codes in [-levels, levels],
    in the original element order. levels: q4 -> 7, q3 -> 3."""
    n = f32.size
    pad = (-n) % BLOCK
    flat = np.concatenate([f32, np.zeros(pad, np.float32)]) if pad else f32
    blk = flat.reshape(-1, BLOCK)
    scale = np.abs(blk).max(axis=1) / levels
    scale[scale == 0] = 1.0  # all-zero block -> codes all 0
    codes = np.rint(blk / scale[:, None]).astype(np.int32)
    np.clip(codes, -levels, levels, out=codes)
    return codes.reshape(-1)[:n]


def cat_of(name):
    if "embed_tokens" in name:
        return "embed"
    return "linear"


def new_acc():
    return dict(
        h16=np.zeros(65536, np.int64),
        q4=np.zeros(15, np.int64),   # codes -7..7  -> +7 -> 0..14
        q3=np.zeros(7, np.int64),    # codes -3..3  -> +3 -> 0..6
        jl=np.zeros(15 * 15, np.int64),  # q4 joint (left_prev, cur)
        ju=np.zeros(15 * 15, np.int64),  # q4 joint (up_prev,   cur)
        nw=0,
    )


def main():
    with open(PATH, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    data_start = 8 + hlen
    fh = open(PATH, "rb")  # keep handle alive for the mmap
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    acc = {"linear": new_acc(), "embed": new_acc()}
    n_norm = 0

    for name, info in header.items():
        if info["dtype"] != "BF16":
            continue
        shape = info["shape"]
        if len(shape) == 1:          # norms: tiny, different stats -> excluded
            n_norm += 1
            continue
        s, e = info["data_offsets"]
        buf = np.frombuffer(mm, dtype=np.uint16, count=(e - s) // 2,
                            offset=data_start + s)
        A = acc[cat_of(name)]
        A["h16"] += np.bincount(buf, minlength=65536)
        f32 = bf16_to_f32(buf)
        A["nw"] += f32.size

        c4 = quantize(f32, 7)
        A["q4"] += np.bincount((c4 + 7).astype(np.int64), minlength=15)
        c3 = quantize(f32, 3)
        A["q3"] += np.bincount((c3 + 3).astype(np.int64), minlength=7)

        R, C = shape
        g = (c4 + 7).reshape(R, C).astype(np.int64)
        pl, cl = g[:, :-1].ravel(), g[:, 1:].ravel()     # left neighbour (within row)
        A["jl"] += np.bincount(pl * 15 + cl, minlength=225)
        pu, cu = g[:-1, :].ravel(), g[1:, :].ravel()     # up neighbour (across rows)
        A["ju"] += np.bincount(pu * 15 + cu, minlength=225)

    # 'all' = linear + embed
    allc = new_acc()
    for k in ("h16", "q4", "q3", "jl", "ju", "nw"):
        allc[k] = acc["linear"][k] + acc["embed"][k]

    idx = np.arange(65536)
    sign_sel, exp_sel, mant_sel = idx >> 15, (idx >> 7) & 0xFF, idx & 0x7F

    def cond(joint):
        j = joint.reshape(15, 15)
        return H(j) - H(j.sum(axis=1)), H(j.sum(axis=0))  # H(cur|prev), H(cur)

    print(f"\nLlama-3.2-1B  |  norms excluded: {n_norm}  |  block={BLOCK}, "
          f"scale overhead = {SCALE_BITS/BLOCK:.3f} b/w\n")
    print(f"{'category':<8} {'Mweights':>9} | {'bf16_H':>7} {'sign':>5} {'exp':>5} "
          f"{'mant':>5} | {'q4_H':>5} {'q4+s':>5} {'q3_H':>5} {'q3+s':>5} | "
          f"{'q4|left':>8} {'q4|up':>7}")
    print("-" * 104)
    for label, A in (("linear", acc["linear"]), ("embed", acc["embed"]),
                     ("ALL", allc)):
        h16 = A["h16"]
        bf = H(h16)
        sH = H(np.bincount(sign_sel, weights=h16, minlength=2))
        eH = H(np.bincount(exp_sel, weights=h16, minlength=256))
        mH = H(np.bincount(mant_sel, weights=h16, minlength=128))
        q4 = H(A["q4"])
        q3 = H(A["q3"])
        ov = SCALE_BITS / BLOCK
        hl, _ = cond(A["jl"])
        hu, _ = cond(A["ju"])
        print(f"{label:<8} {A['nw']/1e6:>9.1f} | {bf:>7.3f} {sH:>5.2f} {eH:>5.2f} "
              f"{mH:>5.2f} | {q4:>5.2f} {q4+ov:>5.2f} {q3:>5.2f} {q3+ov:>5.2f} | "
              f"{hl:>8.2f} {hu:>7.2f}")

    print("\nlegend: bf16_H = order-0 entropy of 16-bit weights (true-lossless floor).")
    print("        q4_H/q3_H = code entropy; +s adds fp16 scale overhead (eff. rate).")
    print("        q4|left, q4|up = H(code | neighbour) -> lever-#2 (spatial) upside.")


if __name__ == "__main__":
    main()
