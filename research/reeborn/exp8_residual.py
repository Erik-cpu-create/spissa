#!/usr/bin/env python3
"""REEBORN E8 - the speculative lane: W = lossy baseline + LOSSLESS residual.

Agent-B flagged "W = U·V + lossless residual R, or residual vs an already-lossy q4/q3 baseline"
as apparently-unclaimed. The low-rank variant (U·V) is already disproven by REEFORM (net 11.73 >
10.5). This tests the q4-baseline twist with a DECISIVE chain-rule measurement:

  To reconstruct bf16 W exactly from a q4 baseline you must store: q4 code + block scale +
  residual (= the bf16 symbol given its q4 code). By the chain rule,
      H(q4code) + H(u16 | q4code) = H(u16) + H(q4code | u16)  >=  H(u16),
  and you ALSO pay the scale stream. So q4+residual is PROVABLY >= the direct floor H(u16)=10.55,
  i.e. it cannot beat coding W directly for STANDALONE lossless. We measure the exact numbers to
  confirm, and report the one real niche: the residual H(u16|q4code) = cost to UPGRADE an
  already-stored q4 model to exact bf16 (scalable/progressive coding).

Run: /tmp/reeborn-venv/bin/python research/reeborn/exp8_residual.py
"""
import json
import struct
import mmap
import numpy as np

PATH = "models/downloads/llama-3.2-1b-instruct-unsloth/model.safetensors"
BLOCK = 32
LEVELS = 7          # q4 symmetric: codes -7..7 (15 values)
SCALE_BW = 16 / BLOCK  # fp16 scale per block -> 0.5 b/w


def H(counts):
    c = np.asarray(counts, np.float64)
    N = c.sum()
    if N <= 0:
        return 0.0
    p = c[c > 0] / N
    return float(-(p * np.log2(p)).sum())


def q4_codes(buf):
    """Per-block symmetric absmax q4 code (0..14) for a bf16 u16 buffer."""
    f32 = (buf.astype(np.uint32) << 16).view(np.float32)
    n = f32.size
    pad = (-n) % BLOCK
    flat = np.concatenate([f32, np.zeros(pad, np.float32)]) if pad else f32
    blk = flat.reshape(-1, BLOCK)
    scale = np.abs(blk).max(axis=1) / LEVELS
    scale[scale == 0] = 1.0
    code = np.clip(np.rint(blk / scale[:, None]), -LEVELS, LEVELS).astype(np.int64)
    return (code.reshape(-1)[:n] + LEVELS)  # 0..14


def main():
    with open(PATH, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    fh = open(PATH, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    ds = 8 + hlen
    K = 2 * LEVELS + 1  # 15

    m_u16 = np.zeros(65536, np.int64)
    m_code = np.zeros(K, np.int64)
    joint = np.zeros(K * 65536, np.int64)   # (q4code, u16)
    for info in header.values():
        if info["dtype"] != "BF16" or len(info["shape"]) == 1:
            continue
        s, e = info["data_offsets"]
        buf = np.frombuffer(mm, dtype=np.uint16, count=(e - s) // 2, offset=ds + s)
        code = q4_codes(buf)
        m_u16 += np.bincount(buf, minlength=65536)
        m_code += np.bincount(code, minlength=K)
        joint += np.bincount(code * 65536 + buf.astype(np.int64), minlength=K * 65536)

    Hu16 = H(m_u16)
    Hcode = H(m_code)
    Hjoint = H(joint)
    Hu16_given_code = Hjoint - Hcode          # residual: bits to recover W given its q4 code
    Hcode_given_u16 = Hjoint - Hu16           # block-scale redundancy carried by the code

    q4_store = Hcode + SCALE_BW               # cost of the q4 baseline itself (codes + scale)
    total_lossless = q4_store + Hu16_given_code  # q4 + scale + residual = full lossless

    print(f"\nLlama-3.2-1B. Speculative lane: W = q4 baseline + lossless residual.\n")
    print(f"  direct floor   H(u16)            = {Hu16:.4f} b/w   <- code W directly (DFloat11-class)")
    print(f"  H(q4 code)                        = {Hcode:.4f} b/w")
    print(f"  H(u16 | q4 code)  [the residual]  = {Hu16_given_code:.4f} b/w")
    print(f"  H(q4 code | u16)  [scale redund.] = {Hcode_given_u16:.4f} b/w  (>0 => code carries block info)")
    print(f"  fp16 scale stream                 = {SCALE_BW:.4f} b/w\n")
    print(f"  q4 baseline store (code+scale)    = {q4_store:.4f} b/w")
    print(f"  TOTAL q4+scale+residual (lossless)= {total_lossless:.4f} b/w")
    print(f"  vs direct floor                   = {Hu16:.4f} b/w  ->  "
          f"{'WIN' if total_lossless < Hu16 - 0.01 else f'LOSES by {total_lossless-Hu16:+.4f} b/w (null, chain rule)'}\n")
    print(f"  NICHE (real): if a q4 model is ALREADY stored, upgrading it to exact bf16 costs the")
    print(f"  residual H(u16|q4)={Hu16_given_code:.2f} b/w, vs {Hu16:.2f} for a separate bf16 copy")
    print(f"  -> saves {Hu16 - Hu16_given_code:.2f} b/w in the store-BOTH case (scalable/progressive coding).")


if __name__ == "__main__":
    main()
