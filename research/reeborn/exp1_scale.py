#!/usr/bin/env python3
"""REEBORN Experiment 1 - the scale stream (lever #1).

E0 showed the fp16 per-block scale overhead (0.50 b/w at block 32) now EXCEEDS the
code-entropy savings (0.40). E1 attacks it three ways:

  (1) Block-size sweep B in {16,32,64,128,256}: bigger block -> fewer scales (less
      overhead) AND a larger per-block absmax -> codes pulled toward 0 -> lower code
      entropy, but coarser quant -> worse quality (lower SQNR). Pure bits-vs-quality.

  (2) Scale-stream compressibility: raw fp16 (16/B) vs entropy-coded fp16 scale
      (lossless re-code of the SAME scales) vs log2-scale delta-coding (smart codec
      lower bound). How far below 16/B can the scale stream go.

  (3) SQNR(dB) per config = quality cost, so the operating point is an informed choice.

Run: /tmp/reeborn-venv/bin/python research/reeborn/exp1_scale.py
"""
import json
import struct
import mmap
import math
import numpy as np

PATH = "models/downloads/llama-3.2-1b-instruct-unsloth/model.safetensors"
BLOCKS = [16, 32, 64, 128, 256]
LEVELS = {"q4": 7, "q3": 3}
LOGDELTA_STEP = 2 ** -6   # fine quant of log2-scale deltas (~near-lossless on scale)


def H(counts):
    c = np.asarray(counts, dtype=np.float64)
    t = c.sum()
    if t <= 0:
        return 0.0
    p = c[c > 0] / t
    return float(-(p * np.log2(p)).sum())


def H_dict(d):
    c = np.fromiter(d.values(), dtype=np.float64)
    return H(c)


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def quant(f32, B, levels):
    n = f32.size
    pad = (-n) % B
    flat = np.concatenate([f32, np.zeros(pad, np.float32)]) if pad else f32
    blk = flat.reshape(-1, B)
    scale = np.abs(blk).max(axis=1) / levels
    scale[scale == 0] = 1.0
    codes = np.clip(np.rint(blk / scale[:, None]), -levels, levels).astype(np.int32)
    deq = (codes * scale[:, None]).reshape(-1)[:n]
    return codes.reshape(-1)[:n], scale, deq


def main():
    with open(PATH, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    data_start = 8 + hlen
    fh = open(PATH, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    # accumulators[B][q] = dict of histos/sums
    acc = {B: {q: dict(code=np.zeros(2 * LEVELS[q] + 1, np.int64),
                       spat=np.zeros(65536, np.int64),   # fp16 scale bit-pattern
                       dlt={}, sse=0.0, sxx=0.0, nb=0, nw=0)
               for q in LEVELS} for B in BLOCKS}

    for name, info in header.items():
        if info["dtype"] != "BF16" or len(info["shape"]) == 1:
            continue
        s, e = info["data_offsets"]
        buf = np.frombuffer(mm, dtype=np.uint16, count=(e - s) // 2,
                            offset=data_start + s)
        f32 = bf16_to_f32(buf)
        sxx = float(np.dot(f32, f32))
        for B in BLOCKS:
            for q, lv in LEVELS.items():
                A = acc[B][q]
                codes, scale, deq = quant(f32, B, lv)
                A["code"] += np.bincount((codes + lv).astype(np.int64),
                                         minlength=2 * lv + 1)
                err = f32 - deq
                A["sse"] += float(np.dot(err, err))
                A["sxx"] += sxx
                A["nw"] += f32.size
                A["nb"] += scale.size
                # scale as fp16 bit-pattern (entropy-coded lossless re-code)
                sp = scale.astype(np.float16).view(np.uint16)
                A["spat"] += np.bincount(sp, minlength=65536)
                # log2-scale delta (smart codec lower bound), per-tensor
                if scale.size >= 2:
                    l = np.log2(scale)
                    dq = np.rint(np.diff(l) / LOGDELTA_STEP).astype(np.int64)
                    u, c = np.unique(dq, return_counts=True)
                    for k, v in zip(u.tolist(), c.tolist()):
                        A["dlt"][k] = A["dlt"].get(k, 0) + v

    for q in LEVELS:
        print(f"\n=== {q.upper()} (levels +/-{LEVELS[q]}) ===")
        print(f"{'B':>4} {'code_H':>7} {'scale_raw':>9} {'scale_ent':>9} "
              f"{'scale_dlt':>9} | {'total_raw':>9} {'total_best':>10} {'SQNR_dB':>8}")
        print("-" * 78)
        for B in BLOCKS:
            A = acc[B][q]
            ch = H(A["code"])
            raw = 16.0 / B
            ent = H(A["spat"]) / B
            dlt = H_dict(A["dlt"]) / B
            best_scale = min(ent, dlt)
            sqnr = 10 * math.log10(A["sxx"] / A["sse"]) if A["sse"] > 0 else float("inf")
            print(f"{B:>4} {ch:>7.3f} {raw:>9.3f} {ent:>9.3f} {dlt:>9.3f} | "
                  f"{ch+raw:>9.3f} {ch+best_scale:>10.3f} {sqnr:>8.2f}")

    print("\nlegend: code_H = q-code entropy; scale_raw = 16/B; scale_ent = entropy-coded")
    print("        fp16 scale; scale_dlt = log2-scale delta-coded; total = code+scale b/w;")
    print("        SQNR_dB = quality (higher = closer to fp). lossless is vs the q-checkpoint.")


if __name__ == "__main__":
    main()
