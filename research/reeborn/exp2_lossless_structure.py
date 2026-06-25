#!/usr/bin/env python3
"""REEBORN Experiment 2 - true-lossless structure hunt (the ratio gamble).

REEFORM (docs/reeform-phase0-results.md) already found standalone raw-weight true-lossless
WALLED at order-0 10.54 b/w on SmolLM2 + Qwen2.5, with every structural lever NULL. This
experiment (a) reproduces the wall on Llama-3.2-1B (a model REEFORM did not test), and
(b) probes levers REEFORM did NOT explicitly try on raw weights:

  - exp | (left, up)   2D joint context
  - exp | colclass8    coarse 8-bucket per-output-channel magnitude class (BIAS-SAFE version
                       of REEFORM's per-column lever, which their high-context-count test
                       could not separate from finite-sample bias)
  - exp | rowclass8    coarse 8-bucket per-input-dim magnitude class
  - mant | exp         is the mantissa less random for some magnitudes?
  - mant | left, sign | left

CRITICAL: every conditional entropy is reported BOTH plugin AND Miller-Madow bias-corrected.
Only a gain that SURVIVES the correction counts as real structure (the exact trap REEFORM
fell into and corrected). A surviving gain => REEBORN beats REEFORM on ratio; none => the
wall is confirmed for Llama-arch and we pivot to the speed axis.

Run: /tmp/reeborn-venv/bin/python research/reeborn/exp2_lossless_structure.py
"""
import json
import struct
import mmap
import math
import numpy as np

PATH = "models/downloads/llama-3.2-1b-instruct-unsloth/model.safetensors"
LN2 = math.log(2.0)
REAL_THRESHOLD = 0.01  # bits/weight; gain below this (after MM) is treated as null


def entropy_mm(counts):
    """Marginal entropy: (plugin, Miller-Madow-corrected), in bits."""
    c = np.asarray(counts, dtype=np.float64)
    N = c.sum()
    if N <= 0:
        return 0.0, 0.0
    nz = c > 0
    Hp = -(c[nz] * np.log2(c[nz] / N)).sum() / N
    m = int(nz.sum())
    return Hp, Hp + (m - 1) / (2 * N * LN2)


def cond_entropy_mm(joint):
    """H(X|C) from a joint count matrix [n_ctx, n_sym]: (plugin, MM-corrected), bits."""
    j = np.asarray(joint, dtype=np.float64)
    N = j.sum()
    if N <= 0:
        return 0.0, 0.0
    nz = j > 0
    H_joint = -(j[nz] * np.log2(j[nz] / N)).sum() / N
    ctx = j.sum(axis=1)
    cnz = ctx > 0
    H_ctx = -(ctx[cnz] * np.log2(ctx[cnz] / N)).sum() / N
    Hp = H_joint - H_ctx
    bias = (int(nz.sum()) - int(cnz.sum())) / (2 * N * LN2)  # conditional MM term
    return Hp, Hp + bias


def klass(values, k=8):
    """Bucket a 1-D array into k coarse quantile classes 0..k-1 (bias-safe context)."""
    qs = np.quantile(values, [(i + 1) / k for i in range(k - 1)])
    return np.searchsorted(qs, values).astype(np.int64)


def accumulate(header, mm, data_start):
    """One pass over every 2-D bf16 tensor; return all marginal + joint histograms."""
    a = {k: np.zeros(n, np.int64) for k, n in {
        "u16": 65536, "sign": 2, "exp": 256, "mant": 128,
        "exp_left": 256 * 256, "exp_up": 256 * 256, "exp_leftup": 256 ** 3,
        "exp_col": 8 * 256, "exp_row": 8 * 256,
        "mant_exp": 256 * 128, "mant_left": 128 * 128, "sign_left": 4,
    }.items()}
    for info in header.values():
        if info["dtype"] != "BF16" or len(info["shape"]) == 1:
            continue
        s, e = info["data_offsets"]
        buf = np.frombuffer(mm, dtype=np.uint16, count=(e - s) // 2,
                            offset=data_start + s)
        R, C = info["shape"]
        exp = ((buf >> 7) & 0xFF).astype(np.int64)
        mant = (buf & 0x7F).astype(np.int64)
        sign = (buf >> 15).astype(np.int64)
        a["u16"] += np.bincount(buf, minlength=65536)
        a["sign"] += np.bincount(sign, minlength=2)
        a["exp"] += np.bincount(exp, minlength=256)
        a["mant"] += np.bincount(mant, minlength=128)
        e2 = exp.reshape(R, C)
        a["exp_left"] += np.bincount(e2[:, :-1].ravel() * 256 + e2[:, 1:].ravel(), minlength=65536)
        a["exp_up"] += np.bincount(e2[:-1, :].ravel() * 256 + e2[1:, :].ravel(), minlength=65536)
        left, up, cur = e2[1:, :-1].ravel(), e2[:-1, 1:].ravel(), e2[1:, 1:].ravel()
        a["exp_leftup"] += np.bincount((left * 256 + up) * 256 + cur, minlength=256 ** 3)
        col_cls = np.broadcast_to(klass(e2.mean(0)), (R, C)).ravel()
        row_cls = np.broadcast_to(klass(e2.mean(1))[:, None], (R, C)).ravel()
        a["exp_col"] += np.bincount(col_cls * 256 + exp, minlength=8 * 256)
        a["exp_row"] += np.bincount(row_cls * 256 + exp, minlength=8 * 256)
        a["mant_exp"] += np.bincount(exp * 128 + mant, minlength=256 * 128)
        mt = mant.reshape(R, C)
        a["mant_left"] += np.bincount(mt[:, :-1].ravel() * 128 + mt[:, 1:].ravel(), minlength=128 * 128)
        sg = sign.reshape(R, C)
        a["sign_left"] += np.bincount(sg[:, :-1].ravel() * 2 + sg[:, 1:].ravel(), minlength=4)
    return a


def report(a):
    _, Hsign = entropy_mm(a["sign"])
    _, Hexp = entropy_mm(a["exp"])
    _, Hmant = entropy_mm(a["mant"])
    _, Hu16 = entropy_mm(a["u16"])
    field_sum = Hsign + Hexp + Hmant
    print(f"\nLlama-3.2-1B raw bf16 (norms excluded).")
    print(f"  TRUE order-0 floor  H(u16 symbol) = {Hu16:.4f} b/w   <- what REEFORM/rANS reach")
    print(f"  fields: sign {Hsign:.4f}  exp {Hexp:.4f}  mant {Hmant:.4f}  (sum {field_sum:.4f})")
    print(f"  field_sum - H(u16) = {field_sum - Hu16:.4f} = I(exp;mant), ALREADY captured by")
    print(f"  coding the joint u16 symbol -> NOT a new lever (this is the mant|exp 'gain').\n")
    # kind 'within' = inside the u16 order-0 floor (subsumed); 'cross' = conditions on
    # OTHER weights, so it can genuinely go below the per-symbol order-0 floor.
    levers = [
        ("exp | left",      a["exp_left"].reshape(256, 256),         Hexp,  "cross"),
        ("exp | up",        a["exp_up"].reshape(256, 256),           Hexp,  "cross"),
        ("exp | (left,up)",  a["exp_leftup"].reshape(256 * 256, 256), Hexp,  "cross"),
        ("exp | colclass8",  a["exp_col"].reshape(8, 256),           Hexp,  "cross"),
        ("exp | rowclass8",  a["exp_row"].reshape(8, 256),           Hexp,  "cross"),
        ("mant | left",      a["mant_left"].reshape(128, 128),       Hmant, "cross"),
        ("sign | left",      a["sign_left"].reshape(2, 2),           Hsign, "cross"),
        ("mant | exp",       a["mant_exp"].reshape(256, 128),        Hmant, "within"),
    ]
    print(f"{'lever':<18} {'kind':>7} {'H_MM':>9} {'marg':>8} {'gain_MM':>8}  verdict")
    print("-" * 74)
    best_cross = 0.0
    for label, joint, marg, kind in levers:
        _, Hmm = cond_entropy_mm(joint)
        gain = marg - Hmm
        real = gain > REAL_THRESHOLD
        if kind == "cross" and real:
            best_cross = max(best_cross, gain)
        tag = ("SUBSUMED by u16" if kind == "within"
               else ("*** real ***" if real else "null"))
        print(f"{label:<18} {kind:>7} {Hmm:>9.4f} {marg:>8.4f} {gain:>8.4f}  {tag}")
    achievable = Hu16 - best_cross
    pct = 100 * best_cross / Hu16
    print(f"\nbest cross-weight gain below the u16 floor = {best_cross:.4f} b/w "
          f"-> achievable {achievable:.4f} vs floor {Hu16:.4f} ({pct:.2f}%)")
    verdict = ("WALL CONFIRMED" if best_cross <= REAL_THRESHOLD
               else f"WALL effectively holds (only a {pct:.2f}% marginal lever)"
               if best_cross < 0.05 else "GENUINE CRACK - quantify + validate held-out")
    print(f"verdict: {verdict}")


def main():
    with open(PATH, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    fh = open(PATH, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    report(accumulate(header, mm, 8 + hlen))


if __name__ == "__main__":
    main()
