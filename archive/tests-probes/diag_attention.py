"""
Step 2: find the FIRST divergent intermediate value in attention2.

Runs the diagnostic shader (attn_dbg) which mirrors attention2's addressing and
dumps, per (head, query row, key):
    slot 0  raw dot product q.k          <- step 1 (Q x K^T)
    slot 1  scaled + masked score
    slot 2  exp(score - rowmax)
    slot 3  normalised weight            <- step 2 (softmax/mask)
    slot 4  q offset used
    slot 5  k offset used
    slot 6  q value at lane 0
    slot 7  k value at lane 0

Then compares each against an independent PyTorch FP32 reference and reports the
first divergence.
"""
import os
import struct
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import _Buf  # noqa: E402

NH, NKV, HD = 32, 8, 128
SCALING = 1.0 / (HD ** 0.5)
S = 2


def main():
    tvk.init(0)
    torch.manual_seed(0)

    # Deliberate inputs: q row s is axis-aligned on axis s, k row s scaled on
    # axis s, so the raw score for key m is 4.0 when m == s and 0.0 otherwise.
    q = torch.zeros(NH, S, HD)
    k = torch.zeros(NKV, S, HD)
    for s in range(S):
        q[:, s, s] = 1.0
        k[:, s, s] = 4.0
    allow = torch.arange(S).unsqueeze(0) <= torch.arange(S).unsqueeze(1)
    mask = torch.zeros(NH, S, S).masked_fill(~allow, float("-inf"))

    # ---- PyTorch FP32 reference for each intermediate
    kh = k.repeat_interleave(NH // NKV, dim=0)
    ref_raw = torch.matmul(q, kh.transpose(-1, -2))      # raw q.k
    ref_scaled = ref_raw * SCALING + mask
    ref_w = torch.softmax(ref_scaled, dim=-1)

    # ---- run the diagnostic shader
    dbg = tvk.VkBuffer(NH * S * S * 8 * 4, "host")
    kb = _Buf(k.contiguous())
    mb = _Buf(mask.contiguous())
    push = (struct.pack("<I", 0) + struct.pack("<I", S) + struct.pack("<I", NH) +
            struct.pack("<I", NH // NKV) + struct.pack("<I", HD) +
            struct.pack("<I", S) + struct.pack("<I", S) +
            struct.pack("<f", SCALING))
    tvk.dispatch("attn_dbg", [_Buf(q.contiguous()).buf, kb.buf, mb.buf, dbg], push,
                 1, NH, 1)
    raw = dbg.download((NH, S, S, 8))

    print("=== attention2 intermediate diagnosis (S=2, head 0) ===\n")
    print(f"{'slot':<28}{'GPU':>16}{'CPU ref':>16}{'match':>8}")
    print("-" * 70)

    h = 0
    for s in range(S):
        for m in range(S):
            base = raw[h, s, m]
            gpu_raw = float(base[0])
            gpu_scaled = float(base[1])
            gpu_exp = float(base[2])
            gpu_w = float(base[3])
            cpu_raw = float(ref_raw[h, s, m])
            cpu_scaled = float(ref_scaled[h, s, m])
            cpu_w = float(ref_w[h, s, m])
            cpu_exp = float(torch.exp(ref_scaled[h, s, m] - ref_scaled[h, s].max()))

            checks = [
                ("raw q.k", gpu_raw, cpu_raw),
                ("scaled+masked", gpu_scaled, cpu_scaled),
                ("exp(s-max)", gpu_exp, cpu_exp),
                ("weight", gpu_w, cpu_w),
            ]
            print(f"--- row {s} key {m} ---")
            for name, g, c in checks:
                ok = abs(g - c) < 1e-3
                print(f"{name:<28}{g:>16.6f}{c:>16.6f}{'OK' if ok else 'DIFF':>8}")
            print(f"{'q offset':<28}{float(base[4]):>16.0f}")
            print(f"{'k offset':<28}{float(base[5]):>16.0f}")
            print(f"{'q[off]':<28}{float(base[6]):>16.6f}")
            print(f"{'k[off]':<28}{float(base[7]):>16.6f}")
            print()

    # ---- verdict: which stage diverges first
    print("=== first divergence ===")
    for s in range(S):
        for m in range(S):
            b = raw[h, s, m]
            if abs(float(b[0]) - float(ref_raw[h, s, m])) > 1e-3:
                print(f"STAGE 1 (Q x K^T) diverges at row {s} key {m}")
                return 1
    for s in range(S):
        for m in range(S):
            b = raw[h, s, m]
            if abs(float(b[1]) - float(ref_scaled[h, s, m])) > 1e-3:
                print(f"STAGE 2a (scale+mask) diverges at row {s} key {m}")
                return 2
    for s in range(S):
        for m in range(S):
            b = raw[h, s, m]
            if abs(float(b[3]) - float(ref_w[h, s, m])) > 1e-3:
                print(f"STAGE 2b (softmax) diverges at row {s} key {m}")
                return 3
    print("all intermediates match -- defect is in stage 3 (x V) or output write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
