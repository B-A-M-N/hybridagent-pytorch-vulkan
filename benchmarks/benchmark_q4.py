"""Benchmark the Q4 matmul kernel against the fp32 tiled matmul.

The point of Q4 is not just VRAM: a 4x smaller weight means 4x less weight
traffic per multiply, which should show up directly in kernel time.

Measures the same shape both ways so the comparison is like-for-like.
"""
import os
import struct
import sys
import time

import numpy as np
import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk
from backend.hybrid_offload import _Buf
from backend.quantize_q4 import quantize_weight

SHAPES = [
    # (M, K, N, label)
    (8, 4096, 4096, "q_proj  M=8"),
    (8, 4096, 1024, "k_proj  M=8"),
    (8, 4096, 12288, "gate_up M=8"),
    (8, 12288, 4096, "down    M=8"),
    (512, 4096, 4096, "q_proj  M=512"),
    (512, 4096, 12288, "gate_up M=512"),
    (1, 4096, 4096, "q_proj  M=1 (decode)"),
    (1, 4096, 12288, "gate_up M=1 (decode)"),
]


def time_fp32(a, w, b, M, K, N, reps=20):
    ab, wb, bb = _Buf(a), _Buf(w), _Buf(b)
    ob = tvk.VkBuffer(max(1, M * N) * 4, "host")
    for _ in range(3):
        tvk.matmul_t_bias(ab.buf, wb.buf, bb.buf, ob, M, K, N)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        tvk.matmul_t_bias(ab.buf, wb.buf, bb.buf, ob, M, K, N)
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2]


def time_q4(a, packed, M, K, N, group_size=128, reps=20):
    ab = _Buf(a)
    raw = np.frombuffer(packed, dtype=np.uint8).astype(np.uint8)
    pad = (-len(raw)) % 4
    if pad:
        raw = np.concatenate([raw, np.zeros(pad, dtype=np.uint8)])
    nbytes = len(raw)
    fb = torch.from_numpy(np.frombuffer(raw.tobytes(), dtype=np.float32).copy())
    wb = tvk.VkBuffer(nbytes, "device")
    wb.upload(fb)
    out = tvk.VkBuffer(max(1, M * N) * 4, "host")
    tiles_x = (N + 15) // 16
    tiles_y = (M + 15) // 16
    push = (struct.pack("<I", M) + struct.pack("<I", K) + struct.pack("<I", N) +
            struct.pack("<I", tiles_x) + struct.pack("<I", group_size))
    for _ in range(3):
        tvk.dispatch("matmul_q4", [ab.buf, wb, out], push,
                     tiles_x * tiles_y, 1, 1)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        tvk.dispatch("matmul_q4", [ab.buf, wb, out], push,
                     tiles_x * tiles_y, 1, 1)
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    tvk.init(0)
    torch.manual_seed(0)
    print(f"{'shape':24s} {'fp32 ms':>10s} {'q4 ms':>10s} {'speedup':>9s} {'bytes':>12s}")
    print("-" * 70)
    for M, K, N, label in SHAPES:
        a = torch.randn(M, K)
        w = torch.randn(N, K) * 0.02
        b = torch.zeros(N)
        t32 = time_fp32(a, w, b, M, K, N)
        packed = quantize_weight(w)
        t4 = time_q4(a, packed, M, K, N)
        print(f"{label:24s} {t32*1000:10.2f} {t4*1000:10.2f} "
              f"{t32/t4:8.2f}x {len(packed)/1e6:11.1f}M")


if __name__ == "__main__":
    main()
