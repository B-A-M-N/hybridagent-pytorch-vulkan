"""Measure the pure dispatch cost vs GPU execution time.

If the GPU work is small relative to the per-op dispatch overhead, batching is
the right fix. If the GPU work dominates, batching will not help and the kernel
itself needs work.
"""
import os
import struct
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, ROOT)
import torch_vulkan_ext as tvk  # noqa: E402

M, K, N = 12, 4096, 4096


def main():
    torch.manual_seed(0)
    tvk.init(0)
    x = torch.randn(M, K)
    w = torch.randn(N, K) * 0.02
    b = torch.zeros(N)
    xb = tvk.VkBuffer(x.numel() * 4, "device"); xb.upload(x)
    wb = tvk.VkBuffer(w.numel() * 4, "device"); wb.upload(w)
    bb = tvk.VkBuffer(b.numel() * 4, "device"); bb.upload(b)
    ob = tvk.VkBuffer(M * N * 4, "host")

    # warm
    for _ in range(5):
        tvk.matmul_t_bias(xb, wb, bb, ob, M, K, N)
    for _ in range(5):
        tvk.matmul_t_bias(xb, wb, bb, ob, M, K, N, sync=False)
        tvk.submit()

    # sync: full round trip per op
    ts = []
    for _ in range(20):
        t0 = time.time()
        tvk.matmul_t_bias(xb, wb, bb, ob, M, K, N)
        ts.append(time.time() - t0)
    ts.sort()
    sync_ms = ts[10] * 1000

    # async: N dispatches then one submit
    for reps in (1, 10, 50):
        ts = []
        for _ in range(10):
            t0 = time.time()
            for _ in range(reps):
                tvk.matmul_t_bias(xb, wb, bb, ob, M, K, N, sync=False)
            tvk.submit()
            ts.append((time.time() - t0) / reps)
        ts.sort()
        print(f"async x{reps:3d}: {ts[5]*1000:7.3f} ms/op")

    print(f"sync     : {sync_ms:7.3f} ms/op")

    # GPU-side time via a long chain: 500 dispatches, one submit
    t0 = time.time()
    for _ in range(500):
        tvk.matmul_t_bias(xb, wb, bb, ob, M, K, N, sync=False)
    tvk.submit()
    gpu_ms = (time.time() - t0) / 500 * 1000
    print(f"500-chain: {gpu_ms:7.3f} ms/op (pure GPU + queue)")


if __name__ == "__main__":
    main()
