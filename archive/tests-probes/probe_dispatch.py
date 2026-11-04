"""Measure pure dispatch overhead: same op, same buffers, repeated."""
import os
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, ROOT)
import torch_vulkan_ext as tvk  # noqa: E402

M, K, N = 4, 4096, 4096


def main():
    torch.manual_seed(0)
    tvk.init(0)
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    ab = tvk.VkBuffer(a.numel() * 4, "device"); ab.upload(a)
    bb = tvk.VkBuffer(b.numel() * 4, "device"); bb.upload(b)
    ob = tvk.VkBuffer(M * N * 4, "device")
    push = tvk.u32(M) + tvk.u32(K) + tvk.u32(N)
    gx, gy = (N + 15) // 16, (M + 15) // 16

    # warm
    for _ in range(5):
        tvk.matmul(ab, bb, ob, M, K, N)
    for _ in range(5):
        tvk.matmul(ab, bb, ob, M, K, N, sync=False)
    tvk.submit()

    ts = []
    for _ in range(20):
        t0 = time.time()
        tvk.matmul(ab, bb, ob, M, K, N)
        ts.append(time.time() - t0)
    ts.sort()
    print(f"matmul 4x4096x4096 sync : {ts[10]*1000:7.3f} ms")

    ts = []
    for _ in range(20):
        t0 = time.time()
        for _ in range(50):
            tvk.matmul(ab, bb, ob, M, K, N, sync=False)
        tvk.submit()
        ts.append((time.time() - t0) / 50)
    ts.sort()
    print(f"matmul 4x4096x4096 async: {ts[10]*1000:7.3f} ms")

    # tiny op to isolate fixed overhead
    xb = tvk.VkBuffer(4096 * 4, "device")
    ob2 = tvk.VkBuffer(4096 * 4, "device")
    p2 = tvk.u32(4096) + tvk.u32(4096) + tvk.f32(1e-6)
    for _ in range(5):
        tvk.rmsnorm(xb, xb, ob2, 1, 4096, 1e-6)
    ts = []
    for _ in range(50):
        t0 = time.time()
        tvk.rmsnorm(xb, xb, ob2, 1, 4096, 1e-6)
        ts.append(time.time() - t0)
    ts.sort()
    print(f"rmsnorm 1x4096 sync     : {ts[25]*1000:7.3f} ms")


if __name__ == "__main__":
    main()
