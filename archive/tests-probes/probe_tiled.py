"""Benchmark the tiled matmul_t_bias against the measured naive baseline."""
import os
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, ROOT)
import torch_vulkan_ext as tvk  # noqa: E402


def main():
    torch.manual_seed(0)
    tvk.init(0)

    # q_proj shape: M=12 tokens, K=4096, N=4096
    for M, K, N in ((12, 4096, 4096), (12, 4096, 12288), (1, 4096, 4096)):
        x = torch.randn(M, K)
        w = torch.randn(N, K) * 0.02
        b = torch.zeros(N)
        ref = x @ w.t()

        xb = tvk.VkBuffer(x.numel() * 4, "device"); xb.upload(x)
        wb = tvk.VkBuffer(w.numel() * 4, "device"); wb.upload(w)
        bb = tvk.VkBuffer(b.numel() * 4, "device"); bb.upload(b)
        ob = tvk.VkBuffer(M * N * 4, "host")

        tvk.matmul_t_bias(xb, wb, bb, ob, M, K, N)
        err = float((ob.download((M, N)) - ref).abs().max())

        # warm
        for _ in range(3):
            tvk.matmul_t_bias(xb, wb, bb, ob, M, K, N)
        ts = []
        for _ in range(10):
            t0 = time.time()
            tvk.matmul_t_bias(xb, wb, bb, ob, M, K, N)
            ts.append(time.time() - t0)
        ts.sort()
        print(f"M={M:3d} K={K} N={N:6d}  {ts[5]*1000:7.2f} ms  err={err:.2e}")

    # async variant
    M, K, N = 12, 4096, 4096
    x = torch.randn(M, K)
    w = torch.randn(N, K) * 0.02
    xb = tvk.VkBuffer(x.numel() * 4, "device"); xb.upload(x)
    wb = tvk.VkBuffer(w.numel() * 4, "device"); wb.upload(w)
    bb = tvk.VkBuffer(N * 4, "device"); bb.upload(torch.zeros(N))
    ob = tvk.VkBuffer(M * N * 4, "host")
    for _ in range(3):
        tvk.matmul_t_bias(xb, wb, bb, ob, M, K, N, sync=False)
        tvk.submit()
    ts = []
    for _ in range(10):
        t0 = time.time()
        tvk.matmul_t_bias(xb, wb, bb, ob, M, K, N, sync=False)
        tvk.submit()
        ts.append(time.time() - t0)
    ts.sort()
    print(f"M={M} K={K} N={N} async        {ts[5]*1000:7.2f} ms")


if __name__ == "__main__":
    main()
