"""
Verify the matmul_t_bias test harness itself before trusting its verdict.

The tiled rewrite was reverted because this test reported errors, but the test
was built with `torch.arange(...)/10` inputs, which makes it easy to mistake a
plausible-looking wrong answer for a real one. This script prints the raw
matrices so the failure mode can be read directly.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, ROOT)
import torch_vulkan_ext as tvk  # noqa: E402


def main():
    torch.manual_seed(0)
    tvk.init(0)

    for M, K, N in ((4, 8, 4), (2, 4, 2), (1, 1, 1), (16, 16, 16), (4, 4096, 4096)):
        torch.manual_seed(1)
        x = torch.randn(M, K)
        w = torch.randn(N, K) * 0.1
        b = torch.zeros(N)
        ref = x @ w.t() + b

        xb = tvk.VkBuffer(x.numel() * 4, "device"); xb.upload(x)
        wb = tvk.VkBuffer(w.numel() * 4, "device"); wb.upload(w)
        bb = tvk.VkBuffer(b.numel() * 4, "device"); bb.upload(b)
        ob = tvk.VkBuffer(M * N * 4, "host")
        tvk.matmul_t_bias(xb, wb, bb, ob, M, K, N)
        got = ob.download((M, N))
        err = float((got - ref).abs().max())
        print(f"M={M:3d} K={K:5d} N={N:5d}  err={err:.3e}  "
              f"{'OK' if err < 1e-3 else 'FAIL'}")

    # print the tiny case in full
    M, K, N = 4, 8, 4
    torch.manual_seed(1)
    x = torch.randn(M, K)
    w = torch.randn(N, K) * 0.1
    ref = x @ w.t()
    xb = tvk.VkBuffer(x.numel() * 4, "device"); xb.upload(x)
    wb = tvk.VkBuffer(w.numel() * 4, "device"); wb.upload(w)
    bb = tvk.VkBuffer(N * 4, "device"); bb.upload(torch.zeros(N))
    ob = tvk.VkBuffer(M * N * 4, "host")
    tvk.matmul_t_bias(xb, wb, bb, ob, M, K, N)
    print("\ntiny case ref:\n", ref.tolist())
    print("tiny case got:\n", ob.download((M, N)).tolist())


if __name__ == "__main__":
    main()
