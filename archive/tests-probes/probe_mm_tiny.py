"""Isolate the tiled matmul_t_bias with a tiny hand-checkable case."""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, ROOT)
import torch_vulkan_ext as tvk  # noqa: E402

M, K, N = 4, 8, 4


def main():
    torch.manual_seed(0)
    tvk.init(0)
    x = torch.arange(M * K, dtype=torch.float32).reshape(M, K) / 10
    w = torch.arange(N * K, dtype=torch.float32).reshape(N, K) / 10
    b = torch.zeros(N)
    ref = x @ w.t() + b

    xb = tvk.VkBuffer(x.numel() * 4, "device"); xb.upload(x)
    wb = tvk.VkBuffer(w.numel() * 4, "device"); wb.upload(w)
    bb = tvk.VkBuffer(b.numel() * 4, "device"); bb.upload(b)
    ob = tvk.VkBuffer(M * N * 4, "host")
    tvk.matmul_t_bias(xb, wb, bb, ob, M, K, N)
    got = ob.download((M, N))
    print("ref:\n", ref.tolist())
    print("got:\n", got.tolist())
    print(f"err: {float((got - ref).abs().max()):.3e}")


if __name__ == "__main__":
    main()
