"""Benchmark: synchronous (one host sync per op) vs batched async dispatch."""
import os
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, ROOT)
import torch_vulkan_ext as tvk  # noqa: E402

M, K, N = 4, 4096, 4096


def build():
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    ab = tvk.VkBuffer(a.numel() * 4, "device"); ab.upload(a)
    bb = tvk.VkBuffer(b.numel() * 4, "device"); bb.upload(b)
    ob = tvk.VkBuffer(M * N * 4, "device")
    ref = a @ b
    return ab, bb, ob, ref


def main():
    torch.manual_seed(0)
    tvk.init(0)
    ab, bb, ob, ref = build()
    push = tvk.u32(M) + tvk.u32(K) + tvk.u32(N)
    gx = (N + 15) // 16
    gy = (M + 15) // 16

    # warmup
    tvk.matmul(ab, bb, ob, M, K, N)
    got = ob.download((M, N))
    print(f"correctness err: {float((got - ref).abs().max()):.3e}\n")

    for label, sync in (("sync (per-op host wait)", True), ("async (batched)", False)):
        for _ in range(3):
            if sync:
                for _ in range(50):
                    tvk.matmul(ab, bb, ob, M, K, N)
            else:
                for _ in range(50):
                    tvk.matmul(ab, bb, ob, M, K, N, sync=False)
                tvk.submit()
        ts = []
        for _ in range(10):
            t0 = time.time()
            if sync:
                for _ in range(50):
                    tvk.matmul(ab, bb, ob, M, K, N)
            else:
                for _ in range(50):
                    tvk.matmul(ab, bb, ob, M, K, N, sync=False)
                tvk.submit()
            ts.append((time.time() - t0) / 50)
        ts.sort()
        print(f"{label:26s} median {ts[5]*1000:7.3f} ms/op")


if __name__ == "__main__":
    main()
