"""Isolate _Buf construction (allocate + upload) cost."""
import os
import sys
import time

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, ROOT)
import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import _Buf  # noqa: E402

H = 4096
S = 12


def main():
    torch.manual_seed(0)
    tvk.init(0)
    t = torch.randn(S, H)

    for _ in range(5):
        _Buf(t)
    ts = []
    for _ in range(50):
        t0 = time.time()
        _Buf(t)
        ts.append(time.time() - t0)
    ts.sort()
    print(f"_Buf(12x4096) alloc+upload : {ts[25]*1000:7.3f} ms")

    o = _Buf.out((S, H))
    ts = []
    for _ in range(50):
        t0 = time.time()
        _Buf.out((S, H))
        ts.append(time.time() - t0)
    ts.sort()
    print(f"_Buf.out(12x4096)         : {ts[25]*1000:7.3f} ms")

    b = _Buf(t)
    ts = []
    for _ in range(50):
        t0 = time.time()
        b.cpu()
        ts.append(time.time() - t0)
    ts.sort()
    print(f".cpu() download            : {ts[25]*1000:7.3f} ms")


if __name__ == "__main__":
    main()
