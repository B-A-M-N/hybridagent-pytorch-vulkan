"""
Verify the Vulkan Q4 matmul kernel against the Python dequantizer.

The kernel (`matmul_q4.comp`) and `backend/quantize_q4.py` are two independent
implementations of the same format. If they agree, the format is unambiguous;
if they disagree, one of them is wrong and the kernel must be treated as
unverified.
"""
import os
import sys

import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

import struct

import torch_vulkan_ext as tvk  # noqa: E402
from backend.hybrid_offload import _Buf  # noqa: E402
from backend.quantize_q4 import quantize_weight, dequantize_weight  # noqa: E402


def run_kernel(a: torch.Tensor, packed: bytes, M: int, K: int, N: int,
               group_size: int) -> torch.Tensor:
    import numpy as np
    ab = _Buf(a.contiguous())
    # `VkBuffer.upload` converts any non-float32 tensor to float32, which would
    # destroy the packed byte layout. Upload the raw bytes through a float32
    # view of the same memory instead.
    raw = np.frombuffer(packed, dtype=np.uint8).astype(np.uint8)
    pad = (-len(raw)) % 4
    if pad:
        raw = np.concatenate([raw, np.zeros(pad, dtype=np.uint8)])
    nbytes = len(raw)
    fb = torch.from_numpy(np.frombuffer(raw.tobytes(), dtype=np.float32).copy())
    assert fb.numel() * 4 == nbytes, f"{fb.numel() * 4} != {nbytes}"
    wb = tvk.VkBuffer(nbytes, "device")
    wb.upload(fb)
    out = tvk.VkBuffer(max(1, M * N) * 4, "host")
    tiles_x = (N + 15) // 16
    tiles_y = (M + 15) // 16
    push = (struct.pack("<I", M) + struct.pack("<I", K) + struct.pack("<I", N) +
            struct.pack("<I", tiles_x) + struct.pack("<I", group_size))
    tvk.dispatch("matmul_q4", [ab.buf, wb, out], push,
                 tiles_x * tiles_y, 1, 1)
    return out.download((M, N))


def main():
    import numpy as np
    global np
    torch.manual_seed(0)
    tvk.init(0)

    for (M, K, N) in ((8, 256, 256), (12, 4096, 4096)):
        a = torch.randn(M, K)
        w = torch.randn(N, K) * 0.02
        ref = a @ w.t()

        packed = quantize_weight(w)
        deq = dequantize_weight(packed, N, K)

        # 1. dequantized weights must be close to the originals
        dq_err = float((deq - w).abs().max())
        # 2. the kernel must match the dequantized matmul
        got = run_kernel(a, packed, M, K, N, 128)
        ref_deq = a @ deq.t()
        kern_err = float((got - ref_deq).abs().max())
        # 3. the kernel must be close to the exact matmul
        exact_err = float((got - ref).abs().max())

        print(f"M={M} K={K} N={N}:")
        print(f"   dequant vs original      : {dq_err:.3e}")
        print(f"   kernel vs dequant matmul : {kern_err:.3e}")
        print(f"   kernel vs exact matmul   : {exact_err:.3e}")
        ok = kern_err < 1e-3
        print(f"   {'PASS' if ok else 'FAIL'}")
        print()


if __name__ == "__main__":
    main()
