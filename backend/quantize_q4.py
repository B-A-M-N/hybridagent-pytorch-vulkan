"""
Symmetric 4-bit weight quantization for the Vulkan backend.

Why this exists
---------------
Each transformer layer costs **771.8 MB in fp32**. On an 8.6 GB card that makes
**8 layers = 6.17 GB**, leaving ~2.4 GB for activations -- not enough at
S >= 512. Measured consequence: at 8 layers the RADV context is lost
(`logit_err 6.16`, `CS has been cancelled because the context is lost`).

At Q4 the same layer is ~96.5 MB, so 8 layers = 0.77 GB and the whole 36-layer
model would be ~3.5 GB. That is the difference between "4 layers fits" and
"the model is almost entirely GPU-resident".

Format
------
Per output row (one `nn.Linear` weight row, length K):
    group_bytes = group_size/2 + 2
    row_stride  = (K / group_size) * group_bytes
    per group:  2 bytes little-endian fp16 scale, then group_size/2 bytes of
                packed nibbles (low nibble = even index, high nibble = odd).

Quantization:  q = clamp(round(x / scale), -8, 7) + 8
Dequantization: x ~= (q - 8) * scale
"""
from __future__ import annotations

import numpy as np
import torch

GROUP_SIZE = 128


def quantize_row(row: np.ndarray, group_size: int = GROUP_SIZE) -> bytes:
    """Pack one weight row into the Q4 format."""
    K = row.shape[0]
    assert K % group_size == 0, f"K={K} not a multiple of group_size"
    out = bytearray()
    for g in range(0, K, group_size):
        blk = row[g:g + group_size]
        amax = float(np.abs(blk).max())
        scale = amax / 7.0 if amax > 0 else 1.0
        q = np.clip(np.round(blk / scale), -8, 7).astype(np.int8) + 8
        q = q.astype(np.uint8)
        # fp16 scale, little-endian
        scale16 = float(np.float16(scale))
        out += np.float16(scale16).tobytes()
        # pack nibbles: low = even index, high = odd index
        even = q[0::2] & 0x0F
        odd = q[1::2] & 0x0F
        out += (even | (odd << 4)).tobytes()
    return bytes(out)


def quantize_weight(w: torch.Tensor, group_size: int = GROUP_SIZE) -> bytes:
    """Quantize an `nn.Linear` weight of shape [N, K] into the packed format."""
    arr = w.detach().float().cpu().numpy()
    N, K = arr.shape
    rows = [quantize_row(arr[n], group_size) for n in range(N)]
    return b"".join(rows)


def dequantize_weight(packed: bytes, N: int, K: int,
                      group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Inverse of `quantize_weight`, for validation."""
    group_bytes = group_size // 2 + 2
    row_stride = (K // group_size) * group_bytes
    out = np.zeros((N, K), dtype=np.float32)
    for n in range(N):
        base = n * row_stride
        p = 0
        for g in range(K // group_size):
            gbase = base + g * group_bytes
            h = packed[gbase] | (packed[gbase + 1] << 8)
            # fp16 -> fp32
            sgn = (h >> 15) & 1
            exp = (h >> 10) & 0x1F
            man = h & 0x3FF
            if exp == 0:
                sc = man * 5.9604644775390625e-8
            elif exp == 31:
                sc = np.inf
            else:
                sc = (man | 0x400) * 2.0 ** (int(exp) - 25)
            sc = -sc if sgn else sc
            for j in range(group_size):
                byte = packed[gbase + 2 + j // 2]
                nib = (byte & 0x0F) if (j % 2 == 0) else (byte >> 4)
                out[n, g * group_size + j] = (nib - 8) * sc
    return torch.from_numpy(out)


def quantization_error(w: torch.Tensor, group_size: int = GROUP_SIZE) -> dict:
    """Measure the error Q4 introduces on a weight tensor."""
    packed = quantize_weight(w, group_size)
    deq = dequantize_weight(packed, *w.shape, group_size)
    d = (deq - w.float()).abs()
    rel = float(d.max() / (w.float().abs().max() + 1e-12))
    return {
        "max_abs_err": float(d.max()),
        "mean_abs_err": float(d.mean()),
        "relative_max": rel,
        "original_bytes": w.numel() * 4,
        "packed_bytes": len(packed),
        "ratio": len(packed) / (w.numel() * 4),
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    for name, shape in (("q_proj", (4096, 4096)), ("gate_up", (24576, 4096))):
        w = torch.randn(*shape) * 0.02
        e = quantization_error(w)
        print(f"{name} {shape}:")
        for k, v in e.items():
            print(f"   {k:15s} {v}")
        print()
