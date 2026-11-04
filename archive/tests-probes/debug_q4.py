"""Debug the Q4 byte-extraction: dump what the kernel reads vs what it should."""
import os
import struct
import sys

import numpy as np
import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk
from backend.quantize_q4 import quantize_weight

tvk.init(0)

# Tiny case: one row, K=8, group_size=8 -> group_bytes = 8/2+2 = 6 bytes
K = 8
N = 1
w = torch.randn(N, K)
packed = quantize_weight(w, group_size=8)
print(f"K={K} group_size=8 -> packed {len(packed)} bytes")
print("packed hex:", packed.hex())

# Expected: group 0 = fp16 scale + 4 bytes of nibbles
h = packed[0] | (packed[1] << 8)
sgn = (h >> 15) & 1
exp = (h >> 10) & 0x1F
man = h & 0x3FF
sc = (man | 0x400) * 2.0 ** (int(exp) - 25) if exp not in (0, 31) else man * 5.96e-8
sc = -sc if sgn else sc
print(f"expected scale: {sc:.6f}")

q = w[0].numpy() / sc
q = np.clip(np.round(q), -8, 7) + 8
print("expected nibbles:", [int(x) for x in q])

# Read back the raw bytes the kernel would see
raw = np.frombuffer(packed, dtype=np.uint8)
print("raw bytes:", [hex(int(x)) for x in raw])
print()
print("=> byte 0,1 = scale; bytes 2..5 = packed nibbles")
print("   nibble j is in byte (2 + j//2), low nibble for even j")
for j in range(K):
    byte = raw[2 + j // 2]
    nib = (byte & 0x0F) if j % 2 == 0 else (byte >> 4)
    print(f"   j={j}: byte={byte:#04x} nibble={nib} -> dequant {(nib-8)*sc:.6f} vs {w[0,j]:.6f}")
