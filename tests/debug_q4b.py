"""Test the Q4 byte-addressing path in isolation with a known byte pattern."""
import os
import struct
import sys

import numpy as np
import torch

ROOT = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan"
sys.path.insert(0, os.path.join(ROOT, "backend", "torch_vulkan_ext"))
sys.path.insert(0, ROOT)

import torch_vulkan_ext as tvk

tvk.init(0)

# A weight buffer whose bytes are 0,1,2,3,...,N-1 so any addressing error
# shows up as a wrong value. K=16, group_size=16 -> group_bytes=10.
K = 16
group_size = 16
group_bytes = group_size // 2 + 2   # 10
N = 1

# Build the expected scale/nibbles for a known input.
w = torch.zeros(N, K)
w[0, 0] = 8.0   # nibble should be round(8/scale)+8 with scale = 8/7
scale = 8.0 / 7.0
import struct as _s
scale16 = _s.unpack('<e', _s.pack('<e', scale))[0]
q = [0] * K
q[0] = int(round(8.0 / scale16)) + 8
print(f"scale={scale16:.6f} q[0]={q[0]}")

# Pack by hand: bytes 0..1 = fp16 scale, bytes 2..9 = 16 nibbles in 8 bytes
packed = bytearray()
packed += _s.pack('<e', scale16)
for j in range(0, K, 2):
    lo = q[j] & 0x0F
    hi = q[j + 1] & 0x0F
    packed.append(lo | (hi << 4))
packed = bytes(packed)
print("packed:", packed.hex(), f"({len(packed)} bytes)")

# Upload and read back through the kernel's byte_at logic by using a probe
# shader -- instead, verify the byte extraction in Python against the shader's
# formula: byte_at(i) = (w[i/4] >> ((i%4)*8)) & 0xFF
words = np.frombuffer(packed, dtype=np.uint8).astype(np.uint32)
pad = (-len(words)) % 4
if pad:
    words = np.concatenate([words, np.zeros(pad, dtype=np.uint32)])
arr = np.frombuffer(packed + b'\x00' * ((-len(packed)) % 4), dtype=np.uint32)
print("words:", [hex(int(x)) for x in arr])
for i in range(len(packed)):
    word = int(arr[i // 4])
    got = (word >> ((i % 4) * 8)) & 0xFF
    print(f"  byte_at({i:2d}) = {got:#04x}   expected {packed[i]:#04x}   "
          f"{'OK' if got == packed[i] else 'MISMATCH'}")
