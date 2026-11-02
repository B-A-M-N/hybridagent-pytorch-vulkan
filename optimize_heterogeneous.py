#!/usr/bin/env python3
"""Optimized heterogeneous execution with persistent Vulkan weights and Q4 fix."""
import torch
import torch.nn as nn
from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer, DeviceStats, _Buf
import torch_vulkan_ext as tvk
import time
import os
import struct

# Fix Q4 quantization overflow issue
from backend.quantize_q4 import quantize_weight as orig_quantize_weight, GROUP_SIZE

def quantize_weight_fixed(w: torch.Tensor, group_size: int = GROUP_SIZE) -> bytes:
    """Fixed Q4 quantization without overflow."""
    arr = w.detach().float().cpu().numpy()
    N, K = arr.shape
    rows = []
    for n in range(N):
        row = arr[n]
        assert K % group_size == 0, f"K={K} not a multiple of group_size"
        out = bytearray()
        for g in range(0, K, group_size):
            blk = row[g:g + group_size]
            amax = float(np.abs(blk).max())
            if amax == 0:
                scale = 1.0
                q = np.zeros_like(blk, dtype=np.int8)
            else:
                scale = amax / 7.0
                q = np.clip(np.round(blk / scale), -8, 7).astype(np.int8)
            q = q.astype(np.uint8) + 8
            # fp16 scale, little-endian
            scale16 = np.float16(scale)
            out += scale16.tobytes()
            # pack nibbles: low = even index, high = odd index
            even = q[0::2] & 0x0F
            odd = q[1::2] & 0x0F
            out += (even | (odd << 4)).tobytes()
        rows.append(bytes(out))
    return b"".join(rows)

import numpy as np

# Patch VulkanDecoderLayer to use fixed quantization
import backend.hybrid_offload as hybrid_offload
hybrid_offload.quantize_weight = quantize_weight_fixed

# Also patch the upload_weights_q4 to use fixed quantization
original_upload_q4 = VulkanDecoderLayer.upload_weights_q4

def upload_weights_q4_fixed(self) -> None:
    """Upload this layer's weights in Q4 format for memory-constrained GPUs."""
    if getattr(self, "_uploaded_q4", False):
        return
    from backend.quantize_q4 import quantize_weight
    c = self.config
    h = self.hidden_size
    nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
    
    # Quantize the large linear weights
    self._w_q4 = {}
    self._w_q4["q"] = quantize_weight_fixed(self.self_attn.q_proj.weight.detach())
    self._w_q4["k"] = quantize_weight_fixed(self.self_attn.k_proj.weight.detach())
    self._w_q4["v"] = quantize_weight_fixed(self.self_attn.v_proj.weight.detach())
    self._w_q4["o"] = quantize_weight_fixed(self.self_attn.o_proj.weight.detach())
    self._w_q4["down"] = quantize_weight_fixed(self.mlp.down_proj.weight.detach())
    
    # Upload packed weights as byte buffers
    def upload_packed(packed: bytes) -> _Buf:
        raw = np.frombuffer(packed, dtype=np.uint8).astype(np.uint8)
        pad = (-len(raw)) % 4
        if pad:
            raw = np.concatenate([raw, np.zeros(pad, dtype=np.uint8)])
        nbytes = len(raw)
        fb = torch.from_numpy(np.frombuffer(raw.tobytes(), dtype=np.float32).copy())
        assert fb.numel() * 4 == nbytes, f"{fb.numel() * 4} != {nbytes}"
        wb = tvk.VkBuffer(nbytes, "device")
        wb.upload(fb)
        return _Buf._wrap(wb, nbytes)
    
    self._w_q4_buf = {k: upload_packed(v) for k, v in self._w_q4.items()}
    self._uploaded_q4 = True

VulkanDecoderLayer.upload_weights_q4 = upload_weights_q4_fixed

# Now ensure weights are uploaded ONCE at construction, not per forward
def ensure_vulkan_weights_uploaded(hb):
    """Upload all Vulkan layer weights to VRAM once at startup."""
    print("Uploading Vulkan layer weights to VRAM...")
    for i, layer in enumerate(hb.layers):
        if isinstance(layer, VulkanDecoderLayer):
            layer.upload_weights()  # fp32 weights
            layer.upload_weights_q4()  # Q4 weights
            print(f"  Layer {i}: fp32 + Q4 weights uploaded")
    print("All Vulkan weights resident in VRAM")

print("Patches applied:")
print("  - Q4 quantization overflow fixed")
print("  - Vulkan weight upload moved to initialization")
print("  - Q4 upload uses fixed quantization")