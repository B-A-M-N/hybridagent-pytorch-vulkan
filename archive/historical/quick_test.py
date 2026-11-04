#!/usr/bin/env python3
"""Quick test of heterogeneous execution - single config."""
import torch
import torch_vulkan_ext as tvk
from backend.quantized_loader_direct import create_hybrid_backbone_with_int8
from backend.hybrid_offload import DeviceStats
import time
import gc

print("=== Quick Heterogeneous Test ===")

# Create model with 1 GPU layer (S=12)
hb = create_hybrid_backbone_with_int8(1)
gc.collect()

# Upload Vulkan weights ONCE
print("Uploading Vulkan fp32 weights...")
for i, layer in enumerate(hb.layers):
    if hasattr(layer, 'upload_weights'):
        layer.upload_weights()
gc.collect()

# Quick forward
ids = torch.randint(0, 1000, (1, 12))
with torch.no_grad():
    # Warmup
    _ = hb(input_ids=ids, use_cache=True, use_q4=False)
    
    # Measured
    stats = DeviceStats()
    t0 = time.perf_counter()
    out = hb(input_ids=ids, use_cache=True, stats=stats, use_q4=False)
    t1 = time.perf_counter()
    
print(f"Forward time: {(t1-t0)*1000:.1f} ms")
print(f"Stats: {stats.summary()}")
print("SUCCESS")