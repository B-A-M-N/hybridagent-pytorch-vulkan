#!/usr/bin/env python3
"""Test S=2048, 3 GPU layers (optimal from sweeps)."""
import torch
import torch_vulkan_ext as tvk
from backend.quantized_loader_direct import create_hybrid_backbone_with_int8
from backend.hybrid_offload import DeviceStats
import time
import gc

print("=== S=2048, 3 GPU layers ===")

hb = create_hybrid_backbone_with_int8(3)
gc.collect()

print("Uploading Vulkan weights...")
for i, layer in enumerate(hb.layers):
    if hasattr(layer, 'upload_weights'):
        layer.upload_weights()
gc.collect()

ids = torch.randint(0, 1000, (1, 2048))
with torch.no_grad():
    # Warmup
    _ = hb(input_ids=ids, use_cache=True, use_q4=False)
    
    # Measured
    stats = DeviceStats()
    t0 = time.perf_counter()
    out = hb(input_ids=ids, use_cache=True, stats=stats, use_q4=False)
    t1 = time.perf_counter()

print(f"Prefill: {(t1-t0)*1000:.1f} ms")
print(f"Stats: {stats.summary()}")

# Decode steps
stats = DeviceStats()
decode_times = []
with torch.no_grad():
    for step in range(10):
        next_id = torch.randint(0, 1000, (1, 1))
        t0 = time.perf_counter()
        _ = hb(input_ids=next_id, use_cache=True, stats=stats, use_q4=False)
        decode_times.append(time.perf_counter() - t0)

decode_times = [t*1000 for t in decode_times]
decode_times.sort()
print(f"Decode (median of 10): {decode_times[len(decode_times)//2]:.1f} ms")
print(f"Decode stats: {stats.summary()}")