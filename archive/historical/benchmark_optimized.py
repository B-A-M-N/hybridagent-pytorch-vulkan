#!/usr/bin/env python3
"""Benchmark heterogeneous execution - fp32 Vulkan path first (no Q4)."""
import torch
import torch_vulkan_ext as tvk
from backend.quantized_loader_direct import create_hybrid_backbone_with_int8
from backend.hybrid_offload import DeviceStats
import time
import gc

def benchmark_heterogeneous(gpu_layers: int, seq_len: int, steps: int = 10):
    """Benchmark full heterogeneous forward pass with stats."""
    
    print(f"\n=== Benchmark: {gpu_layers} GPU layers, S={seq_len}, {steps} steps ===")
    
    # Create model
    hb = create_hybrid_backbone_with_int8(gpu_layers)
    gc.collect()
    
    # Upload Vulkan weights ONCE (fp32 only, no Q4)
    print("  Uploading Vulkan fp32 weights...")
    for i, layer in enumerate(hb.layers):
        if hasattr(layer, 'upload_weights'):
            layer.upload_weights()
    gc.collect()
    
    # Warmup
    ids = torch.randint(0, 1000, (1, seq_len))
    with torch.no_grad():
        _ = hb(input_ids=ids, use_cache=True, use_q4=False)
    
    # Measure prefill
    stats = DeviceStats()
    with torch.no_grad():
        t0 = time.perf_counter()
        out = hb(input_ids=ids, use_cache=True, stats=stats, use_q4=False)
        prefill_time = time.perf_counter() - t0
    
    print(f"  Prefill: {prefill_time*1000:.1f} ms")
    print(f"  Stats: {stats.summary()}")
    
    # Measure sustained decode
    stats = DeviceStats()
    decode_times = []
    with torch.no_grad():
        for step in range(steps):
            next_id = torch.randint(0, 1000, (1, 1))
            t0 = time.perf_counter()
            _ = hb(input_ids=next_id, use_cache=True, stats=stats, use_q4=False)
            decode_times.append(time.perf_counter() - t0)
    
    decode_times = [t*1000 for t in decode_times]
    decode_times.sort()
    median_decode = decode_times[len(decode_times)//2]
    
    print(f"  Decode (median of {steps}): {median_decode:.1f} ms")
    print(f"  Decode stats: {stats.summary()}")
    
    return {
        "prefill_ms": prefill_time * 1000,
        "decode_median_ms": median_decode,
        "decode_all_ms": decode_times,
        "stats": stats.summary()
    }


if __name__ == "__main__":
    # Test optimal configurations from sweeps
    configs = [
        (1, 12),
        (2, 12),
        (5, 12),
        (2, 128),
        (5, 128),
        (5, 512),
        (4, 1024),
        (3, 2048),
    ]
    
    results = {}
    for gpu_layers, seq_len in configs:
        try:
            result = benchmark_heterogeneous(gpu_layers, seq_len, steps=20)
            results[f"gpu{gpu_layers}_s{seq_len}"] = result
        except Exception as e:
            print(f"FAILED: gpu{gpu_layers}_s{seq_len} - {e}")
            results[f"gpu{gpu_layers}_s{seq_len}"] = {"error": str(e)}
    
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        if "error" in v:
            print(f"  {k}: FAILED - {v['error']}")
        else:
            print(f"  {k}: prefill={v['prefill_ms']:.1f}ms, decode={v['decode_median_ms']:.1f}ms")