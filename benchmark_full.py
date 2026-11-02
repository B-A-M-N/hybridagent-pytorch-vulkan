#!/usr/bin/env python3
"""Full heterogeneous benchmark across all optimal configs."""
import torch
import torch_vulkan_ext as tvk
from backend.quantized_loader_direct import create_hybrid_backbone_with_int8
from backend.hybrid_offload import DeviceStats
import time
import gc

def benchmark_config(gpu_layers: int, seq_len: int, decode_steps: int = 10):
    """Benchmark one configuration."""
    
    print(f"\n  Testing {gpu_layers} GPU layers, S={seq_len}...")
    
    # Create model
    hb = create_hybrid_backbone_with_int8(gpu_layers)
    gc.collect()
    
    # Upload Vulkan weights ONCE
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
    
    # Measure decode
    stats = DeviceStats()
    decode_times = []
    with torch.no_grad():
        for step in range(decode_steps):
            next_id = torch.randint(0, 1000, (1, 1))
            t0 = time.perf_counter()
            _ = hb(input_ids=next_id, use_cache=True, stats=stats, use_q4=False)
            decode_times.append(time.perf_counter() - t0)
    
    decode_times = [t*1000 for t in decode_times]
    decode_times.sort()
    median_decode = decode_times[len(decode_times)//2]
    
    print(f"    prefill={prefill_time*1000:.1f}ms, decode={median_decode:.1f}ms, vk_time={stats.vk_time_s:.3f}s, transfers={stats.transfers}")
    
    return {
        "prefill_ms": prefill_time * 1000,
        "decode_median_ms": median_decode,
        "vk_time_s": stats.vk_time_s,
        "transfers": stats.transfers,
        "transfer_bytes": stats.transfer_bytes
    }


if __name__ == "__main__":
    # Optimal configs from sweeps
    configs = [
        # (gpu_layers, seq_len)
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
            result = benchmark_config(gpu_layers, seq_len, decode_steps=10)
            results[f"gpu{gpu_layers}_s{seq_len}"] = result
        except Exception as e:
            print(f"  FAILED: gpu{gpu_layers}_s{seq_len} - {e}")
            results[f"gpu{gpu_layers}_s{seq_len}"] = {"error": str(e)}
    
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        if "error" in v:
            print(f"  {k}: FAILED - {v['error']}")
        else:
            print(f"  {k}: prefill={v['prefill_ms']:.1f}ms, decode={v['decode_median_ms']:.1f}ms, vk={v['vk_time_s']:.3f}s, xfers={v['transfers']}")