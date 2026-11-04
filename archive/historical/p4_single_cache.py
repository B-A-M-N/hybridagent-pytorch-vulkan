#!/usr/bin/env python3
"""
P4 Validation: Single cached-decode benchmark at one cache length.
Run with: python p4_single_cache.py <cache_len>
"""

import os
import sys
import time
import json
import torch
import gc
import psutil

sys.path.insert(0, '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan')

from p2_gpu_kv_test import build_model as build_gpu_model
from p0_reconcile import build_model as build_cpu_model
from backend.hybrid_offload import VulkanDecoderLayer
from p2_gpu_kv_test import _Buf
import torch_vulkan_ext as tvk


def get_memory_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


def run_cpu_cached_decode(model, hb, tokenizer, config, cache_len, num_decode_steps=20, warmup_steps=5):
    results = {"cache_len": cache_len, "decode_steps": [], "peak_rss_mb": 0}
    
    ids = torch.randint(0, 1000, (1, cache_len))
    torch.manual_seed(42)
    
    with torch.no_grad():
        out = model.backbone(input_ids=ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        actual_cache_len = past_kv.get_seq_length()
    
    assert actual_cache_len == cache_len
    
    # Warmup
    with torch.no_grad():
        for step in range(warmup_steps):
            cache_position = torch.tensor([cache_len + step], device=last_hidden.device)
            out = hb(
                inputs_embeds=last_hidden,
                past_key_values=past_kv,
                use_cache=True,
                use_q4=False,
                cache_position=cache_position,
                output_hidden_states=True
            )
            past_kv = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1:, :]
    
    # Measured decode
    times = []
    with torch.no_grad():
        for step in range(num_decode_steps):
            cache_position = torch.tensor([cache_len + warmup_steps + step], device=last_hidden.device)
            t_start = time.perf_counter()
            out = hb(
                inputs_embeds=last_hidden,
                past_key_values=past_kv,
                use_cache=True,
                use_q4=False,
                cache_position=cache_position,
                output_hidden_states=True
            )
            t_end = time.perf_counter()
            
            step_time = (t_end - t_start) * 1000
            times.append(step_time)
            
            past_kv = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1:, :]
            
            rss = get_memory_mb()
            results["peak_rss_mb"] = max(results.get("peak_rss_mb", 0), rss)
    
    times_sorted = sorted(times)
    results["decode_steps"] = times
    results["median_ms"] = times_sorted[len(times_sorted)//2]
    results["avg_ms"] = sum(times) / len(times)
    results["min_ms"] = min(times)
    results["max_ms"] = max(times)
    results["steps_per_sec"] = 1000 / results["avg_ms"]
    
    return results


def run_gpu_cached_decode(model, hb, tokenizer, config, kv_manager, cache_len, num_decode_steps=20, warmup_steps=5, use_q4=True):
    results = {"cache_len": cache_len, "decode_steps": [], "peak_rss_mb": 0, "peak_vram_mb": 0}
    
    ids = torch.randint(0, 1000, (1, cache_len))
    torch.manual_seed(42)
    
    with torch.no_grad():
        out = model.backbone(input_ids=ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        actual_cache_len = past_kv.get_seq_length()
    
    assert actual_cache_len == cache_len
    
    # Seed GPU KV cache
    for layer in hb.layers:
        if isinstance(layer, VulkanDecoderLayer) and hasattr(layer, '_gpu_kv_cache'):
            kv_mgr = layer._gpu_kv_cache
            cached_layer = past_kv.layers[layer.layer_idx]
            cached_k = cached_layer.keys
            cached_v = cached_layer.values
            S = cached_k.shape[2]
            KVD = layer.num_kv_heads * layer.head_dim
            k_gpu = cached_k.permute(0, 2, 1, 3).contiguous().view(1, S * KVD)
            v_gpu = cached_v.permute(0, 2, 1, 3).contiguous().view(1, S * KVD)
            kv_mgr.append_kv(layer.layer_idx, _Buf(k_gpu), _Buf(v_gpu), S, KVD)
    
    # Warmup
    with torch.no_grad():
        for step in range(warmup_steps):
            cache_position = torch.tensor([cache_len + step], device=last_hidden.device)
            out = hb(
                inputs_embeds=last_hidden,
                past_key_values=past_kv,
                use_cache=True,
                use_q4=use_q4,
                cache_position=cache_position,
                output_hidden_states=True
            )
            past_kv = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1:, :]
    
    # Measured decode
    times = []
    with torch.no_grad():
        for step in range(num_decode_steps):
            cache_position = torch.tensor([cache_len + warmup_steps + step], device=last_hidden.device)
            t_start = time.perf_counter()
            out = hb(
                inputs_embeds=last_hidden,
                past_key_values=past_kv,
                use_cache=True,
                use_q4=use_q4,
                cache_position=cache_position,
                output_hidden_states=True
            )
            t_end = time.perf_counter()
            
            step_time = (t_end - t_start) * 1000
            times.append(step_time)
            
            past_kv = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1:, :]
            
            rss = get_memory_mb()
            vram = tvk.vram_used() / 1024 / 1024
            results["peak_rss_mb"] = max(results.get("peak_rss_mb", 0), rss)
            results["peak_vram_mb"] = max(results.get("peak_vram_mb", 0), vram)
    
    times_sorted = sorted(times)
    results["decode_steps"] = times
    results["median_ms"] = times_sorted[len(times_sorted)//2]
    results["avg_ms"] = sum(times) / len(times)
    results["min_ms"] = min(times)
    results["max_ms"] = max(times)
    results["steps_per_sec"] = 1000 / results["avg_ms"]
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python p4_single_cache.py <cache_len>")
        sys.exit(1)
    
    cache_len = int(sys.argv[1])
    num_decode_steps = 20
    warmup_steps = 5
    
    print(f"=" * 70)
    print(f"P4 CACHED DECODE: cache_len={cache_len}")
    print(f"=" * 70)
    
    # Build models
    print("Building GPU model...")
    model_gpu, hb_gpu, tokenizer, config, kv_manager = build_gpu_model(gpu_layers=1)
    for layer in hb_gpu.layers:
        if isinstance(layer, VulkanDecoderLayer):
            layer._gpu_kv_cache = kv_manager
    
    print("Building CPU model...")
    model_cpu, hb_cpu, _, _ = build_cpu_model(gpu_layers=0, use_vulkan=False, use_int8=True)
    
    # CPU
    print(f"\n--- CPU Cached Decode at cache_len={cache_len} ---")
    cpu_results = run_cpu_cached_decode(model_cpu, hb_cpu, tokenizer, config, cache_len, num_decode_steps=20, warmup_steps=5)
    print(f"  Median: {cpu_results['median_ms']:.1f} ms")
    print(f"  Avg:    {cpu_results['avg_ms']:.1f} ms")
    print(f"  Min/Max: {cpu_results['min_ms']:.1f} / {cpu_results['max_ms']:.1f} ms")
    print(f"  Steps/s: {cpu_results['steps_per_sec']:.2f}")
    print(f"  Peak RSS: {cpu_results['peak_rss_mb']:.0f} MB")
    
    # GPU
    print(f"\n--- GPU Cached Decode at cache_len={cache_len} ---")
    gpu_results = run_gpu_cached_decode(model_gpu, hb_gpu, tokenizer, config, kv_manager, cache_len, num_decode_steps=20, warmup_steps=5)
    print(f"  Median: {gpu_results['median_ms']:.1f} ms")
    print(f"  Avg:    {gpu_results['avg_ms']:.1f} ms")
    print(f"  Min/Max: {gpu_results['min_ms']:.1f} / {gpu_results['max_ms']:.1f} ms")
    print(f"  Steps/s: {gpu_results['steps_per_sec']:.2f}")
    print(f"  Peak RSS: {gpu_results['peak_rss_mb']:.0f} MB, Peak VRAM: {gpu_results['peak_vram_mb']:.0f} MB")
    
    # Summary
    speedup = cpu_results['median_ms'] / gpu_results['median_ms']
    winner = "GPU" if speedup > 1 else "CPU"
    print(f"\n  Speedup (CPU/GPU median): {speedup:.2f}x -> {winner} wins")
    
    all_results = {
        "cache_len": cache_len,
        "cpu": cpu_results,
        "gpu": gpu_results,
        "speedup_median": speedup,
        "winner": winner
    }
    
    with open(f"p4_cache_{cache_len}_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to p4_cache_{cache_len}_results.json")


if __name__ == "__main__":
    main()