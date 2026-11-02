#!/usr/bin/env python3
"""
P4 Validation: Proper cached-decode benchmark at specific KV cache lengths.
Tests ONE new token at existing KV lengths 128, 512, 1024, 2048.
Prefill measured separately. Decode measured with warmed execution.
"""

import os
import sys
import time
import json
import torch
import gc
import psutil

sys.path.insert(0, '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan')

# Use the working build functions
from p2_gpu_kv_test import build_model as build_gpu_model
from p0_reconcile import build_model as build_cpu_model
from backend.hybrid_offload import VulkanDecoderLayer
from p2_gpu_kv_test import _Buf
import torch_vulkan_ext as tvk


def get_memory_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


def run_cached_decode_benchmark(model, hb, tokenizer, config, kv_manager, cache_len, num_decode_steps=20, warmup_steps=5, use_q4=True, gpu_layers=1):
    """Run cached decode benchmark at a specific cache length.
    Returns median decode latency for 1 new token per step.
    """
    results = {
        "cache_len": cache_len,
        "decode_steps": [],
        "warmup_steps": [],
        "peak_rss_mb": 0,
        "peak_vram_mb": 0,
    }
    
    # Create prefill context of exactly cache_len tokens
    ids = torch.randint(0, 1000, (1, cache_len))
    torch.manual_seed(42)
    
    # Prefill - build KV cache
    with torch.no_grad():
        out = model.backbone(input_ids=ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        actual_cache_len = past_kv.get_seq_length()
    
    assert actual_cache_len == cache_len, f"Expected cache_len={cache_len}, got {actual_cache_len}"
    
    # Seed GPU KV cache with prefill data
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
    
    # Warmup steps
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
    
    # Measured decode steps - 1 new token per step
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
            vram = tvk.vram_used() / 1024 / 1024 if gpu_layers > 0 else 0
            results["peak_rss_mb"] = max(results["peak_rss_mb"], rss)
            results["peak_vram_mb"] = max(results["peak_vram_mb"], vram)
    
    times_sorted = sorted(times)
    results["decode_steps"] = times
    results["median_ms"] = times_sorted[len(times_sorted)//2]
    results["avg_ms"] = sum(times) / len(times)
    results["min_ms"] = min(times)
    results["max_ms"] = max(times)
    results["steps_per_sec"] = 1000 / results["avg_ms"]
    
    return results


def run_cpu_cached_decode(model, hb, tokenizer, config, cache_len, num_decode_steps=20, warmup_steps=5):
    """Run CPU-only cached decode benchmark at specific cache length."""
    results = {
        "cache_len": cache_len,
        "decode_steps": [],
        "warmup_steps": [],
        "peak_rss_mb": 0,
    }
    
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
            results["peak_rss_mb"] = max(results["peak_rss_mb"], rss)
    
    times_sorted = sorted(times)
    results["decode_steps"] = times
    results["median_ms"] = times_sorted[len(times_sorted)//2]
    results["avg_ms"] = sum(times) / len(times)
    results["min_ms"] = min(times)
    results["max_ms"] = max(times)
    results["steps_per_sec"] = 1000 / results["avg_ms"]
    
    return results


def run_prefill_benchmark(model, hb, tokenizer, config, cache_len, kv_manager=None, warmup=3, use_q4=True, gpu_layers=1):
    """Measure prefill latency separately."""
    results = {"cache_len": cache_len, "prefill_times": []}
    
    for trial in range(warmup + 5):
        ids = torch.randint(0, 1000, (1, cache_len))
        torch.manual_seed(42 + trial)
        
        if gpu_layers > 0:
            gc.collect()
            tvk.vram_used()  # ensure initialized
        
        t_start = time.perf_counter()
        with torch.no_grad():
            out = model.backbone(input_ids=ids, use_cache=True, output_hidden_states=True)
        t_end = time.perf_counter()
        
        prefill_ms = (t_end - t_start) * 1000
        if trial >= warmup:
            results["prefill_times"].append(prefill_ms)
    
    results["prefill_median_ms"] = sorted(results["prefill_times"])[len(results["prefill_times"])//2]
    results["prefill_avg_ms"] = sum(results["prefill_times"]) / len(results["prefill_times"])
    results["prefill_min_ms"] = min(results["prefill_times"])
    results["prefill_max_ms"] = max(results["prefill_times"])
    
    return results


def main():
    print("=" * 70)
    print("P4 VALIDATION: CACHED DECODE AT SPECIFIC KV LENGTHS")
    print("=" * 70)
    print("Methodology:")
    print("  1. Prefill to exact cache length (measured separately)")
    print("  2. Warmup decode steps")
    print("  3. Measure 1-token decode latency at fixed cache length")
    print("  4. Report median, avg, min, max over 20 steps")
    print("  5. CPU-only vs GPU (1 layer, Vulkan Q4, GPU-local KV)")
    print()
    
    cache_lengths = [128, 512, 1024, 2048]
    num_decode_steps = 20
    warmup_steps = 5
    
    # Build models
    print("Building GPU model (1 layer, Vulkan Q4, GPU-local KV)...")
    model_gpu, hb_gpu, tokenizer, config, kv_manager = build_gpu_model(gpu_layers=1)
    print(f"  Built. RSS: {get_memory_mb():.0f} MB, VRAM: {tvk.vram_used()/1024/1024:.0f} MB")
    
    # Attach GPU KV manager to Vulkan layers
    for layer in hb_gpu.layers:
        if isinstance(layer, VulkanDecoderLayer):
            layer._gpu_kv_cache = kv_manager
    
    print("Building CPU-only model (INT8)...")
    model_cpu, hb_cpu, _, _ = build_cpu_model(gpu_layers=0, use_vulkan=False, use_int8=True)
    print(f"  Built. RSS: {get_memory_mb():.0f} MB")
    
    all_results = {}
    
    for cache_len in cache_lengths:
        print(f"\n{'='*70}")
        print(f"CACHE LENGTH: {cache_len}")
        print(f"{'='*70}")
        
        # Prefill benchmark
        print(f"\n--- Prefill ({cache_len} tokens) ---")
        prefill_cpu = run_prefill_benchmark(model_cpu, hb_cpu, tokenizer, config, cache_len, kv_manager=None, use_q4=False, gpu_layers=0)
        prefill_gpu = run_prefill_benchmark(model_gpu, hb_gpu, tokenizer, config, cache_len, kv_manager=kv_manager, use_q4=True, gpu_layers=1)
        print(f"  CPU prefill: {prefill_cpu['prefill_median_ms']:.1f} ms median ({prefill_cpu['prefill_avg_ms']:.1f} avg)")
        print(f"  GPU prefill: {prefill_gpu['prefill_median_ms']:.1f} ms median ({prefill_gpu['prefill_avg_ms']:.1f} avg)")
        
        # CPU cached decode
        print(f"\n--- CPU Cached Decode (1 token x {num_decode_steps} steps) ---")
        cpu_results = run_cpu_cached_decode(model_cpu, hb_cpu, tokenizer, config, cache_len, num_decode_steps, warmup_steps)
        print(f"  Median: {cpu_results['median_ms']:.1f} ms")
        print(f"  Avg:    {cpu_results['avg_ms']:.1f} ms")
        print(f"  Min/Max: {cpu_results['min_ms']:.1f} / {cpu_results['max_ms']:.1f} ms")
        print(f"  Steps/s: {cpu_results['steps_per_sec']:.2f}")
        print(f"  Peak RSS: {cpu_results['peak_rss_mb']:.0f} MB")
        
        # GPU cached decode
        print(f"\n--- GPU Cached Decode (1 token x {num_decode_steps} steps) ---")
        gpu_results = run_cached_decode_benchmark(model_gpu, hb_gpu, tokenizer, config, kv_manager, cache_len, num_decode_steps, warmup_steps, use_q4=True, gpu_layers=1)
        print(f"  Median: {gpu_results['median_ms']:.1f} ms")
        print(f"  Avg:    {gpu_results['avg_ms']:.1f} ms")
        print(f"  Min/Max: {gpu_results['min_ms']:.1f} / {gpu_results['max_ms']:.1f} ms")
        print(f"  Steps/s: {gpu_results['steps_per_sec']:.2f}")
        print(f"  Peak RSS: {gpu_results['peak_rss_mb']:.0f} MB, Peak VRAM: {gpu_results['peak_vram_mb']:.0f} MB")
        
        # Speedup
        speedup = cpu_results['median_ms'] / gpu_results['median_ms']
        winner = "GPU" if speedup > 1 else "CPU"
        print(f"\n  Speedup (CPU/GPU median): {speedup:.2f}x -> {winner} wins")
        
        all_results[f"cache_{cache_len}"] = {
            "prefill_cpu": prefill_cpu,
            "prefill_gpu": prefill_gpu,
            "decode_cpu": cpu_results,
            "decode_gpu": gpu_results,
            "speedup_median": speedup,
            "winner": winner
        }
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: CACHED DECODE MEDIAN LATENCY (1 new token per step)")
    print("=" * 70)
    print(f"{'Cache Len':<12} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10} {'Winner':<8}")
    print("-" * 70)
    
    for cache_len in cache_lengths:
        r = all_results[f"cache_{cache_len}"]
        cpu_med = r["decode_cpu"]["median_ms"]
        gpu_med = r["decode_gpu"]["median_ms"]
        sp = r["speedup_median"]
        w = r["winner"]
        print(f"{cache_len:<12} {cpu_med:<12.1f} {gpu_med:<12.1f} {sp:<10.2f}x {w:<8}")
    
    # Save results
    with open("p4_validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to p4_validation_results.json")


if __name__ == "__main__":
    main()