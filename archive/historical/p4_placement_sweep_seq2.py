#!/usr/bin/env python3
"""
P4: Placement sweep at multiple sequence lengths (S=128, 512, 1024, 2048)
Tests prefill and sustained decode at various sequence lengths to find
where heterogeneous execution wins.
"""

import os
import sys
import time
import json
import torch
import gc
import psutil

sys.path.insert(0, '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan')

# Reuse working build_model from p2_gpu_kv_test.py for GPU layers
from p2_gpu_kv_test import build_model as build_gpu_model, GPUKVCacheManager, VulkanDecoderLayer, _Buf
# For CPU-only, use p0_reconcile which works
from p0_reconcile import build_model as build_cpu_model
import torch_vulkan_ext as tvk


def get_memory_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


def run_gpu_benchmark_at_seqlen(model, hb, tokenizer, config, kv_manager, seq_len, num_decode_steps=20, warmup_steps=3, use_q4=True):
    """Run benchmark at a specific sequence length for GPU model."""
    results = {
        "seq_len": seq_len,
        "prefill_ms": 0,
        "decode_steps": [],
        "warmup_steps": [],
        "total_ms": 0,
        "peak_rss_mb": 0,
        "peak_vram_mb": 0,
        "final_cache_len": 0,
    }
    
    # Create input of specified sequence length
    ids = torch.randint(0, 1000, (1, seq_len))
    torch.manual_seed(42)
    
    # Attach GPU KV manager
    for layer in hb.layers:
        if isinstance(layer, VulkanDecoderLayer):
            layer._gpu_kv_cache = kv_manager
    
    # Prefill
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        out = model.backbone(input_ids=ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        cache_len = past_kv.get_seq_length()
    t_prefill_end = time.perf_counter()
    results["prefill_ms"] = (t_prefill_end - t_prefill_start) * 1000
    
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
    
    print(f"  Prefill (S={seq_len}): {results['prefill_ms']:.1f} ms, cache_len={cache_len}")
    
    # Decode steps
    times = []
    with torch.no_grad():
        for step in range(num_decode_steps + warmup_steps):
            cache_position = torch.tensor([cache_len], device=last_hidden.device)
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
            cache_len = past_kv.get_seq_length()
            
            if step < warmup_steps:
                results["warmup_steps"].append(step_time)
            else:
                results["decode_steps"].append(step_time)
            
            if step % 10 == 0:
                rss = get_memory_mb()
                vram = tvk.vram_used() / 1024 / 1024
                results["peak_rss_mb"] = max(results["peak_rss_mb"], rss)
                results["peak_vram_mb"] = max(results["peak_vram_mb"], vram)
                print(f"    Step {step}: {step_time:.1f} ms, cache_len={cache_len}, RSS={rss:.0f}MB, VRAM={vram:.0f}MB")
    
    results["total_ms"] = sum(times)
    results["final_cache_len"] = cache_len
    
    decode = results["decode_steps"]
    if decode:
        decode_sorted = sorted(decode)
        results["median_ms"] = decode_sorted[len(decode_sorted)//2]
        results["min_ms"] = min(decode)
        results["max_ms"] = max(decode)
        results["avg_ms"] = sum(decode) / len(decode)
        results["steps_per_sec"] = 1000 / results["avg_ms"]
    
    warmup = results["warmup_steps"]
    if warmup:
        warmup_sorted = sorted(warmup)
        results["warmup_avg_ms"] = sum(warmup) / len(warmup)
        results["warmup_median_ms"] = warmup_sorted[len(warmup_sorted)//2]
    
    return results


def run_cpu_benchmark_at_seqlen(model, hb, tokenizer, config, seq_len, num_decode_steps=20, warmup_steps=3):
    """Run benchmark at a specific sequence length for CPU-only model."""
    results = {
        "seq_len": seq_len,
        "prefill_ms": 0,
        "decode_steps": [],
        "warmup_steps": [],
        "total_ms": 0,
        "peak_rss_mb": 0,
        "peak_vram_mb": 0,
        "final_cache_len": 0,
    }
    
    # Create input of specified sequence length
    ids = torch.randint(0, 1000, (1, seq_len))
    torch.manual_seed(42)
    
    # Prefill
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        out = model.backbone(input_ids=ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        cache_len = past_kv.get_seq_length()
    t_prefill_end = time.perf_counter()
    results["prefill_ms"] = (t_prefill_end - t_prefill_start) * 1000
    
    print(f"  Prefill (S={seq_len}): {results['prefill_ms']:.1f} ms, cache_len={cache_len}")
    
    # Decode steps
    times = []
    with torch.no_grad():
        for step in range(num_decode_steps + warmup_steps):
            cache_position = torch.tensor([cache_len], device=last_hidden.device)
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
            cache_len = past_kv.get_seq_length()
            
            if step < warmup_steps:
                results["warmup_steps"].append(step_time)
            else:
                results["decode_steps"].append(step_time)
            
            if step % 10 == 0:
                rss = get_memory_mb()
                results["peak_rss_mb"] = max(results["peak_rss_mb"], rss)
                print(f"    Step {step}: {step_time:.1f} ms, cache_len={cache_len}, RSS={rss:.0f}MB")
    
    results["total_ms"] = sum(times)
    results["final_cache_len"] = cache_len
    
    decode = results["decode_steps"]
    if decode:
        decode_sorted = sorted(decode)
        results["median_ms"] = decode_sorted[len(decode_sorted)//2]
        results["min_ms"] = min(decode)
        results["max_ms"] = max(decode)
        results["avg_ms"] = sum(decode) / len(decode)
        results["steps_per_sec"] = 1000 / results["avg_ms"]
    
    warmup = results["warmup_steps"]
    if warmup:
        warmup_sorted = sorted(warmup)
        results["warmup_avg_ms"] = sum(warmup) / len(warmup)
        results["warmup_median_ms"] = warmup_sorted[len(warmup_sorted)//2]
    
    return results


def main():
    print("=" * 70)
    print("P4: PLACEMENT SWEEP AT MULTIPLE SEQUENCE LENGTHS")
    print("=" * 70)
    
    seq_lens = [128, 512, 1024, 2048]
    all_configs = {}
    
    # Test 0 GPU layers (CPU-only baseline)
    print(f"\n{'='*70}")
    print(f"TESTING: 0 GPU layer(s) at seq_lens={seq_lens}")
    print(f"{'='*70}")
    
    model, hb, tokenizer, config = build_cpu_model(gpu_layers=0, use_vulkan=False, use_int8=True)
    print(f"  Built CPU-only. RSS: {get_memory_mb():.0f} MB")
    
    all_results_cpu = {}
    for seq_len in seq_lens:
        print(f"\n  --- Sequence length {seq_len} ---")
        try:
            results = run_cpu_benchmark_at_seqlen(model, hb, tokenizer, config, seq_len, num_decode_steps=20, warmup_steps=3)
            all_results_cpu[f"seq_{seq_len}"] = results
            print(f"    Decode: avg={results.get('avg_ms', 0):.1f} ms, median={results.get('median_ms', 0):.1f} ms, {results.get('steps_per_sec', 0):.2f} steps/s")
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results_cpu[f"seq_{seq_len}"] = {"error": str(e)}
    
    all_configs["gpu_layers_0"] = all_results_cpu
    
    # Test 1 GPU layer
    print(f"\n{'='*70}")
    print(f"TESTING: 1 GPU layer(s) at seq_lens={seq_lens}")
    print(f"{'='*70}")
    
    model, hb, tokenizer, config, kv_manager = build_gpu_model(gpu_layers=1)
    print(f"  Built. RSS: {get_memory_mb():.0f} MB, VRAM: {tvk.vram_used()/1024/1024:.0f} MB")
    
    all_results_1 = {}
    for seq_len in seq_lens:
        print(f"\n  --- Sequence length {seq_len} ---")
        try:
            results = run_gpu_benchmark_at_seqlen(model, hb, tokenizer, config, kv_manager, seq_len, num_decode_steps=20, warmup_steps=3)
            all_results_1[f"seq_{seq_len}"] = results
            print(f"    Decode: avg={results.get('avg_ms', 0):.1f} ms, median={results.get('median_ms', 0):.1f} ms, {results.get('steps_per_sec', 0):.2f} steps/s")
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results_1[f"seq_{seq_len}"] = {"error": str(e)}
    
    all_configs["gpu_layers_1"] = all_results_1
    
    # Test 2 GPU layers
    print(f"\n{'='*70}")
    print(f"TESTING: 2 GPU layer(s) at seq_lens={seq_lens}")
    print(f"{'='*70}")
    
    model, hb, tokenizer, config, kv_manager = build_gpu_model(gpu_layers=2)
    print(f"  Built. RSS: {get_memory_mb():.0f} MB, VRAM: {tvk.vram_used()/1024/1024:.0f} MB")
    
    all_results_2 = {}
    for seq_len in seq_lens:
        print(f"\n  --- Sequence length {seq_len} ---")
        try:
            results = run_gpu_benchmark_at_seqlen(model, hb, tokenizer, config, kv_manager, seq_len, num_decode_steps=20, warmup_steps=3)
            all_results_2[f"seq_{seq_len}"] = results
            print(f"    Decode: avg={results.get('avg_ms', 0):.1f} ms, median={results.get('median_ms', 0):.1f} ms, {results.get('steps_per_sec', 0):.2f} steps/s")
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results_2[f"seq_{seq_len}"] = {"error": str(e)}
    
    all_configs["gpu_layers_2"] = all_results_2
    
    # Test 3 GPU layers if VRAM permits
    try:
        print(f"\n{'='*70}")
        print(f"TESTING: 3 GPU layer(s) at seq_lens={seq_lens}")
        print(f"{'='*70}")
        
        model, hb, tokenizer, config, kv_manager = build_gpu_model(gpu_layers=3)
        print(f"  Built. RSS: {get_memory_mb():.0f} MB, VRAM: {tvk.vram_used()/1024/1024:.0f} MB")
        
        all_results_3 = {}
        for seq_len in seq_lens:
            print(f"\n  --- Sequence length {seq_len} ---")
            try:
                results = run_gpu_benchmark_at_seqlen(model, hb, tokenizer, config, kv_manager, seq_len, num_decode_steps=20, warmup_steps=3)
                all_results_3[f"seq_{seq_len}"] = results
                print(f"    Decode: avg={results.get('avg_ms', 0):.1f} ms, median={results.get('median_ms', 0):.1f} ms, {results.get('steps_per_sec', 0):.2f} steps/s")
            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_results_3[f"seq_{seq_len}"] = {"error": str(e)}
        
        all_configs["gpu_layers_3"] = all_results_3
    except Exception as e:
        print(f"3 GPU layers failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("P4 MULTI-SEQUENCE-LENGTH SUMMARY")
    print("=" * 70)
    print(f"{'Config':<15} {'Seq':<8} {'Prefill(ms)':<12} {'Decode Avg(ms)':<15} {'Decode Med(ms)':<15} {'Steps/s':<10} {'Speedup':<8}")
    print("-" * 70)
    
    for config_name, config_results in all_configs.items():
        for seq_key, results in config_results.items():
            if "error" in results:
                continue
            seq = seq_key.split("_")[1]
            prefill = results.get("prefill_ms", 0)
            avg = results.get("avg_ms", 0)
            median = results.get("median_ms", 0)
            steps = results.get("steps_per_sec", 0)
            
            # Calculate speedup vs CPU-only at same seq_len
            cpu_key = "gpu_layers_0"
            speedup = ""
            if cpu_key in all_configs and seq_key in all_configs[cpu_key]:
                cpu_avg = all_configs[cpu_key][seq_key].get("avg_ms", 0)
                if cpu_avg > 0 and avg > 0:
                    speedup = f"{cpu_avg/avg:.2f}x"
            
            print(f"{config_name:<15} {seq:<8} {prefill:<12.1f} {avg:<15.1f} {median:<15.1f} {steps:<10.2f} {speedup:<8}")
    
    # Save results
    with open("p4_multi_seq_results.json", "w") as f:
        json.dump(all_configs, f, indent=2)
    print("\nResults saved to p4_multi_seq_results.json")


if __name__ == "__main__":
    main()