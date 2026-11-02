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

from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer, _Buf
from third_party.HybridAgent.hybrid_agent.model import HybridAgentModel
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import safe_open
import torch_vulkan_ext as tvk


def get_memory_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


def build_model(gpu_layers=1, use_q4=True):
    """Build model with specified number of GPU layers."""
    # Load tokenizer and config from original model (has tokenizer and config)
    model_base = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
    config = AutoConfig.from_pretrained(model_base)
    
    # Determine GPU layer indices
    gpu_layer_indices = list(range(min(gpu_layers, config.num_hidden_layers)))
    placement = Placement(layers_on_gpu=gpu_layer_indices)
    
    # Load backbone
    from transformers import AutoModelForCausalLM
    backbone = AutoModelForCausalLM.from_config(config)
    
    # Build state dict - INT8 for CPU layers, FP32 for GPU layers
    state_dict = {}
    
    int8_path = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
    with safe_open(int8_path, framework="pt", device="cpu") as f:
        for i in range(config.num_hidden_layers):
            is_vulkan = i in gpu_layer_indices
            if not is_vulkan:
                # CPU layers get INT8 weights
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    base = f"model.layers.{i}.mlp.{name}"
                    weight_int8 = f.get_tensor(f"{base}.weight_int8")
                    weight_scales = f.get_tensor(f"{base}.weight_scales")
                    state_dict[f"model.layers.{i}.mlp.{name}.weight_int8"] = weight_int8
                    state_dict[f"model.layers.{i}.mlp.{name}.scales"] = weight_scales
                    try:
                        bias = f.get_tensor(f"{base}.bias")
                        state_dict[f"model.layers.{i}.mlp.{name}.bias"] = bias
                    except: pass
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    base = f"model.layers.{i}.self_attn.{proj_name}"
                    weight_int8 = f.get_tensor(f"{base}.weight_int8")
                    weight_scales = f.get_tensor(f"{base}.weight_scales")
                    state_dict[f"model.layers.{i}.self_attn.{proj_name}.weight_int8"] = weight_int8
                    state_dict[f"model.layers.{i}.self_attn.{proj_name}.scales"] = weight_scales
                    try:
                        bias = f.get_tensor(f"{base}.bias")
                        state_dict[f"model.layers.{i}.self_attn.{proj_name}.bias"] = bias
                    except: pass
            try:
                ln1 = f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
                state_dict[f"model.layers.{i}.input_layernorm.weight"] = ln1
            except: pass
            try:
                ln2 = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
                state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = ln2
            except: pass
            try:
                q_norm = f.get_tensor(f"model.layers.{i}.self_attn.q_norm.weight")
                state_dict[f"model.layers.{i}.self_attn.q_norm.weight"] = q_norm
            except: pass
            try:
                k_norm = f.get_tensor(f"model.layers.{i}.self_attn.k_norm.weight")
                state_dict[f"model.layers.{i}.self_attn.k_norm.weight"] = k_norm
            except: pass
        embed = f.get_tensor("model.embed_tokens.weight")
        state_dict["model.embed_tokens.weight"] = embed
        norm_weight = f.get_tensor("model.norm.weight")
        state_dict["model.norm.weight"] = norm_weight
        lm_head = f.get_tensor("lm_head.weight")
        state_dict["lm_head.weight"] = lm_head
    
    # Load FP32 weights for GPU layers from original model
    if gpu_layer_indices:
        original_path = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic/model.safetensors"
        with safe_open(original_path, framework="pt", device="cpu") as f:
            for i in gpu_layer_indices:
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    base = f"model.layers.{i}.mlp.{name}"
                    try:
                        weight = f.get_tensor(f"{base}.weight")
                        if hasattr(backbone.model.layers[i].mlp, name):
                            getattr(backbone.model.layers[i].mlp, name).weight.data.copy_(weight)
                    except: pass
                    try:
                        bias = f.get_tensor(f"{base}.bias")
                        if hasattr(backbone.model.layers[i].mlp, name) and getattr(backbone.model.layers[i].mlp, name).bias is not None:
                            getattr(backbone.model.layers[i].mlp, name).bias.data.copy_(bias)
                    except: pass
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    base = f"model.layers.{i}.self_attn.{proj_name}"
                    try:
                        weight = f.get_tensor(f"{base}.weight")
                        if hasattr(backbone.model.layers[i].self_attn, proj_name):
                            getattr(backbone.model.layers[i].self_attn, proj_name).weight.data.copy_(weight)
                    except: pass
                    try:
                        bias = f.get_tensor(f"{base}.bias")
                        if hasattr(backbone.model.layers[i].self_attn, proj_name) and getattr(backbone.model.layers[i].self_attn, proj_name).bias is not None:
                            getattr(backbone.model.layers[i].self_attn, proj_name).bias.data.copy_(bias)
                    except: pass
                try:
                    ln1 = f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
                    backbone.model.layers[i].input_layernorm.weight.data.copy_(ln1)
                except: pass
                try:
                    ln2 = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
                    backbone.model.layers[i].post_attention_layernorm.weight.data.copy_(ln2)
                except: pass
                try:
                    q_norm = f.get_tensor(f"model.layers.{i}.self_attn.q_norm.weight")
                    if hasattr(backbone.model.layers[i].self_attn, 'q_norm'):
                        backbone.model.layers[i].self_attn.q_norm.weight.data.copy_(q_norm)
                except: pass
                try:
                    k_norm = f.get_tensor(f"model.layers.{i}.self_attn.k_norm.weight")
                    if hasattr(backbone.model.layers[i].self_attn, 'k_norm'):
                        backbone.model.layers[i].self_attn.k_norm.weight.data.copy_(k_norm)
                except: pass
    
    gc.collect()
    
    backbone.load_state_dict(state_dict, assign=True)
    backbone = backbone.to_empty(device="cpu")
    
    model = HybridAgentModel(backbone)
    model.resize_to_tokenizer(tokenizer)
    
    adapter_path = os.path.join(model_base, "adapter.pt")
    if os.path.exists(adapter_path):
        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        model.hybridlink.load_state_dict(state, strict=True)
    
    qm = model.backbone.model
    hb = HybridBackbone(qm, placement)
    gc.collect()
    
    print(f"  Uploading Vulkan weights (Q4={'yes' if use_q4 else 'no'})...")
    for i, layer in enumerate(hb.layers):
        if isinstance(layer, VulkanDecoderLayer):
            layer.upload_weights()
            if use_q4:
                layer.upload_weights_q4()
    gc.collect()
    
    # Create GPU KV cache manager for GPU layers
    from p2_gpu_kv_test import GPUKVCacheManager
    kv_manager = GPUKVCacheManager(config, set(gpu_layer_indices))
    
    return model, hb, tokenizer, config, kv_manager


class LatencyStats:
    def __init__(self):
        self.times = {}
        self.counts = {}
    
    def record(self, name, duration):
        if name not in self.times:
            self.times[name] = 0.0
            self.counts[name] = 0
        self.times[name] += duration
        self.counts[name] += 1
    
    def get_avg(self, name):
        if name in self.times and self.counts[name] > 0:
            return self.times[name] / self.counts[name] * 1000
        return 0.0


def run_benchmark_at_seqlen(model, hb, tokenizer, config, kv_manager, seq_len, num_decode_steps=20, warmup_steps=3, gpu_layers=1, use_q4=True):
    """Run benchmark at a specific sequence length."""
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


def test_config(gpu_layers, seq_lens=[128, 512, 1024, 2048], use_q4=True):
    """Test a specific GPU layer configuration at multiple sequence lengths."""
    print(f"\n{'='*70}")
    print(f"TESTING: {gpu_layers} GPU layer(s) at seq_lens={seq_lens}")
    print(f"{'='*70}")
    
    model, hb, tokenizer, config, kv_manager = build_model(gpu_layers=gpu_layers, use_q4=use_q4)
    print(f"  Built. RSS: {get_memory_mb():.0f} MB, VRAM: {tvk.vram_used()/1024/1024:.0f} MB")
    
    all_results = {}
    for seq_len in seq_lens:
        print(f"\n  --- Sequence length {seq_len} ---")
        try:
            results = run_benchmark_at_seqlen(model, hb, tokenizer, config, kv_manager, seq_len, num_decode_steps=20, warmup_steps=3, gpu_layers=gpu_layers, use_q4=use_q4)
            all_results[f"seq_{seq_len}"] = results
            print(f"    Decode: avg={results.get('avg_ms', 0):.1f} ms, median={results.get('median_ms', 0):.1f} ms, {results.get('steps_per_sec', 0):.2f} steps/s")
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[f"seq_{seq_len}"] = {"error": str(e)}
    
    return all_results


def main():
    print("=" * 70)
    print("P4: PLACEMENT SWEEP AT MULTIPLE SEQUENCE LENGTHS")
    print("=" * 70)
    
    seq_lens = [128, 512, 1024, 2048]
    all_configs = {}
    
    # Test 0 GPU layers (CPU-only baseline)
    all_configs["gpu_layers_0"] = test_config(0, seq_lens)
    
    # Test 1 GPU layer
    all_configs["gpu_layers_1"] = test_config(1, seq_lens)
    
    # Test 2 GPU layers
    all_configs["gpu_layers_2"] = test_config(2, seq_lens)
    
    # Test 3 GPU layers if VRAM permits
    try:
        all_configs["gpu_layers_3"] = test_config(3, seq_lens)
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