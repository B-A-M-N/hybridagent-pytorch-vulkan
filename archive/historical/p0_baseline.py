#!/usr/bin/env python3
"""
P0: Authentic baseline — CPU-only vs Heterogeneous (1 GPU layer) on 100-step latent workload.
Mirrors the exact HybridAgent latent loop: prefill -> 100 steps of (HybridLink + Backbone).
"""

import torch
import torch.nn as nn
import torch_vulkan_ext as tvk
from hybrid_agent import HybridAgentModel, HybridLink, add_anchor_tokens
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer, _Buf, u32
from int8_linear_v3 import Int8Linear
from safetensors import safe_open
import os
import gc
import time
import psutil
import json

MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
ORIGINAL_SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic/model.safetensors"

def get_memory_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

def build_model(gpu_layers=0, use_vulkan=False):
    """Build model with specified GPU layers. Returns (model, hb, tokenizer, config)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(MODEL_DIR)

    if use_vulkan:
        tvk.init(0)

    torch.manual_seed(0)

    placement = Placement.first_n(gpu_layers, config.num_hidden_layers)
    gpu_layer_indices = set()
    for i in range(config.num_hidden_layers):
        if placement.device_of(i) == "vulkan":
            gpu_layer_indices.add(i)

    with torch.device("meta"):
        backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

    # Replace CPU layers with Int8Linear
    for name, module in list(backbone.named_modules()):
        if isinstance(module, nn.Linear):
            if name == 'lm_head' or name.endswith('.lm_head'):
                continue
            is_vulkan = False
            if 'model.layers.' in name:
                try:
                    layer_idx = int(name.split('model.layers.')[1].split('.')[0])
                    if layer_idx in gpu_layer_indices:
                        is_vulkan = True
                except:
                    pass
            if is_vulkan:
                continue
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            parent = backbone.get_submodule(parent_name) if parent_name else backbone
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            dtype = module.weight.dtype
            quantized = Int8Linear(in_features, out_features, bias=bias, dtype=dtype)
            setattr(parent, attr_name, quantized)

    state_dict = {}
    if gpu_layer_indices:
        with safe_open(ORIGINAL_SAFETENSORS_PATH, framework="pt", device="cpu") as f:
            for i in gpu_layer_indices:
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    base = f"model.layers.{i}.mlp.{name}"
                    try:
                        weight = f.get_tensor(f"{base}.weight")
                        state_dict[f"model.layers.{i}.mlp.{name}.weight"] = weight
                    except: pass
                    try:
                        bias = f.get_tensor(f"{base}.bias")
                        state_dict[f"model.layers.{i}.mlp.{name}.bias"] = bias
                    except: pass
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    base = f"model.layers.{i}.self_attn.{proj_name}"
                    try:
                        weight = f.get_tensor(f"{base}.weight")
                        state_dict[f"model.layers.{i}.self_attn.{proj_name}.weight"] = weight
                    except: pass
                    try:
                        bias = f.get_tensor(f"{base}.bias")
                        state_dict[f"model.layers.{i}.self_attn.{proj_name}.bias"] = bias
                    except: pass

    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        for i in range(config.num_hidden_layers):
            is_vulkan = i in gpu_layer_indices
            if not is_vulkan:
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

    backbone.load_state_dict(state_dict, assign=True)
    backbone = backbone.to_empty(device="cpu")

    if gpu_layer_indices:
        with safe_open(ORIGINAL_SAFETENSORS_PATH, framework="pt", device="cpu") as f:
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

    model = HybridAgentModel(backbone)
    model.resize_to_tokenizer(tokenizer)

    adapter_path = os.path.join(MODEL_DIR, "adapter.pt")
    if os.path.exists(adapter_path):
        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        model.hybridlink.load_state_dict(state, strict=True)

    qm = model.backbone.model
    hb = HybridBackbone(qm, Placement.first_n(gpu_layers, 36))
    gc.collect()

    # Upload Vulkan weights if using GPU
    if use_vulkan and gpu_layers > 0:
        print("Uploading Vulkan weights (Q4)...")
        for i, layer in enumerate(hb.layers):
            if isinstance(layer, VulkanDecoderLayer):
                layer.upload_weights()
                layer.upload_weights_q4()
        gc.collect()

    return model, hb, tokenizer, config

def run_latent_benchmark(model, hb, tokenizer, config, num_steps=100, use_q4=False, warmup_steps=5):
    """Run the exact HybridAgent latent loop: prefill -> HybridLink + Backbone x N steps."""
    results = {
        "prefill_ms": 0,
        "steps": [],
        "warmup_steps": [],
        "total_ms": 0,
        "peak_rss_mb": 0,
        "final_cache_len": 0,
    }

    # Prefill
    ids = torch.randint(0, 1000, (1, 12))
    torch.manual_seed(42)
    
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        out = model.backbone(input_ids=ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        cache_len = past_kv.get_seq_length()
    t_prefill_end = time.perf_counter()
    results["prefill_ms"] = (t_prefill_end - t_prefill_start) * 1000
    print(f"  Prefill: {results['prefill_ms']:.1f} ms, cache_len={cache_len}")

    # 100-step latent loop (with warmup separation)
    times = []
    hybridlink_times = []
    backbone_times = []
    transfer_times = []
    kv_concat_times = []
    
    with torch.no_grad():
        for step in range(num_steps):
            # HybridLink injection
            t_hl_start = time.perf_counter()
            injected = model.hybridlink(last_hidden)
            next_input = injected
            cache_position = torch.tensor([cache_len], device=next_input.device)
            t_hl_end = time.perf_counter()
            hybridlink_times.append((t_hl_end - t_hl_start) * 1000)
            
            # Backbone forward
            t_bb_start = time.perf_counter()
            out = hb(
                inputs_embeds=next_input,
                past_key_values=past_kv,
                use_cache=True,
                use_q4=use_q4,
                cache_position=cache_position,
                output_hidden_states=True
            )
            t_bb_end = time.perf_counter()
            backbone_times.append((t_bb_end - t_bb_start) * 1000)
            
            past_kv = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1:, :]
            cache_len = past_kv.get_seq_length()
            
            step_time = (t_bb_end - t_hl_start) * 1000  # end-to-end step
            times.append(step_time)
            
            if step < warmup_steps:
                results["warmup_steps"].append(step_time)
            else:
                results["steps"].append(step_time)
            
            if step % 20 == 0:
                rss = get_memory_mb()
                results["peak_rss_mb"] = max(results["peak_rss_mb"], rss)
                print(f"  Step {step}: {step_time:.1f} ms (HL: {hybridlink_times[-1]:.1f}, BB: {backbone_times[-1]:.1f}), cache_len={cache_len}, RSS={rss:.0f}MB")

    results["total_ms"] = sum(times)
    results["final_cache_len"] = cache_len
    
    # Compute statistics on sustained steps only
    sustained = results["steps"]
    if sustained:
        sustained_sorted = sorted(sustained)
        results["median_ms"] = sustained_sorted[len(sustained_sorted)//2]
        results["min_ms"] = min(sustained)
        results["max_ms"] = max(sustained)
        results["avg_ms"] = sum(sustained) / len(sustained)
        results["steps_per_sec"] = 1000 / results["avg_ms"]
        results["hybridlink_avg_ms"] = sum(hybridlink_times[warmup_steps:]) / len(sustained)
        results["backbone_avg_ms"] = sum(backbone_times[warmup_steps:]) / len(sustained)
    
    warmup_sorted = sorted(results["warmup_steps"])
    results["warmup_avg_ms"] = sum(results["warmup_steps"]) / len(results["warmup_steps"]) if results["warmup_steps"] else 0
    results["warmup_median_ms"] = warmup_sorted[len(warmup_sorted)//2] if warmup_sorted else 0
    
    return results

def main():
    print("=" * 70)
    print("P0: AUTHENTIC BASELINE - CPU-only vs Heterogeneous (1 GPU layer)")
    print("=" * 70)
    
    all_results = {}
    
    # 1. CPU-only baseline (0 GPU layers)
    print("\n>>> BUILDING CPU-ONLY MODEL (0 GPU layers)")
    print(f"  Initial RSS: {get_memory_mb():.0f} MB")
    model_cpu, hb_cpu, tokenizer, config = build_model(gpu_layers=0, use_vulkan=False)
    print(f"  After build RSS: {get_memory_mb():.0f} MB")
    
    print("\n>>> RUNNING CPU-ONLY 100-STEP LATENT BENCHMARK")
    cpu_results = run_latent_benchmark(model_cpu, hb_cpu, tokenizer, config, num_steps=100, use_q4=False)
    all_results["cpu_only"] = cpu_results
    
    del model_cpu, hb_cpu
    gc.collect()
    torch.cuda.empty_cache() if hasattr(torch, 'cuda') else None
    print(f"  After cleanup RSS: {get_memory_mb():.0f} MB")
    
    # 2. Heterogeneous: 1 GPU layer (Vulkan Q4)
    print("\n>>> BUILDING HETEROGENEOUS MODEL (1 GPU layer, Vulkan Q4)")
    print(f"  Initial RSS: {get_memory_mb():.0f} MB")
    model_hyb, hb_hyb, tokenizer, config = build_model(gpu_layers=1, use_vulkan=True)
    print(f"  After build RSS: {get_memory_mb():.0f} MB")
    print(f"  VRAM used: {tvk.vram_used() / 1024 / 1024:.0f} MB / {tvk.vram_total() / 1024 / 1024:.0f} MB")
    
    print("\n>>> RUNNING HETEROGENEOUS 100-STEP LATENT BENCHMARK")
    hyb_results = run_latent_benchmark(model_hyb, hb_hyb, tokenizer, config, num_steps=100, use_q4=True)
    all_results["heterogeneous_1gpu"] = hyb_results
    
    # Summary
    print("\n" + "=" * 70)
    print("P0 BASELINE SUMMARY")
    print("=" * 70)
    
    for name, r in all_results.items():
        print(f"\n{name.upper()}:")
        print(f"  Prefill:         {r['prefill_ms']:.1f} ms")
        print(f"  Warmup (5 steps): {r['warmup_avg_ms']:.1f} ms avg, {r['warmup_median_ms']:.1f} ms median")
        print(f"  Sustained (95 steps):")
        print(f"    Median:         {r['median_ms']:.1f} ms")
        print(f"    Avg:            {r['avg_ms']:.1f} ms")
        print(f"    Min/Max:        {r['min_ms']:.1f} / {r['max_ms']:.1f} ms")
        print(f"    Steps/sec:      {r['steps_per_sec']:.2f}")
        print(f"    HybridLink:     {r['hybridlink_avg_ms']:.1f} ms")
        print(f"    Backbone:       {r['backbone_avg_ms']:.1f} ms")
        print(f"  Total 100 steps:  {r['total_ms']:.1f} ms")
        print(f"  Final cache:      {r['final_cache_len']} tokens")
        print(f"  Peak RSS:         {r['peak_rss_mb']:.0f} MB")
    
    # Speedup comparison
    cpu_avg = all_results["cpu_only"]["avg_ms"]
    hyb_avg = all_results["heterogeneous_1gpu"]["avg_ms"]
    speedup = cpu_avg / hyb_avg if hyb_avg > 0 else 0
    print(f"\n>>> SPEEDUP (CPU-only / Heterogeneous): {speedup:.2f}x")
    if speedup > 1:
        print(f"    Heterogeneous is {speedup:.2f}x FASTER")
    else:
        print(f"    CPU-only is {1/speedup:.2f}x FASTER")
    
    # Save results
    with open("p0_baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to p0_baseline_results.json")

if __name__ == "__main__":
    main()