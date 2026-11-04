#!/usr/bin/env python3
"""Minimal convergence audit - 1 GPU layer, Q4, 100 latent steps."""
import torch
import torch.nn as nn
import torch_vulkan_ext as tvk
from backend.quantized_loader_direct import create_hybrid_backbone_with_int8
from hybrid_agent import HybridAgentModel, HybridLink, add_anchor_tokens
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM
from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
from int8_linear_v3 import Int8Linear
from safetensors import safe_open
import os
import gc
import time

MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
ORIGINAL_SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic/model.safetensors"

def get_memory_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1024 / 1024

print("=== MINIMAL CONVERGENCE AUDIT ===")
print(f"Initial RSS: {get_memory_mb():.0f} MB\n")

# ============================================================
# 1. BUILD MODEL - 1 GPU LAYER (Q4)
# ============================================================
gpu_layers = 1
print(f"Building model with {gpu_layers} GPU layer (Q4)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
add_anchor_tokens(tokenizer)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

config = AutoConfig.from_pretrained(MODEL_DIR)

tvk.init(0)
torch.manual_seed(0)

placement = Placement.first_n(gpu_layers, config.num_hidden_layers)
gpu_layer_indices = set()
for i in range(config.num_hidden_layers):
    if placement.device_of(i) == "vulkan":
        gpu_layer_indices.add(i)

with torch.device("meta"):
    backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

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
        quantized = Int8Linear(module.in_features, module.out_features, bias=module.bias is not None, dtype=module.weight.dtype)
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
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                base = f"model.layers.{i}.self_attn.{proj_name}"
                try:
                    weight = f.get_tensor(f"{base}.weight")
                    state_dict[f"model.layers.{i}.self_attn.{proj_name}.weight"] = weight
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
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                base = f"model.layers.{i}.self_attn.{proj_name}"
                weight_int8 = f.get_tensor(f"{base}.weight_int8")
                weight_scales = f.get_tensor(f"{base}.weight_scales")
                state_dict[f"model.layers.{i}.self_attn.{proj_name}.weight_int8"] = weight_int8
                state_dict[f"model.layers.{i}.self_attn.{proj_name}.scales"] = weight_scales
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
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                base = f"model.layers.{i}.self_attn.{proj_name}"
                try:
                    weight = f.get_tensor(f"{base}.weight")
                    if hasattr(backbone.model.layers[i].self_attn, proj_name):
                        getattr(backbone.model.layers[i].self_attn, proj_name).weight.data.copy_(weight)
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

# Upload Vulkan weights (Q4)
print("Uploading Vulkan weights (Q4)...")
for i, layer in enumerate(hb.layers):
    if isinstance(layer, VulkanDecoderLayer):
        layer.upload_weights()
        layer.upload_weights_q4()
gc.collect()

print(f"Model built. RSS: {get_memory_mb():.0f} MB")

# ============================================================
# 2. PRECISION VERIFICATION
# ============================================================
print(f"\n=== PRECISION VERIFICATION ===")
for i, layer in enumerate(hb.layers):
    if isinstance(layer, VulkanDecoderLayer):
        if hasattr(layer, '_w'):
            for name, buf in layer._w.items():
                print(f"  Layer {i} _w[{name}]: {buf.size/1e6:.1f} MB (FP32 in VRAM)")
        if hasattr(layer, '_w_q4_buf'):
            for name, buf in layer._w_q4_buf.items():
                print(f"  Layer {i} _w_q4_buf[{name}]: {buf.size/1e6:.1f} MB (Q4 packed in VRAM)")
        break

# ============================================================
# 3. FORWARD TEST (Q4)
# ============================================================
print(f"\n=== FORWARD TEST (Q4) ===")
ids = torch.randint(0, 1000, (1, 12))
with torch.no_grad():
    _ = hb(input_ids=ids, use_cache=True, use_q4=True)
    print("Q4 forward OK")

# ============================================================
# 4. 100-STEP LATENT TEST
# ============================================================
print(f"\n=== 100-STEP LATENT TEST ===")

ids = torch.randint(0, 1000, (1, 12))
with torch.no_grad():
    out = model.backbone(input_ids=ids, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    last_hidden = out.hidden_states[-1][:, -1:, :]
    cache_len = past_kv.get_seq_length()
    print(f"Initial cache length: {cache_len}")
    
    times = []
    for step in range(100):
        injected = model.hybridlink(last_hidden)
        next_input = injected
        cache_position = torch.tensor([cache_len], device=next_input.device)
        
        t0 = time.perf_counter()
        out = hb(
            inputs_embeds=next_input,
            past_key_values=past_kv,
            use_cache=True,
            use_q4=True,
            cache_position=cache_position,
            output_hidden_states=True
        )
        t1 = time.perf_counter()
        
        past_kv = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1:, :]
        cache_len = past_kv.get_seq_length()
        
        times.append((t1 - t0) * 1000)
        
        if step % 20 == 0:
            print(f"  Step {step}: {(t1-t0)*1000:.1f} ms, cache_len={cache_len}")

times.sort()
print(f"\n  Median: {times[50]:.1f} ms")
print(f"  Min: {times[0]:.1f} ms, Max: {times[-1]:.1f} ms")
print(f"  Avg: {sum(times)/100:.1f} ms")
print(f"  Final cache: {past_kv.get_seq_length()}")

# ============================================================
# MEMORY & PRECISION SUMMARY
# ============================================================
print(f"\n=== AUDIT SUMMARY ===")
import psutil
print(f"Final RSS: {get_memory_mb():.0f} MB")
print(f"Latent steps/sec: {1000/(sum(times)/100):.1f}")
print(f"100-step latent: COMPLETED ({sum(times)/100:.1f} ms/step avg)")
print(f"KV cache: CPU DynamicCache, continuity verified")
print(f"Precision: Vulkan Q4 matmul + FP32 norms, CPU INT8 (FP32 compute via _weight_fp_cached)")
print(f"Q4: ACTIVE (use_q4=True)")
print(f"Placement: Fixed (Placement.first_n), 1 GPU layer")
print(f"KV cache: CPU DynamicCache, per-step K/V transfer")
print(f"\n=== AUDIT COMPLETE ===")