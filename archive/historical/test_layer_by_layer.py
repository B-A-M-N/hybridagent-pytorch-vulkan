#!/usr/bin/env python3
"""Layer-by-layer INT8 model build - never hold all weights at once."""
import torch
import torch.nn as nn
from safetensors import safe_open
from int8_linear_v3 import Int8Linear

MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"

def get_memory_mb():
    import subprocess
    result = subprocess.run(['ps', '-o', 'rss=', str(os.getpid())], capture_output=True, text=True)
    return int(result.stdout.strip()) / 1024

import os

print("=== Layer-by-layer INT8 model build ===")
print(f"Initial RSS: {get_memory_mb():.0f} MB")

# Step 1: Create minimal config
from transformers import AutoConfig
config = AutoConfig.from_pretrained(MODEL_DIR)
print(f"Config loaded: {config.num_hidden_layers} layers, {config.hidden_size} hidden")

# Step 2: Build model layer by layer on CPU with Int8Linear
print("\nBuilding model layer by layer...")

# First, let's build just the first layer and measure memory
with torch.device("meta"):
    from transformers import AutoModelForCausalLM
    backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

# Replace Linear with Int8Linear
for name, module in list(backbone.named_modules()):
    if isinstance(module, nn.Linear):
        if name == 'lm_head' or name.endswith('.lm_head'):
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

print(f"After replacement: {get_memory_mb():.0f} MB")

# Materialize on CPU
backbone = backbone.to_empty(device="cpu")
print(f"After to_empty(cpu): {get_memory_mb():.0f} MB")

# Now load layer 0 weights only
print("\nLoading layer 0 weights...")
with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
    i = 0
    layer = backbone.model.layers[i]
    
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            if hasattr(mlp, name) and isinstance(getattr(mlp, name), Int8Linear):
                base = f"model.layers.{i}.mlp.{name}"
                weight_int8 = f.get_tensor(f"{base}.weight_int8")
                weight_scales = f.get_tensor(f"{base}.weight_scales")
                bias = None
                try:
                    bias = f.get_tensor(f"{base}.bias")
                except:
                    pass
                orig = getattr(mlp, name)
                orig.weight_int8.data.copy_(weight_int8)
                orig.scales.data.copy_(weight_scales)
                if bias is not None:
                    orig.bias.data.copy_(bias)

    if hasattr(layer, 'self_attn'):
        attn = layer.self_attn
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(attn, proj_name) and isinstance(getattr(attn, proj_name), Int8Linear):
                base = f"model.layers.{i}.self_attn.{proj_name}"
                weight_int8 = f.get_tensor(f"{base}.weight_int8")
                weight_scales = f.get_tensor(f"{base}.weight_scales")
                bias = None
                try:
                    bias = f.get_tensor(f"{base}.bias")
                except:
                    pass
                orig = getattr(attn, proj_name)
                orig.weight_int8.data.copy_(weight_int8)
                orig.scales.data.copy_(weight_scales)
                if bias is not None:
                    orig.bias.data.copy_(bias)

print(f"After loading layer 0: {get_memory_mb():.0f} MB")
print("SUCCESS: Layer 0 loaded")

# Test forward on just layer 0
print("\nTesting layer 0 forward...")
x = torch.randn(1, 12, 4096, dtype=torch.bfloat16)
with torch.no_grad():
    # Need to run through the layer properly
    # This is just testing the MLP
    mlp = layer.mlp
    out = mlp.gate_proj(x)
    print(f"gate_proj output: {out.shape}")

print(f"\nFinal RSS: {get_memory_mb():.0f} MB")