#!/usr/bin/env python3
"""Minimal test - create Int8Linear directly on CPU, load from INT8 checkpoint."""
import torch
import torch.nn as nn
from safetensors import safe_open
from int8_linear_v3 import Int8Linear

MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"

print("=== Minimal INT8 load test ===")

# Test Int8Linear creation directly on CPU
print("Creating Int8Linear on CPU...")
lin = Int8Linear(4096, 4096, bias=False, dtype=torch.bfloat16)
print(f"  weight_int8: {lin.weight_int8.shape}, dtype={lin.weight_int8.dtype}, device={lin.weight_int8.device}")
print(f"  scales: {lin.scales.shape}, dtype={lin.scales.dtype}, device={lin.scales.device}")

# Load one weight from safetensors
print("Loading weight from INT8 checkpoint...")
with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
    weight_int8 = f.get_tensor("model.layers.0.self_attn.q_proj.weight_int8")
    weight_scales = f.get_tensor("model.layers.0.self_attn.q_proj.weight_scales")
    print(f"  loaded weight_int8: {weight_int8.shape}, dtype={weight_int8.dtype}")
    print(f"  loaded weight_scales: {weight_scales.shape}, dtype={weight_scales.dtype}")
    lin.weight_int8.data.copy_(weight_int8)
    lin.scales.data.copy_(weight_scales)
print("SUCCESS: INT8 weight loaded directly")

# Test forward
print("Testing forward...")
x = torch.randn(1, 12, 4096, dtype=torch.bfloat16)
with torch.no_grad():
    out = lin(x)
print(f"Forward OK: output shape {out.shape}")