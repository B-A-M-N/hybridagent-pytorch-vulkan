#!/usr/bin/env python3
"""Direct INT8 state_dict loading with assign=True + to_empty for remaining meta tensors."""
import torch
import torch.nn as nn
from safetensors import safe_open
from int8_linear_v3 import Int8Linear
import os
import gc

MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"

def get_memory_mb():
    import subprocess
    result = subprocess.run(['ps', '-o', 'rss=', str(os.getpid())], capture_output=True, text=True)
    return int(result.stdout.strip()) / 1024

print("=== Direct INT8 state_dict loading with assign=True + to_empty ===")
print(f"Initial RSS: {get_memory_mb():.0f} MB")

# Step 1: Create tokenizer and config
from transformers import AutoTokenizer, AutoConfig
from hybrid_agent import add_anchor_tokens
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
add_anchor_tokens(tokenizer)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

config = AutoConfig.from_pretrained(MODEL_DIR)
print(f"Config loaded: {config.num_hidden_layers} layers, {config.hidden_size} hidden")

# Step 2: Create model architecture on meta device
print("\nCreating model on meta device...")
with torch.device("meta"):
    from transformers import AutoModelForCausalLM
    backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

# Step 3: Replace ALL Linear layers with Int8Linear (except lm_head)
print("Replacing Linear layers with Int8Linear...")
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

print(f"After replacement (meta): {get_memory_mb():.0f} MB")

# Step 4: Build state_dict from INT8 checkpoint matching the model's expected format
print("\nBuilding state_dict from INT8 checkpoint...")
state_dict = {}

with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
    # Layer weights
    for i in range(config.num_hidden_layers):
        if i % 6 == 0:
            print(f"  Building layer {i}... RSS: {get_memory_mb():.0f} MB")
        
        # MLP
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            base = f"model.layers.{i}.mlp.{name}"
            weight_int8 = f.get_tensor(f"{base}.weight_int8")
            weight_scales = f.get_tensor(f"{base}.weight_scales")
            state_dict[f"model.layers.{i}.mlp.{name}.weight_int8"] = weight_int8
            state_dict[f"model.layers.{i}.mlp.{name}.scales"] = weight_scales
            try:
                bias = f.get_tensor(f"{base}.bias")
                state_dict[f"model.layers.{i}.mlp.{name}.bias"] = bias
            except:
                pass

        # Attention
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            base = f"model.layers.{i}.self_attn.{proj_name}"
            weight_int8 = f.get_tensor(f"{base}.weight_int8")
            weight_scales = f.get_tensor(f"{base}.weight_scales")
            state_dict[f"model.layers.{i}.self_attn.{proj_name}.weight_int8"] = weight_int8
            state_dict[f"model.layers.{i}.self_attn.{proj_name}.scales"] = weight_scales
            try:
                bias = f.get_tensor(f"{base}.bias")
                state_dict[f"model.layers.{i}.self_attn.{proj_name}.bias"] = bias
            except:
                pass

        # Layer norms
        try:
            ln1 = f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
            state_dict[f"model.layers.{i}.input_layernorm.weight"] = ln1
        except: pass
        try:
            ln2 = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
            state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = ln2
        except: pass

        # QKV norms
        try:
            q_norm = f.get_tensor(f"model.layers.{i}.self_attn.q_norm.weight")
            state_dict[f"model.layers.{i}.self_attn.q_norm.weight"] = q_norm
        except: pass
        try:
            k_norm = f.get_tensor(f"model.layers.{i}.self_attn.k_norm.weight")
            state_dict[f"model.layers.{i}.self_attn.k_norm.weight"] = k_norm
        except: pass

    # Embeddings, final norm, lm_head
    embed = f.get_tensor("model.embed_tokens.weight")
    state_dict["model.embed_tokens.weight"] = embed

    norm_weight = f.get_tensor("model.norm.weight")
    state_dict["model.norm.weight"] = norm_weight

    lm_head = f.get_tensor("lm_head.weight")
    state_dict["lm_head.weight"] = lm_head

print(f"State dict built with {len(state_dict)} tensors. RSS: {get_memory_mb():.0f} MB")

# Step 5: Load state_dict directly with assign=True - materializes with loaded data!
print("\nLoading state_dict with assign=True...")
backbone.load_state_dict(state_dict, assign=True)
print(f"After load_state_dict: {get_memory_mb():.0f} MB")

# Step 6: Materialize remaining meta tensors (rotary embeddings, etc.)
print("Materializing remaining meta tensors...")
backbone = backbone.to_empty(device="cpu")
print(f"After to_empty(cpu): {get_memory_mb():.0f} MB")

# Step 7: Wrap in HybridAgentModel
from hybrid_agent import HybridAgentModel, HybridLink
model = HybridAgentModel(backbone)
model.resize_to_tokenizer(tokenizer)

# Load HybridLink
adapter_path = os.path.join(MODEL_DIR, "adapter.pt")
if os.path.exists(adapter_path):
    state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    model.hybridlink.load_state_dict(state, strict=True)
print('HybridLink loaded')

# Step 8: Create HybridBackbone with placement
from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
qm = model.backbone.model
hb = HybridBackbone(qm, Placement.first_n(0, 36))
gc.collect()
print(f"Final RSS: {get_memory_mb():.0f} MB")

# Step 9: Test forward pass
print("\nTesting forward pass...")
import torch
ids = torch.randint(0, 1000, (1, 12))
with torch.no_grad():
    out = hb(input_ids=ids, use_cache=True)
# HybridBackbone returns the model's output (BaseModelOutputWithPast)
# Apply lm_head to get logits
logits = model.backbone.lm_head(out.last_hidden_state)
print(f'Forward OK: logits shape {logits.shape}')

print("\n=== SUCCESS: Direct INT8 loading works ===")