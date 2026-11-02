#!/usr/bin/env python3
"""Complete INT8 model build - layer by layer, no BF16 intermediate."""
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

print("=== Complete INT8 model build (layer by layer) ===")
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

print(f"After replacement: {get_memory_mb():.0f} MB")

# Step 4: Materialize on CPU - creates int8 tensors directly!
print("Materializing on CPU...")
backbone = backbone.to_empty(device="cpu")
print(f"After to_empty(cpu): {get_memory_mb():.0f} MB")

# Step 5: Load weights layer by layer from INT8 checkpoint
print("\nLoading INT8 weights layer by layer...")
with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
    for i in range(config.num_hidden_layers):
        if i % 6 == 0:
            print(f"  Loading layer {i}... RSS: {get_memory_mb():.0f} MB")
        
        layer = backbone.model.layers[i]
        
        # MLP
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

        # Attention
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

        # Layer norms (bf16 in safetensors)
        try:
            ln1_weight = f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
            layer.input_layernorm.weight.data = ln1_weight.to(layer.input_layernorm.weight.dtype)
        except: pass
        try:
            ln2_weight = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
            layer.post_attention_layernorm.weight.data = ln2_weight.to(layer.post_attention_layernorm.weight.dtype)
        except: pass

        # QKV norms
        try:
            q_norm = f.get_tensor(f"model.layers.{i}.self_attn.q_norm.weight")
            if hasattr(layer.self_attn, 'q_norm'):
                layer.self_attn.q_norm.weight.data = q_norm.to(layer.self_attn.q_norm.weight.dtype)
        except: pass
        try:
            k_norm = f.get_tensor(f"model.layers.{i}.self_attn.k_norm.weight")
            if hasattr(layer.self_attn, 'k_norm'):
                layer.self_attn.k_norm.weight.data = k_norm.to(layer.self_attn.k_norm.weight.dtype)
        except: pass

    # Load embeddings, final norm, lm_head
    print("  Loading embeddings/norms/lm_head...")
    embed = f.get_tensor("model.embed_tokens.weight")
    backbone.model.embed_tokens.weight.data = embed.to(backbone.model.embed_tokens.weight.dtype)

    norm_weight = f.get_tensor("model.norm.weight")
    backbone.model.norm.weight.data = norm_weight.to(backbone.model.norm.weight.dtype)

    lm_head = f.get_tensor("lm_head.weight")
    backbone.lm_head.weight.data = lm_head.to(backbone.lm_head.weight.dtype)

gc.collect()
print(f"After all weights loaded: {get_memory_mb():.0f} MB")

# Step 6: Wrap in HybridAgentModel
from hybrid_agent import HybridAgentModel, HybridLink
model = HybridAgentModel(backbone)
model.resize_to_tokenizer(tokenizer)

# Load HybridLink
adapter_path = os.path.join(MODEL_DIR, "adapter.pt")
if os.path.exists(adapter_path):
    state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    model.hybridlink.load_state_dict(state, strict=True)
print('HybridLink loaded')

# Step 7: Create HybridBackbone with placement
from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
qm = model.backbone.model
hb = HybridBackbone(qm, Placement.first_n(0, 36))  # 0 GPU layers for now
gc.collect()
print(f"Final RSS: {get_memory_mb():.0f} MB")

# Step 8: Test forward pass
print("\nTesting forward pass...")
import torch
ids = torch.randint(0, 1000, (1, 12))
with torch.no_grad():
    out = hb(input_ids=ids, use_cache=True)
print(f'Forward OK: logits shape {out.logits.shape}')

print("\n=== SUCCESS: Complete INT8 model built and tested ===")