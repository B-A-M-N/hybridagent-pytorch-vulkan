#!/usr/bin/env python3
"""Direct INT8 model loader - create model on CPU with Int8Linear, load from safetensors layer by layer."""
import os
import gc
import torch
import torch.nn as nn
from safetensors import safe_open

MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"


def create_hybrid_backbone_with_int8(gpu_layers: int = 0):
    """
    Create HybridBackbone with INT8 CPU layers and Vulkan GPU layers.
    CRITICAL: Create model on CPU with Int8Linear directly, load from INT8 checkpoint - NO BF16 intermediate.
    """
    import torch
    import torch_vulkan_ext as tvk
    from transformers import AutoTokenizer, AutoConfig
    from hybrid_agent import add_anchor_tokens, HybridAgentModel, HybridLink
    from transformers import AutoModelForCausalLM
    from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
    from int8_linear_v3 import Int8Linear
    from safetensors import safe_open
    import gc

    tvk.init(0)
    torch.manual_seed(0)

    MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
    SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"

    print("=== Building INT8 model directly (NO BF16 intermediate) ===")

    # Step 1: Create tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(MODEL_DIR)

    # Step 2: Create model architecture on CPU with Int8Linear directly
    print("Creating model architecture on CPU with Int8Linear...")
    
    # First create the model on meta to get the structure
    with torch.device("meta"):
        backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    
    # Replace ALL linear layers with Int8Linear (except lm_head)
    print("Replacing ALL Linear layers with Int8Linear...")
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

    print('All Linear layers (except lm_head) replaced with Int8Linear')

    # Materialize on CPU
    backbone = backbone.to_empty(device="cpu")
    print('Backbone materialized on CPU')

    # NOW load INT8 weights from safetensors directly into the int8 tensors
    print('Loading INT8 weights from safetensors...')
    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        # Load layer by layer to control memory
        for i in range(36):
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

            gc.collect()

        # Embeddings
        embed = f.get_tensor("model.embed_tokens.weight")
        backbone.model.embed_tokens.weight.data = embed.to(backbone.model.embed_tokens.weight.dtype)

        # Final norm
        norm_weight = f.get_tensor("model.norm.weight")
        backbone.model.norm.weight.data = norm_weight.to(backbone.model.norm.weight.dtype)

        # LM head (bf16 in safetensors)
        lm_head = f.get_tensor("lm_head.weight")
        backbone.lm_head.weight.data = lm_head.to(backbone.lm_head.weight.dtype)

    print('INT8 weights loaded')

    # Wrap in HybridAgentModel
    from hybrid_agent import HybridAgentModel, HybridLink
    model = HybridAgentModel(backbone)
    model.resize_to_tokenizer(tokenizer)

    # Load HybridLink
    adapter_path = os.path.join(MODEL_DIR, "adapter.pt")
    if os.path.exists(adapter_path):
        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        model.hybridlink.load_state_dict(state, strict=True)
    print('HybridLink loaded')

    # Create HybridBackbone with proper placement
    from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
    qm = model.backbone.model
    hb = HybridBackbone(qm, Placement.first_n(gpu_layers, 36))
    gc.collect()
    return hb


if __name__ == "__main__":
    hb = create_hybrid_backbone_with_int8(0)
    print('SUCCESS: CPU INT8 model created')
    import torch
    ids = torch.randint(0, 1000, (1, 12))
    with torch.no_grad():
        out = hb(input_ids=ids, use_cache=True)
    print(f'Forward pass OK: logits shape {out.logits.shape}')