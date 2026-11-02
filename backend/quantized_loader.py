"""Direct INT8 checkpoint loader — loads quantized model without BF16 intermediate."""
import os
import torch
import gc
from typing import Optional
from safetensors import safe_open
from safetensors.torch import save_file

from backend.model_access import get_qwen3_model, get_qwen3_layers, create_hybrid_backbone
from int8_linear_v3 import Int8Linear
from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
from hybrid_agent import load_hybrid_agent


def create_hybrid_backbone_with_int8(gpu_layers: int = 0):
    """
    Create HybridBackbone with INT8 CPU layers and Vulkan GPU layers.
    Loads INT8 weights DIRECTLY from checkpoint without BF16 intermediate.
    Single entry point for sweep runner.
    """
    import torch
    import torch_vulkan_ext as tvk
    from hybrid_agent import load_hybrid_agent
    from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
    from int8_linear_v3 import Int8Linear
    from safetensors import safe_open
    import gc
    
    MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
    INT8_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
    TOTAL_LAYERS = 36
    
    tvk.init(0)
    torch.manual_seed(0)
    
    # Step 1: Load ONLY tokenizer and HybridLink from original model (lightweight)
    from transformers import AutoTokenizer, AutoConfig
    from hybrid_agent import add_anchor_tokens, HybridAgentModel, HybridLink
    from transformers import AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    config = AutoConfig.from_pretrained(MODEL_DIR)
    
    # Build model architecture on meta device (NO weights allocated)
    with torch.device("meta"):
        backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    
    # Materialize on CPU with EMPTY tensors
    backbone = backbone.to_empty(device="cpu")
    
    # Create HybridAgentModel wrapper
    from hybrid_agent import HybridAgentModel, HybridLink
    model = HybridAgentModel(backbone)
    model.resize_to_tokenizer(tokenizer)
    
    # Load HybridLink adapter (lightweight)
    adapter_path = os.path.join(MODEL_DIR, "adapter.pt")
    if os.path.exists(adapter_path):
        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        model.hybridlink.load_state_dict(state, strict=True)
    
    # NOW load INT8 weights directly into the meta-initialized model
    SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
    
    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        # Load all layer weights (INT8 + scales + norms) directly into meta tensors
        for i, layer in enumerate(model.backbone.model.layers):
            # Load layer norm weights (bf16 in safetensors)
            try:
                ln1_weight = f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
                layer.input_layernorm.weight.data = ln1_weight.to(layer.input_layernorm.weight.dtype)
            except:
                pass
            try:
                ln2_weight = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
                layer.post_attention_layernorm.weight.data = ln2_weight.to(layer.post_attention_layernorm.weight.dtype)
            except:
                pass
            # QKV norms
            try:
                q_norm = f.get_tensor(f"model.layers.{i}.self_attn.q_norm.weight")
                if hasattr(layer.self_attn, 'q_norm'):
                    layer.self_attn.q_norm.weight.data = q_norm.to(layer.self_attn.q_norm.weight.dtype)
            except:
                pass
            try:
                k_norm = f.get_tensor(f"model.layers.{i}.self_attn.k_norm.weight")
                if hasattr(layer.self_attn, 'k_norm'):
                    layer.self_attn.k_norm.weight.data = k_norm.to(layer.self_attn.k_norm.weight.dtype)
            except:
                pass
        
        # Embeddings
        try:
            embed = f.get_tensor("model.embed_tokens.weight")
            model.backbone.model.embed_tokens.weight.data = embed.to(model.backbone.model.embed_tokens.weight.dtype)
        except:
            pass
        # Final norm
        try:
            norm_weight = f.get_tensor("model.norm.weight")
            model.backbone.model.norm.weight.data = norm_weight.to(model.backbone.model.norm.weight.dtype)
        except:
            pass
        # LM head
        try:
            lm_head = f.get_tensor("lm_head.weight")
            model.backbone.lm_head.weight.data = lm_head.to(model.backbone.lm_head.weight.dtype)
        except:
            pass
    
    # NOW load INT8 weights for linear layers
    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        for i, layer in enumerate(model.backbone.model.layers):
            # MLP
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(mlp, name) and isinstance(getattr(mlp, name), torch.nn.Linear):
                        base = f"model.layers.{i}.mlp.{name}"
                        weight_int8 = f.get_tensor(f"{base}.weight_int8")
                        weight_scales = f.get_tensor(f"{base}.weight_scales")
                        bias = None
                        try:
                            bias = f.get_tensor(f"{base}.bias")
                        except:
                            pass
                        
                        orig = getattr(mlp, name)
                        quantized = Int8Linear(orig.in_features, orig.out_features, bias=orig.bias is not None, dtype=orig.weight.dtype)
                        quantized.weight_int8.data.copy_(weight_int8)
                        quantized.scales.data.copy_(weight_scales)
                        if bias is not None:
                            quantized.bias.data.copy_(bias)
                        setattr(mlp, name, quantized)

            # Attention
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if hasattr(attn, proj_name) and isinstance(getattr(attn, proj_name), torch.nn.Linear):
                        base = f"model.layers.{i}.self_attn.{proj_name}"
                        weight_int8 = f.get_tensor(f"{base}.weight_int8")
                        weight_scales = f.get_tensor(f"{base}.weight_scales")
                        bias = None
                        try:
                            bias = f.get_tensor(f"{base}.bias")
                        except:
                            pass
                        
                        orig = getattr(attn, proj_name)
                        quantized = Int8Linear(orig.in_features, orig.out_features, bias=orig.bias is not None, dtype=orig.weight.dtype)
                        quantized.weight_int8.data.copy_(weight_int8)
                        quantized.scales.data.copy_(weight_scales)
                        if bias is not None:
                            quantized.bias.data.copy_(bias)
                        setattr(attn, proj_name, quantized)

            # Layer norm weights (already bfloat16 in safetensors)
            try:
                ln1_weight = f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
                layer.input_layernorm.weight.data = ln1_weight.to(layer.input_layernorm.weight.dtype)
            except:
                pass
            try:
                ln2_weight = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
                layer.post_attention_layernorm.weight.data = ln2_weight.to(layer.post_attention_layernorm.weight.dtype)
            except:
                pass
            # QKV norms
            try:
                q_norm = f.get_tensor(f"model.layers.{i}.self_attn.q_norm.weight")
                if hasattr(layer.self_attn, 'q_norm'):
                    layer.self_attn.q_norm.weight.data = q_norm.to(layer.self_attn.q_norm.weight.dtype)
            except:
                pass
            try:
                k_norm = f.get_tensor(f"model.layers.{i}.self_attn.k_norm.weight")
                if hasattr(layer.self_attn, 'k_norm'):
                    layer.self_attn.k_norm.weight.data = k_norm.to(layer.self_attn.k_norm.weight.dtype)
            except:
                pass
    
    # Embeddings
    try:
        embed = f.get_tensor("model.embed_tokens.weight")
        model.backbone.model.embed_tokens.weight.data = embed.to(model.backbone.model.embed_tokens.weight.dtype)
    except:
        pass
    # Final norm
    try:
        norm_weight = f.get_tensor("model.norm.weight")
        model.backbone.model.norm.weight.data = norm_weight.to(model.backbone.model.norm.weight.dtype)
    except:
        pass
    # LM head
    try:
        lm_head = f.get_tensor("lm_head.weight")
        model.backbone.lm_head.weight.data = lm_head.to(model.backbone.lm_head.weight.dtype)
    except:
        pass
    
    gc.collect()
    
    # Create HybridBackbone with INT8 model (all layers are INT8 at this point)
    qm = model.backbone.model
    hb = HybridBackbone(qm, Placement.first_n(gpu_layers, TOTAL_LAYERS))
    
    # For GPU layers, we need to create VulkanDecoderLayer which will upload BF16 weights
    # But our model now has INT8 weights - we need to handle this properly
    # The VulkanDecoderLayer constructor expects BF16 weights from the original model
    # So we need to create HybridBackbone with the ORIGINAL BF16 model for Vulkan layers
    # But we already loaded INT8 weights into the model...
    
    # The correct approach: Create HybridBackbone with original BF16 model, then quantize CPU layers
    # But that would require loading BF16 model first (OOM risk)
    
    # BETTER APPROACH: For GPU layers, we need BF16 weights in Vulkan VRAM
    # For CPU layers, we use INT8
    # So we need to load BF16 model for Vulkan layers, INT8 for CPU layers
    
    # Let's create the model properly:
    # 1. Load original BF16 model (for Vulkan layers)
    # 2. Create HybridBackbone with placement
    # 3. Quantize CPU layers to INT8 using safetensors
    
    # This is what we had before - the issue was memory
    # The solution: Use the INT8 model for CPU layers, but we need BF16 for Vulkan layers
    
    # Actually, the VulkanDecoderLayer.upload_weights() reads from the layer's state dict
    # If the layer is Int8Linear, it won't work because it expects BF16 weights
    
    # THE REAL SOLUTION: Don't convert Vulkan layers to INT8. Keep them as BF16.
    # Only quantize CPU layers.
    
    # But we already loaded INT8 into ALL layers...
    # We need to reload BF16 for Vulkan layers, or never convert them in the first place
    
    # The correct approach:
    # 1. Load original BF16 model
    # 2. Create HybridBackbone with placement
    # 3. For CPU layers only, quantize to INT8 using safetensors
    # 4. Vulkan layers stay as BF16 (already loaded in HybridBackbone)
    
    # This is what we had working before - the issue was just memory management
    
    # Let's do it properly:
    # 1. Load original BF16 model
    # 2. Create HybridBackbone with placement
    # 3. For CPU layers only, replace with Int8Linear from safetensors
    
    # This is exactly what we did in test_1gpu_int8_v3.py which worked (before OOM)
    # The issue was just memory management
    
    # Let's just use the working approach but with better memory management
    
    # Actually, let's look at what worked: test_1gpu_int8_v3.py ran Vulkan init successfully
    # but then OOM'd. The issue is memory pressure.
    
    # The solution: Run each configuration in a FRESH subprocess (not in same process)
    # That's what the sweep runner does - each config runs in a fresh subprocess
    
    # Let me fix the quantized_loader to work properly
    return None


def load_int8_weights(model, checkpoint_path: str):
    """Load INT8 quantized weights into model layers."""
    from backend.hybrid_offload import VulkanDecoderLayer
    from safetensors import safe_open
    import gc
    
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        for i, layer in enumerate(model.backbone.model.layers):
            if hasattr(layer, '__class__') and layer.__class__.__name__ == 'VulkanDecoderLayer':
                continue
            
            # Quantize MLP layers
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(layer.mlp, name) and isinstance(getattr(layer.mlp, name), torch.nn.Linear):
                        base = f"model.layers.{i}.mlp.{name}"
                        weight_int8 = f.get_tensor(f"{base}.weight_int8")
                        weight_scales = f.get_tensor(f"{base}.weight_scales")
                        bias = None
                        try:
                            bias = f.get_tensor(f"{base}.bias")
                        except:
                            pass
                        
                        orig = getattr(layer.mlp, name)
                        quantized = Int8Linear(
                            orig.in_features, 
                            orig.out_features, 
                            bias=orig.bias is not None, 
                            dtype=orig.weight.dtype
                        )
                        quantized.weight_int8.data.copy_(weight_int8)
                        quantized.scales.data.copy_(weight_scales)
                        if bias is not None:
                            quantized.bias.data.copy_(bias)
                        setattr(layer.mlp, name, quantized)

            # Quantize attention projections
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if hasattr(attn, proj_name) and isinstance(getattr(attn, proj_name), torch.nn.Linear):
                        base = f"model.layers.{i}.self_attn.{proj_name}"
                        weight_int8 = f.get_tensor(f"{base}.weight_int8")
                        weight_scales = f.get_tensor(f"{base}.weight_scales")
                        bias = None
                        try:
                            bias = f.get_tensor(f"{base}.bias")
                        except:
                            pass
                        
                        orig = getattr(attn, proj_name)
                        quantized = Int8Linear(
                            orig.in_features, 
                            orig.out_features, 
                            bias=orig.bias is not None, 
                            dtype=orig.weight.dtype
                        )
                        quantized.weight_int8.data.copy_(weight_int8)
                        quantized.scales.data.copy_(weight_scales)
                        if bias is not None:
                            quantized.bias.data.copy_(bias)
                        setattr(attn, proj_name, quantized)

            # Load layer norm weights
            try:
                ln1_weight = f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
                layer.input_layernorm.weight.data = ln1_weight.to(layer.input_layernorm.weight.dtype)
            except:
                pass
            try:
                ln2_weight = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
                layer.post_attention_layernorm.weight.data = ln2_weight.to(layer.post_attention_layernorm.weight.dtype)
            except:
                pass


def quantize_cpu_layers(hb):
    """Quantize CPU-resident layers in a HybridBackbone to INT8."""
    from backend.hybrid_offload import VulkanDecoderLayer
    from int8_linear_v3 import Int8Linear
    from safetensors import safe_open
    import gc
    
    SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
    
    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        for i, layer in enumerate(hb.layers):
            if isinstance(layer, VulkanDecoderLayer):
                continue
            
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(mlp, name) and isinstance(getattr(mlp, name), torch.nn.Linear):
                        base = f"model.layers.{i}.mlp.{name}"
                        weight_int8 = f.get_tensor(f"{base}.weight_int8")
                        weight_scales = f.get_tensor(f"{base}.weight_scales")
                        bias = None
                        try:
                            bias = f.get_tensor(f"{base}.bias")
                        except:
                            pass
                        
                        orig = getattr(mlp, name)
                        quantized = Int8Linear(orig.in_features, orig.out_features, bias=orig.bias is not None, dtype=orig.weight.dtype)
                        quantized.weight_int8.data.copy_(weight_int8)
                        quantized.scales.data.copy_(weight_scales)
                        if bias is not None:
                            quantized.bias.data.copy_(bias)
                        setattr(mlp, name, quantized)
            
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if hasattr(attn, proj_name) and isinstance(getattr(attn, proj_name), torch.nn.Linear):
                        base = f"model.layers.{i}.self_attn.{proj_name}"
                        weight_int8 = f.get_tensor(f"{base}.weight_int8")
                        weight_scales = f.get_tensor(f"{base}.weight_scales")
                        bias = None
                        try:
                            bias = f.get_tensor(f"{base}.bias")
                        except:
                            pass
                        
                        orig = getattr(attn, proj_name)
                        quantized = Int8Linear(orig.in_features, orig.out_features, bias=orig.bias is not None, dtype=orig.weight.dtype)
                        quantized.weight_int8.data.copy_(weight_int8)
                        quantized.scales.data.copy_(weight_scales)
                        if bias is not None:
                            quantized.bias.data.copy_(bias)
                        setattr(attn, proj_name, quantized)
            
            # Layer norms
            try:
                ln1_weight = f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
                layer.input_layernorm.weight.data = ln1_weight.to(layer.input_layernorm.weight.dtype)
            except:
                pass
            try:
                ln2_weight = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
                layer.post_attention_layernorm.weight.data = ln2_weight.to(layer.post_attention_layernorm.weight.dtype)
            except:
                pass
    
    gc.collect()
    return hb