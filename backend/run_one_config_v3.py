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
    
    # Step 2: Create model architecture on meta device (NO weights allocated)
    with torch.device("meta"):
        backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    
    # Materialize on CPU with EMPTY tensors
    backbone = backbone.to_empty(device="cpu")
    
    # NOW load INT8 weights directly into the empty tensors
    print("Loading INT8 weights directly into empty tensors...")
    from safetensors import safe_open
    from int8_linear_v3 import Int8Linear
    
    SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
    
    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        # Load all layer weights (INT8 + scales + norms) directly into meta tensors
        for i in range(36):
            layer = model.backbone.model.layers[i]
            
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
            # Load INT8 weights for MLP
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
    embed = f.get_tensor("model.embed_tokens.weight")
    model.backbone.model.embed_tokens.weight.data = embed.to(model.backbone.model.embed_tokens.weight.dtype)
    
    # Final norm
    norm_weight = f.get_tensor("model.norm.weight")
    model.backbone.model.norm.weight.data = norm_weight.to(model.backbone.model.norm.weight.dtype)
    
    # LM head
    lm_head = f.get_tensor("lm_head.weight")
    model.backbone.lm_head.weight.data = lm_head.to(model.backbone.lm_head.weight.dtype)
    
    # Load HybridLink
    adapter_path = os.path.join(MODEL_DIR, "adapter.pt")
    if os.path.exists(adapter_path):
        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        model.hybridlink.load_state_dict(state, strict=True)
    print('HybridLink loaded')
    
    model.eval()
    gc.collect()
    return model, tokenizer


if __name__ == "__main__":
    import sys
    import gc
    import time
    
    if len(sys.argv) < 4:
        print("Usage: python run_one.py <seq_len> <gpu_layers> <reps>")
        sys.exit(1)
    
    seq_len = int(sys.argv[1])
    gpu_layers = int(sys.argv[2])
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    
    import torch
    import torch_vulkan_ext as tvk
    from hybrid_agent import load_hybrid_agent
    from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
    from int8_linear_v3 import Int8Linear
    from safetensors import safe_open
    import gc
    
    tvk.init(0)
    torch.manual_seed(0)
    
    # Step 1: Load original BF16 model
    print('Loading original BF16 model...')
    model, tok = load_hybrid_agent(
        '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic',
        dtype=torch.bfloat16, device='cpu'
    )
    print('Model loaded')
    
    # Step 2: Create HybridBackbone with original BF16 weights (Vulkan layers get correct weights)
    qm = model.backbone.model  # Qwen3Model
    hb = HybridBackbone(qm, Placement.first_n(gpu_layers, 36))
    print(f'HybridBackbone created with {gpu_layers} GPU layers')
    
    # CRITICAL: Delete original model immediately to free memory
    del model
    gc.collect()
    
    # Step 3: Quantize ONLY CPU layers (non-Vulkan) with INT8 from safetensors
    SAFETENSORS_PATH = '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors'
    
    with safe_open(SAFETENSORS_PATH, framework='pt', device='cpu') as f:
        for i, layer in enumerate(hb.layers):
            if isinstance(layer, VulkanDecoderLayer):
                continue
            
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(mlp, name) and isinstance(getattr(mlp, name), torch.nn.Linear):
                        base = f'model.layers.{i}.mlp.{name}'
                        weight_int8 = f.get_tensor(f'{base}.weight_int8')
                        weight_scales = f.get_tensor(f'{base}.weight_scales')
                        bias = None
                        try:
                            bias = f.get_tensor(f'{base}.bias')
                        except:
                            pass
                        
                        orig = getattr(mlp, name)
                        quantized = Int8Linear(orig.in_features, orig.out_features, bias=orig.bias is not None, dtype=orig.weight.dtype)
                        quantized.weight_int8.data.copy_(weight_int8)
                        quantized.scales.data.copy_(weight_scales)
                        if bias is not None:
                            quantized.bias.data.copy_(bias)
                        setattr(mlp, name, quantized)
                        del orig
                        gc.collect()

            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if hasattr(attn, proj_name) and isinstance(getattr(attn, proj_name), torch.nn.Linear):
                        base = f'model.layers.{i}.self_attn.{proj_name}'
                        weight_int8 = f.get_tensor(f'{base}.weight_int8')
                        weight_scales = f.get_tensor(f'{base}.weight_scales')
                        bias = None
                        try:
                            bias = f.get_tensor(f'{base}.bias')
                        except:
                            pass
                        
                        orig = getattr(attn, proj_name)
                        quantized = Int8Linear(orig.in_features, orig.out_features, bias=orig.bias is not None, dtype=orig.weight.dtype)
                        quantized.weight_int8.data.copy_(weight_int8)
                        quantized.scales.data.copy_(weight_scales)
                        if bias is not None:
                            quantized.bias.data.copy_(bias)
                        setattr(attn, proj_name, quantized)
                        del orig
                        gc.collect()

            # Layer norms (already bfloat16 in safetensors)
            try:
                ln1_weight = f.get_tensor(f'model.layers.{i}.input_layernorm.weight')
                layer.input_layernorm.weight.data = ln1_weight.to(layer.input_layernorm.weight.dtype)
            except: pass
            try:
                ln2_weight = f.get_tensor(f'model.layers.{i}.post_attention_layernorm.weight')
                layer.post_attention_layernorm.weight.data = ln2_weight.to(layer.post_attention_layernorm.weight.dtype)
            except: pass
            # QKV norms
            try:
                q_norm = f.get_tensor(f'model.layers.{i}.self_attn.q_norm.weight')
                if hasattr(attn, 'q_norm'):
                    attn.q_norm.weight.data = q_norm.to(attn.q_norm.weight.dtype)
            except: pass
            try:
                k_norm = f.get_tensor(f'model.layers.{i}.self_attn.k_norm.weight')
                if hasattr(attn, 'k_norm'):
                    attn.k_norm.weight.data = k_norm.to(attn.k_norm.weight.dtype)
            except: pass

    gc.collect()
    print('INT8 quantization done')
    
    # Run inference
    ids = torch.randint(0, 1000, (1, seq_len))
    with torch.no_grad():
        hb(input_ids=ids, use_cache=True)  # warmup
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            with torch.no_grad():
                hb(input_ids=torch.randint(0, 1000, (1, seq_len)), use_cache=True)
            ts.append(time.perf_counter() - t0)
    ts.sort()
    result = ts[len(ts) // 2] * 1000
    print(f"RESULT {result:.2f}")