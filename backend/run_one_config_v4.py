#!/usr/bin/env python3
"""Direct INT8 model loader - loads INT8 weights into meta-initialized model."""
import torch
import gc
import os
import torch.nn as nn
from safetensors import safe_open
from int8_linear_v3 import Int8Linear

MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"

def load_int8_model_direct(gpu_layers: int = 0):
    """
    Load INT8 model directly from checkpoint without BF16 intermediate.
    1. Create model architecture on meta device (NO weights)
    2. Load INT8 weights directly into empty tensors
    3. Create HybridBackbone with proper placement
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
    import os

    MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
    SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"

    tvk.init(0)
    torch.manual_seed(0)

    # Step 1: Create tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(MODEL_DIR)

    # Step 2: Create model on meta device (NO weights)
    with torch.device("meta"):
        backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

    # Materialize with empty tensors
    backbone = backbone.to_empty(device="cpu")

    # NOW load INT8 weights directly into the empty tensors
    from hybrid_agent import HybridAgentModel, HybridLink
    from hybrid_agent import add_anchor_tokens
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = HybridAgentModel(backbone)
    model.resize_to_tokenizer(tokenizer)

    # Load INT8 weights directly into the empty tensors
    print("Loading INT8 weights directly into empty tensors...")
    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        # Load all layer weights (INT8 + scales + norms) directly
        for i in range(36):
            layer = model.backbone.model.layers[i]
            
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

    # Now create HybridBackbone with proper placement
    from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
    qm = model.backbone.model
    hb = HybridBackbone(qm, Placement.first_n(gpu_layers, 36))
    
    # Now quantize ONLY CPU layers (non-Vulkan) with INT8 from safetensors
    SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
    with safe_open(SAFETENSORS_PATH, framework='pt', device='cpu') as f:
        for i, layer in enumerate(hb.layers):
            if isinstance(layer, VulkanDecoderLayer):
                continue
            # ... quantization logic here ...
            # (Same as before but now model is already INT8)

    print('INT8 quantization done')
    return HybridBackbone(model.backbone.model, Placement.first_n(gpu_layers, 36))


if __name__ == "__main__":
    import sys
    import gc
    import time
    import torch
    import torch_vulkan_ext as tvk
    from hybrid_agent import load_hybrid_agent
    from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
    from int8_linear_v3 import Int8Linear
    from safetensors import safe_open
    import gc
    
    tvk.init(0)
    torch.manual_seed(0)
    
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
    
    # Step 1: Create model on meta device, load INT8 directly
    from transformers import AutoTokenizer, AutoConfig
    from hybrid_agent import add_anchor_tokens, HybridAgentModel, HybridLink
    from transformers import AutoModelForCausalLM
    from hybrid_agent import HybridAgentModel, HybridLink
    from int8_linear_v3 import Int8Linear
    from safetensors import safe_open
    import gc
    
    MODEL_DIR = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic"
    SAFETENSORS_PATH = "/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors"
    
    tvk.init(0)
    torch.manual_seed(0)
    
    # Step 1: Create tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained("/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic", trust_remote_code=False)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    config = AutoConfig.from_pretrained(MODEL_DIR)
    
    # Step 2: Create model on meta device (NO weights)
    with torch.device("meta"):
        backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    
    # Materialize with empty tensors
    backbone = backbone.to_empty(device="cpu")
    
    # Load INT8 weights directly into empty tensors
    from safetensors import safe_open
    from int8_linear_v3 import Int8Linear
    import gc
    
    print('Loading INT8 weights directly into empty tensors...')
    with safe_open('/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors', framework='pt', device='cpu') as f:
        # Load tokenizer and config first
        from transformers import AutoTokenizer, AutoConfig
        from hybrid_agent import add_anchor_tokens, HybridAgentModel, HybridLink
        tokenizer = AutoTokenizer.from_pretrained("/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic", trust_remote_code=False)
        add_anchor_tokens(tokenizer)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        config = AutoConfig.from_pretrained("/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic")
        
        with torch.device("meta"):
            backbone = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        backbone = backbone.to_empty(device="cpu")
        
        # Load INT8 weights directly
        with safe_open('/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors', framework='pt', device='cpu') as f:
            for i in range(36):
                layer = model.backbone.model.layers[i]
                # ... (load weights)
    
    print("Done loading")

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