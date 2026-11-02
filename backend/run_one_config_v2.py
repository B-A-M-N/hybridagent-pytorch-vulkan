#!/usr/bin/env python3
"""Single configuration runner - quantize ONE LAYER AT A TIME to minimize peak memory."""
import torch
import gc
import os
import sys
import time

sys.path.insert(0, '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan')
sys.path.insert(0, '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/third_party/HybridAgent')
sys.path.insert(0, '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/backend/torch_vulkan_ext')

import torch
import gc

def run_one_config(seq_len: int, gpu_layers: int, reps: int = 2) -> float:
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
    from hybrid_agent import load_hybrid_agent
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
    # Do it ONE LAYER AT A TIME to minimize peak memory
    print('Quantizing CPU layers with INT8 (one layer at a time)...')
    SAFETENSORS_PATH = '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors'
    
    with safe_open(SAFETENSORS_PATH, framework='pt', device='cpu') as f:
        for i, layer in enumerate(hb.layers):
            if isinstance(layer, VulkanDecoderLayer):
                continue
            
            print(f'  Quantizing layer {i}...')
            
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
                        # Explicitly delete original weight to free memory
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
                        # Explicitly delete original weight to free memory
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
    return ts[len(ts) // 2] * 1000


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
    from hybrid_agent import load_hybrid_agent
    model, tok = load_hybrid_agent(
        '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic',
        dtype=torch.bfloat16, device='cpu'
    )
    print('Model loaded')
    
    # Step 2: Create HybridBackbone with original BF16 weights (Vulkan layers get correct weights)
    from backend.hybrid_offload import HybridBackbone, Placement, VulkanDecoderLayer
    qm = model.backbone.model  # Qwen3Model
    hb = HybridBackbone(qm, Placement.first_n(gpu_layers, 36))
    print(f'HybridBackbone created with {gpu_layers} GPU layers')
    
    # CRITICAL: Delete original model immediately to free memory
    del model
    gc.collect()
    
    # Step 3: Quantize ONLY CPU layers (non-Vulkan) with INT8 from safetensors
    # Do it ONE LAYER AT A TIME to minimize peak memory
    print('Quantizing CPU layers with INT8 (one layer at a time)...')
    SAFETENSORS_PATH = '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan/weights/HybridAgent-8B-Agentic-INT8/model.safetensors'
    
    with safe_open(SAFETENSORS_PATH, framework='pt', device='cpu') as f:
        for i, layer in enumerate(hb.layers):
            if isinstance(layer, VulkanDecoderLayer):
                continue
            
            print(f'  Quantizing layer {i}...')
            
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
                        # Explicitly delete original weight to free memory
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
                        # Explicitly delete original weight to free memory
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